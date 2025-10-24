import os
import json
import logging
import redis
from celery import Celery
from typing import Dict, Any, List
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from models import Files, FilesChunks
from utils.helpers import (
    get_s3_client,
    extraer_texto_mejorado_async,
    dividir_en_chunks_semanticos,
    generar_embeddings_async
)
import asyncio
import traceback

# Logging simple pero informativo
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("job_scanear")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

celery = Celery("jobs", broker=BROKER_URL, backend=BACKEND_URL)
r = redis.Redis.from_url(REDIS_URL)

# DB setup para el worker (usar misma URL que la app)
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL no está definida en el entorno")
engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine)

def set_job_state(job_id: str, payload: dict):
    try:
        r.set(f"job:{job_id}", json.dumps(payload))
    except Exception as e:
        logger.error(f"Error al setear estado en Redis para job {job_id}: {e}")

@celery.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def procesar_archivos(self, payload: Dict[str, Any]):
    """
    Entrada: payload debe contener:
      - prefixes: List[str]
      - chunk_size: int
      - max_keys: Optional[int]
      - chunk_overlap: int
      - concurrency: int
      - bucket: str
    """
    job_id = self.request.id
    logger.info(f"[JOB {job_id}] Iniciando tarea")
    set_job_state(job_id, {"status": "queued", "progress": 0})

    try:
        asyncio.run(_procesar_archivos_async(job_id, payload))
        set_job_state(job_id, {"status": "completed", "progress": 100})
        logger.info(f"[JOB {job_id}] Completado correctamente")
        return {"status": "completed"}
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception(f"[JOB {job_id}] Error crítico: {e}")
        set_job_state(job_id, {"status": "failed", "progress": 0, "error": str(e), "traceback": tb})
        raise

async def _procesar_archivos_async(job_id: str, payload: Dict[str, Any]):
    prefixes: List[str] = payload.get("prefixes", [])
    chunk_size: int = payload.get("chunk_size", 1000)
    max_keys = payload.get("max_keys")
    chunk_overlap: int = payload.get("chunk_overlap", 200)
    concurrency: int = payload.get("concurrency", 5)
    bucket_name: str = payload.get("bucket")

    logger.info(f"[JOB {job_id}] Parámetros: prefixes={prefixes}, chunk_size={chunk_size}, max_keys={max_keys}, chunk_overlap={chunk_overlap}, concurrency={concurrency}, bucket={bucket_name}")
    set_job_state(job_id, {"status": "listing", "progress": 1, "message": "Listando objetos en S3"})

    s3 = get_s3_client()
    objetos = []

    # Listar objetos (mismo comportamiento que tenías)
    for prefix in prefixes:
        logger.info(f"[JOB {job_id}] Listando prefijo: {prefix}")
        try:
            if max_keys:
                resp = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=max_keys)
                objetos.extend(resp.get("Contents", []))
            else:
                continuation_token = None
                while True:
                    kwargs = {"Bucket": bucket_name, "Prefix": prefix}
                    if continuation_token:
                        kwargs["ContinuationToken"] = continuation_token
                    resp = s3.list_objects_v2(**kwargs)
                    objetos.extend(resp.get("Contents", []))
                    if resp.get("IsTruncated"):
                        continuation_token = resp["NextContinuationToken"]
                    else:
                        break
        except Exception as e:
            logger.exception(f"[JOB {job_id}] Error listando objetos para prefix {prefix}: {e}")
            # No abortamos todo el job por un prefijo fallido, seguimos con los demás
            continue

    total = len(objetos)
    logger.info(f"[JOB {job_id}] Total objetos encontrados: {total}")
    if total == 0:
        set_job_state(job_id, {"status": "completed", "progress": 100, "total": 0, "message": "No se encontraron objetos"})
        return

    # Variables de control
    processed = 0
    total_chunks = 0

    # Crear sesión DB por worker
    db = SessionLocal()

    # Semáforo para concurrencia de coroutines
    sem = asyncio.Semaphore(concurrency)

    async def procesar_obj(obj: dict):
        nonlocal processed, total_chunks
        key = obj.get("Key")
        size = obj.get("Size", 0)
        if not key:
            logger.warning(f"[JOB {job_id}] Objeto sin Key: {obj}")
            return

        async with sem:
            logger.info(f"[JOB {job_id}] Procesando archivo: {key}")
            set_job_state(job_id, {
                "status": "in_progress",
                "progress": int((processed / total) * 100) if total else 0,
                "current_file": key,
                "processed_files": processed,
                "total_files": total
            })

            # Filtrado por tamaño y extensión
            if size == 0 or not key.lower().endswith(".pdf"):
                logger.info(f"[JOB {job_id}] Skipping {key} size={size}")
                return

            # Verificar existencia
            try:
                archivo_existente = db.query(Files).filter(Files.name == key).first()
                if archivo_existente:
                    chunks_existentes = db.query(FilesChunks).filter(FilesChunks.files_id == archivo_existente.id).count()
                    if chunks_existentes > 0:
                        logger.info(f"[JOB {job_id}] {key} ya procesado, saltando")
                        return
            except Exception:
                logger.exception(f"[JOB {job_id}] Error consultando DB para {key}, continuar")
                # no abortar, intentar procesar

            # Presigned URL
            try:
                url_presignada = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket_name, "Key": key},
                    ExpiresIn=int(os.getenv("PRESIGNED_EXPIRATION", 3600)),
                )
                logger.info(f"[JOB {job_id}] Presigned URL generada para {key}")
            except Exception as e:
                logger.exception(f"[JOB {job_id}] Error generando presigned URL para {key}: {e}")
                return

            # Extraer texto
            try:
                logger.info(f"[JOB {job_id}] Extrayendo texto de {key}")
                texto_pdf = await extraer_texto_mejorado_async(url_presignada)
                if not texto_pdf or not texto_pdf.strip():
                    logger.warning(f"[JOB {job_id}] No se extrajo texto de {key}")
                    return
            except Exception as e:
                logger.exception(f"[JOB {job_id}] Error extrayendo texto de {key}: {e}")
                return

            # Dividir en chunks
            try:
                logger.info(f"[JOB {job_id}] Dividiendo en chunks {key}")
                chunks = dividir_en_chunks_semanticos(texto_pdf, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if not chunks:
                    logger.warning(f"[JOB {job_id}] No se generaron chunks para {key}")
                    return
                logger.info(f"[JOB {job_id}] {len(chunks)} chunks generados para {key}")
            except Exception as e:
                logger.exception(f"[JOB {job_id}] Error dividiendo en chunks para {key}: {e}")
                return

            # Guardar archivo principal
            try:
                archivo_db = Files(
                    name=key,
                    nit=key.split("/")[0] if "/" in key else key,
                    size=size,
                    url_preassigned=url_presignada,
                    text_extracted=texto_pdf,
                )
                db.add(archivo_db)
                db.flush()
                db.commit()
                logger.info(f"[JOB {job_id}] Archivo guardado en DB id={archivo_db.id}")
            except IntegrityError:
                db.rollback()
                logger.warning(f"[JOB {job_id}] Archivo duplicado (IntegrityError) para {key}, skip")
                return
            except Exception as e:
                db.rollback()
                logger.exception(f"[JOB {job_id}] Error guardando archivo {key}: {e}")
                return

            # Generar embeddings
            try:
                logger.info(f"[JOB {job_id}] Generando embeddings para {key}")
                embeddings = await generar_embeddings_async(chunks, max_concurrent=concurrency)
                logger.info(f"[JOB {job_id}] Embeddings generados para {key}")
            except Exception as e:
                logger.exception(f"[JOB {job_id}] Error generando embeddings para {key}: {e}")
                return

            # Guardar chunks en bulk
            try:
                chunks_db = []
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    if not chunk.strip() or embedding is None:
                        continue
                    chunks_db.append(
                        FilesChunks(
                            files_id=archivo_db.id,
                            content=chunk,
                            chunk_number=i + 1,
                            embedding=embedding,
                        )
                    )
                if chunks_db:
                    db.add_all(chunks_db)
                    db.commit()
                    saved = len(chunks_db)
                    total_chunks += saved
                    logger.info(f"[JOB {job_id}] Guardados {saved} chunks para {key}")
            except Exception as e:
                db.rollback()
                logger.exception(f"[JOB {job_id}] Error guardando chunks para {key}: {e}")
                return

            # Actualizar contador y progreso al finalizar este archivo
            processed += 1
            progress = int((processed / total) * 100)
            set_job_state(job_id, {
                "status": "in_progress",
                "progress": progress,
                "processed_files": processed,
                "total_files": total,
                "total_chunks": total_chunks,
                "current_file": key
            })
            logger.info(f"[JOB {job_id}] Progreso {progress}% ({processed}/{total})")

    # Construir y ejecutar tasks async respetando concurrencia
    tasks = [procesar_obj(obj) for obj in objetos]
    # Ejecutar y capturar excepciones para no abortar todo si una coroutine falla
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        # En teoría cada coroutine maneja sus errores, pero registramos si algo global ocurre
        logger.exception(f"[JOB {job_id}] Excepción en gather: {e}")
        set_job_state(job_id, {"status": "failed", "progress": int((processed / total) * 100), "error": str(e)})
    finally:
        try:
            db.close()
        except Exception:
            logger.warning(f"[JOB {job_id}] Error cerrando sesión DB")
