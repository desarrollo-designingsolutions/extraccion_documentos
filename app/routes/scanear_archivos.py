from fastapi import APIRouter, Depends, HTTPException, Query  # type: ignore
from sqlalchemy.orm import Session  # type: ignore
from sqlalchemy.exc import IntegrityError  # type: ignore
from botocore.exceptions import ClientError  # type: ignore
from database import get_db
from models import ArchivoS3, ChunkArchivo
from .schemas import RespuestaExito
from utils.helpers import (
    get_s3_client,
    extraer_texto_mejorado,
    dividir_en_chunks_semanticos,
    generar_embedding_openai,
)
import os

router = APIRouter()


@router.get("/scanear_archivos/", response_model=RespuestaExito)
async def importar_y_guardar_archivos_mejorado(
    prefix: str = Query(
        "", description="Prefijo opcional para filtrar archivos (ej. carpeta/)"
    ),
    chunk_size: int = Query(
        1000, description="Tamaño de chunks para división semántica"
    ),
    max_keys: int = Query(3),
    chunk_overlap: int = Query(200, description="Solapamiento entre chunks"),
    db: Session = Depends(get_db),
):
    bucket_name = os.getenv("AWS_BUCKET")
    if not bucket_name:
        raise HTTPException(
            status_code=500, detail="Variable de entorno AWS_BUCKET no configurada"
        )

    expiration = int(os.getenv("PRESIGNED_EXPIRATION", 3600))
    s3 = get_s3_client()

    try:
        response = s3.list_objects_v2(
            Bucket=bucket_name, Prefix=prefix, MaxKeys=max_keys
        )
        objetos = response.get("Contents", [])

        total_guardados = 0
        total_chunks = 0

        for obj in objetos:
            if obj["Size"] > 0 and obj["Key"].lower().endswith(".pdf"):
                key = obj["Key"]

                # Verificar si ya existe
                archivo_existente = (
                    db.query(ArchivoS3).filter(ArchivoS3.nombre == key).first()
                )
                if archivo_existente:
                    # Verificar si ya tiene chunks
                    chunks_existentes = (
                        db.query(ChunkArchivo)
                        .filter(ChunkArchivo.archivo_s3_id == archivo_existente.id)
                        .count()
                    )

                    if chunks_existentes > 0:
                        continue

                nit_parts = key.split("/")
                nit = nit_parts[0] if nit_parts else key

                # Generar presigned URL
                try:
                    url_presignada = s3.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": bucket_name, "Key": key},
                        ExpiresIn=expiration,
                    )
                except Exception as e:
                    print(f"Error al generar presigned URL para {key}: {str(e)}")
                    continue

                # Extraer texto mejorado
                texto_pdf = extraer_texto_mejorado(url_presignada)
                if not texto_pdf.strip():
                    print(f"Advertencia: No se pudo extraer texto de {key}")
                    continue

                # Dividir en chunks semánticos
                chunks = dividir_en_chunks_semanticos(
                    texto_pdf, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                if not chunks:
                    continue

                # Guardar archivo principal
                try:
                    archivo_db = ArchivoS3(
                        nombre=key,
                        nit=nit,
                        tamaño=obj["Size"],
                        url_presignada=url_presignada,
                        texto_extraido=texto_pdf,
                    )
                    db.add(archivo_db)
                    db.flush()  # Para obtener el ID
                    db.commit()
                    print(f"Archivo: {archivo_db.id} creado")
                except IntegrityError:
                    db.rollback()
                    continue
                except Exception as e:
                    db.rollback()
                    continue

                print(
                    f"Cantidad de chucks generados: {len(chunks)} para el archivo: {archivo_db.id}"
                )

                # Procesar y guardar chunks
                chunks_guardados = 0
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue

                    embedding = generar_embedding_openai(chunk)
                    if not embedding:
                        print(
                            f"Advertencia: No se pudo generar embedding para chunk {i+1}"
                        )
                        continue

                    try:
                        chunk_db = ChunkArchivo(
                            archivo_s3_id=archivo_db.id,
                            contenido=chunk,
                            numero_chunk=i + 1,
                            embedding=embedding,
                        )

                        db.add(chunk_db)
                        db.commit()
                        chunks_guardados += 1

                    except Exception as e:
                        print(f"Error al guardar chunk {i+1}: {str(e)}")
                        continue

                # Commit final para todos los chunks
                try:
                    total_guardados += 1
                    total_chunks += chunks_guardados

                except Exception as e:
                    db.rollback()
                    print(f"Error en commit final para {key}: {str(e)}")
                    continue

        return RespuestaExito(
            mensaje="PDFs procesados con chunks semánticos y embeddings",
            total_guardados=total_guardados,
            total_chunks=total_chunks,
            objetos=objetos,
        )

    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchBucket":
            raise HTTPException(status_code=404, detail="Bucket no encontrado")
        else:
            raise HTTPException(
                status_code=500, detail=f"Error en S3: {e.response['Error']['Message']}"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {str(e)}")
