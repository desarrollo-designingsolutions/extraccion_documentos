from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import boto3
import os
import requests
from io import BytesIO
import fitz  # PyMuPDF
import pdfplumber
from openai import OpenAI
from botocore.exceptions import ClientError, NoCredentialsError
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text
from database import get_db
from models import ArchivoS3, ChunkArchivo
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from collections import defaultdict
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import re
import asyncio
import aiohttp
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

# Router dedicado para APIs de AWS
router = APIRouter()

# Cliente de OpenAI
openai_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_key)

# Cargar modelos al inicio
model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class RespuestaExito(BaseModel):
    mensaje: str
    total_guardados: int
    total_chunks: int
    objetos: List[dict]


class PreguntaInput(BaseModel):
    id: int  # ID del registro en DB
    pregunta: str  # Texto de la pregunta
    score_threshold: float = 0.3  # Umbral mínimo de relevancia para chunks (0.0-1.0)


class RespuestaLLM(BaseModel):
    respuesta: str  # Respuesta generada por el LLM
    distancia: float  # Distancia pgvector (0=alta similitud, 1=baja)
    chunks_utilizados: int  # Número de chunks usados para la respuesta


class EstadisticasNIT(BaseModel):
    nit: str
    quantity_folder: int
    quantity_files: int


class ConsultaIARequest(BaseModel):
    pregunta: str
    modelo: str = "gpt-4"


# Función helper para conectar a S3
def get_s3_client():
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
        return s3
    except NoCredentialsError:
        raise HTTPException(status_code=401, detail="Credenciales AWS no configuradas")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al conectar a AWS: {str(e)}"
        )


# Función para limpiar y preprocesar texto
def limpiar_texto(texto: str) -> str:
    if not texto:
        return ""

    # Eliminar múltiples espacios y saltos de línea
    texto = re.sub(r"\s+", " ", texto)

    # Eliminar encabezados/pies de página comunes (patrones básicos)
    lineas = texto.split("\n")
    lineas_filtradas = []

    for linea in lineas:
        linea = linea.strip()
        # Filtrar números de página solos y encabezados muy cortos
        if (
            len(linea) > 10
            and not re.match(r"^\d+$", linea)  # Números solos
            and not re.match(r"^(Página|Page)\s+\d+$", linea, re.IGNORECASE)
        ):
            lineas_filtradas.append(linea)

    return " ".join(lineas_filtradas)


# Función mejorada para extraer texto de PDF
def extraer_texto_mejorado(url: str) -> str:
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        if len(response.content) == 0:
            print("ERROR: Descarga vacía.")
            return ""

        texto_completo = ""

        # ESTRATEGIA 1: PyMuPDF (rápido y efectivo para texto nativo)
        try:
            doc = fitz.open(stream=response.content, filetype="pdf")
            for pagina_num, pagina in enumerate(doc):
                texto_pagina = pagina.get_text()
                if texto_pagina:
                    texto_completo += texto_pagina + "\n"

            if texto_completo.strip():
                return limpiar_texto(texto_completo)
        except Exception as e:
            print(f"PyMuPDF falló: {str(e)}")
            texto_completo = ""

        # ESTRATEGIA 2: pdfplumber (mejor para layouts complejos y tablas)
        try:
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                for pagina_num, pagina in enumerate(pdf.pages):
                    # Extraer texto principal
                    texto_pagina = pagina.extract_text()
                    if texto_pagina:
                        texto_completo += texto_pagina + "\n"

                    # Intentar extraer y reconstruir tablas
                    try:
                        tablas = pagina.extract_tables()
                        for tabla_num, tabla in enumerate(tablas):
                            if tabla:
                                texto_tabla = "| TABLA | "
                                for fila in tabla:
                                    fila_limpia = [
                                        str(celda).replace("\n", " ") if celda else ""
                                        for celda in fila
                                    ]
                                    texto_tabla += " | ".join(fila_limpia) + " | "
                                texto_completo += texto_tabla + "\n"
                    except Exception as e_tabla:
                        print(
                            f"Error extrayendo tablas página {pagina_num + 1}: {str(e_tabla)}"
                        )

            if texto_completo.strip():
                return limpiar_texto(texto_completo)
        except Exception as e:
            print(f"pdfplumber falló: {str(e)}")
            texto_completo = ""

        # ESTRATEGIA 3: OCR como último recurso
        try:
            images = convert_from_bytes(response.content, dpi=200, fmt="RGB")
            for i, image in enumerate(images):
                ocr_text = pytesseract.image_to_string(image, lang="spa")
                texto_completo += ocr_text + "\n"

            texto_final = texto_completo.strip()
            return limpiar_texto(texto_final)

        except Exception as e:
            print(f"ERROR en OCR: {str(e)}")
            return ""

    except requests.exceptions.RequestException as e:
        print(f"ERROR descarga: {str(e)}")
        return ""
    except Exception as e:
        print(f"ERROR general en extracción mejorada: {str(e)}")
        return ""


# Función para dividir texto en chunks semánticos
def dividir_en_chunks_semanticos(
    texto: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[str]:
    if not texto.strip():
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    chunks = text_splitter.split_text(texto)

    return chunks


# Función para generar embedding con validación
def generar_embedding_openai(texto: str) -> Optional[List[float]]:
    if not texto.strip():
        return None

    try:
        response = openai_client.embeddings.create(
            input=texto, model="text-embedding-3-large"
        )
        # response = openai_client.embeddings.create(
        #     input=texto, model="text-embedding-3-small"
        # )
        embedding = response.data[0].embedding

        # Validar el embedding
        norma = np.linalg.norm(embedding)
        if norma < 0.01:  # Embedding muy pequeño puede indicar problema
            print(f"Advertencia: embedding con norma muy baja: {norma}")
            return None

        return embedding

    except Exception as e:
        print(f"Error al generar embedding: {str(e)}")
        return None


# API MEJORADA: Guarda PDFs con chunks semánticos y embeddings
@router.get("/scanear_archivos/", response_model=RespuestaExito)
async def importar_y_guardar_archivos_mejorado(
    prefix: str = Query(
        "", description="Prefijo opcional para filtrar archivos (ej. carpeta/)"
    ),
    chunk_size: int = Query(
        1000, description="Tamaño de chunks para división semántica"
    ),
    chunk_overlap: int = Query(200, description="Solapamiento entre chunks"),
    db: Session = Depends(get_db),
):
    bucket_name = os.getenv("AWS_BUCKET")
    if not bucket_name:
        raise HTTPException(
            status_code=500, detail="Variable de entorno AWS_BUCKET no configurada"
        )

    if not openai_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")

    expiration = int(os.getenv("PRESIGNED_EXPIRATION", 3600))
    s3 = get_s3_client()

    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=10)
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
                    else:
                        print("PDF existe pero sin chunks, reprocesando...")

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
                    print(f"Advertencia: No se pudieron generar chunks para {key}")
                    continue

                # Guardar archivo principal
                try:
                    archivo_db = ArchivoS3(
                        nombre=key,
                        nit=nit,
                        tamaño=obj["Size"],
                        url_presignada=url_presignada,
                        texto_extraido=texto_pdf,
                        embedding=None,  # Ya no guardamos embedding a nivel de documento
                    )
                    db.add(archivo_db)
                    db.flush()  # Para obtener el ID
                    print(f"Archivo: {archivo_db.id} creado")
                except IntegrityError:
                    db.rollback()
                    print(f"Duplicado: {key}")
                    continue
                except Exception as e:
                    db.rollback()
                    continue

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

                        print(
                            f"Chunk: {i + 1} genrado para el archivo: {archivo_db.id} creado"
                        )

                        db.add(chunk_db)
                        chunks_guardados += 1

                    except Exception as e:
                        print(f"Error al guardar chunk {i+1}: {str(e)}")
                        continue

                # Commit final para todos los chunks
                try:
                    db.commit()
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


# API MEJORADA: Responde pregunta usando chunks semánticos y reranking
@router.post("/responder-pregunta/", response_model=RespuestaLLM)
def responder_pregunta_mejorado(
    input_data: PreguntaInput, db: Session = Depends(get_db)
):
    if not openai_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY no configurada")

    # Generar embedding de la pregunta
    embedding_pregunta = generar_embedding_openai(input_data.pregunta)
    if not embedding_pregunta:
        raise HTTPException(
            status_code=400, detail="No se pudo generar embedding para la pregunta"
        )

    # Desactivar index scan para asegurar búsqueda exacta
    db.execute(text("SET enable_indexscan = off;"))

    # Query de debug SIN vector para verificar datos raw
    query_debug = text(
        """
        SELECT COUNT(*) as total 
        FROM chunks_archivo 
        WHERE archivo_s3_id = :id
        """
    )
    count = db.execute(query_debug, {"id": input_data.id}).scalar()

    if count == 0:
        # Chequea si el archivo principal existe
        query_archivo = text("SELECT id FROM archivos_s3 WHERE id = :id")
        archivo_exists = db.execute(query_archivo, {"id": input_data.id}).fetchone()

    # Buscar los chunks más relevantes usando pgvector
    query = text(
        """
        SELECT ca.id, ca.contenido, ca.embedding <=> CAST(:query_emb AS vector) AS distancia
        FROM chunks_archivo ca
        WHERE ca.archivo_s3_id = :id
        ORDER BY distancia ASC
        LIMIT 20
    """
    )

    results = db.execute(
        query, {"query_emb": embedding_pregunta, "id": input_data.id}
    ).fetchall()

    if not results:
        raise HTTPException(
            status_code=404, detail="No se encontraron chunks para este PDF"
        )

    # Mostrar información de los chunks encontrados
    for i, result in enumerate(results):
        print(
            f"Chunk {i+1}: distancia={result.distancia:.4f}, chars={len(result.contenido)}"
        )

    # Preparar datos para reranking
    chunks_candidatos = [result._asdict() for result in results]

    # Aplicar reranking con cross-encoder
    pares_reranking = [
        [input_data.pregunta, chunk["contenido"]] for chunk in chunks_candidatos
    ]
    scores_reranking = reranker.predict(pares_reranking)

    # Combinar scores de reranking con distancias vectoriales
    for i, chunk in enumerate(chunks_candidatos):
        chunk["score_reranking"] = float(scores_reranking[i])
        # Score combinado (ajustamos ponderación a 60/40)
        chunk["score_combinado"] = 0.6 * chunk["score_reranking"] + 0.4 * (
            1 - chunk["distancia"]
        )
        print(
            f"Chunk {i+1}: rerank={chunk['score_reranking']:.4f}, dist={chunk['distancia']:.4f}, combinado={chunk['score_combinado']:.4f}"
        )

    # Ordenar por score combinado y filtrar por umbral
    chunks_ordenados = sorted(
        chunks_candidatos, key=lambda x: x["score_combinado"], reverse=True
    )
    chunks_finales = [
        chunk
        for chunk in chunks_ordenados
        if chunk["score_combinado"] > input_data.score_threshold
    ]

    print(
        f"Chunks después de filtrar (umbral {input_data.score_threshold}): {len(chunks_finales)}"
    )

    if not chunks_finales:
        # Si no hay chunks que superen el umbral, usar los 2 mejores sin filtrar
        print("No hay chunks que superen el umbral, usando los 2 mejores disponibles")
        chunks_finales = chunks_ordenados[:2]

        if not chunks_finales:
            raise HTTPException(
                status_code=400,
                detail="No se encontraron chunks suficientemente relevantes para la pregunta",
            )

    # Limitar a los 3 chunks más relevantes para el contexto
    chunks_contexto = chunks_finales[:3]

    # Construir contexto para el LLM
    contexto = "\n\n".join(
        [
            f"[Fragmento {i+1}]: {chunk['contenido']}"
            for i, chunk in enumerate(chunks_contexto)
        ]
    )

    print(
        f"Usando {len(chunks_contexto)} chunks para contexto (total: {len(contexto)} caracteres)"
    )
    print(
        f"Distancia promedio: {sum(chunk['distancia'] for chunk in chunks_contexto) / len(chunks_contexto):.4f}"
    )
    print(
        f"Score combinado promedio: {sum(chunk['score_combinado'] for chunk in chunks_contexto) / len(chunks_contexto):.4f}"
    )

    # Enviar al LLM con contexto mejorado
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """Eres un asistente especializado en análisis de documentos. 
                    Responde la pregunta del usuario basándote ÚNICAMENTE en los fragmentos de contexto proporcionados.
                    Si la información en los fragmentos no es suficiente para responder completamente, 
                    indica qué aspectos puedes responder y cuáles no.""",
                },
                {
                    "role": "user",
                    "content": f"""CONTEXTO DEL DOCUMENTO:
                    {contexto}

                    PREGUNTA: {input_data.pregunta}

                    Instrucciones: Responde usando solo la información del contexto proporcionado. 
                    Si no hay información suficiente para alguna parte de la pregunta, explícita claramente qué no se puede responder.""",
                },
            ],
            max_tokens=800,
            temperature=0.3,
        )
        respuesta_llm = completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en LLM: {str(e)}")

    # Calcular distancia promedio de los chunks utilizados
    distancia_promedio = sum(chunk["distancia"] for chunk in chunks_contexto) / len(
        chunks_contexto
    )

    # Opcional: Reset indexscan para no afectar otras queries
    db.execute(text("SET enable_indexscan = on;"))

    return RespuestaLLM(
        respuesta=respuesta_llm,
        distancia=distancia_promedio,
        chunks_utilizados=len(chunks_contexto),
    )


# TERCERA API: Genera un archivo Excel con estadísticas por NIT
@router.get("/estadisticas-archivos/")
def obtener_estadisticas_archivos(
    prefix: str = Query(
        "", description="Prefijo opcional para filtrar objetos (ej. carpeta/)"
    ),
):
    bucket_name = os.getenv("AWS_BUCKET")
    if not bucket_name:
        raise HTTPException(
            status_code=500, detail="Variable de entorno AWS_BUCKET no configurada"
        )

    s3 = get_s3_client()
    try:
        stats_por_nit = defaultdict(lambda: {"subfolders": set(), "files": 0})

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        for page in pages:
            objetos = page.get("Contents", [])
            for obj in objetos:
                key = obj["Key"]
                segmentos = key.split("/")
                if len(segmentos) < 2:
                    continue

                nit = segmentos[0]
                if not nit:
                    continue

                if obj["Size"] > 0 and not key.endswith("/"):
                    stats_por_nit[nit]["files"] += 1
                    subfolder = "/".join(segmentos[1:-1])
                    if subfolder:
                        stats_por_nit[nit]["subfolders"].add(subfolder)

                elif key.endswith("/"):
                    subfolder = "/".join(segmentos[1:])
                    if subfolder:
                        stats_por_nit[nit]["subfolders"].add(subfolder)

        datos_excel = []
        for nit, data in stats_por_nit.items():
            datos_excel.append(
                {
                    "Carpeta NIT": nit,
                    "Cantidad de subcarpetas": len(data["subfolders"]),
                    "Cantidad de archivos": data["files"],
                }
            )

        if not datos_excel:
            datos_excel = [
                {
                    "Carpeta NIT": "",
                    "Cantidad de subcarpetas": 0,
                    "Cantidad de archivos": 0,
                }
            ]

        df = pd.DataFrame(datos_excel)

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Estadísticas")

        output.seek(0)

        excel_filename = "estadisticas_nit.xlsx"

        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={excel_filename}"},
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


# TERCERA API: Genera un archivo Excel con estadísticas por NIT
@router.post("/consultas_ia/")
async def consulta_ia(request: ConsultaIARequest):
    try:
        # Contexto del sistema especializado en programación
        system_context = """
        Eres un asistente especializado en programación y desarrollo de software. 
        Proporciona respuestas técnicas precisas, incluye ejemplos de código cuando sea relevante,
        y menciona mejores prácticas. Responde en el mismo idioma que la pregunta del usuario.
        """

        response = openai_client.chat.completions.create(
            model=request.modelo,
            messages=[
                {"role": "system", "content": system_context},
                {"role": "user", "content": request.pregunta},
            ],
            temperature=0.7,
            max_tokens=1500,
        )

        return {
            "respuesta": response.choices[0].message.content,
            "modelo_utilizado": response.model,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la consulta: {str(e)}")
