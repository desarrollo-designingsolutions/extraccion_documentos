import numpy as np
import re
import pytesseract
import fitz
import pdfplumber
import boto3
import os
import asyncio
import httpx
import logging
from io import BytesIO
from typing import List, Optional
from openai import OpenAI
from botocore.exceptions import NoCredentialsError
from fastapi import HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_bytes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Cliente de OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

# ---------------------------
# Funciones auxiliares sync
# ---------------------------
def extraer_con_pymupdf(content: bytes) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    textos = [pagina.get_text() for pagina in doc if pagina.get_text()]
    return "\n".join(textos)

def extraer_con_pdfplumber(content: bytes) -> str:
    textos = []
    with pdfplumber.open(BytesIO(content)) as pdf:
        for pagina in pdf.pages:
            if (txt := pagina.extract_text()):
                textos.append(txt)
    return "\n".join(textos)

def extraer_con_ocr(content: bytes, lang: str = "spa") -> str:
    images = convert_from_bytes(content, dpi=200, fmt="RGB")
    textos = [pytesseract.image_to_string(img, lang=lang, config="--psm 6") for img in images]
    return "\n".join(textos)

# Función mejorada para extraer texto de PDF
async def extraer_texto_mejorado_async(url: str, lang: str = "spa") -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content = resp.content

    # Estrategia 1: PyMuPDF
    try:
        texto = await asyncio.to_thread(extraer_con_pymupdf, content)
        if texto.strip():
            return texto
    except Exception as e:
        logger.warning(f"[PyMuPDF] Falló: {e}")

    # Estrategia 2: pdfplumber
    try:
        texto = await asyncio.to_thread(extraer_con_pdfplumber, content)
        if texto.strip():
            return texto
    except Exception as e:
        logger.warning(f"[pdfplumber] Falló: {e}")

    # Estrategia 3: OCR
    try:
        texto = await asyncio.to_thread(extraer_con_ocr, content, lang)
        return texto
    except Exception as e:
        logger.error(f"[OCR] Falló: {e}")
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


async def generar_embeddings_async(
    chunks: List[str], 
    max_concurrent: int = 5
) -> List[Optional[List[float]]]:
    sem = asyncio.Semaphore(max_concurrent)

    async def wrapper(idx: int, chunk: str):
        async with sem:
            # logger.info(f"[Embeddings] Iniciando chunk {idx+1}/{len(chunks)}")
            result = await asyncio.to_thread(generar_embedding_openai, chunk)
            # logger.info(f"[Embeddings] Finalizado chunk {idx+1}/{len(chunks)}")
            return result

    tasks = [wrapper(i, chunk) for i, chunk in enumerate(chunks)]
    return await asyncio.gather(*tasks)