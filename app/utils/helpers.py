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
import hashlib
from datetime import datetime
from typing import Tuple

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


# ---------------------------
# Validation helpers for LLM outputs
# ---------------------------
def _parse_date_to_iso(date_str: str) -> str:
    if not date_str or not isinstance(date_str, str):
        return ""
    date_str = date_str.strip()
    # Common patterns: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD
    patterns = [r"^(\d{2})/(\d{2})/(\d{4})$", r"^(\d{2})-(\d{2})-(\d{4})$", r"^(\d{4})-(\d{2})-(\d{2})$"]
    for pat in patterns:
        m = re.match(pat, date_str)
        if m:
            try:
                if pat.startswith('^(\\d{2})'):
                    d, mth, y = m.groups()
                else:
                    y, mth, d = m.groups()
                return f"{int(y):04d}-{int(mth):02d}-{int(d):02d}"
            except Exception:
                return ""
    # Fallback: try parsing with datetime
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    return ""


def _clean_digits(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    return re.sub(r"\D+", "", s)


def validate_parsed_response(parsed: dict, context_text: str = "") -> dict:
    """
    Validate and normalize the parsed JSON coming from the LLM.
    - Ensures required keys exist and normalizes number/date fields.
    - Removes clearly invalid or hallucinated values based on simple heuristics.
    """
    if not isinstance(parsed, dict):
        return parsed

    out = parsed.copy()

    # Ensure top-level keys exist
    for key in ("prestacion_servicios_de_salud", "documentos_administrativos", "documentos_contractuales", "otros", "datos_prestacion_servicios", "resumen"):
        out.setdefault(key, [] if key != "datos_prestacion_servicios" and key != "resumen" else {})

    datos = out.get("datos_prestacion_servicios") or {}
    # Normalize nombre_paciente: keep if non-empty and appears in context or has at least two words
    nombre = datos.get("nombre_paciente", "")
    if isinstance(nombre, str) and nombre.strip():
        name_ok = False
        if nombre.lower() in context_text.lower():
            name_ok = True
        elif len([w for w in nombre.split() if w]) >= 2:
            name_ok = True
        if not name_ok:
            datos["nombre_paciente"] = ""
    else:
        datos["nombre_paciente"] = ""

    # tipo_documento: leave as-is but ensure string
    tipo = datos.get("tipo_documento", "")
    datos["tipo_documento"] = tipo if isinstance(tipo, str) else ""

    # numero_documento: keep only digits, require length >=6 (heuristic)
    num = _clean_digits(datos.get("numero_documento", ""))
    if len(num) < 6:
        datos["numero_documento"] = ""
    else:
        datos["numero_documento"] = num

    # numero_factura: allow alphanumeric but strip weird chars
    nf = datos.get("numero_factura", "")
    if isinstance(nf, str):
        datos["numero_factura"] = nf.strip()
    else:
        datos["numero_factura"] = ""

    # fecha_expedicion_factura: normalize to YYYY-MM-DD when possible
    fecha_norm = _parse_date_to_iso(datos.get("fecha_expedicion_factura", ""))
    datos["fecha_expedicion_factura"] = fecha_norm

    # emisor_factura, pagador keep as strings
    for fld in ("emisor_factura", "pagador"):
        v = datos.get(fld, "")
        datos[fld] = v.strip() if isinstance(v, str) else ""

    # autorizacion: dedupe and keep alnum+hyphen
    auths = datos.get("autorizacion", []) or []
    cleaned = []
    for a in auths:
        if not isinstance(a, str):
            continue
        ca = re.sub(r"[^A-Z0-9\-]", "", a.upper())
        if ca and ca not in cleaned:
            cleaned.append(ca)
    datos["autorizacion"] = cleaned

    out["datos_prestacion_servicios"] = datos

    # resumen: ensure string
    out["resumen"] = out.get("resumen", "") if isinstance(out.get("resumen", ""), str) else ""

    return out

# Agregar en helpers.py después de las funciones de extracción existentes
def extract_text(file_content: bytes, filename: str) -> str:
    """
    Extrae texto de contenido de archivo en bytes (similar a extraer_texto_mejorado_async pero sincrónico)
    """
    if not file_content:
        return ""
    
    # Determinar el tipo de archivo por extensión
    ext = filename.lower().split('.')[-1] if filename else ''
    
    # Si es PDF, usar las mismas estrategias que extraer_texto_mejorado_async
    if ext == 'pdf':
        # Estrategia 1: PyMuPDF
        try:
            texto = extraer_con_pymupdf(file_content)
            if texto and texto.strip():
                logger.info(f"Texto extraído con PyMuPDF: {len(texto)} caracteres")
                return limpiar_texto(texto)
        except Exception as e:
            logger.warning(f"[PyMuPDF] Falló: {e}")

        # Estrategia 2: pdfplumber
        try:
            texto = extraer_con_pdfplumber(file_content)
            if texto and texto.strip():
                logger.info(f"Texto extraído con pdfplumber: {len(texto)} caracteres")
                return limpiar_texto(texto)
        except Exception as e:
            logger.warning(f"[pdfplumber] Falló: {e}")

        # Estrategia 3: OCR
        try:
            texto = extraer_con_ocr(file_content, 'spa')
            if texto and texto.strip():
                logger.info(f"Texto extraído con OCR: {len(texto)} caracteres")
                return limpiar_texto(texto)
        except Exception as e:
            logger.error(f"[OCR] Falló: {e}")
    
    # Si es imagen, usar OCR directamente
    elif ext in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
        try:
            texto = extraer_con_ocr(file_content, 'spa')
            if texto and texto.strip():
                logger.info(f"Texto extraído con OCR: {len(texto)} caracteres")
                return limpiar_texto(texto)
        except Exception as e:
            logger.error(f"[OCR para imagen] Falló: {e}")
    
    # Para archivos de texto plano
    elif ext in ['txt', 'md']:
        try:
            texto = file_content.decode('utf-8')
            return limpiar_texto(texto)
        except UnicodeDecodeError:
            try:
                texto = file_content.decode('latin-1')
                return limpiar_texto(texto)
            except Exception as e:
                logger.error(f"[Decodificación texto] Falló: {e}")
    
    logger.warning(f"No se pudo extraer texto del archivo {filename}")
    return ""

def split_text(text: str) -> List[str]:
    """
    Función alias para dividir_en_chunks_semanticos para mantener compatibilidad
    """
    return dividir_en_chunks_semanticos(text)