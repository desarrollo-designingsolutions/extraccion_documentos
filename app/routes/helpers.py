import numpy as np # type: ignore
import re
import pytesseract # type: ignore
import fitz # type: ignore
import pdfplumber # type: ignore
import boto3 # type: ignore
import os
import requests
from io import BytesIO
from typing import List, Optional
from openai import OpenAI # type: ignore
from botocore.exceptions import NoCredentialsError # type: ignore
from fastapi import HTTPException # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore
from pdf2image import convert_from_bytes # type: ignore

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

