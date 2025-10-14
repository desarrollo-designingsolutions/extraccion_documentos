from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from .schemas import DescomprimirArchivos
from pydantic import BaseModel
from typing import List
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from utils.helpers import get_s3_client
import os
from datetime import datetime
from dotenv import load_dotenv
import zipfile
import io
from pathlib import Path

load_dotenv()

router = APIRouter()
AWS_BUCKET = os.getenv("AWS_BUCKET", "soportescoosalud")

class NitsRequest(BaseModel):
    nits: List[str]
    descomprimir: bool = False

class ArchivoZip(BaseModel):
    nombre: str
    ruta_completa: str
    tamaño: int
    ultima_modificacion: str
    etag: str
    archivos_descomprimidos: int = 0

class ResultadoNit(BaseModel):
    nit: str
    zips_encontrados: List[ArchivoZip]
    total_zips: int
    total_tamaño: int
    total_archivos_descomprimidos: int
    error: str = None

class DescomprimirArchivos(BaseModel):
    data: str
    nits: List[str]
    resultados: List[ResultadoNit]
    timestamp: str

async def get_s3_client_dependency():
    try:
        return get_s3_client()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error conectando a S3: {str(e)}")

def listar_todos_zips_recursivo(s3_client, bucket_name: str, prefix: str) -> List[ArchivoZip]:
    """Busca RECURSIVAMENTE todos los archivos ZIP"""
    zips = []
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    if not obj['Key'].endswith('/') and obj['Key'].lower().endswith('.zip'):
                        archivo_zip = ArchivoZip(
                            nombre=os.path.basename(obj['Key']),
                            ruta_completa=obj['Key'],
                            tamaño=obj['Size'],
                            ultima_modificacion=obj['LastModified'].isoformat(),
                            etag=obj['ETag']
                        )
                        zips.append(archivo_zip)
    except ClientError as e:
        raise Exception(f"Error listando ZIPs: {str(e)}")
    
    return zips

def descomprimir_zip_completo(s3_client, bucket_name: str, zip_file: ArchivoZip) -> int:
    """Descomprime ZIP en el MISMO NIVEL donde se encontró, con prefijo ZIP_EXTRA_"""
    try:
        # Descargar ZIP
        print(f"📥 Descargando {zip_file.nombre}...")
        response = s3_client.get_object(Bucket=bucket_name, Key=zip_file.ruta_completa)
        zip_content = response['Body'].read()
        print(f"✅ ZIP descargado: {len(zip_content)} bytes")
        
        # Extraer directorio base del ZIP (sin el nombre del archivo)
        dir_base = Path(zip_file.ruta_completa).parent  # Ej: "900986941/facturas/"
        nombre_zip_sin_ext = Path(zip_file.nombre).stem
        carpeta_prefijo = f"ZIP_EXTRA_{nombre_zip_sin_ext}"
        
        # Carpeta destino: {dir_base}/ZIP_EXTRA_{nombre}/
        # Ej: 900986941/facturas/ZIP_EXTRA_UV27232/
        print(f"📁 Carpeta destino: {dir_base}/{carpeta_prefijo}/")
        
        archivos_subidos = 0
        
        with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
            # Contar archivos primero
            total_en_zip = sum(1 for f in zip_ref.infolist() if not f.is_dir())
            print(f"📄 Archivos en ZIP: {total_en_zip}")
            
            for file_info in zip_ref.infolist():
                if file_info.is_dir():
                    continue
                
                # Construir ruta nueva EN EL MISMO NIVEL del ZIP
                ruta_relativa = file_info.filename
                if Path(ruta_relativa).parent != '.':
                    # Mantener estructura interna: {dir_base}/ZIP_EXTRA_xxx/{subcarpeta}/archivo
                    ruta_nueva = f"{dir_base}/{carpeta_prefijo}/{ruta_relativa}"
                else:
                    # Archivo en raíz del ZIP: {dir_base}/ZIP_EXTRA_xxx/archivo
                    ruta_nueva = f"{dir_base}/{carpeta_prefijo}/{Path(ruta_relativa).name}"
                
                # Leer contenido
                with zip_ref.open(file_info) as file:
                    content = file.read()
                
                print(f"📤 Subiendo: {ruta_nueva} ({len(content)} bytes)")
                
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=ruta_nueva,
                    Body=content,
                    ContentType='application/octet-stream',
                    Metadata={
                        'fuente_zip': zip_file.ruta_completa,
                        'nombre_original': file_info.filename,
                        'descomprimido_en': datetime.utcnow().isoformat(),
                        'carpeta_destino': str(Path(ruta_nueva).parent)
                    }
                )
                archivos_subidos += 1
        
        print(f"✅ {archivos_subidos}/{total_en_zip} archivos descomprimidos en {dir_base}")
        return archivos_subidos
        
    except zipfile.BadZipFile as e:
        raise Exception(f"ZIP corrupto {zip_file.nombre}: {str(e)}")
    except Exception as e:
        raise Exception(f"Error descomprimiendo {zip_file.nombre}: {str(e)}")
@router.post("/descomprimir_archivos/", response_model=DescomprimirArchivos)
async def descomprimir_archivos(
    request: NitsRequest,
    s3_client: boto3.client = Depends(get_s3_client_dependency),
    background_tasks: BackgroundTasks = None
):
    """Endpoint principal: lista y opcionalmente descomprime ZIPs"""
    print(f"🔍 Bucket: {AWS_BUCKET}, Descomprimir: {request.descomprimir}")
    
    resultados = []
    bucket_name = AWS_BUCKET
    
    # Verificar bucket
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            raise HTTPException(status_code=404, detail=f"Bucket '{bucket_name}' no encontrado")
        raise HTTPException(status_code=403, detail="Acceso denegado al bucket")
    
    for nit in request.nits:
        print(f"\n🔍 Procesando NIT: {nit}")
        resultado_nit = ResultadoNit(
            nit=nit,
            zips_encontrados=[],
            total_zips=0,
            total_tamaño=0,
            total_archivos_descomprimidos=0
        )
        
        try:
            prefix = f"{nit}/"
            zips_encontrados = listar_todos_zips_recursivo(s3_client, bucket_name, prefix)
            
            print(f"📦 ZIPs encontrados: {len(zips_encontrados)}")
            for zip_file in zips_encontrados:
                print(f"   - {zip_file.nombre} ({zip_file.tamaño} bytes)")
            
            total_zips = len(zips_encontrados)
            total_tamaño = sum(z.tamaño for z in zips_encontrados)
            
            resultado_nit.zips_encontrados = zips_encontrados
            resultado_nit.total_zips = total_zips
            resultado_nit.total_tamaño = total_tamaño
            
            # DESCOMPRESIÓN
            if request.descomprimir and zips_encontrados:
                print("🚀 INICIANDO DESCOMPRESIÓN...")
                total_descomprimidos = 0
                
                if len(zips_encontrados) > 3:  # Background para muchos ZIPs
                    background_tasks.add_task(
                        descomprimir_zips_en_background,
                        s3_client,
                        bucket_name,
                        zips_encontrados
                    )
                    resultado_nit.error = f"Descompresión de {total_zips} ZIPs en background"
                else:
                    # Síncrono para pocos ZIPs
                    for zip_file in zips_encontrados:
                        try:
                            archivos_extraidos = descomprimir_zip_completo(
                                s3_client, bucket_name, zip_file
                            )
                            zip_file.archivos_descomprimidos = archivos_extraidos
                            total_descomprimidos += archivos_extraidos
                        except Exception as e:
                            print(f"❌ Error {zip_file.nombre}: {str(e)}")
                            zip_file.archivos_descomprimidos = 0
                
                resultado_nit.total_archivos_descomprimidos = total_descomprimidos
                print(f"📊 Total descomprimidos: {total_descomprimidos}")
            else:
                print("⏸️ Descompresión deshabilitada")
            
        except Exception as e:
            print(f"❌ Error NIT {nit}: {str(e)}")
            resultado_nit.error = str(e)
        
        resultados.append(resultado_nit)
    
    return DescomprimirArchivos(
        data=f"Completado. ZIPs: {sum(r.total_zips for r in resultados)}",
        nits=request.nits,
        resultados=resultados,
        timestamp=datetime.utcnow().isoformat()
    )

def descomprimir_zips_en_background(s3_client, bucket_name: str, zips: List[ArchivoZip]):
    """Background task para múltiples ZIPs"""
    for zip_file in zips:
        try:
            archivos = descomprimir_zip_completo(s3_client, bucket_name, zip_file)
            zip_file.archivos_descomprimidos = archivos
            print(f"✅ Background: {zip_file.nombre} -> {archivos} archivos")
        except Exception as e:
            print(f"❌ Background error {zip_file.nombre}: {str(e)}")