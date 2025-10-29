from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
import requests
import os

router = APIRouter()

# Configurar templates
templates = Jinja2Templates(directory="templates")
api_base_url = "http://app:8000/api/v1"

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Ruta principal"""
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/list_files", response_class=HTMLResponse)
async def list_files(request: Request):
    """Ruta de lista de archivos"""
    return templates.TemplateResponse("list_files.html", {"request": request})

@router.get("/download_files_csv")
async def download_files_csv():
    """Endpoint para descargar CSV de archivos"""
    # Datos de ejemplo
    sample = [
        {"nombre": "factura_enero.pdf", "peso_kb": 125},
        {"nombre": "contrato_cliente.docx", "peso_kb": 3420},
        {"nombre": "informe_2024.xlsx", "peso_kb": 875},
        {"nombre": "escaneo_venta.png", "peso_kb": 210},
    ]
    df = pd.DataFrame(sample)
    
    def human_size(kb: int) -> str:
        if kb >= 1024:
            mb = kb / 1024
            return f"{mb:.2f} MB"
        return f"{kb} KB"

    df['peso'] = df['peso_kb'].apply(human_size)
    df_display = df[['nombre', 'peso']]
    
    # Crear CSV en memoria
    output = io.StringIO()
    df_display.to_csv(output, index=False)
    output.seek(0)
    
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8')),
        media_type='text/csv',
        headers={'Content-Disposition': 'attachment; filename=lista_archivos.csv'}
    )