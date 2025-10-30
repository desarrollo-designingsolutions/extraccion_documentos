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

@router.get("/chat")
async def chat_page(request: Request, file_id: int = None, file_name: str = None):
    return templates.TemplateResponse("chat_file.html", {
        "request": request,
        "file_id": file_id,
        "file_name": file_name
    })