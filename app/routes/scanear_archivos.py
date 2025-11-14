import os
from fastapi import APIRouter, Body, HTTPException, Query
from typing import List, Optional
from app.jobs.job_scanear import procesar_archivos
from .schemas import RespuestaExito

router = APIRouter()

@router.post("/scanear_archivos/", response_model=RespuestaExito)
async def importar_y_guardar_archivos_mejorado(
    prefixes: List[str] = Body(..., embed=True, description="Lista de NITs/prefijos a procesar"),
    chunk_size: int = Query(1000),
    max_keys: Optional[int] = Query(None),
    chunk_overlap: int = Query(200),
    concurrency: int = Query(5)
):
    bucket_name = os.getenv("AWS_BUCKET")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="Variable de entorno AWS_BUCKET no configurada")

    task = procesar_archivos.delay({
        "prefixes": prefixes,
        "chunk_size": chunk_size,
        "max_keys": max_keys,
        "chunk_overlap": chunk_overlap,
        "concurrency": concurrency,
        "bucket": bucket_name,
    })
    return RespuestaExito(
        code=200,
        objetos=[{"job_id": task.id}],
    )
