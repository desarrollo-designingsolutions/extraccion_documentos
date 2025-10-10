from fastapi import APIRouter # type: ignore
from routes.scanear_archivos import router as scanear_router

# Router principal
router = APIRouter()

# Incluir el router de scanear archivos
router.include_router(scanear_router, tags=["scanear-archivos"])