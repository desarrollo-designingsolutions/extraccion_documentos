from fastapi import APIRouter # type: ignore
from routes.scanear_archivos import router as scanear_archivos
from routes.responder_pregunta import router as responder_pregunta
from routes.estadisticas_archivos import router as estadisticas_archivos
from routes.descomprimir_archivos import router as descomprimir_archivos

# Router principal
router = APIRouter()

# Incluir el router de scanear archivos
router.include_router(scanear_archivos)

# Incluir el router de responder pregunta
router.include_router(responder_pregunta)

# Incluir el router de estadisticas archivos
router.include_router(estadisticas_archivos)

# Incluir el router de descomprimir archivos
router.include_router(descomprimir_archivos)