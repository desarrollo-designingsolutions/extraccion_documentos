from fastapi import APIRouter
from app.routes.scanear_archivos import router as scanear_archivos
from app.routes.jobs_status import router as jobs_status
from app.routes.responder_pregunta import router as responder_pregunta
from app.routes.responder_pregunta_multiple import router as responder_pregunta_multiple
from app.routes.estadisticas_archivos import router as estadisticas_archivos
from app.routes.search_invoice_number import router as search_invoice_number
from app.routes.list_files import router as list_files
from app.routes.chat_router import router as chat_router
from app.routes.notebooklm import router as notebooklm_router

# Router principal
router = APIRouter()

@router.get("/")
def root():
    return {
        "status": "success",
        "status_code": 200,
        "mensaje": "API de AWS S3 corriendo."
    }

# Incluir el router de scanear archivos
router.include_router(scanear_archivos)

# Incluir el router de jobs status
router.include_router(jobs_status)

# Incluir el router de responder pregunta
router.include_router(responder_pregunta)

# Incluir el router de responder pregunta multiple
router.include_router(responder_pregunta_multiple)

# Incluir el router de estadisticas archivos
router.include_router(estadisticas_archivos)

# Incluir el router de search invoice number
router.include_router(search_invoice_number)

# api de listar los archivos de la base de datos
router.include_router(list_files)

# api de chat con ia sobre documento individual
router.include_router(chat_router)

# Agregar en la lista de routers incluidos
router.include_router(notebooklm_router)