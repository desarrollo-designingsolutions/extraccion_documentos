from fastapi import APIRouter
from routes.scanear_archivos import router as scanear_archivos
from routes.jobs_status import router as jobs_status
from routes.responder_pregunta import router as responder_pregunta
from routes.responder_pregunta_multiple import router as responder_pregunta_multiple
from routes.estadisticas_archivos import router as estadisticas_archivos
from routes.search_invoice_number import router as search_invoice_number

# Router principal
router = APIRouter()

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