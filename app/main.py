from fastapi import FastAPI
import os

# Importar DB y modelos para registrarlos
from database import engine  # Para migraciones
from models import ArchivoS3  # Registra el modelo en Base
from migrations import run_migrations  # Función de migración

# Instancia de FastAPI
app = FastAPI(title="API de Consultas a AWS S3", version="1.0.0")

# Evento de startup: Ejecuta migraciones al iniciar la app
@app.on_event("startup")
async def startup_event():
    run_migrations()

# Importar y montar el router de APIs
from routers import router as api_router
app.include_router(api_router, prefix="/api/v1")

# Root endpoint para verificar que la app corre
@app.get("/")
def root():
    return {"mensaje": "API de AWS S3 corriendo."}

# Ejecutar con: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)