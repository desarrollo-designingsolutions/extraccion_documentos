from fastapi import FastAPI
from migrations import run_migrations
from sentence_transformers import CrossEncoder
from openai import AsyncOpenAI
from redis.asyncio import Redis
import logging
import os

app = FastAPI(title="API de Consultas a AWS S3", version="1.0.0")
prefix = "/api/v1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    run_migrations()
    logger.info("Cargando modelo reranker en startup")
    app.state.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device='cpu' or 'cuda')
    logger.info("Reranker cargado")
    app.state.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    app.state.redis = await Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))

@app.on_event("shutdown")
async def shutdown_event():
    try:
        model = getattr(app.state, "reranker", None)
        if model is not None:
            del app.state.reranker
            logger.info("Reranker liberado")

        # Cierra Redis async
        redis = getattr(app.state, "redis", None)
        if redis is not None:
            await redis.aclose()
            logger.info("Redis cerrado")
    except Exception:
        logger.exception("Error en shutdown")

# Importar y montar el router de APIs
from routers import router as api_router
app.include_router(api_router, prefix=prefix)

@app.get("/")
def root():
    return {"mensaje": "API de AWS S3 corriendo."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)