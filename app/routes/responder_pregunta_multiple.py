from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from utils.helpers import generar_embedding_openai
from jobs.job_pregunta_multiple import process_nit
import json
import logging
import asyncio
from typing import List, Dict, Any
from redis.asyncio import Redis
from pydantic import BaseModel, Field

class PreguntaMultipleInput(BaseModel):
    pregunta: str
    nits: List[str]
    concurrency: int = Field(default=5, ge=1, le=20)  # Limita concurrency entre 1 y 20
    score_threshold: float = 0.5

router = APIRouter()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("job_scanear_multiple")

async def get_cached_embedding(redis: Redis, question: str) -> List[float] | None:
    cache_key = f"embedding:{hash(question)}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)
    embedding = generar_embedding_openai(question)
    if embedding:
        await redis.set(cache_key, json.dumps(embedding), ex=3600)  # Cache 1 hora
    return embedding


@router.post("/responder_pregunta_multiple/", response_model=List[Dict[str, Any]])
async def responder_pregunta_multiple(request: Request, input_data: PreguntaMultipleInput, db: AsyncSession = Depends(get_db)):
    start_time = asyncio.get_event_loop().time()  # Para métricas

    redis: Redis = request.app.state.redis
    embedding = await get_cached_embedding(redis, input_data.pregunta)
    if not embedding or not isinstance(embedding, list):
        logger.error("Embedding inválido")
        raise HTTPException(status_code=400, detail="No se pudo generar embedding")

    sem = asyncio.Semaphore(input_data.concurrency)

    # Launch per-NIT jobs from the jobs module
    tasks = [process_nit(request, nit, embedding, input_data, sem) for nit in input_data.nits]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter valid responses and log errors
    valid_responses = []
    for resp in responses:
        if isinstance(resp, Exception):
            logger.error(f"Error processing NIT: {resp}")
        else:
            valid_responses.append(resp)

    logger.info(f"Procesado multiple en {asyncio.get_event_loop().time() - start_time:.2f}s")
    return valid_responses