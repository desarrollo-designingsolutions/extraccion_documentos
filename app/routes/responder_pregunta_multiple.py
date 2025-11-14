from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from typing import List
import logging
from app.jobs.job_pregunta_multiple import procesar_pregunta_multiple
from .schemas import RespuestaExito


class PreguntaMultipleInput(BaseModel):
    pregunta: str
    nits: List[str]
    concurrency: int = Field(default=5, ge=1, le=20)
    score_threshold: float = 0.5

router = APIRouter()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("responder_pregunta_multiple")


@router.post("/responder_pregunta_multiple/", response_model=RespuestaExito)
async def responder_pregunta_multiple(request: Request, input_data: PreguntaMultipleInput):
    """
    Encola un job que procesará cada archivo de los NITs solicitados.
    El job guardará en la columna `json_category` de la tabla `files` el JSON devuelto por la LLM
    para cada archivo. Se devuelve el job_id para consultar el estado.
    """
    payload = {
        "pregunta": input_data.pregunta,
        "nits": input_data.nits,
        "concurrency": input_data.concurrency,
        "score_threshold": input_data.score_threshold,
    }

    task = procesar_pregunta_multiple.delay(payload)

    logger.info(f"Job encolado responder_pregunta_multiple id={task.id}")
    return RespuestaExito(code=200, objetos=[{"job_id": task.id}])