from pydantic import BaseModel
from typing import List, Optional

class RespuestaExito(BaseModel):
    code: int
    objetos: List[dict]

class RespuestaLLM(BaseModel):
    respuesta: dict
    distancia: float
    chunks_utilizados: int

class PreguntaInput(BaseModel):
    id: int
    pregunta: str
    score_threshold: float = 0.3