from pydantic import BaseModel # type: ignore
from typing import List, Optional

class RespuestaExito(BaseModel):
    mensaje: str
    total_guardados: int
    total_chunks: int
    objetos: List[dict]

class RespuestaLLM(BaseModel):
    respuesta: str
    distancia: float
    chunks_utilizados: int

class PreguntaInput(BaseModel):
    id: int
    pregunta: str
    score_threshold: float = 0.3