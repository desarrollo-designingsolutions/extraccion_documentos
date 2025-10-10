from pydantic import BaseModel # type: ignore
from typing import List, Optional

class RespuestaExito(BaseModel):
    mensaje: str
    total_guardados: int
    total_chunks: int
    objetos: List[dict]