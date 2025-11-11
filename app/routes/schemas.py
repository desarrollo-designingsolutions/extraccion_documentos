from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

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

class ConversationMessageBase(BaseModel):
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class ConversationMessageCreate(ConversationMessageBase):
    pass

class ConversationMessageResponse(ConversationMessageBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class ConversationSessionBase(BaseModel):
    session_id: str
    file_id: Optional[int] = None
    title: Optional[str] = None

class ConversationSessionCreate(ConversationSessionBase):
    pass

class ConversationSessionResponse(BaseModel):
    id: int
    session_id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    messages: List[ConversationMessageResponse] = []
    
    class Config:
        from_attributes = True

class ChatHistoryRequest(BaseModel):
    session_id: str
    limit: Optional[int] = 10

class ChatWithHistoryRequest(BaseModel):
    session_id: str
    question: str
    score_threshold: float = 0.5
    use_history: bool = True