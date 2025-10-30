from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from utils.helpers import generar_embedding_openai
import json
import logging
import asyncio
from typing import List, Dict, Any, AsyncGenerator
from redis.asyncio import Redis
import hashlib
import uuid

router = APIRouter()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chat_router")

# Reutilizar funciones auxiliares del responder_pregunta.py
def dict_from_row(row) -> Dict:
    try:
        return dict(row._mapping)
    except Exception:
        return dict(row)

def _predict_sync(reranker, pairs: List[List[str]]) -> List[float]:
    return reranker.predict(pairs)

async def predict_reranker(request: Request, pairs: List[List[str]], batch_size: int = 16) -> List[float]:
    reranker = getattr(request.app.state, "reranker", None)
    if reranker is None:
        logger.error("Reranker no disponible")
        raise HTTPException(status_code=500, detail="Reranker no disponible")
    loop = asyncio.get_running_loop()
    scores: List[float] = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i: i + batch_size]
        batch_scores = await loop.run_in_executor(None, _predict_sync, reranker, batch)
        scores.extend(batch_scores)
    return scores

def normalize_distance(d: float) -> float:
    try:
        v = float(d)
    except Exception:
        return 1.0
    return max(0.0, min(1.0, v))

async def get_cached_embedding(redis: Redis, question: str) -> List[float] | None:
    cache_key = f"embedding:{hashlib.sha256(question.encode()).hexdigest()}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)
    embedding = generar_embedding_openai(question)
    if embedding:
        await redis.set(cache_key, json.dumps(embedding), ex=3600)
    return embedding

async def fetch_chunks(db: AsyncSession, embedding: List[float], file_id: int) -> List[Dict]:
    embedding_str = f"[{','.join(map(str, embedding))}]"
    
    query = text(
        """
            SELECT ca.id, ca.content, ca.embedding <=> CAST(:query_emb AS vector) AS distancia
            FROM files_chunks ca
            WHERE ca.files_id = :id
            ORDER BY distancia ASC
            LIMIT 20
        """
    )
    try:
        result = db.execute(query, {"query_emb": embedding_str, "id": file_id})
        return result.fetchall()
    except Exception:
        logger.exception("Error en búsqueda vectorial")
        raise HTTPException(status_code=500, detail="Error en búsqueda por vector")

async def stream_llm_response(request: Request, context: str, question: str, chat_history: List[Dict] = None) -> AsyncGenerator[str, None]:
    """Streaming response from LLM"""
    openai_client = request.app.state.openai_client
    
    # Construir el historial de mensajes
    messages = [
        {
            "role": "system",
            "content": f"""Eres un asistente especializado en analizar documentos. 
            
INSTRUCCIONES:
- Responde ÚNICAMENTE basándote en la información proporcionada en el CONTEXTO del documento.
- Si la información no está en el CONTEXTO, di claramente que no tienes esa información.
- Sé preciso y conciso en tus respuestas.
- Si el usuario pregunta sobre información que requiere análisis complejo, desglosa tu respuesta en partes.
- Mantén un tono profesional pero amigable.

CONTEXTO DEL DOCUMENTO:
{context}

Recuerda: Solo usa la información del CONTEXTO proporcionado."""
        }
    ]
    
    # Agregar historial de conversación si existe
    if chat_history:
        for turn in chat_history:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
    
    # Agregar la pregunta actual
    messages.append({"role": "user", "content": question})
    
    try:
        # Usar streaming para la respuesta
        stream = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=1500,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except asyncio.TimeoutError:
        yield "⏰ Lo siento, la solicitud está tomando demasiado tiempo. Por favor intenta nuevamente."
    except Exception as e:
        logger.error(f"Error en LLM streaming: {str(e)}")
        yield "❌ Lo siento, hubo un error procesando tu pregunta. Por favor intenta nuevamente."

async def generate_full_response(request: Request, context: str, question: str, chat_history: List[Dict] = None) -> str:
    """Versión no-streaming para compatibilidad"""
    openai_client = request.app.state.openai_client
    
    messages = [
        {
            "role": "system",
            "content": f"""Eres un asistente especializado en analizar documentos. Responde ÚNICAMENTE basándote en la información proporcionada en el CONTEXTO del documento.

CONTEXTO DEL DOCUMENTO:
{context}"""
        }
    ]
    
    if chat_history:
        for turn in chat_history:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
    
    messages.append({"role": "user", "content": question})
    
    try:
        completion = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=1500,
            ),
            timeout=45.0
        )
        return completion.choices[0].message.content.strip()
    except asyncio.TimeoutError:
        return "⏰ Lo siento, la solicitud está tomando demasiado tiempo. Por favor intenta nuevamente."
    except Exception as e:
        logger.error(f"Error en LLM: {str(e)}")
        return "❌ Lo siento, hubo un error procesando tu pregunta. Por favor intenta nuevamente."

@router.post("/chat_with_document")
async def chat_with_document(
    request: Request, 
    data: Dict[str, Any], 
    db: AsyncSession = Depends(get_db)
):
    """
    Endpoint principal para chat con documentos
    Soporta streaming y respuestas completas
    """
    try:
        file_id = data.get("file_id")
        question = data.get("message")
        chat_history = data.get("chat_history", [])
        use_streaming = data.get("streaming", True)
        
        if not file_id or not question:
            raise HTTPException(status_code=400, detail="Faltan file_id o message")
        
        logger.info(f"Iniciando chat para archivo {file_id}: {question[:100]}...")
        
        # Verificar que el archivo existe
        query_archivo = text("SELECT id, name FROM files WHERE id = :id")
        archivo_result = db.execute(query_archivo, {"id": file_id})
        archivo = archivo_result.fetchone()
        
        if not archivo:
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        # Generar embedding para la pregunta
        redis: Redis = request.app.state.redis
        embedding = await get_cached_embedding(redis, question)
        if not embedding:
            raise HTTPException(status_code=400, detail="No se pudo generar embedding para la pregunta")
        
        # Buscar chunks relevantes
        results = await fetch_chunks(db, embedding, file_id)
        if not results:
            raise HTTPException(status_code=404, detail="No se encontraron fragmentos relevantes en el documento")
        
        # Procesar chunks y aplicar reranking
        chunks_candidatos = [
            {"id": d.get("id"), "content": d.get("content", ""), "distancia": normalize_distance(d.get("distancia", 1.0))}
            for r in results if (d := dict_from_row(r))
        ]
        
        # Aplicar reranking
        pares = [[question, c["content"]] for c in chunks_candidatos]
        try:
            scores = await predict_reranker(request, pares, batch_size=8)
        except Exception:
            logger.exception("Error en reranking")
            # Continuar sin reranking en caso de error
            scores = [1.0] * len(chunks_candidatos)
        
        for i, chunk in enumerate(chunks_candidatos):
            chunk["score_reranking"] = float(scores[i]) if i < len(scores) else 1.0
            chunk["score_combinado"] = 0.6 * chunk["score_reranking"] + 0.4 * (1.0 - chunk["distancia"])
        
        # Seleccionar mejores chunks
        chunks_ordenados = sorted(chunks_candidatos, key=lambda x: x["score_combinado"], reverse=True)
        chunks_finales = [c for c in chunks_ordenados if c["score_combinado"] > 0.3] or chunks_ordenados[:3]
        
        # Construir contexto
        contexto = "\n\n".join([f"--- Fragmento {i+1} ---\n{chunk['content']}" for i, chunk in enumerate(chunks_finales[:3])])
        
        logger.info(f"Contexto preparado con {len(chunks_finales)} fragmentos para archivo {file_id}")
        
        # Preparar respuesta
        if use_streaming:
            # Devolver streaming response
            return StreamingResponse(
                stream_llm_response(request, contexto, question, chat_history),
                media_type="text/plain"
            )
        else:
            # Devolver respuesta completa
            respuesta = await generate_full_response(request, contexto, question, chat_history)
            return {
                "response": respuesta,
                "chunks_utilizados": len(chunks_finales),
                "distancia_promedio": sum(c["distancia"] for c in chunks_finales) / len(chunks_finales) if chunks_finales else 0,
                "file_id": file_id,
                "file_name": archivo.name if archivo else "Desconocido"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error inesperado en chat_with_document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@router.get("/chat_with_document/health")
async def health_check():
    """Endpoint de salud para el servicio de chat"""
    return {"status": "healthy", "service": "chat_with_document"}