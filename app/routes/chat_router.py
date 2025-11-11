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

# Constantes
MAX_FILES_PER_CHAT = 10  # Límite máximo de archivos por chat

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

async def validate_files_exist(db: AsyncSession, file_ids: List[int]) -> List[Dict]:
    """Validar que todos los archivos existen y obtener sus nombres"""
    if not file_ids:
        return []
    
    query_archivo = text("SELECT id, name FROM files WHERE id IN :ids")
    archivo_result = db.execute(query_archivo, {"ids": tuple(file_ids)})
    archivos = [dict_from_row(row) for row in archivo_result.fetchall()]
    
    if len(archivos) != len(file_ids):
        found_ids = {a['id'] for a in archivos}
        not_found = set(file_ids) - found_ids
        raise HTTPException(
            status_code=404, 
            detail=f"Archivos no encontrados: {not_found}"
        )
    
    return archivos

async def fetch_chunks_multiple_files(
    db: AsyncSession, 
    embedding: List[float], 
    file_ids: List[int],
    chunks_per_file: int = 10
) -> List[Dict]:
    """Buscar chunks relevantes en múltiples archivos"""
    embedding_str = f"[{','.join(map(str, embedding))}]"
    
    query = text(
        """
        WITH ranked_chunks AS (
            SELECT 
                ca.id, 
                ca.content, 
                ca.files_id,
                ca.embedding <=> CAST(:query_emb AS vector) AS distancia,
                ROW_NUMBER() OVER (PARTITION BY ca.files_id ORDER BY ca.embedding <=> CAST(:query_emb AS vector)) as rank
            FROM files_chunks ca
            WHERE ca.files_id IN :ids
        )
        SELECT id, content, files_id, distancia
        FROM ranked_chunks
        WHERE rank <= :chunks_per_file
        ORDER BY distancia ASC
        LIMIT 50
        """
    )
    
    try:
        result = db.execute(query, {
            "query_emb": embedding_str, 
            "ids": tuple(file_ids),
            "chunks_per_file": chunks_per_file
        })
        return [dict_from_row(row) for row in result.fetchall()]
    except Exception:
        logger.exception("Error en búsqueda vectorial múltiple")
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
                - Cuando uses información de diferentes documentos, indica claramente de cuál documento proviene cada información.

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
    Soporta un solo archivo o múltiples archivos
    """
    try:
        # Soporte para file_id (individual) y file_ids (múltiple)
        file_id = data.get("file_id")
        file_ids = data.get("file_ids", [])
        question = data.get("message")
        chat_history = data.get("chat_history", [])
        use_streaming = data.get("streaming", True)
        
        # Determinar lista de archivos
        final_file_ids = []
        if file_ids:
            if isinstance(file_ids, list):
                final_file_ids = file_ids
            else:
                final_file_ids = [file_ids]
        elif file_id:
            final_file_ids = [file_id]
        
        if not final_file_ids or not question:
            raise HTTPException(status_code=400, detail="Faltan file_id/file_ids o message")
        
        # Validar límite de archivos
        if len(final_file_ids) > MAX_FILES_PER_CHAT:
            raise HTTPException(
                status_code=400, 
                detail=f"Máximo {MAX_FILES_PER_CHAT} archivos permitidos para chat simultáneo"
            )
        
        logger.info(f"Iniciando chat para archivos {final_file_ids}: {question[:100]}...")
        
        # Validar que los archivos existen
        archivos = await validate_files_exist(db, final_file_ids)
        
        # Generar embedding para la pregunta
        redis: Redis = request.app.state.redis
        embedding = await get_cached_embedding(redis, question)
        if not embedding:
            raise HTTPException(status_code=400, detail="No se pudo generar embedding para la pregunta")
        
        # Buscar chunks relevantes en todos los documentos
        results = await fetch_chunks_multiple_files(db, embedding, final_file_ids)
        if not results:
            raise HTTPException(
                status_code=404, 
                detail="No se encontraron fragmentos relevantes en los documentos"
            )
        
        # Procesar chunks y aplicar reranking
        chunks_candidatos = [
            {
                "id": d.get("id"), 
                "content": d.get("content", ""), 
                "files_id": d.get("files_id"),
                "distancia": normalize_distance(d.get("distancia", 1.0))
            }
            for d in results
        ]
        
        # Aplicar reranking
        pares = [[question, c["content"]] for c in chunks_candidatos]
        try:
            scores = await predict_reranker(request, pares, batch_size=2)
        except Exception:
            logger.exception("Error en reranking")
            scores = [1.0] * len(chunks_candidatos)
        
        for i, chunk in enumerate(chunks_candidatos):
            chunk["score_reranking"] = float(scores[i]) if i < len(scores) else 1.0
            chunk["score_combinado"] = 0.6 * chunk["score_reranking"] + 0.4 * (1.0 - chunk["distancia"])
        
        # Seleccionar mejores chunks
        chunks_ordenados = sorted(chunks_candidatos, key=lambda x: x["score_combinado"], reverse=True)
        chunks_finales = [c for c in chunks_ordenados if c["score_combinado"] > 0.3] or chunks_ordenados[:5]
        
        # Construir contexto con información de origen
        contexto_parts = []
        for i, chunk in enumerate(chunks_finales[:5]):
            # Encontrar nombre del archivo para este chunk
            archivo_nombre = next(
                (a['name'] for a in archivos if a['id'] == chunk['files_id']), 
                f"Documento {chunk['files_id']}"
            )
            contexto_parts.append(
                f"--- Fragmento {i+1} (de {archivo_nombre}) ---\n{chunk['content']}"
            )
        
        contexto = "\n\n".join(contexto_parts)
        
        logger.info(f"Contexto preparado con {len(chunks_finales)} fragmentos de {len(archivos)} archivos")
        
        # Preparar respuesta
        if use_streaming:
            return StreamingResponse(
                stream_llm_response(request, contexto, question, chat_history),
                media_type="text/plain"
            )
        else:
            respuesta = await generate_full_response(request, contexto, question, chat_history)
            return {
                "response": respuesta,
                "chunks_utilizados": len(chunks_finales),
                "archivos_utilizados": len(set(c['files_id'] for c in chunks_finales)),
                "total_archivos": len(archivos),
                "file_ids": final_file_ids,
                "file_names": [a['name'] for a in archivos]
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