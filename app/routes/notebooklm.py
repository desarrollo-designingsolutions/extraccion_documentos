from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import text
from openai import AsyncOpenAI
import asyncio
from database import get_db
from models import TemporaryFiles, TemporaryFilesChunks, ConversationSession, ConversationMessage
from utils.helpers import extract_text, split_text, generar_embeddings_async
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging
from pydantic import BaseModel
from routes.responder_pregunta import (
    get_cached_embedding,
    predict_reranker,
    normalize_distance,
    dict_from_row,
)


logger = logging.getLogger(__name__)

router = APIRouter()

# Schemas para request/response
class NotebookAskRequest(BaseModel):
    session_id: str
    question: str
    score_threshold: float = 0.5

class NotebookUploadResponse(BaseModel):
    session_id: str
    file_id: int
    filename: str
    chunks_created: int

class NotebookAskResponse(BaseModel):
    respuesta: str
    distancia: float
    chunks_utilizados: int
    session_id: str

class ChatRequest(BaseModel):
    session_id: str
    conversation_id: Optional[int] = None
    question: str
    score_threshold: float = 0.5
    use_history: bool = True

class CreateConversationRequest(BaseModel):
    session_id: str
    name: Optional[str] = "Nueva Conversación"

class UpdateConversationRequest(BaseModel):
    name: str

# Función para procesar archivos en background
async def process_temporary_file(
    file_content: bytes,
    filename: str,
    session_id: str,
    file_id: int,
    conversation_id: Optional[int] = None,
    db: Session = None
):
    """Procesa un archivo temporal: extrae texto, divide en chunks y genera embeddings"""
    try:
        # Extraer texto
        text = extract_text(file_content, filename)
        if not text:
            logger.warning(f"No se pudo extraer texto del archivo {filename}")
            return

        # Dividir en chunks
        chunks = split_text(text)
        
        if not chunks:
            logger.warning(f"No se pudieron generar chunks para {filename}")
            return

        # Generar embeddings de forma asíncrona
        embeddings = await generar_embeddings_async(chunks, 2)
        
        # Guardar chunks con sus embeddings
        chunks_guardados = 0
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding is None:
                logger.error(f"No se pudo generar embedding para el chunk {i} del archivo {filename}")
                continue

            chunk_record = TemporaryFilesChunks(
                temporary_files_id=file_id,
                content=chunk,
                chunk_number=i,
                embedding=embedding
            )
            db.add(chunk_record)
            chunks_guardados += 1

        db.commit()
        logger.info(f"Procesados {chunks_guardados} chunks para archivo temporal {filename}")
        
        # Actualizar el registro del archivo
        db.query(TemporaryFiles).filter(TemporaryFiles.id == file_id).update({
            'text_extracted': text[:10000],
            'conversation_id': conversation_id
        })
        db.commit()
        
    except Exception as e:
        logger.error(f"Error procesando archivo temporal {filename}: {str(e)}")
        db.rollback()
        raise

@router.post("/upload", response_model=NotebookUploadResponse)
async def upload_temporary_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Sube un archivo temporal para NotebookLM"""
    
    # Generar session_id si no se proporciona
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Procesar conversation_id
    conversation_id_int = None
    if conversation_id and conversation_id != "null":
        try:
            conversation_id_int = int(conversation_id)
        except (ValueError, TypeError):
            conversation_id_int = None
    
    # Leer el archivo
    contents = await file.read()
    
    # Crear registro en la base de datos
    expires_at = datetime.utcnow() + timedelta(hours=24)
    
    temporary_file = TemporaryFiles(
        session_id=session_id,
        name=file.filename,
        original_filename=file.filename,
        size=len(contents),
        expires_at=expires_at,
        conversation_id=conversation_id_int
    )
    
    db.add(temporary_file)
    db.commit()
    db.refresh(temporary_file)
    
    # Procesar el archivo en background
    background_tasks.add_task(
        process_temporary_file,
        contents,
        file.filename,
        session_id,
        temporary_file.id,
        conversation_id_int,
        db
    )
    
    return NotebookUploadResponse(
        session_id=session_id,
        file_id=temporary_file.id,
        filename=file.filename,
        chunks_created=0
    )

# ===== APIs PARA GESTIÓN DE CONVERSACIONES =====

@router.get("/conversations/{session_id}")
async def get_conversations(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Obtiene todas las conversaciones de una sesión"""
    
    conversations = db.query(ConversationSession).filter(
        ConversationSession.session_id == session_id
    ).order_by(ConversationSession.updated_at.desc()).all()
    
    return {
        "session_id": session_id,
        "conversations": [
            {
                "id": conv.id,
                "name": conv.title,
                "session_id": conv.session_id,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "message_count": db.query(ConversationMessage).filter(
                    ConversationMessage.session_id == conv.id
                ).count()
            }
            for conv in conversations
        ]
    }

@router.post("/conversations")
async def create_conversation(
    request: CreateConversationRequest,
    db: Session = Depends(get_db)
):
    """Crea una nueva conversación"""
    
    conversation = ConversationSession(
        session_id=request.session_id,
        title=request.name or "Nueva Conversación"
    )
    
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    
    return {
        "conversation_id": conversation.id,
        "name": conversation.title,
        "session_id": conversation.session_id,
        "created_at": conversation.created_at
    }

@router.get("/conversation/{conversation_id}")
async def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Obtiene una conversación específica con sus mensajes y archivos"""
    
    conversation = db.query(ConversationSession).filter(
        ConversationSession.id == conversation_id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    
    # Obtener mensajes
    messages = db.query(ConversationMessage).filter(
        ConversationMessage.session_id == conversation_id
    ).order_by(ConversationMessage.created_at.asc()).all()
    
    # Obtener archivos asociados
    files = db.query(TemporaryFiles).filter(
        TemporaryFiles.conversation_id == conversation_id,
        TemporaryFiles.expires_at > datetime.utcnow()
    ).all()
    
    return {
        "id": conversation.id,
        "name": conversation.title,
        "session_id": conversation.session_id,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at,
                "metadata": msg.message_metadata
            }
            for msg in messages
        ],
        "files": [
            {
                "id": f.id,
                "name": f.name,
                "original_filename": f.original_filename,
                "size": f.size,
                "created_at": f.created_at
            }
            for f in files
        ]
    }

@router.put("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: int,
    request: UpdateConversationRequest,
    db: Session = Depends(get_db)
):
    """Actualiza el nombre de una conversación"""
    
    conversation = db.query(ConversationSession).filter(
        ConversationSession.id == conversation_id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    
    conversation.title = request.name
    conversation.updated_at = datetime.utcnow()
    
    db.commit()
    
    return {
        "conversation_id": conversation.id,
        "name": conversation.title,
        "updated_at": conversation.updated_at
    }

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Elimina una conversación y todos sus mensajes"""
    
    conversation = db.query(ConversationSession).filter(
        ConversationSession.id == conversation_id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    
    # Eliminar mensajes de la conversación
    db.query(ConversationMessage).filter(
        ConversationMessage.session_id == conversation_id
    ).delete()
    
    # Eliminar la conversación
    db.delete(conversation)
    db.commit()
    
    return {"message": f"Conversación {conversation_id} eliminada correctamente"}

@router.delete("/conversations/{conversation_id}/messages")
async def clear_conversation_messages(
    conversation_id: int,
    db: Session = Depends(get_db)
):
    """Limpia todos los mensajes de una conversación"""
    
    conversation = db.query(ConversationSession).filter(
        ConversationSession.id == conversation_id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    
    # Eliminar mensajes
    deleted_count = db.query(ConversationMessage).filter(
        ConversationMessage.session_id == conversation_id
    ).delete()
    
    conversation.updated_at = datetime.utcnow()
    db.commit()
    
    return {
        "message": f"Eliminados {deleted_count} mensajes",
        "conversation_id": conversation_id
    }

# ===== APIs PRINCIPALES OPTIMIZADAS =====

@router.get("/files/{session_id}")
async def get_session_files(session_id: str, db: Session = Depends(get_db)):
    """Obtiene los archivos de una sesión temporal"""
    
    files = db.query(TemporaryFiles).filter(
        TemporaryFiles.session_id == session_id,
        TemporaryFiles.expires_at > datetime.utcnow()
    ).all()
    
    return {
        "session_id": session_id,
        "files": [
            {
                "id": f.id,
                "name": f.name,
                "original_filename": f.original_filename,
                "size": f.size,
                "created_at": f.created_at,
                "chunks_count": len(f.chunks) if f.chunks else 0,
                "conversation_id": f.conversation_id
            }
            for f in files
        ]
    }

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Elimina una sesión temporal y todos sus archivos y conversaciones"""
    
    # Obtener todas las conversaciones de la sesión
    conversations = db.query(ConversationSession).filter(
        ConversationSession.session_id == session_id
    ).all()
    
    # Eliminar mensajes de cada conversación
    for conv in conversations:
        db.query(ConversationMessage).filter(
            ConversationMessage.session_id == conv.id
        ).delete()
    
    # Eliminar las conversaciones
    db.query(ConversationSession).filter(
        ConversationSession.session_id == session_id
    ).delete()
    
    # Eliminar archivos temporales
    files = db.query(TemporaryFiles).filter(TemporaryFiles.session_id == session_id).all()
    for file in files:
        db.delete(file)
    
    db.commit()
    
    return {"message": f"Sesión {session_id} eliminada correctamente"}

async def fetch_temporary_chunks(
    db: Session, 
    embedding: List[float], 
    session_id: str, 
    conversation_id: Optional[int] = None
) -> List[Dict]:
    """Busca chunks temporales similares al embedding"""
    embedding_str = f"[{','.join(map(str, embedding))}]"

    # Construir query base
    query_base = """
        SELECT tc.id, tc.content, tc.embedding <=> CAST(:query_emb AS vector) AS distancia
        FROM temporary_files_chunks tc
        JOIN temporary_files tf ON tc.temporary_files_id = tf.id
        WHERE tf.session_id = :session_id 
          AND tf.expires_at > NOW()
    """
    
    # Agregar filtro por conversación si se proporciona
    if conversation_id:
        query_base += " AND tf.conversation_id = :conversation_id"
    
    query_base += " ORDER BY distancia ASC LIMIT 20"
    
    query = text(query_base)
    
    try:
        params = {"query_emb": embedding_str, "session_id": session_id}
        if conversation_id:
            params["conversation_id"] = conversation_id
            
        result = db.execute(query, params)
        return result.fetchall()
    except Exception:
        logger.exception("Error en búsqueda vectorial temporal")
        raise HTTPException(status_code=500, detail="Error en búsqueda por vector temporal")

@router.post("/chat", response_model=NotebookAskResponse)
async def notebook_chat(
    request: Request,
    input_data: ChatRequest,
    db: Session = Depends(get_db)
):
    """Endpoint principal para chat con documentos - reemplaza ask y ask_with_history"""
    
    # Verificar que la sesión tenga archivos
    session_files = db.query(TemporaryFiles).filter(
        TemporaryFiles.session_id == input_data.session_id,
        TemporaryFiles.expires_at > datetime.utcnow()
    ).count()
    
    if session_files == 0:
        raise HTTPException(status_code=404, detail="Sesión no encontrada o expirada")
    
    # Obtener o crear sesión de conversación
    if input_data.conversation_id:
        conversation_session = db.query(ConversationSession).filter(
            ConversationSession.id == input_data.conversation_id,
            ConversationSession.session_id == input_data.session_id
        ).first()
        
        if not conversation_session:
            raise HTTPException(status_code=404, detail="Conversación no encontrada")
    else:
        conversation_session = ConversationSession(
            session_id=input_data.session_id,
            title=f"Conversación {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        )
        db.add(conversation_session)
        db.commit()
        db.refresh(conversation_session)
    
    # Obtener historial reciente si se solicita
    history_context = ""
    if input_data.use_history:
        recent_messages = db.query(ConversationMessage).filter(
            ConversationMessage.session_id == conversation_session.id
        ).order_by(ConversationMessage.created_at.desc()).limit(6).all()
        
        recent_messages.reverse()
        for msg in recent_messages:
            role_label = "Usuario" if msg.role == "user" else "Asistente"
            history_context += f"{role_label}: {msg.content}\n"
        
        if history_context:
            history_context = "## Historial de conversación anterior:\n" + history_context + "\n"

    # Guardar pregunta del usuario
    user_message = ConversationMessage(
        session_id=conversation_session.id,
        role="user",
        content=input_data.question
    )
    db.add(user_message)
    
    # Procesar la pregunta
    redis = request.app.state.redis
    embedding = await get_cached_embedding(redis, input_data.question)
    if not embedding or not isinstance(embedding, list):
        logger.error("Embedding inválido")
        raise HTTPException(status_code=400, detail="No se pudo generar embedding")

    # Buscar chunks relevantes
    results = await fetch_temporary_chunks(
        db, embedding, input_data.session_id, conversation_session.id
    )
    
    if not results:
        raise HTTPException(
            status_code=404, 
            detail="No se encontraron chunks relevantes en los archivos temporales"
        )

    # Procesar y rankear chunks
    chunks_candidatos = [
        {
            "id": d.get("id"), 
            "content": d.get("content", ""), 
            "distancia": normalize_distance(d.get("distancia", 1.0))
        }
        for r in results if (d := dict_from_row(r))
    ]

    # Reranking
    pares = [[input_data.question, c["content"]] for c in chunks_candidatos]
    try:
        scores = await predict_reranker(request, pares, batch_size=8)
    except Exception:
        logger.exception("Error en reranking temporal")
        raise HTTPException(status_code=500, detail="Error en reranking")

    for i, chunk in enumerate(chunks_candidatos):
        chunk["score_reranking"] = float(scores[i])
        chunk["score_combinado"] = (
            0.6 * chunk["score_reranking"] + 
            0.4 * (1.0 - chunk["distancia"])
        )

    chunks_ordenados = sorted(
        chunks_candidatos, 
        key=lambda x: x["score_combinado"], 
        reverse=True
    )
    
    chunks_finales = [
        c for c in chunks_ordenados 
        if c["score_combinado"] > input_data.score_threshold
    ] or chunks_ordenados[:2]

    if not chunks_finales:
        raise HTTPException(
            status_code=400, 
            detail="No se encontraron chunks relevantes"
        )

    chunks_contexto = chunks_finales[:3]
    contexto_documentos = "\n\n".join([
        f"[Fragmento {i+1}]: {chunk['content']}" 
        for i, chunk in enumerate(chunks_contexto)
    ])
    
    # Combinar historial con contexto de documentos
    contexto_completo = (
        history_context + 
        "## Contexto actual de los documentos:\n" + 
        contexto_documentos
    )
    distancia_promedio = sum(c["distancia"] for c in chunks_contexto) / len(chunks_contexto)

    # Llamada LLM con contexto completo
    respuesta_texto = await call_llm_markdown(
        request, contexto_completo, input_data.question, history_context
    )

    # Guardar respuesta del asistente
    assistant_message = ConversationMessage(
        session_id=conversation_session.id,
        role="assistant",
        content=respuesta_texto,
        message_metadata={
            "chunks_utilizados": len(chunks_contexto),
            "distancia_promedio": distancia_promedio,
            "modelo": "gpt-4o-mini",
            "tipo_respuesta": "markdown"
        }
    )
    db.add(assistant_message)
    
    # Actualizar timestamp de la sesión
    conversation_session.updated_at = datetime.utcnow()
    db.commit()

    return NotebookAskResponse(
        respuesta=respuesta_texto,
        distancia=distancia_promedio,
        chunks_utilizados=len(chunks_contexto),
        session_id=input_data.session_id
    )

async def call_llm_markdown(
    request: Request, 
    context: str, 
    current_question: str, 
    history: str = ""
) -> str:
    """Genera respuesta en formato Markdown usando LLM"""
    openai_client: AsyncOpenAI = request.app.state.openai_client

    system_content = """
        Eres un asistente especializado en analizar documentos del sector salud. 
        Responde preguntas basándote EXCLUSIVAMENTE en el contenido proporcionado.

        INSTRUCCIONES:
        - Usa **Markdown** para estructura clara
        - Organiza información con negritas, listas y encabezados
        - Sé conciso y profesional
        - NO inventes información
        - Si no hay suficiente información, indícalo claramente
    """

    user_content = f"""
        ## CONTEXTO DE DOCUMENTOS:
        {context}

        ## HISTORIAL DE CONVERSACIÓN:
        {history}

        ## PREGUNTA ACTUAL:
        {current_question}

        Responde basándote ÚNICAMENTE en el contexto proporcionado.
    """

    try:
        logger.info(f"Enviando pregunta a LLM: {current_question}")

        completion = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                max_tokens=2000,
            ),
            timeout=60.0
        )
        
        respuesta = completion.choices[0].message.content.strip()
        logger.info(f"Respuesta LLM recibida: {len(respuesta)} caracteres")
        
        return respuesta
        
    except asyncio.TimeoutError:
        logger.error("Timeout en llamada a LLM")
        raise HTTPException(status_code=504, detail="Timeout en llamada a LLM")
    except Exception as e:
        logger.error(f"Error en LLM: {str(e)}")
        raise HTTPException(status_code=500, detail="Error en LLM")

@router.post("/cleanup")
async def cleanup_expired_files(db: Session = Depends(get_db)):
    """Limpia archivos temporales expirados"""
    
    deleted_count = db.query(TemporaryFiles).filter(
        TemporaryFiles.expires_at <= datetime.utcnow()
    ).delete()
    
    db.commit()
    
    return {"message": f"Eliminados {deleted_count} archivos temporales expirados"}