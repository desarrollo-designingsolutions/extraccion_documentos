from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import get_db
from models import TemporaryFiles, TemporaryFilesChunks
from utils.helpers import extract_text, split_text, generar_embeddings_async
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging
from pydantic import BaseModel
from sqlalchemy.orm import Session

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
    respuesta: Dict
    distancia: float
    chunks_utilizados: int
    session_id: str

# Función para procesar archivos en background
async def process_temporary_file(
    file_content: bytes,
    filename: str,
    session_id: str,
    file_id: int,
    db: Session
):
    """Procesa un archivo temporal: extrae texto, divide en chunks y genera embeddings"""
    try:
        # Extraer texto (sincrónico)
        text = extract_text(file_content, filename)
        if not text:
            logger.warning(f"No se pudo extraer texto del archivo {filename}")
            return

        # Dividir en chunks (sincrónico)
        chunks = split_text(text)
        
        if not chunks:
            logger.warning(f"No se pudieron generar chunks para {filename}")
            return

        # Generar embeddings de forma asíncrona para todos los chunks
        embeddings = await generar_embeddings_async(chunks)
        
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
        
        # Actualizar el registro del archivo con el texto extraído
        db.query(TemporaryFiles).filter(TemporaryFiles.id == file_id).update({
            'text_extracted': text[:10000]  # Guardar primeros 10k chars como muestra
        })
        db.commit()
        
    except Exception as e:
        logger.error(f"Error procesando archivo temporal {filename}: {str(e)}")
        # Revertir en caso de error
        db.rollback()
        raise


@router.post("/upload_temporary", response_model=NotebookUploadResponse)
async def upload_temporary_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Sube un archivo temporal para NotebookLM"""
    
    # Generar session_id si no se proporciona
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Leer el archivo
    contents = await file.read()
    
    # Crear registro en la base de datos
    expires_at = datetime.utcnow() + timedelta(hours=24)  # Expira en 24 horas
    
    temporary_file = TemporaryFiles(
        session_id=session_id,
        name=file.filename,
        original_filename=file.filename,
        size=len(contents),
        expires_at=expires_at
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
        db
    )
    
    return NotebookUploadResponse(
        session_id=session_id,
        file_id=temporary_file.id,
        filename=file.filename,
        chunks_created=0  # Se actualizará después del procesamiento
    )

@router.get("/session_files/{session_id}")
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
                "chunks_count": len(f.chunks) if f.chunks else 0
            }
            for f in files
        ]
    }

@router.delete("/session/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Elimina una sesión temporal y todos sus archivos"""
    
    files = db.query(TemporaryFiles).filter(TemporaryFiles.session_id == session_id).all()
    
    for file in files:
        db.delete(file)
    
    db.commit()
    
    return {"message": f"Sesión {session_id} eliminada correctamente"}

# Importar funciones necesarias de responder_pregunta.py
from routes.responder_pregunta import (
    get_cached_embedding,
    predict_reranker,
    normalize_distance,
    call_llm,
    dict_from_row,
    validate_parsed_response
)

async def fetch_temporary_chunks(db: Session, embedding: List[float], session_id: str) -> List[Dict]:
    """Busca chunks temporales similares al embedding"""
    embedding_str = f"[{','.join(map(str, embedding))}]"

    query = text(
        """
        SELECT tc.id, tc.content, tc.embedding <=> CAST(:query_emb AS vector) AS distancia
        FROM temporary_files_chunks tc
        JOIN temporary_files tf ON tc.temporary_files_id = tf.id
        WHERE tf.session_id = :session_id 
          AND tf.expires_at > NOW()
        ORDER BY distancia ASC
        LIMIT 20
        """
    )
    try:
        result = db.execute(query, {"query_emb": embedding_str, "session_id": session_id})
        return result.fetchall()
    except Exception:
        logger.exception("Error en búsqueda vectorial temporal")
        raise HTTPException(status_code=500, detail="Error en búsqueda por vector temporal")

@router.post("/ask", response_model=NotebookAskResponse)
async def notebook_ask(
    request: Request,
    input_data: NotebookAskRequest,
    db: Session = Depends(get_db)
):
    """Hace preguntas sobre los archivos temporales de una sesión"""
    
    # Verificar que la sesión tenga archivos
    session_files = db.query(TemporaryFiles).filter(
        TemporaryFiles.session_id == input_data.session_id,
        TemporaryFiles.expires_at > datetime.utcnow()
    ).count()
    
    if session_files == 0:
        raise HTTPException(status_code=404, detail="Sesión no encontrada o expirada")
    
    # Usar la misma lógica de responder_pregunta pero con chunks temporales
    redis = request.app.state.redis
    embedding = await get_cached_embedding(redis, input_data.question)
    if not embedding or not isinstance(embedding, list):
        logger.error("Embedding inválido")
        raise HTTPException(status_code=400, detail="No se pudo generar embedding")

    # Fetch chunks temporales
    results = await fetch_temporary_chunks(db, embedding, input_data.session_id)
    if not results:
        raise HTTPException(status_code=404, detail="No se encontraron chunks relevantes en los archivos temporales")

    # Aplicar la misma lógica de ranking que en responder_pregunta
    chunks_candidatos = [
        {"id": d.get("id"), "content": d.get("content", ""), "distancia": normalize_distance(d.get("distancia", 1.0))}
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
        chunk["score_combinado"] = 0.6 * chunk["score_reranking"] + 0.4 * (1.0 - chunk["distancia"])

    chunks_ordenados = sorted(chunks_candidatos, key=lambda x: x["score_combinado"], reverse=True)
    chunks_finales = [c for c in chunks_ordenados if c["score_combinado"] > input_data.score_threshold] or chunks_ordenados[:2]

    if not chunks_finales:
        raise HTTPException(status_code=400, detail="No se encontraron chunks relevantes")

    chunks_contexto = chunks_finales[:3]
    contexto = "\n\n".join([f"[Fragmento {i+1}]: {chunk['content']}" for i, chunk in enumerate(chunks_contexto)])
    distancia_promedio = sum(c["distancia"] for c in chunks_contexto) / len(chunks_contexto)

    # Llamada LLM
    parsed = await call_llm(request, contexto)
    try:
        parsed = validate_parsed_response(parsed, contexto)
    except Exception:
        logger.exception("Error validando respuesta LLM (notebook)")

    return NotebookAskResponse(
        respuesta=parsed,
        distancia=distancia_promedio,
        chunks_utilizados=len(chunks_contexto),
        session_id=input_data.session_id
    )


@router.post("/cleanup_expired")
async def cleanup_expired_files(db: Session = Depends(get_db)):
    """Limpia archivos temporales expirados (para llamar periódicamente)"""
    
    deleted_count = db.query(TemporaryFiles).filter(
        TemporaryFiles.expires_at <= datetime.utcnow()
    ).delete()
    
    db.commit()
    
    return {"message": f"Eliminados {deleted_count} archivos temporales expirados"}