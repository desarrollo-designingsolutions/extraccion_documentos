from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from database import AsyncSessionLocal
import json
import logging
import asyncio
from typing import List, Dict, Any
from redis.asyncio import Redis
from openai import AsyncOpenAI
import re
import string

logger = logging.getLogger("job_scanear_multiple")


def dict_from_row(row) -> Dict:
    try:
        return dict(row._mapping)
    except Exception:
        return dict(row)


def _predict_sync(reranker, pairs: List[List[str]]) -> List[float]:
    return reranker.predict(pairs)


async def predict_reranker(request, pairs: List[List[str]], batch_size: int = 16) -> List[float]:
    reranker = getattr(request.app.state, "reranker", None)
    if reranker is None:
        logger.error("Reranker no disponible")
        raise Exception("Reranker no disponible")
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


# Simple Spanish stopwords (small set to avoid new deps)
STOPWORDS = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
    "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero",
    "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando",
}


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    # Lower, remove punctuation, split on whitespace
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    tokens = [t for t in re.split(r"\s+", text) if t and t not in STOPWORDS]
    return tokens


def lexical_overlap_score(query: str, chunk: str) -> float:
    q_tokens = tokenize(query)
    c_tokens = tokenize(chunk)
    if not q_tokens or not c_tokens:
        return 0.0
    inter = set(q_tokens).intersection(c_tokens)
    # normalize by query length to measure how much of the query is covered
    return len(inter) / max(1, len(set(q_tokens)))


async def fetch_chunks_by_nit(db: AsyncSession, embedding: List[float], nit: str) -> List[Dict]:
    embedding_str = f"[{','.join(map(str, embedding))}]"

    query = text(
    """
        SELECT ca.id, ca.content, ca.embedding <=> CAST(:query_emb AS vector) AS distancia
        FROM files_chunks ca
        JOIN files f ON f.id = ca.files_id
        WHERE f.nit = :nit
        ORDER BY distancia ASC
    LIMIT 50
        """
    )
    try:
        result = await db.execute(query, {"query_emb": embedding_str, "nit": nit})
        return result.fetchall()
    except Exception:
        logger.exception("Error en búsqueda vectorial por NIT")
        raise


async def call_llm(request, context: str) -> Dict:
    openai_client: AsyncOpenAI = request.app.state.openai_client

    system_content = """
        PRINCIPIOS
        1) Usa EXCLUSIVAMENTE el texto en CONTEXTO DEL DOCUMENTO (procede de búsqueda semántica por embeddings).
        2) El CONTEXTO puede contener fragmentos parciales, duplicados o contradictorios, separados por “----- DOC n -----”.
        3) NO inventes. Si un dato no aparece con claridad, déjalo vacío ("") o no incluyas la categoría.
        4) SALIDA = SOLO un objeto JSON VÁLIDO con el esquema, claves y orden exactos definidos abajo. Sin texto extra, sin Markdown, sin comentarios.
        5) La petición del servicio será: "clasifica los soportes presentes y extrae datos cuando existan". Devuelve SIEMPRE el JSON COMPLETO, incluyendo "resumen". Si no hay base suficiente para un resumen clínico, deja "resumen" = "".

        ESQUEMA Y ORDEN DE SALIDA (inmutable)
        {
        "prestacion_servicios_de_salud": [],
        "documentos_administrativos": [],
        "documentos_contractuales": [],
        "otros": [],
        "datos_prestacion_servicios": {
        "nombre_paciente": "",
        "tipo_documento": "",
        "numero_documento": "",
        "numero_factura": "",
        "fecha_expedicion_factura": "",
        "emisor_factura": "",
        "pagador": "",
        "autorizacion": []
        },
        "resumen": ""
        }
    """

    user_content = f"""
        {context}   (delimitado por “----- DOC n -----” entre fragmentos)

        PREGUNTA:
        clasifica los soportes presentes y extrae datos cuando existan
    """

    try:
        completion = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=800,
            ),
            timeout=30.0,
        )
        raw_text = completion.choices[0].message.content.strip()
        return json.loads(raw_text)
    except Exception:
        logger.exception("Error en LLM")
        raise


async def process_nit(request, nit: str, embedding: List[float], input_data, sem: asyncio.Semaphore) -> Dict[str, Any]:
    async with sem:
        async with AsyncSessionLocal() as task_db:
            # Debug count (async) by NIT
            query_debug = text("SELECT COUNT(*) as total FROM files_chunks ca JOIN files f ON f.id = ca.files_id WHERE f.nit = :nit")
            try:
                count_result = await task_db.execute(query_debug, {"nit": nit})
                count = count_result.scalar()
            except Exception:
                logger.exception("Error en query debug por NIT")
                raise

            if count == 0:
                query_archivo = text("SELECT id FROM files WHERE nit = :nit")
                try:
                    archivo_result = await task_db.execute(query_archivo, {"nit": nit})
                    if archivo_result.fetchone() is None:
                        raise Exception(f"NIT {nit} no encontrado")
                except Exception:
                    logger.exception("Error comprobando NIT")
                    raise

            # Fetch chunks while session is open
            raw_results = await fetch_chunks_by_nit(task_db, embedding, nit)
            results = [dict_from_row(r) for r in raw_results]

        # Después de cerrar la sesión, seguimos con datos independientes de la sesión
        if not results:
            raise Exception(f"No chunks encontrados para NIT {nit}")

        chunks_candidatos = [
            {"id": d.get("id"), "content": d.get("content", ""), "distancia": normalize_distance(d.get("distancia", 1.0))}
            for d in results
        ]

        # Reranking
        pares = [[input_data.pregunta, c["content"]] for c in chunks_candidatos]
        scores = await predict_reranker(request, pares, batch_size=8)

        # Combine signals: reranker (alpha), distance (beta), lexical overlap (gamma)
        ALPHA = 0.75  # reranker weight (most important)
        BETA = 0.15   # distance weight
        GAMMA = 0.10  # lexical overlap weight

        for i, chunk in enumerate(chunks_candidatos):
            chunk["score_reranking"] = float(scores[i])
            lex = lexical_overlap_score(input_data.pregunta, chunk["content"]) if input_data.pregunta else 0.0
            chunk["lexical_score"] = lex
            chunk["score_combinado"] = (
                ALPHA * chunk["score_reranking"] +
                BETA * (1.0 - chunk["distancia"]) +
                GAMMA * chunk["lexical_score"]
            )

        chunks_ordenados = sorted(chunks_candidatos, key=lambda x: x["score_combinado"], reverse=True)

        # adapt number of chunks selected: prefer top by score, but allow more context (up to 5)
        top_n = max(3, min(5, len(chunks_ordenados)))
        chunks_finales = [c for c in chunks_ordenados if c["score_combinado"] > input_data.score_threshold] or chunks_ordenados[:top_n]

        if not chunks_finales:
            raise Exception(f"Chunks no relevantes para NIT {nit}")

        chunks_contexto = chunks_finales[:3]
        contexto = "\n\n".join([f"[Fragmento {i + 1}]: {chunk['content']}" for i, chunk in enumerate(chunks_contexto)])
        distancia_promedio = sum(chunk["distancia"] for chunk in chunks_contexto) / len(chunks_contexto)
        chunk_ids = [chunk["id"] for chunk in chunks_contexto]

        # Llamada LLM
        parsed = await call_llm(request, contexto)

        return {
            "nit": nit,
            "respuesta": parsed,
            "distancia": distancia_promedio,
            "chunks_utilizados": len(chunks_contexto),
            "chunk_ids": chunk_ids,
        }
