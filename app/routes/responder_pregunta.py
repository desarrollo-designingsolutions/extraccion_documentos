from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from .schemas import RespuestaLLM, PreguntaInput
from app.utils.helpers import generar_embedding_openai
import json
import logging
import asyncio
from typing import List, Dict, Any
from redis.asyncio import Redis  # Import corregido
from openai import AsyncOpenAI  # Para tipado
import hashlib
from app.utils.helpers import validate_parsed_response

router = APIRouter()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("job_scanear")

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
        await redis.set(cache_key, json.dumps(embedding), ex=3600)  # Cache 1 hora
    return embedding

async def fetch_chunks(db: AsyncSession, embedding: List[float], file_id: int) -> List[Dict]:
    # Convert list to PG vector string format
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

async def call_llm(request: Request, context: str) -> Dict:
    openai_client: AsyncOpenAI = request.app.state.openai_client

    # Primero definimos el system content como string constante
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

        OBJETIVO
        Clasificar los tipos documentales presentes y extraer datos clave del CONTEXTO. Generar un resumen estructurado SOLO con información soportada por el CONTEXTO. Si algo no está, usar "" o [] según corresponda.

        CATÁLOGO DE TIPOS (palabras clave y reglas)
        Detecta si existen documentos de los siguientes tipos (insensible a mayúsculas, tildes, plural/singular y pequeñas variaciones). Agrega SOLO los detectados con los literales exactos:

        1) "prestacion_servicios_de_salud"
        - "Factura de servicios médicos"
        Indicadores: factura, cuenta de cobro/cta de cobro, nro/número de factura, prefijo+consecutivo.
        - "Autorizaciones"
        Indicadores: autorización, nro/código/radicado de autorización.
        - "Historia clínica"
        Indicadores: historia clínica, epicrisis, evolución, nota de ingreso/egreso, enfermería.
        - "Evidencia de entrega"
        Indicadores: exámenes/resultados (laboratorio, imagenología, rx, tac, rm, ecografía), reporte/nota de procedimiento/consulta, descripción quirúrgica.

        2) "documentos_administrativos"
        - "Recibos de pago"
        - "Certificados Médicos"
        - "Constancia de Remisión"
        - "Pre autorización"

        3) "documentos_contractuales"
        - "Contrato"
        - "Otro si"
        - "Tarifario"
        - "Cotización"

        4) "otros"
        - Para documentos válidos no cubiertos arriba.
        - Incluir como mínimo:
        - "Correos electrónicos" si hay cabeceras o contenido de emails.
        - "Acta de conciliación" cuando aparezca (aunque sea “de conciliación”, el esquema no tiene grupo propio; ubicarla aquí).

        REGLAS ANTI-ALUCINACIÓN (críticas)
        - NUNCA inventes datos de identificación del paciente (nombre, tipo/numero de documento). NO los derives de:
        (a) nombres de archivo/rutas,
        (b) direcciones de email o usuarios,
        (c) campos parciales no rotulados.
        - Solo llena "nombre_paciente", "tipo_documento" y "numero_documento" si están explícitamente rotulados en el CONTEXTO (p. ej., “Paciente:”, “Usuario:”, “Afiliado:”, “CC:”, “Tipo de documento:”, “Número de documento:”).
        - Para "numero_documento", conserva SOLO dígitos (remueve puntos, comas y espacios). Si falta cualquier parte, deja "".
        - Para "numero_factura", usa el literal detectado (alfanumérico permitido con guiones/puntos). No completes prefijos ni consecutivos faltantes.
        - Para "fecha_expedicion_factura", normaliza a YYYY-MM-DD solo si es inequívoco. Si hay ambigüedad, deja "".
        - Nunca extrapoles información clínica o administrativa no presente. En el "resumen", incluye SOLO lo que esté explícitamente en el CONTEXTO.

        REGLAS DE DEDUCCIÓN Y DESEMPATE (ruido RAG)
        - Trata cada “----- DOC n -----” como fuente independiente.
        - Ante duplicados/contradicciones, prioriza:
        (1) rótulos claros (“Factura No.”, “Fecha de expedición”, “NÚMERO DE AUTORIZACIÓN”),
        (2) presencia de razón social/sellos/NIT del emisor,
        (3) especificidad frente a términos genéricos.
        - Ignora boilerplate (avisos legales, pies de página) para extracción.

        EXTRACCIONES → llenar "datos_prestacion_servicios"
        Completar si existe; en caso contrario, dejar "" (o []).
        - "nombre_paciente": etiquetas “Paciente”, “Usuario”, “Afiliado”, “Nombre del paciente”.
        - "tipo_documento": valores tal como aparezcan (CC, TI, CE, PA, RC, etc.). No inventes.
        - "numero_documento": solo dígitos (quitar puntos/espacios). No “reconstruyas” números parciales.
        - "numero_factura": alfanumérico con guiones/prefijos ([A-Z0-9\-/.]+). Conservar literal.
        - "fecha_expedicion_factura": normalizar a YYYY-MM-DD cuando sea inequívoco (desde DD/MM/YYYY, YYYY-MM-DD o DD-MM-YYYY); si no, "".
        - "emisor_factura": IPS/Prestador emisor (razón social explícita o NIT rotulado como emisor/prestador).
        - "pagador": EPS/Entidad pagadora (p. ej., “Coosalud EPS”).
        - "autorizacion": lista de números de autorización detectados (sin duplicados). Patrones típicos:
        (?i)(autorizaci[oó]n|c[oó]digo de autorizaci[oó]n|radicado)\s*[:#\-]*\s*([A-Z0-9\-]+)

        EVIDENCIA DE “TRES CORREOS” (autorizaciones)
        - Si se evidencia el envío de tres correos relacionados con autorización (tres cabeceras/hilos distintos con asunto/tema de autorización), incluir "Correos electrónicos" en "otros".
        - No infieras si no hay evidencia clara.

        RESUMEN ESTRUCTURADO → "resumen"
        - Si existe base clínica en Historia clínica o Evidencia de entrega, genera SIEMPRE un resumen **estructurado** con el siguiente formato (solo incluye líneas con evidencia; si un ítem no existe, **omite la línea completa**):
        • Caso/Atención: [motivo consulta/diagnóstico/servicio prestado].
        • Fechas clave: [atención/ingreso/egreso, exámenes, autorización, factura] (solo las disponibles).
        • Autorización: [números detectados] y, si corresponde, mención “se evidencian tres correos de gestión de autorización”.
        • Soportes localizados: [lista de tipos presentes: Historia clínica, Evidencia de entrega, Autorizaciones, Factura, Correos, Acta de conciliación].
        • Hallazgos relevantes: [hechos clínicos/administrativos soportados en el CONTEXTO].
        • Conclusión/Acción sugerida: [síntesis breve basada en lo encontrado].
        - Si NO hay base clínica suficiente para un resumen (p. ej., solo administrativos), deja "resumen" = "".

        VALIDACIÓN DE SALIDA
        1) Devuelve SOLO el JSON del esquema con las claves y orden exactos.
        2) No agregues ni renombres claves. Sin comentarios ni Markdown.
        3) Cuando algo no exista en CONTEXTO, deja vacío ("") o [] según corresponda.

        PROCEDIMIENTO INTERNO (aplícalo, no lo expliques)
        1) Separar y leer por DOC; ignorar boilerplate.
        2) Detectar categorías (insensible a mayúsculas/tildes).
        3) Extraer campos con los patrones/normalizaciones indicados.
        4) Resolver contradicciones con reglas de desempate.
        5) Deduplicar listas y mantener solo valores sustentados.
        6) Rellenar el JSON exacto en el orden definido.
    """

    # Luego creamos el user content sin usar f-strings complejos
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
                max_tokens=2000,
            ),
            timeout=30.0
        )
        raw_text = completion.choices[0].message.content.strip()
        return json.loads(raw_text)  # Asume JSON válido; maneja errores abajo
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Timeout en llamada a LLM")
    except json.JSONDecodeError:
        logger.warning("JSON inválido de LLM")
        raise HTTPException(status_code=500, detail="Respuesta LLM malformada")
    except Exception:
        logger.exception("Error en LLM")
        raise HTTPException(status_code=500, detail="Error en LLM")

@router.post("/responder_pregunta/", response_model=RespuestaLLM)
async def responder_pregunta_mejorado(request: Request, input_data: PreguntaInput, db: AsyncSession = Depends(get_db)):
    start_time = asyncio.get_event_loop().time()  # Para métricas

    redis: Redis = request.app.state.redis
    embedding = await get_cached_embedding(redis, input_data.pregunta)
    if not embedding or not isinstance(embedding, list):
        logger.error("Embedding inválido")
        raise HTTPException(status_code=400, detail="No se pudo generar embedding")

    # Debug count (async)
    query_debug = text("SELECT COUNT(*) as total FROM files_chunks WHERE files_id = :id")
    try:
        count_result = db.execute(query_debug, {"id": input_data.id})
        count = count_result.scalar()
    except Exception:
        logger.exception("Error en query debug")
        raise HTTPException(status_code=500, detail="Error en DB")

    if count == 0:
        query_archivo = text("SELECT id FROM files WHERE id = :id")
        try:
            archivo_result = db.execute(query_archivo, {"id": input_data.id})
            if archivo_result.fetchone() is None:
                raise HTTPException(status_code=404, detail="Archivo no encontrado")
        except Exception:
            logger.exception("Error comprobando archivo")
            raise HTTPException(status_code=500, detail="Error en DB")

    # Fetch chunks
    results = await fetch_chunks(db, embedding, input_data.id)
    if not results:
        raise HTTPException(status_code=404, detail="No chunks encontrados")

    chunks_candidatos = [
        {"id": d.get("id"), "content": d.get("content", ""), "distancia": normalize_distance(d.get("distancia", 1.0))}
        for r in results if (d := dict_from_row(r))
    ]

    # Reranking
    pares = [[input_data.pregunta, c["content"]] for c in chunks_candidatos]
    try:
        scores = await predict_reranker(request, pares, batch_size=2)  # Ajusta batch si CPU/GPU
    except Exception:
        logger.exception("Error en reranking")
        raise HTTPException(status_code=500, detail="Error en reranking")

    for i, chunk in enumerate(chunks_candidatos):
        chunk["score_reranking"] = float(scores[i])
        chunk["score_combinado"] = 0.6 * chunk["score_reranking"] + 0.4 * (1.0 - chunk["distancia"])

    chunks_ordenados = sorted(chunks_candidatos, key=lambda x: x["score_combinado"], reverse=True)
    chunks_finales = [c for c in chunks_ordenados if c["score_combinado"] > input_data.score_threshold] or chunks_ordenados[:2]

    if not chunks_finales:
        raise HTTPException(status_code=400, detail="Chunks no relevantes")

    chunks_contexto = chunks_finales[:3]
    contexto = "\n\n".join([f"[Fragmento {i+1}]: {chunk['content']}" for i, chunk in enumerate(chunks_contexto)])
    distancia_promedio = sum(c["distancia"] for c in chunks_contexto) / len(chunks_contexto)

    # Llamada LLM
    parsed = await call_llm(request, contexto)
    try:
        parsed = validate_parsed_response(parsed, contexto)
    except Exception:
        logger.exception("Error validando respuesta LLM (individual)")

    logger.info(f"Procesado en {asyncio.get_event_loop().time() - start_time:.2f}s")
    return RespuestaLLM(
        respuesta=parsed,
        distancia=distancia_promedio,
        chunks_utilizados=len(chunks_contexto),
    )