import os
import json
import logging
import asyncio
from typing import List, Dict, Any
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from app.jobs import celery, set_job_state
from sentence_transformers import CrossEncoder
from openai import AsyncOpenAI
from app.utils.helpers import generar_embedding_openai, validate_parsed_response
import re
import string

logger = logging.getLogger("job_pregunta_multiple")

# DB setup para el worker (usar misma URL que la app)
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL no está definida en el entorno")
engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine)

# Simple Spanish stopwords (small set to avoid new deps)
STOPWORDS = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
    "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero",
    "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando",
}

# Boilerplate keywords that indicate non-patient, contractual or metadata content
BOILERPLATE_KEYWORDS = {
    "el contratante", "tarifario", "acta de conciliación", "suministro de medicamentos",
    "stock de medicamentos", "registro individual", "suministro de medicamentos", "contrato", 
    "cláusula", "clausula", "condiciones", "términos", "terminos", "ordenado en la fórmula", 
    "entrega de la información"
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
    return len(inter) / max(1, len(set(q_tokens)))

def normalize_distance(d: float) -> float:
    try:
        v = float(d)
    except Exception:
        return 1.0
    return max(0.0, min(1.0, v))

def _predict_sync(reranker, pairs: List[List[str]]) -> List[float]:
    return reranker.predict(pairs)

async def predict_reranker(reranker, pairs: List[List[str]], batch_size: int = 16) -> List[float]:
    if reranker is None:
        logger.error("Reranker no disponible")
        raise Exception("Reranker no disponible")
    
    loop = asyncio.get_event_loop()
    scores: List[float] = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i: i + batch_size]
        batch_scores = await loop.run_in_executor(None, _predict_sync, reranker, batch)
        scores.extend(batch_scores)
    return scores

def safe_json_parse(json_string: str) -> Dict:
    """
    Intenta parsear JSON de manera segura, con múltiples estrategias de fallback
    """
    if not json_string or not json_string.strip():
        return {}
    
    # Intentar parseo directo primero
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}. Attempting cleanup...")
    
    # Estrategia 1: Buscar el primer { y el último }
    start_idx = json_string.find('{')
    end_idx = json_string.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            cleaned = json_string[start_idx:end_idx+1]
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    
    # Estrategia 2: Remover caracteres problemáticos comunes
    try:
        # Remover saltos de línea y tabs dentro de strings
        cleaned = re.sub(r'(?<!\\)\\n', ' ', json_string)
        cleaned = re.sub(r'(?<!\\)\\t', ' ', cleaned)
        # Escapar comillas simples no escapadas
        cleaned = re.sub(r"(?<!\\)'", '"', cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Estrategia 3: Si todo falla, retornar estructura vacía
    logger.error(f"Could not parse JSON after multiple attempts. Raw: {json_string[:500]}...")
    return {
        "error": "Failed to parse LLM response",
        "raw_response_preview": json_string[:200] + "..." if len(json_string) > 200 else json_string
    }

async def call_llm(openai_client: AsyncOpenAI, context: str) -> Dict:
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
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=2000,
        )
        raw_text = completion.choices[0].message.content.strip()
        
        # Limpiar respuesta - remover posibles code blocks
        if raw_text.startswith('```json'):
            raw_text = raw_text[7:]
        if raw_text.startswith('```'):
            raw_text = raw_text[3:]
        if raw_text.endswith('```'):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()
        
        return safe_json_parse(raw_text)
        
    except Exception as e:
        logger.exception(f"Error en LLM: {e}")
        return {
            "error": f"LLM call failed: {str(e)}",
            "status": "llm_error"
        }

async def procesar_archivo_async(
    db, 
    openai_client: AsyncOpenAI, 
    reranker,
    file_id: int, 
    file_name: str,
    pregunta: str,
    score_threshold: float
):
    """Procesa un archivo individual de manera asíncrona"""
    
    try:
        # Generar embedding para la pregunta
        emb = await asyncio.to_thread(generar_embedding_openai, pregunta)
        if not emb:
            logger.error(f"No se pudo generar embedding para pregunta en archivo {file_name}")
            return None

        # Buscar chunks similares
        emb_str = f"[{','.join(map(str, emb))}]"
        q_chunks = text(
            """
            SELECT ca.id, ca.content, ca.embedding <=> CAST(:query_emb AS vector) AS distancia
            FROM files_chunks ca
            WHERE ca.files_id = :file_id
            ORDER BY distancia ASC
            LIMIT 50
            """
        )
        
        try:
            result = db.execute(q_chunks, {"query_emb": emb_str, "file_id": file_id})
            rows = result.fetchall()
            if not rows:
                logger.info(f"No chunks para archivo {file_name}")
                return None
            
            # Convertir resultados a dict
            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "content": row[1],
                    "distancia": row[2]
                })
        except Exception as e:
            logger.exception(f"Error fetch chunks for file {file_name}: {e}")
            return None

        # Preparar pares para reranking
        pares = [[pregunta, r.get("content", "")[:8000]] for r in results]  # Limitar contenido

        # Aplicar reranking
        try:
            scores = await predict_reranker(reranker, pares, batch_size=2)
        except Exception as e:
            logger.exception(f"Error en reranking para {file_name}: {e}")
            return None

        # Calcular scores combinados
        candidates = []
        for i, r in enumerate(results):
            distancia = normalize_distance(r.get("distancia", 1.0))
            content = r.get("content", "")
            score_rer = float(scores[i])
            lex = lexical_overlap_score(pregunta, content)
            
            combined = 0.75 * score_rer + 0.15 * (1.0 - distancia) + 0.10 * lex
            
            # Aplicar penalización por contenido boilerplate
            lower = content.lower()
            penalty = 0.0
            for kw in BOILERPLATE_KEYWORDS:
                if kw in lower:
                    penalty += 0.25
            combined = max(0.0, combined - penalty)
            
            candidates.append({
                "id": r.get("id"), 
                "content": content, 
                "distancia": distancia, 
                "score_combinado": combined
            })

        # Seleccionar mejores chunks
        candidates_sorted = sorted(candidates, key=lambda x: x["score_combinado"], reverse=True)
        top_n = max(3, min(5, len(candidates_sorted)))
        chunks_finales = [c for c in candidates_sorted if c["score_combinado"] > score_threshold] or candidates_sorted[:top_n]
        
        if not chunks_finales:
            logger.info(f"No chunks relevantes para {file_name}")
            return None

        # Construir contexto para LLM (limitar tamaño)
        contexto = "\n\n".join([f"[Fragmento {i+1}]: {c['content'][:2000]}" for i, c in enumerate(chunks_finales[:3])])

        # Llamar al LLM
        try:
            parsed = await call_llm(openai_client, contexto)
        except Exception as e:
            logger.exception(f"Error LLM para archivo {file_name}: {e}")
            return None

        # Validar respuesta
        try:
            parsed_valid = validate_parsed_response(parsed, contexto)
        except Exception as e:
            logger.exception(f"Error validando respuesta LLM para {file_name}: {e}")
            parsed_valid = parsed  # Usar respuesta aunque falle validación

        # Guardar resultado en base de datos
        try:
            db.execute(
                text("UPDATE files SET json_category = :json WHERE id = :id"), 
                {"json": json.dumps(parsed_valid), "id": file_id}
            )
            db.commit()
            logger.info(f"Guardado json_category para file_id={file_id}")
            return parsed_valid
        except Exception as e:
            db.rollback()
            logger.exception(f"Error guardando json_category file_id={file_id}: {e}")
            return None
            
    except Exception as e:
        logger.exception(f"Error general procesando archivo {file_name}: {e}")
        return None

@celery.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3})
def procesar_pregunta_multiple(self, payload: Dict[str, Any]):
    """
    Tarea Celery para procesar pregunta múltiple - Versión síncrona que llama a la versión async
    """
    job_id = self.request.id
    logger.info(f"[JOB {job_id}] Iniciando tarea procesar_pregunta_multiple")
    set_job_state(job_id, {"status": "queued", "progress": 0})

    try:
        # Ejecutar la versión asíncrona en un event loop
        result = asyncio.run(_procesar_pregunta_multiple_async(job_id, payload))
        set_job_state(job_id, {"status": "completed", "progress": 100})
        logger.info(f"[JOB {job_id}] Completado correctamente")
        return result
    except Exception as e:
        logger.exception(f"[JOB {job_id}] Error crítico: {e}")
        set_job_state(job_id, {"status": "failed", "progress": 0, "error": str(e)})
        raise

async def _procesar_pregunta_multiple_async(job_id: str, payload: Dict[str, Any]):
    """
    Versión asíncrona del procesamiento
    """
    pregunta = payload.get("pregunta") or payload.get("question")
    nits = payload.get("nits", [])
    requested_concurrency = int(payload.get("concurrency", 5))
    concurrency = max(1, min(requested_concurrency, 4))
    score_threshold = float(payload.get("score_threshold", 0.5))

    logger.info(f"[JOB {job_id}] Parámetros: nits={nits}, concurrency={concurrency}, score_threshold={score_threshold}")

    # Inicializar modelos (una vez por job)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device='cpu')
    openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Crear sesión DB
    db = SessionLocal()

    try:
        # Contar archivos totales para progreso
        total_files = 0
        for nit in nits:
            count_query = text("SELECT COUNT(*) FROM files WHERE nit = :nit AND json_category IS NULL")
            result = db.execute(count_query, {"nit": nit})
            count = result.scalar() or 0
            total_files += count

        if total_files == 0:
            set_job_state(job_id, {"status": "completed", "progress": 100, "message": "No se encontraron archivos"})
            return {"status": "completed", "message": "No se encontraron archivos"}

        logger.info(f"[JOB {job_id}] Total archivos a procesar: {total_files}")
        set_job_state(job_id, {"status": "processing", "progress": 5, "total_files": total_files})

        processed_files = 0
        successful_files = 0
        tasks = []

        # Crear semáforo para controlar concurrencia
        sem = asyncio.Semaphore(concurrency)

        async def procesar_con_semaforo(file_id, file_name, nit):
            async with sem:
                nonlocal processed_files, successful_files
                
                # Actualizar progreso
                progress = int((processed_files / total_files) * 100) if total_files > 0 else 0
                set_job_state(job_id, {
                    "status": "in_progress",
                    "progress": progress,
                    "current_file": file_name,
                    "processed_files": processed_files,
                    "total_files": total_files,
                    "current_nit": nit
                })

                # Procesar archivo
                result = await procesar_archivo_async(
                    db, openai_client, reranker, file_id, file_name, pregunta, score_threshold
                )
                
                processed_files += 1
                if result is not None:
                    successful_files += 1
                
                # Actualizar progreso después de procesar
                progress = int((processed_files / total_files) * 100) if total_files > 0 else 0
                set_job_state(job_id, {
                    "status": "in_progress", 
                    "progress": progress,
                    "processed_files": processed_files,
                    "successful_files": successful_files,
                    "total_files": total_files
                })
                
                return result

        # Recolectar todos los archivos a procesar
        for nit in nits:
            files_query = text("SELECT id, name FROM files WHERE nit = :nit AND json_category IS NULL")
            result = db.execute(files_query, {"nit": nit})
            files = result.fetchall()
            
            for file_row in files:
                file_id, file_name = file_row
                task = procesar_con_semaforo(file_id, file_name, nit)
                tasks.append(task)

        # Ejecutar todas las tareas
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Contar resultados exitosos y errores
        errors = sum(1 for r in results if isinstance(r, Exception))
        
        logger.info(f"[JOB {job_id}] Procesamiento completado: {successful_files}/{len(tasks)} archivos procesados exitosamente, {errors} errores")
        
        return {
            "status": "completed",
            "total_files": total_files,
            "processed_files": processed_files,
            "successful_files": successful_files,
            "errors": errors
        }

    except Exception as e:
        logger.exception(f"[JOB {job_id}] Error en procesamiento: {e}")
        raise
    finally:
        # Cerrar sesión de base de datos
        try:
            db.close()
        except Exception:
            logger.warning(f"[JOB {job_id}] Error cerrando sesión DB")