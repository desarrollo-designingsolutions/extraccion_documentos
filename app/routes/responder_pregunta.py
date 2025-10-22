from fastapi import APIRouter, Depends, HTTPException  # type: ignore
from sqlalchemy import text  # type: ignore
from sqlalchemy.orm import Session  # type: ignore
from sentence_transformers import CrossEncoder  # type: ignore
from openai import OpenAI  # type: ignore
from database import get_db
from .schemas import RespuestaLLM, PreguntaInput
from utils.helpers import (
    generar_embedding_openai,
)
import os
import json

router = APIRouter()

# Cargar modelos al inicio
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Cliente de OpenAI
openai_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_key)


@router.post("/responder_pregunta/", response_model=RespuestaLLM)
def responder_pregunta_mejorado(
    input_data: PreguntaInput, db: Session = Depends(get_db)
):
    # Generar embedding de la pregunta
    embedding_pregunta = generar_embedding_openai(input_data.pregunta)
    if not embedding_pregunta:
        raise HTTPException(
            status_code=400, detail="No se pudo generar embedding para la pregunta"
        )

    # Query de debug SIN vector para verificar datos raw
    query_debug = text(
        """
        SELECT COUNT(*) as total 
        FROM files_chunks 
        WHERE files_id = :id
        """
    )
    count = db.execute(query_debug, {"id": input_data.id}).scalar()

    if count == 0:
        # Chequea si el archivo principal existe
        query_archivo = text("SELECT id FROM files WHERE id = :id")
        archivo_exists = db.execute(query_archivo, {"id": input_data.id}).fetchone()

    # Buscar los chunks más relevantes usando pgvector
    query = text(
        """
        SELECT ca.id, ca.content, ca.embedding <=> CAST(:query_emb AS vector) AS distancia
        FROM files_chunks ca
        WHERE ca.files_id = :id
        ORDER BY distancia ASC
        LIMIT 20
        """
    )

    results = db.execute(
        query, {"query_emb": embedding_pregunta, "id": input_data.id}
    ).fetchall()

    if not results:
        raise HTTPException(
            status_code=404, detail="No se encontraron chunks para este PDF"
        )

    # Preparar datos para reranking
    chunks_candidatos = [result._asdict() for result in results]

    # Aplicar reranking con cross-encoder
    pares_reranking = [
        [input_data.pregunta, chunk["content"]] for chunk in chunks_candidatos
    ]
    scores_reranking = reranker.predict(pares_reranking)

    # Combinar scores de reranking con distancias vectoriales
    for i, chunk in enumerate(chunks_candidatos):
        chunk["score_reranking"] = float(scores_reranking[i])
        # Score combinado (ajustamos ponderación a 60/40)
        chunk["score_combinado"] = 0.6 * chunk["score_reranking"] + 0.4 * (
            1 - chunk["distancia"]
        )
        print(
            f"Chunk {i+1}: rerank={chunk['score_reranking']:.4f}, dist={chunk['distancia']:.4f}, combinado={chunk['score_combinado']:.4f}"
        )

    # Ordenar por score combinado y filtrar por umbral
    chunks_ordenados = sorted(
        chunks_candidatos, key=lambda x: x["score_combinado"], reverse=True
    )
    chunks_finales = [
        chunk
        for chunk in chunks_ordenados
        if chunk["score_combinado"] > input_data.score_threshold
    ]

    print(
        f"Chunks después de filtrar (umbral {input_data.score_threshold}): {len(chunks_finales)}"
    )

    if not chunks_finales:
        # Si no hay chunks que superen el umbral, usar los 2 mejores sin filtrar
        print("No hay chunks que superen el umbral, usando los 2 mejores disponibles")
        chunks_finales = chunks_ordenados[:2]

        if not chunks_finales:
            raise HTTPException(
                status_code=400,
                detail="No se encontraron chunks suficientemente relevantes para la pregunta",
            )

    # Limitar a los 3 chunks más relevantes para el contexto
    chunks_contexto = chunks_finales[:3]

    # Construir contexto para el LLM
    contexto = "\n\n".join(
        [
            f"[Fragmento {i+1}]: {chunk['content']}"
            for i, chunk in enumerate(chunks_contexto)
        ]
    )

    print(
        f"Usando {len(chunks_contexto)} chunks para contexto (total: {len(contexto)} caracteres)"
    )
    print(
        f"Distancia promedio: {sum(chunk['distancia'] for chunk in chunks_contexto) / len(chunks_contexto):.4f}"
    )
    print(
        f"Score combinado promedio: {sum(chunk['score_combinado'] for chunk in chunks_contexto) / len(chunks_contexto):.4f}"
    )

    try:
        # Primero definimos el system content como string constante
        system_content = """
        PRINCIPIOS
        1) Usa EXCLUSIVAMENTE el texto en CONTEXTO DEL DOCUMENTO (procede de búsqueda semántica/embeddings).
        2) El CONTEXTO puede tener fragmentos parciales, duplicados o contradicciones y viene separado con delimitadores "----- DOC n -----".
        3) NO inventes. Si un dato no aparece con claridad, déjalo vacío ("") o no incluyas la categoría.
        4) SALIDA = SOLO un objeto JSON VÁLIDO con el esquema, claves y orden exactos que se definen abajo. Sin texto extra, sin Markdown, sin comentarios.

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
        Clasificar los tipos documentales presentes y extraer datos clave. Si el usuario pide un resumen, escribirlo brevemente. Si no hay información suficiente, deja campos vacíos.

        TIPOS Y LITERALES PERMITIDOS
        1) "prestacion_servicios_de_salud"  (agrega SOLO los literales exactos detectados)
        - "Factura de servicios médicos"
            Indicadores: factura, cta de cobro, nro/número de factura, prefijo+consecutivo.
        - "Autorizaciones"
            Indicadores: autorización, nro/código/radicado de autorización.
        - "Historia clínica"
            Indicadores: historia clínica, epicrisis, evolución, nota de ingreso/egreso, enfermería.
        - "Evidencia de entrega"
            Indicadores: exámenes/resultados (laboratorio, imagenología, rx, tac, rm, ecografía), reportes de consulta/procedimiento, descripción quirúrgica.

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
            - "Acta de conciliación" cuando aparezca (el esquema no tiene grupo propio de conciliación; por eso va aquí).

        REGLAS DE DEDUCCIÓN Y DESEMPATE (por ruido de RAG)
        - Trata cada "----- DOC n -----" como fuente independiente.
        - Ante duplicados o contradicciones, prioriza:
        (1) fragmentos con rótulos claros ("Factura No.", "Fecha de expedición", "NÚMERO DE AUTORIZACIÓN"),
        (2) presencia de razón social/sellos/NIT del emisor,
        (3) mayor especificidad frente a términos genéricos.
        - Ignora boilerplate (avisos legales, pies de página) para la extracción.

        EXTRACCIONES → llenar "datos_prestacion_servicios"
        De existir, completar; si no, dejar "" (o [] para listas).
        - "nombre_paciente": etiquetas "Paciente", "Usuario", "Afiliado", "Nombre del paciente".
        - "tipo_documento": valores tal como aparezcan (CC, TI, CE, PA, RC, etc.).
        - "numero_documento": solo dígitos (quitar puntos/espacios).
        - "numero_factura": alfanumérico con guiones/prefijos ([A-Z0-9\-/.]+). Conservar el literal detectado.
        - "fecha_expedicion_factura": normalizar a YYYY-MM-DD cuando sea inequívoco a partir de DD/MM/YYYY, YYYY-MM-DD o DD-MM-YYYY; si hay ambigüedad, dejar "".
        - "emisor_factura": IPS/Prestador emisor (razón social más explícita o NIT rotulado).
        - "pagador": EPS/Entidad pagadora (p. ej., campo "Pagador", "Asegurador", "EPS").
        - "autorizacion": lista de números de autorización detectados (sin duplicados). Patrones típicos:
        (?i)(autorizaci[oó]n|c[oó]digo de autorizaci[oó]n|radicado)\s*[:#\-]*\s*([A-Z0-9\-]+)

        EVIDENCIA DE "TRES CORREOS" (autorizaciones)
        - Si se evidencia el envío de tres correos relacionados a la solicitud/gestión de autorización (tres cabeceras o hilos distintos con asunto/tema de autorización), incluir "Correos electrónicos" en "otros".
        - No infieras si no hay evidencia clara de tres correos distintos.

        RESUMEN CLÍNICO → "resumen"
        - Solo si el usuario lo solicita o el flujo lo requiere.
        - Breve (máx. 5-6 líneas) y derivado de Historia clínica o Evidencia de entrega del CONTEXTO.
        - Si no hay base clínica suficiente, dejar "".

        VALIDACIÓN DE SALIDA
        1) Devuelve SOLO el JSON del esquema, con las claves y orden exactos.
        2) No agregues ni renombres claves. No repitas categorías. Sin comentarios ni Markdown.
        3) Cuando algo no exista en CONTEXTO, deja vacío ("") o [] según corresponda.

        PROCEDIMIENTO INTERNO (no lo expliques; solo aplícalo)
        1) Separar y leer por DOC, ignorar boilerplate.
        2) Detectar categorías con las palabras clave (insensible a mayúsculas/tildes).
        3) Extraer campos con los patrones y normalizaciones indicados.
        4) Resolver contradicciones con las reglas de desempate.
        5) Deduplicar listas y ordenar alfabéticamente cuando aplique (no obligatorio).
        6) Rellenar JSON exacto en el orden definido.

        SINÓNIMOS FRECUENTES (ayuda a la detección)
        - Factura: "cuenta de cobro", "cta de cobro", "invoice".
        - Autorización: "código autorización", "radicado", "autoriz.".
        - Historia clínica: "epicrisis", "evolución", "nota médica". 
        - Evidencia de entrega: "resultado", "reporte", "descripción quirúrgica", "imagenología", "laboratorio".
        - Recibo de pago: "comprobante", "soporte de pago", "transferencia".
        - Remisión: "contrarreferencia", "remisión".
        """

        # Luego creamos el user content sin usar f-strings complejos
        user_content = f"""
        {contexto}   (delimitado por "----- DOC n -----" entre fragmentos)

        PREGUNTA:
        {input_data.pregunta}

        PARÁMETROS RECOMENDADOS (fuera del prompt, en tu cliente)
        - temperature: 0
        - top_p: 1
        - max_tokens: suficiente para cubrir el JSON
        - Si tu proveedor lo soporta: usa validación JSON/JSON schema o function/tool-calling para fijar el formato.

        EJEMPLOS (few-shots compactos dentro del system)
        --------------------------------------------------------------------------------
        [Ejemplo 1 - Factura + Autorización + Correos]
        CONTEXTO DEL DOCUMENTO (fragmentos):
        ----- DOC 1 -----
        Factura No.: ABC-123
        Fecha expedición: 14/08/2025
        Prestador (Emisor): Clínica San José S.A.S. NIT 900111222
        Pagador: Coosalud EPS
        Paciente: Juan Pérez
        CC 1.234.567.890

        ----- DOC 2 -----
        NÚMERO DE AUTORIZACIÓN: A-98765

        ----- DOC 3 -----
        De: autorizaciones@ips.com
        Asunto: Solicitud de Autorización paciente Juan Pérez
        Fecha: 2025-08-10

        ----- DOC 4 -----
        De: autorizaciones@ips.com
        Asunto: Seguimiento Autorización A-98765
        Fecha: 2025-08-11

        ----- DOC 5 -----
        De: autorizaciones@ips.com
        Asunto: Escalamiento Autorización A-98765
        Fecha: 2025-08-12

        PREGUNTA: Clasificar y extraer datos.

        SALIDA ESPERADA (solo JSON):
        {{
        "prestacion_servicios_de_salud": [
            "Factura de servicios médicos",
            "Autorizaciones"
        ],
        "documentos_administrativos": [],
        "documentos_contractuales": [],
        "otros": [
            "Correos electrónicos"
        ],
        "datos_prestacion_servicios": {{
            "nombre_paciente": "Juan Pérez",
            "tipo_documento": "CC",
            "numero_documento": "1234567890",
            "numero_factura": "ABC-123",
            "fecha_expedicion_factura": "2025-08-14",
            "emisor_factura": "Clínica San José S.A.S. (NIT 900111222)",
            "pagador": "Coosalud EPS",
            "autorizacion": ["A-98765"]
        }},
        "resumen": ""
        }}

        [Ejemplo 2 - Administrativos + Acta de conciliación, sin datos clínicos]
        CONTEXTO DEL DOCUMENTO:
        ----- DOC 1 -----
        Recibo de pago transferencia #77889 por servicios prestados agosto 2025.
        ----- DOC 2 -----
        Certificado Médico laboral: apto.
        ----- DOC 3 -----
        Acta de conciliación entre IPS X y EPS Y, acuerdos de pago.

        PREGUNTA: Solo clasificación.

        SALIDA ESPERADA (solo JSON):
        {{
            "prestacion_servicios_de_salud": [],
            "documentos_administrativos": [
                "Recibos de pago",
                "Certificados Médicos"
            ],
            "documentos_contractuales": [],
            "otros": [
                "Acta de conciliación"
            ],
            "datos_prestacion_servicios": {{
                "nombre_paciente": "",
                "tipo_documento": "",
                "numero_documento": "",
                "numero_factura": "",
                "fecha_expedicion_factura": "",
                "emisor_factura": "",
                "pagador": "",
                "autorizacion": []
            }},
            "resumen": ""
        }}
        """
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user", 
                    "content": user_content
                }
        ])

        respuesta_llm = completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en LLM: {str(e)}")

    # Calcular distancia promedio de los chunks utilizados
    distancia_promedio = sum(chunk["distancia"] for chunk in chunks_contexto) / len(
        chunks_contexto
    )

    return RespuestaLLM(
        respuesta=json.loads(respuesta_llm),
        distancia=distancia_promedio,
        chunks_utilizados=len(chunks_contexto),
    )
