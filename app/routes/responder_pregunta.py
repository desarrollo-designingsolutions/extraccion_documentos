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
        FROM chunks_archivo 
        WHERE archivo_s3_id = :id
        """
    )
    count = db.execute(query_debug, {"id": input_data.id}).scalar()

    if count == 0:
        # Chequea si el archivo principal existe
        query_archivo = text("SELECT id FROM archivos_s3 WHERE id = :id")
        archivo_exists = db.execute(query_archivo, {"id": input_data.id}).fetchone()

    # Buscar los chunks más relevantes usando pgvector
    query = text(
        """
        SELECT ca.id, ca.contenido, ca.embedding <=> CAST(:query_emb AS vector) AS distancia
        FROM chunks_archivo ca
        WHERE ca.archivo_s3_id = :id
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
        [input_data.pregunta, chunk["contenido"]] for chunk in chunks_candidatos
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
            f"[Fragmento {i+1}]: {chunk['contenido']}"
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
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """
                        Eres un asistente especializado en análisis de documentos.
                        El asistente especializado en análisis de documentos, debe tener la capacidad de procesar grandes volúmenes de información, identificar patrones y contenido relevante dentro de los documentos de servicios médicos, y ofrecer respuestas rápidas y precisas que faciliten la gestión de cuentas médicas en Colombia, mejorando la eficiencia y la precisión en la administración de los servicios de salud. 

                       Tu tarea es responder preguntas del usuario basándote ÚNICAMENTE en el contexto proporcionado.

                        Tipos de preguntas que puedes recibir:
                        1. Clasificación del documento:
                        - "Prestación de servicios de salud": documentos que detallan información de un paciente y los procedimientos médicos realizados.
                        - "Documento administrativo": documentos dirigidos a Coosalud (ej. contratos, actas de conciliación, cartas, comunicaciones administrativas).
                        - Si no hay información suficiente, responde: "No se puede determinar".
                        - En este caso, responde SOLO con la categoría.

                        - "Prestación de servicios de salud":
                        Factura de servicios médicos: Detalla los servicios prestados y los costos asociados a un paciente, que deben ser aprobados y validados por la EPS correspondiente.
                        Autorizaciones: Solicitudes previas de aprobación para ciertos procedimientos o tratamientos médicos. Se valida como autorización el trámite del envió de los tres correos por parte de la IPS
                        Historia clínica: Documento que recoge el registro detallado de la atención médica prestada al paciente.
                        Evidencia de entrega: Resultados de análisis, pruebas médicas solicitadas como parte del tratamiento, así como, reportes de consultas, procedentitos y descripciones quirúrgicas.
                        - "Documento de conciliación":
                        Acta de conciliación: Es un documento formal que recoge los acuerdos alcanzados entre IPS y EPS, normalmente relacionadas con la prestación de servicios de salud. Esta acta se elabora tras un proceso de negociación o conciliación, donde se buscan soluciones mutuas a los desacuerdos.

                        - "Documento administrativo":
                        Recibos de pago: Comprobantes que validan los pagos realizados por los servicios prestados. 
                        Certificados Médicos: Soporte para justificar la condición de salud del paciente ante EPS, empleadores o entidades gubernamentales.
                        Constancia de Remisión: Documento que evidencia la remisión de un paciente de una IPS a otra o a un especialista, indicando el motivo y el tratamiento recomendado.
                        Pre autorización:  Documento que autoriza el uso o la dispensación de ciertos servicios, medicamentos o tratamientos no cubiertos automáticamente por el plan de salud.

                        - "Documento contractual":
                        Contrato: Es el acuerdo formal entre las EPS (Entidades Promotoras de Salud) y las IPS (Instituciones Prestadoras de Salud) o los prestadores individuales de servicios médicos, donde se establecen las condiciones, términos, tarifas y responsabilidades sobre la prestación de servicios de salud.
                        Otro si: Es un documento adicional o anexo a un contrato principal (como el contrato de prestación de servicios de salud) que modifica, amplía o aclara ciertos términos, condiciones o acuerdos establecidos inicialmente.
                        Tarifario: Es un documento que contiene la lista oficial de tarifas que las EPS y las IPS deben seguir para facturar los servicios médicos prestados. Establece los precios estándar por cada tipo de servicio, tratamiento o procedimiento.
                        Cotización: Es una estimación de los costos de un servicio o conjunto de servicios médicos proporcionada por un prestador de salud a un paciente o a una EPS. Generalmente, la cotización especifica los precios por procedimiento, tratamiento o consulta.


                        El asistente debe identificar si existe o no dentro de los soportes las categorías de los siguientes documentos: 
                        •	Factura de servicios médicos
                        •	Autorizaciones
                        •	Historia clínica
                        •	Evidencia de entrega
                        •	Acta de conciliación
                        •	Documento administrativo
                        •	Documento contractual

                        Del listado anterior el asistente debe responder en formato JSON si existe o no la categoría documental.
                        Adicionalmente en la categoría autorizaciones se debe identificar:
                        •	Si existe un numero de autorización -  Este se debe extraer este número.
                        •	Si existe evidencia del envío de tres coreos electrónicos solicitando una autorización - se debe reportar que existen los tres correos.

                      2. Extracción de datos:
                        - Si el usuario pide datos específicos (ej. nombre del paciente, fecha, número de documento), respóndelos únicamente si aparecen en el contexto.
                        - Si no están, indica claramente: "No se encuentra en el contexto".

                        En las siguientes categorías:
                        •	Factura: debe extraer el número de factura, usuario (paciente), emisor (quien genera la factura - IPS) y pagador (a quien va dirigida la factura - EPS).
                        •	Autorización: debe extraer el numero de autorización cuando exista.

                        3. Resúmenes:
                        - Si el usuario pide un resumen, genera un texto breve y claro con la información más relevante del documento.
                        
                        El resumen debe generarse a partir de la información contenida en la Historia Clínica o evidencia del servicio.

                        Reglas generales:
                        - Nunca inventes información que no esté en el contexto.
                        - Si no puedes responder algo, dilo explícitamente.
                        - Ajusta tu respuesta al tipo de pregunta: clasificación, datos específicos o resumen.
                    """,
                },
                {
                    "role": "user",
                    "content": f"""
                        CONTEXTO DEL DOCUMENTO:
                        {contexto}

                        PREGUNTA: {input_data.pregunta}
                    """,
                },
            ],
        )

        respuesta_llm = completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en LLM: {str(e)}")

    # Calcular distancia promedio de los chunks utilizados
    distancia_promedio = sum(chunk["distancia"] for chunk in chunks_contexto) / len(
        chunks_contexto
    )

    return RespuestaLLM(
        respuesta=respuesta_llm,
        distancia=distancia_promedio,
        chunks_utilizados=len(chunks_contexto),
    )
