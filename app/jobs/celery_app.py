import os
import json
import logging
import redis
from celery import Celery

logger = logging.getLogger("jobs")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)


# Celery app centralizada para todo el package `jobs`
celery = Celery("jobs", broker=BROKER_URL, backend=BACKEND_URL)

# Cliente redis para persistir estados simples
try:
    r = redis.Redis.from_url(REDIS_URL)
except Exception:
    r = None
    logger.exception("No se pudo inicializar cliente Redis en celery_app")


def set_job_state(job_id: str, payload: dict):
    """Guardar estado simple del job en Redis (si está disponible)."""
    if not r:
        logger.debug("Redis no disponible, set_job_state no guardará estado")
        return
    try:
        r.set(f"job:{job_id}", json.dumps(payload))
    except Exception:
        logger.exception(f"Error al setear estado en Redis para job {job_id}")


# Nota: no importamos módulos de tareas aquí para evitar import cycles en tiempo de carga.
# Los módulos de tarea deben importar `celery` desde `jobs` o `jobs.celery_app`.
