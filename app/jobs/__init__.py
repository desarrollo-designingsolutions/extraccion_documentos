"""Paquete de tareas `jobs`.

Este módulo expone el objeto `celery` y la función `set_job_state` desde
`jobs.celery_app` para que el worker pueda importarlos como `jobs.celery`.
"""

try:
	from .celery_app import celery, set_job_state  # re-export
except Exception:
	# No elevar excepción en import time; los errores se registrarán cuando se use
	celery = None
	def set_job_state(job_id: str, payload: dict):
		return

# Importar módulos con tareas para registrarlas cuando se importe `jobs`.
try:
	# módulos que definen tareas decoradas con @celery.task
	from . import job_scanear  # noqa: F401
	from . import job_pregunta_multiple  # noqa: F401
except Exception:
	# No fallar el import del paquete; los errores aparecerán en logs cuando el worker arranque
	pass