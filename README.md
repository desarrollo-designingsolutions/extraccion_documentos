Propuesta para integrar Streamlit como frontend separado consumiendo API FastAPI:

- Carpetas:
  - /app/              # Backend FastAPI existente
  - /frontend/         # Nuevo frontend en Streamlit

- Ejecución:
  - Backend: uvicorn app.main:app --reload
  - Frontend: streamlit run frontend/main.py

Esta estructura permite escalabilidad y desacople claro.
