from sqlalchemy import inspect, text # type: ignore
from database import engine, Base

def run_migrations():
    # Habilitar pgvector (solo una vez)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

    inspector = inspect(engine)

    # Verificar y crear tabla archivos_s3
    tabla_archivos_existe = inspector.has_table("archivos_s3")
    tabla_chunks_existe = inspector.has_table("chunks_archivo")

    if not tabla_archivos_existe:
        Base.metadata.create_all(bind=engine)

    # Verificar y crear tabla chunks_archivo
    if not tabla_chunks_existe:
        Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    print("🚀 Iniciando migraciones de base de datos...")
    run_migrations()
    print("🎯 Migraciones completadas!")
