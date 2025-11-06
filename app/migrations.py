from sqlalchemy import inspect, text
from database import engine, Base

def run_migrations():
    # Habilitar pgvector (solo una vez)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

    inspector = inspect(engine)

    tablas_existentes = inspector.get_table_names()
    tablas_requeridas = ['files', 'files_chunks', 'invoice_audits', 'temporary_files', 'temporary_files_chunks']

    # Crear tablas que no existen
    for tabla in tablas_requeridas:
        if tabla not in tablas_existentes:
            print(f"Creando tabla: {tabla}")
    
    # Crear todas las tablas definidas en los modelos
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    print("🚀 Iniciando migraciones de base de datos...")
    run_migrations()
    print("🎯 Migraciones completadas!")