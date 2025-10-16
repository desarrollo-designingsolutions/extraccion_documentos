from sqlalchemy import inspect, text
from database import engine, Base

def run_migrations():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

    inspector = inspect(engine)
    tablas_existentes = inspector.get_table_names()

    if not tablas_existentes:
        Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    print("🚀 Iniciando migraciones de base de datos...")
    run_migrations()
    print("🎯 Migraciones completadas!")
