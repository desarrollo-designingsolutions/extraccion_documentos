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
        print("Tabla 'archivos_s3' no existe. Creándola con Vector nativo...")
        Base.metadata.create_all(bind=engine)
        print("✅ Tabla 'archivos_s3' creada exitosamente.")


    # Verificar y crear tabla chunks_archivo
    if not tabla_chunks_existe:
        print("Tabla 'chunks_archivo' no existe. Creándola...")
        Base.metadata.create_all(bind=engine)
        print("✅ Tabla 'chunks_archivo' creada exitosamente.")

        # Crear índice para búsquedas vectoriales en chunks
        try:
            with engine.connect() as conn:
                conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding 
                    ON chunks_archivo 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """
                    )
                )
                conn.commit()
                print("✅ Índice vectorial creado para 'chunks_archivo'")
        except Exception as e:
            print(f"⚠️  Error creando índice vectorial: {e}")


if __name__ == "__main__":
    print("🚀 Iniciando migraciones de base de datos...")
    run_migrations()
    print("🎯 Migraciones completadas!")
