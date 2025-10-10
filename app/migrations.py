from sqlalchemy import inspect, text
from database import engine, Base
from models import ArchivoS3, ChunkArchivo


def run_migrations():
    """
    Chequea si las tablas 'archivos_s3' y 'chunks_archivo' existen.
    Si no existen, las crea con Vector(1536).
    Siempre habilita extensión 'vector' si no está.
    """
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
    else:
        print("✅ Tabla 'archivos_s3' ya existe.")

        # Verificar si tiene la columna embedding con la dimensión correcta
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'archivos_s3' 
                    AND column_name = 'embedding'
                """
                    )
                )
                embedding_column = result.fetchone()

                if embedding_column:
                    print("✅ Columna 'embedding' existe en 'archivos_s3'")
                else:
                    print("⚠️  Columna 'embedding' no existe, se necesitará migración")
        except Exception as e:
            print(f"⚠️  Error verificando columna embedding: {e}")

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
    else:
        print("✅ Tabla 'chunks_archivo' ya existe.")

        # Verificar foreign key y relaciones
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                    SELECT COUNT(*) as chunk_count 
                    FROM chunks_archivo
                """
                    )
                )
                chunk_count = result.fetchone()[0]
                print(f"📊 Tabla 'chunks_archivo' tiene {chunk_count} registros")
        except Exception as e:
            print(f"⚠️  Error contando chunks: {e}")

    # Verificar la relación entre tablas
    if tabla_archivos_existe and tabla_chunks_existe:
        try:
            with engine.connect() as conn:
                # Verificar si hay archivos sin chunks
                result = conn.execute(
                    text(
                        """
                    SELECT COUNT(*) as archivos_sin_chunks
                    FROM archivos_s3 a
                    LEFT JOIN chunks_archivo c ON a.id = c.archivo_s3_id
                    WHERE c.id IS NULL
                """
                    )
                )
                archivos_sin_chunks = result.fetchone()[0]

                if archivos_sin_chunks > 0:
                    print(
                        f"🔄 {archivos_sin_chunks} archivos necesitan ser reprocesados con chunks"
                    )
                else:
                    print("✅ Todos los archivos tienen chunks asociados")

        except Exception as e:
            print(f"⚠️  Error verificando relaciones: {e}")


def check_database_health():
    """
    Verifica la salud de la base de datos y las configuraciones
    """
    print("\n🔍 Verificando salud de la base de datos...")

    with engine.connect() as conn:
        try:
            # Verificar extensión vector
            result = conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            )
            if result.fetchone():
                print("✅ Extensión 'vector' está habilitada")
            else:
                print("❌ Extensión 'vector' no está habilitada")

            # Verificar dimensiones de embeddings
            result = conn.execute(
                text(
                    """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name IN ('archivos_s3', 'chunks_archivo')
                AND column_name = 'embedding'
            """
                )
            )

            embedding_columns = result.fetchall()
            for col in embedding_columns:
                print(f"✅ Columna 'embedding' existe en {col[0]}")

            # Verificar cantidad de tablas
            result = conn.execute(
                text(
                    """
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                AND table_name IN ('archivos_s3', 'chunks_archivo')
            """
                )
            )
            table_count = result.fetchone()[0]
            print(f"📊 {table_count}/2 tablas necesarias existen")

        except Exception as e:
            print(f"❌ Error en verificación de salud: {e}")


if __name__ == "__main__":
    print("🚀 Iniciando migraciones de base de datos...")
    run_migrations()
    check_database_health()
    print("🎯 Migraciones completadas!")
