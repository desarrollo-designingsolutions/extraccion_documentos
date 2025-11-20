from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Preferir DATABASE_URL completo si está presente (útil en Docker .env)
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Construir DATABASE_URL desde variables de entorno individuales
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "mi_proyecto_db")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "usuario_db")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

    DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Dependencia para obtener sesión de DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
