from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
import os

DATABASE_URL = os.getenv("DATABASE_URL").replace("postgresql://", "postgresql+asyncpg://")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL no configurada")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):  # Subclasea DeclarativeBase correctamente
    pass

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session