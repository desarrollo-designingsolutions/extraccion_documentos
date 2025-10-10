from sqlalchemy import Column, Integer, String, BigInteger, DateTime, Text, ForeignKey # type: ignore
from sqlalchemy.orm import relationship # type: ignore
from sqlalchemy.sql import func # type: ignore
from pgvector.sqlalchemy import Vector # type: ignore
from database import Base

class ArchivoS3(Base):
    __tablename__ = "archivos_s3"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, index=True, unique=True, nullable=False)
    nit = Column(String, index=True, nullable=False)
    tamaño = Column(BigInteger, nullable=False)
    url_presignada = Column(String, nullable=False)
    texto_extraido = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relación con chunks
    chunks = relationship("ChunkArchivo", back_populates="archivo_s3", cascade="all, delete-orphan")

class ChunkArchivo(Base):
    __tablename__ = "chunks_archivo"
    
    id = Column(Integer, primary_key=True, index=True)
    archivo_s3_id = Column(Integer, ForeignKey('archivos_s3.id'))
    contenido = Column(Text, nullable=False)
    numero_chunk = Column(Integer, nullable=False)
    embedding = Column(Vector(3072))
    
    # Relación
    archivo_s3 = relationship("ArchivoS3", back_populates="chunks")