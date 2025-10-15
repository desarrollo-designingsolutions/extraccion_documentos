from sqlalchemy import Column, Integer, String, BigInteger, DateTime, Text, ForeignKey # type: ignore
from sqlalchemy.orm import relationship # type: ignore
from sqlalchemy.sql import func # type: ignore
from pgvector.sqlalchemy import Vector # type: ignore
from database import Base

class Files(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True, nullable=False)
    nit = Column(String, index=True, nullable=False)
    invoice_number = Column(String, nullable=True)
    size = Column(BigInteger, nullable=False)
    url_preassigned = Column(String, nullable=False)
    text_extracted = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relación con chunks
    chunks = relationship("FilesChunks", back_populates="archivo_s3", cascade="all, delete-orphan")

class FilesChunks(Base):
    __tablename__ = "files_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    files_id = Column(Integer, ForeignKey('files.id'))
    content = Column(Text, nullable=False)
    chunk_number = Column(Integer, nullable=False)
    embedding = Column(Vector(3072))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relación
    archivo_s3 = relationship("Files", back_populates="chunks")