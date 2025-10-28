from sqlalchemy import Column, Integer, String, BigInteger, DateTime, Text, ForeignKey, Numeric
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
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
    # JSON categorizado por la LLM (puede ser NULL)
    json_category = Column(JSONB, nullable=True)
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

class InvoiceAudits(Base):
    __tablename__ = "invoice_audits"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    company_id = Column(Integer, nullable=True)
    third_id = Column(Integer, nullable=True)
    invoice_number = Column(String, nullable=True)
    total_value = Column(Numeric(15, 2), nullable=True)
    origin = Column(String, nullable=True)
    expedition_date = Column(DateTime(timezone=True), nullable=True)
    date_entry = Column(DateTime(timezone=True), nullable=True)
    date_departure = Column(DateTime(timezone=True), nullable=True)
    modality = Column(String, nullable=True)
    regimen = Column(String, nullable=True)
    coverage = Column(String, nullable=True)
    contract_number = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)