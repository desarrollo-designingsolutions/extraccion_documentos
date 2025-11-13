from sqlalchemy import Column, Integer, String, BigInteger, DateTime, Text, ForeignKey, Numeric
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.database import Base

class Files(Base):
    __tablename__ = "files"
    __table_args__ = {'extend_existing': True}

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
    __table_args__ = {'extend_existing': True}
    
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
    __table_args__ = {'extend_existing': True}

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

class TemporaryFiles(Base):
    __tablename__ = "temporary_files"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    name = Column(String, index=True, nullable=False)
    original_filename = Column(String, nullable=False)
    size = Column(BigInteger, nullable=False)
    text_extracted = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    # NUEVO CAMPO: Para asociar archivos temporales con conversaciones
    conversation_id = Column(Integer, ForeignKey('conversation_sessions.id'), nullable=True)
    
    # Relación con chunks temporales
    chunks = relationship("TemporaryFilesChunks", back_populates="temporary_file", cascade="all, delete-orphan")
    # NUEVA RELACIÓN: Para acceder a la conversación desde el archivo temporal
    conversation = relationship("ConversationSession", back_populates="temporary_files")

class TemporaryFilesChunks(Base):
    __tablename__ = "temporary_files_chunks"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    temporary_files_id = Column(Integer, ForeignKey('temporary_files.id'))
    content = Column(Text, nullable=False)
    chunk_number = Column(Integer, nullable=False)
    embedding = Column(Vector(3072))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relación
    temporary_file = relationship("TemporaryFiles", back_populates="chunks")

class ConversationSession(Base):
    __tablename__ = "conversation_sessions"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    # CAMBIO: Quitamos unique=True para permitir múltiples conversaciones por sesión
    session_id = Column(String, index=True, nullable=False)  # Para NotebookLM
    file_id = Column(Integer, ForeignKey('files.id'), nullable=True)     # Para chat individual (opcional)
    title = Column(String, nullable=True)                                # Título automático de la conversación
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relaciones
    messages = relationship("ConversationMessage", back_populates="session", cascade="all, delete-orphan")
    file = relationship("Files")  # Opcional, para chats de archivos específicos
    # NUEVA RELACIÓN: Para acceder a los archivos temporales de esta conversación
    temporary_files = relationship("TemporaryFiles", back_populates="conversation")

class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey('conversation_sessions.id'))
    role = Column(String, nullable=False)  # 'user' o 'assistant'
    content = Column(Text, nullable=False)
    message_metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relación
    session = relationship("ConversationSession", back_populates="messages")