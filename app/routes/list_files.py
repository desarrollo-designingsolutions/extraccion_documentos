from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.models import Files
from app.database import get_db
from .schemas import RespuestaExito
import logging
import requests
import os
from pydantic import BaseModel
from app.utils.helpers import (
    get_s3_client
)

router = APIRouter()

# Logging simple pero informativo
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("list_files")

@router.get("/list_files", response_model=RespuestaExito)
async def list_files(db: Session = Depends(get_db)):
    files = db.query(Files).order_by(Files.id.desc()).all()
    objetos = [
        {
            "id": f.id,
            "name": f.name,
            "nit": f.nit,
            "invoice_number": f.invoice_number,
            "size": f.size,
            "url_preassigned": f.url_preassigned,
            "json_category": f.json_category,
            "created_at": f.created_at,
        }
        for f in files
    ]
    return RespuestaExito(code=200, objetos=objetos)

class GetUrlRequest(BaseModel):
    file_id: int

@router.post("/get_url_file")
async def get_url_file(payload: GetUrlRequest, db: Session = Depends(get_db)):
    file_id = payload.file_id

    if not file_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="La 'file_id' es requerida")

    try:
        file_record = db.query(Files).filter(Files.id == payload.file_id).first()

        # Si hay URL en BD, intentar validar su vigencia
        if file_record and file_record.url_preassigned:
            resp = None
            try:
                resp = requests.head(file_record.url_preassigned, timeout=5, allow_redirects=True)
            except Exception:
                resp = None

            # Si HEAD no confirma (no es 200), intentar GET para leer posible XML de error (p.ej. "Request has expired")
            if resp is None or resp.status_code != 200:
                try:
                    resp = requests.get(file_record.url_preassigned, timeout=5, allow_redirects=True)
                except Exception:
                    resp = None

            # Si HEAD/GET devolvió 200, usar la URL almacenada
            if resp is not None and resp.status_code == 200:
                return {
                    "code": 200,
                    "url": file_record.url_preassigned
                }

        # Generar nueva URL presignada
        s3 = get_s3_client()
        expires = int(os.getenv("PRESIGNED_EXPIRATION", "3600"))
        try:
            url_presignada = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": os.getenv("AWS_BUCKET"), "Key": file_record.name},
                ExpiresIn=expires,
            )
        except Exception as e:
            logger.exception(f"Error generando presigned URL desde S3 para {file_record.name}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error generando presigned URL")

        # Guardar/actualizar en BD si existe registro
        if file_record:
            try:
                file_record.url_preassigned = url_presignada
                db.add(file_record)
                db.commit()
                db.refresh(file_record)
            except Exception as e:
                logger.exception(f"Error guardando URL presignada en DB para {file_record.name}: {e}")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error generando presigned URL para {file_record.name}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error generando presigned URL")

    return {
        "code": 200,
        "url": url_presignada
    }