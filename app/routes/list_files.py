from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from models import Files
from database import get_db
from .schemas import RespuestaExito

router = APIRouter()

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
    return RespuestaExito(mensaje="OK", objetos=objetos)