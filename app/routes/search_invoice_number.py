from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Optional
from app.database import get_db
from app.models import Files, InvoiceAudits

router = APIRouter()

def extract_invoice_number(file_name: str) -> Optional[str]:
    """
    Extrae el invoice_number de un path como:
    '800033723/800033723/800033723-FEBQ170778.pdf'

    Regla: el NIT antes del '-' debe coincidir con el primer nivel.
    Retorna el invoice_number o None si no es válido.
    """
    try:
        parts = file_name.split("/")  # dividir en niveles
        if not parts:
            return None

        nit_base = parts[0]  # primer nivel siempre es el NIT
        invoice_number = None

        # Revisar segundo y tercer nivel
        for i in range(1, min(3, len(parts))):
            if "-" in parts[i]:
                segment = parts[i].replace(".pdf", "")
                nit_candidate, invoice_candidate = segment.split("-", 1)
                if nit_candidate == nit_base:  # validar coincidencia
                    invoice_number = invoice_candidate
                    break

        # Si no se encontró en niveles, revisar archivo final
        if not invoice_number and "-" in parts[-1]:
            filename = parts[-1].replace(".pdf", "")
            nit_candidate, invoice_candidate = filename.split("-", 1)
            if nit_candidate == nit_base:  # validar coincidencia
                invoice_number = invoice_candidate

        return invoice_number
    except Exception:
        return None


@router.post("/search_invoice_number/")
def search_invoice_number(db: Session = Depends(get_db)):
    try:
        files = db.query(Files).filter(Files.invoice_number == None).all()

        results = []
        exists = None
        for f in files:
            invoice_number = extract_invoice_number(f.name)

            if invoice_number:
                # 👇 Validar existencia en InvoiceAudits
                exists = db.query(InvoiceAudits).filter(
                    InvoiceAudits.invoice_number == invoice_number
                ).first()

                if exists:
                    f.invoice_number = invoice_number
                    db.commit()
                    db.refresh(f)

            results.append({
                "id": f.id,
                "name": f.name,
                "invoice_number": f.invoice_number
            })

        return {
            "results": results,
            "exists": exists
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error inesperado: {str(e)}"
        )