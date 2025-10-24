from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from botocore.exceptions import ClientError
import pandas as pd
from io import BytesIO
from collections import defaultdict
from utils.helpers import get_s3_client
import os

router = APIRouter()


@router.get("/estadisticas_archivos/")
def obtener_estadisticas_archivos(
    prefix: str = Query(
        "", description="Prefijo opcional para filtrar objetos (ej. carpeta/)"
    ),
):
    bucket_name = os.getenv("AWS_BUCKET")
    if not bucket_name:
        raise HTTPException(
            status_code=500, detail="Variable de entorno AWS_BUCKET no configurada"
        )

    s3 = get_s3_client()
    try:
        stats_por_nit = defaultdict(lambda: {"subfolders": set(), "files": 0})

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

        for page in pages:
            objetos = page.get("Contents", [])
            for obj in objetos:
                key = obj["Key"]
                segmentos = key.split("/")
                if len(segmentos) < 2:
                    continue

                nit = segmentos[0]
                if not nit:
                    continue

                if obj["Size"] > 0 and not key.endswith("/"):
                    stats_por_nit[nit]["files"] += 1
                    subfolder = "/".join(segmentos[1:-1])
                    if subfolder:
                        stats_por_nit[nit]["subfolders"].add(subfolder)

                elif key.endswith("/"):
                    subfolder = "/".join(segmentos[1:])
                    if subfolder:
                        stats_por_nit[nit]["subfolders"].add(subfolder)

        datos_excel = []
        for nit, data in stats_por_nit.items():
            datos_excel.append(
                {
                    "Carpeta NIT": nit,
                    "Cantidad de subcarpetas": len(data["subfolders"]),
                    "Cantidad de archivos": data["files"],
                }
            )

        if not datos_excel:
            datos_excel = [
                {
                    "Carpeta NIT": "",
                    "Cantidad de subcarpetas": 0,
                    "Cantidad de archivos": 0,
                }
            ]

        df = pd.DataFrame(datos_excel)

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Estadísticas")

        output.seek(0)

        excel_filename = "estadisticas_nit.xlsx"

        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={excel_filename}"},
        )

    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchBucket":
            raise HTTPException(status_code=404, detail="Bucket no encontrado")
        else:
            raise HTTPException(
                status_code=500, detail=f"Error en S3: {e.response['Error']['Message']}"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {str(e)}")
