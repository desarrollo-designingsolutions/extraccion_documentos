import json
import redis
from fastapi import APIRouter
from typing import Any
import os


router = APIRouter()
r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))

@router.get("/jobs/{job_id}/status")
async def job_status(job_id: str) -> Any:
    data = r.get(f"job:{job_id}")
    if not data:
        return {"job_id": job_id, "status": "not_found"}
    return json.loads(data)
