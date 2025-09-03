# app/routes/health.py
from fastapi import APIRouter, HTTPException, Depends
import logging
from app.services.auth_service import verify_token, verify_token_v2

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/", response_model=dict)
async def health_check(current_user: dict = Depends(verify_token)):
    """Kiểm tra trạng thái API."""
    logger.info("Health check requested")
    return {"status": "ok"}


@router.get("/v2", response_model=dict)
async def health_check_v2(current_user: dict = Depends(verify_token_v2)):
    """Kiểm tra trạng thái API."""
    logger.info("Health check requested")
    return {"status": "ok"}