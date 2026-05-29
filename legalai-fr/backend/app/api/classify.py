from fastapi import APIRouter

from backend.app.schemas.classification import (
    ClassificationRequest,
    ClassificationResponse,
)
from backend.app.services.ml_service import classify_text


router = APIRouter(prefix="/ai", tags=["AI Classification"])


@router.post("/classify", response_model=ClassificationResponse)
def classify(request: ClassificationRequest) -> ClassificationResponse:
    result = classify_text(request.text)
    return ClassificationResponse(**result)