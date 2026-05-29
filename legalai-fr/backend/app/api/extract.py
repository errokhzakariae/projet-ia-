from fastapi import APIRouter

from backend.app.schemas.extraction import ExtractionRequest, ExtractionResponse
from backend.app.services.nlp_service import extract_information


router = APIRouter(prefix="/ai", tags=["NLP Extraction"])


@router.post("/extract", response_model=ExtractionResponse)
def extract(request: ExtractionRequest) -> ExtractionResponse:
    result = extract_information(request.text)
    return ExtractionResponse(**result)