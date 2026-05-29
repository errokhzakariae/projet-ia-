from fastapi import APIRouter

from backend.app.schemas.summarization import (
    SummarizationRequest,
    SummarizationResponse,
)
from backend.app.services.llm_service import summarize_text


router = APIRouter(prefix="/ai", tags=["LLM Summarization"])


@router.post("/summarize", response_model=SummarizationResponse)
def summarize(request: SummarizationRequest) -> SummarizationResponse:
    result = summarize_text(
        text=request.text,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    return SummarizationResponse(**result)
