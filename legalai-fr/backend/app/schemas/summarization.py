from pydantic import BaseModel, Field


class SummarizationRequest(BaseModel):
    text: str = Field(..., min_length=20)
    max_tokens: int = Field(default=700, ge=100, le=2000)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


class SummarizationResponse(BaseModel):
    summary: str
    model: str
    usage: dict | None = None
