from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):
    text: str = Field(..., min_length=5)


class ClassificationResponse(BaseModel):
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]