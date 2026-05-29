from pydantic import BaseModel, Field


class ExtractionRequest(BaseModel):
    text: str = Field(..., min_length=5)


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class ExtractionResponse(BaseModel):
    entities: list[Entity]
    legal_keywords: list[str]
    amounts: list[str]
    articles: list[str]