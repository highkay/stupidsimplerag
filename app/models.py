from typing import List, Optional

from pydantic import BaseModel, Field, validator


class ChatRequest(BaseModel):
    """Incoming query body for /chat."""

    query: str = Field(..., description="User question, plain text")
    start_date: Optional[str] = Field(None, description="YYYY-MM-DD lower bound")
    end_date: Optional[str] = Field(None, description="YYYY-MM-DD upper bound")


class SourceItem(BaseModel):
    filename: str
    date: Optional[str] = None
    score: Optional[float] = None
    keywords: Optional[str] = None
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = Field(default_factory=list)


class IngestResponse(BaseModel):
    status: str
    chunks: int
    filename: str


class LLMAnalysis(BaseModel):
    summary: str = ""
    table_narrative: str = ""
    keywords: List[str] = Field(default_factory=list)

    @validator("keywords", pre=True, always=True)
    def normalize_keywords(cls, value):
        if not value:
            return []
        normalized = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text and text not in normalized:
                normalized.append(text)
        return normalized[:8]
