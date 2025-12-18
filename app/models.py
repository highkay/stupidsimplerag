from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Incoming query body for /chat."""

    query: str = Field(..., description="User question, plain text")
    start_date: Optional[str] = Field(None, description="YYYY-MM-DD lower bound")
    end_date: Optional[str] = Field(None, description="YYYY-MM-DD upper bound")
    filename: Optional[str] = Field(None, description="Exact filename to filter on")
    filename_contains: Optional[str] = Field(
        None, description="Case-insensitive substring/full-text match on filename"
    )
    keywords_any: Optional[List[str]] = Field(
        None, description="Return slices that contain at least one of these keywords"
    )
    keywords_all: Optional[List[str]] = Field(
        None, description="Return slices that contain all of these keywords"
    )
    skip_rerank: bool = Field(
        False, description="If true, skip rerank and return similarity results directly"
    )
    skip_generation: bool = Field(
        False, description="If true, skip LLM generation and only return retrieved sources"
    )


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
    error: Optional[str] = None
    doc_hash: Optional[str] = None


class TextIngestRequest(BaseModel):
    content: str = Field(..., description="Raw Markdown or text content")
    filename: Optional[str] = Field(None, description="Optional logical filename for tracking")


class LLMAnalysis(BaseModel):
    summary: str = ""
    table_narrative: str = ""
    keywords: List[str] = Field(default_factory=list)

    @field_validator("keywords", mode="before")
    @classmethod
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
