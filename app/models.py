from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


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
    scope: Optional[str] = Field(
        None, description="Optional logical scope/namespace for filtering (e.g. 'reports/2025')"
    )


class SourceItem(BaseModel):
    filename: str
    date: Optional[str] = None
    score: Optional[float] = None
    keywords: Optional[str] = None
    text: str
    scope: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = Field(default_factory=list)


class IngestResponse(BaseModel):
    status: str
    chunks: int
    filename: str
    error: Optional[str] = None
    doc_hash: Optional[str] = None
    scope: Optional[str] = None


class DocumentInfo(BaseModel):
    filename: str
    date: Optional[str] = None
    chunks: int
    scope: Optional[str] = None


class TextIngestRequest(BaseModel):
    content: str = Field(..., description="Raw Markdown or text content")
    filename: Optional[str] = Field(None, description="Optional logical filename for tracking")
    force_update: bool = Field(False, description="If true, overwrite existing document with same filename")
    scope: Optional[str] = Field(None, description="Optional logical scope/namespace")


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


class GroundingDocumentSelector(BaseModel):
    doc_hash: Optional[str] = Field(
        None, description="Preferred stable document identity returned by ingest APIs"
    )
    filename: Optional[str] = Field(
        None, description="Fallback logical filename for locating one document"
    )
    scope: Optional[str] = Field(
        None, description="Optional logical scope/namespace for filename or doc_hash lookup"
    )

    @model_validator(mode="after")
    def validate_selector(self):
        if not self.doc_hash and not self.filename:
            raise ValueError("document selector requires doc_hash or filename")
        return self


class GroundingCandidate(BaseModel):
    identifier: Optional[str] = Field(
        None, description="Optional external identifier such as a ticker or entity id"
    )
    name: Optional[str] = Field(None, description="Primary display name")
    aliases: List[str] = Field(default_factory=list, description="Known aliases or short names")
    candidate_type: Optional[str] = Field(
        None, description="Optional type hint such as stock, company, product, topic"
    )

    @field_validator("aliases", mode="before")
    @classmethod
    def normalize_aliases(cls, value):
        if not value:
            return []
        normalized = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    @model_validator(mode="after")
    def validate_identity(self):
        if not self.name and not self.identifier:
            raise ValueError("grounding candidate requires name or identifier")
        return self


class GroundingRequest(BaseModel):
    document: GroundingDocumentSelector
    candidates: List[GroundingCandidate] = Field(
        default_factory=list,
        description="Candidates to ground against a single document",
    )
    assets: List[GroundingCandidate] = Field(
        default_factory=list,
        description="Deprecated alias for candidates",
    )
    scope: Optional[str] = Field(
        None, description="Optional scope override when document.scope is omitted"
    )
    max_excerpts: int = Field(
        3, ge=1, le=10, description="Maximum number of excerpts per asset result"
    )
    skip_rerank: bool = Field(
        False, description="If true, skip rerank and only use deterministic ranking"
    )

    @model_validator(mode="after")
    def normalize_candidates(self):
        if not self.candidates and self.assets:
            self.candidates = list(self.assets)
        if not self.candidates:
            raise ValueError("grounding request requires at least one candidate")
        return self


class GroundingExcerpt(BaseModel):
    section_type: str
    score: float
    text: str
    is_alias_hit: bool


class GroundingResult(BaseModel):
    identifier: Optional[str] = None
    name: str
    relevance_tier: str
    source_zone: str
    source_reason: str
    body_hit_count: int = 0
    qa_hit_count: int = 0
    list_hit_count: int = 0
    candidate_brief: str
    excerpts: List[GroundingExcerpt] = Field(default_factory=list)


class GroundingDocumentInfo(BaseModel):
    doc_hash: Optional[str] = None
    filename: str
    scope: Optional[str] = None
    date: Optional[str] = None
    summary: str = ""
    chunk_count: int = 0


class GroundingResponse(BaseModel):
    document: GroundingDocumentInfo
    candidate_results: List[GroundingResult] = Field(default_factory=list)
