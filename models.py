"""Inbound ingestion contract."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class IngestMetadata(BaseModel):
    course_code: str
    course_title: str
    course_offering_id: str
    academic_session: str
    semester: str
    start_week: int
    end_week: int
    weeks: list[int]
    material_type: str
    title: str
    document_id: str


class IngestRequest(BaseModel):
    document_id: str
    file_url: HttpUrl
    file_name: str
    mime_type: str
    callback_url: HttpUrl
    metadata: IngestMetadata


class IngestResponse(BaseModel):
    status: Literal["queued"]
    document_id: str


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: Literal["queued", "processing", "done", "failed"]
    chunk_count: int
    attempts: int
    last_error: str | None
    created_at: datetime
    updated_at: datetime


class ExplainRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    course_offering_id: str = Field(..., min_length=1)
    week: int = Field(..., ge=1)


class ExplainSource(BaseModel):
    document_id: str
    title: str
    material_type: str
    chunk_preview: str


class RetrievalInfo(BaseModel):
    chunks_available: int
    chunks_above_threshold: int
    similarity_threshold: float


class ExplainResponse(BaseModel):
    answer: str
    sources: list[ExplainSource]
    retrieval_info: RetrievalInfo
