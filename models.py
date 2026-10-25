"""Inbound ingestion contract."""

from typing import Literal

from pydantic import BaseModel, HttpUrl


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
