"""Inbound ingestion contract."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class AttachmentResponse(BaseModel):
    attachment_id: str
    filename: str
    mime_type: str
    size_bytes: int
    created_at: datetime
    expires_at: datetime


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
    attachment_ids: list[str] = Field(default_factory=list)
    session_id: str = ""
    concept_id: str = ""
    concept_description: str = ""


class ChatMessage(BaseModel):
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    sources_json: list | None = None
    created_at: datetime


class ExplainSource(BaseModel):
    document_id: str | None = None
    title: str
    material_type: str
    chunk_preview: str
    source_type: Literal["course_material", "attachment", "ai_generated"] = "course_material"
    attachment_id: str | None = None
    filename: str | None = None


class RetrievalInfo(BaseModel):
    chunks_available: int
    chunks_above_threshold: int
    similarity_threshold: float
    attachment_chunks_available: int = 0
    attachment_chunks_above_threshold: int = 0
    attachment_images_count: int = 0


class ExplainResponse(BaseModel):
    answer: str
    sources: list[ExplainSource]
    retrieval_info: RetrievalInfo


class AnalyseMaterialRequest(BaseModel):
    document_id: str = Field(..., min_length=1)
    max_concepts: int = Field(default=12, ge=3, le=20)


class LessonPlanConcept(BaseModel):
    concept_id: str
    title: str
    description: str
    learning_objective: str
    difficulty: Literal["introductory", "intermediate", "advanced"]
    prerequisite_concept_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)


class LessonPlanSection(BaseModel):
    section_id: str
    title: str
    concept_ids: list[str]


class LessonPlan(BaseModel):
    document_id: str
    sections: list[LessonPlanSection]
    concepts: list[LessonPlanConcept]
    missing_prerequisites: list[str] = Field(default_factory=list)
    notes: str | None = None


class AnalyseMaterialResponse(BaseModel):
    document_id: str
    lesson_plan: LessonPlan
    chunks_analysed: int


class GenerateExampleTasksRequest(BaseModel):
    concept_id: str = Field(..., min_length=1)
    concept_description: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    max_tasks: int = Field(default=6, ge=3, le=10)
    example_text: str = ""


class ExampleTask(BaseModel):
    task_id: str
    sequence: int
    prompt: str
    response_type: Literal["text", "number", "multiple_choice", "step"]
    expected_answer: str
    evaluation_criteria: list[str] = Field(default_factory=list)
    hint_strategy: str | None = None
    required: bool = True


class WorkedExample(BaseModel):
    source: Literal["LECTURE_NOTE", "AI_GENERATED"]
    text: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class GenerateExampleTasksResponse(BaseModel):
    concept_id: str
    example: WorkedExample
    tasks: list[ExampleTask]


class Misconception(BaseModel):
    label: str
    evidence: str


class EvaluatePracticeRequest(BaseModel):
    concept_id: str = Field(..., min_length=1)
    concept_description: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    expected_answer: str = Field(..., min_length=1)
    evaluation_criteria: list[str] = Field(default_factory=list)
    student_answer: str = Field(..., min_length=1)
    student_reasoning: str = ""
    response_type: Literal["text", "number", "multiple_choice", "step"] = "text"


class EvaluatePracticeResponse(BaseModel):
    concept_id: str
    answer_correct: bool
    answer_feedback: str
    reasoning_quality: Literal["strong", "adequate", "weak", "absent"]
    reasoning_feedback: str
    misconceptions: list[Misconception] = Field(default_factory=list)
    evidence_strength: Literal["independent", "assisted", "insufficient"]
    needs_review: bool
    recommended_next_step: Literal["progress", "reinforce", "teach_prerequisite"]
