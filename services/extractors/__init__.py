"""Dispatch document extraction using the declared MIME type."""

from . import docx, pdf, pptx
from .errors import ExtractionError


def extract_text(file_path: str, mime_type: str) -> str:
    extractors = {
        "application/pdf": pdf.extract_text,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": docx.extract_text,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": pptx.extract_text,
    }
    extractor = extractors.get(mime_type)
    if extractor is None:
        raise ExtractionError("Unsupported file type. Supported: PDF, DOCX, PPTX.")
    return extractor(file_path)
