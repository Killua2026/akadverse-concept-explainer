"""PDF text extraction preserving page order and whitespace."""

from pathlib import Path

from pypdf import PdfReader

from .errors import ExtractionError


def extract_text(file_path: str) -> str:
    try:
        with Path(file_path).open("rb") as handle:
            reader = PdfReader(handle)
            content = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as error:
        raise ExtractionError("Unable to extract text from PDF") from error
    if not content.strip():
        raise ExtractionError("No extractable text (scanned PDF?)")
    return content
