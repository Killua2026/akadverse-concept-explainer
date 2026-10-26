"""Pure extraction and chunking contract checks."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from pypdf import PdfWriter

from services.chunking import chunk_text
from services.extractors import ExtractionError, extract_text
from services.extractors import docx, pdf, pptx

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


# PDF
def test_pdf_text():
    assert pdf.extract_text(str(Path(__file__).parent / "fixtures/sample.pdf")) == "Lecture PDF content"


def test_pdf_empty(tmp_path):
    path = tmp_path / "empty.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.write(path)
    with pytest.raises(ExtractionError, match=r"^No extractable text \(scanned PDF\?\)$"):
        pdf.extract_text(str(path))


# DOCX
def test_docx_paragraphs(office_document):
    assert docx.extract_text(str(office_document("docx", paragraph="Lecture paragraph"))) == "Lecture paragraph"


def test_docx_tables(office_document):
    assert docx.extract_text(str(office_document("docx", table="Table cell"))) == "Table cell"


def test_docx_paragraph_then_table(office_document):
    assert docx.extract_text(str(office_document("docx", paragraph="First", table="Second"))) == "First\nSecond"


@pytest.mark.parametrize("module, kind, message", [
    (docx, "docx", "No extractable text in DOCX"),
    (pptx, "pptx", "No extractable text in PPTX"),
])
def test_office_empty(module, kind, message, office_document):
    with pytest.raises(ExtractionError, match=f"^{message}$"):
        module.extract_text(str(office_document(kind)))


@pytest.mark.parametrize("module, kind", [(pdf, "pdf"), (docx, "docx"), (pptx, "pptx")])
def test_corrupt_file_safe_error(module, kind, tmp_path):
    path = tmp_path / f"private-lecture.{kind}"
    path.write_bytes(b"corrupt document")
    with pytest.raises(ExtractionError) as caught:
        module.extract_text(str(path))
    assert str(path) not in str(caught.value)
    assert path.name not in str(caught.value)


# PPTX
def test_pptx_slide(office_document):
    assert pptx.extract_text(str(office_document("pptx", paragraph="Slide content"))) == "Slide content"


def test_pptx_notes(office_document):
    assert pptx.extract_text(str(office_document("pptx", paragraph="Slide", notes="Speaker notes"))) == "Slide\nSpeaker notes"


def test_pptx_notes_only(office_document):
    assert pptx.extract_text(str(office_document("pptx", notes="Lecture in notes"))) == "Lecture in notes"


# Dispatcher
@pytest.mark.parametrize("mime, module", [("application/pdf", pdf), (DOCX_MIME, docx), (PPTX_MIME, pptx)])
def test_dispatcher(mime, module, monkeypatch):
    extractor = Mock(return_value="Expected text")
    monkeypatch.setattr(module, "extract_text", extractor)
    assert extract_text("lecture", mime) == "Expected text"
    extractor.assert_called_once_with("lecture")


def test_dispatcher_unknown():
    with pytest.raises(ExtractionError, match="Unsupported file type"):
        extract_text("lecture", "application/zip")


# Chunking
def test_chunk_boundaries():
    content = "".join(chr(0x4E00 + i) for i in range(1801))
    chunks = chunk_text(content)
    assert chunks == [content[:1000], content[800:1800], content[1600:]]
    assert chunks[0][-200:] == chunks[1][:200]


def test_chunk_short():
    assert chunk_text("Short lecture") == ["Short lecture"]


def test_chunk_empty():
    assert chunk_text("") == []
