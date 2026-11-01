"""Shared ingestion chunk boundaries."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str) -> list[str]:
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)
