import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from config import GeminiSettings, StorageSettings
from db import Database
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from routes.ingest import router as ingest_router

logger = logging.getLogger(__name__)


async def probe_ingestion_database(database: Database) -> None:
    async with database.engine.connect() as connection:
        await connection.execute(text("SELECT 1"))


async def configure_ingestion(application: FastAPI) -> None:
    application.state.ingest_database = None
    if not os.getenv("DATABASE_URL", "").strip():
        application.state.ingest_status = "disabled (missing DATABASE_URL)"
        return
    if not os.getenv("AI_SERVICE_API_KEY", "").strip():
        application.state.ingest_status = "disabled (missing AI_SERVICE_API_KEY)"
        return
    try:
        database = Database(StorageSettings.from_env())
        application.state.ingest_database = database
        await asyncio.wait_for(probe_ingestion_database(database), timeout=2.0)
    except (TimeoutError, OSError, RuntimeError, SQLAlchemyError):
        logger.warning("Ingestion startup probe failed: database unavailable.")
        application.state.ingest_status = "disabled (db unreachable)"
        return
    application.state.ingest_status = "ready"

from routes import legacy
from routes.grounding import make_grounding_router
from routes.legacy import make_legacy_router

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
gemini_settings = GeminiSettings.from_env()
GoogleGenerativeAIEmbeddings = legacy.GoogleGenerativeAIEmbeddings
_get_or_load_vector_store = legacy._get_or_load_vector_store

@asynccontextmanager
async def lifespan(application: "FastAPI") -> AsyncIterator[None]:
    """
    Lifespan handler: runs on startup and shutdown.
    Replaces the deprecated @app.on_event("startup") pattern.
    """
    gemini_settings.validate()
    await configure_ingestion(application)

    print("[Startup] AkadVerse Concept Explainer initializing...")
    print(f"[Startup] FAISS index path: '{legacy.FAISS_INDEX_PATH}'")
    print(f"[Startup] Max upload size: {legacy.MAX_UPLOAD_SIZE_MB} MB")
    if os.path.exists(legacy.FAISS_INDEX_PATH):
        print("[Startup] Existing FAISS index found on disk — will load on first request.")
    else:
        print("[Startup] No existing index found — awaiting first PDF upload.")
    try:
        yield
    finally:
        database = application.state.ingest_database
        if database is not None:
            await database.dispose()
    print("[Shutdown] AkadVerse Concept Explainer stopped.")


app = FastAPI(
    title="AkadVerse Concept Explainer API",
    description=(
        "Tier 5 RAG-powered study assistant. Upload lecture notes as PDF, "
        "then ask questions to receive structured pedagogical explanations. "
        "Also supports web-grounded explanations via /explain-with-grounding."
    ),
    version="1.2",
    lifespan=lifespan
)

legacy_router = make_legacy_router(GOOGLE_API_KEY, settings=gemini_settings)
# Resolve these module symbols at call time so existing test monkeypatches apply.
# The router captures the API key separately, keeping environment reloads isolated.
legacy_router.legacy_dependencies["embeddings"] = (
    lambda *args, **kwargs: GoogleGenerativeAIEmbeddings(*args, **kwargs)
)
legacy_router.legacy_dependencies["vector_store"] = (
    lambda embeddings: _get_or_load_vector_store(embeddings)
)
app.include_router(legacy_router)
grounding_router = make_grounding_router(GOOGLE_API_KEY, settings=gemini_settings)
app.include_router(grounding_router)
app.include_router(ingest_router)

@app.get("/health", tags=["System"])
async def health_check():
    """
    Reports service status, FAISS index state, and available endpoints.
    """
    index_in_memory = legacy._vector_store is not None
    index_on_disk = os.path.exists(legacy.FAISS_INDEX_PATH)

    return {
        "status": "ok",
        "provider": gemini_settings.llm_provider,
        "ingest_status": getattr(app.state, "ingest_status", "disabled (db unreachable)"),
        "version": "1.2",
        "index_in_memory": index_in_memory,
        "index_on_disk": index_on_disk,
        "rag_status": (
            "Ready — index loaded." if index_in_memory
            else "Ready — index on disk, loads on first question." if index_on_disk
            else "Awaiting PDF upload via /upload-note."
        ),
        "grounding_status": "Ready — no PDF required for /explain-with-grounding.",
        "endpoints": {
            "POST /upload-note": "Upload a PDF lecture note for RAG Q&A.",
            "POST /explain-concept": "Ask a question answered from your uploaded notes.",
            "POST /explain-with-grounding": "Ask any question answered from live web search.",
            "GET  /health": "This endpoint."
        }
    }

