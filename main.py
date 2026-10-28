import os
import asyncio
import logging
import httpx
from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi.responses import JSONResponse
from workers.ingestion import recover_stuck_jobs, worker_loop
from workers.ingestion import logger as worker_logger

from fastapi import FastAPI
from config import GeminiSettings, StorageSettings, get_worker_count
from db import Database
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from routes.ingest import router as ingest_router
from routes.documents import router as documents_router
from services.callback import CallbackSettings

logger = logging.getLogger(__name__)


def get_ingestion_database(application: FastAPI) -> Database | None:
    return getattr(application.state, "ingest_database", None)


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
    if not os.getenv("AI_SERVICE_CALLBACK_SECRET", "").strip():
        application.state.ingest_status = "disabled (missing AI_SERVICE_CALLBACK_SECRET)"
        return
    if not any(host.strip() for host in os.getenv("AI_SERVICE_CALLBACK_ALLOWED_HOSTS", "").split(",")):
        application.state.ingest_status = "disabled (missing AI_SERVICE_CALLBACK_ALLOWED_HOSTS)"
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

from routes.grounding import make_grounding_router

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
gemini_settings = GeminiSettings.from_env()

@asynccontextmanager
async def lifespan(application: "FastAPI") -> AsyncIterator[None]:
    """
    Lifespan handler: runs on startup and shutdown.
    Replaces the deprecated @app.on_event("startup") pattern.
    """
    gemini_settings.validate()
    worker_count = get_worker_count()
    application.state.worker_count = worker_count
    application.state.worker_tasks = []
    callback_client = httpx.AsyncClient(trust_env=False, follow_redirects=False)
    application.state.callback_client = callback_client
    callback_settings = CallbackSettings.from_env()
    application.state.callback_settings = callback_settings
    def worker_log_context(record):
        if record.getMessage() == "worker.started":
            task = asyncio.current_task()
            tasks = application.state.worker_tasks
            if task in tasks:
                record.worker_index = tasks.index(task)
        return True
    worker_logging_enabled = False
    application.state.ingest_database = None
    if not gemini_settings.use_enterprise:
        application.state.ingest_status = "disabled (ingestion requires enterprise mode)"
        application.state.worker_status = "disabled (ingestion requires enterprise mode)"
    else:
        await configure_ingestion(application)
        database = get_ingestion_database(application)
        application.state.worker_status = (
            "disabled (ingest not configured)" if application.state.ingest_status.startswith("disabled (missing ")
            else "stopped (db unavailable)"
        )
        if application.state.ingest_status == "ready" and database is not None:
            try:
                await recover_stuck_jobs(database)
            except (SQLAlchemyError, OSError):
                logger.error("worker.db_unavailable")
                application.state.ingest_status = "disabled (db unreachable)"
            else:
                worker_logger.addFilter(worker_log_context)
                worker_logging_enabled = True
                application.state.worker_tasks = [
                    asyncio.create_task(worker_loop(database, gemini_settings, callback_client, callback_settings))
                    for _ in range(worker_count)
                ]
                application.state.worker_status = "running"
                def worker_finished(task):
                    if not task.cancelled() and task.exception() is not None:
                        application.state.worker_status = "stopped (db unavailable)"
                        application.state.ingest_status = "disabled (db unreachable)"
                for task in application.state.worker_tasks:
                    task.add_done_callback(worker_finished)

    print("[Startup] AkadVerse Concept Explainer initializing...")
    try:
        yield
    finally:
        tasks = application.state.worker_tasks
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await callback_client.aclose()
        if worker_logging_enabled:
            worker_logger.removeFilter(worker_log_context)
        database = get_ingestion_database(application)
        if database is not None:
            await database.dispose()
    print("[Shutdown] AkadVerse Concept Explainer stopped.")


app = FastAPI(
    title="AkadVerse Concept Explainer API",
    description=(
        "Study assistant with durable PDF, DOCX, and PPTX ingestion. "
        "Document retrieval is planned for Phase 3. "
        "Also supports web-grounded explanations via /explain-with-grounding."
    ),
    version="1.2",
    lifespan=lifespan
)

grounding_router = make_grounding_router(GOOGLE_API_KEY, settings=gemini_settings)
app.include_router(grounding_router)
app.include_router(ingest_router)
app.include_router(documents_router)


@app.middleware("http")
async def enterprise_ingestion_guard(request, call_next):
    if request.url.path == "/ingest" and getattr(
        request.app.state, "ingest_status", None
    ) == "disabled (ingestion requires enterprise mode)":
        return JSONResponse(status_code=503, content={
            "detail": "Ingestion requires LLM_PROVIDER=gemini_enterprise. Developer API mode is not supported for ingestion."
        })
    return await call_next(request)

@app.get("/health", tags=["System"])
async def health_check():
    """Reports service status and available endpoints."""

    return {
        "status": "ok",
        "provider": gemini_settings.llm_provider,
        "ingest_status": getattr(app.state, "ingest_status", "disabled (db unreachable)"),
        "worker_status": getattr(app.state, "worker_status", "disabled (ingest not configured)"),
        "worker_count": getattr(app.state, "worker_count", 0),
        "version": "1.2",
        "grounding_status": "Requires shared-secret auth and ready ingestion for /explain-with-grounding.",
        "endpoints": {
            "POST /ingest": "Queue a course document for ingestion.",
            "DELETE /documents/{id}": "Delete a document and its ingestion jobs.",
            "POST /explain-with-grounding": "Ask any question answered from live web search.",
            "GET /health": "This endpoint."
        }
    }

