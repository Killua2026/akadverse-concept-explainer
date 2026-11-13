import os
import asyncio
import logging
import sys
import re
import tempfile
import time
from pathlib import Path
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
from services.attachments import sweep_expired_attachments
from services.chat_history import sweep_expired_chat_messages
from routes.attachments import make_attachments_router

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(name)s - %(message)s",
    stream=sys.stdout,
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
cleanup_logger = logger.getChild("cleanup")

_WORKER_TEMP_FILE_PATTERN = re.compile(r"^[0-9a-f]{32}_")


def sweep_stale_temp_files(max_age_seconds: int = 3600) -> int:
    """Remove orphaned worker temp files left by a previous crashed run.

    The worker writes /tmp/{uuid4().hex}_{filename} and deletes it in a
    finally block on every code path except SIGKILL. On Railway, /tmp
    is ephemeral and cleared on container restart, so this sweep is
    defensive insurance for the case where a worker crashes mid-job but
    the container keeps running.
    """
    cutoff = time.time() - max_age_seconds
    removed = 0
    tmp_dir = Path(tempfile.gettempdir())
    try:
        for tmp_file in tmp_dir.iterdir():
            if not tmp_file.is_file():
                continue
            if not _WORKER_TEMP_FILE_PATTERN.match(tmp_file.name):
                continue
            try:
                if tmp_file.stat().st_mtime < cutoff:
                    tmp_file.unlink()
                    removed += 1
            except OSError:
                # File already gone, or permission denied: ignore and continue.
                continue
    except OSError:
        logger.warning("temp_sweep_dir_unreadable")
    return removed


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
    database = None
    for attempt in range(1, 4):
        try:
            if database is None:
                database = Database(StorageSettings.from_env())
            await asyncio.wait_for(probe_ingestion_database(database), timeout=20.0)
        except TimeoutError:
            logger.warning("Startup DB probe attempt %s/3: timeout", attempt)
        except (OSError, RuntimeError, SQLAlchemyError):
            logger.warning("Startup DB probe attempt %s/3: error", attempt)
        else:
            logger.info("Startup DB probe attempt %s/3: success", attempt)
            break
        if attempt < 3:
            await asyncio.sleep(1)
    else:
        if database is not None:
            await database.dispose()
        application.state.ingest_database = None
        application.state.ingest_status = "disabled (db unreachable)"
        return
    application.state.ingest_database = database
    if not os.getenv("AI_SERVICE_API_KEY", "").strip():
        application.state.ingest_status = "disabled (missing AI_SERVICE_API_KEY)"
        return
    if not os.getenv("AI_SERVICE_CALLBACK_SECRET", "").strip():
        application.state.ingest_status = "disabled (missing AI_SERVICE_CALLBACK_SECRET)"
        return
    if not any(host.strip() for host in os.getenv("AI_SERVICE_CALLBACK_ALLOWED_HOSTS", "").split(",")):
        application.state.ingest_status = "disabled (missing AI_SERVICE_CALLBACK_ALLOWED_HOSTS)"
        return
    application.state.ingest_status = "ready"

from routes.grounding import make_grounding_router
from routes.explain import make_explain_router
from routes.lesson_planning import make_lesson_planning_router

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
gemini_settings = GeminiSettings.from_env()

async def _run_cleanup_sweeps(database: Database) -> None:
    for resource, sweep in (
        ("attachments", sweep_expired_attachments),
        ("chat_messages", sweep_expired_chat_messages),
    ):
        try:
            count = await sweep(database)
        except Exception as exc:
            cleanup_logger.warning("%s.sweep_failed error_type=%s", resource, type(exc).__name__)
        else:
            cleanup_logger.info("%s.swept count=%s", resource, count)


async def _periodic_cleanup_sweep(database: Database) -> None:
    while True:
        await asyncio.sleep(3600)
        await _run_cleanup_sweeps(database)


# Preserve the pre-Phase 4a lifecycle hooks for existing callers.
_periodic_attachment_sweep = _periodic_cleanup_sweep


@asynccontextmanager
async def lifespan(application: "FastAPI") -> AsyncIterator[None]:
    """
    Lifespan handler: runs on startup and shutdown.
    Replaces the deprecated @app.on_event("startup") pattern.
    """
    gemini_settings.validate()
    application.state.gemini_settings = gemini_settings
    worker_count = get_worker_count()
    application.state.worker_count = worker_count
    application.state.worker_tasks = []
    application.state.cleanup_sweep_task = None
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
    swept = await asyncio.to_thread(sweep_stale_temp_files)
    if swept:
        logger.info("temp_sweep_removed count=%s", swept)
    if not gemini_settings.use_enterprise:
        application.state.ingest_status = "disabled (ingestion requires enterprise mode)"
        application.state.worker_status = "disabled (ingestion requires enterprise mode)"
    else:
        await configure_ingestion(application)
        database = get_ingestion_database(application)
        if database is not None:
            await _run_cleanup_sweeps(database)
            application.state.cleanup_sweep_task = asyncio.create_task(_periodic_cleanup_sweep(database))
            application.state.attachment_sweep_task = application.state.cleanup_sweep_task
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
        cleanup_sweep_task = application.state.cleanup_sweep_task
        if cleanup_sweep_task is not None:
            cleanup_sweep_task.cancel()
            await asyncio.gather(cleanup_sweep_task, return_exceptions=True)
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
        "Course material retrieval is available via /explain. "
        "Also supports web-grounded explanations via /explain-with-grounding."
    ),
    version="1.2",
    lifespan=lifespan
)

grounding_router = make_grounding_router(GOOGLE_API_KEY, settings=gemini_settings)
app.include_router(grounding_router)
app.include_router(make_explain_router(gemini_settings))
app.include_router(ingest_router)
app.include_router(documents_router)
app.include_router(make_attachments_router())
app.include_router(make_lesson_planning_router(gemini_settings))


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
        "grounding_status": "Ready - requires shared-secret auth.",
        "endpoints": {
            "POST /explain": "Answer from course material for an offering and week.",
            "GET /documents/{id}/status": "Read ingestion status.",
            "POST /ingest": "Queue a course document for ingestion.",
            "DELETE /documents/{id}": "Delete a document and its ingestion jobs.",
            "POST /explain-with-grounding": "Ask any question answered from live web search.",
            "GET /health": "This endpoint."
        }
    }

