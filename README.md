# AkadVerse Concept Explainer

FastAPI microservice for the AkadVerse Study Zone, reserved port **8010**.
The intended caller is the Next.js backend, not a browser client.

## Current status: Phase 2c complete

The current checkout includes Phase 2c (`f2f7334`): durable ingestion of PDF,
DOCX, and PPTX documents into Supabase Postgres with pgvector. Configurable
background workers process accepted jobs. Legacy FAISS endpoints and Google
Search Grounding remain available.

| Capability | Current state |
|---|---|
| Authenticated `POST /ingest` | Implemented; returns `202` after queue admission |
| Durable jobs and configurable workers | Implemented; startup recovery and concurrent claims |
| PDF / DOCX / PPTX extraction | Implemented |
| Shared chunking and Enterprise embeddings | Implemented |
| Delete endpoint and outbound callback client | Next: Phase 2d; not implemented |
| Legacy FAISS endpoint retirement | Phase 2e; not implemented |
| Retrieval over ingested pgvector chunks | Phase 3; not implemented |
| Stewart chat history | Phase 4; not implemented |
| ZIP ingestion | Rejected; deferred to a later phase |

`/ingest` writes pgvector chunks. It does not populate the legacy FAISS index,
and `/explain-concept` does not retrieve the newly ingested pgvector chunks.
Callbacks currently produce a `callback.would_fire` log only. A job marked
`done` means its chunks were stored, not that a callback was delivered.

## Contents

- [Architecture](#architecture)
- [Installation and startup](#installation-and-startup)
- [Configuration](#configuration)
- [Database setup](#database-setup)
- [API endpoints](#api-endpoints)
- [Extraction and processing](#extraction-and-processing)
- [Jobs and workers](#jobs-and-workers)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Project structure](#project-structure)
- [Integration and next steps](#integration-and-next-steps)

## Architecture

Current ingestion flow:

```text
Next.js backend -> POST /ingest -> Postgres jobs -> background worker
  -> signed-URL download -> MIME dispatcher -> PDF / DOCX / PPTX extraction
  -> chunking (1000 characters, 200 overlap) -> Enterprise embeddings
  -> transaction: replace document chunks and mark job done
  -> callback.would_fire log; temp file removed in finally
```

Separate explanation flows remain in place:

- Legacy: `/upload-note` accepts a PDF and builds a local FAISS index;
  `/explain-concept` retrieves from that index and generates an explanation.
- Grounding: `/explain-with-grounding` uses the selected Gemini provider and
  Google Search, returning an explanation with cited sources.

Explanation responses contain `concept`, `simple_explanation`, `analogy`,
`example`, and `worked_problem`. Grounding also returns `sources` entries
with `title` and `url`. These response fields do not describe ingestion responses.

## Installation and startup

Use Python **3.12** and **uv**, matching the parent project's `pyproject.toml`.
From the parent `akadverse-ai` directory, synchronize its environment:

```powershell
uv sync
```

Then, from `akadverse-concept-explainer/`, install this service's requirements:

```powershell
uv pip install -r requirements.txt
```

Run that install again after synchronizing or updating the root environment;
`uv run` does not automatically install the service's requirements file.
The root project supplies pytest, httpx, and multipart support.

Service requirements include FastAPI, Uvicorn, `google-genai>=1.67.0`,
LangChain integrations, `langchain-text-splitters`, `faiss-cpu`, Pydantic,
`pypdf`, `python-docx==1.2.0`, `python-pptx==1.0.2`,
`sqlalchemy[asyncio]==2.1.4`, `psycopg[binary]==3.2.6`, and
`python-dotenv==1.0.1`. `requirements.txt` is authoritative for service pins.

Copy `.env.example` to your private `.env` and supply the required settings.
Never commit or share credential values. Existing process variables take
precedence over values loaded by python-dotenv.

Start the service from this directory:

```powershell
uv run python run.py
```

`run.py` sets the Windows SelectorEventLoop policy before Uvicorn starts,
binds to `0.0.0.0:8010`, and selects the asyncio loop. It does not enable
reload. Direct Uvicorn startup bypasses this Windows policy setup.
Use [Swagger UI](http://127.0.0.1:8010/docs) to inspect the five current API
operations and their request schemas. Port 8010 belongs to this service;
consult the root `AGENTS.md` before choosing any other local testing port.

## Configuration

| Variable | Current use |
|---|---|
| `LLM_PROVIDER` | `gemini` by default; explicitly use `gemini_enterprise` for ingestion |
| `GOOGLE_API_KEY` | Required for Developer mode; never an HTTP form field |
| `GOOGLE_CLOUD_PROJECT` | Required for Enterprise mode |
| `GOOGLE_CLOUD_LOCATION` | Enterprise generation/discovery location; default `global` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Optional path to an external ADC credential file; runtime ADC is also supported |
| `DATABASE_URL` | Required for ingestion; private Postgres connection string |
| `AI_SERVICE_API_KEY` | Required shared secret for inbound ingestion authentication |
| `AI_SERVICE_CALLBACK_SECRET` | Reserved for Phase 2d; currently unused by a callback client |
| `WORKER_COUNT` | Positive integer, default `1`; restart required after changes |

Developer mode supports the existing explanation routes but disables ingestion
and workers. `/ingest` returns `503` with an Enterprise-mode requirement.
A Cloud project setting alone does not select Enterprise mode. Unknown providers,
missing required provider credentials/settings, and invalid worker counts fail
startup validation.

The implemented Enterprise path uses ADC with project and location. This
service does not implement an Enterprise Express Mode API-key configuration.
Configure credentials outside the repository and enable the necessary model
access for the selected Cloud project. A successful health response is not
proof of ADC permissions, billing, quota, or live model availability.

Generation models are discovered synchronously while loading settings and
cached. Discovery failures retain the configured priority list. Request errors
do not switch providers. Ingestion embeddings use the explicit constants in
`config.py`: `gemini-embedding-2`, `global`, and 768 output dimensions. Their
location is independent of `GOOGLE_CLOUD_LOCATION`; these are implementation
settings, not a claim that a particular project can access the model.

## Database setup

The service owns the `concept_explainer` schema in AkadVerse's Postgres instance.
It does not own or modify another service's tables.

1. Configure a reachable Supabase Postgres connection with permissions to apply
   the service migrations. Use the actual connection details from your dashboard.
   The documented local setup uses the Session pooler on port 5432.
2. Verify pgvector is enabled, using the Supabase dashboard or an authorized SQL
   session: `SELECT extname FROM pg_extension WHERE extname = 'vector';`.
3. Apply `migrations/001_pgvector_storage.sql`, then
   `migrations/002_jobs_table.sql`, manually through the SQL editor or psql.
   Run each complete transaction, including its local search path.

Migration 001 creates the schema, pgvector prerequisite, chunks table, and
indexes. Migration 002 creates durable jobs and the status/creation-time index.
They are not automatically applied at application startup, and neither migrates
existing FAISS data. `IF NOT EXISTS` supports reruns but does not reconcile
incompatible existing table definitions.

The `chunks` table stores a generated ID, document ID, content, `vector(768)`
embedding, JSONB metadata, and creation time. The `jobs` table stores a UUID,
unique document ID, status, JSONB payload, attempts, last error, and timestamps.

`DATABASE_URL` may use `postgresql://`; `db.py` selects `postgresql+psycopg`
internally. URL-encode special characters in the password component only.
The async engine uses `NullPool`, hidden SQL parameters, a 15-second driver
connection timeout, and disabled automatic prepared statements. Hidden
parameters do not make every possible traceback safe to share.

## API endpoints

### POST /ingest

Backend-only queue admission, authenticated with `Authorization: Bearer <shared-secret>`.
The token is compared using `secrets.compare_digest`. Provider authentication
is configured on the server and is separate from this inbound shared secret.

Example request with illustrative, non-secret values:

```json
{
  "document_id": "lecture-001",
  "file_url": "https://example.com/signed/lecture.docx",
  "file_name": "lecture.docx",
  "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "callback_url": "https://example.com/api/webhooks/ai-ingest",
  "metadata": {
    "course_code": "CSC 201",
    "course_title": "Data Structures",
    "course_offering_id": "offering-001",
    "academic_session": "2025/2026",
    "semester": "First",
    "start_week": 1,
    "end_week": 2,
    "weeks": [1, 2],
    "material_type": "NOTES",
    "title": "Introduction",
    "document_id": "lecture-001"
  }
}
```

All displayed request and metadata fields are required. The URL fields are
validated as HTTP URLs. `callback_url` is required by the request schema even
though outbound delivery is not implemented yet.

Success: HTTP `202`, `{"status":"queued","document_id":"lecture-001"}`.
This confirms queue admission, not extraction or embedding success.

| Response | Meaning |
|---|---|
| `401` | Missing or incorrect Bearer token when ingestion is otherwise ready |
| `422` | Invalid request fields or unsupported MIME type, including ZIP |
| `503` | Enterprise mode required, configuration missing, or database unavailable |

A new document starts queued with attempts zero. A duplicate document ID
requeues the existing job and increments attempts while preserving its original
payload. The worker increments attempts again when claiming a job. Therefore,
`attempts` is not purely a count of completed processing attempts. Reposting a
changed signed URL does not replace the existing job's original URL today.

### GET /health

Returns provider and legacy index status plus `ingest_status`, `worker_status`,
and `worker_count`. In ready Enterprise mode these include `ready`, `running`,
and the configured count. Disabled states identify missing ingestion settings,
Developer mode, or database unavailability. The configured count is not a count
of currently healthy tasks. Startup probes the database with a two-second
application timeout; health does not perform a fresh live LLM request.

The response's embedded `endpoints` map still lists the four historical
operations; `/ingest` is mounted and appears in OpenAPI despite that map omission.

### Existing explanation routes

| Route | Input | Behavior |
|---|---|---|
| `POST /upload-note` | Multipart `file`, PDF only, maximum 50 MB | Legacy FAISS indexing; success includes filename, chunk count, and page count |
| `POST /explain-concept` | Form `question` | Legacy FAISS retrieval and five-field explanation; returns `400` if notes are absent |
| `POST /explain-with-grounding` | Form `question` | Gemini with Google Search; five-field explanation plus cited sources |

The two FAISS routes remain available until Phase 2e. Do not build new
integrations against them. Grounding is retained independently of ingestion.
The local `akadverse_concept_faiss_index/` folder is generated legacy data and
must not be committed. Removing it does not delete pgvector chunks or jobs.
`[KAFKA MOCK]` logs do not implement Kafka or callback delivery.

## Extraction and processing

| MIME type | Extractor | Extracted content |
|---|---|---|
| `application/pdf` | pypdf `PdfReader` | Page text joined with newlines |
| `application/vnd.openxmlformats-officedocument.wordprocessingml.document` | python-docx | Paragraphs first, then table cells |
| `application/vnd.openxmlformats-officedocument.presentationml.presentation` | python-pptx | Slide shape text frames and existing non-empty speaker notes |

DOCX interleaved paragraph/table order is not preserved. The accepted policy is
to upgrade traversal only if a real lecturer document demonstrates lost
adjacency during chunking; see `AGENTS.md`. PDF OCR is not implemented, and
image-only content does not become text automatically.

`services.extractors.extract_text(file_path, mime_type)` dispatches to the
format modules. They share `ExtractionError`. Empty documents and corrupt
files fail rather than silently producing empty output; error messages do not
include the file path. Empty PDF text reports
`No extractable text (scanned PDF?)`. Office formats identify DOCX or PPTX in
their empty-text message. Unsupported dispatcher MIME types also raise
`ExtractionError`; the HTTP route rejects unsupported types before enqueueing.

`services.chunking.chunk_text(text)` preserves the splitter's 1000-character
size and 200-character overlap settings. Actual chunks can be shorter because
of separator boundaries. Extraction and chunking run through `asyncio.to_thread`
in the worker to keep blocking library work off the event loop.

The worker downloads to `/tmp/<uuid>_<sanitized-basename>`, follows redirects,
and enforces a 60-second download timeout and 20 MB byte limit. These limits
belong to ingestion, separate from the legacy upload route's 50 MB limit.
The remote Supabase object is never deleted or modified. The temporary file
is removed in `finally` on success or failure.

Embeddings are requested in batches of at most 10 chunks with eight seconds
between batches. The embedding service validates the number of vectors,
768 dimensions, and finite values. It requests one SDK attempt and does not
implement automatic retries. Failure marks the job failed with a safe message.
No fixed worker count or throughput is guaranteed safe for every Cloud project;
verify the applicable quota before increasing concurrency.

## Jobs and workers

Startup recovers processing jobs older than five minutes by requeueing them,
then launches `WORKER_COUNT` tasks. Each task claims one job at a time using
`FOR UPDATE SKIP LOCKED`; an empty queue is polled every two seconds. Recovery
runs at startup, not periodically. Shutdown cancels and awaits the tasks and
disposes the database engine.

| Status | Meaning in Phase 2c |
|---|---|
| `queued` | Waiting for a worker |
| `processing` | Claimed; attempts incremented |
| `done` | Document chunks replaced and job completion committed |
| `failed` | Processing failed; safe `last_error` stored, at most 500 characters |

Chunk deletion, insertion, and marking the job done occur in one transaction.
A failed insert rolls back replacement so previously stored chunks survive.
`ExtractionError` messages are preserved up to 500 characters; other failures
use approved safe messages. Callback logs are emitted for success or failure.
Failed jobs are not automatically retried; another admission request can requeue
them with the original payload. Worker database failures stop the affected task
and disable ingestion; no automatic worker restart is implemented.

## Testing

From this service directory:

```powershell
uv run pytest tests/ -q --tb=short
```

Latest full-suite verification on 26 October 2026: **86 passed**, with two
existing deprecation warnings (Starlette/AnyIO and langchain-community).
The latest worker-only verification reported **21 passed**; extractor-only
verification reported **20 passed**.

Tests cover provider configuration, authentication, queue admission, embedding
validation, extraction/dispatch, chunk boundaries, startup recovery, concurrent
workers, document replacement, and safe worker failures. Worker integration
uses real isolated Postgres inserts, real file extraction, mocked downloads,
and mocked embeddings. The PDF fixture is committed at
`tests/fixtures/sample.pdf`; DOCX/PPTX fixtures are generated programmatically.
The PDF ignore exception is intentional.

The test database must be reachable with pgvector enabled and permissions for
isolated schemas. Tests create unique `concept_explainer_test_<uuid>` schemas
and clean up or roll back their changes. Do not point test writes at production
service tables. `tests/conftest.py` sets the Windows selector policy.

Always use `--tb=short`; traceback locals can expose secrets. If more detail is
necessary, run only the failing test with `--tb=long` and redact before sharing.
Automated tests do not establish live ADC access, model availability, Cloud
quota, or actual callback delivery. Provider tests mock external generation;
legacy FAISS behavior is not an end-to-end live-service acceptance test.

## Troubleshooting

- **Ingestion unavailable:** check `/health`, Enterprise selection, private
  database/auth configuration, both migrations, and database reachability.
- **Windows Psycopg/Proactor error:** launch through `uv run python run.py`;
  do not switch to a synchronous driver to hide the loop configuration problem.
- **Missing packages:** run `uv pip install -r requirements.txt` in the root uv
  environment. Locally installed Office packages do not replace declared pins.
- **Expired download URL:** a download `403` records
  `Signed URL expired, retry needed`. Reposting the same ID currently retains
  the old payload; coordinate URL refresh behavior rather than assuming it updates.
- **Empty/scanned/corrupt file:** inspect `jobs.last_error`; there is no OCR fallback.
  Unlike queue validation errors, these failures happen after the `202` response.
- **Embedding failure:** verify access and quota for the configured model and
  location without exposing SDK exceptions or credentials.
- **Port conflict:** stop the conflicting service on 8010 or consult the platform
  registry for an available testing port. Do not reuse another tool's reserved port.
- **No legacy lecture notes:** `/explain-concept` requires its separate FAISS
  upload path; successful `/ingest` processing does not satisfy it.

## Project structure

```text
akadverse-concept-explainer/
  main.py                      FastAPI lifespan, router wiring, health
  run.py                       Windows-compatible launcher, port 8010
  auth.py                      Ingestion shared-secret authentication
  models.py                    Ingestion request/response schemas
  config.py                    Provider/storage settings and worker constants
  db.py                        Async Postgres engine and session factory
  routes/
    ingest.py                  POST /ingest
    grounding.py               Retained Google Search Grounding route
    legacy.py                  FAISS routes retained until Phase 2e
  workers/ingestion.py          Claims, recovery, processing, temp-file cleanup
  services/
    embedding.py               Enterprise batch embeddings
    chunking.py                Shared text splitting
    gemini_provider.py         Provider clients and Enterprise discovery
    model_discovery.py         Developer model discovery
    extractors/
      __init__.py              MIME dispatcher
      errors.py                Shared ExtractionError
      pdf.py                   PDF text
      docx.py                  DOCX paragraphs and tables
      pptx.py                  PPTX slides and speaker notes
  migrations/
    001_pgvector_storage.sql
    002_jobs_table.sql
  tests/
    conftest.py                Isolated database and generated Office fixtures
    test_api_key_env.py
    test_auth.py
    test_db.py
    test_embedding.py
    test_extractors.py
    test_gemini_provider.py
    test_ingest.py
    test_refactor.py
    test_worker.py
    test_worker_extractors.py
    fixtures/sample.pdf
  AGENTS.md
  README.md
  requirements.txt
  .env.example
```

## Integration and next steps

The request contract requires a signed file URL and callback URL from Next.js.
Until Phase 2d, inspect the job row and worker logs for results; there is no
job-status HTTP endpoint or outbound callback implementation. Enable PDF,
DOCX, and PPTX in lecturer integrations, and continue rejecting ZIP.

Phase 2d adds `DELETE /documents/{document_id}` and callback delivery using the
existing planned contract in `AGENTS.md`. Neither route/client should be assumed
to exist today. Phase 2e retires the FAISS endpoints, Phase 3 adds pgvector-based
retrieval, and Phase 4 adds Stewart chat history. Rotate previously exposed
shared secrets before deployment and exchange replacements privately.
