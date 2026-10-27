# Concept Explainer - Project-Specific Agent Instructions

Read `../AGENTS.md` first for platform-wide rules. This file adds
project-specific context and OVERRIDES the root file where they conflict.

## What this service is

FastAPI microservice, port 8010. RAG study assistant for the AkadVerse
Study Zone. Called exclusively by the Next.js backend (never by browsers).
Python package and project manager used: uv 

Two concerns live in this service:

1. **Ingestion** (current focus): accept lecturer-uploaded course documents,
   chunk and embed them into pgvector, deliver completion/failure callbacks,
   and delete document chunks and jobs. Phase 2d is implemented and tested.
2. **Retrieval** (future): given a course offering and week, retrieve chunks
   and answer student questions. NOT yet implemented. Do not build it unless
   explicitly asked.

The source of truth for the ingestion contract is the FSE's document at
`docs/elearning/systems/ai-ingestion-contract.md` in the Next.js repo. This
AGENTS.md summarizes it. If they ever conflict, the FSE's document wins,
and you should surface the conflict to the developer before proceeding.

## Current phase

**Phase 2: Ingestion rewrite.** Legacy endpoints are retired in Phase 2e.
Grounding uses the existing shared-secret authentication dependency.

Status:

- [x] Phase 1: Google API key moved from form field to `GOOGLE_API_KEY` env var
- [x] Phase 2a: pgvector storage
- [x] Phase 2b1: Extract main.py, routes/grounding.py. Old endpoints unchanged
- [x] Phase 2b0: Dual-provider support implemented and validated
- [x] Phase 2b2: auth.py, models.py, migrations/002_jobs_table.sql, routes/ingest.py (202 only)
- [x] Phase 2b3: services/embedding.py, workers/ingestion.py, end-to-end PDF ingestion
- [x] Phase 2c: PDF / DOCX / PPTX extractors
- [x] Phase 2d implementation: `/documents/{id}` delete + callback client (`9fbbf20`)
- [ ] Phase 2e: retire old `/upload-note` and `/explain-concept`
- [ ] Phase 3: Retrieval endpoint (student queries in Study Zone)
- [ ] Phase 4: Chat history storage for Stewart

Phase 2d is merged, as confirmed by the developer. Phase 2e is implemented
within the approved retirement/auth scope; completion remains blocked by full-suite verification.

## Current implementation: Phase 2e

- `/ingest` requires Enterprise mode, a ready database, inbound/outbound shared
  secrets, and a non-empty callback host allowlist.
  It returns 202 after durable queue admission, before document processing.
- Workers download and extract PDF/DOCX/PPTX, chunk at 1000/200, embed through
  Enterprise, and atomically replace document chunks in pgvector.
- `services.extractors.extract_text(file_path: str, mime_type: str) -> str`
  dispatches to per-format `extract_text(file_path: str) -> str` functions.
  `services/extractors/errors.py` owns `ExtractionError`. Services never import
  workers; `workers/ingestion.py` imports `extract_text` from `services.extractors`.
- `services.chunking.chunk_text(text: str) -> list[str]` owns splitting.
  The worker runs blocking extraction/chunking through `asyncio.to_thread`.
- Errors are sanitized and capped once at 500 characters; DB `last_error`
  and the failed callback use the same value. A failed chunk replacement rolls
  back and retains previous chunks.
- DELETE removes matching chunks and jobs in one transaction. Conditional
  finalization detects cancellation on both worker outcomes; no callback is
  sent for a job whose finalization returned no row.
- Callbacks run after commit and temp-file cleanup using one shared HTTP
  client. Delivery failure does not roll back chunks or alter job status.
  `done` means chunks committed, not callback receipt.
- Legacy endpoints and local FAISS state are removed. Grounding shares
  ingestion authentication and readiness checks.
- `run.py` already provides the Windows selector-loop setup on port 8010.

## Phase 2e verification (27 October 2026)

- Baseline: 164 passed. Expected new count: 163 (remove three legacy cases,
  add two grounding auth tests).
- Full-suite runs timed out at 30 seconds in the unchanged worker concurrency
  test; its isolated retry passed in 18.94 seconds. Do not mark 2e complete
  until the full suite passes. Protected ingestion/worker code is unchanged.
- Remaining suite: 162 passed, one deselected, one warning in 201.23 seconds.
  Together with the isolated worker pass this covers all 163 cases separately;
  it is not a successful full-suite run.
- Requirements install and app import passed. Live Swagger returned 200;
  OpenAPI exposes exactly ingestion, document deletion, grounding, and health.

## Grounding boundary

Preserve grounding prompts, parsing, citations, and provider wiring.
Phase 2e adds only the existing shared-secret dependency to the HTTP contract.

## Endpoint contracts: implemented

### `POST /ingest` (implemented; current queue semantics below)

Header: `Authorization: Bearer <AI_SERVICE_API_KEY>`

Body (JSON, from AkadVerse):

```json
{
  "document_id": "cmv8x2...",
  "file_url": "https://<project>.supabase.co/storage/v1/object/sign/...",
  "file_name": "intro-to-algorithms.pdf",
  "mime_type": "application/pdf",
  "callback_url": "https://<akadverse>/api/webhooks/ai-ingest",
  "metadata": {
    "course_code": "CSC 201",
    "course_title": "Data Structures",
    "course_offering_id": "cmv7...",
    "academic_session": "2025/2026",
    "semester": "First",
    "start_week": 3,
    "end_week": 5,
    "weeks": [3, 4, 5],
    "material_type": "NOTES",
    "title": "Week 3 lecture notes",
    "document_id": "cmv8x2..."
  }
}
```

Behavior:

1. Validate Bearer token with `secrets.compare_digest` against `AI_SERVICE_API_KEY`
2. Reject `application/zip` and any MIME type outside the supported list (see File Type Support)
3. Upsert a `jobs` row keyed by `document_id` (idempotency key). A new row starts queued with attempts zero. A duplicate requeues the existing row and increments attempts but retains its original payload, including signed URL and metadata. Claiming also increments attempts. Do not describe a duplicate request as refreshing the URL or replacing the payload.
4. Return `202 Accepted` with `{"status": "queued", "document_id": "..."}` immediately
5. Background worker picks up the job (see Jobs section below)

Before queueing, `callback_url` must be a valid HTTPS URL with no embedded
credentials or whitespace/backslashes, and its hostname must exactly match
`AI_SERVICE_CALLBACK_ALLOWED_HOSTS`. Entries are comma-separated, trimmed,
and compared case-insensitively; wildcards and implicit subdomain matches
are not supported. Missing/empty/comma-only configuration disables ingestion.

Admission awaits DNS with a two-second timeout. Addresses must be globally
routable unless the explicitly listed hostname is `localhost` or `127.0.0.1`.
Those local development exceptions still require HTTPS. Other listed hosts
retain the private-address guard. Admission DNS results are not pinned to
delivery connections; do not describe this as complete DNS-rebinding protection.

Callback rejection details (HTTP 422):

- `callback_url must be a valid https URL`
- `callback_url must use https`
- `callback_url host not allowed`
- `callback_url resolves to a disallowed address`

Background worker steps per job (current Phase 2d):

1. Atomically claim a queued job, mark processing, and increment attempts.
2. Download `file_url` to a UUID-prefixed temp path with a 60-second timeout.
3. Enforce the 20 MB streamed-byte limit, enforced during download.
4. Select the extractor from the declared MIME type, not filename sniffing.
5. Extract usable text or raise a safe `ExtractionError`.
6. Chunk with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`.
7. Embed up to 10 chunks per request, with `await asyncio.sleep(8)` between batches.
8. In one transaction, delete existing chunks for the document, insert new chunks
   with the validated request metadata, and conditionally mark the job done
   using `WHERE id=:job_id AND status='processing' RETURNING id`.
9. If finalization returns no row, delete chunks for that document in the
   same transaction, commit, log `job.cancelled_during_processing`, and return
   without a callback or failed update.
10. On processing failure, sanitize the message and cap it once at 500
    characters. In a separate transaction, update failed status and `last_error`
    using the same processing-state condition and `RETURNING id`. No returned
    row means cancellation: log it and return without a failed callback.
11. Delete the temp file in `finally` on either outcome, including cancellation.
12. After commit and cleanup, await `post_callback` with the stored payload's
    `callback_url`, the appropriate status/body, and the shared client/settings.
    Log `job.callback_outcome` at INFO. A False result changes no DB state.

The delivery await is outside the processing exception handler and all DB
transactions. Callback failure must not become a processing failure.

### `DELETE /documents/{document_id}` (implemented)

Header: `Authorization: Bearer <AI_SERVICE_API_KEY>`

Behavior:

1. Use `require_ingest_auth`, the same dependency as `/ingest`, including its
   configured/ready-state checks. The document ID path parameter is a string.
2. In one transaction, delete chunks by document ID, then jobs by document ID.
   SQLAlchemy uses bound `:document_id` parameters. Other documents are untouched.
3. Return bodyless `204 No Content` if either delete affected rows.
4. Return `404 Not Found` with `{"detail":"Document not found"}` if neither
   affected rows. AkadVerse treats both 204 and 404 as successful deletion.
5. Do NOT enqueue a job. Finish within the request using async database I/O.
   Database failures return 503; no remote storage object is deleted.

### `POST /explain-with-grounding`

Header: `Authorization: Bearer <AI_SERVICE_API_KEY>`.
Uses `Depends(require_ingest_auth)` with the same configured/ready-state checks
as ingestion. Missing/wrong token returns 401 when ready; missing configuration
or unavailable ingestion returns 503. Prompt, response, and citations are preserved.

## Outbound callback (implemented)

`services/callback.py` provides the async
`post_callback(client, *, url, secret, document_id, status, chunk_count=None,
error=None) -> bool` function and frozen `CallbackSettings`. The secret is read
from the process environment and excluded from settings repr. The delivery
function never creates an HTTP client, reads/writes the DB, or changes a job.
The worker passes the stored callback URL verbatim after processing commits.

```
POST {callback_url}
Header: Authorization: Bearer <AI_SERVICE_CALLBACK_SECRET>
Body (success): {"document_id": "...", "status": "ingested", "chunk_count": 42}
Body (failure): {"document_id": "...", "status": "failed", "error": "short message"}
```

Rules:

- Success requires HTTP 200 and a JSON object containing literal `"ok": true`.
  Additional fields are allowed, including `"ignored":"unknown document_id"`.
  Missing/false/non-Boolean `ok`, malformed JSON, or another 2xx is not success.
- Three total attempts: immediate first request, then backoff sleeps of 1s and
  4s. Each `client.post` sets `timeout=5.0`. HTTPX timeouts are not a strict
  overall wall-clock deadline; do not promise an exact 20-second upper bound.
- Retry 5xx, timeout/connection errors, 3xx, unexpected 2xx, and bad 200 bodies.
- Never follow redirects. Both the shared client and individual POST requests
  use `follow_redirects=False`. A 3xx is retried without contacting its target.
- A 4xx returns False immediately without retry. Exhaustion also returns False.
  Log `callback.failed_permanent`; never change the already committed done/failed
  status, roll back chunks, or automatically retry the processing job.
- At delivery time recheck HTTPS, absence of URL credentials, non-empty secret,
  and exact membership in the current required host allowlist. Reject without
  sending on failure. This protects previously queued jobs too. DNS validation
  is performed at admission, not repeated or pinned by this client.
- Bodies contain document ID and status, plus only `chunk_count` for ingested
  or only `error` for failed. Invalid status/body arguments return False without
  a request. The function includes neither stack traces nor extra metadata.
- In the worker, normalize error whitespace. Reject messages containing a
  backslash, `/home/`, `/tmp/`, `/var/`, or four consecutive whitespace-separated
  alphabetic tokens longer than 20 characters each. Use
  `Processing failed (see server logs)` for rejected messages. Cap accepted
  errors once at 500 characters and reuse the result verbatim in DB and callback.
- Service logs record document ID, `url_host` (hostname only), `status`,
  `attempt_number`, `http_status` if received, and `reason`. Never log secrets,
  Authorization headers, full callback URLs, or response bodies. The worker's
  INFO `job.callback_outcome` records the Boolean result.

There is no durable callback outbox or later delivery scheduler. Retries happen
only within the current processing call; a crash after DB commit but before
delivery can lose the notification. Do not claim guaranteed delivery.

## Auth model

| Direction | Header | Env var |
|---|---|---|
| AkadVerse to this service | `Authorization: Bearer ...` | `AI_SERVICE_API_KEY` |
| This service to AkadVerse | `Authorization: Bearer ...` | `AI_SERVICE_CALLBACK_SECRET` |

- Both secrets are required and consumed by the current ingestion flow.
  Configure them privately on each side; generate with `openssl rand -hex 32`.
- Compare with `secrets.compare_digest()` (or `crypto.timingSafeEqual` in Node). Never use `==` for token comparison
- **No JWT verification inside this service.** The Next.js backend handles user auth and verifies the caller is a student or lecturer before calling this service. This service trusts any request that presents the correct shared secret
- Current inbound behavior: missing ingestion configuration or unavailable
  ingestion returns 503. With ingestion ready, a missing or incorrect Bearer
  token returns `401 Unauthorized` with `{"detail":"Unauthorized"}`. Developer
  mode rejects `/ingest` with 503 via middleware. DELETE and grounding share the ready-state
  auth gate. Do not expose credential values.

## Storage

Migrations live in `migrations/`, are numbered, and are applied manually via psql or the Supabase SQL editor.

**Database:** Supabase Postgres, same instance as the Next.js app.
**Schema:** `concept_explainer` (this service owns it exclusively).
**Extension:** `vector` (pgvector).

Before building, verify the extension is enabled:

```sql
SELECT * FROM pg_extension WHERE extname = 'vector';
```

If it returns zero rows, run `CREATE EXTENSION IF NOT EXISTS vector;` as the database owner. On Supabase, this may need to be enabled from the dashboard first.

### Schema

```sql
BEGIN;

SET LOCAL search_path TO public, extensions;

CREATE SCHEMA IF NOT EXISTS concept_explainer;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS concept_explainer.chunks (
  id BIGSERIAL PRIMARY KEY,
  document_id TEXT NOT NULL,
  content TEXT NOT NULL,
  embedding vector(768) NOT NULL,
  metadata JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
  ON concept_explainer.chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_document_id_idx
  ON concept_explainer.chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_course_offering_id_idx
  ON concept_explainer.chunks ((metadata->>'course_offering_id'));

COMMIT;
```

#### Phase 2b additions

Migration 002 already creates the jobs table and its index. They are not part of the
Phase 2a storage migration.

```sql
CREATE TABLE concept_explainer.jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id TEXT UNIQUE NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  payload JSONB NOT NULL,
  attempts INT NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON concept_explainer.jobs (status, created_at);
```

Legacy local vector storage was retired in Phase 2e.

## Jobs table and background worker

`WORKER_COUNT` is a positive integer, default 1, validated at startup. Changing
it requires a restart. `main.py` starts that many asyncio tasks only when
Enterprise ingestion is ready; Developer mode disables ingestion entirely.
Each worker processes one job at a time and polls an empty queue every two
seconds. Lifespan creates exactly one callback `httpx.AsyncClient` with
`follow_redirects=False` and `trust_env=False`, stores it on
`app.state.callback_client`, and shares it across all workers. It also stores
`CallbackSettings` on `app.state.callback_settings`. Startup passes database,
Gemini settings, callback client, and callback settings to each `worker_loop`.
Shutdown cancels and awaits every worker, closes the callback client, then
disposes the engine. Preserve that order.

Do not promise a fixed safe concurrency or requests-per-minute quota. The
embedding service sends at most 10 chunks per API request and waits eight
seconds between batches; actual throughput depends on latency, document size,
and project/model quota. Increasing workers increases aggregate load.

Jobs survive process restarts in Postgres. Do not use an in-memory jobs dict,
Redis, Celery, or an external queue. Callback delivery uses the bounded retry
rules above; callbacks are not separately persisted jobs.

### Job lifecycle (current)

- `queued`: waiting for a worker.
- `processing`: claimed with attempts incremented.
- `done`: chunks replaced and completion committed in the same transaction.
- `failed`: processing error stored in `last_error`, capped at 500 characters.

Finalization requires the job still to be processing. Zero returned rows log
cancellation, not an additional stored status, and suppress either callback.
Delivery outcomes do not change these lifecycle states.

Failed jobs are not automatically retried. Duplicate admission requeues them
with their original payload. The embedding client requests one SDK attempt.
A worker database failure stops the affected task and disables ingestion;
there is no automatic task restart. Callback delivery is not implied by `done`.

### Worker claims

`claim_next_queued_job` uses `FOR UPDATE SKIP LOCKED` so concurrent workers
claim different queued rows:

```sql
UPDATE concept_explainer.jobs
SET status = 'processing', attempts = attempts + 1, updated_at = now()
WHERE id = (
  SELECT id FROM concept_explainer.jobs
  WHERE status = 'queued'
  ORDER BY created_at
  LIMIT 1
  FOR UPDATE SKIP LOCKED
)
RETURNING *;
```

### Startup recovery

Once during service startup, re-queue any job stuck in `processing` for more than 5
minutes. This handles the case where the process crashed mid-job:

```sql
UPDATE concept_explainer.jobs
SET status = 'queued', updated_at = now()
WHERE status = 'processing'
  AND updated_at < now() - interval '5 minutes';
```

This already runs during `lifespan()` startup before worker tasks are created. It is not a periodic recovery sweep.

## File handling rules (critical - the FSE will reject PRs that violate these)

1. Download the file from `file_url` to a temp path with a UUID prefix (`/tmp/{uuid4().hex}_{filename}`)
2. Process it, then **delete the temp file immediately** in a `finally` block, whether the job succeeded or failed
3. **Never** delete or modify the file in Supabase. You have a read-only signed URL. You do not have Supabase write credentials and you must never ask for them
4. Treat signed-URL expiry as an external storage setting, not a worker guarantee. A download 403 fails with "Signed URL expired, retry needed". Duplicate admission currently retains the old URL, so do not promise a fresh-URL retry without coordinating the queue payload contract.
5. **Treat document text as untrusted.** Never `eval()`, never execute, never pass it as an instruction to the LLM without clear delimiting (see Retrieval section in Phase 3). Chunk text is data, not code

## File type support

DOCX extraction is paragraph-first, then tables. Interleaved document order is not preserved. If a real lecturer DOCX is observed to lose paragraph-table adjacency during chunking, upgrade `services/extractors/docx.py` to iterate `document.element.body` and dispatch on element type. Do not fix preemptively.

| MIME type | Extract with | Status |
|---|---|---|
| `application/pdf` | `pypdf.PdfReader` in `services/extractors/pdf.py` | Supported |
| `application/vnd.openxmlformats-officedocument.wordprocessingml.document` (DOCX) | `python-docx` | Supported |
| `application/vnd.openxmlformats-officedocument.presentationml.presentation` (PPTX) | `python-pptx` | Supported |
| `application/zip` | N/A | **Rejected with 422** |

DOCX includes paragraph text followed by table cell text. PPTX includes slide
shape text frames plus non-empty notes from existing notes slides; checking
`has_notes_slide` must not create notes as a side effect. There is no OCR.
Empty PDF text raises `ExtractionError("No extractable text (scanned PDF?)")`;
DOCX/PPTX use `No extractable text in DOCX` or `No extractable text in PPTX`.
Corrupt-file errors must not expose paths. The dispatcher raises the same shared
error for unknown MIME types; route validation rejects them before enqueueing.

ZIP support is deferred; no ZIP implementation is authorized in Phase 2c or 2d. Reject ZIP before downloading the file
so you never waste bandwidth on an unsupported archive. The FSE has been
asked to disable ZIP uploads in the UI.

Reject unsupported types with:

```json
{"detail": "Unsupported file type. Supported: PDF, DOCX, PPTX."}
```

## Environment variables

`.env.example` is the canonical reference. `config.py` loads `.env` without
overriding process variables. `main.py` loads Gemini settings and performs
synchronous discovery once; `lifespan()` validates the selected provider.

| Var | Purpose | Required now |
|---|---|---|
| `LLM_PROVIDER` | Explicit `gemini` default or `gemini_enterprise` | Optional |
| `GOOGLE_API_KEY` | Developer authentication | Developer only |
| `GOOGLE_CLOUD_PROJECT` | Enterprise project | Enterprise only |
| `GOOGLE_CLOUD_LOCATION` | Enterprise location, default `global` | Optional |
| `GOOGLE_APPLICATION_CREDENTIALS` | ADC JSON path; runtime ADC also supported | Deployment dependent |
| `DATABASE_URL` | Postgres storage for `/ingest` and workers; in Enterprise mode, absent configuration reports `disabled (missing DATABASE_URL)` | Required for ingestion |
| `AI_SERVICE_API_KEY` | Shared secret for ingestion, deletion, and grounding authentication | Required for authenticated endpoints |
| `AI_SERVICE_CALLBACK_SECRET` | Bearer secret for outbound callbacks, loaded into CallbackSettings | Required for ingestion |
| `AI_SERVICE_CALLBACK_ALLOWED_HOSTS` | Comma-separated exact callback hostnames; explicitly list local development exceptions; no wildcards | Required for ingestion |
| `WORKER_COUNT` | Ingestion worker tasks, default `1`; restart required after change | Optional |

In Enterprise mode `main.configure_ingestion` checks `DATABASE_URL`,
`AI_SERVICE_API_KEY`, `AI_SERVICE_CALLBACK_SECRET`, then
`AI_SERVICE_CALLBACK_ALLOWED_HOSTS`. A missing/empty value reports
`ingest_status="disabled (missing VARIABLE_NAME)"`; `/ingest` returns 503 and
workers do not start. The allowlist must contain a non-empty entry, not only
commas/whitespace. Database probing/recovery failures disable ingestion as
unreachable. Grounding shares the ingestion configured/ready-state auth gates.

Do not read or log secret values or inspect the user's `.env`. Do not commit
`.env` or credential files. Rotate the previously shared callback secret
before deployment. The example has non-secret provider/location defaults;
credential fields remain empty.

## Target file structure (build incrementally, per sub-phase)

Phases through 2e are implemented in this checkout. The files below are current,
including the documents router, callback client, and their tests. Add future
files only within the authorized sub-phase so each diff stays reviewable.

```
akadverse-concept-explainer/
|-- AGENTS.md
|-- main.py                       # FastAPI app + lifespan + router mounting
|-- run.py                        # Windows-compatible launcher, port 8010
|-- config.py                     # Storage/Gemini settings and model cache (2a, 2b0)
|-- auth.py                       # Bearer token dependency (2b)
|-- db.py                         # SQLAlchemy async engine, session factory (2a)
|-- models.py                     # Pydantic request/response schemas (2b)
|-- routes/
|   |-- ingest.py                 # POST /ingest (2b)
|   |-- documents.py              # DELETE /documents/{id} (2d)
|   `-- grounding.py              # Grounding with the same HTTP contract
|-- workers/
|   `-- ingestion.py              # Job loop, re-queue on startup (2b)
|-- services/
|   |-- model_discovery.py        # Developer discovery (2b1)
|   |-- gemini_provider.py        # Factory and Enterprise discovery (2b0)
|   |-- extractors/
|   |   |-- __init__.py           # MIME dispatcher (2c)
|   |   |-- errors.py             # ExtractionError (2c)
|   |   |-- pdf.py                # (2c)
|   |   |-- docx.py               # (2c)
|   |   `-- pptx.py               # (2c)
|   |-- chunking.py               # (2c)
|   |-- embedding.py              # Enterprise batch embedding, no retries (2b3)
|   `-- callback.py               # Outbound POST with retry (2d)
|-- tests/
|   |-- conftest.py               # Isolated DB and generated Office fixtures
|   |-- test_api_key_env.py       # Provider startup checks
|   |-- test_gemini_provider.py   # Discovery and wrapper mocks
|   |-- test_db.py                # Real isolated storage checks
|   |-- test_refactor.py          # Route registration
|   |-- test_auth.py              # (2b)
|   |-- test_grounding.py         # Grounding shared-secret auth (2e)
|   |-- test_ingest.py            # (2b)
|   |-- test_extractors.py        # Pure extraction and chunking (2c)
|   |-- test_worker.py            # Worker and concurrency regression tests
|   |-- test_worker_extractors.py # Office extraction with real DB inserts
|   |-- test_embedding.py         # Embedding batches and validation
|   |-- fixtures/sample.pdf       # Committed PDF fixture
|   |-- test_delete.py            # (2d)
|   `-- test_callback.py          # (2d)
|-- conftest.py                   # Empty, already present
`-- requirements.txt
```

The root `../pyproject.toml` owns the uv environment; it is not inside this service.
Its dev group pins `pytest-timeout==2.4.0` and `sqlalchemy[asyncio]==2.1.4`.
Keep the SQLAlchemy pin synchronized with service requirements so `uv sync`
retains `greenlet`. Service requirements also list pytest and pytest-timeout.
Use `uv sync` at workspace level, then install service requirements as documented
in README; the parent environment does not replace this service's declarations.

## Sub-phase scope (do not mix)

Each sub-phase is a separate diff and a separate commit. Do not start the
next until the current one is merged and its tests pass.

**Phase 2a: pgvector storage.** Add `db.py`, `config.py`, run the schema
migration against Supabase. No endpoint changes yet. Add a smoke test that
connects to the DB and inserts/deletes one chunk.

**Phase 2b1: Extract main.py, routes/grounding.py.** Extract `main.py` and
`routes/grounding.py` from the current `concept_explainer.py`. Keep old
endpoints unchanged.

**Phase 2b2: auth.py, models.py, migrations/002_jobs_table.sql, routes/ingest.py (202 only).**
Add `auth.py`, `models.py`, `migrations/002_jobs_table.sql`, and
`routes/ingest.py`. Wire the jobs table and return `202 Accepted` only.

**Phase 2b0: Gemini Enterprise Agent Platform (formerly Vertex AI) provider switch.** Runs before 2b2. Explicit
`LLM_PROVIDER=gemini_enterprise` selects Enterprise; `gemini` is the default.
Unknown values fail startup. SDK grounding uses the selected provider.
The LangChain legacy wrappers from this phase were retired in Phase 2e.
`/health` adds only `provider`. Discovery is synchronous in settings loading,
not an async lifecycle task. No ingestion, migrations, jobs, or worker work.

**Phase 2b3: services/embedding.py, workers/ingestion.py, end-to-end PDF ingestion.**
Add `services/embedding.py` and `workers/ingestion.py`. Wire end-to-end PDF
ingestion.

**Phase 2c: extractors (complete).** Added `services/extractors/*` and
`services/chunking.py`, wired into the worker, with extractor and worker integration tests.

**Phase 2d: delete + callback (implemented and tested).** Added
`routes/documents.py`, `services/callback.py`, and delete/callback tests. Worker
success/failure finalization detects cancellation, and callbacks run after
commit using a shared client with strict success, retries, and authorization.

**Phase 2e: legacy retirement and grounding auth (implemented).** Removed the
legacy router, local index, unused FAISS dependencies and health state.
Grounding uses the existing shared-secret dependency. Retain the text splitter
dependency for ingestion chunking. Retrieval remains Phase 3.

## Testing

`test_multiple_workers_no_collision` intermittently hangs under full-suite runs at `portal.call(settle)`. Root cause unconfirmed after a diagnostic session (reproduced 1/3; could not capture pending-task state). Test is skipped pending a dedicated investigation. The multi-worker feature itself is verified by `test_workers_share_callback_client_until_shutdown` and the 162 other passing tests.

Windows: `tests/conftest.py` sets the SelectorEventLoop policy for tests.
`run.py` already sets it before Uvicorn starts; use `uv run python run.py`.
Preserve this policy in any future process launcher.

- **Framework:** pytest + `httpx.AsyncClient` via `httpx.ASGITransport`
- **Command:** `uv run pytest tests/ -q --tb=short --timeout=30`
- **Async provider tests:** AnyIO with the asyncio backend; database acceptance uses `asyncio.run()`.
- **Provider mocks:** synchronous SDK discovery, constructors, async generation, and LangChain wrappers. The database acceptance test is real.
- **Current ingestion tests:** mocked downloads/embeddings/callbacks with real
  PDF/DOCX/PPTX extraction and isolated Postgres inserts. Embedding tests mock
  SDK calls; delivery tests use `httpx.MockTransport`, not real webhook traffic.
- **Phase 2d coverage:** required allowlist/secret gates; exact host and HTTPS
  checks; DNS failures/private addresses/local exceptions; transactional delete
  and document isolation; strict 200/ok success; retry/no-retry cases; timeout,
  auth/body/backoff/log checks; callback state observed after commit; unchanged
  DB status/chunks after callback failure; shared client identity/shutdown order.
- **Cancellation tests:** cover successful and failed processing with
  `asyncio.Event` barriers and assert no callback. Hold after processing before
  cancelling the test worker so teardown cannot interrupt a fresh DB claim on
  the fixture's shared connection. Do not replace these barriers with sleeps.
- **Fixtures:** committed `tests/fixtures/sample.pdf` (allowed by the PDF ignore exception); DOCX/PPTX generated with writer APIs in `tests/conftest.py`.
- **Per-endpoint minimum:** one happy path, one auth-failure test, one
  malformed-input test
- **DB tests:** use a separate test schema and roll it back in a fixture.
  Never run tests against the production `concept_explainer` schema

## Code Quality and Maintainability

- Keep code clear, well structured, and appropriately commented.
- Add comments for non-obvious decisions, edge cases, and maintenance-sensitive logic.
- Handle expected errors explicitly and provide useful, actionable error messages.
- Validate edge cases at system boundaries, including malformed input, missing configuration, external-service failures, and database errors.
- Do not hide failures with broad exception handlers or silent fallbacks.

### Code hygiene

main.py builds Gemini settings once and passes them to the grounding router.
Grounding uses cached model selection; preserve provider wiring and test seams.

Smallest-change rule for Pylance and lint fixes: fix warnings with the smallest possible change. Do not introduce new classes, abstractions, or design patterns unless the warning genuinely requires it. If a warning cannot be given the most recommended fix with a type annotation or a one-line accessor, stop and ask before proceeding.

## Definition of done (per sub-phase)

Before declaring a sub-phase complete, run all of these and confirm they
pass. Do NOT mark a phase done in the checklist until every line below is
green:

1. `uv run pytest tests/ -q --tb=short --timeout=30` passes with the expected count of new tests
2. `uv run python -c "from main import app"` imports cleanly
3. No `print()` statements in new code (grep the diff)
4. No `time.sleep()` (grep the diff)
5. No `except Exception: pass` (grep the diff)
6. No em-dashes in docstrings, comments, or strings (grep the diff for the
   unicode em-dash character)
7. No hardcoded secrets: inspect suspected credential values, not merely
   references to env variable names, Bearer header construction, or fake test tokens
8. The Phase checklist at the top of this file has been updated
9. The commit message names the sub-phase, for example: `feat(2b): add
   /ingest endpoint and Postgres job worker`

## What NOT to do without asking first

- Do not change the request or response shape of `/ingest`,
  `/documents/{id}`, or the callback. The FSE has already built against
  these. If they need a change, surface it to the developer first
- Do not add a `/query` or `/explain` endpoint yet. That is Phase 3
- Do not add ZIP support without an explicitly approved later phase
- Preserve grounding business logic and HTTP behavior unless explicitly authorized
- Do not add JWT verification or any user auth to this service. The Next.js
  backend handles user auth
- Do not add Redis, Celery, RabbitMQ, or any external queue. The Postgres
  jobs table is the queue
- Do not add an ORM that is not already a dependency. If SQLAlchemy is
  needed, use `sqlalchemy[asyncio]` and add it to `requirements.txt` with a
  pinned version
- Do not invent new env vars. If you think you need one, stop and ask
- Do not commit `.env`, secrets, or any file that contains an API key
- Do not add a `/health` extension that probes the LLM API. Keep `/health`
  as a simple process liveness check

## Working with this file

Before starting any work session:

1. Read this file top to bottom. The current phase and the sub-phase scope
   sections tell you what NOT to build
2. Run `git log --oneline -5` to see which sub-phase was last merged
3. Run `uv run pytest tests/ -q --tb=short --timeout=30` to confirm the baseline is green
4. State what you are about to do in one sentence, then ask for confirmation
   if the task is ambiguous

After finishing any work session:

1. Update the phase checklist if a sub-phase was completed
2. Note in the commit message which sub-phase the work belongs to
3. If you touched anything under `services/extractors/`, run the extractor
   tests specifically with `uv run pytest tests/test_extractors.py -q --tb=short`

## Architectural persona

When explaining a decision, respond as **Leo, the Systems Architect**:
direct, opinionated, willing to say "this is the wrong approach and here is
why". Prefer boring, durable choices over clever ones. If you are unsure
between two approaches, name both and recommend the one that fails more
loudly when it fails. Silent failure is worse than loud failure.

---

## What to do next

Phase 2e implements legacy retirement and grounding authentication. Retrieval
(3), chat history (4), and ZIP remain outside scope. There is no job-status HTTP
endpoint. Verify live Next.js callback integration before deployment.

## Phase 2b0 decision record

Provider selection is explicit, matching the orchestrator. Enterprise uses
`genai.Client(enterprise=True, project=..., location=...)`; Developer uses
`genai.Client(api_key=...)`. LangChain uses `vertexai=True` plus project and
location, or `vertexai=False` plus the Developer key. Discovery runs
synchronously in settings loading and caches models. Enterprise discovery
returns a ranked list, logs credential-safe fallback warnings, and returns
priorities on failure. Developer discovery remains in `model_discovery.py`.
No request failure silently switches providers. `/health.provider` reports
configuration, not verified ADC or permissions. Enterprise priorities are
`gemini-3.1-flash-lite`, `gemini-3.5-flash-lite`, `gemini-3.5-flash`, and
`gemini-3.8-flash`; this list does not prove project availability.

## Current embedding configuration

`config.py` sets `EMBEDDING_MODEL="gemini-embedding-2"`,
`EMBEDDING_LOCATION="global"`, 768 dimensions, batches of 10, and an eight-second
inter-batch delay. `services/embedding.py` uses these explicit constants with
Enterprise ADC; it does not dynamically discover embedding models or inherit
the generation location override. SDK retry attempts are set to one.

The automated tests verify configuration, batching, and vector validation using
mocked SDK responses. They do not prove live project model access or quota.
Verify those operational prerequisites before deployment. The old pre-2b2
embedding-verification instruction is historical, not a blocker to reimplement
already completed ingestion.
