# Concept Explainer - Project-Specific Agent Instructions

Read `../AGENTS.md` first for platform-wide rules. This file adds
project-specific context and OVERRIDES the root file where they conflict.

## What this service is

FastAPI microservice, port 8010. RAG study assistant for the AkadVerse
Study Zone. Called exclusively by the Next.js backend (never by browsers).
Python package and project manager used: uv 

Two concerns live in this service:

1. **Ingestion** (current focus): accept lecturer-uploaded course documents,
   chunk and embed them into pgvector. Callback delivery is planned for 2d.
2. **Retrieval** (future): given a course offering and week, retrieve chunks
   and answer student questions. NOT yet implemented. Do not build it unless
   explicitly asked.

The source of truth for the ingestion contract is the FSE's document at
`docs/elearning/systems/ai-ingestion-contract.md` in the Next.js repo. This
AGENTS.md summarizes it. If they ever conflict, the FSE's document wins,
and you should surface the conflict to the developer before proceeding.

## Current phase

**Phase 2: Ingestion rewrite.** The old FAISS-based endpoints are being
retired and replaced incrementally. Current implementation and the remaining
target contracts are labeled separately below.

Status:

- [x] Phase 1: Google API key moved from form field to `GOOGLE_API_KEY` env var
- [x] Phase 2a: pgvector storage (legacy FAISS retained until 2e)
- [x] Phase 2b1: Extract main.py, routes/grounding.py. Old endpoints unchanged
- [x] Phase 2b0: Dual-provider support implemented and validated
- [x] Phase 2b2: auth.py, models.py, migrations/002_jobs_table.sql, routes/ingest.py (202 only)
- [x] Phase 2b3: services/embedding.py, workers/ingestion.py, end-to-end PDF ingestion
- [x] Phase 2c: PDF / DOCX / PPTX extractors
- [ ] Phase 2d: `/documents/{id}` delete + callback client
- [ ] Phase 2e: retire old `/upload-note` and `/explain-concept`
- [ ] Phase 3: Retrieval endpoint (student queries in Study Zone)
- [ ] Phase 4: Chat history storage for Stewart

Update this checklist when a phase or sub-phase is merged. Do not proceed
to the next phase until the current one is complete, tested, and merged.

## Current implementation: Phase 2c

The checkout includes `f2f7334` (`feat(2c): add DOCX and PPTX extractors`).
Phase 2d is next. Do not implement it without an explicit task.

- `/ingest` requires Enterprise mode, a ready database, and shared-secret auth.
  It returns 202 after durable queue admission, before document processing.
- Workers download and extract PDF/DOCX/PPTX, chunk at 1000/200, embed through
  Enterprise, and atomically replace document chunks in pgvector.
- `services.extractors.extract_text(file_path: str, mime_type: str) -> str`
  dispatches to per-format `extract_text(file_path: str) -> str` functions.
  `services/extractors/errors.py` owns `ExtractionError`. Services never import
  workers; `workers/ingestion.py` imports `extract_text` from `services.extractors`.
- `services.chunking.chunk_text(text: str) -> list[str]` owns splitting.
  The worker runs blocking extraction/chunking through `asyncio.to_thread`.
- Extraction errors become failed jobs with their safe message capped at 500
  characters. A failed chunk replacement rolls back and retains previous chunks.
- Callback delivery and the delete endpoint are not implemented. Only
  `callback.would_fire` is logged. `done` currently means chunks were committed.
- Legacy `/upload-note` and `/explain-concept` still use FAISS. New ingestion
  does not feed them. Grounding remains independent and unchanged.
- `run.py` already provides the Windows selector-loop setup on port 8010.
- Latest full-suite verification on 26 October 2026: 86 passed, two existing
  deprecation warnings. Focused results: 21 worker tests and 20 extractor tests.
  These tests do not establish live Cloud model access or callback delivery.

## Legacy retention boundary

Retain the FAISS index constant, vector-store global, lock, imports, and routes
until Phase 2e. Legacy blocking sleeps and print statements are existing debt;
do not expand or refactor them during extractor or callback work without scope
approval. New ingestion uses async I/O, async sleep, and logging.

`POST /explain-with-grounding` is retained. Preserve its prompt, parsing,
citations, provider wiring, and HTTP contract unless explicitly authorized.

## Endpoint contracts: current and planned

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

Background worker steps per job (current Phase 2c):

1. Atomically claim a queued job, mark processing, and increment attempts.
2. Download `file_url` to a UUID-prefixed temp path with a 60-second timeout.
3. Enforce the 20 MB streamed-byte limit, independent of the legacy 50 MB limit.
4. Select the extractor from the declared MIME type, not filename sniffing.
5. Extract usable text or raise a safe `ExtractionError`.
6. Chunk with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`.
7. Embed up to 10 chunks per request, with `await asyncio.sleep(8)` between batches.
8. In one transaction, delete existing chunks for the document, insert new chunks
   with the validated request metadata, and mark the job done.
9. Log completion and `callback.would_fire`; do not send a callback yet.
10. On failure, mark failed with a safe message capped at 500 characters.
11. Delete the temp file in `finally` on either outcome.

The Phase 2d callback delivery requirements are specified separately below.
They do not describe current job completion behavior.

### `DELETE /documents/{document_id}` (Phase 2d target; not implemented)

Header: `Authorization: Bearer <AI_SERVICE_API_KEY>`

Behavior:

1. Validate Bearer token
2. `DELETE FROM concept_explainer.chunks WHERE document_id = $1`
3. Return `204 No Content` on success
4. Return `404 Not Found` if zero rows were deleted. AkadVerse treats both 204 and 404 as success
5. Do NOT enqueue a job. Complete the deletion within the HTTP request using async database I/O

### `POST /explain-with-grounding`

**Unchanged from current implementation.** Keep as-is. It does not touch
the database and is not part of the ingestion flow. Do not modify it while
working on any Phase 2 sub-phase.

### Legacy endpoints still active (delete in Phase 2e)

- `POST /upload-note`: replaced by `POST /ingest`
- `POST /explain-concept`: replaced by a future retrieval endpoint in Phase 3

Do not add features to these. Do not fix bugs in these. Delete them in
Phase 2e once `/ingest` is working and tested end-to-end.

## Outbound callback (Phase 2d target; not implemented)

Planned delivery after job completion or failure. Today the worker only logs `callback.would_fire`; the required inbound `callback_url` is stored but not called. The rules below are the future contract, not implemented retry behavior.

```
POST {callback_url}
Header: Authorization: Bearer <AI_SERVICE_CALLBACK_SECRET>
Body (success): {"document_id": "...", "status": "ingested", "chunk_count": 42}
Body (failure): {"document_id": "...", "status": "failed", "error": "short message"}
```

Rules:

- Retry on `5xx` responses with exponential backoff (1s, 4s, 16s). Max 3 attempts, then give up and mark the job `failed`
- Do NOT retry on `4xx` responses. Log and mark the job `failed` immediately
- Treat `200 {"ok": true, "ignored": "unknown document_id"}` as success. The document was deleted before the callback landed, this is expected
- The `error` field is user-facing: max 500 characters, no stack traces, no internal identifiers beyond the document ID

## Auth model

| Direction | Header | Env var |
|---|---|---|
| AkadVerse to this service | `Authorization: Bearer ...` | `AI_SERVICE_API_KEY` |
| This service to AkadVerse | `Authorization: Bearer ...` | `AI_SERVICE_CALLBACK_SECRET` |

- Both are shared secrets configured privately on each side; only the inbound secret is consumed today. Generate with `openssl rand -hex 32`
- Compare with `secrets.compare_digest()` (or `crypto.timingSafeEqual` in Node). Never use `==` for token comparison
- **No JWT verification inside this service.** The Next.js backend handles user auth and verifies the caller is a student or lecturer before calling this service. This service trusts any request that presents the correct shared secret
- Current inbound behavior: missing database or shared-secret configuration returns 503; unavailable ingestion also returns 503. With ingestion ready, a missing or incorrect Bearer token returns `401 Unauthorized` with `{"detail": "Unauthorized"}`. Developer mode is rejected with 503 by middleware. Do not expose credential values

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

**FAISS remains in the current runtime until Phase 2e.** In 2e, delete `akadverse_concept_faiss_index/`, the
`FAISS_INDEX_PATH` constant, and every import of
`langchain_community.vectorstores.FAISS`. Also delete the `faiss-cpu` line
from `requirements.txt` once the migration is complete.

## Jobs table and background worker

`WORKER_COUNT` is a positive integer, default 1, validated at startup. Changing
it requires a restart. `main.py` starts that many asyncio tasks only when
Enterprise ingestion is ready; Developer mode disables ingestion entirely.
Each worker processes one job at a time and polls an empty queue every two
seconds. Shutdown cancels and awaits all worker tasks, then disposes the engine.

Do not promise a fixed safe concurrency or requests-per-minute quota. The
embedding service sends at most 10 chunks per API request and waits eight
seconds between batches; actual throughput depends on latency, document size,
and project/model quota. Increasing workers increases aggregate load.

Jobs survive process restarts in Postgres. Do not use an in-memory jobs dict,
Redis, Celery, or an external queue. Callback retry rules remain Phase 2d work.

### Job lifecycle (current)

- `queued`: waiting for a worker.
- `processing`: claimed with attempts incremented.
- `done`: chunks replaced and completion committed in the same transaction.
- `failed`: processing error stored in `last_error`, capped at 500 characters.

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
| `AI_SERVICE_API_KEY` | Shared secret for inbound `/ingest` authentication, implemented in Phase 2b2 | Required for ingestion |
| `AI_SERVICE_CALLBACK_SECRET` | Reserved for future outbound callbacks (Phase 2d); callback delivery is not yet implemented | Not required until Phase 2d |
| `WORKER_COUNT` | Ingestion worker tasks, default `1`; restart required after change | Optional |

Do not read or log secret values or inspect the user's `.env`. Do not commit
`.env` or credential files. Rotate the previously shared callback secret
before deployment. The example has non-secret provider/location defaults;
credential fields remain empty.

## Target file structure (build incrementally, per sub-phase)

Phases through 2c are implemented. The tree below includes future files only where explicitly labeled 2d.
Add future files only as each sub-phase lands. Do NOT create the full tree in one shot; add files
per sub-phase so each diff stays reviewable.

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
|   |-- legacy.py                 # Existing FAISS routes, provider wiring only
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
Unknown values fail startup. Both SDK grounding and LangChain legacy calls
use the selected provider. Preserve all business logic and FAISS code.
`/health` adds only `provider`. Discovery is synchronous in settings loading,
not an async lifecycle task. No ingestion, migrations, jobs, or worker work.

**Phase 2b3: services/embedding.py, workers/ingestion.py, end-to-end PDF ingestion.**
Add `services/embedding.py` and `workers/ingestion.py`. Wire end-to-end PDF
ingestion.

**Phase 2c: extractors (complete).** Added `services/extractors/*` and
`services/chunking.py`, wired into the worker, with extractor and worker integration tests.

**Phase 2d: delete + callback.** Add `routes/documents.py` and
`services/callback.py`. Wire the callback into the worker's completion path.

**Phase 2e: retire old endpoints.** Delete `POST /upload-note`,
`POST /explain-concept`, FAISS code, `faiss-cpu` requirement, and the
`akadverse_concept_faiss_index/` folder. Update the README.

## Testing

Windows: `tests/conftest.py` sets the SelectorEventLoop policy for tests.
`run.py` already sets it before Uvicorn starts; use `uv run python run.py`.
Preserve this policy in any future process launcher.

- **Framework:** pytest + `httpx.AsyncClient` via `httpx.ASGITransport`
- **Command:** `uv run pytest tests/ -q --tb=short`
- **Async provider tests:** AnyIO with the asyncio backend; database acceptance uses `asyncio.run()`.
- **Provider mocks:** synchronous SDK discovery, constructors, async generation, and LangChain wrappers. The database acceptance test is real.
- **Current ingestion tests:** mocked downloads/embeddings with real PDF/DOCX/PPTX extraction and isolated Postgres inserts. Embedding service tests mock SDK calls. Callback delivery tests belong to Phase 2d.
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

`main.py` builds Gemini settings once and passes them to both router
factories. `LegacyRouter.legacy_dependencies` exposes typed embedding and
vector-store callables resolved at request time. Both routers use cached
settings model selection. Keep provider arguments separate from business
logic and preserve existing monkeypatch seams when changing wiring.

Smallest-change rule for Pylance and lint fixes: fix warnings with the smallest possible change. Do not introduce new classes, abstractions, or design patterns unless the warning genuinely requires it. If a warning cannot be given the most recommended fix with a type annotation or a one-line accessor, stop and ask before proceeding.

## Definition of done (per sub-phase)

Before declaring a sub-phase complete, run all of these and confirm they
pass. Do NOT mark a phase done in the checklist until every line below is
green:

1. `uv run pytest tests/ -q --tb=short` passes with the expected count of new tests
2. `uv run python -c "from main import app"` imports cleanly
3. No `print()` statements in new code (grep the diff)
4. No `time.sleep()` (grep the diff)
5. No `except Exception: pass` (grep the diff)
6. No em-dashes in docstrings, comments, or strings (grep the diff for the
   unicode em-dash character)
7. No hardcoded secrets (grep for `AIza`, `sk-`, `Bearer `, and the two
   secret env var names appearing literally)
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
3. Run `uv run pytest tests/ -q --tb=short` to confirm the baseline is green
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

Phase 2c is complete. Next is Phase 2d: the document delete endpoint and
callback client, following the planned contracts above after explicit scope
approval. Do not retire legacy routes (2e), implement retrieval (3), chat history
(4), or ZIP support as part of 2d.

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
