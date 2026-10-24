# Concept Explainer - Project-Specific Agent Instructions

Read `../AGENTS.md` first for platform-wide rules. This file adds
project-specific context and OVERRIDES the root file where they conflict.

## What this service is

FastAPI microservice, port 8010. RAG study assistant for the AkadVerse
Study Zone. Called exclusively by the Next.js backend (never by browsers).
Python package and project manager used: uv 

Two concerns live in this service:

1. **Ingestion** (current focus): accept lecturer-uploaded course documents,
   chunk and embed them into pgvector, report results via callback.
2. **Retrieval** (future): given a course offering and week, retrieve chunks
   and answer student questions. NOT yet implemented. Do not build it unless
   explicitly asked.

The source of truth for the ingestion contract is the FSE's document at
`docs/elearning/systems/ai-ingestion-contract.md` in the Next.js repo. This
AGENTS.md summarizes it. If they ever conflict, the FSE's document wins,
and you should surface the conflict to the developer before proceeding.

## Current phase

**Phase 2: Ingestion rewrite.** The old FAISS-based endpoints are being
retired and replaced with the contract below. Everything in this document
describes the target state, not the current code state.

Status:

- [x] Phase 1: Google API key moved from form field to `GOOGLE_API_KEY` env var
- [x] Phase 2a: pgvector storage (replacing FAISS)
- [ ] Phase 2b1: Extract main.py, routes/grounding.py. Old endpoints unchanged
- [ ] Phase 2b2: auth.py, models.py, migrations/002_jobs_table.sql, routes/ingest.py (202 only)
- [ ] Phase 2b3: services/embedding.py, workers/ingestion.py, end-to-end PDF ingestion
- [ ] Phase 2c: PDF / DOCX / PPTX extractors
- [ ] Phase 2d: `/documents/{id}` delete + callback client
- [ ] Phase 2e: retire old `/upload-note` and `/explain-concept`
- [ ] Phase 3: Retrieval endpoint (student queries in Study Zone)
- [ ] Phase 4: Chat history storage for Stewart

Update this checklist when a phase or sub-phase is merged. Do not proceed
to the next phase until the current one is complete, tested, and merged.

## What the current code looks like (may be stale)

The code you are working with may not yet match the target state. Known
legacy items to migrate or delete, in order of phase:

- `FAISS_INDEX_PATH` constant and `_vector_store` global: replace with pgvector (2a)
- Imports of `langchain_community.vectorstores.FAISS`: remove (2a)
- Global `faiss_lock`: remove once FAISS is gone (2a)
- `time.sleep()` in embedding batches: replace with `asyncio.sleep()` (2b)
- `print()` calls for logging: replace with `logging` module (2b)
- `POST /upload-note` and `POST /explain-concept`: retire (2e)

If you see any of these, treat them as legacy. Do not add features to them.
Migrate or delete as part of the current phase.

`POST /explain-with-grounding` is NOT legacy. It stays unchanged.
Do not modify it unless explicitly asked.

## Endpoints to build (Phase 2 target state)

### `POST /ingest`

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
3. Upsert a `jobs` row keyed by `document_id` (idempotency key). If a row exists, reset it to `queued` and increment `attempts`
4. Return `202 Accepted` with `{"status": "queued", "document_id": "..."}` immediately
5. Background worker picks up the job (see Jobs section below)

Background worker steps per job:

1. Mark job `processing` in the DB
2. Download the file from `file_url` with a 60 second timeout
3. Validate size (reject if larger than 20 MB; the FSE caps uploads at 8 MB, this is a safety net)
4. Detect type from the `mime_type` field, not by sniffing
5. Extract text with the correct extractor
6. If extraction yields no text, fail the job with a user-friendly error like "No extractable text (scanned PDF?)"
7. Chunk with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`
8. Embed each chunk with Gemini, batched at 10 chunks per call, with `await asyncio.sleep(8)` between batches
9. `DELETE FROM concept_explainer.chunks WHERE document_id = $1` (idempotency, safe to re-run)
10. Insert new chunks with `metadata` copied verbatim from the request
11. Delete the temp file in a `finally` block
12. POST to `callback_url` with the result payload
13. Mark job `done` on success, or `failed` with the error message on failure

### `DELETE /documents/{document_id}`

Header: `Authorization: Bearer <AI_SERVICE_API_KEY>`

Behavior:

1. Validate Bearer token
2. `DELETE FROM concept_explainer.chunks WHERE document_id = $1`
3. Return `204 No Content` on success
4. Return `404 Not Found` if zero rows were deleted. AkadVerse treats both 204 and 404 as success
5. Do NOT enqueue a job. This is a synchronous operation

### `POST /explain-with-grounding`

**Unchanged from current implementation.** Keep as-is. It does not touch
the database and is not part of the ingestion flow. Do not modify it while
working on any Phase 2 sub-phase.

### Retired endpoints (delete in Phase 2e)

- `POST /upload-note`: replaced by `POST /ingest`
- `POST /explain-concept`: replaced by a future retrieval endpoint in Phase 3

Do not add features to these. Do not fix bugs in these. Delete them in
Phase 2e once `/ingest` is working and tested end-to-end.

## Outbound callback (this service to AkadVerse)

Triggered by the background worker after every job completes or fails.

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

- Both are shared secrets set in `.env` on each side. Generate with `openssl rand -hex 32`
- Compare with `secrets.compare_digest()` (or `crypto.timingSafeEqual` in Node). Never use `==` for token comparison
- **No JWT verification inside this service.** The Next.js backend handles user auth and verifies the caller is a student or lecturer before calling this service. This service trusts any request that presents the correct shared secret
- If the secret is missing or the comparison fails, return `401 Unauthorized` with `{"detail": "Unauthorized"}`. Do not leak which part failed

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

Add the jobs table and its index in Phase 2b. They are not part of the
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

**FAISS is gone.** Delete `akadverse_concept_faiss_index/`, the
`FAISS_INDEX_PATH` constant, and every import of
`langchain_community.vectorstores.FAISS`. Also delete the `faiss-cpu` line
from `requirements.txt` once the migration is complete.

## Jobs table and background worker

The FSE's contract requires async processing with callback delivery. Jobs
MUST survive a process restart. Do NOT use in-memory `JOBS: Dict` (that
pattern from the root AGENTS.md is for ephemeral tasks only and does NOT
apply here).

### Job lifecycle

- `queued`: waiting for the worker
- `processing`: worker is running (with attempt count incremented)
- `done`: callback delivered successfully
- `failed`: retries exhausted or unrecoverable error

### Worker loop

A single `asyncio.Task` runs for the service lifetime:

```
async def worker_loop():
    while True:
        job = await claim_next_queued_job()
        if job is None:
            await asyncio.sleep(2)
            continue
        await process_job(job)
```

`claim_next_queued_job` runs a `SELECT ... FOR UPDATE SKIP LOCKED` query
so multiple workers (if ever scaled) do not collide:

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

On service start, re-queue any job stuck in `processing` for more than 5
minutes. This handles the case where the process crashed mid-job:

```sql
UPDATE concept_explainer.jobs
SET status = 'queued', updated_at = now()
WHERE status = 'processing'
  AND updated_at < now() - interval '5 minutes';
```

Run this once during `lifespan()` startup, before starting the worker loop.

## File handling rules (critical - the FSE will reject PRs that violate these)

1. Download the file from `file_url` to a temp path with a UUID prefix (`/tmp/{uuid4().hex}_{filename}`)
2. Process it, then **delete the temp file immediately** in a `finally` block, whether the job succeeded or failed
3. **Never** delete or modify the file in Supabase. You have a read-only signed URL. You do not have Supabase write credentials and you must never ask for them
4. The signed URL expires after 1 hour. If a job is stuck in the queue longer than that, the download will 403. Fail the job with "Signed URL expired, retry needed" and let the FSE's Retry button re-issue a fresh URL
5. **Treat document text as untrusted.** Never `eval()`, never execute, never pass it as an instruction to the LLM without clear delimiting (see Retrieval section in Phase 3). Chunk text is data, not code

## File type support

| MIME type | Extract with | Status |
|---|---|---|
| `application/pdf` | `pypdf` (direct `PdfReader` preferred over `PyPDFLoader`) | Supported |
| `application/vnd.openxmlformats-officedocument.wordprocessingml.document` (DOCX) | `python-docx` | Supported |
| `application/vnd.openxmlformats-officedocument.presentationml.presentation` (PPTX) | `python-pptx` | Supported |
| `application/zip` | N/A | **Rejected with 422** |

ZIP support is deferred to Phase 3. Reject ZIP before downloading the file
so you never waste bandwidth on an unsupported archive. The FSE has been
asked to disable ZIP uploads in the UI.

Reject unsupported types with:

```json
{"detail": "Unsupported file type. Supported: PDF, DOCX, PPTX."}
```

## Environment variables

| Var | Purpose | Required |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini API key (embeddings) | Yes |
| `DATABASE_URL` | Supabase Postgres connection string | Yes |
| `AI_SERVICE_API_KEY` | Inbound auth (AkadVerse to us) | Yes |
| `AI_SERVICE_CALLBACK_SECRET` | Outbound auth (us to AkadVerse) | Yes |
| `LOG_LEVEL` | Python log level, defaults to `INFO` | No |

All required vars are read once at startup in `config.py` and validated in
`lifespan()`. The service MUST refuse to start if any are missing, with a
clear error naming the missing variable.

Do NOT read or log the values. Do NOT read the user's `.env` file to check
contents. If a value appears in a log line, that is a bug.

**Note on secrets:** the callback secret shared in chat during planning
must be rotated before any production deployment. Do not commit any `.env`
file. `.env.example` contains only the variable names and empty values.

## Target file structure (build incrementally, per sub-phase)

The current `concept_explainer.py` is a single 500-line file. Split it as
each sub-phase lands. Do NOT create the full tree in one shot; add files
per sub-phase so each diff stays reviewable.

```
akadverse-concept-explainer/
|-- AGENTS.md
|-- main.py                       # FastAPI app + lifespan + router mounting (2b)
|-- config.py                     # Settings from env, startup validation (2a)
|-- auth.py                       # Bearer token dependency (2b)
|-- db.py                         # SQLAlchemy async engine, session factory (2a)
|-- models.py                     # Pydantic request/response schemas (2b)
|-- routes/
|   |-- ingest.py                 # POST /ingest (2b)
|   |-- documents.py              # DELETE /documents/{id} (2d)
|   `-- grounding.py              # POST /explain-with-grounding (moved unchanged)
|-- workers/
|   `-- ingestion.py              # Job loop, re-queue on startup (2b)
|-- services/
|   |-- extractors/
|   |   |-- pdf.py                # (2c)
|   |   |-- docx.py               # (2c)
|   |   `-- pptx.py               # (2c)
|   |-- chunking.py               # (2c)
|   |-- embedding.py              # Gemini embedding, retry/backoff (2b)
|   `-- callback.py               # Outbound POST with retry (2d)
|-- tests/
|   |-- conftest.py               # Root fixture: async client, test DB (2a)
|   |-- test_auth.py              # (2b)
|   |-- test_ingest.py            # (2b)
|   |-- test_extractors.py        # (2c)
|   |-- test_delete.py            # (2d)
|   `-- test_callback.py          # (2d)
|-- conftest.py                   # Empty, already present
|-- pyproject.toml
`-- requirements.txt
```

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

**Phase 2b3: services/embedding.py, workers/ingestion.py, end-to-end PDF ingestion.**
Add `services/embedding.py` and `workers/ingestion.py`. Wire end-to-end PDF
ingestion.

**Phase 2c: extractors.** Add `services/extractors/*` and
`services/chunking.py`. Wire them into the worker. Add unit tests per
extractor.

**Phase 2d: delete + callback.** Add `routes/documents.py` and
`services/callback.py`. Wire the callback into the worker's completion path.

**Phase 2e: retire old endpoints.** Delete `POST /upload-note`,
`POST /explain-concept`, FAISS code, `faiss-cpu` requirement, and the
`akadverse_concept_faiss_index/` folder. Update the README.

## Testing

Windows: the test suite requires a SelectorEventLoop. `tests/conftest.py`
sets the policy at import time. When Phase 2b creates `run.py` or any
other process launcher, include the same policy setup before uvicorn runs.
The same applies when the FastAPI app itself starts using `db.py` on Windows.

- **Framework:** pytest + `httpx.AsyncClient` via `httpx.ASGITransport`
- **Command:** `uv run pytest tests/ -q --tb=short`
- **Async tests:** `pytest-asyncio` with `asyncio_mode = "auto"` in
  `pyproject.toml`
- **Mock everything external:** Gemini SDK (`monkeypatch` the embed method),
  Supabase signed-URL fetch (`respx` or `httpx.MockTransport`), callback
  POST (`respx`)
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
- Do not add ZIP support yet. That is Phase 3
- Do not touch `/explain-with-grounding`
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
```

---

## What to do next

**Do the smoke tests before any real work.** In a fresh Codex session, run these two prompts in sequence. Do not write code until both pass.

**Smoke test 1:**

> Read both AGENTS.md files. In one paragraph, tell me what this service does, what phase it is in, and what you would build next. Do not write any code.

You want Codex to say: this is the AkadVerse concept explainer on port 8010, it is in Phase 2 (ingestion rewrite), next step is Phase 2a (pgvector migration). It should **not** say anything about extending `/upload-note` or building `/explain-concept`.

**Smoke test 2:**

> Open `concept_explainer.py`. List every piece of code in it that is marked as legacy in the AGENTS.md and would need to be removed or replaced during Phase 2. Do not modify anything yet.

You want Codex to name: FAISS imports, `FAISS_INDEX_PATH`, `_vector_store`, `faiss_lock`, `time.sleep()`, `print()` calls, `/upload-note`, `/explain-concept`. If it misses any of these or names something that is not on the list, the AGENTS.md is not being read properly and you should stop and debug that before proceeding.

**If both pass:** come back and I will write the Phase 2a prompt (pgvector migration) calibrated for Codex. That phase is small and self-contained, and it will confirm whether Codex is following the file structure and phase discipline as intended before you hand it the bigger Phase 2b work.

**One last thing:** the `[file content begin]` blocks you have been pasting include a raw `AI_SERVICE_CALLBACK_SECRET` value. Do not copy that into `.env` locally either, even for testing. Generate a fresh one with `openssl rand -hex 32`, use that for your dev environment, and tell the FSE to rotate theirs before production. The one in chat is now treated as public.
