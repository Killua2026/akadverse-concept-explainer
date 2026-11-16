# AkadVerse Concept Explainer

FastAPI microservice for the AkadVerse Study Zone, reserved port **8010**.
The intended caller is the Next.js backend, not a browser client.

## Current status

- Phase 3 (`POST /explain`) is committed and verified end-to-end on Railway
  with real PDF and PPTX lecture documents.
- Service deployed to Railway at `<RAILWAY_URL>` (staging).
- Smoke test verified end-to-end ingestion: 202, chunks in
  `concept_explainer.chunks` (pgvector), and job `done`.
- Callback delivery to `https://akadverse.vercel.app/api/webhooks/ai-ingest`
  is verified working (200 OK).
- Two smoke tests passed: INS412 PDF produced 28 chunks; CSC424 PPTX produced
  17 chunks. `/explain` returned grounded answers with correct source citations
  and correct "not covered" behaviour for questions outside the material.
- App logs route to stdout, appearing as `[info]` in Railway instead of
  `[error]`. `httpx` INFO logging is suppressed to prevent signed URL leakage.
- Railway trial expires early December 2026: plan an upgrade or migration.
- Phase 3.5c adds image attachments to Stewart retrieval alongside the Phase 3.5b text attachments.
- The known skipped concurrency test remains skipped.
- The deployed URL is staging-only; production deploy is a separate decision.

| Capability | Current state |
|---|---|
| Authenticated `POST /ingest` | Implemented; returns `202` after queue admission |
| Durable jobs and configurable workers | Implemented; startup recovery and concurrent claims |
| PDF / DOCX / PPTX extraction | Implemented |
| Shared chunking and Enterprise embeddings | Implemented |
| Delete endpoint and outbound callback client | Implemented; transactional deletion and bounded post-commit delivery |
| Legacy endpoint retirement and grounding auth | Implemented and committed in Phase 2e |
| Concurrent job-claim collision test | Retained but skipped; dedicated investigation pending for intermittent teardown hang |
| Retrieval over ingested pgvector chunks | Implemented: authenticated synchronous `POST /explain` |
| `POST /analyse-material` | Implemented: decomposes ingested material into a structured lesson plan with sections, concepts, prerequisites, difficulty, and source chunk citations |
| Concept-scoped `/explain` | Implemented: optional `concept_id` and `concept_description` fields scope Stewart's answer to a single concept |
| `POST /generate-example-tasks` | Implemented: decomposes a worked example into ordered interactive student tasks (multiple-choice, text, number, step responses) with expected answers and evaluation criteria |
| `POST /evaluate-practice` | Implemented: LLM-graded answer and reasoning evaluation with misconception diagnosis; DB-free, returns 200 with `needs_review=true` on degraded fallback |
| `POST /generate-practice` | Implemented: independent-practice question generation, source-grounded, solution hidden from student until submission; distinct from guided `/generate-example-tasks` |
| `POST /generate-revision-questions` | Implemented: generates revision questions from a taught concept and submits them to FSE's `/webhooks/tutor/questions` via HMAC-SHA256; requires `FSE_WEBHOOK_URL` and `FSE_WEBHOOK_SECRET` on the deployed service |
| `POST /generate-hint` | Implemented: generates one Assistance Ladder hint (L1-L6) at a caller-specified level; stateless, DB-free; FSE owns escalation policy |
| Stewart chat history | Implemented in Phase 4a: 105-day persistence, last-10-message context, opt-in via `session_id` |
| File attachments on `/explain` | Implemented in Phases 3.5a, 3.5b, 3.5c: text files and images, up to 8 per request |
| AI-generated worked examples | Implemented in Phase 4a: labeled with `[AI-GENERATED EXAMPLE]` and `source_type="ai_generated"` |
| ZIP ingestion | Rejected; deferred to a later phase |

`/ingest` writes pgvector chunks. `/explain` retrieves from these chunks.
Callbacks are posted after processing commits. A job marked `done` means its
chunks were stored, not that a callback was delivered. Delivery failure does
not change the committed job status or roll back chunks.

## Contents

- [Architecture](#architecture)
- [Installation and startup](#installation-and-startup)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Database setup](#database-setup)
- [API endpoints](#api-endpoints)
  - [Analyse material](#post-analyse-material)
  - [Generate example tasks](#post-generate-example-tasks)
  - [Evaluate practice](#post-evaluate-practice)
  - [Generate practice](#post-generate-practice)
  - [Generate revision questions](#post-generate-revision-questions)
  - [Generate hint](#post-generate-hint)
  - [Demo flow: lesson planning → teaching](#demo-flow-lesson-planning--teaching)
  - [Attachments](#attachments)
- [FSE Integration Testing Guide](#fse-integration-testing-guide)
- [Extraction and processing](#extraction-and-processing)
- [Jobs and workers](#jobs-and-workers)
- [Outbound callbacks](#outbound-callbacks)
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
  -> transaction: replace document chunks and conditionally mark job done
  -> temp-file cleanup -> outbound callback using the shared HTTP client
```

Processing failures commit a sanitized error before attempting a failed
callback. DELETE removes document chunks and jobs in one transaction; workers
detect cancellation during finalization and skip delivery.

Grounding uses the selected Gemini provider and Google Search, returning an
explanation with cited sources. It requires shared-secret authentication.

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
Its dev group pins `pytest-timeout==2.4.0` and
`sqlalchemy[asyncio]==2.1.4`; the latter keeps async SQLAlchemy's `greenlet`
dependency installed during `uv sync`. Keep that pin synchronized with the
service requirements. The service requirements also list pytest and
pytest-timeout; synchronization does not replace installing service dependencies.

Service requirements include FastAPI, Uvicorn, `google-genai>=1.67.0`,
`langchain-text-splitters`, Pydantic,
`pypdf`, `python-docx==1.2.0`, `python-pptx==1.0.2`,
`sqlalchemy[asyncio]==2.1.4`, `psycopg[binary]==3.2.6`, and
`python-dotenv==1.0.1`, and `python-multipart>=0.0.9`.
`requirements.txt` is authoritative for service pins.
`python-multipart` is required by FastAPI's `Form(...)` parameter parsing. Must
be in `requirements.txt`, not just the parent `pyproject.toml`, because the
container installs only from `requirements.txt`.

Copy `.env.example` to your private `.env` and supply the required settings.
Never commit or share credential values. Existing process variables take
precedence over values loaded by python-dotenv.

Start the service from this directory:

```powershell
uv run python run.py
```

`run.py` sets the Windows SelectorEventLoop policy before Uvicorn starts,
selects the asyncio loop, and disables reload. Bind address is environment-aware:
local development (no `$PORT`) uses `127.0.0.1:8010`; container deployment
(`$PORT` set) uses `0.0.0.0:$PORT`.
Direct Uvicorn startup bypasses this Windows policy setup.
Use [Swagger UI](http://127.0.0.1:8010/docs) to inspect the four current API
operations and their request schemas. Port 8010 belongs to this service;
consult the root `AGENTS.md` before choosing any other local testing port.

For container deployment, see [Deployment](#deployment). The `Dockerfile` at the
service root installs from `requirements.txt` only; anything that runs at runtime
must be listed there. This was surfaced in practice: `python-multipart` was
missing from `requirements.txt` and only worked locally because the parent
workspace `pyproject.toml` supplies it. The container had no parent workspace
and crashed at import.

### Docker

```bash
cd akadverse-concept-explainer
docker build -t akadverse-concept-explainer .
docker run --rm -p 8010:8010 --env-file .env akadverse-concept-explainer
```

The image installs from `requirements.txt`, runs `run.py`, and respects
`$PORT` when the platform injects it (Railway, Render, Fly). Credentials
must be provided via `GOOGLE_APPLICATION_CREDENTIALS_JSON` (base64) rather
than `GOOGLE_APPLICATION_CREDENTIALS` (file path) because the file path
does not exist inside the container. For this local Docker command, set
`PORT=8010` in the private `.env` so the server accepts container traffic.

## Deployment

The ingestion worker is an `asyncio.Task` inside the FastAPI lifespan and must
run continuously. Request-scoped serverless platforms cannot host it reliably:
Vercel Functions end execution after requests, and Cloud Run without
`min-instances=1` can stop the process between requests. Railway,
[Fly.io](https://fly.io/), and Render Starter are suitable; Render Free and
Vercel are not. Keep CPU allocated between requests on any chosen platform.

The container build uses the `Dockerfile` at the service root. The image installs
from `requirements.txt` only. The parent workspace `pyproject.toml` is not
available in the container, so anything needed at runtime must be listed in
`requirements.txt`, including `python-multipart` for `Form(...)` parameters.

Environment variables are similar to local development: `AI_SERVICE_API_KEY`,
`AI_SERVICE_CALLBACK_ALLOWED_HOSTS`, `AI_SERVICE_CALLBACK_SECRET`, `DATABASE_URL`,
`GOOGLE_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS_JSON`, `GOOGLE_CLOUD_LOCATION`,
`GOOGLE_CLOUD_PROJECT`, `LLM_PROVIDER`, `PORT`, and `WORKER_COUNT`, with two changes:

- `GOOGLE_APPLICATION_CREDENTIALS_JSON` (base64 of the service-account JSON)
  replaces `GOOGLE_APPLICATION_CREDENTIALS` (a local file path).
- The platform injects `$PORT`, which `run.py` reads to bind `0.0.0.0:$PORT`.

Apply both migrations to the target Supabase database before the first request.
The deployed service does not run migrations on startup.

Point the platform healthcheck at `/health` with a 120-second grace period.
Startup runs up to three 20-second DB probes plus recovery and worker startup.
Set the restart policy to **On Failure**. Do not disable restart; a worker crash
should trigger a restart. This restarts an exited process; an individual task
stopping does not itself guarantee process exit (see Jobs and workers).
Do not enable scale-to-zero, Serverless sleep, or any serverless sleep setting.
The worker stops when the process stops.

## Configuration

| Variable | Current use |
|---|---|
| `LLM_PROVIDER` | `gemini` by default; explicitly use `gemini_enterprise` for ingestion |
| `GOOGLE_API_KEY` | Required for Developer mode; never an HTTP form field |
| `GOOGLE_CLOUD_PROJECT` | Required for Enterprise mode |
| `GOOGLE_CLOUD_LOCATION` | Enterprise generation/discovery location; default `global` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Optional path to an external ADC credential file; runtime ADC is also supported |
| `DATABASE_URL` | Required for ingestion; private Postgres connection string |
| `AI_SERVICE_API_KEY` | Shared secret for ingestion, deletion, and grounding authentication |
| `AI_SERVICE_CALLBACK_SECRET` | Required Bearer secret for outbound callbacks |
| `AI_SERVICE_CALLBACK_ALLOWED_HOSTS` | Required comma-separated, exact callback hostnames; no wildcard matching |
| `WORKER_COUNT` | Positive integer, default `1`; restart required after changes |

Developer mode disables ingestion and workers; grounding requires shared-secret authentication independently of ingestion readiness. `/ingest` returns `503` with an Enterprise-mode requirement.
A Cloud project setting alone does not select Enterprise mode. Unknown providers,
missing required provider credentials/settings, and invalid worker counts fail
startup validation.

In Enterprise mode, ingestion requires non-empty `DATABASE_URL`,
`AI_SERVICE_API_KEY`, `AI_SERVICE_CALLBACK_SECRET`, and
`AI_SERVICE_CALLBACK_ALLOWED_HOSTS`. Missing settings report
`disabled (missing VARIABLE_NAME)` through `ingest_status`; `/ingest` returns
503 and workers do not start. An empty or comma/whitespace-only allowlist is
missing configuration. Grounding requires only the inbound shared secret for authentication and remains available when ingestion is disabled.

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
data from retired local indexes. `IF NOT EXISTS` supports reruns but does not reconcile
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

All displayed request and metadata fields are required. `file_url` is validated
as an HTTP URL. Before queueing, `callback_url` must use HTTPS, have no embedded
credentials, and match an explicitly listed hostname. Host comparison ignores
case and trims whitespace around allowlist entries; it does not match wildcards
or implicitly authorize subdomains. The example requires `example.com` in the
operator's allowlist; it is not a default authorized destination.

Admission awaits DNS resolution with a two-second timeout. Listed hosts must
resolve only to globally routable addresses, except explicitly listed
`localhost` and `127.0.0.1`, which bypass the private-address guard for local
development. HTTPS is still required for those local hosts. Invalid URLs,
HTTP callbacks, unlisted hosts, and disallowed/unresolvable addresses return
422 with a callback-specific detail. DNS is checked at admission, not pinned
to the later connection, so this is not a guarantee against DNS rebinding.

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

### DELETE /documents/{document_id}

Uses the same `require_ingest_auth` dependency and Bearer token as `/ingest`,
including its configured/ready-state checks. The path parameter is a string.
The request deletes matching chunks, then matching jobs, in one database
transaction using async I/O. It does not enqueue work or delete the remote file.

- `204 No Content`, with no body, if either table had matching rows.
- `404 Not Found`, with `{"detail":"Document not found"}`, if neither did.
- `401` for missing/wrong auth when ready; `503` for unavailable configuration
  or database operations. Next.js treats both 204 and 404 as successful deletion.

Other document IDs are untouched. An in-flight worker detects a removed job
when its conditional finalization returns no row, as described below.

### GET /documents/{document_id}/status

Returns the current state of an ingestion job. Authenticated with the same
Bearer token as `/ingest` and `DELETE`. Use this to reconcile missed callbacks:
after a `POST /ingest`, poll every 10 seconds for up to 5 minutes. If the
callback arrives first, stop polling.

Response:

```json
{
  "document_id": "lecture-001",
  "status": "done",
  "chunk_count": 42,
  "attempts": 1,
  "last_error": null,
  "created_at": "2026-11-04T16:20:00Z",
  "updated_at": "2026-11-04T16:21:00Z"
}
```

- `200` if the job exists.
- `404 {"detail": "Document not found"}` if the job does not exist. Retry after
  5s if the `POST /ingest` 202 was just received. This is rare: the row is
  committed synchronously before 202 returns, so 404 usually means the document
  was never submitted to this service. Worker claiming is not required for visibility.
- `401` if auth is missing or wrong.
- `503` if the database is unavailable.

### POST /explain

Answers are Markdown-formatted. Mathematical expressions use LaTeX (`$...$` inline, `$$...$$` for display). AI-generated worked examples are marked with `[AI-GENERATED EXAMPLE]` on its own line as plain text. The frontend detects this literal string and applies its own visual styling. The response's `sources` array is separate from the answer text; render them independently.

`POST /explain` is synchronous JSON. Send `Authorization: Bearer <AI_SERVICE_API_KEY>`
and a body such as `{"question":"Explain recursion","course_offering_id":"offering-id","week":3}`.
Question length is 1-4000 characters, offering ID is non-empty, and week is at least 1.
An optional nonempty `session_id` enables conversation context and persistence.
Stewart sees up to 10 prior messages for the session, and the service saves
the current question and answer, including canned retrieval responses. The
service sweeps chat messages older than 105 days at startup and hourly.
Apply migration 004 before starting this version. An omitted or empty
`session_id` skips history. The FSE backend must scope session IDs to the
authorized student. The request field defaults to `session_id: str = ""`.

The microservice does not create sessions. There is no `POST /sessions` endpoint.
The FSE backend generates a UUID (or any stable string) per authorized
conversation, stores it alongside the conversation record, and sends it on every
`/explain` call for that conversation. To continue a conversation, send the same
`session_id`. To start a fresh conversation, send a new one. Stewart loads the
stored history, so send the current question and the session ID along with the
usual request fields.

For a student opening Week 3 of CSC424 in the Study Zone, generate one `session_id` per (student, course_offering, week) tuple and reuse it across all questions in that study session.

Optionally include `attachment_ids`, an array of up to eight IDs from
`POST /attachments`. PDF, DOCX, PPTX, TXT, and MD attachments are supported.
A missing or expired ID or an extraction failure returns 422 for the entire request.
See [Attachments](#attachments) for the upload workflow and expiry handling.

Two optional fields, `concept_id` and `concept_description`, scope the answer to a
single concept from a lesson plan, typically from `/analyse-material`. When both
are non-empty, the prompt instructs Stewart to answer for that concept and its
immediate prerequisites only. When either field is empty, the prompt is
byte-identical to the unscoped prompt. Retrieval still uses the requested course
offering and week, so Gemini may occasionally mention adjacent concepts. See
[Demo flow](#demo-flow-lesson-planning--teaching) below.

Image attachments support `image/png`, `image/jpeg`, and `image/webp` and can be mixed
with text files. The cap is 4 MiB per image and eight total attachments, with no
aggregate image byte cap. Each image appears after the text attachment sources, in
request order, with `source_type="attachment"`, `document_id=null`, `attachment_id`,
`filename`, and `chunk_preview="[image: filename]"`. A vision rejection returns a
sanitized 422 for the entire request.

Per-model inline limits can be lower than the API-wide limit. Smoke-test a real 4 MiB image on the deployed model (`gemini-3.5-flash-lite` or higher).

The response has `answer`, `sources` (`document_id`, `title`, `material_type`,
`chunk_preview`, the first 200 content characters), and `retrieval_info`
(`chunks_available`, `chunks_above_threshold`, `similarity_threshold`).

| `source_type` | When it appears | UI suggestion |
| --- | --- | --- |
| `course_material` | Retrieved from a lecturer-ingested document for the requested offering and week | Cite the `title` and `document_id`. Link to the source document if your UI supports it. |
| `attachment` | Retrieved from a text or image file the student uploaded via `POST /attachments` | Show the `filename`. For images, `chunk_preview` is `"[image: {filename}]"`. Render a thumbnail if you have one. |
| `ai_generated` | A worked example Stewart generated because the source material contained none | Render with a distinct style (icon, tint, or badge) and the marker `[AI-GENERATED EXAMPLE]`. Never present as lecturer content. |

With attachment IDs, sources also include `source_type`, `attachment_id`, and `filename`. Attachment sources have a null
`document_id`. The retrieval info adds `attachment_chunks_available` (all chunks
before the cap) and `attachment_chunks_above_threshold` (capped chunks meeting
the threshold before the top-eight cutoff). These counters cover text attachments.
`attachment_images_count` reports how many images were sent to Gemini and is zero
for requests with only text attachments. Course sources keep their existing
shape when `attachment_ids` is omitted or empty.

Stewart starts directly with the answer and only greets in response to a
greeting with no academic content. When Gemini is called and the student
requests a worked example, Stewart uses a suitable source example if available.
Otherwise it can generate one with `[AI-GENERATED EXAMPLE]` on its own line.
The marker stays in the answer, and the response adds one source with
`source_type="ai_generated"` and `material_type="AI_GENERATED"`. Its preview
includes the first marker followed by up to 200 characters of the following
text with whitespace normalized.

The question uses the ingestion embedding pipeline: gemini-embedding-2, global,
768 dimensions, no retries. Retrieval filters offering ID and JSONB weeks membership,
counts available/qualifying chunks, and retrieves up to eight at cosine distance
at most 0.6 (similarity at least 0.4). Attachments share a total budget of 100
embedded chunks, allocated proportionally across files with at least one per
nonempty file. The same question embedding retrieves up to eight attachment
chunks at the same threshold. Attachment chunks stay in memory for the request.
Gemini uses the first discovered model and temperature 0.3, with separate course
material and attached-file context when both have qualifying chunks.
Other generation failures or empty output return a fallback answer. Embedding
failures return sanitized 503. Database failures and Developer mode return 503.
Missing or wrong Bearer credentials return 401 when the shared secret is configured.

Database initialization and its existing retry/probe policy precede shared-secret
and callback configuration checks. A reachable database remains available for
retrieval when callbacks are disabled. Failed probes dispose the partial database
and clear its state. Ingestion and workers still require ready status.
No callback secret or allowlist is required by retrieval. Callers handle rate
limiting.

With no images, these short-circuit branches return HTTP 200 with `sources: []`
and skip LLM generation:

- No attachment IDs and no material (`chunks_available == 0`): exact canned answer is
  `No course material has been ingested for Week {week} of this offering yet.`
  Its prefix is `No course material has been ingested`.
- Neither course material nor attachments has a qualifying chunk, in all other cases: exact
  canned answer is `The course material for Week {week} does not cover this question.`
  Its prefix is `The course material for Week {week}`.

`{week}` is replaced with the requested week. When attachment chunks qualify
and course chunks do not, Stewart can answer using only those attachments and
cite their filenames.

When the 100-chunk total cap is active, each attachment contributes only its first N chunks in document order, where N is its proportional allocation. For very large files, this means only the beginning of the document is retrievable. Callers with large source documents should either split them into topic-scoped uploads or accept that questions about later sections will not find supporting chunks.

Latency budget for /explain: requests without attachments typically complete in 10-20 seconds (course material retrieval + LLM generation). Requests with attachments add the embedding cost of the attached chunks: a small attachment (10-20 chunks) adds ~10-15 seconds; a request hitting the 100-chunk cap adds ~80 seconds of inter-batch sleeps plus embedding compute, for a worst case of roughly 2 minutes end-to-end. Frontend clients must set a request timeout of at least 150 seconds to accommodate the worst case. Callers needing faster responses for large attachments should use smaller attachments, or wait for the asynchronous /explain variant planned for Phase 4.

Attachment ownership is enforced by the FSE backend before calling this service. The microservice stores and serves any attachment by ID to any caller presenting a valid `AI_SERVICE_API_KEY`. The FSE backend must verify that a student is authorized to view a course/material before including its `attachment_ids` in an `/explain` request, and must verify student identity before calling `/attachments` or `/explain` at all.

### POST /analyse-material

Builds a structured lesson plan from an ingested document. The plan provides the
foundation for lesson state and concept-scoped explanations.

Use the same `Authorization: Bearer <AI_SERVICE_API_KEY>` header as `/explain`.

```json
{"document_id": "csc424-lecture-1-2027", "max_concepts": 8}
```

`max_concepts` is optional, defaults to 12, and accepts values from 3 to 20.

The response contains `document_id`, `lesson_plan`, and `chunks_analysed`.
`chunks_analysed` counts chunks whose content was included in the prompt.

| `lesson_plan` field | Shape |
| --- | --- |
| `document_id` | String |
| `sections` | Array of objects with `section_id`, `title`, and `concept_ids` |
| `concepts` | Array of objects with `concept_id`, `title`, `description`, `learning_objective`, `difficulty`, `prerequisite_concept_ids`, and `source_chunk_ids` |
| `missing_prerequisites` | Array of strings |
| `notes` | String or null |

The service reads all chunks for the document and calls Gemini with
`response_schema` for structured JSON output. It repairs minor reference defects
before validation and returns a single-concept plan if generation and recovery
fail.

- `200` returns the generated plan or a fallback plan.
- `401` indicates a missing or invalid Bearer token.
- `404` indicates that the document has no ingested chunks.
- `503` indicates unavailable database access, developer mode, or missing auth configuration.
- `422` indicates an invalid request body.

The prompt includes up to 80,000 characters of concatenated chunk content and
markers, so long documents may lose later topics. Plan quality depends on
Gemini's structured output, and pathological inputs may trigger the fallback.

Smoke-test concept `concept-4`, **Modes of Data Transmission**, was intermediate,
required `concept-1`, and cited chunks `98` through `100`.

### POST /generate-example-tasks

Takes a concept from a lesson plan and returns a worked example with 3 to 10
ordered interactive micro-tasks. Use the same Bearer token as `/explain`.

```json
{
  "concept_id": "concept-4",
  "concept_description": "The classification of data flow directions between devices, categorized as simplex, half-duplex, and full-duplex.",
  "document_id": "csc424-lecture-1-2027",
  "max_tasks": 5,
  "example_text": ""
}
```

`concept_id`, `concept_description`, and `document_id` are required.
`max_tasks` defaults to 6 and accepts 3 to 10. `example_text` is optional.

Response fields

- `concept_id` (string)
- `example` (object) with `source` (`LECTURE_NOTE` or `AI_GENERATED`),
  `text` (string), and `source_chunk_ids` (list of strings)
- `tasks` (list) with `task_id` (string), `sequence` (integer), `prompt`
  (string), `response_type` (`text`, `number`, `multiple_choice`, or `step`),
  `expected_answer` (string), `evaluation_criteria` (list of strings),
  `hint_strategy` (string or null), and `required` (boolean)

- An empty `example_text` triggers two Gemini calls, one to produce a worked
  example and one to decompose it. A supplied example goes straight to
  decomposition. Both stages use `response_schema` for structured JSON.
  The service validates tasks, drops duplicates, renumbers them, and caps
  the result at `max_tasks`.
- `LECTURE_NOTE` identifies an example found in the ingested material.
  The FSE should label `AI_GENERATED` examples in the UI, consistent with
  `source_type="ai_generated"` in `/explain` sources.
- Keep `expected_answer` and `evaluation_criteria` server-side until the
  student submits. They support immediate feedback and the future
  `/evaluate-practice` evaluation flow.
- Status codes include 200 for success and fallback, 401 for auth failure,
  404 for an unknown document or one with no chunks, and 503 when the
  database is unavailable or the service is in developer mode.
- The 30,000-character source cap can omit a lecture example and lead to an
  AI-generated one. Task quality depends on Gemini's structured output.
  Generation failure or fewer than three valid tasks yields a single-task plan.
- **`concept_description` has the biggest influence on task quality.**
  A vague description such as "transmission modes" produces vague tasks.
  A specific description such as "classification of data flow directions
  between devices, categorized as simplex, half-duplex, and full-duplex"
  produces concrete, checkable tasks. The FSE should pass the concept's
  `description` from `/analyse-material` verbatim as `concept_description`.
  Both descriptions pass validation. The difference is in the tasks Gemini returns.

This is the third teaching stage after lesson planning and concept-scoped
explanation. See [Demo flow](#demo-flow-lesson-planning--teaching).

### POST /evaluate-practice

Grades a student's practice answer against the expected answer and criteria
returned by `/generate-example-tasks`. Returns structured feedback, reasoning
assessment, misconception diagnosis, and a recommended next step. This
endpoint is DB-free; it does not read from `concept_explainer.chunks` and
does not require an ingested document.

Use the same `Authorization: Bearer <AI_SERVICE_API_KEY>` header as `/explain`.

```json
{
  "concept_id": "concept-4",
  "concept_description": "The classification of data flow directions between devices, categorized as simplex, half-duplex, and full-duplex.",
  "question": "<task prompt from /generate-example-tasks>",
  "expected_answer": "<that task's expected_answer>",
  "evaluation_criteria": ["<that task's criteria>"],
  "student_answer": "<student's answer>",
  "student_reasoning": "<optional; student's reasoning>",
  "response_type": "text"
}
```

`concept_id`, `concept_description`, `question`, `expected_answer`, and
`student_answer` are required non-empty strings. `evaluation_criteria`
defaults to `[]`. `student_reasoning` defaults to `""`. `response_type`
defaults to `"text"` and accepts `text`, `number`, `multiple_choice`, or
`step`.

The response contains nine fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `concept_id` | string | Echo of the request's `concept_id` |
| `answer_correct` | boolean | Whether the answer matches the expected answer and criteria |
| `answer_feedback` | string | Specific feedback on the answer |
| `reasoning_quality` | `"strong"` \| `"adequate"` \| `"weak"` \| `"absent"` | Assessment of the student's reasoning |
| `reasoning_feedback` | string | Specific feedback on the reasoning |
| `misconceptions` | array of `{label, evidence}` | Detected misconceptions with quoted evidence; empty when none |
| `evidence_strength` | `"independent"` \| `"assisted"` \| `"insufficient"` | Whether the student appears to have solved without hints |
| `needs_review` | boolean | True when the answer is ambiguous, the reasoning conflicts with the answer, or the LLM is uncertain |
| `recommended_next_step` | `"progress"` \| `"reinforce"` \| `"teach_prerequisite"` | How the FSE should route the student after this attempt |

`needs_review` is a required boolean, not a defaulted field. The LLM sets
it per response. When `needs_review` is `true`, the FSE should route the
attempt to instructor review or flag it for later attention rather than
committing to `recommended_next_step` blindly.

Response notes for FSE:

- The grading prompt reserves `"progress"` for correct answers with adequate
  or strong reasoning. For a correct answer without reasoning, it asks the
  model to set `answer_correct=true`, `reasoning_quality="absent"`, and
  `needs_review=false`. The response schema allows `"progress"` with
  `"absent"`, so check `reasoning_quality` when gating advancement on reasoning.
- `misconceptions` is empty when no specific misconception was diagnosed.
  A wrong answer with no identifiable misconception returns an empty list,
  not a generic entry. The FSE should not assume "wrong" and "misconception"
  are synonymous.
- `evidence_strength` reflects the absence or presence of hint context in
  the submission; it is not a judgment of the student's overall ability.

If the LLM call fails, the schema is violated, or the response cannot be
parsed, the endpoint returns HTTP 200 with a degraded payload:
`answer_correct=false`, `reasoning_quality="absent"`,
`evidence_strength="insufficient"`, `needs_review=true`,
`recommended_next_step="reinforce"`, and explanatory fallback strings in
`answer_feedback` and `reasoning_feedback`. The same `needs_review` flag also
covers completed evaluations that need review. The response has no dedicated
fallback indicator, so pause automatic routing and show the feedback when
`needs_review` is `true`.

| Status | Meaning |
| --- | --- |
| `200` | Completed evaluation or degraded fallback |
| `401` | Missing or incorrect Bearer token when shared-secret authentication is configured |
| `422` | Invalid request body, such as a missing required field or an unsupported `response_type` |
| `503` | Shared-secret authentication or the LLM provider is unconfigured |

The endpoint requires no ingested document or database connection.

Expected latency is 3-10 seconds for a single evaluation.

### POST /generate-practice

Generates complete, independent-practice questions for a specific concept.
Distinct from `/generate-example-tasks` in learning objective, prompt, and
solution visibility. Use this when the student is ready to demonstrate
understanding without guidance; use `/generate-example-tasks` when introducing
the concept.

| Aspect | `/generate-example-tasks` (guided) | `/generate-practice` (independent) |
| --- | --- | --- |
| Primary objective | Teach how to approach and solve | Measure independent understanding |
| Solution visibility | Revealed progressively as micro-tasks | `expected_answer` and `explanation` returned to caller, kept server-side, NOT shown to student before submission |
| Task structure | Worked example decomposed into 3-10 ordered micro-tasks | 1-5 standalone questions, one attempt each |
| Response field | `tasks[]` with `sequence`, `prompt`, `hint_strategy` | `questions[]` with `question_text`, `options`, `expected_answer`, `explanation` |
| `activity_type` | (implicit) | **`"INDEPENDENT_PRACTICE"`** (fixed literal) |
| FSE flow | Student sees steps one at a time, guided | Student sees question only, attempts, then FSE calls `/evaluate-practice` |

Learning progression: explanation → `/generate-example-tasks` (guided) →
`/generate-practice` (independent) → `/evaluate-practice` (grading) → adaptive
next teaching action.

Use the same `Authorization: Bearer <AI_SERVICE_API_KEY>` header as `/explain`.

```json
{
  "concept_id": "concept-4",
  "concept_description": "The classification of data flow directions between devices, categorized as simplex, half-duplex, and full-duplex.",
  "document_id": "csc424-lecture-1-2027",
  "difficulty": "medium",
  "question_type": "short_answer",
  "max_questions": 2
}
```

`concept_id`, `concept_description`, and `document_id` are required.
`difficulty` defaults to `"medium"` and accepts `easy`, `medium`, or `hard`.
`question_type` defaults to `"short_answer"` and accepts `short_answer`,
`multiple_choice`, `numeric`, or `step`. `max_questions` defaults to 1 and
accepts 1 to 5.

The response contains four top-level fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `concept_id` | string | Echo of the request's `concept_id` |
| `activity_type` | `"INDEPENDENT_PRACTICE"` | Fixed literal distinguishing this endpoint from `/generate-example-tasks` |
| `questions` | array | 1 to `max_questions` entries; empty on degraded fallback |
| `needs_review` | boolean | `false` on success; `true` on degraded fallback; set by the service, not the LLM |

Each question in `questions` has: `question_id` (format `pq-{concept_id}-{n}`),
`concept_id`, `question_text`, `question_type`, `options` (empty for non-multiple-choice),
`expected_answer`, `explanation`, `evaluation_criteria` (list of strings),
`difficulty`, and `source_chunk_ids` (chunks whose content informed the question).

FSE integration notes:

- **Keep `expected_answer` and `explanation` server-side.** The student should
  see only `question_text` and (for multiple choice) `options` before submitting.
  Revealing `expected_answer` before submission defeats the entire purpose.
- **Route attempts through `/evaluate-practice`.** When the student submits,
  pass their answer and reasoning to `/evaluate-practice` with the same
  `concept_id`, `concept_description`, `question_text` (as `question`), and
  `expected_answer`. `/generate-practice` does not grade.
- **Distinguish guided vs independent evidence.** Per FSE's learning-evidence
  vocabulary: guided example completed (from `/generate-example-tasks`),
  practice completed independently, practice completed with hints, practice
  completed with substantial assistance, or incorrect response/misconception
  identified (from `/evaluate-practice`). Do not conflate.
- **Do not treat `source_chunk_ids` as "these specific chunks caused this
  question."** On a small document under the 30,000-character prompt cap, all
  chunks are included in the prompt and therefore all are listed. On a large
  document, the list is a subset.

If the LLM call, schema, parsing, or repair fails, the endpoint returns HTTP
200 with `activity_type="INDEPENDENT_PRACTICE"`, `questions=[]`, and
`needs_review=true`. FSE should surface "practice unavailable" rather than
displaying a fake question. The degraded path never returns 5xx.

Status codes: `200` for success or degraded fallback; `401` for missing or
incorrect Bearer token; `404` when the document has no ingested chunks; `503`
when the ingest database is unavailable or the service runs in developer mode.

Expected latency is 8-25 seconds.

### POST /generate-revision-questions

Generates revision questions from a taught concept and submits them to FSE's `/webhooks/tutor/questions` webhook with HMAC-SHA256 auth. This endpoint completes the revision-question-bank loop: FSE detects a passed teaching stage, calls us, we generate, we submit, FSE persists and returns IDs, we return the combined result to FSE's original call.

Stateless and DB-free. FSE owns persistence, semantic dedup, embeddings, FSRS scheduling, student cards, and review logs. We own generation, wire-format transformation, and HMAC submission.

Flow:

```
FSE stage tracker (student passed) -> POST /generate-revision-questions
 -> Gemini generates 1-5 questions
 -> HMAC-signed POST to FSE_WEBHOOK_URL
 -> FSE persists, dedups, creates FSRS cards, returns results[]
 -> Service returns GenerateRevisionQuestionsResponse to FSE's original call
```

Use the same `Authorization: Bearer <AI_SERVICE_API_KEY>` header as other authenticated endpoints.

Request from FSE:

```json
{
  "teaching_event_id": "teach_9f31...",
  "student_id": "stu_123",
  "concept_id": "con_concept-4",
  "concept_description": "The classification of data flow directions...",
  "taught_summary": "Half-duplex allows bidirectional communication but only one side transmits at a time.",
  "subject": "Data Communication",
  "curriculum": "WAEC",
  "level": "SS2",
  "topic": "Modes of Data Transmission",
  "language": "en",
  "max_questions": 3,
  "question_types": ["short_answer", "mcq", "true_false"]
}
```

Required: `teaching_event_id`, `student_id`, `concept_id`, `concept_description`, `taught_summary`, `subject`. `taught_summary` is capped at 2,000 characters. `curriculum`, `level`, `topic` default to empty. `language` defaults to `"en"`. `max_questions` defaults to 3, accepts 1-5. `question_types` defaults to `["short_answer"]`; allowed values are `mcq`, `short_answer`, `true_false`.

Response to FSE:

| Field | Type | Meaning |
| --- | --- | --- |
| `teaching_event_id` | string | Echo of the request's ID |
| `generated_count` | int | Number of questions we generated and attempted to submit |
| `webhook_attempted` | bool | `true` if we called FSE's webhook, `false` on generation failure or missing config |
| `webhook_status` | enum | `processed` \| `accepted_async` \| `failed_permanent` \| `failed_transient` \| `not_attempted` |
| `webhook_results` | array | FSE's `results[]` array echoed verbatim when `webhook_status` is `processed` or `accepted_async`; empty list otherwise |
| `needs_review` | bool | `true` on any degraded state; `false` only when `webhook_status="processed"` AND all questions returned a non-rejected outcome AND at least one result present |
| `error_detail` | string \| null | Short sanitized reason on degraded states; never includes URL, secret, signature, or response body |

`needs_review` semantics for FSE:

- `false`: safe to treat results as canonical. Use `webhook_results` to attach FSRS cards.
- `true` with `webhook_status="processed"` and results present but one or more `rejected_invalid`: FSE processed the batch but rejected some questions. Use only the non-rejected results; do not retry the batch.
- `true` with `webhook_status in {"failed_permanent", "failed_transient"}`: our submission to your webhook failed. Question IDs not available. Consider re-firing the endpoint after investigating (same `teaching_event_id` gets a new `Idempotency-Key` value only if you generate a new teaching event; same event reuses the same key).
- `true` with `webhook_status="not_attempted"`: generation failed before submission. No questions exist. Retry after a minute or investigate our logs.

#### Production smoke test handoff (16 November 2026)

- The first smoke test exposed missing canonical answers on generated short-answer questions. The Gemini prompt now explicitly requires a non-empty `correct_text`, and later tests generated two questions and attempted delivery.
- The initial webhook `401` came from the receiver's auth-header timestamp parser rejecting six fractional digits before HMAC verification. The sender now emits UTC timestamps with millisecond precision. The FSE also deployed support for microsecond precision while preserving freshness checks.
- Delivery `teach_test-e2e-2026-11-16-016` used a millisecond UTC header and received HTTP `400 INVALID_PAYLOAD`. The FSE correlation ID is `ff40a328-54df-48f6-aa32-13ff925e98c0`. The separate `timestamp_outside_window` entry was an intentional stale-timestamp probe.
- Before retrying, get the failing field path and validation rule from the FSE for that correlation ID. One candidate is the JSON body's separate `occurred_at` field, which can still carry six fractional digits. Its role in this rejection is unconfirmed.
- After resolving the schema issue, run one smoke test with a fresh teaching event ID. `processed` with usable results confirms end-to-end processing. `accepted_async` confirms delivery and leaves processing pending.

### POST /generate-hint

Generates one Assistance Ladder hint at a caller-specified level. This endpoint completes the adaptive teaching loop: when a student gets a practice question wrong and asks for help, the FSE calls `/generate-hint` at the appropriate level for that turn, Stewart produces the hint content, and the FSE shows it to the student.

**Model A (locked):** FSE owns escalation policy. The FSE decides which level to request on each turn based on the student's attempt, previous hints, and explicit requests. This endpoint generates the hint content at exactly the level requested. It does not track ladder state, does not decide when to escalate, does not persist attempts, and does not maintain "highest level reached."

The six levels follow the founder doc's Assistance Ladder:

| Level | `hint_type` | Purpose |
| --- | --- | --- |
| L1 | `L1_METACOGNITIVE` | Ask a question that helps the student inspect their approach |
| L2 | `L2_DIRECTIONAL` | Point toward a useful next step or method |
| L3 | `L3_CONCEPT_REMINDER` | Recall a relevant principle, formula, or lecture-note reference |
| L4 | `L4_ANALOGOUS_EXAMPLE` | Show a structurally similar but different problem |
| L5 | `L5_PARTIAL_SOLUTION` | Reveal part of the current problem's solution |
| L6 | `L6_FULL_SOLUTION` | Provide and explain the full solution |

Use the same `Authorization: Bearer <AI_SERVICE_API_KEY>` header as `/explain`.

Request:

```json
{
  "concept_id": "concept-4",
  "concept_description": "The classification of data flow directions between devices, categorized as simplex, half-duplex, and full-duplex.",
  "question": "Which mode of data communication allows two devices to send and receive data in both directions, but not at the same time?",
  "expected_answer": "Half-duplex",
  "evaluation_criteria": [
    "Must identify the mode as half-duplex",
    "Must distinguish half-duplex from full-duplex by simultaneity"
  ],
  "student_attempt": "Full-duplex",
  "target_level": 2,
  "previous_hints": ["Have you considered what the qualifier 'not at the same time' is testing?"]
}
```

Required: `concept_id`, `concept_description`, `question`, `expected_answer`, `student_attempt`, `target_level`. `target_level` is 1-6. `evaluation_criteria` and `previous_hints` default to `[]`.

Response:

| Field | Type | Meaning |
| --- | --- | --- |
| `level_given` | int | Echo of `target_level` on success; `-1` on degraded fallback |
| `hint_type` | enum | `L1_METACOGNITIVE` \| `L2_DIRECTIONAL` \| `L3_CONCEPT_REMINDER` \| `L4_ANALOGOUS_EXAMPLE` \| `L5_PARTIAL_SOLUTION` \| `L6_FULL_SOLUTION` \| `NO_HINT` |
| `hint_content` | string | The student-facing hint text; empty on degraded fallback |
| `rationale` | string | Tutor-facing one-sentence explanation of what this hint targets; not shown to the student |
| `needs_review` | bool | `false` on success; `true` on degraded fallback |

`needs_review` is a required field with no default. It is `false` only when a hint was successfully generated at the requested level. Any failure returns `true` with `level_given=-1`, `hint_type="NO_HINT"`, empty `hint_content`, and `rationale="Hint generation is temporarily unavailable."` The degraded path never returns 5xx.

FSE integration notes:

- **Pass `previous_hints` accurately.** The prompt tells the model not to repeat a prior hint in wording or substance. If you omit this list, or include only the current hint, the model may regenerate material the student has already seen. Send every hint shown in this attempt's ladder history.
- **Do not call at a new level until the student has responded to the current one.** The endpoint itself imposes no constraint — you can call L6 immediately — but pedagogically, escalation without a student attempt is not a hint, it is a solution reveal disguised as help. FSE enforces the timing.
- **L5 does not guarantee the answer is hidden.** Levels 1-5 rely on prompt instructions, not schema validation, to prevent premature reveal. L5 is the highest-risk level for boundary leakage. In practice the smoke test showed a clean L5 (eliminated the wrong answer, used analogy, left the identification to the student), but the boundary is prompt-enforced. If leakage is observed in the wild, tighten the level rule in `services/hint_generation.py` — one-line prompt change, no code change.
- **L4 generates a similar problem, not a hint about this problem.** Level 4 is qualitatively different from L1-L3 and L5-L6: it presents a *different* problem to illustrate the approach. The FSE may want to render L4 differently (visually distinct, or as a sub-panel) rather than inline with the other hint levels.
- **`rationale` is not shown to students.** It is a tutor-facing note about what the hint targets, useful for logging or instructor review. Render only `hint_content` to the student.

Status codes: `200` on success or degraded fallback; `401` for missing or incorrect Bearer token; `422` if `target_level` is outside 1-6; `503` only when no LLM provider is configured. No `404`, no `500`.

Expected latency: 3-8 seconds per hint.

### Demo flow: lesson planning → teaching

1. The lecturer submits a document to `/ingest`. The worker chunks and embeds it.
2. After ingestion completes, the FSE backend calls `/analyse-material` with the
   `document_id` and receives ordered concepts with prerequisite links.
3. The FSE frontend renders the plan as a course structure with per-concept entry points.
4. The student opens a concept. The backend calls `/explain` with the plan's
   `concept_id` and concept description as `concept_description`.
5. Stewart answers for the selected concept with Markdown structure and LaTeX
   where relevant. The response's `sources` array carries citation metadata.
6. From the concept entry, the FSE calls `/generate-example-tasks` with the
   concept's `concept_id`, its `description` as `concept_description`, and
   the `document_id`. The response includes a worked example and ordered tasks.
7. The FSE renders the example and presents the tasks one at a time.
   `expected_answer` and `evaluation_criteria` stay server-side until the
   student submits, then drive immediate feedback.
8. Optionally, when the student is ready to demonstrate independent
   understanding, the FSE calls `/generate-practice` with the concept and
   a target difficulty. The response provides complete questions; the FSE
   reveals only `question_text` (and `options` for multiple choice) to the
   student and keeps `expected_answer` and `explanation` server-side.
9. As the student submits each answer, the FSE calls `/evaluate-practice`
   with the plan's `concept_id` and `concept_description`, the task's
   `prompt` as `question`, its `expected_answer`, `evaluation_criteria`,
   and `response_type`, plus the student's `student_answer` and optional
   `student_reasoning`. The FSE displays `answer_feedback`,
   `reasoning_feedback`, and any diagnosed misconceptions. When
   `needs_review` is `true`, flag the attempt for review and pause automatic
   routing. Otherwise, use `recommended_next_step` to select the next step.

Note: `/generate-revision-questions` is built and tested against stubs but is not part of the current demo walkthrough. Its end-to-end flow requires FSE's `/webhooks/tutor/questions` handler to be live. The demo uses `/generate-practice` and `/evaluate-practice` for the teaching loop.

**Adaptive teaching loop with hints.** If the student gets a practice question wrong, the FSE calls `/evaluate-practice` to grade the attempt, then `/generate-hint` at level 1 to offer a metacognitive prompt. If the student retries and is still stuck, the FSE calls `/generate-hint` at level 2, passing the level-1 hint in `previous_hints`, and so on through L6. After any successful attempt, `/evaluate-practice` grades again to record evidence and recommend the next step. This is the full adaptive loop; the current demo shows analyze → explain → guided example → independent practice → evaluation, and hints extend the loop when the student struggles.

### Attachments

1. **Upload a file.** `POST /attachments` uploads a file and returns an `attachment_id`.
2. **Reference the upload.** `POST /explain` references one or more `attachment_id` values in `attachment_ids`.

Attachments expire 6 hours after upload. `/explain` returns 422 for an expired or unknown ID.

#### POST /attachments

Use the same Bearer token as `/explain`. Send `multipart/form-data` with a single
`file` field. The per-file limit is 4 MiB for all file types, including text and images.

Accepted MIME types

- `application/pdf`
- `application/vnd.openxmlformats-officedocument.wordprocessingml.document`
- `application/vnd.openxmlformats-officedocument.presentationml.presentation`
- `text/plain`
- `text/markdown`
- `image/png`
- `image/jpeg`
- `image/webp`

A successful upload returns `200` with
`{attachment_id, filename, mime_type, size_bytes, created_at, expires_at}`.
An oversized file returns `413`. An unsupported MIME type or empty file returns `422`.

#### GET /attachments/{attachment_id}

Returns the same metadata as `POST /attachments`, with `404` after expiry or deletion.

#### DELETE /attachments/{attachment_id}

Returns `204` on deletion or `404` if the attachment is missing or already expired.

#### End-to-end example

1. Upload a local `recursion.txt` file from your backend environment.

   ```bash
   curl -X POST 'https://<RAILWAY_URL>/attachments' \
     -H 'Authorization: Bearer <AI_SERVICE_API_KEY>' \
     -F 'file=@recursion.txt;type=text/plain'
   ```

   Example `200` response

   ```json
   {
     "attachment_id": "8ce28683-406d-4f61-aa75-957c032be16c",
     "filename": "recursion.txt",
     "mime_type": "text/plain",
     "size_bytes": 128,
     "created_at": "2026-11-09T10:00:00Z",
     "expires_at": "2026-11-09T16:00:00Z"
   }
   ```

2. Ask a question and reference the returned ID in `attachment_ids`.

   ```bash
   curl -X POST 'https://<RAILWAY_URL>/explain' \
     -H 'Authorization: Bearer <AI_SERVICE_API_KEY>' \
     -H 'Content-Type: application/json' \
     -d '{
       "question": "What does the base case do in recursion?",
       "course_offering_id": "offering-id",
       "week": 3,
       "attachment_ids": ["8ce28683-406d-4f61-aa75-957c032be16c"]
     }'
   ```

3. Read the answer and its attachment source. This fragment shows the `answer` and `sources` fields.

   ```json
   {
     "answer": "The base case stops a recursive function from calling itself once a stopping condition is met.",
     "sources": [
       {
         "document_id": null,
         "title": "recursion.txt",
         "material_type": "ATTACHMENT",
         "source_type": "attachment",
         "attachment_id": "8ce28683-406d-4f61-aa75-957c032be16c",
         "filename": "recursion.txt",
         "chunk_preview": "A base case is a stopping condition for a recursive function."
       }
     ]
   }
   ```

#### Handling expiry

The `expires_at` field in the upload response is an ISO 8601 timestamp. If a user
drafts a question and returns more than 6 hours later, `/explain` returns
`422 Attachment {id} not found or expired`. The FSE UI should detect this error,
prompt the student to re-attach the file, and call `POST /attachments` again to
obtain a fresh `attachment_id`. Do not cache `attachment_id` past its `expires_at`.

### GET /health

Returns status, provider, version, grounding status, available endpoints, plus `ingest_status`, `worker_status`,
and `worker_count`. In ready Enterprise mode these include `ready`, `running`,
and the configured count. Disabled states identify missing ingestion settings,
Developer mode, or database unavailability. The configured count is not a count
of currently healthy tasks. Startup probes the database with a two-second
application timeout; health does not perform a fresh live LLM request.

The embedded `endpoints` map lists ingestion, document deletion, grounding, and health.

### POST /explain-with-grounding

Header: `Authorization: Bearer <AI_SERVICE_API_KEY>`.
Form field: `question`.

In Swagger at `/docs`, click **Authorize**, enter the `AI_SERVICE_API_KEY`
value without the `Bearer` prefix, and authorize. Swagger attaches the Bearer
header to protected requests. The secret is never prefilled from the server environment.

Uses `Depends(require_shared_secret_auth)` with the same inbound shared secret
as ingestion. Missing or incorrect auth returns 401 when the secret is configured;
an unset or blank `AI_SERVICE_API_KEY` returns 503. Database, callback configuration,
and ingestion readiness do not gate this endpoint. No new env var.
Gemini with Google Search returns the five explanation fields and cited sources.

## FSE Integration Testing Guide

This section is for the AkadVerse backend team. It assumes you have received the staging base URL and the two shared secrets via password manager. The service is backend-only; callers must never expose `AI_SERVICE_API_KEY` to a browser.

### Prerequisites

- [ ] Staging base URL: `<RAILWAY_URL>`
- [ ] `AI_SERVICE_API_KEY` value received
- [ ] `AI_SERVICE_CALLBACK_SECRET` value received (must match this service's value)
- [ ] Your `/api/webhooks/ai-ingest` endpoint deployed and reachable over HTTPS
- [ ] A signed Supabase URL for a test PDF or DOCX

### Verification steps

Commands use Bash-compatible syntax and `jq`. Replace angle-bracket placeholders
with your staging hostname, receiver hostname, secrets, and material values.
In commands, `<RAILWAY_URL>` is the hostname without the `https://` prefix.
Run these from your backend environment.

1. **Confirm the service is healthy (no auth).**

   ```bash
   curl -s 'https://<RAILWAY_URL>/health' | jq
   ```

   Expect `status: "ok"`, `ingest_status: "ready"`, and `worker_status: "running"`.
   If `ingest_status` is not `"ready"`, ingestion is misconfigured on our side.
   Include both `ingest_status` and `worker_status` when reporting to the AI team.

2. **Confirm your callback endpoint accepts a test POST.**

   Before real ingestion, verify your endpoint responds with `{"ok": true}` to
   a legitimate-looking callback body. Send from your own environment, not
   through the AI service:

   ```bash
   curl -i -X POST 'https://<your-app>/api/webhooks/ai-ingest' \
     -H 'Authorization: Bearer <AI_SERVICE_CALLBACK_SECRET>' \
     -H 'Content-Type: application/json' \
     -d '{"document_id": "test-ignore-me", "status": "ingested", "chunk_count": 1}'
   ```

   Expect `200 {"ok": true}` or
   `200 {"ok": true, "ignored": "unknown document_id"}`.
   A 4xx fails permanently immediately. Redirects, timeouts, and responses
   without literal `"ok": true` exhaust after three attempts. Redirects are
   never followed. See Outbound callbacks for the preserved delivery rules.

3. **Submit a real ingestion.**

   ```bash
   curl -i -X POST 'https://<RAILWAY_URL>/ingest' \
     -H 'Authorization: Bearer <AI_SERVICE_API_KEY>' \
     -H 'Content-Type: application/json' \
     -d '{
       "document_id": "your-material-uuid",
       "file_url": "https://<project>.supabase.co/storage/v1/object/sign/...",
       "file_name": "intro.pdf",
       "mime_type": "application/pdf",
       "callback_url": "https://<your-app>/api/webhooks/ai-ingest",
       "metadata": {
         "course_code": "CSC 201", "course_title": "Data Structures",
         "course_offering_id": "offering-uuid", "academic_session": "2025/2026",
         "semester": "First", "start_week": 3, "end_week": 5,
         "weeks": [3, 4, 5], "material_type": "NOTES",
         "title": "Week 3 notes", "document_id": "your-material-uuid"
       }
     }'
   ```

   Expect `202 {"status": "queued", "document_id": "your-material-uuid"}`.
   The 202 means queue admission only. Processing and callback delivery happen
   asynchronously. For DOCX, use its filename and supported DOCX MIME type
   listed under Extraction and processing.

4. **Watch your callback receive the notification.**

   Within approximately 60 seconds for a small test document, your endpoint
   should receive the following. This is an estimate, not a processing deadline;
   document size, queue load, and provider latency can increase it.

   - `Authorization: Bearer <AI_SERVICE_CALLBACK_SECRET>`
   - Success: `{"document_id": "your-material-uuid", "status": "ingested", "chunk_count": N}`
   - Processing failure: `{"document_id": "your-material-uuid", "status": "failed", "error": "..."}`
     (error is capped at 500 characters and path-sanitized)

   Respond `200 {"ok": true}`. Anything else triggers retry or permanent failure
   according to the callback rules above.

5. **Verify and clean up.**

   Verify the received document ID and outcome, then send a delete:

   ```bash
   curl -i -X DELETE 'https://<RAILWAY_URL>/documents/your-material-uuid' \
     -H 'Authorization: Bearer <AI_SERVICE_API_KEY>'
   ```

   Expect `204` if the document existed, `404` if not. Your contract treats both
   as success. Chunks and the job row are removed in one transaction.

### Expected behaviours during integration

- The first callback after deployment may fail if your endpoint is not yet live.
  Retries exhaust in roughly 21 seconds for repeated timeouts, not a guaranteed
  deadline; 4xx fails immediately. The job is still `done` with chunks committed.
  You will not receive a second callback after exhaustion. Resubmit the same
  `document_id` to re-queue with the original payload. Resubmitting currently
  keeps the original signed URL, so if it expired, ask the AI team about URL
  refresh behavior before relying on retry.
- `422 callback_url host not allowed`: your hostname is not in
  `AI_SERVICE_CALLBACK_ALLOWED_HOSTS`. Confirm your hostname to the AI team.
- `503 Ingestion is not configured on this service`: one of the four required
  ingestion-gate environment variables may be missing on our side. Include
  `/health.ingest_status` in the report. Provider/database readiness can also
  cause 503 (see API endpoints).
- `401 Unauthorized`: the token is wrong or your environment is not sending the
  header. Confirm through `/health` (no auth needed) that the service is up.

6. **Reconcile with the status endpoint if callbacks were missed.**

   ```bash
   curl https://<RAILWAY_URL>/documents/your-material-uuid/status \
     -H "Authorization: Bearer <AI_SERVICE_API_KEY>"
   ```

   Poll every 10 seconds for the first 5 minutes after `POST /ingest`. When the
   response status is `done` or `failed`, stop polling: the callback may have
   been missed, but the truth is in this response. `chunk_count > 0` when
   `status="done"` confirms chunks landed. Reconcile before Step 5 cleanup,
   which removes the job and makes status return 404.

7. **Step 7: verify retrieval answers a question about ingested material.**

   After an ingestion completes (`status="done"`), send:

   ```bash
   curl -X POST 'https://<RAILWAY_URL>/explain' \
     -H "Authorization: Bearer <AI_SERVICE_API_KEY>" \
     -H "Content-Type: application/json" \
     -d '{"question": "Explain the main concept from this material", "course_offering_id": "your-offering-uuid", "week": 1}'
   ```

   Use the ingested offering ID and a week included in its metadata. Run this
   before Step 5 cleanup, or ingest a new document if it was already deleted.
   Expect `answer` grounded in the ingested material with `sources` citing the
   `document_id`. If `retrieval_info.chunks_above_threshold == 0` but
   `chunks_available > 0`, the material does not cover the question: route to
   `/explain-with-grounding` as a fallback and show the student the source-type
   difference (course material versus Google Search grounding).

### Revision question bank integration

The `/generate-revision-questions` endpoint requires two environment variables on our deployed service before it will function:

| Variable | Value | Provider |
| --- | --- | --- |
| `FSE_WEBHOOK_URL` | Full URL to your `/webhooks/tutor/questions` endpoint | FSE, out of band |
| `FSE_WEBHOOK_SECRET` | Shared HMAC-SHA256 signing secret | FSE, out of band |

Until both are set on the Railway service, every call returns `503 {"detail": "Revision submission is not configured."}`. This is intentional: the endpoint is inert until configured. Confirm with the AI team when both are provisioned.

#### What we POST to your webhook

We sign and submit to `FSE_WEBHOOK_URL` with HMAC-SHA256. This is a different auth mechanism than our existing ingestion callbacks, which use `Authorization: Bearer <AI_SERVICE_CALLBACK_SECRET>`. Do not reuse the Bearer pattern for this webhook.

Headers:

```
Content-Type: application/json
X-Akadverse-Signature: <hex HMAC-SHA256 of body bytes with FSE_WEBHOOK_SECRET>
X-Akadverse-Timestamp: <UTC ISO-8601, e.g. 2026-11-15T10:42:00.123456+00:00>
Idempotency-Key: teach_<teaching_event_id>
```

Signature verification on your side:

1. Read the raw request body bytes.
2. Compute `hmac.new(FSE_WEBHOOK_SECRET.encode("utf-8"), raw_body_bytes, hashlib.sha256).hexdigest()`.
3. Compare with `X-Akadverse-Signature` using constant-time comparison.
4. Reject if `X-Akadverse-Timestamp` is older than 5 minutes.

Body (matches your spec §6.3):

```json
{
  "event_id": "<fresh UUID we generate per submission>",
  "teaching_event_id": "<echoed from your request>",
  "student_id": "<echoed>",
  "occurred_at": "<our generation time, UTC ISO-8601>",
  "context": {
    "subject": "<echoed>",
    "curriculum": "<echoed>",
    "level": "<echoed>",
    "topic": "<echoed>",
    "concept_id": "<echoed>",
    "concept_title": "<echoed from concept_description>",
    "language": "<echoed>"
  },
  "taught_summary": "<echoed>",
  "questions": [
    {
      "client_ref": "q1",
      "type": "mcq",
      "stem": "...",
      "options": [{"key": "A", "text": "..."}, ...],
      "correct_answer": "A",
      "explanation": "...",
      "difficulty_estimate": 0.5,
      "bloom_level": "understand",
      "tags": []
    }
  ]
}
```

Specific alignment notes:

- `event_id` is our UUID4, fresh per submission. Use it for our-side traceability.
- `occurred_at` is our generation time, not your request time.
- `concept_title` is our internal name; we populate it from your `concept_description` field. Your spec calls the field `concept_title` on the wire, we carry `concept_description` inbound. Same value, two names.
- `client_ref` is `q1`, `q2`, etc., sequential per submission, stable within the retry window.
- `correct_answer` for `true_false` is a Boolean (`true`/`false`), not a string. For `mcq`, it is the option key letter. For `short_answer`, it is the exact correct string.
- For `short_answer` questions, we include `accepted_answers` (list, possibly empty) at the top level of the question object as a sibling to `correct_answer`.

Retries: we attempt the webhook up to 3 times with delays of 0s, 1s, 4s. The body, signature, timestamp, and `Idempotency-Key` are byte-identical across retries. Our retry window is ~5 seconds total, well within your 5-minute freshness check. If your webhook returns the same `Idempotency-Key` twice with the same body, return the original response unchanged per your spec §6.6.

We do not retry on 400, 401, 409, or any other 4xx. We retry on 429, 5xx, timeouts, request errors, and malformed 200/202 bodies.

Response we expect back (your §6.5):

```json
{
  "status": "processed",
  "event_id": "...",
  "results": [
    {
      "client_ref": "q1",
      "question_id": "qst_8a21...",
      "outcome": "created",
      "similarity": null,
      "student_card_id": "card_77c..."
    }
  ]
}
```

We treat these as success:
- HTTP 200 with body `status="processed"` (must be exact string match)
- HTTP 202 with any valid JSON object body

We treat everything else as failure per the retry rules above.

Latency and timeout budget for FSE:

- Our full round-trip: 8-18 seconds typical (Gemini generation 5-15s + your webhook 3s target).
- Worst case: ~30 seconds if Gemini is slow and your webhook retries internally.
- Set the timeout on your call to `/generate-revision-questions` to at least 30 seconds.
- Do not set 3 seconds — that is your webhook target, not the full loop.

End-to-end integration checklist for FSE:

1. Deploy `/webhooks/tutor/questions` with HMAC-SHA256 signature verification and timestamp freshness check.
2. Provision `FSE_WEBHOOK_URL` and `FSE_WEBHOOK_SECRET` on the AI service (Railway env vars).
3. Confirm `question_types` v1 = `mcq`, `short_answer`, `true_false` (drop `fill_blank`, `numeric`, `step` from v1 scope).
4. Set your timeout on calls to `/generate-revision-questions` to 30s minimum.
5. Trigger a manual call with a synthetic `teaching_event_id`. Verify signature passes, timestamp is fresh, body shape matches §6.3, and your response returns `status="processed"` with `results[]` populated.

### Do not

- Expose `AI_SERVICE_API_KEY` to any browser or client-side code.
- Retry `/ingest` automatically on 5xx; contact the AI team first.
- Assume a `done` job implies a delivered callback. Delivery is best-effort with
  bounded retries.

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
and enforces a 60-second download timeout and 20 MB byte limit. These limits belong to ingestion.
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
runs at startup, not periodically. Lifespan creates one shared callback
`httpx.AsyncClient` with `follow_redirects=False` and `trust_env=False`, stores
it on `app.state.callback_client`, and passes it with `CallbackSettings` to
every worker. Shutdown cancels and awaits all workers, closes that client,
then disposes the database engine.

| Status | Current meaning, unchanged by Phase 2e |
|---|---|
| `queued` | Waiting for a worker |
| `processing` | Claimed; attempts incremented |
| `done` | Document chunks replaced and job completion committed |
| `failed` | Processing failed; safe `last_error` stored, at most 500 characters |

Chunk deletion, insertion, and marking the job done occur in one transaction.
A failed insert rolls back replacement so previously stored chunks survive.
Both done and failed finalization use `WHERE id=:job_id AND status='processing'
RETURNING id`. If no row is returned, the worker logs
`job.cancelled_during_processing` and sends no callback or failed update.
The success path also deletes chunks it just wrote for the document in that
transaction. Cancellation is not an additional persistent job status.

Errors use safe stage messages or extraction messages, normalize whitespace,
and reject backslashes, `/home/`, `/tmp/`, `/var/`, or four consecutive
whitespace-separated alphabetic tokens each longer than 20 characters. Rejected messages become
`Processing failed (see server logs)`. Accepted messages are capped once at
500 characters; DB `last_error` and the failed callback reuse the same value.
Failed jobs are not automatically retried; another admission request can requeue
them with the original payload. Worker database failures stop the affected task
and disable ingestion; no automatic worker restart is implemented.

## Outbound callbacks

`services/callback.py` owns `CallbackSettings` and
`post_callback(client, *, url, secret, document_id, status, chunk_count=None,
error=None) -> bool`. The worker uses the stored job payload's `callback_url`.
The client never constructs its own HTTP client or accesses the database.
It rechecks HTTPS, absence of URL credentials, exact host authorization, and
a non-empty secret at delivery time, including for previously queued jobs.

POST authentication is `Authorization: Bearer <AI_SERVICE_CALLBACK_SECRET>`.
Success body: `{"document_id":"lecture-001","status":"ingested","chunk_count":42}`.
Failure body: `{"document_id":"lecture-001","status":"failed","error":"Safe error"}`.
Only the status-appropriate field is included, never both `chunk_count` and
`error`. Callbacks run after the transaction and temp-file cleanup, outside the
processing exception handler. The worker logs the Boolean delivery outcome.

- Success requires HTTP 200 with a JSON object containing literal `"ok": true`.
  Additional fields, including `"ignored":"unknown document_id"`, are allowed.
- Three total attempts use backoff sleeps of one second, then four seconds.
  Each request uses HTTPX's five-second timeout setting; this is not a strict
  overall delivery deadline.
- Retry 5xx, timeouts/connection errors, 3xx, unexpected 2xx, and 200 responses
  with malformed JSON or without literal `"ok": true`.
- Never follow callback redirects, including when a supplied client enables
  them. A 4xx response returns False immediately without retries.
- Exhaustion logs `callback.failed_permanent` and returns False. Authorization
  or body validation failures also return False without sending a request.

Callback service logs contain document ID, hostname only, processing status, attempt,
HTTP status when available, and reason. They do not include the secret,
Authorization header, full URL, or response body. `job.callback_outcome` records
delivery success/failure without changing the committed job or chunks.

Delivery is best effort: there is no durable callback outbox or later redelivery
after exhaustion. A crash after DB commit and before delivery can lose the
notification. Admission-time DNS checks are not repeated or pinned by the
callback client; the required host allowlist is rechecked at delivery time.

## Testing

Before running `uv run pytest`, run `Remove-Item Env:DATABASE_URL` in the same shell.

From this service directory:

```powershell
uv run pytest tests/ -q --tb=short --timeout=30
```

Recorded verification on 27 October 2026, after the developer-approved skip:

| Full-suite run | Result | Pytest duration |
|---|---|---|
| 1 | 162 passed, 1 skipped, 1 warning | 189.51 seconds |
| 2 | 162 passed, 1 skipped, 1 warning | 184.04 seconds |
| 3 | 162 passed, 1 skipped, 1 warning | 193.38 seconds |

All three commands exited successfully. The warning is the existing
Starlette/AnyIO `BlockingPortal` deprecation. None met the requested
under-60-second target. These are recorded test results, not a claim that the
suite is rerun for each documentation edit.

There are 163 cases: 164 baseline cases minus one obsolete API-key case and
two legacy-wrapper cases, plus two grounding auth cases. One case is skipped,
not deselected or deleted.

`tests/test_worker.py::test_multiple_workers_no_collision` has an approved skip
for an intermittent hang at `portal.call(settle)`. Diagnosis reproduced the
timeout in one of three full-suite runs. Root cause is unconfirmed, and no
hanging-run pending-task snapshot was captured. The test remains pending a
dedicated investigation. The passing
`test_workers_share_callback_client_until_shutdown` covers callback-client
sharing and shutdown order with three mocked workers; it does not replace
the skipped test's collision-free concurrent job-claim coverage.

Tests cover provider configuration, authentication, queue admission, embedding
validation, extraction/dispatch, chunk boundaries, startup recovery, worker
client sharing and shutdown, document replacement, safe worker failures,
transactional deletion,
strict callback response handling, retries, redirect refusal, and host
authorization. Worker tests verify DB status at callback time, preserved status
after delivery failure, and cancellation on both success and failure paths.
Cancellation uses `asyncio.Event` barriers rather than timing sleeps.
Worker integration
uses real isolated Postgres inserts, real file extraction, mocked downloads,
and mocked embeddings/callbacks. Delivery tests use `httpx.MockTransport`, not
live webhook requests. The PDF fixture is committed at
`tests/fixtures/sample.pdf`; DOCX/PPTX fixtures are generated programmatically.
The PDF ignore exception is intentional.

The test database must be reachable with pgvector enabled and permissions for
isolated schemas. Tests create unique `concept_explainer_test_<uuid>` schemas
and clean up or roll back their changes. Do not point test writes at production
service tables. `tests/conftest.py` sets the Windows selector policy.

Always use `--tb=short`; traceback locals can expose secrets. If more detail is
necessary, run only the failing test with `--tb=long` and redact before sharing.
Automated tests do not establish live ADC access, model availability, Cloud
quota, or actual callback delivery. Provider tests mock external generation.

## Troubleshooting

- **`/explain` returns "does not cover this question":** material exists for
  that offering/week but no chunk passed the similarity threshold. The question
  may be outside the material's scope or use vocabulary that does not match
  the document. Do not lower the threshold; route to `/explain-with-grounding`.
- **`/explain` returns "No course material has been ingested":** no chunks exist
  for that `course_offering_id` + `week` combination. Verify the ingestion
  metadata's `weeks` array includes the queried week.
- **Railway Deploy Logs show `[error] INFO:` for routine events:** upgrade to
  the committed logging configuration (`2f3baee`) documented in this update;
  logs must route to stdout via `logging.basicConfig(stream=sys.stdout)`.

- **First deploy fails at container start with `RuntimeError: Form data requires "python-multipart"`:**
  a package used at runtime is not in `requirements.txt`. Verify all runtime
  dependencies are listed. Docker layer caching may serve a stale image;
  force a no-cache rebuild.
- **Healthcheck fails for 120s, "service unavailable":** the container is not
  listening on `$PORT` or is bound to `127.0.0.1`. Confirm `run.py` reads `$PORT`
  and binds `0.0.0.0`. Or the container crashed on startup: check Deploy Logs,
  not Build Logs.
- **`callback.failed_permanent` in logs but job is `done` in DB:** expected when
  the FSE's endpoint is not live or returns something other than `200 {"ok": true}`.
  Delivery is best-effort; the job succeeded. Check the receiver, not the AI service.

- **Ingestion unavailable:** check `/health`, Enterprise selection, private
  database/auth configuration, callback secret, required host allowlist, both
  migrations, and database reachability.
- **Callback URL rejected:** check HTTPS, exact hostname membership, and DNS.
  Localhost/127.0.0.1 must be explicitly listed and still require HTTPS. An
  empty allowlist disables ingestion instead of permitting arbitrary hosts.
- **Callback delivery failed:** inspect `callback.attempt`,
  `callback.failed_permanent`, and `job.callback_outcome`. The receiver must
  return HTTP 200 with `{"ok":true}`; redirects are not followed and 4xx is
  not retried. A done job can still have an undelivered callback.
- **Windows Psycopg/Proactor error:** launch through `uv run python run.py`;
  do not switch to a synchronous driver to hide the loop configuration problem.
- **Missing packages:** run `uv pip install -r requirements.txt` in the root uv
  environment. Locally installed Office packages do not replace declared pins.
- **Database test timeout:** check database reachability and rerun the failing
  test in isolation with short tracebacks before changing application code.
- **Expired download URL:** a download `403` records
  `Signed URL expired, retry needed`. Reposting the same ID currently retains
  the old payload; coordinate URL refresh behavior rather than assuming it updates.
- **Empty/scanned/corrupt file:** inspect `jobs.last_error`; there is no OCR fallback.
  Unlike queue validation errors, these failures happen after the `202` response.
- **Embedding failure:** verify access and quota for the configured model and
  location without exposing SDK exceptions or credentials.
- **Port conflict:** stop the conflicting service on 8010 or consult the platform
  registry for an available testing port. Do not reuse another tool's reserved port.
## Project structure

```text
akadverse-concept-explainer/
  Dockerfile                   Container build from service requirements
  .dockerignore                Build exclusions, including .env
  main.py                      FastAPI lifespan, router wiring, health
  run.py                       Windows-compatible launcher, port 8010
  auth.py                      Shared-secret authentication and ingestion readiness
  models.py                    Request and response schemas
  config.py                    Provider/storage settings and worker constants
  db.py                        Async Postgres engine and session factory
  routes/
    explain.py                 Authenticated synchronous POST /explain
    lesson_planning.py         Authenticated POST /analyse-material
    example_tasks.py           Authenticated POST /generate-example-tasks
    practice_evaluation.py    Authenticated POST /evaluate-practice
    attachments.py            Authenticated attachment upload, metadata, and deletion
    ingest.py                  POST /ingest
    documents.py               Document deletion and ingestion status
    grounding.py               Retained Google Search Grounding route
  workers/ingestion.py          Claims, recovery, processing, temp-file cleanup
  services/
    retrieval.py               Course-scoped retrieval and grounded answers
    lesson_planning.py         Structured plans from ingested chunks
    example_tasks.py           Worked examples and interactive tasks
    practice_evaluation.py    Answer and reasoning evaluation
    attachments.py            Attachment storage and expiry sweeps
    chat_history.py           Conversation persistence and retention sweeps
    embedding.py               Enterprise batch embeddings
    callback.py                CallbackSettings and bounded authorized delivery
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
    003_attachments_table.sql
    004_chat_messages_table.sql
  tests/
    conftest.py                Isolated database and generated Office fixtures
    test_analyse_material.py
    test_example_tasks.py
    test_practice_evaluation.py
    test_attachments.py
    test_health.py
    test_status.py
    test_api_key_env.py
    test_auth.py
    test_callback.py
    test_db.py
    test_delete.py
    test_embedding.py
    test_explain.py
    test_extractors.py
    test_gemini_provider.py
    test_grounding.py
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

Next.js supplies a signed file URL and an HTTPS callback URL whose hostname
the operator has explicitly authorized. The receiver validates the shared
Bearer secret and responds with HTTP 200 and `{"ok":true}`, including for an
ignored unknown document. Use DELETE to remove chunks and jobs; treat 204 and
404 as success. Enable PDF/DOCX/PPTX in lecturer integrations and reject ZIP.
`GET /documents/{id}/status` reconciles missed callbacks.

Phase 3 is committed and verified end-to-end on Railway with real INS412 PDF
(28 chunks) and CSC424 PPTX (17 chunks) lecture documents. Retrieval returned
grounded answers, correct citations, and correct not-covered responses.
Callbacks to `https://akadverse.vercel.app/api/webhooks/ai-ingest` returned 200 OK.
Exchange secrets privately. Production deployment remains a separate decision.

Attachments, chat history, AI-generated worked examples, lesson planning,
concept-scoped explanations, interactive micro-tasks, practice evaluation, independent practice
generation, revision question generation (pending FSE webhook deployment), and Assistance Ladder hint generation are complete. The remaining deferred items are the skipped concurrency test
investigation, the durable callback outbox (only when an actual delivery
problem is observed), streaming `/explain` responses, and the full Stewart
teaching roadmap (learner model, assistance ladder L0-L6, FSRS integration)
pending Phase 0 alignment with the FSE and founder. A separate post-demo
cleanup pass will address a latent metadata-read guard in
`services/example_tasks.py`, strengthen the parsed-read test in
`tests/test_practice_evaluation.py`, remove an unused import, and replace
pre-rule `print()` calls in `services/model_discovery.py` and `main.py`.

Plan a Railway
upgrade or migration before the trial expires early December 2026.
