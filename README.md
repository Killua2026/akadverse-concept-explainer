# AkadVerse Concept Explainer

FastAPI microservice for the AkadVerse Study Zone, reserved port **8010**.
The intended caller is the Next.js backend, not a browser client.

## Current status: Phase 2e implemented and committed

Phase 2e retires the legacy explanation endpoints and authenticates Google Search
Grounding with the existing ingestion shared secret. Durable ingestion, document
deletion, and callbacks remain unchanged. The local Phase 2e commit is
`54a6ecb`. Three consecutive full-suite runs each recorded 162 passed and one
intentionally skipped concurrency test. Remote merge and deployment are not
verified. Retrieval is planned for Phase 3 and has not started.

| Capability | Current state |
|---|---|
| Authenticated `POST /ingest` | Implemented; returns `202` after queue admission |
| Durable jobs and configurable workers | Implemented; startup recovery and concurrent claims |
| PDF / DOCX / PPTX extraction | Implemented |
| Shared chunking and Enterprise embeddings | Implemented |
| Delete endpoint and outbound callback client | Implemented; transactional deletion and bounded post-commit delivery |
| Legacy endpoint retirement and grounding auth | Implemented and committed in Phase 2e |
| Concurrent job-claim collision test | Retained but skipped; dedicated investigation pending for intermittent teardown hang |
| Retrieval over ingested pgvector chunks | Phase 3; not implemented |
| Stewart chat history | Phase 4; not implemented |
| ZIP ingestion | Rejected; deferred to a later phase |

`/ingest` writes pgvector chunks. Retrieval over these chunks is not implemented.
Callbacks are posted after processing commits. A job marked `done` means its
chunks were stored, not that a callback was delivered. Delivery failure does
not change the committed job status or roll back chunks.

## Contents

- [Architecture](#architecture)
- [Installation and startup](#installation-and-startup)
- [Configuration](#configuration)
- [Database setup](#database-setup)
- [API endpoints](#api-endpoints)
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
Use [Swagger UI](http://127.0.0.1:8010/docs) to inspect the four current API
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
| `AI_SERVICE_API_KEY` | Shared secret for ingestion, deletion, and grounding authentication |
| `AI_SERVICE_CALLBACK_SECRET` | Required Bearer secret for outbound callbacks |
| `AI_SERVICE_CALLBACK_ALLOWED_HOSTS` | Required comma-separated, exact callback hostnames; no wildcard matching |
| `WORKER_COUNT` | Positive integer, default `1`; restart required after changes |

Developer mode disables ingestion and workers; grounding shares the ingestion readiness gate. `/ingest` returns `503` with an Enterprise-mode requirement.
A Cloud project setting alone does not select Enterprise mode. Unknown providers,
missing required provider credentials/settings, and invalid worker counts fail
startup validation.

In Enterprise mode, ingestion requires non-empty `DATABASE_URL`,
`AI_SERVICE_API_KEY`, `AI_SERVICE_CALLBACK_SECRET`, and
`AI_SERVICE_CALLBACK_ALLOWED_HOSTS`. Missing settings report
`disabled (missing VARIABLE_NAME)` through `ingest_status`; `/ingest` returns
503 and workers do not start. An empty or comma/whitespace-only allowlist is
missing configuration. Grounding shares the same configured/ready-state auth checks and returns 503 when ingestion is unavailable.

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

Uses `Depends(require_ingest_auth)`, the same shared secret and readiness checks
as ingestion. Missing or incorrect auth returns 401 when ingestion is ready;
missing configuration or unavailable ingestion returns 503. No new env var.
Gemini with Google Search returns the five explanation fields and cited sources.

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
  main.py                      FastAPI lifespan, router wiring, health
  run.py                       Windows-compatible launcher, port 8010
  auth.py                      Ingestion shared-secret authentication
  models.py                    Ingestion request/response schemas
  config.py                    Provider/storage settings and worker constants
  db.py                        Async Postgres engine and session factory
  routes/
    ingest.py                  POST /ingest
    documents.py               Transactional DELETE /documents/{document_id}
    grounding.py               Retained Google Search Grounding route
  workers/ingestion.py          Claims, recovery, processing, temp-file cleanup
  services/
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
  tests/
    conftest.py                Isolated database and generated Office fixtures
    test_api_key_env.py
    test_auth.py
    test_callback.py
    test_db.py
    test_delete.py
    test_embedding.py
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
There is still no job-status HTTP endpoint.

Phase 2e is implemented and committed locally in `54a6ecb`. Before deployment, verify
the real receiver, TLS/DNS configuration, both shared secrets, allowlisted
hosts, ADC/model access, and quotas without exposing credentials. Automated
tests do not establish those live integration prerequisites or remote merge
status. Rotate previously exposed secrets and exchange replacements privately.

Verify Phase 2e remote merge status before the next implementation phase.
Investigate and restore the skipped concurrency test separately. Phase 3 adds
pgvector retrieval and requires its own explicit scope and authorization;
Phase 4 adds Stewart chat history. Neither phase is implemented. ZIP support
remains deferred.
