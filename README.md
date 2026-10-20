# AkadVerse Concept Explainer
### Tier 5 Learning AI Tool | Microservice Port: `8010`

> A RAG-powered study assistant that explains academic concepts in two modes:
> from your own uploaded lecture notes, or from live web sources via
> Google Search Grounding. Produces structured pedagogical responses
> covering plain-language explanation, analogy, example, and worked problem.

---

## Table of Contents

1. [What This Microservice Does](#what-this-microservice-does)
2. [The Two Explanation Modes](#the-two-explanation-modes)
3. [Architecture Overview](#architecture-overview)
4. [Prerequisites](#prerequisites)
   - [Port Registry](#port-registry)
5. [Getting Your API Key](#getting-your-api-key)
   - [Supabase Setup](#supabase-setup)
6. [Configuration](#configuration)
7. [Installation](#installation)
8. [Running the Server](#running-the-server)
9. [API Endpoints](#api-endpoints)
   - [POST /upload-note](#1-post-upload-note)
   - [POST /explain-concept](#2-post-explain-concept)
   - [POST /explain-with-grounding](#3-post-explain-with-grounding)
   - [GET /health](#4-get-health)
10. [Testing with Swagger UI](#testing-with-swagger-ui)
11. [Example Test Inputs](#example-test-inputs)
12. [Understanding the Responses](#understanding-the-responses)
13. [Choosing the Right Mode](#choosing-the-right-mode)
14. [Generated Files](#generated-files)
15. [Common Errors and Fixes](#common-errors-and-fixes)
16. [Project Structure](#project-structure)
17. [Part of the AkadVerse Platform](#part-of-the-akadverse-platform)
18. [Integration Notes for Full Stack Engineer](#integration-notes-for-full-stack-engineer)

Additional Phase 2a sections:

- [Current Status: Phase 2a](#current-status-phase-2a)
- [Postgres Storage](#postgres-storage)
- [Database Migrations](#database-migrations)
- [Automated Tests](#automated-tests)

---

## What This Microservice Does

This service is a **Tier 5 component** of the AkadVerse AI-first e-learning
platform. It serves the *Study Zone* and acts as a personal
AI tutor for students.

The existing endpoints below describe the legacy runtime, which still uses
FAISS. Phase 2a adds Postgres storage independently; requests do not yet use it.
The intended integration is through the Next.js backend, not direct browser calls.

A student can:

- **Upload their lecture notes as a PDF** and ask questions about the specific
  content their lecturer taught. The AI answers strictly from those notes and
  honestly says "I do not know" rather than hallucinating if the answer is not
  in the document.
- **Ask any academic question without uploading anything** and receive a
  structured explanation sourced from the live web, complete with citations
  so they can verify and explore further.

Every response, regardless of mode, is structured into five fields:
the concept, a plain-language explanation, a
relatable analogy, a concrete example, and a step-by-step worked problem.

### Current Status: Phase 2a

Storage foundations are implemented and the migration has been applied and
verified in the development Supabase project. The last verified suite result
was five passing tests. No endpoint was changed as part of Phase 2a.
`concept_explainer.py` still runs the FAISS-backed upload and explanation routes.
`config.py` and `db.py` are independent storage modules, not wired into FastAPI's
lifespan yet. The jobs table and ingestion worker are deferred to Phase 2b;
student retrieval is not implemented. See `AGENTS.md` for the phased roadmap.

---

## The Two Explanation Modes

| | Mode A: Notes-Based RAG | Mode B: Search Grounding |
|---|---|---|
| **Endpoint** | `POST /explain-concept` | `POST /explain-with-grounding` |
| **Source of answers** | Student's uploaded PDF | Live Google Search |
| **PDF required?** | Yes | No |
| **Returns citations?** | No (answers from your own notes) | Yes (URLs to web sources) |
| **If concept not in notes** | Honestly says "I do not know" | Searches the web for the answer |
| **Best for** | Exam prep from lecturer's material | Filling gaps, deeper understanding |

Both modes return the same five-field pedagogical structure. They are
designed as complements - use Mode A to study from your lecturer's
exact material, then use Mode B to fill any gaps the notes do not cover.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                     POST /upload-note                          │
│  PDF → PyPDFLoader → Chunker (1000 chars) → Gemini Embeddings  │
│                         → FAISS Index (saved to disk)          │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                    POST /explain-concept                        │
│  Question → Load FAISS (from disk if needed) → Top-3 Retrieval │
│           → Gemini LLM (no web access) → ExplanationResponse   │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                 POST /explain-with-grounding                    │
│  Question → Gemini + Google Search Tool (live web retrieval)   │
│           → Extract JSON block → GroundedExplanationResponse   │
│           → + cited sources from grounding metadata            │
└────────────────────────────────────────────────────────────────┘
```

Phase 2a also provides a separate storage path: process environment to
`StorageSettings`, then `Database`, then Supabase Postgres. The existing HTTP
flows shown above do not call this path yet.

**Key design decisions:**

- The FAISS index is **saved to disk** after every upload and reloaded
  automatically on first use after a server restart. Students never need
  to re-upload their notes between sessions.
- Uploaded PDFs are saved with a **UUID-prefixed temp filename** so two
  concurrent uploads of files with the same name cannot corrupt each other.
- **File size is checked before any processing begins.** Files over 50 MB
  are rejected immediately with a clear 413 error.
- The generative model is discovered via `client.models.list()` when needed.
  Legacy embedding code tries a hardcoded candidate list; it does not discover
  embedding models dynamically.
- Search Grounding and `with_structured_output()` **cannot run together**
  in one API call. The grounding endpoint solves this by prompting Gemini
  to embed a JSON block in its free-text response, which is then extracted
  by a regex-based helper with three fallback strategies.

---

## Prerequisites

- **Python 3.12**, matching the repository's root `pyproject.toml`
- **uv**, the Python package and project manager
- A **Google Gemini API key** (one key covers both modes - free tier available)
- Internet connection (for embedding calls and grounding search)
- A **Supabase Postgres project** with pgvector enabled for storage and DB tests
- Database credentials permitted to create an isolated test schema and roll it back

> **Windows users:** All commands below work in the VS Code integrated
> terminal or Windows PowerShell. Use `python` instead of `python3` if needed.

### Port Registry

| Port | Service | Notes |
|---|---|---|
| 8010 | Concept Explainer (this service) | Fixed port. Do not change without updating `AGENTS.md` |
| 8009 | Slide Generator | Different microservice. Never use as a fallback |
| 8008 | Notes Creator | Different microservice |
| 8011 | Resource Puller | Different microservice |
| 8000 | Orchestrator | Routes requests to Tier 5 services |

This is a subset of the platform registry. Check the root `AGENTS.md` before
choosing an unregistered local testing port.

---

## Getting Your API Key

1. Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with a Google account.
3. Click **Create API Key**.
4. Copy the key and provide it to the service through the `GOOGLE_API_KEY`
  environment variable.

> The free tier includes access to Gemini 2.5 Flash, sufficient embedding
> quota for lecture-sized PDFs, and Search Grounding support.

> **Important for Search Grounding:** `/explain-with-grounding` uses
> Google's native Search Grounding tool. This is available on the free tier
> but may require billing to be configured on your Google Cloud account in
> some regions. If you receive a `400` or `403` error on the grounding
> endpoint specifically, check your quota settings at
> [https://aistudio.google.com](https://aistudio.google.com).

## Supabase Setup

### (a) Create the project

Create a project at [database.new](https://database.new). Phase 2a pgvector
storage lives in Postgres. Free-tier projects pause after one week of
inactivity; if connections fail later, resume the project from the dashboard.
See [Supabase plan details](https://supabase.com/pricing).

### (b) Enable pgvector before running the migration

Use this dashboard-first setup procedure for this project:

> On hosted Supabase, `CREATE EXTENSION IF NOT EXISTS vector` in a migration often fails with `permission denied to create extension "vector"` because extension installation is restricted to the dashboard on hosted projects. Enable it from the UI first:
>
> 1. In the Supabase dashboard, go to **Database** in the left sidebar.
> 2. Click **Extensions**.
> 3. Search for `vector`.
> 4. Toggle it on and wait a few seconds for it to go green.
>
> Once enabled, the `IF NOT EXISTS` clause in `migrations/001_pgvector_storage.sql` sees it and skips cleanly. If you skip this step, the migration may error out on the `CREATE EXTENSION` line even though nothing is wrong with the file.

Permissions depend on the database role and extension. The quoted setup
guidance is not a universal dashboard-only restriction: Supabase also
documents SQL-based extension installation for authorized roles. Follow
the dashboard procedure here to establish the prerequisite before migration.
See [Supabase extensions](https://supabase.com/docs/guides/database/extensions).

### (c) Copy the connection string

Click **Connect**, choose **Session pooler**, and copy that URL on port 5432.
Use it for this service's local development setup. Direct connections use
IPv6 by default and can fail on IPv4-only home networks unless the project
has the IPv4 add-on. Session pooling supports IPv4 and prepared statements.
Transaction pooling on port 6543 does not support prepared statements unless
the client disables them; it is not the connection mode selected here.
Copy the actual pooler hostname from the dashboard rather than constructing it.
See [Supabase connection modes](https://supabase.com/docs/guides/database/connecting-to-postgres).

### (d) DATABASE_URL format

Store a bare Postgres URL in the service's ignored `.env` file. The following
is a format illustration containing placeholders, not usable credentials:

```dotenv
DATABASE_URL=postgresql://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:5432/postgres
```

Use `postgresql://` in `.env`, not `postgresql+psycopg://` or
`postgresql+asyncpg://`. Internally, `db.py` applies
`url.set(drivername="postgresql+psycopg")` when constructing the SQLAlchemy
async engine. This changes the engine URL, not the environment variable.
It explains the different scheme shown in SQLAlchemy diagnostics; do not
share diagnostics containing credentials.

### (e) Password encoding note

URL-encode the password component if it contains `@`, `:`, `/`, `?`, `#`,
`[`, `]`, or `%` before inserting it into the URL. Use `urllib.parse.quote_plus`
for those characters; `urllib.parse.quote(password, safe="")` also handles
spaces correctly in URI user information. Encode only the password, not the
whole URL, and do not encode it twice. A long, randomly generated alphanumeric
password avoids this encoding issue when resetting credentials.

### (f) Verification before migration

Run this query in the Supabase SQL editor before applying the migration:

```sql
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
```

One row means pgvector is enabled and ready. Zero rows means step (b) is
still required; enable the extension from the dashboard and verify again
before proceeding.

## Installation

### Step 1 - Set up your project folder

Use the existing repository checkout, including its storage modules, migrations,
and tests. Run the following commands from `akadverse-concept-explainer/`:

```
cd akadverse-concept-explainer
```

### Step 2 - Create a virtual environment (recommended)

```bash
# Synchronize the root uv project and its Python 3.12 environment
uv sync
```

### Step 3 - Install dependencies

```bash
uv pip install -r requirements.txt
```

> **Why `google-genai>=1.67.0` specifically?** The Search Grounding tool
> uses `types.Tool(google_search=types.GoogleSearch())` which requires
> the unified SDK at version 1.67.0 or above. Older versions do not have
> this API and will throw an `AttributeError`.

Full dependency reference:

| Package | Min Version | Purpose |
|---|---|---|
| `fastapi` | latest | Web framework for the API |
| `uvicorn` | latest | ASGI server to run FastAPI |
| `google-genai` | 1.67.0 | Gemini SDK: generation, embeddings, and Search Grounding |
| `langchain-google-genai` | latest | LangChain wrapper for Gemini structured output |
| `langchain-community` | latest | FAISS vector store integration |
| `langchain-text-splitters` | latest | Recursive character text splitter for PDF chunking |
| `faiss-cpu` | latest | Local vector similarity search index |
| `pydantic` | latest | Data validation and response schemas |
| `pypdf` | latest | PDF text extraction backend for PyPDFLoader |
| `sqlalchemy[asyncio]` | pinned: 2.1.4 | Async engine and session factory |
| `psycopg[binary]` | pinned: 3.2.6 | Postgres async driver and binary distribution |
| `python-dotenv` | pinned: 1.0.1 | Test environment loading |

The root uv project supplies pytest and httpx. This service's additional
dependencies are recorded in `requirements.txt`; install that file into the
root environment after synchronizing. Do not assume `uv run` installs a
service-level requirements file automatically.

---

## Running the Server

From inside your project folder with the virtual environment activated:

```bash
uvicorn concept_explainer:app --host 127.0.0.1 --port 8010 --reload
```

**Expected startup output (first run, no prior uploads):**

```
[Startup] AkadVerse Concept Explainer initializing...
[Startup] FAISS index path: 'akadverse_concept_faiss_index'
[Startup] Max upload size: 50 MB
[Startup] No existing index found - awaiting first PDF upload.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8010 (Press CTRL+C to quit)
```

**Expected startup output (after a previous upload):**

```
[Startup] AkadVerse Concept Explainer initializing...
[Startup] FAISS index path: 'akadverse_concept_faiss_index'
[Startup] Max upload size: 50 MB
[Startup] Existing FAISS index found on disk - will load on first request.
INFO:     Application startup complete.
```

The second message confirms FAISS persistence is working - the index from
your last session is on disk and will be loaded automatically when a student
asks their first question, without requiring a re-upload.

Keep this terminal open while testing. The `--reload` flag restarts the
server automatically when you edit the code.

---

## Configuration

| Variable | Current use |
|---|---|
| `GOOGLE_API_KEY` | Required by the existing application's startup validation |
| `DATABASE_URL` | Required when `StorageSettings.from_env()` is called for storage or tests |
| `AI_SERVICE_API_KEY` | Planned inbound ingestion authentication; not wired in Phase 2a |
| `AI_SERVICE_CALLBACK_SECRET` | Planned outbound callback authentication; not wired in Phase 2a |

`tests/conftest.py` calls `load_dotenv()` at import time. Tests therefore load
the ignored service `.env` automatically; existing process variables take
precedence. Neither `config.py` nor `concept_explainer.py` loads `.env` itself.
For the current application, provide `GOOGLE_API_KEY` in the process environment
before startup, or use Uvicorn's environment-file option with the same app and
port. `DATABASE_URL` alone does not make the HTTP routes use Postgres.

The service reads the Gemini key once when `concept_explainer.py` is imported.
The key is never accepted in an HTTP request, stored in the FAISS index, or
shown in Swagger UI. Set it before starting Uvicorn.

**Windows PowerShell, current terminal only:**

```powershell
$env:GOOGLE_API_KEY = "your-gemini-api-key"
```

**Windows Command Prompt, current terminal only:**

```bat
set GOOGLE_API_KEY=your-gemini-api-key
```

**Persist for future Windows terminals:**

```bat
setx GOOGLE_API_KEY "your-gemini-api-key"
```

After using `setx`, open a new terminal before starting the service. Startup
fails immediately with a clear `RuntimeError` if the variable is missing or
empty. Do not commit API keys to source control, README files, or `.env`
files that are not excluded from version control.

---

## Postgres Storage

Phase 2a introduces `config.py` and `db.py` independently of the HTTP routes.
`StorageSettings.from_env()` validates that `DATABASE_URL` is present and
nonempty, stores it as a `SecretStr`, and excludes it from the settings repr.
`Database` creates an async SQLAlchemy engine and session factory, uses
`NullPool`, hides SQL parameter values, disables automatic prepared statements,
and provides async engine disposal. Hidden SQL parameters do not guarantee
that every traceback is free of secrets.

The service owns the `concept_explainer` schema in the same Postgres instance
as AkadVerse. It owns no other service's tables. The Phase 2a `chunks` table is:

| Column | Postgres type | Constraint |
|---|---|---|
| `id` | `BIGSERIAL` / `bigint` | Primary key |
| `document_id` | `TEXT` | Not null |
| `content` | `TEXT` | Not null |
| `embedding` | `vector(768)` | Not null |
| `metadata` | `JSONB` | Not null |
| `created_at` | `TIMESTAMPTZ` | Not null, defaults to `now()` |

Indexes are `chunks_embedding_hnsw_idx` (HNSW with cosine distance),
`chunks_document_id_idx`, and `chunks_course_offering_id_idx` (the
`metadata->>'course_offering_id'` expression), plus `chunks_pkey`.
The jobs table is not created by Phase 2a. FAISS data is not migrated by this SQL.

## Database Migrations

Schema changes are numbered SQL files in `migrations/`, applied manually.
`001_pgvector_storage.sql` creates the schema, extension prerequisite, chunks
table, and indexes using `IF NOT EXISTS`. Reapplying it safely skips existing
objects; it does not reconcile an existing table with an incompatible definition.

Run the complete file as a single transaction using `psql -f` with a privately
configured connection or paste the whole block into the Supabase SQL editor.
Keep its `BEGIN`, `SET LOCAL search_path TO public, extensions`, and `COMMIT`
together. Do not run statements one at a time: the local search path would not
persist and `vector(768)` could fail to resolve. Do not put real credentials
in commands copied into chat or logs.

After application, confirm the extension exists, all six columns have the
types above, and all three named indexes plus the primary key exist. Applying
this file does not enable `/ingest`, callbacks, or student retrieval.

## API Endpoints

### 1. `POST /upload-note`

> **Legacy. Being replaced by `POST /ingest` in Phase 2b. Do not build new integrations against this endpoint.**

**What it does:** Accepts a PDF lecture note, splits it into 1000-character
overlapping chunks, generates embeddings using Gemini, and stores everything
in a FAISS vector index on disk. This must be called before using
`/explain-concept`.

**Form fields:**

| Field | Required | Description | Example |
|---|---|---|---|
| `file` | Yes | A PDF file to index (max 50 MB) | `lecture_notes.pdf` |

**Success response (200 OK):**

```json
{
  "status": "success",
  "message": "'lecture_notes.pdf' has been processed and indexed. You can now ask questions about its content.",
  "filename": "lecture_notes.pdf",
  "chunks_embedded": 64,
  "pages_loaded": 64
}
```

**File too large (413):**

```json
{
  "detail": "File too large: 72.3 MB. Maximum allowed size is 50 MB. Please upload a focused set of lecture notes rather than a full textbook."
}
```

**Non-PDF file (400):**

```json
{
  "detail": "Only PDF files are supported."
}
```

**Expected terminal output during upload:**

```
[Upload] Processing 'lecture_notes.pdf' (1470.0 KB)...
[Upload] Loaded 64 pages from 'lecture_notes.pdf'.
[Upload] Split into 64 chunks. Starting embedding...
[Embedding] Trying model: models/gemini-embedding-001
[Embedding] 11/64 chunks embedded...
[Embedding] 21/64 chunks embedded...
...
[FAISS] Index saved to 'akadverse_concept_faiss_index'.
[Embedding] All 64 chunks embedded successfully with models/gemini-embedding-001.
[KAFKA MOCK] Published event 'note.uploaded'. Document indexed for student session.
[Upload] Temp file 'a3f1b2c4_lecture_notes.pdf' cleaned up.
```

> **Note on upload time:** Embedding is rate-limited to 10 chunks per batch
> with an 8-second pause between batches to stay within the Gemini API limit
> of 100 requests per minute. A 64-chunk PDF takes approximately 55 seconds.
> A progress log prints after every batch so you know it has not frozen.

> **Note on the UUID prefix:** The `a3f1b2c4_` prefix in the temp filename
> log is a randomly generated 8-character identifier that prevents filename
> collisions when two users upload files with the same name simultaneously.

---

### 2. `POST /explain-concept`

> **Legacy. Being replaced by a student retrieval endpoint in Phase 3 (not yet implemented). Do not build new integrations against this endpoint.**

**What it does:** Answers a question using only the content from your uploaded
PDF. Retrieves the three most relevant chunks from the FAISS index and
generates a structured pedagogical explanation. If the concept is not
covered in the notes, it says so rather than hallucinating.

Requires a prior call to `/upload-note`. If the server was restarted since
the last upload, the index is loaded from disk automatically - no
re-upload needed.

**Form fields:**

| Field | Required | Description | Example |
|---|---|---|---|
| `question` | Yes | The concept or question to explain | `Explain the stepping-stone method` |

**Success response (200 OK):**

```json
{
  "concept": "Stepping-Stone Method for Transportation Problems",
  "simple_explanation": "The Stepping-Stone Method is a technique used to optimise transportation problems...",
  "analogy": "Imagine you're finding the quickest route through a series of islands to deliver packages...",
  "example": "I do not know based on the provided notes. The notes describe the algorithm steps but do not provide a concrete numerical example.",
  "worked_problem": "I do not know based on the provided notes. The notes outline the general steps but do not walk through a specific numerical problem."
}
```

> The `"I do not know based on the provided notes"` responses above are
> **correct, expected behaviour** - not an error. The model is grounded
> strictly in your uploaded notes and refuses to invent content. Use
> `/explain-with-grounding` to supplement these gaps.

**No notes uploaded (400):**

```json
{
  "detail": "No lecture notes found. Please upload a PDF first via the /upload-note endpoint, then ask your question."
}
```

**Expected terminal output:**

```
[FAISS] Loaded existing index from 'akadverse_concept_faiss_index'.
[Model] Using generative model: gemini-2.5-flash
[Explain] Generating explanation for: 'Explain the stepping-stone method...' using gemini-2.5-flash...
[KAFKA MOCK] Published event 'session.ended' - study interaction logged.
```

---

### 3. `POST /explain-with-grounding`

**What it does:** Answers a question by searching the live web using Gemini's
native Google Search Grounding tool. No PDF upload is required. Returns a
structured explanation plus the web sources Gemini cited so students can
verify the information.

**Form fields:**

| Field | Required | Description | Example |
|---|---|---|---|
| `question` | Yes | Any academic concept or question | `Explain Dijkstra's shortest path algorithm` |

**Success response (200 OK):**

```json
{
  "concept": "Dijkstra's algorithm is a greedy algorithm used to find shortest paths...",
  "simple_explanation": "Imagine you're finding the quickest way from your starting point to every location on a map...",
  "analogy": "Consider a search-and-rescue operation in a city where rescue teams always go to the closest unrescued location...",
  "example": "A common real-world application is GPS navigation systems. When you request directions, the GPS treats intersections as nodes and roads as weighted edges...",
  "worked_problem": "Let's find the shortest path from node A to all other nodes...\n\n1. Initialization: distances = {A: 0, B: ∞, C: ∞, D: ∞, E: ∞}...\n2. Current Node A (distance 0): Update neighbors B: 4, C: 2...",
  "sources": [
    {
      "title": "wikipedia.org",
      "url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/..."
    },
    {
      "title": "stanford.edu",
      "url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/..."
    },
    {
      "title": "freecodecamp.org",
      "url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/..."
    }
  ]
}
```

> **About the source URLs:** The `vertexaisearch.cloud.google.com/grounding-api-redirect/...`
> format is normal and expected. Google proxies all grounding citation URLs
> through their redirect service. Clicking them in a browser takes you to
> the original source page (Wikipedia, Stanford, FreeCodeCamp, etc.).

**Expected terminal output:**

```
[Model] Using generative model: gemini-2.5-flash
[Grounding] Using model: gemini-2.5-flash with Google Search grounding...
[Grounding] Received grounded response (11325 chars).
[Grounding] Extracted 7 cited source(s).
[KAFKA MOCK] Published event 'session.ended' - grounded study interaction logged.
```

---

### 4. `GET /health`

**What it does:** Reports service status, FAISS index state, and a map of
all available endpoints. Call this first to confirm the service started
correctly before testing other endpoints.

**No parameters required.**

**Success response (200 OK):**

```json
{
  "status": "ok",
  "version": "1.2",
  "index_in_memory": false,
  "index_on_disk": true,
  "rag_status": "Ready - index on disk, loads on first question.",
  "grounding_status": "Ready - no PDF required for /explain-with-grounding.",
  "endpoints": {
    "POST /upload-note": "Upload a PDF lecture note for RAG Q&A.",
    "POST /explain-concept": "Ask a question answered from your uploaded notes.",
    "POST /explain-with-grounding": "Ask any question answered from live web search.",
    "GET  /health": "This endpoint."
  }
}
```

**Reading the index status fields:**

| `index_in_memory` | `index_on_disk` | What it means |
|---|---|---|
| `false` | `false` | No PDF uploaded yet. Upload one before using `/explain-concept`. |
| `false` | `true` | Index exists from a prior session. Will load on the first question. |
| `true` | `true` | Index is loaded in memory and ready. `/explain-concept` responds immediately. |

---

## Testing with Swagger UI

FastAPI auto-generates an interactive testing interface. Once the server is
running, open your browser and go to:

```
http://127.0.0.1:8010/docs
```

You will see all four endpoints listed. To test any endpoint:

1. Click the endpoint name to expand it.
2. Click **"Try it out"** (top right of the endpoint card).
3. Fill in the displayed request fields. The Gemini key is already configured
  in the server environment and is not a request field. For file uploads,
  click **"Choose File"**.
4. Click **"Execute"**.
5. The response appears below, along with the exact `curl` command used.

> Keep your terminal visible alongside the browser to see server logs in
> real time as requests come in.

---

## Automated Tests

From `akadverse-concept-explainer/`, with the service dependencies installed
in the root uv environment, run:

```bash
uv run pytest tests/ -q --tb=short
```

The last verified result after Phase 2a was **5 passed, 2 deprecation warnings**.
Three tests cover existing API-key behavior; two cover storage round-trip
and missing-`DATABASE_URL` validation. The database test needs a reachable
Supabase project with pgvector already enabled; it is not a mocked database test.

`tests/conftest.py` sets the Windows SelectorEventLoop policy before other
imports, then loads `.env` through python-dotenv. It creates a unique
`concept_explainer_test_<uuid>` schema, rewrites the migration's schema name,
runs the DDL twice to check idempotency, and rolls back the outer transaction.
The round-trip test inserts, reads, verifies metadata and 768 dimensions,
checks indexes, and deletes a synthetic chunk. It does not write test data
into `concept_explainer`.

Do not run pytest without `--tb=short`: longer tracebacks can expose local
variables containing secrets. If needed, debug only the failing test with
`--tb=long` and redact the output before sharing. When Phase 2b introduces a
launcher or starts using `db.py` in FastAPI on Windows, set the same selector
policy before Uvicorn starts. The current production app does not use `db.py`.

---

## Example Test Inputs

Run these in order for a complete end-to-end verification.

These are manual checks of the existing runtime, separate from the automated
storage tests. Response timings, citation counts, model names, and example
outputs below are illustrative and vary with input, available models, and quota.

---

### Test 1 - Health check

Call `GET /health`. Confirm `"status": "ok"` and grounding status shows
`"Ready"`. This verifies the server started correctly.

---

### Test 2 - Search Grounding (no PDF needed)

Call `POST /explain-with-grounding`:

| Field | Value |
|---|---|
| `question` | `Explain Dijkstra's shortest path algorithm` |

**Expected:** All five explanation fields richly populated, plus a `sources`
list with 3 to 8 citations from sites like Wikipedia, Stanford, FreeCodeCamp,
and Brilliant. Response takes 5 to 15 seconds.

---

### Test 3 - Upload a PDF

Call `POST /upload-note`:

| Field | Value |
|---|---|
| `file` | Any PDF lecture note you have (under 50 MB) |

**Expected:** `200 OK` with `chunks_embedded` and `pages_loaded` counts.
Progress logs appear in your terminal every 10 chunks. After completion,
`GET /health` should show `index_on_disk: true`.

---

### Test 4 - Ask from your notes

Call `POST /explain-concept` with a question about a topic in your PDF:

| Field | Value |
|---|---|
| `question` | A concept from your uploaded lecture note |

**Expected:** A structured response grounded in your lecturer's material.
If a concept is not explicitly covered in the notes, `"I do not know based
on the provided notes"` will appear in the relevant fields. This is correct
behaviour, not an error.

---

### Test 5 - FAISS persistence across restarts

Stop the server with `Ctrl+C`. Restart it. Without re-uploading any PDF,
immediately call `POST /explain-concept` with the same question from Test 4.

**Expected:** The same quality response. Terminal should show:

```
[FAISS] Loaded existing index from 'akadverse_concept_faiss_index'.
```

This confirms the FAISS persistence is working - the index survived the
server restart and was reloaded automatically.

---

### Test 6 - Compare modes on the same question

Use a topic from your uploaded PDF that the notes cover conceptually but
without a worked example. Call both endpoints with the same question:

**`/explain-concept` result:** `simple_explanation` and `analogy` populated
from lecturer's notes; `example` and `worked_problem` may say "I do not know."

**`/explain-with-grounding` result:** All five fields populated from the web;
`sources` lists the sites used.

This side-by-side comparison is the clearest demonstration of how the two
modes complement each other in a real student workflow.

---

### Test 7 - File size guard

Attempt to upload a file over 50 MB via `/upload-note`.

**Expected:** A `413 Request Entity Too Large` response showing the actual
file size and the limit. No embedding or API calls should start.

---

## Understanding the Responses

### Why some fields say "I do not know"

In `/explain-concept`, the model is explicitly instructed to ground its
answer strictly in the uploaded notes and not to invent information. If
your PDF describes a concept's procedure but does not include a numerical
example or worked calculation, those fields return `"I do not know based
on the provided notes"`. This is honest, safe behaviour - not a bug.

Use `/explain-with-grounding` to fill those gaps from the web.

### About the grounding source URLs

URLs in the `sources` array always look like:

```
https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQ...
```

This is normal. Google routes all grounding citations through their redirect
service. Clicking the URL in a browser takes you to the original page.

### Why uploads take time

The legacy service targets a nominal 100 embedding requests per minute by
sending 10 chunks at a time with an 8-second pause between batches.
Actual Gemini quotas depend on the project and model; upload duration varies.
A progress line prints after
each batch so you can track it. The service is not frozen - it is waiting
to respect the rate limit.

### Why grounding responses take longer than RAG responses

`/explain-with-grounding` performs a live Google Search and synthesises
content from multiple web pages before responding. This takes 5 to 15
seconds in the illustrative examples, depending on the topic and Gemini's
current load. `/explain-concept` retrieves locally but still calls Gemini
for generation, so relative response times are not guaranteed.

---

## Choosing the Right Mode

| I want to... | Use this endpoint |
|---|---|
| Study content my lecturer specifically taught | `/explain-concept` after uploading their PDF |
| Get a worked example my notes do not have | `/explain-with-grounding` |
| Understand a concept not yet covered in class | `/explain-with-grounding` |
| See where the AI's information came from | `/explain-with-grounding` (returns cited URLs) |
| Prepare for an exam using my exact lecture material | `/explain-concept` |
| Use the service without uploading anything | `/explain-with-grounding` |

---

## Generated Files

Running this service creates the following files in your project folder.
**Do not commit these to version control** - they are listed in `.gitignore`.

| File / Folder | What it is |
|---|---|
| `akadverse_concept_faiss_index/` | FAISS vector index folder, created on first upload |
| `akadverse_concept_faiss_index/index.faiss` | The binary FAISS index file |
| `akadverse_concept_faiss_index/index.pkl` | Serialised document metadata |

To reset to a clean state and wipe all uploaded notes:

```bash
# Windows
rmdir /s /q akadverse_concept_faiss_index

# macOS/Linux
rm -rf akadverse_concept_faiss_index
```

After resetting, `GET /health` will show `index_on_disk: false` and you
will need to upload a PDF again before using `/explain-concept`.

---

## Common Errors and Fixes

**`ModuleNotFoundError: No module named 'google.genai'`**
```bash
uv pip install "google-genai>=1.67.0"
```

**`ModuleNotFoundError: No module named 'faiss'`**
```bash
uv pip install faiss-cpu
```

**`ModuleNotFoundError: No module named 'pypdf'`**
```bash
uv pip install pypdf
```

**`AttributeError` related to `types.Tool` or `types.GoogleSearch`**

Your `google-genai` version is below 1.67.0. Upgrade it:
```bash
uv pip install --upgrade "google-genai>=1.67.0"
```

**`400` or `403` on `/explain-with-grounding` mentioning grounding not supported**

Your API key may need Google Search Grounding enabled or billing configured
in your Google Cloud project. Check quota settings at
[https://aistudio.google.com](https://aistudio.google.com). Grounding
requires at least the Pay-as-you-go tier in some regions.

**`413 Request Entity Too Large` on `/upload-note`**

Your PDF exceeds 50 MB. Upload a focused set of lecture slides rather than
a full textbook. To raise the limit, change `MAX_UPLOAD_SIZE_MB` at the
top of `concept_explainer.py`.

**`400 Bad Request`: "No lecture notes found"**

You called `/explain-concept` before uploading a PDF, or the FAISS index
was deleted. Call `GET /health` to check `index_on_disk`, then upload a
PDF if needed.

**`422 Unprocessable Entity`: "Could not extract any text from the PDF"**

The PDF is likely a scanned image rather than a text-based document.
Image-only PDFs cannot be processed - the text must be selectable (you
should be able to highlight and copy it in a PDF viewer).

**Upload appears frozen in the terminal**

This is normal during the 8-second sleep between embedding batches. Wait
for the next batch log line to appear. For a 64-chunk document you will
see roughly 7 batch logs in total before completion.

**`Address already in use` on startup**

Port 8010 is occupied by another process. Two options:

1. Free 8010 by stopping the conflicting process. On Windows: `Get-NetTCPConnection -LocalPort 8010 | Select-Object OwningProcess`, then `Stop-Process -Id <pid>`. On macOS/Linux: `lsof -i :8010`, then `kill <pid>`.
2. Use an unregistered port in the 8020+ range for local testing only. Do not reuse any port in the Port Registry section. Do not change the service's default port in `AGENTS.md` or in production configuration without coordinating with the platform team.

**`RuntimeError`: DATABASE_URL environment variable is not set**

Storage validation needs `DATABASE_URL`. Tests load `.env` from their fixture;
the application does not load it automatically. Check the configured variable
without printing its value, and use the bare Postgres URL described above.

**`ModuleNotFoundError`: No module named 'psycopg'**

Install the service requirements into the uv environment with
`uv pip install -r requirements.txt`. A package installed into an isolated
`uv run --with` environment is not thereby installed into the root project.

**Psycopg cannot use ProactorEventLoop on Windows**

The database test requires the selector policy already set at the top of
`tests/conftest.py`. Future Windows process launchers using `db.py` must set
the policy before starting Uvicorn. Do not switch to a synchronous driver
to mask this configuration error.

**Extension permissions, missing vector type, or connection failures**

Follow Supabase Setup: enable `vector`, use Session pooler port 5432, encode
password characters correctly, and resume a paused project. Execute the
whole migration transaction to retain its local search path. Keep diagnostic
output free of credentials.

---

## Project Structure

```
akadverse-concept-explainer/
│
├── concept_explainer.py                  # Existing FastAPI app and legacy FAISS routes
├── config.py                             # Storage environment validation
├── db.py                                 # Async Postgres engine and session factory
├── migrations/
│   └── 001_pgvector_storage.sql           # Manual Phase 2a storage migration
├── tests/
│   ├── conftest.py                        # Windows policy, dotenv, isolated test schema
│   ├── test_api_key_env.py                # Existing API-key tests
│   └── test_db.py                         # Storage and configuration tests
├── conftest.py                            # Existing root test configuration
├── AGENTS.md                              # Service instructions and roadmap
├── .env.example                           # Variable names with empty values
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
├── .gitignore                            # Excludes index folder and env files
│
└── akadverse_concept_faiss_index/        # Generated on first upload - DO NOT COMMIT
    ├── index.faiss
    └── index.pkl
```

---

## Part of the AkadVerse Platform

This microservice is **Tier 5** in the AkadVerse AI architecture.

Concept Explainer owns port 8010. Related platform services include Notes
Creator (8008), Slide Generator (8009), Resource Puller (8011), Quiz Generator
(8016), and the Orchestrator (8000). The root `AGENTS.md` is the full port registry.
The intended ingestion caller is the Next.js backend. `[KAFKA MOCK]` lines
are log messages only; they do not implement Kafka publishing or webhooks.
Ingestion callbacks and durable job processing are planned work, not Phase 2a
runtime behavior.

## Integration Notes for Full Stack Engineer

These are forward-looking coordination notes, not a list of implemented APIs.
Phase 2a provides storage only; `concept_explainer.jobs` does not exist yet.
The service roadmap in `AGENTS.md` schedules delete/callback work for Phase 2d
and chat history for Phase 4, so delivery dates in the architecture handoff
below require coordination before anyone builds against them.

(a) **Architecture.** "Chat history is a separate phase. We ship ingestion first. Phase 2b delivers `/ingest` and `/documents/{id}`. Phase 3 delivers the retrieval endpoint and Stewart's chat history. Do not build against the retrieval endpoint yet."

(b) **Upload UI.** "Please enable DOCX and PPTX uploads in the lecturer UI. Please disable ZIP for now; ZIP support is deferred to a later phase."

(c) **Database.** "What Postgres instance does AkadVerse use in production? Please confirm `CREATE EXTENSION vector` is available on it, or enable it from the Supabase dashboard. This service uses the same Postgres as AkadVerse, in a schema it owns (`concept_explainer`). No shared tables, no cross-service coupling."

(d) **Async jobs.** "Async ingestion tracking uses a Postgres `jobs` table. No Redis, no Celery, no external queue. If a job appears stuck, you can query `concept_explainer.jobs` directly."

(e) **Secrets.** "The callback secret shared earlier appeared in plain text in chat. Please rotate before production and share the new value via password manager. Treat any secret that has appeared in a chat log or terminal traceback as compromised."

(f) **Migrations.** "Schema changes on our side are numbered SQL files in `migrations/`, applied manually. You do not need to coordinate timing for now. When we deploy to a shared environment, we will hand you the file and confirm application."

---

*AkadVerse AI Architecture v1.5*
