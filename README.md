# AkadVerse Concept Explainer
### Tier 5 Learning AI Tool | Microservice Port: `8006`

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
5. [Getting Your API Key](#getting-your-api-key)
6. [Installation](#installation)
7. [Running the Server](#running-the-server)
8. [API Endpoints](#api-endpoints)
   - [POST /upload-note](#1-post-upload-note)
   - [POST /explain-concept](#2-post-explain-concept)
   - [POST /explain-with-grounding](#3-post-explain-with-grounding)
   - [GET /health](#4-get-health)
9. [Testing with Swagger UI](#testing-with-swagger-ui)
10. [Example Test Inputs](#example-test-inputs)
11. [Understanding the Responses](#understanding-the-responses)
12. [Choosing the Right Mode](#choosing-the-right-mode)
13. [Generated Files](#generated-files)
14. [Common Errors and Fixes](#common-errors-and-fixes)
15. [Project Structure](#project-structure)

---

## What This Microservice Does

This service is a **Tier 5 component** of the AkadVerse AI-first e-learning
platform. It lives inside the *My Learning* module and acts as a personal
AI tutor for students.

A student can:

- **Upload their lecture notes as a PDF** and ask questions about the specific
  content their lecturer taught. The AI answers strictly from those notes and
  honestly says "I do not know" rather than hallucinating if the answer is not
  in the document.
- **Ask any academic question without uploading anything** and receive a
  structured explanation sourced from the live web, complete with citations
  so they can verify and explore further.

Every response, regardless of mode, is structured into five pedagogical
components proven to aid understanding: a plain-language explanation, a
relatable analogy, a concrete example, and a step-by-step worked problem.

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
designed as complements -- use Mode A to study from your lecturer's
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

**Key design decisions:**

- The FAISS index is **saved to disk** after every upload and reloaded
  automatically on first use after a server restart. Students never need
  to re-upload their notes between sessions.
- Uploaded PDFs are saved with a **UUID-prefixed temp filename** so two
  concurrent uploads of files with the same name cannot corrupt each other.
- **File size is checked before any processing begins.** Files over 50 MB
  are rejected immediately with a clear 413 error.
- The embedding model is **discovered dynamically at runtime** via
  `ListModels` rather than hardcoded, so the service adapts automatically
  if Google renames or deprecates models.
- Search Grounding and `with_structured_output()` **cannot run together**
  in one API call. The grounding endpoint solves this by prompting Gemini
  to embed a JSON block in its free-text response, which is then extracted
  by a regex-based helper with three fallback strategies.

---

## Prerequisites

- **Python 3.10 or higher**
- **pip** (Python package manager)
- A **Google Gemini API key** (one key covers both modes -- free tier available)
- Internet connection (for embedding calls and grounding search)

> **Windows users:** All commands below work in the VS Code integrated
> terminal or Windows PowerShell. Use `python` instead of `python3` if needed.

---

## Getting Your API Key

1. Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with a Google account.
3. Click **Create API Key**.
4. Copy the key -- you will paste it into Swagger UI form fields when testing.

> The free tier includes access to Gemini 2.5 Flash, sufficient embedding
> quota for lecture-sized PDFs, and Search Grounding support.

> **Important for Search Grounding:** `/explain-with-grounding` uses
> Google's native Search Grounding tool. This is available on the free tier
> but may require billing to be configured on your Google Cloud account in
> some regions. If you receive a `400` or `403` error on the grounding
> endpoint specifically, check your quota settings at
> [https://aistudio.google.com](https://aistudio.google.com).

---

## Installation

### Step 1 — Set up your project folder

Place `concept_explainer.py` in a dedicated folder:

```
akadverse-concept-explainer/
└── concept_explainer.py
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Create the environment
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install fastapi uvicorn "google-genai>=1.67.0" langchain-google-genai langchain-community langchain-text-splitters faiss-cpu pydantic pypdf
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

---

## Running the Server

From inside your project folder with the virtual environment activated:

```bash
uvicorn concept_explainer:app --host 127.0.0.1 --port 8006 --reload
```

**Expected startup output (first run, no prior uploads):**

```
[Startup] AkadVerse Concept Explainer initializing...
[Startup] FAISS index path: 'akadverse_concept_faiss_index'
[Startup] Max upload size: 50 MB
[Startup] No existing index found — awaiting first PDF upload.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8006 (Press CTRL+C to quit)
```

**Expected startup output (after a previous upload):**

```
[Startup] AkadVerse Concept Explainer initializing...
[Startup] FAISS index path: 'akadverse_concept_faiss_index'
[Startup] Max upload size: 50 MB
[Startup] Existing FAISS index found on disk — will load on first request.
INFO:     Application startup complete.
```

The second message confirms FAISS persistence is working -- the index from
your last session is on disk and will be loaded automatically when a student
asks their first question, without requiring a re-upload.

Keep this terminal open while testing. The `--reload` flag restarts the
server automatically when you edit the code.

---

## API Endpoints

### 1. `POST /upload-note`

**What it does:** Accepts a PDF lecture note, splits it into 1000-character
overlapping chunks, generates embeddings using Gemini, and stores everything
in a FAISS vector index on disk. This must be called before using
`/explain-concept`.

**Form fields:**

| Field | Required | Description | Example |
|---|---|---|---|
| `file` | Yes | A PDF file to index (max 50 MB) | `lecture_notes.pdf` |
| `google_api_key` | Yes | Your Gemini API key | `AIza...` |

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

**What it does:** Answers a question using only the content from your uploaded
PDF. Retrieves the three most relevant chunks from the FAISS index and
generates a structured pedagogical explanation. If the concept is not
covered in the notes, it says so rather than hallucinating.

Requires a prior call to `/upload-note`. If the server was restarted since
the last upload, the index is loaded from disk automatically -- no
re-upload needed.

**Form fields:**

| Field | Required | Description | Example |
|---|---|---|---|
| `question` | Yes | The concept or question to explain | `Explain the stepping-stone method` |
| `google_api_key` | Yes | Your Gemini API key | `AIza...` |

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
> **correct, expected behaviour** -- not an error. The model is grounded
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
[KAFKA MOCK] Published event 'session.ended' — study interaction logged.
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
| `google_api_key` | Yes | Your Gemini API key | `AIza...` |

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
[KAFKA MOCK] Published event 'session.ended' — grounded study interaction logged.
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
  "rag_status": "Ready — index on disk, loads on first question.",
  "grounding_status": "Ready — no PDF required for /explain-with-grounding.",
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
http://127.0.0.1:8006/docs
```

You will see all four endpoints listed. To test any endpoint:

1. Click the endpoint name to expand it.
2. Click **"Try it out"** (top right of the endpoint card).
3. Fill in the form fields. For file uploads, click **"Choose File"**.
4. Click **"Execute"**.
5. The response appears below, along with the exact `curl` command used.

> Keep your terminal visible alongside the browser to see server logs in
> real time as requests come in.

---

## Example Test Inputs

Run these in order for a complete end-to-end verification.

---

### Test 1 — Health check

Call `GET /health`. Confirm `"status": "ok"` and grounding status shows
`"Ready"`. This verifies the server started correctly.

---

### Test 2 — Search Grounding (no PDF needed)

Call `POST /explain-with-grounding`:

| Field | Value |
|---|---|
| `question` | `Explain Dijkstra's shortest path algorithm` |
| `google_api_key` | Your key |

**Expected:** All five explanation fields richly populated, plus a `sources`
list with 3 to 8 citations from sites like Wikipedia, Stanford, FreeCodeCamp,
and Brilliant. Response takes 5 to 15 seconds.

---

### Test 3 — Upload a PDF

Call `POST /upload-note`:

| Field | Value |
|---|---|
| `file` | Any PDF lecture note you have (under 50 MB) |
| `google_api_key` | Your key |

**Expected:** `200 OK` with `chunks_embedded` and `pages_loaded` counts.
Progress logs appear in your terminal every 10 chunks. After completion,
`GET /health` should show `index_on_disk: true`.

---

### Test 4 — Ask from your notes

Call `POST /explain-concept` with a question about a topic in your PDF:

| Field | Value |
|---|---|
| `question` | A concept from your uploaded lecture note |
| `google_api_key` | Your key |

**Expected:** A structured response grounded in your lecturer's material.
If a concept is not explicitly covered in the notes, `"I do not know based
on the provided notes"` will appear in the relevant fields. This is correct
behaviour, not an error.

---

### Test 5 — FAISS persistence across restarts

Stop the server with `Ctrl+C`. Restart it. Without re-uploading any PDF,
immediately call `POST /explain-concept` with the same question from Test 4.

**Expected:** The same quality response. Terminal should show:

```
[FAISS] Loaded existing index from 'akadverse_concept_faiss_index'.
```

This confirms the FAISS persistence is working -- the index survived the
server restart and was reloaded automatically.

---

### Test 6 — Compare modes on the same question

Use a topic from your uploaded PDF that the notes cover conceptually but
without a worked example. Call both endpoints with the same question:

**`/explain-concept` result:** `simple_explanation` and `analogy` populated
from lecturer's notes; `example` and `worked_problem` may say "I do not know."

**`/explain-with-grounding` result:** All five fields populated from the web;
`sources` lists the sites used.

This side-by-side comparison is the clearest demonstration of how the two
modes complement each other in a real student workflow.

---

### Test 7 — File size guard

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
on the provided notes"`. This is honest, safe behaviour -- not a bug.

Use `/explain-with-grounding` to fill those gaps from the web.

### About the grounding source URLs

URLs in the `sources` array always look like:

```
https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQ...
```

This is normal. Google routes all grounding citations through their redirect
service. Clicking the URL in a browser takes you to the original page.

### Why uploads take time

The Gemini embedding API is rate-limited to 100 requests per minute. The
service sends 10 chunks at a time with an 8-second pause between batches.
A 64-chunk PDF takes approximately 55 seconds. A progress line prints after
each batch so you can track it. The service is not frozen -- it is waiting
to respect the rate limit.

### Why grounding responses take longer than RAG responses

`/explain-with-grounding` performs a live Google Search and synthesises
content from multiple web pages before responding. This takes 5 to 15
seconds depending on the topic and Gemini's current load. RAG responses
from `/explain-concept` are faster because all retrieval is local.

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
**Do not commit these to version control** -- they are listed in `.gitignore`.

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
pip install "google-genai>=1.67.0"
```

**`ModuleNotFoundError: No module named 'faiss'`**
```bash
pip install faiss-cpu
```

**`ModuleNotFoundError: No module named 'pypdf'`**
```bash
pip install pypdf
```

**`AttributeError` related to `types.Tool` or `types.GoogleSearch`**

Your `google-genai` version is below 1.67.0. Upgrade it:
```bash
pip install --upgrade "google-genai>=1.67.0"
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
Image-only PDFs cannot be processed -- the text must be selectable (you
should be able to highlight and copy it in a PDF viewer).

**Upload appears frozen in the terminal**

This is normal during the 8-second sleep between embedding batches. Wait
for the next batch log line to appear. For a 64-chunk document you will
see roughly 7 batch logs in total before completion.

**`Address already in use` on startup**

Port 8006 is occupied by another process. Use a different port:
```bash
uvicorn concept_explainer:app --host 127.0.0.1 --port 8009 --reload
```

---

## Project Structure

```
akadverse-concept-explainer/
│
├── concept_explainer.py                  # Main microservice — all logic here
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
├── .gitignore                            # Excludes index folder and env files
│
└── akadverse_concept_faiss_index/        # Generated on first upload — DO NOT COMMIT
    ├── index.faiss
    └── index.pkl
```

---

## Part of the AkadVerse Platform

This microservice is **Tier 5** in the AkadVerse AI architecture, operating
within the E-Learning module alongside:

- External Resources Puller (Port 8007)
- Notes Creator (Port 8008)
- Quiz Generator (Port 8009)
- Slides Creator (Port 8010)

All Tier 5 services communicate with the rest of the platform via Apache
Kafka event bus (simulated locally with FastAPI webhooks during development).
The `[KAFKA MOCK]` log lines in the terminal represent events that would be
published to the real Kafka bus in production.

---

*AkadVerse AI Architecture v1.0*