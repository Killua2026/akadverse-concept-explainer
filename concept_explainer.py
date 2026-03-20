"""
AkadVerse — Concept Explainer (RAG-powered Study Assistant)
Tier 5 | Microservice Port: 8006
========================================================================
v1.0 — Original build:
  - /upload-note: Upload a PDF, chunk it, embed it, store in FAISS.
  - /explain-concept: RAG Q&A over uploaded notes with structured
    pedagogical output (explanation, analogy, example, worked problem).
  - Dynamic model discovery via genai.Client().models.list().
  - Rate-limit protection: 8s sleep between embedding batches.
  - Embedding model fallback chain for deprecated model resilience.

v1.1 — Three bug fixes applied:
  - FIX 1 (FAISS Persistence): Replaced the global in-memory vector_store
    with a disk-persisted FAISS index. On upload, the index is saved to
    FAISS_INDEX_PATH. On /explain-concept, the index is loaded from disk
    if not already in memory. A server restart no longer wipes uploaded
    notes — students do not lose their context between sessions.

  - FIX 2 (Temp File Collision): The original temp file was named
    temp_{filename}, a predictable name that causes file corruption when
    two students upload files with the same name concurrently. Fixed by
    prepending a uuid4() string to the temp filename, guaranteeing
    uniqueness per upload regardless of concurrency.

  - FIX 3 (File Size Guard): No file size limit meant a student uploading
    a 500-page textbook would run for ~4 minutes and hammer API quota.
    Added a MAX_UPLOAD_SIZE_MB constant checked before any processing.
    If the upload exceeds the limit, a clear 413 error is returned
    immediately, before any disk write or embedding call.

v1.2 — Search Grounding mode added:
  - NEW ENDPOINT /explain-with-grounding: A second explanation mode that
    bypasses FAISS entirely and uses Gemini's native Google Search Grounding
    tool to answer questions from live web content. No PDF upload required.

    Key technical decisions:
    (a) Uses the unified google-genai SDK with types.Tool(google_search=
        types.GoogleSearch()) — NOT the older google-generativeai syntax
        from the gemini_search_grounding_aistudio.py reference script.
    (b) Grounding and with_structured_output() cannot be used together in
        the same API call. Instead, the prompt explicitly instructs Gemini
        to embed a JSON block in its response. A json_extraction helper
        then locates, extracts, and parses that block into ExplanationResponse.
    (c) Grounding metadata (cited source URLs and titles) is returned in a
        separate 'sources' field so students can see exactly where the
        information came from and verify it.
    (d) The endpoint works even with no PDF uploaded — it is fully
        independent of the FAISS pipeline.
"""

import os
import re
import json
import time
import shutil
import threading
from uuid import uuid4                          # FIX 2: For unique temp file names
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field, SecretStr

# Unified Google GenAI SDK (v1.67.0+)
# types is needed for the Search Grounding tool configuration (v1.2)
from google import genai
from google.genai import types

# LangChain ecosystem components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# =========================================================
# CONSTANTS
# =========================================================

# FIX 1: Path where the FAISS index folder is saved to disk.
# On every successful upload, the index is written here.
# On every /explain-concept call, the index is loaded from here if
# not already held in memory, surviving server restarts cleanly.
FAISS_INDEX_PATH = "akadverse_concept_faiss_index"

# FIX 3: Maximum allowed upload size in megabytes.
# 50 MB is generous for lecture note PDFs while guarding against
# textbook-sized uploads that would consume excessive API quota.
MAX_UPLOAD_SIZE_MB = 50
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Embedding batch size and inter-batch sleep to stay under the
# Gemini API rate limit of 100 embedding requests per minute.
EMBEDDING_BATCH_SIZE = 10
EMBEDDING_BATCH_SLEEP_SECONDS = 8

# FIX 1: A threading lock guards all FAISS write operations to prevent
# index corruption if two uploads arrive concurrently.
faiss_lock = threading.Lock()

# FIX 1: Module-level reference to the loaded FAISS index.
# Populated on first use (either fresh upload or load from disk).
# This replaces the old bare `vector_store = None` global.
_vector_store: Optional[FAISS] = None


# =========================================================
# PYDANTIC MODELS
# =========================================================

class ExplanationResponse(BaseModel):
    """Schema for the structured pedagogical explanation output."""
    concept: str = Field(description="The core concept being explained.")
    simple_explanation: str = Field(description="A plain-language explanation.")
    analogy: str = Field(description="A relatable analogy to help understand the concept.")
    example: str = Field(description="A concrete example of the concept in action.")
    worked_problem: str = Field(description="A step-by-step breakdown or worked problem.")


class UploadSuccessResponse(BaseModel):
    """Schema for a successful note upload response."""
    status: str
    message: str
    filename: str
    chunks_embedded: int
    pages_loaded: int


class GroundedSource(BaseModel):
    """A single cited web source returned by Gemini's grounding metadata."""
    title: str = Field(description="Page title of the source.")
    url: str = Field(description="URL of the source.")


class GroundedExplanationResponse(BaseModel):
    """
    Schema for the /explain-with-grounding endpoint response.

    Extends ExplanationResponse with a 'sources' field that lists the
    web pages Gemini cited when generating the answer. This lets students
    verify the information and read further if they want to.
    """
    concept: str
    simple_explanation: str
    analogy: str
    example: str
    worked_problem: str
    sources: List[GroundedSource] = Field(
        default_factory=list,
        description="Web sources cited by Gemini when generating this answer."
    )


# =========================================================
# DYNAMIC MODEL DISCOVERY
# =========================================================

def get_valid_model_name(api_key_str: str) -> str:
    """
    Dynamically discovers the best available Gemini generative model
    by calling client.models.list() and matching against a priority list.

    Mirrors the pattern from the Resources Puller for consistency
    across all Tier 5 microservices.

    Falls back to 'gemini-1.5-flash' if discovery fails entirely.
    """
    try:
        client = genai.Client(api_key=api_key_str)

        # Collect all available model names, stripping the "models/" prefix
        all_models = [
            m.name.replace("models/", "")
            for m in client.models.list()
            if m.name
        ]

        # Prefer models in order of capability vs cost
        priority_order = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-pro",
        ]
        for preferred in priority_order:
            if preferred in all_models:
                print(f"[Model] Using generative model: {preferred}")
                return preferred

        if all_models:
            print(f"[Model] Fallback to first available: {all_models[0]}")
            return all_models[0]

    except Exception as e:
        print(f"[Model WARNING] Discovery failed ({e}). Defaulting to 'gemini-1.5-flash'.")

    return "gemini-1.5-flash"


# =========================================================
# FIX 1: FAISS PERSISTENCE HELPERS
# =========================================================

def _load_index_from_disk(embeddings: GoogleGenerativeAIEmbeddings) -> Optional[FAISS]:
    """
    FIX 1: Loads the FAISS index from disk if it exists.

    Uses allow_dangerous_deserialization=True which is required by FAISS
    when loading a pickled index from disk. This is safe here because the
    index is written by our own service — it is not user-supplied data.

    Returns None if no index has been saved yet (first-run or clean state).
    """
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            index = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"[FAISS] Loaded existing index from '{FAISS_INDEX_PATH}'.")
            return index
        except Exception as e:
            # If the on-disk index is corrupt, log and continue as a fresh start
            print(f"[FAISS WARNING] Could not load index from disk ({e}). Starting fresh.")
    return None


def _save_index_to_disk(index: FAISS) -> None:
    """
    FIX 1: Persists the FAISS index to the FAISS_INDEX_PATH folder.

    Called after every successful upload so the index survives server
    restarts. The faiss_lock must be held by the caller.
    """
    try:
        index.save_local(FAISS_INDEX_PATH)
        print(f"[FAISS] Index saved to '{FAISS_INDEX_PATH}'.")
    except Exception as e:
        # Log but do not crash — the in-memory index is still usable this session
        print(f"[FAISS ERROR] Failed to save index to disk: {e}")


def _get_or_load_vector_store(embeddings: GoogleGenerativeAIEmbeddings) -> Optional[FAISS]:
    """
    FIX 1: Returns the current in-memory FAISS index, loading it from disk
    if it has not been loaded yet this session.

    This is the single access point for reading the vector store, ensuring
    that a server restart does not force students to re-upload their notes.
    """
    global _vector_store
    if _vector_store is None:
        # Attempt to hydrate from disk on first access
        _vector_store = _load_index_from_disk(embeddings)
    return _vector_store


# =========================================================
# FASTAPI APPLICATION
# =========================================================

@asynccontextmanager
async def lifespan(_: "FastAPI") -> AsyncIterator[None]:
    """
    Lifespan handler: runs on startup and shutdown.
    Replaces the deprecated @app.on_event("startup") pattern.
    """
    print("[Startup] AkadVerse Concept Explainer initializing...")
    print(f"[Startup] FAISS index path: '{FAISS_INDEX_PATH}'")
    print(f"[Startup] Max upload size: {MAX_UPLOAD_SIZE_MB} MB")
    if os.path.exists(FAISS_INDEX_PATH):
        print("[Startup] Existing FAISS index found on disk — will load on first request.")
    else:
        print("[Startup] No existing index found — awaiting first PDF upload.")
    yield
    print("[Shutdown] AkadVerse Concept Explainer stopped.")


app = FastAPI(
    title="AkadVerse Concept Explainer API",
    description=(
        "Tier 5 RAG-powered study assistant. Upload lecture notes as PDF, "
        "then ask questions to receive structured pedagogical explanations. "
        "Also supports web-grounded explanations via /explain-with-grounding."
    ),
    version="1.2",
    lifespan=lifespan
)

# RAG prompt — instructs Gemini to act as a university professor and
# respond in the exact structured format our Pydantic schema expects.
rag_prompt_template = PromptTemplate(
    template="""You are an expert university professor tutoring a student.
Use the following pieces of retrieved context from the student's lecture notes to explain the concept.
If the answer is not clearly covered in the context, say that you do not know based on the
provided notes — do not invent or hallucinate information.

Context from lecture notes:
{context}

Student's Question:
{question}

Respond with a JSON object matching this exact structure:
- concept: The core concept the student is asking about.
- simple_explanation: A plain-language explanation a first-year student could follow.
- analogy: A relatable, everyday analogy that makes the concept click.
- example: A concrete, specific example of the concept in action.
- worked_problem: A step-by-step worked problem or breakdown that demonstrates the concept.
""",
    input_variables=["context", "question"]
)

# Grounding prompt — used by /explain-with-grounding.
# Critically, this prompt instructs Gemini to embed a JSON block inside its
# grounded response. We cannot use with_structured_output() when grounding
# is enabled (the two features cannot run in the same API call), so we parse
# the JSON block out of the free-text response ourselves using json_extraction.
GROUNDING_PROMPT_TEMPLATE = """You are an expert university professor answering a student's question.
Use your search results to provide a thorough, accurate answer.

Student's Question: {question}

Your response MUST end with a JSON block (inside ```json and ``` markers) matching this exact structure:
{{
  "concept": "the core concept being explained",
  "simple_explanation": "a plain-language explanation a first-year student could follow",
  "analogy": "a relatable everyday analogy that makes the concept click",
  "example": "a concrete specific example of the concept in action",
  "worked_problem": "a step-by-step worked problem or breakdown demonstrating the concept"
}}

Make all five fields thorough and educational. Do not leave any field empty.
"""


def _extract_json_from_grounded_response(raw_text: str) -> dict:
    """
    Extracts and parses the JSON block that Gemini embeds inside its grounded
    free-text response.

    Why this is needed: Gemini's Search Grounding tool and structured output
    (with_structured_output) cannot be used in the same API call. As a
    workaround, the grounding prompt instructs Gemini to append a ```json
    block at the end of its response. This function finds that block using
    a regex, strips the markdown fences, and parses it.

    Fallback strategy: if no fenced block is found, we attempt to find any
    raw JSON object in the response using a broader regex. This handles cases
    where Gemini omits the fences but still returns valid JSON.

    Args:
        raw_text: The full text content of Gemini's grounded response.

    Returns:
        Parsed dict matching the ExplanationResponse schema fields.

    Raises:
        ValueError if no valid JSON block can be found or parsed.
    """
    # Strategy 1: Look for a fenced ```json ... ``` block (the expected format)
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1))
        except json.JSONDecodeError as e:
            print(f"[Grounding] Fenced JSON found but failed to parse: {e}")

    # Strategy 2: Find any JSON object in the response as a broader fallback
    # This handles cases where Gemini returns valid JSON without the fences.
    raw_match = re.search(r"\{[^{}]*\"concept\"[^{}]*\}", raw_text, re.DOTALL)
    if raw_match:
        try:
            return json.loads(raw_match.group(0))
        except json.JSONDecodeError as e:
            print(f"[Grounding] Raw JSON found but failed to parse: {e}")

    # Strategy 3: If both strategies fail, construct a degraded response so
    # the endpoint still returns something meaningful rather than a 500 error.
    # This happens if Gemini ignores the JSON instruction entirely.
    print("[Grounding WARNING] Could not extract JSON from response. Building degraded fallback.")
    return {
        "concept": "Unable to parse structured response",
        "simple_explanation": raw_text[:800] if raw_text else "No explanation available.",
        "analogy": "Not available for this response.",
        "example": "Not available for this response.",
        "worked_problem": "Not available for this response."
    }


# =========================================================
# ENDPOINT 1: Upload a PDF lecture note
# =========================================================

@app.post("/upload-note", response_model=UploadSuccessResponse, tags=["Document Ingestion"])
async def upload_note(
    file: UploadFile = File(..., description="A PDF lecture note to index."),
    google_api_key: str = Form(..., description="Your Google Gemini API key.")
):
    """
    Upload a PDF lecture note, chunk it, embed it with Gemini, and
    persist it to the local FAISS index so it survives server restarts.

    FIX 1: Index is saved to disk after every successful upload.
    FIX 2: Temp file uses a uuid4 prefix to prevent concurrent collision.
    FIX 3: File size is checked before any disk write or API call.
    """
    global _vector_store

    # -- Input validation --
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # FIX 3: Read the file content into memory first so we can check its size
    # before writing anything to disk or calling any external API.
    file_content = await file.read()

    if len(file_content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {len(file_content) / (1024*1024):.1f} MB. "
                f"Maximum allowed size is {MAX_UPLOAD_SIZE_MB} MB. "
                "Please upload a focused set of lecture notes rather than a full textbook."
            )
        )

    # FIX 2: Generate a UUID-prefixed temp filename to prevent collisions
    # when two students upload files with the same name concurrently.
    # e.g. "a3f1b2c4_lecture1.pdf" instead of "temp_lecture1.pdf"
    unique_prefix = uuid4().hex[:8]
    temp_file_path = f"{unique_prefix}_{filename}"

    try:
        # Write the already-read content to the uniquely-named temp file
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_content)

        print(f"[Upload] Processing '{filename}' ({len(file_content) / 1024:.1f} KB)...")

        # -- Step 1: Load the PDF pages via LangChain --
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        if not documents:
            raise HTTPException(
                status_code=422,
                detail=f"Could not extract any text from '{filename}'. The PDF may be image-only or corrupt."
            )

        print(f"[Upload] Loaded {len(documents)} pages from '{filename}'.")

        # -- Step 2: Split pages into overlapping chunks --
        # chunk_size=1000 chars with chunk_overlap=200 ensures context
        # continuity across chunk boundaries during retrieval.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        print(f"[Upload] Split into {len(chunks)} chunks. Starting embedding...")

        # -- Step 3: Embed chunks into FAISS with model fallback chain --
        # We try each candidate model in order. If a model returns NOT_FOUND
        # (deprecated or unavailable for this API key), we move to the next.
        # For all other errors (quota, auth, network) we fail fast immediately.
        embedding_model_candidates = [
            "models/gemini-embedding-001",       # Current stable (renamed from text-embedding-004)
            "models/text-embedding-004",          # Prior stable name — try if above fails
            "models/gemini-embedding-2-preview",  # Preview — future-proofing
            "models/embedding-001",               # Legacy last resort
        ]

        last_embed_error: Optional[Exception] = None
        embedded_successfully = False

        for embedding_model in embedding_model_candidates:
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=embedding_model,
                    api_key=SecretStr(google_api_key)
                )

                print(f"[Embedding] Trying model: {embedding_model}")

                # Initialise the index with the very first chunk to validate
                # the embedding model works before committing to the full batch.
                new_index = FAISS.from_documents([chunks[0]], embeddings)

                # Spoon-feed the remaining chunks in small batches with sleeps
                # to stay within the Gemini 100-requests-per-minute rate limit.
                for i in range(1, len(chunks), EMBEDDING_BATCH_SIZE):
                    batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
                    new_index.add_documents(batch)
                    completed = min(i + EMBEDDING_BATCH_SIZE, len(chunks))
                    print(f"[Embedding] {completed}/{len(chunks)} chunks embedded...")

                    # Only sleep if there are more batches remaining
                    if completed < len(chunks):
                        time.sleep(EMBEDDING_BATCH_SLEEP_SECONDS)

                # FIX 1: Persist the completed index to disk under the lock,
                # then update the in-memory reference atomically.
                with faiss_lock:
                    _save_index_to_disk(new_index)
                    _vector_store = new_index

                print(f"[Embedding] All {len(chunks)} chunks embedded successfully with {embedding_model}.")
                embedded_successfully = True
                break   # Exit the candidate loop on first success

            except Exception as embed_error:
                last_embed_error = embed_error

                if "NOT_FOUND" in str(embed_error):
                    # Model not available for this API key — try the next candidate
                    print(f"[Embedding WARNING] {embedding_model} not available. Trying next...")
                    continue

                # Any other error (auth failure, quota exceeded, network issue)
                # is not recoverable by switching models — fail immediately.
                raise HTTPException(
                    status_code=500,
                    detail=f"Embedding failed with '{embedding_model}': {embed_error}"
                )

        if not embedded_successfully:
            # All candidates failed with NOT_FOUND — nothing left to try
            raise HTTPException(
                status_code=500,
                detail=(
                    "No supported embedding model was found for this API key. "
                    f"Tried: {embedding_model_candidates}. "
                    f"Last error: {last_embed_error}"
                )
            )

        print(f"[KAFKA MOCK] Published event 'note.uploaded' — document indexed for student session.")

        return UploadSuccessResponse(
            status="success",
            message=(
                f"'{filename}' has been processed and indexed. "
                f"You can now ask questions about its content."
            ),
            filename=filename,
            chunks_embedded=len(chunks),
            pages_loaded=len(documents)
        )

    except HTTPException:
        # Re-raise FastAPI errors without wrapping them in a generic 500
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while processing '{filename}': {e}"
        )

    finally:
        # FIX 2: Always clean up the unique temp file, whether the upload
        # succeeded or failed. This prevents orphaned temp files accumulating.
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"[Upload] Temp file '{temp_file_path}' cleaned up.")


# =========================================================
# ENDPOINT 2: Ask a question about the uploaded notes
# =========================================================

@app.post("/explain-concept", response_model=ExplanationResponse, tags=["Q&A"])
async def explain_concept(
    question: str = Form(..., description="The concept or question you want explained."),
    google_api_key: str = Form(..., description="Your Google Gemini API key.")
):
    """
    Ask a question about your uploaded lecture notes. Retrieves the top-3
    most relevant chunks from the FAISS index and generates a structured
    pedagogical explanation using Gemini.

    FIX 1: Loads the FAISS index from disk if the server was restarted
    since the last upload — students do not need to re-upload their notes.
    """
    # We need an embeddings object to load the index from disk if required.
    # Use the first candidate — if it fails here, the user will get a clear error.
    embeddings_for_load = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        api_key=SecretStr(google_api_key)
    )

    # FIX 1: Try to get the in-memory store, or load from disk if missing
    current_store = _get_or_load_vector_store(embeddings_for_load)

    if current_store is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No lecture notes found. Please upload a PDF first via the "
                "/upload-note endpoint, then ask your question."
            )
        )

    try:
        # -- Step 1: Retrieve the 3 most relevant chunks for this question --
        retriever = current_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(question)

        if not relevant_docs:
            raise HTTPException(
                status_code=422,
                detail="Could not retrieve any relevant content for your question. Try rephrasing."
            )

        # Combine retrieved chunks into a single context block for the prompt
        context_text = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)

        # -- Step 2: Select the generative model dynamically --
        selected_model = get_valid_model_name(google_api_key)

        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            api_key=google_api_key,
            temperature=0.3   # Low temperature keeps explanations factual and grounded
        )

        # -- Step 3: Bind the LLM to our structured Pydantic output schema --
        structured_llm = llm.with_structured_output(ExplanationResponse, method="json_mode")

        # -- Step 4: Format the prompt and invoke Gemini --
        prompt_text = rag_prompt_template.format(
            context=context_text,
            question=question
        )

        print(f"[Explain] Generating explanation for: '{question}' using {selected_model}...")
        explanation: ExplanationResponse = structured_llm.invoke(prompt_text)  # type: ignore

        print(f"[KAFKA MOCK] Published event 'session.ended' — study interaction logged.")

        return explanation

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating the explanation: {e}"
        )


# =========================================================
# ENDPOINT 3: Search Grounding — web-sourced explanations (v1.2)
# =========================================================

@app.post("/explain-with-grounding", response_model=GroundedExplanationResponse, tags=["Q&A"])
async def explain_with_grounding(
    question: str = Form(..., description="The concept or question you want explained from the web."),
    google_api_key: str = Form(..., description="Your Google Gemini API key.")
):
    """
    Explains a concept using Gemini's native Google Search Grounding tool.

    Unlike /explain-concept which answers only from uploaded notes, this
    endpoint searches the live web and synthesises an answer from current
    sources. No PDF upload is required — it works as a standalone endpoint.

    How it works:
    1. Sends the question to Gemini with google_search grounding enabled.
    2. Gemini autonomously searches Google and retrieves relevant pages.
    3. The response includes both the synthesised explanation AND a list
       of cited sources so students can verify and explore further.
    4. Because grounding and with_structured_output() cannot run together,
       the prompt instructs Gemini to embed a JSON block in its response.
       _extract_json_from_grounded_response() parses it out.

    Important SDK note: this uses the unified google-genai SDK with
    types.Tool(google_search=types.GoogleSearch()), NOT the older
    google-generativeai syntax (tools='google_search_retrieval').
    """
    try:
        # -- Step 1: Build the grounding-enabled Gemini client --
        client = genai.Client(api_key=google_api_key)

        # Dynamically select the best available generative model
        selected_model = get_valid_model_name(google_api_key)
        # Grounding works best with models that support it; gemini-2.0-flash
        # and above have full grounding support. We fall back gracefully if
        # the selected model doesn't support it (caught in the except block).
        print(f"[Grounding] Using model: {selected_model} with Google Search grounding...")

        # -- Step 2: Format the prompt that asks Gemini to include a JSON block --
        prompt = GROUNDING_PROMPT_TEMPLATE.format(question=question)

        # -- Step 3: Call Gemini with the Search Grounding tool enabled --
        # types.Tool(google_search=types.GoogleSearch()) is the v1.2+ SDK syntax.
        # This is different from the older SDK which used tools='google_search_retrieval'.
        # The grounding tool makes Gemini autonomously search Google during generation.
        response = client.models.generate_content(
            model=selected_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.3   # Low temperature for factual accuracy
            )
        )

        # -- Step 4: Extract the raw text from the response --
        if not response.candidates:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response. Try rephrasing your question."
            )

        raw_text = response.text
        if not raw_text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned empty content. Try rephrasing your question."
            )
        print(f"[Grounding] Received grounded response ({len(raw_text)} chars).")

        # -- Step 5: Extract the JSON block from the free-text grounded response --
        # Gemini cannot use with_structured_output() while grounding is active,
        # so we parse the embedded JSON block ourselves.
        parsed_data = _extract_json_from_grounded_response(raw_text)

        # -- Step 6: Extract cited sources from grounding metadata --
        # The grounding_metadata field contains the web pages Gemini used.
        # We surface these to the student as verifiable references.
        sources: List[GroundedSource] = []
        try:
            candidate = response.candidates[0]
            grounding_meta = getattr(candidate, "grounding_metadata", None)

            if grounding_meta:
                chunks = getattr(grounding_meta, "grounding_chunks", []) or []
                for chunk in chunks:
                    web = getattr(chunk, "web", None)
                    if web:
                        title = getattr(web, "title", "Untitled source") or "Untitled source"
                        url = getattr(web, "uri", "") or ""
                        if url:   # Only include sources with valid URLs
                            sources.append(GroundedSource(title=title, url=url))

            print(f"[Grounding] Extracted {len(sources)} cited source(s).")

        except Exception as meta_err:
            # Metadata extraction failure is non-fatal — we still return the answer
            print(f"[Grounding WARNING] Could not extract source metadata: {meta_err}")

        print(f"[KAFKA MOCK] Published event 'session.ended' — grounded study interaction logged.")

        # -- Step 7: Build and return the grounded response --
        return GroundedExplanationResponse(
            concept=parsed_data.get("concept", question),
            simple_explanation=parsed_data.get("simple_explanation", ""),
            analogy=parsed_data.get("analogy", ""),
            example=parsed_data.get("example", ""),
            worked_problem=parsed_data.get("worked_problem", ""),
            sources=sources
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during grounded explanation: {e}"
        )


# =========================================================
# ENDPOINT 4: Health check
# =========================================================

@app.get("/health", tags=["System"])
async def health_check():
    """
    Reports service status, FAISS index state, and available endpoints.
    """
    index_in_memory = _vector_store is not None
    index_on_disk = os.path.exists(FAISS_INDEX_PATH)

    return {
        "status": "ok",
        "version": "1.2",
        "index_in_memory": index_in_memory,
        "index_on_disk": index_on_disk,
        "rag_status": (
            "Ready — index loaded." if index_in_memory
            else "Ready — index on disk, loads on first question." if index_on_disk
            else "Awaiting PDF upload via /upload-note."
        ),
        "grounding_status": "Ready — no PDF required for /explain-with-grounding.",
        "endpoints": {
            "POST /upload-note": "Upload a PDF lecture note for RAG Q&A.",
            "POST /explain-concept": "Ask a question answered from your uploaded notes.",
            "POST /explain-with-grounding": "Ask any question answered from live web search.",
            "GET  /health": "This endpoint."
        }
    }


# =========================================================
# Run: uvicorn concept_explainer:app --host 127.0.0.1 --port 8006 --reload
# =========================================================