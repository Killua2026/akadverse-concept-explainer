import os
import time
import threading
from uuid import uuid4
from typing import Callable, List, Optional, TypedDict, cast

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field, SecretStr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from services.model_discovery import get_valid_model_name

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

class LegacyDependencies(TypedDict):
    """Keep each dependency's callable type distinct when accessed by key."""

    embeddings: Callable[..., GoogleGenerativeAIEmbeddings]
    vector_store: Callable[[GoogleGenerativeAIEmbeddings], Optional[FAISS]]


class LegacyRouter(APIRouter):
    """Declare legacy dependency wiring as part of the router's interface."""

    def __init__(self) -> None:
        super().__init__()
        # Instance-owned wiring avoids dynamic attributes on FastAPI's base router.
        self.legacy_dependencies: LegacyDependencies = {
            "embeddings": GoogleGenerativeAIEmbeddings,
            "vector_store": _get_or_load_vector_store,
        }


def make_legacy_router(api_key: Optional[str]) -> LegacyRouter:
    router = LegacyRouter()
    GOOGLE_API_KEY = api_key

    @router.post("/upload-note", response_model=UploadSuccessResponse, tags=["Document Ingestion"])
    async def upload_note(
        file: UploadFile = File(..., description="A PDF lecture note to index.")
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
                "models/embedding-001",               # Legacy last resort
            ]

            last_embed_error: Optional[Exception] = None
            embedded_successfully = False

            for embedding_model in embedding_model_candidates:
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model=embedding_model,
                        api_key=SecretStr(cast(str, GOOGLE_API_KEY))
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


    @router.post("/explain-concept", response_model=ExplanationResponse, tags=["Q&A"])
    async def explain_concept(
        question: str = Form(..., description="The concept or question you want explained.")
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
        embeddings_for_load = router.legacy_dependencies["embeddings"](
            model="models/gemini-embedding-001",
            api_key=SecretStr(cast(str, GOOGLE_API_KEY))
        )

        # FIX 1: Try to get the in-memory store, or load from disk if missing
        current_store = router.legacy_dependencies["vector_store"](embeddings_for_load)

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
            selected_model = get_valid_model_name(cast(str, GOOGLE_API_KEY))

            llm = ChatGoogleGenerativeAI(
                model=selected_model,
                api_key=GOOGLE_API_KEY,
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


    return router
