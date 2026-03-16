import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field, SecretStr
import time

# 1. Import the NEW unified Google GenAI SDK (v1.67.0+)
from google import genai

# 2. Import LangChain ecosystem components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------
# Pydantic Models for Structured Output
# ---------------------------------------------------------

class ExplanationResponse(BaseModel):
    """Schema for the structured pedagogical explanation output."""
    concept: str = Field(description="The core concept being explained.")
    simple_explanation: str = Field(description="A plain-language explanation.")
    analogy: str = Field(description="A relatable analogy to help understand the concept.")
    example: str = Field(description="A concrete example of the concept in action.")
    worked_problem: str = Field(description="A step-by-step breakdown or worked problem.")

# ---------------------------------------------------------
# Dynamic Model Discovery Logic (Using Unified SDK)
# ---------------------------------------------------------

def get_valid_model_name(api_key: SecretStr) -> str:
    """
    Dynamically finds a working Google Gemini model.
    Utilizes the modern genai.Client() architecture to prevent missing export errors.
    """
    try:
        # Convert SecretStr to plain string for API usage
        api_key_str = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        # Initialize the new instance-based client
        client = genai.Client(api_key=api_key_str)
        
        all_models = []
        # Access models via the new client.models.list() method
        for m in client.models.list():
            # Null-safety check: ensure the model name actually exists
            if m.name:
                clean_name = m.name.replace("models/", "")
                all_models.append(clean_name)
        
        # Priority list favoring the newest flash models for speed and cost
        priority_order = [
            "gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash", "gemini-pro"
        ]
        
        # Check priority list against available models
        for preferred in priority_order:
            if preferred in all_models:
                print(f"[System]: Selected AI Model dynamically: {preferred}")
                return preferred
                
        # Fallback to whatever is available
        if all_models:
            print(f"[System]: Falling back to first available model: {all_models[0]}")
            return all_models[0]
            
    except Exception as e:
        # Catch structural or network errors during discovery
        print(f"[System Warning]: Model discovery failed ({e}). Defaulting to 'gemini-1.5-flash'.")
    
    return "gemini-1.5-flash"

# ---------------------------------------------------------
# FastAPI Application Setup
# ---------------------------------------------------------

app = FastAPI(
    title="AkadVerse Concept Explainer API",
    description="Tier 5 RAG-powered Q&A agent utilizing the Unified Google GenAI SDK.",
    version="2.0"
)

# Global variable to hold our in-memory FAISS vector store for local testing
vector_store = None

# Custom prompt enforcing the instructional design
rag_prompt_template = PromptTemplate(
    template="""You are an expert university professor tutoring a student.
    Use the following pieces of retrieved context from the student's lecture notes to explain the concept.
    If the answer is not in the context, just say that you do not know, do not try to make up an answer.
    
    Context: {context}
    
    Student's Question: {question}
    
    Provide your response highly structured to match this exact format:
    1. A plain-language explanation in simple terms.
    2. A relatable analogy.
    3. A clear example.
    4. A worked problem or step-by-step breakdown.
    """,
    input_variables=["context", "question"]
)

# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.post("/upload-note")
async def upload_note(file: UploadFile = File(...), google_api_key: str = Form(...)):
    """
    Endpoint to upload a PDF lecture note, split it into chunks, 
    and store it in the local FAISS vector database.
    """
    global vector_store
    
    # Validate the filename before using string methods on it.
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a filename.")
    if not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for this test.")

    temp_file_path = f"temp_{filename}"
    
    try:
        # Save the uploaded file locally temporarily
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"Processing document: {filename}...")

        # Step 1: Load the PDF using LangChain
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        # Step 2: Split the text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Step 3: Generate embeddings and load into FAISS.
        # Important: model validation happens during embedding (FAISS.from_documents),
        # not reliably at client construction time.
        embedding_model_candidates = [
            "models/text-embedding-004",
            "models/gemini-embedding-001",
            "models/embedding-001",
        ]

        last_embed_error: Exception | None = None

        print("Embedding document chunks into FAISS...")
        for embedding_model in embedding_model_candidates:
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=embedding_model,
                    api_key=SecretStr(google_api_key)
                )
                
                print(f"[System]: Embedding model selected: {embedding_model}")
                print(f"Embedding {len(chunks)} chunks into FAISS with rate-limit protection...")
                
                # 1. Initialize the vector store with just the very first chunk
                vector_store = FAISS.from_documents([chunks[0]], embeddings)
                
                # 2. Spoon-feed the rest of the chunks in small batches
                batch_size = 10  # 10 chunks per batch
                for i in range(1, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    vector_store.add_documents(batch)
                    
                    # Print progress so you know it is not frozen
                    print(f"  -> Embedded {min(i + batch_size, len(chunks))} / {len(chunks)} chunks...")
                    
                    # Sleep for 8 seconds between batches to stay under the 100 requests/minute limit
                    time.sleep(8) 
                
                print("Successfully embedded all chunks!")
                break # Success! Break out of the candidate loop
            
            except Exception as embed_error:
                last_embed_error = embed_error
                # Continue trying other models only for model-not-found style failures.
                if "NOT_FOUND" in str(embed_error):
                    print(f"[System Warning]: {embedding_model} unavailable. Trying next candidate.")
                    continue

                # For other errors (auth, quota, network), fail fast with clear detail.
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed during embedding with '{embedding_model}': {embed_error}"
                )
        else:
            # All candidate models failed with NOT_FOUND.
            raise HTTPException(
                status_code=500,
                detail=(
                    "No supported embedding model was found for this API/project. "
                    f"Tried: {embedding_model_candidates}. Last error: {last_embed_error}"
                )
            )

        return {
            "status": "success", 
            "message": f"Successfully processed '{filename}'. {len(chunks)} text chunks embedded and stored in FAISS.",
            "documents_loaded": len(documents)
        }

    except Exception as e:
        # Catch embedding failures, file reading errors, etc.
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
        
    finally:
        # Clean up the temporary file regardless of success or failure
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/explain-concept", response_model=ExplanationResponse)
async def explain_concept(question: str = Form(...), google_api_key: str = Form(...)):
    """
    Endpoint to ask a question. Retrieves relevant context from FAISS 
    and generates an explanation using Gemini.
    """
    global vector_store
    
    # Check if a document has been uploaded yet
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No lecture notes found. Please upload a PDF note first.")

    try:
        # Step 1: Retrieve the most relevant chunks from the uploaded note
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(question)
        
        # Combine the retrieved text into a single context block
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Step 2: Initialize the dynamic LLM
        selected_model = get_valid_model_name(SecretStr(google_api_key))
        
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            api_key=google_api_key,
            temperature=0.5  # Lower temperature ensures factual retrieval
        )
        
        # Step 3: Enforce JSON output matching our Pydantic schema
        structured_llm = llm.with_structured_output(ExplanationResponse, method="json_mode")
        
        # Step 4: Format the prompt and invoke the LLM
        _input = rag_prompt_template.format_prompt(
            context=context_text,
            question=question
        )
        
        print(f"Generating explanation using retrieved context with model {selected_model}...")
        explanation: ExplanationResponse = structured_llm.invoke(_input.to_string()) # type: ignore
        
        # Step 5: Simulate Kafka event logging
        print(f"[KAFKA MOCK] Published event 'session.ended' - Study interaction logged for Concept Explainer.")
        
        return explanation

    except Exception as e:
        # Robust error handling for LLM timeouts or context retrieval failures
        raise HTTPException(status_code=500, detail=f"An error occurred while explaining the concept: {str(e)}")

# Run instructions: uvicorn concept_explainer:app --host 127.0.0.1 --port 8006 --reload