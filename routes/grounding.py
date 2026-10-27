import re
import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Form
from pydantic import BaseModel, Field
from google.genai import types

from auth import require_ingest_auth
from config import GeminiSettings
from services.gemini_provider import create_client


class GroundingRouter(APIRouter):
    selected_model: str | None = None

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


def make_grounding_router(api_key, *, settings: GeminiSettings | None = None):
    router = GroundingRouter()
    provider_settings = settings or GeminiSettings(google_api_key=api_key or "")

    @router.post("/explain-with-grounding", response_model=GroundedExplanationResponse, tags=["Q&A"], dependencies=[Depends(require_ingest_auth)])
    async def explain_with_grounding(
        question: str = Form(..., description="The concept or question you want explained from the web.")
    ):
        """
        Explains a concept using Gemini's native Google Search Grounding tool.

        Searches the live web and synthesises an answer from current sources.
        Requires shared-secret authentication and ready ingestion configuration.

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
            selected_model = router.selected_model or provider_settings.model_names[0]
            # Reuse startup discovery for the selected provider.
            print(f"[Grounding] Using model: {selected_model} with Google Search grounding...")

            # -- Step 2: Format the prompt that asks Gemini to include a JSON block --
            prompt = GROUNDING_PROMPT_TEMPLATE.format(question=question)

            # -- Step 3: Call Gemini with the Search Grounding tool enabled --
            # types.Tool(google_search=types.GoogleSearch()) is the v1.2+ SDK syntax.
            # This is different from the older SDK which used tools='google_search_retrieval'.
            # The grounding tool makes Gemini autonomously search Google during generation.
            client = create_client(provider_settings)
            if client is None:
                raise RuntimeError("Gemini provider is not configured.")
            response = await client.aio.models.generate_content(
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


    return router
