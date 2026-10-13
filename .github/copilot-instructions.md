# AkadVerse Concept Explainer — Copilot Instructions

## What this is
FastAPI microservice, port 8010. RAG study assistant. Two modes:
- /explain-concept — answers from the student's uploaded PDF
- /explain-with-grounding — answers from live Google Search

## Locked decisions (do NOT change)
- The Google API key must be read once at startup from
  `os.environ["GOOGLE_API_KEY"]`. It must NEVER be accepted as a
  request parameter, form field, or query string.
- Auth will later come from an `X-User-Id` header — do not add it yet,
  but do not add any other auth mechanism either.
- The five-field response schema is frozen:
  concept, simple_explanation, analogy, example, worked_problem.
- Do not rename existing endpoints or change their response models
  unless explicitly asked.

## Code standards
- Use `logging`, not `print`, in any new code.
- Do not introduce new dependencies.
- Prefer editing the existing file over creating new modules.