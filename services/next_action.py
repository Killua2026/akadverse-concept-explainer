import json
import logging
import re
from typing import Any, Literal

from google.genai import types
from pydantic import BaseModel

from config import GeminiSettings
from models import Decision, NextActionResponse
from services.gemini_provider import create_client

logger = logging.getLogger(__name__)

# Seventeen numbered principles and six hard rules.
NEXT_ACTION_PROMPT_TEMPLATE = """You are the AkadVerse Learning Orchestrator. Your only job is to decide ONE next teaching action for a student, based on their learning context and latest evidence. You do not teach, explain, or generate content \u2014 you produce a structured routing decision.

Current event: {trigger}

Student learning context (snapshot from the backend):
{student_context}

Current lesson state:
{current}

Latest evidence from the most recent interaction:
{latest_evidence}

Actions the backend currently allows you to choose (you MUST pick from this list):
{allowed_actions}

Policy:
{policy}

Decision principles, applied in order of precedence:

1. SAFETY FIRST. If any latest_evidence entry has needs_review=true, return FLAG_FOR_REVIEW. This overrides all other rules.

2. PREREQUISITE GAP. If the current concept has mastery DEVELOPING or lower AND latest_evidence shows the student struggling with a foundational idea (recurring misconception or prerequisite with mastery NEEDS_REINFORCEMENT), return TEACH_PREREQUISITE with target_concept_id set to the specific prerequisite's concept_id.

3. DIAGNOSED MISCONCEPTION. If latest_evidence contains at least one entry in misconceptions[] on an incorrect or partial result, return ADDRESS_MISCONCEPTION targeting the current concept. Set params.misconception_code if the evidence names one.

4. REASONING GAP. If latest_evidence shows result=correct but reasoning_quality is weak or absent, return REASONING_EXERCISE with params.focus="reasoning".

5. INDEPENDENT SUCCESS. If latest_evidence shows result=correct AND
   reasoning_quality is strong or adequate AND BOTH of the following:
     - assistance.hints_used is 0 or absent
     - assistance.assisted is false or absent
   this is independent mastery:
     - If mastery is DEVELOPING or higher and a transfer opportunity
       makes sense, return TRANSFER_TASK with params.difficulty="hard".
     - Otherwise, return ADVANCE_TO_NEXT_CONCEPT.
   In both cases, add "CONCEPT_READY_FOR_REVISION" to signals.

6. ASSISTED SUCCESS. If result=correct but EITHER
     - assistance.hints_used is greater than 0, OR
     - assistance.assisted is true,
   do NOT treat this as independent mastery. Return GENERATE_PRACTICE
   with params.assistance_target="independent".

7. GUIDED FAILED. If trigger is GUIDED_COMPLETED and the guided tasks were not all correct, return SHOW_GUIDED_EXAMPLE with params.difficulty lower than the previous attempt, or REINFORCE_CONCEPT.

8. GUIDED SUCCEEDED. If trigger is GUIDED_COMPLETED and all tasks were correct with low assistance, return GENERATE_PRACTICE.

9. REPEATED FAILURE. If latest_evidence contains 2 or more consecutive incorrect results with no clear misconception to target, return FLAG_FOR_REVIEW.

10. REVISION REVIEW FAILED. If trigger is REVISION_REVIEW_COMPLETED and the review failed, return REINFORCE_CONCEPT for the concept associated with the failed review.

11. CONVERSATION. If trigger is STUDENT_ASKED, return CONTINUE_CONVERSATION.

12. HINT REQUESTED. If trigger is HINT_REQUESTED, return GIVE_HINT with params.hint_level set based on policy.hint_levels_shown so far (if available) or default 1.

13. CONCEPT OPENED (NEW). If trigger is CONCEPT_OPENED and the concept's coverage is NOT_STARTED or missing, return EXPLAIN_CONCEPT.

14. CONCEPT OPENED (IN PROGRESS). If trigger is CONCEPT_OPENED and coverage is EXPLAINED but no practice evidence exists, return SHOW_GUIDED_EXAMPLE or GENERATE_PRACTICE depending on stage.

15. RETURN VISIT. If trigger is RETURN_VISIT, prefer REINFORCE_CONCEPT or GENERATE_PRACTICE with a lower difficulty to re-establish confidence before advancing.

16. LESSON COMPLETE. If trigger is LESSON_COMPLETED, return COMPLETE_LESSON and add "CONCEPT_READY_FOR_REVISION" to signals if any concept was mastered in this lesson.

17. FALLBACK. If none of the above produce a clear choice, return FLAG_FOR_REVIEW with reason_code="NO_CLEAR_ACTION".

Hard rules that override every principle above:

A. action MUST be in the allowed_actions list. If the preferred action from the rules above is not allowed, choose the closest allowed action.

B. target_concept_id MUST be one of: the current concept's concept_id, a prerequisite's concept_id, or a next_concept's concept_id. Never invent an id.

C. If policy.max_consecutive_reinforce is reached (recent decisions on this concept have all been REINFORCE_CONCEPT up to the limit) and you would otherwise return REINFORCE_CONCEPT, return FLAG_FOR_REVIEW instead.

D. rationale is 1 sentence, no student PII, no question text, no lecture content. Purpose is audit traceability, not student-facing.

E. confidence is your honest confidence 0.0-1.0. Use 0.0-0.3 for uncertain decisions, 0.4-0.6 for moderate, 0.7-1.0 for clear-cut. Under 0.5 triggers review logging on the backend.

F. signals is a list of strings. For v1, the only valid signal is "CONCEPT_READY_FOR_REVISION". Emit it when a concept has been independently demonstrated (principle 5) or when the lesson completes (principle 16).

Return JSON matching this schema:
{{ "action": "...", "target_concept_id": "...", "params": {{}},
   "reason_code": "...", "rationale": "...", "confidence": 0.0,
   "signals": [] }}
"""

NEXT_ACTION_PROMPT_VERSION = "next-action-v1"


class _DecisionLLM(BaseModel):
    action: Literal[
        "EXPLAIN_CONCEPT",
        "REINFORCE_CONCEPT",
        "SHOW_GUIDED_EXAMPLE",
        "GIVE_HINT",
        "GENERATE_PRACTICE",
        "ADDRESS_MISCONCEPTION",
        "REASONING_EXERCISE",
        "TRANSFER_TASK",
        "TEACH_PREREQUISITE",
        "ADVANCE_TO_NEXT_CONCEPT",
        "COMPLETE_LESSON",
        "CONTINUE_CONVERSATION",
        "RECAP",
        "FLAG_FOR_REVIEW",
    ]
    target_concept_id: str
    params: dict[str, Any]
    reason_code: str
    rationale: str
    confidence: float
    signals: list[str]


class _NextActionRejected(ValueError):
    """A model response failed validation with a log-safe reason."""


# Preserve the sibling JSON extractor verbatim.
_PlanRejected = _NextActionRejected


def _extract_json_object(raw_text: str) -> dict:
    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL | re.IGNORECASE):
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    stripped = raw_text.lstrip()
    if stripped.startswith("["):
        raise _PlanRejected("expected_json_object")
    for match in re.finditer(r"\{", stripped):
        try:
            data, _ = decoder.raw_decode(stripped, match.start())
        except json.JSONDecodeError:
            continue
        return data
    raise _PlanRejected("no_json_object")


def _validate_decision(parsed: _DecisionLLM, current: dict[str, Any], allowed_actions: list[str]) -> None:
    if parsed.action not in allowed_actions:
        raise _NextActionRejected("action_not_allowed")

    concepts = [current.get("concept")]
    for field in ("prerequisites", "next_concepts"):
        entries = current.get(field)
        if isinstance(entries, list):
            concepts.extend(entries)
    if not any(
        isinstance(concept, dict) and concept.get("concept_id") == parsed.target_concept_id
        for concept in concepts
    ):
        raise _NextActionRejected("target_not_reachable")

    if not parsed.rationale.strip():
        raise _NextActionRejected("rationale_empty")


async def decide_next_action(
    settings: GeminiSettings,
    *,
    request_id: str,
    schema_version: str,
    operation: str,
    trigger: str,
    student_context: dict[str, Any],
    current: dict[str, Any],
    latest_evidence: list[dict[str, Any]],
    allowed_actions: list[str],
    policy: dict[str, Any],
) -> NextActionResponse:
    model = ""
    reason = "prompt_build_failed"
    try:
        model_names = getattr(settings, 'model_names', ())
        if model_names:
            model = model_names[0]
        snapshot = json.dumps(student_context, separators=(",", ":"))
        if len(snapshot) > 20_000:
            logger.warning(
                "next_action.snapshot_truncated original_length=%d truncated_to=20000",
                len(snapshot),
            )
            snapshot = snapshot[:19_900] + "\n\n[NOTE: context truncated at 20000 chars; tail omitted due to size limit]"
        prompt = NEXT_ACTION_PROMPT_TEMPLATE.format(
            trigger=json.dumps(trigger, indent=2),
            student_context=snapshot,
            current=json.dumps(current, indent=2),
            latest_evidence=json.dumps(latest_evidence, indent=2),
            allowed_actions=json.dumps(allowed_actions, indent=2),
            policy=json.dumps(policy, indent=2),
        )

        reason = "generation_failed"
        client = create_client(settings)
        if client is None:
            raise RuntimeError("Gemini client is unavailable")
        if not model:
            raise RuntimeError("No Gemini model is configured")
        async with client.aio as api:
            try:
                response = await api.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                        response_schema=_DecisionLLM,
                    ),
                )
            except Exception as error:
                logger.warning(
                    "next_action.schema_attempt_failed error_type=%s", type(error).__name__,
                )
                response = await api.models.generate_content(
                    model=model,
                    contents=prompt + "\nJSON schema:\n" + json.dumps(_DecisionLLM.model_json_schema()),
                    config=types.GenerateContentConfig(temperature=0.2),
                )

        reason = "decision_processing_failed"
        metadata = getattr(response, "usage_metadata", None)
        usage = {
            "model": model,
            "input_tokens": getattr(metadata, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(metadata, "candidates_token_count", 0) or 0,
            "model_calls": 1,
        }
        data = getattr(response, "parsed", None)
        if data is None:
            data = _extract_json_object(response.text or "")
        parsed = _DecisionLLM.model_validate(data)

        try:
            _validate_decision(parsed, current, allowed_actions)
        except _NextActionRejected as error:
            logger.warning("next_action.decision_rejected reason_code=%s", error)
            raise

        return NextActionResponse(
            request_id=request_id,
            schema_version=schema_version,
            status="ok",
            decision=Decision(
                action=parsed.action,
                target_concept_id=parsed.target_concept_id,
                params=parsed.params or {},
                reason_code=parsed.reason_code,
                rationale=parsed.rationale,
                confidence=parsed.confidence,
            ),
            signals=parsed.signals or [],
            usage=usage,
            model_version=model,
            prompt_version=NEXT_ACTION_PROMPT_VERSION,
        )
    except Exception as error:
        if isinstance(error, _NextActionRejected):
            reason = str(error)
        logger.warning(
            "next_action.fallback error_type=%s reason=%s", type(error).__name__, reason,
        )
        return NextActionResponse(
            request_id=request_id,
            schema_version=schema_version,
            status="degraded",
            decision=None,
            signals=[],
            usage={"model_calls": 0},
            model_version=model,
            prompt_version=NEXT_ACTION_PROMPT_VERSION,
        )
