"""Synchronous Gemini client construction and Enterprise model discovery."""

import logging
from typing import Any

from google import genai

from config import GeminiSettings
from services.model_discovery import MODEL_PRIORITY_ORDER

logger = logging.getLogger(__name__)

ENTERPRISE_MODEL_PRIORITY = [
    "gemini-3.1-flash-lite",   # Most cost-efficient Gemini model, optimized for low latency use cases for high-volume, cost-sensitive LLM traffic.
    "gemini-3.5-flash-lite",   # Highly cost-efficient, high-volume production standard
    "gemini-3.5-flash",        # Higher-throughput mid-tier
    "gemini-3.8-flash",        # Flagship standard utility fallback
]
MEDIA_ONLY_MODEL_MARKERS = (
    "audio", "image", "imagen", "lyria", "music", "veo", "video",
)


def create_client(settings: GeminiSettings) -> genai.Client | None:
    """Construct the selected provider client, or None without configuration."""
    if settings.use_enterprise:
        # Note: Search Grounding on Gemini 3.x requires location='global'. Embeddings for sub-phase 2b2 must be verified at this location too.
        return genai.Client(
            enterprise=True,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )
    if settings.google_api_key:
        return genai.Client(api_key=settings.google_api_key)
    return None


def _model_id(model: Any) -> str:
    name = str(getattr(model, "name", "") or "").strip()
    return name.removeprefix("models/").rstrip("/").split("/")[-1]


def _supports_chat_generation(model: Any, model_id: str) -> bool:
    normalized_id = model_id.lower().replace("_", "-")
    if any(marker in normalized_id for marker in MEDIA_ONLY_MODEL_MARKERS):
        return False
    supported_actions = getattr(model, "supported_actions", None)
    if supported_actions is None and isinstance(model, dict):
        supported_actions = model.get("supported_actions") or model.get("supportedActions")
    if supported_actions is None:
        return True
    actions = {str(action).replace("_", "").lower() for action in supported_actions}
    return "generatecontent" in actions


def discover_enterprise_model_names(
    project: str,
    location: str,
    preferred_model: str | None = None,
    fallback_models: list[str] | None = None,
) -> list[str]:
    """Synchronously discover and rank Enterprise Gemini text model IDs."""
    priority: list[str] = []
    for candidate in [preferred_model, *(fallback_models or []), *ENTERPRISE_MODEL_PRIORITY]:
        if candidate and candidate not in priority:
            priority.append(candidate)
    if not project or not location:
        logger.warning("Enterprise discovery skipped; project or location missing; falling back to priority list.")
        return priority
    try:
        client = genai.Client(enterprise=True, project=project, location=location)
        discovered = {
            model_id
            for model in client.models.list()
            if (model_id := _model_id(model))
            and "gemini" in model_id.lower()
            and "flash" in model_id.lower()
            and _supports_chat_generation(model, model_id)
        }
        ranked = [candidate for candidate in priority if candidate in discovered]
        ranked.extend(sorted(discovered - set(ranked)))
        if not ranked:
            logger.warning("No eligible Enterprise models discovered; falling back to priority list.")
        return ranked or priority
    except Exception:
        # SDK exception text may contain credentials, so do not include it.
        logger.warning("Enterprise model discovery: falling back to priority list; discovery failed.")
        return priority
