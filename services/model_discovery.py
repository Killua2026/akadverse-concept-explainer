from google import genai


MODEL_PRIORITY_ORDER = (
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
)
DEFAULT_MODEL = MODEL_PRIORITY_ORDER[0]


def get_valid_model_name(api_key_str: str) -> str:
    """Dynamically discovers the most cost-efficient Gemini generative model

    by calling client.models.list() and matching against a budget-first priority
    list.

    Uses the configured Gemini 3.x priority order to avoid restricted
    models on new projects/accounts.

    Falls back to the first model in MODEL_PRIORITY_ORDER if discovery fails.
    """
    try:
        client = genai.Client(api_key=api_key_str)

        # Collect all available model names, stripping the "models/" prefix
        all_models = [
            m.name.replace("models/", "")
            for m in client.models.list()
            if m.name
        ]

        # Prioritize the configured 3.x models to avoid limited-access models
        for preferred in MODEL_PRIORITY_ORDER:
            if preferred in all_models:
                print(
                    f"[Model] Using cost-efficient generative model: {preferred}"
                )
                return preferred

        if all_models:
            print(f"[Model] Fallback to first available: {all_models[0]}")
            return all_models[0]

    except Exception as e:
        print(
            f"[Model WARNING] Discovery failed ({e}). Defaulting to '{DEFAULT_MODEL}'."
        )

    return DEFAULT_MODEL
