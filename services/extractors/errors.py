"""Safe extraction failures suitable for job error messages."""


class ExtractionError(ValueError):
    """A document could not yield usable text."""
