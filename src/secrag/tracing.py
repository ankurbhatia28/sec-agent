"""Langfuse setup and the root-trace context manager.

Every unit of work (one question, one smoke run, one judge call batch) opens a root
span via `trace(...)`; spans and generations created inside it nest automatically
through OpenTelemetry context. Without LANGFUSE_* keys the client is a no-op.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langfuse import get_client, propagate_attributes

from secrag import settings  # noqa: F401  (loads .env before the client is created)


def langfuse():
    return get_client()


@contextmanager
def trace(
    name: str,
    *,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
    input: Any = None,
) -> Iterator[Any]:
    """Open a root span named `name` and propagate session/tags/metadata to all children."""
    with langfuse().start_as_current_observation(name=name, as_type="span", input=input) as span:
        with propagate_attributes(
            trace_name=name, session_id=session_id, tags=tags, metadata=metadata
        ):
            yield span


def flush() -> None:
    """Call at the end of short-lived scripts so buffered spans are exported."""
    langfuse().flush()
