"""The single entry point for LLM calls.

`complete(alias, messages)` resolves an alias from config/models.yaml, adds an Anthropic
prompt-cache breakpoint at the end of the stable system prefix, serves repeat calls from
a disk cache, and records every call as a Langfuse generation (tokens, cache tokens, cost)
nested under whatever span is current.

LiteLLM's own Langfuse callback is deliberately not used: it can emit each call as a
separate trace instead of nesting it under ours.
"""

import copy
import hashlib
import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import diskcache
import litellm
import yaml

from secrag import settings
from secrag.tracing import langfuse

log = logging.getLogger(__name__)
litellm.suppress_debug_info = True


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model: str
    family: str
    roles: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def provider(self) -> str:
        return self.model.split("/", 1)[0]


@dataclass
class LLMResult:
    text: str
    alias: str
    model: str
    usage: dict[str, int]
    cost_usd: float | None
    response_cache_hit: bool
    raw: dict[str, Any]


@lru_cache
def load_models(path: Path = settings.CONFIG_DIR / "models.yaml") -> dict[str, ModelSpec]:
    entries = yaml.safe_load(path.read_text())["models"]
    return {
        alias: ModelSpec(
            alias=alias,
            model=e["model"],
            family=e["family"],
            roles=tuple(e.get("roles", [])),
            params=e.get("params") or {},
        )
        for alias, e in entries.items()
    }


def get_model(alias: str) -> ModelSpec:
    models = load_models()
    if alias not in models:
        raise KeyError(f"Unknown model alias {alias!r}; known: {sorted(models)}")
    return models[alias]


def add_cache_breakpoint(messages: list[dict]) -> list[dict]:
    """Mark the end of the last system message as an Anthropic cache breakpoint.

    The breakpoint goes after the stable prefix (system prompt, rules, few-shots), never
    after the per-request tail (retrieved chunks, question), so writes are actually re-read.
    Returns a copy; the input is not mutated.
    """
    out = copy.deepcopy(messages)
    system_idxs = [i for i, m in enumerate(out) if m.get("role") == "system"]
    if not system_idxs:
        return out
    msg = out[system_idxs[-1]]
    content = msg["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
    msg["content"] = content
    return out


def cache_key(model: str, messages: list[dict], params: dict[str, Any], sample: int) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "params": params, "sample": sample},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    """Map provider-specific usage fields onto one schema (also used for Langfuse usage_details)."""
    usage = usage or {}
    details = usage.get("prompt_tokens_details") or {}
    cache_read = usage.get("cache_read_input_tokens") or details.get("cached_tokens") or 0
    return {
        "input": usage.get("prompt_tokens") or 0,
        "output": usage.get("completion_tokens") or 0,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens") or 0,
    }


@lru_cache
def _response_cache() -> diskcache.Cache:
    return diskcache.Cache(str(settings.CACHE_DIR / "llm"))


def complete(
    alias: str,
    messages: list[dict],
    *,
    name: str = "llm",
    use_cache: bool = True,
    sample: int = 0,
    **params: Any,
) -> LLMResult:
    """Call `alias` with `messages`.

    `sample` > 0 gives an independent cached draw (for variance runs).
    """
    spec = get_model(alias)
    call_params = {**spec.params, **params}
    if spec.provider == "anthropic":
        messages = add_cache_breakpoint(messages)
    key = cache_key(spec.model, messages, call_params, sample)
    cache = _response_cache()

    with langfuse().start_as_current_observation(
        name=name,
        as_type="generation",
        model=spec.model,
        input=messages,
        model_parameters=call_params,
    ) as gen:
        hit = use_cache and key in cache
        if hit:
            entry = cache[key]
        else:
            response = litellm.completion(model=spec.model, messages=messages, **call_params)
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:  # model missing from LiteLLM's cost map
                log.warning("No price for %s; cost not recorded", spec.model)
                cost = None
            entry = {"response": response.model_dump(), "cost": cost}
            if use_cache:
                cache[key] = entry

        raw = entry["response"]
        text = raw["choices"][0]["message"].get("content") or ""
        usage = normalize_usage(raw.get("usage"))
        # A cached response costs nothing now; the original cost stays in metadata.
        cost_now = 0.0 if hit else entry["cost"]
        gen.update(
            output=text,
            usage_details=usage,
            cost_details={"total": cost_now} if cost_now is not None else None,
            metadata={
                "alias": alias,
                "response_cache_hit": hit,
                "original_cost_usd": entry["cost"],
            },
        )

    return LLMResult(
        text=text,
        alias=alias,
        model=spec.model,
        usage=usage,
        cost_usd=cost_now,
        response_cache_hit=hit,
        raw=raw,
    )
