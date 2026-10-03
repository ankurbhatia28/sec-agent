# Status

**Current phase:** 0 — Foundations (code done; smoke verification pending API keys)
**Last updated:** 2026-10-03

## Done
- Repo scaffold: uv package `secrag` (Python 3.12), ruff, pytest, Makefile.
- `src/secrag/llm.py` — alias → LiteLLM call, Anthropic cache breakpoint, disk response cache (`.cache/llm`), Langfuse generation per call (tokens, cache tokens, cost).
- `src/secrag/tracing.py` — `trace()` root span + `propagate_attributes` (session, tags, metadata).
- `config/models.yaml` — Claude Sonnet 5.5 / Haiku 4.5, GPT-6 Luna / 6.1 Sol, Qwen 3.8 Flash / free tier, DeepSeek V4 Pro, local Qwen3 (Ollama).
- `secrag smoke` CLI; 7 unit tests (no network).

## Next
1. Copy `.env.example` → `.env`; add Langfuse Cloud keys + at least two provider keys.
2. `make smoke` twice: first run shows real calls with cost; second shows `[response cache]` hits at $0. Confirm in Langfuse that each smoke run is ONE trace with nested generations.
3. Phase 1 (on the ≥48 GB machine): RAGFlow via Docker — **must build the image locally on Apple Silicon** (no official ARM64 images); edgartools download + XBRL facts; chunk snapshot.

## Phase map
0 Foundations → 1 Corpus & ingestion → 2 Baseline pipeline (+ no-retrieval, RAGFlow chat baselines) → 3 Golden set + error analysis → 4 Code checks + aligned judges → 5 Experiment loop → 6 Agent → 7 Write-ups.
Full plan: `docs/plan.md` (read only the section for the current phase).

## Resume prompt (paste into a fresh session)
> Read docs/STATUS.md. Continue the current phase: <phase>. Read only that phase's section of docs/plan.md. Start in plan mode.
