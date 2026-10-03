# SEC RAG Evals

Eval-first retrieval agent over SEC 10-K/10-Q filings. When resuming, read `docs/STATUS.md` first; it points to the relevant section of `docs/plan.md`. Architecture choices: `docs/decisions/`.

## Commands
- `make install` / `make lint` / `make test`
- `make smoke` — one traced call per provider (aliases without keys are skipped)
- `uv run secrag <cmd>` — CLI entry point (`src/secrag/cli.py`)

## Conventions
- Every LLM call goes through `secrag.llm.complete(alias, messages)`; aliases live in `config/models.yaml`. Never call litellm or a provider SDK directly.
- Every unit of work runs inside `secrag.tracing.trace(...)` so spans nest in one Langfuse trace. Don't enable LiteLLM's Langfuse callback.
- Prompts: stable content (system, rules, few-shots) in the system message; per-request content (retrieved chunks, question) in the user message. `llm.py` puts the Anthropic cache breakpoint at the end of the system message.
- Judges must come from a different model family than the generator under test.
- Each phase ends with a saved, versioned output (`data/chunks/<snapshot>.jsonl`, `evals/golden/golden_vN.jsonl`, `evals/runs/<run_id>/`). Later phases read these files, never earlier phases' code paths.
- Large files (filings, chunk snapshots, run outputs) stay on disk; inspect with `jq`/`head`, don't paste into context.
- Update `docs/STATUS.md` at the end of every working session.
- Model IDs and prices: `config/models.yaml` + `litellm.model_cost[...]`; check there before loading external API docs.
