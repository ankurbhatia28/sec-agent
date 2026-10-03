# 0001 — Architecture and stack

**Date:** 2026-10-03 · **Status:** accepted · Supersedes parts of the original (private) planning decision doc.

## Decisions
1. **Two layers.** RAGFlow is a parsing/chunking/embedding/hybrid-search *service* called over its HTTP API (`/api/v1/retrieval`). Our code owns orchestration, generation, the agent, and evals.
   - *Why:* use DeepDoc's document understanding without giving up control of traces. Each RAGFlow call is our own Langfuse span recording chunk IDs, source document, and combined/vector/keyword scores — enough for retrieval error analysis and recall@k. RAGFlow's internal scoring stays opaque; we judge retrieval by its output.
   - RAGFlow's built-in chat (with its native Langfuse connector) runs the same golden set as an off-the-shelf baseline.
2. **Tracing: Langfuse Cloud (free tier),** spans we define via the Langfuse v4 SDK. Not self-hosted, to avoid a second Docker stack alongside RAGFlow.
3. **LLM access: LiteLLM SDK (not proxy) behind `secrag.llm`,** aliases in `config/models.yaml`. Direct Anthropic/OpenAI keys (native caching, Batch API), OpenRouter for Qwen/DeepSeek/others, Ollama for local models. LiteLLM's Langfuse callback is off because it can create a separate trace per call.
4. **Corpus:** Big Tech + NVDA (NVDA, MSFT, GOOGL, META, AMZN); latest 10-K + 2–3 10-Qs each. Eval features: a no-retrieval baseline measures training-data leakage; fiscal-year misalignment (NVDA ends late Jan, MSFT ends June) is an expected "wrong period" failure mode.
5. **Numeric ground truth from XBRL** via `edgartools`; also a candidate agent tool in Phase 6.
6. **Hardware:** RAGFlow needs ≥16 GB RAM and has no official ARM64 images. It runs on the incoming ≥48 GB Apple Silicon machine with a locally built image.

## Eval method (from Hamel Husain's AI Evals course)
Error analysis before metrics (open coding → axial coding into a failure list) → code checks first → one binary LLM judge per failure mode that needs judgment → judges checked against expert labels (train/dev/test split, true positive and true negative rates) → corrected pass rates with bootstrap confidence intervals.
