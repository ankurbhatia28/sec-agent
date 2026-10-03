# SEC 10-K/10-Q Retrieval Agent: Build & Eval Plan

## Context

The project (originally scoped in private overview and decision docs, kept outside the repo) is an eval-first portfolio piece: a cited Q&A agent over SEC filings, measured rigorously using the methods from Hamel Husain's AI Evals course (error analysis, failure-mode-specific binary judges, checking judges against human labels). The directory is currently just those two docs; there's no code and no git repo yet.

Decisions locked in this session (they update the Decision Doc):
- **Architecture: two layers.** RAGFlow is a service for document parsing (its DeepDoc engine), chunking, embeddings, and hybrid search, called through its HTTP API. Our own Python code handles orchestration, generation, the agent, and evals, all traced in **Langfuse** using spans we define. Each RAGFlow call shows up as a step in our trace. RAGFlow's built-in chat runs the same golden set as an off-the-shelf baseline.
- **LLM access:** the LiteLLM library (not its proxy) behind a small `llm.py` wrapper we own, plus `models.yaml` aliases. Providers: OpenRouter (Qwen, DeepSeek, Llama, Gemini, free tiers), Ollama (free local models), and direct Anthropic and OpenAI keys (native caching and Batch API).
- **Corpus:** Big Tech + NVDA. Suggested: NVDA, MSFT, GOOGL, META, AMZN, each with its latest 10-K plus 2–3 10-Qs.
- **Domain expert:** the finance friend is confirmed for a few sessions (question drafting, answer key, judge-alignment labels).

Two corpus points to turn into features of the eval:
1. **Training-data leakage.** Big Tech filings are probably in models' training data. Mitigation: favor the newest 10-Qs (filed after model cutoffs) and run a **no-retrieval baseline** (the model answers from memory). The gap between that and the full system is a headline result.
2. **Fiscal-period mismatch.** NVDA's fiscal year ends in late January and MSFT's in June, while the others follow the calendar year. "Wrong period" is an expected failure mode that cross-company questions will surface.

---

## Guiding principles (the habits this project demonstrates)

1. **Every phase ends in a saved, versioned output on disk.** Any later phase can start fresh from those files without rerunning earlier phases or re-paying for LLM calls.
2. **Look at the data before writing metrics.** No generic out-of-the-box RAG scores (e.g. RAGAS faithfulness) until error analysis shows which failure modes actually occur.
3. **Code checks first, LLM judges only where judgment is needed,** and every judge is checked against human labels before its numbers are reported.
4. **Every setting is a config value.** Model, prompt version, retrieval settings, and chunk snapshot all live in config, so each experiment is just a set of config values.
5. **Never pay twice for the same LLM call:** disk response cache, prompt caching, Batch API for large runs, cheap or local models during development.

---

## Repo layout (target)

```
sec-rag-evals/
  CLAUDE.md                 # conventions + "how to resume" pointer (short)
  docs/STATUS.md            # living handoff: current phase, last outputs, next step
  docs/decisions/           # ADR-style notes (supersedes parts of the Decision Doc)
  config/
    corpus.yaml             # tickers + pinned EDGAR accession numbers
    models.yaml             # alias -> litellm model string, cost, role (gen/judge)
    experiments/*.yaml      # one file per experiment config
  src/secrag/
    llm.py                  # LiteLLM wrapper: Langfuse generation spans, disk cache, cache_control
    tracing.py              # Langfuse setup, @observe helpers, session/trace naming
    edgar.py                # edgartools download + XBRL facts extraction
    ragflow_client.py       # dataset/doc upload, parse polling, chunk export, retrieve()
    pipeline/single_shot.py # retrieve -> generate w/ citations (structured output)
    pipeline/agent.py       # Phase 6 tool-using agent
    evals/                  # code checks, judges, alignment stats, runner
  data/                     # gitignored, reproducible
    raw/                    # downloaded filings (HTML)
    facts/                  # XBRL facts parquet (numeric ground truth)
    chunks/<snapshot_id>.jsonl   # frozen copy of RAGFlow's parsed chunks
  evals/
    golden/golden_v{n}.jsonl     # questions + answer key + evidence chunk IDs
    labels/                      # human labels (open coding, judge alignment)
    failure_taxonomy.md
    judges/*.md                  # versioned judge prompts
    runs/<run_id>/               # outputs, scores, config snapshot, cost
  apps/annotate/            # tiny labeling UI (FastHTML/Streamlit)
  tests/
```

Tooling: `uv` + Python 3.12, `ruff`, `pytest`, `Makefile` targets per phase, `.env` for keys (never committed). RAGFlow runs via its official `docker compose` setup (needs about 16 GB RAM, plus a check for Apple Silicon support). Use **Langfuse Cloud's free tier** rather than self-hosting, so you aren't running two Docker stacks.

---

## Phases (each one can start fresh from saved files)

Each phase lists **Inputs → Work → Output (checkpoint) → Done when**. Start every phase in a fresh Claude Code session, using `docs/STATUS.md` as the handoff.

### Phase 0: Foundations
- `git init`, uv project, CLAUDE.md, STATUS.md, Makefile, `.env.example`.
- `llm.py`: `complete(alias, messages, **kw)` using LiteLLM, wrapped in a Langfuse generation span that records model, tokens, cost, and cached tokens. Adds Anthropic `cache_control` breakpoints when the alias is an Anthropic model, and a disk response cache (LiteLLM's disk cache or our own, keyed by model + prompt + params) that is on by default for evals.
  - Don't use LiteLLM's built-in Langfuse callback. It can create a separate trace per call instead of nesting under our session's trace.
- `models.yaml` with a few aliases: a fast/cheap generator, a strong generator, a local Qwen model (Ollama), judge candidates from a different model family than the generator.
- **Done when:** `make smoke` sends one call through 3 providers and you see 3 nested generations inside one trace in Langfuse, with cost recorded.

### Phase 1: Corpus & ingestion
- `edgar.py` using **edgartools**: download the pinned 10-K/10-Q filings to `data/raw/`, and pull **XBRL financial facts** into `data/facts/*.parquet`. These facts are machine-readable numeric ground truth (revenue, segment figures, EPS, …) from SEC data. Use them for numeric answer keys and later as an agent tool.
- `ragflow_client.py`: create one dataset with document metadata (ticker, form, fiscal year/quarter, period end date) so searches can filter. Upload the files and wait for parsing to finish.
- Try 2 of RAGFlow's chunking methods (e.g. General vs. Manual/Table-oriented) on one filing before committing.
- **Parse-quality spot check (a small eval of its own):** sample about 20 table-derived chunks and compare their numbers to the XBRL facts. Log the error rate. This is the "parsing is where quality is won or lost" result for the write-up.
- Export every chunk from RAGFlow to `data/chunks/<snapshot_id>.jsonl` (chunk ID, doc, position, text, metadata). The snapshot ID goes into every later run config.
- **Done when:** the chunk snapshot exists, the parse spot-check numbers are in `docs/`, and `retrieve("NVDA data center revenue Q2")` returns sensible chunks with similarity scores.

### Phase 2: Baseline pipeline + no-retrieval baseline
- `single_shot.py`: question → (optional metadata filter) → `ragflow_client.retrieve()` (as a span recording chunk IDs and the combined/vector/keyword scores) → generator returns structured output `{answer, citations:[chunk_id], numbers:[{value, unit, period}], abstain:bool}`.
- Prompt layout for caching: fixed system prompt, then citation rules, then few-shot examples first; retrieved chunks and the question last.
- Trace naming: one trace per question, a Langfuse `session_id` per eval run, tags for config hash, pipeline, and golden-set version.
- Two more baselines that use the same interface: **no-retrieval** (model answers from memory) and **RAGFlow's built-in chat** (through its API, with its own Langfuse connector turned on).
- **Done when:** a CLI `secrag ask "..."` returns a cited answer and the trace shows retrieval + generation nested in one trace.

### Phase 3: Golden set + error analysis (the core of the project)
Follow the course method, in order:
1. **Dimensions session with your friend** (about 90 min): agree on the question dimensions. Suggested: *question type* (fact lookup / numeric-table / cross-period / cross-company / unanswerable), *where the answer lives* (narrative / table / footnote / MD&A), *company*, *difficulty*. Hand-write 30–50 real analyst questions.
2. **Synthetic expansion:** generate combinations of dimension values, have an LLM turn them into natural-language questions, and have your friend filter them. Target about 100 questions total. Include about 10% **unanswerable** questions, to test abstaining and refusing to make things up.
3. **Answer key:** for each question, record the answer text, the numbers (value, unit, period, tolerance), and the evidence (filing plus chunk IDs from the frozen snapshot, and an XBRL fact ID where it applies). Use a quick LLM draft from the XBRL facts, then have your friend verify it. Save as `golden_v1.jsonl`.
4. Run the baseline on all questions, then **open coding**: read the traces in `apps/annotate` (question, answer, retrieved chunks, answer key side by side; pass/fail + free-text note). Use Langfuse annotation queues for overflow. Keep going until new traces stop revealing new failure types (usually 60–100 traces).
5. **Axial coding:** group the notes into a failure list, e.g. *retrieval miss*, *wrong period/fiscal-year confusion*, *right number, wrong line item*, *unit/scale error (millions vs. billions)*, *unsupported claim*, *incomplete comparison*, *failed to abstain*, *bad citation*. Count how often each happens and save to `evals/failure_taxonomy.md`.
- **Done when:** golden v1 is frozen, the failure list has counts, and the top 3 failure modes are named. **This alone is write-up #1 material.**

### Phase 4: Automated evaluators + judge alignment
- **Code checks** (cheap, run every time): retrieval recall@k and MRR (how high the first right chunk ranks) against the evidence chunk IDs; citation validity (cited IDs exist and were retrieved); numeric match within tolerance against the answer key and XBRL; period match; abstain correctness on unanswerable questions; cost and latency per question.
- **Binary LLM judges**, one per failure mode that code can't check (e.g. *unsupported claim*, *incomplete comparison*, *wrong-line-item reasoning*). Each judge returns pass/fail with a short critique. Prompts live in `evals/judges/*.md` and are versioned.
- **Checking judges against human labels:** your friend labels about 100–150 traces for each judged failure mode. Split them 20/40/40 into train (few-shot examples), dev (prompt tuning), and test (touched once). Report how often the judge catches real failures (**true positive rate**) and how often it correctly passes good answers (**true negative rate**). Report **corrected pass rates with bootstrap confidence intervals**, adjusting the raw judge pass rate for the judge's known error rates. Compare judge models and pick the **cheapest judge that meets the bar** (e.g. both rates ≥ 0.85).
- **Done when:** `make eval RUN=<config>` produces `evals/runs/<run_id>/report.md` with per-failure-mode rates, intervals, and cost, and there's a judge alignment report.

### Phase 5: Experiment loop (improve by evidence)
- Make the runner work with Langfuse Datasets/Experiments so runs can be compared side by side in the UI. Large runs go through the Anthropic/OpenAI **Batch APIs** (about 50% off).
- Change one setting at a time, chosen to target the top failure modes: chunking method, RAGFlow's keyword/vector weight, top-k, reranker on/off, metadata filtering (fixes wrong-period), query rewriting, generator model (cost vs. quality chart across Claude, GPT, Qwen, DeepSeek).
- Keep a small **regression subset** (about 20 questions, code checks only, cached responses) that runs in `pytest` and CI.
- **Done when:** there's a before/after table per failure mode, plus a cost vs. quality chart.

### Phase 6: Multi-step agent
- `agent.py` (a plain tool-calling loop through `llm.py`; only consider a framework like LangGraph if the loop gets messy). Tools: `list_filings(ticker, form, period)`, `search_filings(query, filters)` (RAGFlow), `get_financial_fact(ticker, concept, period)` (XBRL facts), and `calculate(expr)`. Each tool call is a span.
- Agent-specific evals: right tool chosen, step count, cost per question, and a **failure-at-which-step matrix** (where in the loop it first goes wrong).
- Key comparison: agent vs. single-shot vs. "agent without the XBRL tool," broken down by question type. Expected: big gains on cross-period/cross-company questions, extra cost on simple lookups. Measure it, don't assume it.
- **Done when:** the same eval suite runs on all variants and there's a results table by question type.

### Phase 7: Write-up
- Write-up #1 after Phase 4 (baseline, failure list, judge alignment, leakage result). Write-up #2 after Phase 6 (experiments, agent results, cost). Publish them as artifacts or a blog post. Include an honest-limitations section.

---

## Using context and caching efficiently

**Inside the app (LLM spend):**
- Prompt layout: fixed content first, changing content last. Anthropic caching uses explicit `cache_control` (set by `llm.py`); OpenAI, DeepSeek, and Qwen cache repeated prompt openings automatically. Cache-hit tokens are recorded on Langfuse spans, so savings show up in the reports.
- Disk response cache: iterating on judge prompts or analysis never re-pays for unchanged generator calls. Retrieval results are cached per (query, params, snapshot).
- Batch APIs for full eval runs; cheap or local models (Ollama Qwen) for development and smoke tests; strong models only for final runs.
- Retrieve less but better: metadata filters and top-k tuning cut context tokens as well as errors.

**In Claude Code (building it):**
- One phase per session, starting with `/clear`. `CLAUDE.md` stays short (conventions, commands, a pointer to `docs/STATUS.md`). STATUS.md is updated at the end of every session with what's done, saved outputs, and the next step, so a new session restarts quickly.
- Use plan mode at the start of each phase. Use subagents only for genuinely broad searches. Commit small and often. Run `/code-review` before closing a phase.
- Large files (filings, chunk snapshots, run outputs) stay on disk. Inspect them with targeted commands (`jq`, `head`, small scripts), never by pasting them into the conversation.

## Off-the-shelf tools (don't build what you don't have to)

| Need | Tool |
|---|---|
| EDGAR download + XBRL financial facts | `edgartools` |
| Layout/table-aware parsing, chunking, embeddings, hybrid search, rerank | RAGFlow (DeepDoc), called through its API |
| Unified LLM access + cost tracking | LiteLLM library, OpenRouter, Ollama |
| Tracing, datasets, experiments, annotation queues, scores | Langfuse Cloud |
| Optional parse comparison | Docling on the same tables (a parsing ablation, only if Phase 1 shows table errors) |

## Verification
- Phase 0: `make smoke` → nested multi-provider trace with costs in Langfuse.
- Phase 1: `make ingest` is safe to rerun; chunk snapshot row counts match the docs; parse spot-check report exists.
- Phase 2+: `secrag ask` works end to end; `pytest` covers the code checks (numeric tolerance, citation validity, recall@k) using fixed test inputs.
- Phase 4+: `make eval RUN=baseline` reproduces the same scores from the cache with no new LLM calls (proves the cache and reproducibility work); judge alignment report is checked in.
- After each phase: STATUS.md is updated, so a fresh session can resume using only the saved files.

## First session after approval
Phase 0 only: scaffold the repo, `llm.py`, `tracing.py`, `models.yaml`, and `make smoke`. Plus `docs/decisions/0001-architecture.md` recording the choices above. Needs from you: Langfuse keys, an OpenRouter key, Anthropic/OpenAI keys, and Ollama installed (optional).
