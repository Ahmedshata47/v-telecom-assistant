# Vodafone Signal Intelligence — telecom agent demo

Web demo of the agent graph in the last cell of `..\voda.ipynb`, **with RAGAS
removed**. It runs entirely on the local corpus; the only network call is the
Mistral retrieval critic.

The point of this build is that a developer can watch the agent work: whether
RAG was used at all, exactly which 3GPP chunks came back, what the critic
thought of them, and whether a corrective retry happened before an answer was
allowed.

## Run it

```powershell
cd C:\Users\ASUS\Desktop\3gpp_specs\demo
.\run_demo.ps1 -MistralApiKey "your-key"
# then open http://127.0.0.1:8000
```

`run_demo.ps1` uses the `tf` Conda environment. To point at a different corpus
or model, set `TELECOM_ROOT` and `BASE_MODEL` before starting.

The Mistral key is only needed for the critic. It is read from
`MISTRAL_API_KEY` and is never written to disk.

## Test it

```powershell
C:\Users\ASUS\miniconda3\envs\tf\python.exe test_demo.py
```

Eight checks always run and need no key: RAGAS absence, graph topology,
retrieval constants, citation validation, the four critic branch conditions,
the second RRF pass, the run-summary counters, and a real retrieval against
the local ChromaDB. Eight end-to-end checks (RAG route, TS filter, DIRECT route,
critic rejection and retry, quality flags, router discrimination, HTTP payload,
agent load) run automatically once `MISTRAL_API_KEY` is set, and are skipped
with a printed notice otherwise. Last full run: **16 passed, 0 failed**.

## Architecture

Ported unchanged from the notebook:

```
START -> route -> retrieve -> evaluate_retrieval -> generate -> END
                  ^               |
                  |               +-- prepare_retry -> retrieve_retry --+
                  +---------------------------------------------------------+
```

| Node | Notebook | What it does |
| --- | --- | --- |
| `route` | §20 | Local Qwen answers `RAG: YES` / `RAG: NO`. No RAG goes straight to generation. |
| `retrieve` | §23 | Hybrid retrieval: dense + BM25, fused with RRF, cut to `TOP_K=5`. A TS number in the question restricts both arms to that spec. |
| `evaluate_retrieval` | §24 | Mistral `codestral-2508` returns `{relevant, score, missing_information, search_queries}`. |
| `prepare_retry` | §26 | Records the critic's queries and bumps `retry_count`. |
| `retrieve_retry` | §27 | Retrieves the original question plus every critic query, then a second query-level RRF merges them. |
| `generate` | §28 | Answers from context only. Citations are validated against retrieved metadata and never generated. |

Config matches the notebook: `TOP_K=5`, `CANDIDATE_K=40`, `BM25_CANDIDATE_K=40`,
`RRF_K=60`, `MAX_RETRIEVAL_ATTEMPTS=2`.

Removed: §34 RAGAS (`context_precision`, `context_recall`, `faithfulness`,
`answer_relevancy`) and the HuggingFace/Ragas embedding wrapper it needed.
`requirements.txt` therefore has no `ragas` entry.

## What was added for the demo

Telemetry only — no prompt, threshold or branch condition was changed. The
notebook `print()`s into a cell; a web demo has to return the same facts, so
each node appends to `rounds` and `trace` in the graph state.

- `rounds[]` — one entry per retrieval round: the queries used, the retrieval
  kind, every chunk with its dense rank, BM25 rank, BM25 score, distance and
  hybrid score, plus the critic verdict and the branch that was taken.
- `retry_decision` — one of `generate`, `prepare_retry`, `generate_max_attempts`,
  `generate_no_new_query`, or `skipped` when the router bypassed retrieval.
- `router_output` — the raw router reply, so a surprising RAG choice is
  debuggable instead of mysterious.
- `new_chunks_vs_previous` — how much a retry actually changed.
- `citation_source` — `model` when the model produced a citation that matched a
  retrieved chunk, `fallback` when the server attached the top chunk itself.
- `quality_flags[]` — see below.
- `/api/health`, and `/api/dev_run`, which reproduces notebook §31, §32, §35 and
  §36 (routing split, accepted-on-first-round, needed-a-retry, still
  insufficient, verified citations) over the notebook's own test questions.

## What the demo surfaces about the current model

These are measured, not theoretical. They are reported rather than hidden, and
each is guarded by a test.

**The router is biased toward answering directly.** On a labelled set of 8
questions needing 3GPP evidence and 8 not needing it, the fine-tuned 1.5B router
scores 75% accuracy at **100% precision and 50% recall**. It never sends a
non-RAG question into retrieval, but it misses about half of the questions that
should be retrieved. Over the 20 notebook test questions it routes 8 to RAG.
Three alternative router prompts (few-shot 2-shot, few-shot 4-shot, a tightened
criterion-only version) were measured and all scored *worse*, so the notebook's
own prompt is kept. The consequence for a demo: some 3GPP questions get answered
from model knowledge with no retrieval, and that is visible under **Router said**.

**Generation ignores two of its own rules.** The prompt tells the model not to
answer in multiple-choice style, but it was fine-tuned on TeleQnA multiple-choice
data and frequently replies `option 1: ...`. Flagged as
`multiple_choice_prefix`.

**A rejected context can still produce a citation.** When the critic rejects the
context but the model answers without the required abstention sentence,
`_ensure_retrieved_citation` attaches the top retrieved chunk as a fallback
citation. That citation is locally derived so it cannot be a hallucination, but
it points at evidence the critic just said was insufficient. Flagged as
`citation_attached_after_rejection`, and the UI says so in plain language. The
underlying behaviour is the notebook's and is left untouched.

## API

| Method | Path | Returns |
| --- | --- | --- |
| `GET` | `/` | The demo page. |
| `GET` | `/api/health` | Readiness, resolved paths, `ragas: "excluded"`. |
| `POST` | `/api/ask` | `{"question": "..."}` → full run record. |
| `POST` | `/api/dev_run` | `{"limit": 6}` or `{"questions":[...]}` → per-question rows + summary. |

## Known deviation

The router prompt is truncated inside `voda.ipynb` itself — the cell keeps only
its tail, `"ecom concepts where exact standards evidence is unnecessary"`. The
demo restores the full sentence, documented at `ROUTER_PROMPT` in
`agent_server.py`. Everything else is verbatim.
