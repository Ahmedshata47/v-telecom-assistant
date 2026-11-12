# Architecture

This document describes the full pipeline, from raw 3GPP `.docx` files to a cited answer in the browser.

## 1. Offline corpus build

### 1.1 Parse

`notebooks/voda.ipynb`, cell 0 walks every specification folder and uses
`python-docx` to read headings (`Heading 1..n`) and body text.

- Heading text such as `4.2.3 Service Request procedures` is split into a
  **section number** and a **title**.
- A heading stack reconstructs `parent_section` / `parent_title`, preserving
  document hierarchy.
- Output: `parsed_sections.json` — **6,302 sections** across 7 specifications.

### 1.2 Chunk

Cell 1 splits each section independently with `RecursiveCharacterTextSplitter`:

| Parameter | Value |
| --- | --- |
| `chunk_size` | 1000 |
| `chunk_overlap` | 150 |
| separators | `\n\n`, `\n`, `. `, ` `, `` |

Each chunk keeps the full provenance metadata (`spec`, `source`, `section`,
`title`, `heading_level`, `parent_section`, `parent_title`).

Output: `parsed_chunks.json` — **15,773 chunks**.

### 1.3 Embed

Cell 2 loads `sentence-transformers/multi-qa-mpnet-base-dot-v1` (768-d) and
writes normalized embeddings into the persistent ChromaDB collection
`3gpp_specs`.

> The embedding model is part of the index contract. Querying a 768-d
> collection with a 384-d model (e.g. MiniLM) raises a dimension mismatch and
> the retrieval branch cannot run.

## 2. Fine-tuning

`Fine-tuning/Untitled.ipynb` trains a QLoRA adapter on TeleQnA:

1. TeleQnA JSON → chat messages (`user`: question + options, `assistant`: answer).
2. 80 / 10 / 10 train / validation / test split (seed 42).
3. The Qwen chat template is patched with `{% generation %}` so TRL can mask
   non-assistant tokens (`assistant_only_loss=True`).
4. 4-bit NF4 loading with double quantization and bf16 compute.
5. LoRA `r=16`, `alpha=32`, `dropout=0.05` on `q/k/v/o_proj`.
6. One epoch, `lr=2e-4`, batch 8, gradient accumulation 2, `max_length=512`.

The adapter is pinned to the GPU with `device_map={"": 0}` because automatic
offloading is rejected by the 4-bit quantizer.

## 3. Agent graph

`demo/agent_server.py` is a faithful port of `notebooks/voda.ipynb` (RAGAS
excluded). The only additions are telemetry fields — no prompt, threshold or
branch condition was changed.

```
START
  └─ route ──(no RAG)──────────────────────────────► generate ─► END
        │
        └─(RAG)─ retrieve ─► evaluate_retrieval
                                   │
                     accepted ─────┴────► generate ─► END
                                   │
                     rejected ─────┴────► prepare_retry ─► retrieve_retry
                                                              │
                                                   evaluate_retrieval (loop)
```

### 3.1 Router

The fine-tuned Qwen is asked to answer only `YES` or `NO` to whether retrieval
is required. The raw reply is preserved in `router_output`.

### 3.2 Hybrid retrieval

For each query:

1. **Dense** — embed and query Chroma for `CANDIDATE_K=40` candidates.
2. **BM25** — fuse the same query against an in-memory BM25 index for
   `BM25_CANDIDATE_K=40` candidates.
3. **RRF** — `score = Σ 1 / (RRF_K + rank)` over dense and BM25 ranks, with
   `RRF_K=60`.
4. **Top-K** — keep `TOP_K=5`.

If the question names a spec (e.g. `TS 23.502`), both arms are restricted to that
specification.

### 3.3 Retrieval critic

Mistral `codestral-2508` receives the question and the formatted context and
returns structured JSON:

```json
{
  "relevant": true,
  "score": 0.86,
  "missing_information": "...",
  "search_queries": ["...", "..."]
}
```

The decision node maps the verdict to one of `generate`,
`prepare_retry`, `generate_max_attempts` or `generate_no_new_query`.

### 3.4 Corrective retrieval

When the critic rejects the context and attempts remain:

1. `prepare_retry` records the critic queries.
2. `retrieve_retry` retrieves for the original question **plus** every critic
   query.
3. A **second, query-level RRF** merges the per-query result lists.
4. The merged context is sent back to the critic.

The loop is hard-capped at `MAX_RETRIEVAL_ATTEMPTS=2`.

### 3.5 Generation and citations

The prompt instructs the model to answer only from context and to cite as
`[TS xx.xxx, Section y]`.

The server never trusts a citation blindly:

- `validated_citation` accepts a regex match only if it exists in the retrieved
  metadata.
- `ensure_retrieved_citation` attaches the top retrieved chunk as a **fallback**
  when the model refuses to abstain. The citation is locally derived, so it
  cannot be a hallucination, but the UI flags that it followed a rejected
  context (`citation_attached_after_rejection`).

## 4. Telemetry

Each run returns:

| Field | Purpose |
| --- | --- |
| `rounds[]` | Per-round queries, kind, every chunk with dense/BM25 ranks, distance, fused score, critic verdict and branch |
| `retry_decision` | `generate`, `prepare_retry`, `generate_max_attempts`, `generate_no_new_query`, `skipped` |
| `router_output` | Raw router reply |
| `new_chunks_vs_previous` | How much a retry actually changed |
| `citation_source` | `model`, `fallback` or `none` |
| `quality_flags[]` | Measurable caveats about the answer |

## 5. HTTP API

| Method | Path | Returns |
| --- | --- | --- |
| `GET` | `/` | The single-file demo page |
| `GET` | `/api/health` | Readiness, resolved paths, `ragas: "excluded"` |
| `POST` | `/api/ask` | `{"question": "..."}` → full run record |
| `POST` | `/api/ask/stream` | Server-sent events for the live UI |
| `POST` | `/api/dev_run` | Batch run over the notebook test questions + summary |

## 6. Testing

`demo/test_demo.py` runs 16 checks:

- **8 offline** — RAGAS absence, graph topology, retrieval constants, citation
  validation, critic branch conditions, second RRF pass, run-summary counters,
  and a real retrieval against the local ChromaDB.
- **8 end-to-end** — RAG route, TS filter, direct route, critic rejection and
  retry, quality flags, router discrimination, HTTP payload, agent load. These
  require `MISTRAL_API_KEY`.
