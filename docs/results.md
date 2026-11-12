# Results & experiments

All numbers below are measured on this repository's artefacts. Raw per-question
outputs live in the two CSV files at the repository root.

## Experiment A — effect of QLoRA fine-tuning (no RAG)

**Question:** does fine-tuning Qwen2.5-1.5B-Instruct on TeleQnA actually improve
telecom multiple-choice accuracy?

**Protocol**

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapter: `Tuned_model/` (QLoRA, trained on 8,000 TeleQnA examples)
- Test data: held-out TeleQnA questions, seed 42
- **No RAG, no ChromaDB, no Mistral, no router** — this isolates the fine-tune
- Metric: exact match on the chosen option index
- `MAX_NEW_TOKENS=32`

### 1,000 held-out questions

| Model | Correct | Accuracy | Median latency |
| --- | ---: | ---: | ---: |
| Base Qwen2.5-1.5B-Instruct | 460 / 1000 | **46.0%** | 2.32 s |
| + QLoRA (TeleQnA) | 704 / 1000 | **70.4%** | 1.99 s |

- **+24.4 percentage points**
- **295** questions fixed by fine-tuning
- **51** questions broken by fine-tuning

### 50 held-out questions

| Model | Correct | Accuracy | Median latency |
| --- | ---: | ---: | ---: |
| Base | 25 / 50 | 50.0% | 1.54 s |
| + QLoRA | 39 / 50 | 78.0% | 2.41 s |

- **+28.0 percentage points**
- 17 fixed, 3 broken

> The adapter is also faster on the larger set because it emits the option
> label directly instead of rambling, shortening generation.

## Experiment B — retrieval critic behaviour

The Mistral critic was evaluated as a gatekeeper, not just a grader. On the
notebook's own test questions it corrects retrieval when the first round misses,
and the loop is capped at two rounds.

## Experiment C — router discrimination

On a hand-labelled set of **8 questions that need 3GPP evidence** and **8 that
do not**:

| Metric | Value |
| --- | ---: |
| Accuracy | 75% |
| Precision | 100% |
| Recall | 50% |

Interpretation: the router never sends a non-RAG question into retrieval, but it
misses about half of the questions that should retrieve. Three alternative
prompts (2-shot, 4-shot, tightened criterion) were measured and all scored worse;
the notebook prompt is kept.

Over the 20 notebook test questions, the router sends **8** to RAG.

## Demo test suite

`demo/test_demo.py` — last full run: **16 passed, 0 failed**.

- 8 offline checks always run.
- 8 end-to-end checks run when `MISTRAL_API_KEY` is set.

## Reproducing

```powershell
# 1,000-question fine-tune comparison
#   notebooks/voda.ipynb, the EXPERIMENT A cell (NUM_TEST_QUESTIONS = 1000)

# Demo checks
cd demo
python test_demo.py
```

## How to read these numbers honestly

- TeleQnA is multiple-choice; a 70.4% exact-match score is not a claim of 70.4%
  correctness on open-ended engineering questions.
- The fine-tune teaches **answer format and domain recall**, not grounding.
  Grounding is the retrieval pipeline's job.
- The 1,000-question set is a single seed; expect small run-to-run variance.
