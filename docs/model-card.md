# Model card — `teleqna-qwen-1.5b-qlora`

## Summary

A QLoRA adapter for [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
fine-tuned on the **TeleQnA** telecom question-answering dataset. It improves
multiple-choice telecom accuracy from **46.0% → 70.4%** on a 1,000-question
held-out set, on a single 4 GB GPU.

| | |
| --- | --- |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Library | PEFT 0.21.0 / TRL 1.13.0 |
| Task | Causal LM, conversational SFT |
| Data | 10,000 TeleQnA questions |
| Language | English |

## Intended use

- Telecom multiple-choice question answering.
- The **router** and **answer generator** in the 3GPP RAG agent of this repo.
- Research and demonstration on low-VRAM hardware.

## Out of scope

- Open-ended, safety-critical engineering advice.
- Any claim of factual grounding without retrieval.
- Non-English telecom domains.

## Training procedure

### Data

TeleQnA items were converted to chat format:

```
user:      Question:
           <question>

           Options:
           1.  <option 1>
           2.  <option 2>
           ...

assistant: <reference answer>
```

Split: 80% train, 10% validation, 10% test (seed 42).

### Quantization & LoRA

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

### Training arguments

| Argument | Value |
| --- | --- |
| epochs | 1 |
| learning rate | 2e-4 |
| train batch size | 8 |
| grad accumulation | 2 |
| max length | 512 |
| precision | bf16 |
| eval / save | every 100 steps |
| assistant-only loss | `True` |

The chat template is patched with `{% generation %}` markers so non-assistant
tokens are masked during loss computation.

## Evaluation

| Split | Base | Fine-tuned |
| --- | ---: | ---: |
| 1,000 held-out | 46.0% | **70.4%** |
| 50 held-out | 50.0% | 78.0% |

Exact option-index match, greedy decoding, `max_new_tokens=32`.

## Limitations

- **Format leakage.** The model often prefixes answers with `option 1: ...`,
  a TeleQnA multiple-choice artefact. The demo flags this as
  `multiple_choice_prefix`.
- **Not a retriever.** It has no mechanism to know whether its answer is current
  or complete; the surrounding RAG pipeline supplies evidence.
- **Router bias.** Used as a router, it has 100% precision but only 50% recall on
  a small labelled set — it prefers answering directly.
- **Single dataset.** TeleQnA is multiple-choice; free-form telecom question
  quality is not measured here.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

base = "Qwen/Qwen2.5-1.5B-Instruct"
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    base, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(model, "Tuned_model")

messages = [{"role": "user", "content": "Which network function provides Network Data Analytics services in the 5G System?\n\n1. NSSF\n2. NWDAF\n3. NEF\n4. PCF"}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=32)
print(tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

## Citation

```bibtex
@software{qwen25,
  title  = {Qwen2.5: A Party of Foundation Models},
  author = {Qwen Team},
  year   = {2024},
  url    = {https://qwenlm.github.io/blog/qwen2.5/}
}
```
