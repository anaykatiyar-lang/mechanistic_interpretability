# GPT-2 Small: From Apparent Addition Circuit to Equal-Operand Effect

## Notebook review

This Markdown export preserves the notebook cell order, source code, explanatory text, and text outputs recorded in the uploaded notebook. It is a static export: the notebook was not rerun during conversion. Recorded values below are transcribed from saved cell outputs and have not been recomputed.

### Analysis summary

- The notebook’s current question is whether GPT-2 Small shows an equal-operand preference in the tested arithmetic prompts and whether a small set of candidate components contributes to the measured target-versus-neighbor logit contrast. The metric is not arithmetic accuracy.
- The canonical circuit dataset has 39 rows over discovery target values 4, 6, 10, 12, and 16, plus exploratory-transfer targets 8 and 14. Prompts include an explicit BOS token; the notebook checks single-token answer IDs.
- The final audit names `10_mlp_out`, `L9H9`, and `L10H2` as exploratory candidates. In the recorded outputs, their DLA FDR q-values are about 0.513; their two-sided causal-ablation FDR q-values are 0.150. These results do not establish a confirmed circuit.
- Recorded activation-patching recovery is modest and variable. Transfer is mixed: the circuit ablation reduces the measured advantage at target 8 but increases it at target 14. Filler prompts also show effects, limiting claims of task specificity.
- **Metric consistency to check before reporting:** the final audit prints a “Removal fraction” of `+0.893908`, while its displayed full-model and circuit-ablated means are `+0.289386` and `+0.097351`. The ratio of those two aggregate means’ difference to the full mean is about `0.664`; therefore the `0.893908` value appears to use a different aggregation. Preserve the stored value, but define the calculation and label before using it in a paper.
- Cell 164’s heading says “exploratory transfer,” but legacy names and a save call still refer to “holdout”; it prints a transfer-check filename while saving `equal_operand_circuit_holdout.csv`. This is a naming inconsistency, not a reason to alter results in this export.
- The old corpus analysis is explicitly marked historical and exploratory, with its threshold not independently justified. The newer raw logging cell collects records but does not set an eligibility floor or establish a corpus correlation.

## Notebook contents

### Cell 1 — markdown

# GPT-2 Small: From Apparent Addition Circuit to Equal-Operand Effect

This notebook documents an iterative mechanistic investigation of arithmetic-like
behavior in GPT-2 Small.

The project began by testing whether simple addition prompts were supported by a
general internal addition circuit. Subsequent causal, attributional, population,
tokenization, and baseline analyses weakened several initial circuit hypotheses.

The surviving phenomenon is a reproducible equal-operand/doubles advantage. The
later experiments therefore investigate whether this phenomenon is specific to
addition or reflects a more general computation associated with repeated operands.


### Cell 2 — code (execution count: 1)

```python
! pip install "transformer_lens==3.6.0"
```

#### Recorded output

```text
Requirement already satisfied: transformer_lens==3.6.0 in /usr/local/lib/python3.13/dist-packages (3.6.0)
Requirement already satisfied: accelerate>=0.23.0 in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (1.15.0)
Requirement already satisfied: beartype>=0.14.1 in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (0.22.9)
Requirement already satisfied: better-abc>=0.0.3 in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (0.0.3)
Requirement already satisfied: datasets>=2.7.1 in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (4.8.5)
Requirement already satisfied: einops>=0.6.0 in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (0.8.2)
Requirement already satisfied: fancy-einsum>=0.0.3 in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (0.0.3)
Requirement already satisfied: huggingface-hub>=0.23.2 in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (1.33.0)
Requirement already satisfied: jaxtyping>=0.2.11 in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (0.3.11)
Requirement already satisfied: packaging>=23.0 in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (26.3)
Requirement already satisfied: pandas>=1.1.5 in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (2.2.3)
Requirement already satisfied: protobuf>=3.20.0 in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (6.33.6)
Requirement already satisfied: rich>=12.6.0 in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (13.9.4)
Requirement already satisfied: sentencepiece in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (0.2.2)
Requirement already satisfied: torch>=2.6 in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (2.11.0+cpu)
Requirement already satisfied: tqdm>=4.64.1 in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (4.67.3)
Requirement already satisfied: transformers-stream-generator<0.1,>=0.0.5 in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (0.0.5)
Requirement already satisfied: transformers>=5.9.0 in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (5.18.0)
Requirement already satisfied: typeguard<5,>=4.2 in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (4.6.0)
Requirement already satisfied: typing-extensions in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (4.16.0)
Requirement already satisfied: wandb>=0.13.5 in /usr/local/lib/python3.13/dist-packages (from transformer_lens==3.6.0) (0.28.1)
Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.13/dist-packages (from accelerate>=0.23.0->transformer_lens==3.6.0) (2.1.3)
Requirement already satisfied: psutil in /usr/local/lib/python3.13/dist-packages (from accelerate>=0.23.0->transformer_lens==3.6.0) (5.9.5)
Requirement already satisfied: pyyaml in /usr/local/lib/python3.13/dist-packages (from accelerate>=0.23.0->transformer_lens==3.6.0) (6.0.3)
Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.13/dist-packages (from accelerate>=0.23.0->transformer_lens==3.6.0) (0.8.0)
Requirement already satisfied: filelock in /usr/local/lib/python3.13/dist-packages (from datasets>=2.7.1->transformer_lens==3.6.0) (4.0.7)
Requirement already satisfied: pyarrow>=21.0.0 in /usr/local/lib/python3.13/dist-packages (from datasets>=2.7.1->transformer_lens==3.6.0) (23.0.1)
Requirement already satisfied: dill<0.4.2,>=0.3.0 in /usr/local/lib/python3.13/dist-packages (from datasets>=2.7.1->transformer_lens==3.6.0) (0.4.1)
Requirement already satisfied: requests>=2.32.2 in /usr/local/lib/python3.13/dist-packages (from datasets>=2.7.1->transformer_lens==3.6.0) (2.32.4)
Requirement already satisfied: httpx<1.0.0 in /usr/local/lib/python3.13/dist-packages (from datasets>=2.7.1->transformer_lens==3.6.0) (0.28.1)
Requirement already satisfied: xxhash in /usr/local/lib/python3.13/dist-packages (from datasets>=2.7.1->transformer_lens==3.6.0) (4.0.1)
Requirement already satisfied: multiprocess<0.70.20 in /usr/local/lib/python3.13/dist-packages (from datasets>=2.7.1->transformer_lens==3.6.0) (0.70.19)
Requirement already satisfied: fsspec<=2026.2.0,>=2023.1.0 in /usr/local/lib/python3.13/dist-packages (from fsspec[http]<=2026.2.0,>=2023.1.0->datasets>=2.7.1->transformer_lens==3.6.0) (2025.12.0)
Requirement already satisfied: click<9.0.0,>=8.4.2 in /usr/local/lib/python3.13/dist-packages (from huggingface-hub>=0.23.2->transformer_lens==3.6.0) (8.5.0)
Requirement already satisfied: hf-xet<2.0.0,>=1.6.0 in /usr/local/lib/python3.13/dist-packages (from huggingface-hub>=0.23.2->transformer_lens==3.6.0) (1.6.0)
Requirement already satisfied: wadler-lindig>=0.1.3 in /usr/local/lib/python3.13/dist-packages (from jaxtyping>=0.2.11->transformer_lens==3.6.0) (0.1.7)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.13/dist-packages (from pandas>=1.1.5->transformer_lens==3.6.0) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.13/dist-packages (from pandas>=1.1.5->transformer_lens==3.6.0) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.13/dist-packages (from pandas>=1.1.5->transformer_lens==3.6.0) (2026.4)
Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.13/dist-packages (from rich>=12.6.0->transformer_lens==3.6.0) (4.2.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.13/dist-packages (from rich>=12.6.0->transformer_lens==3.6.0) (2.21.0)
Requirement already satisfied: setuptools<82 in /usr/local/lib/python3.13/dist-packages (from torch>=2.6->transformer_lens==3.6.0) (80.10.2)
Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.13/dist-packages (from torch>=2.6->transformer_lens==3.6.0) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.13/dist-packages (from torch>=2.6->transformer_lens==3.6.0) (3.7)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.13/dist-packages (from torch>=2.6->transformer_lens==3.6.0) (3.1.6)
Requirement already satisfied: regex>=2025.10.22 in /usr/local/lib/python3.13/dist-packages (from transformers>=5.9.0->transformer_lens==3.6.0) (2025.11.3)
Requirement already satisfied: tokenizers<0.24.0,>=0.23.1 in /usr/local/lib/python3.13/dist-packages (from transformers>=5.9.0->transformer_lens==3.6.0) (0.23.2)
Requirement already satisfied: typer in /usr/local/lib/python3.13/dist-packages (from transformers>=5.9.0->transformer_lens==3.6.0) (0.27.2)
Requirement already satisfied: platformdirs in /usr/local/lib/python3.13/dist-packages (from wandb>=0.13.5->transformer_lens==3.6.0) (4.12.2)
Requirement already satisfied: pydantic<3,>=2.6 in /usr/local/lib/python3.13/dist-packages (from wandb>=0.13.5->transformer_lens==3.6.0) (2.13.5)
Requirement already satisfied: sentry-sdk>=2.0.0 in /usr/local/lib/python3.13/dist-packages (from wandb>=0.13.5->transformer_lens==3.6.0) (2.71.0)
Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /usr/local/lib/python3.13/dist-packages (from fsspec[http]<=2026.2.0,>=2023.1.0->datasets>=2.7.1->transformer_lens==3.6.0) (3.14.3)
Requirement already satisfied: anyio in /usr/local/lib/python3.13/dist-packages (from httpx<1.0.0->datasets>=2.7.1->transformer_lens==3.6.0) (4.15.1)
Requirement already satisfied: certifi in /usr/local/lib/python3.13/dist-packages (from httpx<1.0.0->datasets>=2.7.1->transformer_lens==3.6.0) (2026.7.22)
Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.13/dist-packages (from httpx<1.0.0->datasets>=2.7.1->transformer_lens==3.6.0) (1.0.9)
Requirement already satisfied: idna in /usr/local/lib/python3.13/dist-packages (from httpx<1.0.0->datasets>=2.7.1->transformer_lens==3.6.0) (3.20)
Requirement already satisfied: h11>=0.16 in /usr/local/lib/python3.13/dist-packages (from httpcore==1.*->httpx<1.0.0->datasets>=2.7.1->transformer_lens==3.6.0) (0.16.0)
Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.13/dist-packages (from markdown-it-py>=2.2.0->rich>=12.6.0->transformer_lens==3.6.0) (0.1.2)
Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.13/dist-packages (from pydantic<3,>=2.6->wandb>=0.13.5->transformer_lens==3.6.0) (0.8.0)
Requirement already satisfied: pydantic-core==2.46.5 in /usr/local/lib/python3.13/dist-packages (from pydantic<3,>=2.6->wandb>=0.13.5->transformer_lens==3.6.0) (2.46.5)
Requirement already satisfied: typing-inspection>=0.4.2 in /usr/local/lib/python3.13/dist-packages (from pydantic<3,>=2.6->wandb>=0.13.5->transformer_lens==3.6.0) (0.4.4)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.13/dist-packages (from python-dateutil>=2.8.2->pandas>=1.1.5->transformer_lens==3.6.0) (1.17.0)
Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.13/dist-packages (from requests>=2.32.2->datasets>=2.7.1->transformer_lens==3.6.0) (3.4.9)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.13/dist-packages (from requests>=2.32.2->datasets>=2.7.1->transformer_lens==3.6.0) (2.5.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.13/dist-packages (from sympy>=1.13.3->torch>=2.6->transformer_lens==3.6.0) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.13/dist-packages (from jinja2->torch>=2.6->transformer_lens==3.6.0) (3.0.3)
Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.13/dist-packages (from typer->transformers>=5.9.0->transformer_lens==3.6.0) (1.5.4)
Requirement already satisfied: annotated-doc>=0.0.2 in /usr/local/lib/python3.13/dist-packages (from typer->transformers>=5.9.0->transformer_lens==3.6.0) (0.0.5)
Requirement already satisfied: aiohappyeyeballs>=2.5.0 in /usr/local/lib/python3.13/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.2.0,>=2023.1.0->datasets>=2.7.1->transformer_lens==3.6.0) (2.7.1)
Requirement already satisfied: aiosignal>=1.4.0 in /usr/local/lib/python3.13/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.2.0,>=2023.1.0->datasets>=2.7.1->transformer_lens==3.6.0) (1.4.0)
Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.13/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.2.0,>=2023.1.0->datasets>=2.7.1->transformer_lens==3.6.0) (26.1.0)
Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.13/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.2.0,>=2023.1.0->datasets>=2.7.1->transformer_lens==3.6.0) (1.8.0)
Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.13/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.2.0,>=2023.1.0->datasets>=2.7.1->transformer_lens==3.6.0) (6.9.1)
Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.13/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.2.0,>=2023.1.0->datasets>=2.7.1->transformer_lens==3.6.0) (0.5.4)
Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.13/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.2.0,>=2023.1.0->datasets>=2.7.1->transformer_lens==3.6.0) (1.25.1)
```


### Cell 3 — code (execution count: 2)

```python
import torch
from transformer_lens import HookedTransformer

# Correctly target 'cuda' if a GPU is available to match your Colab runtime
device = "cpu" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
print(f"Model loaded on {device.upper()} with {model.cfg.n_layers} layers and {model.cfg.n_heads} heads per layer.")
```

#### Recorded output

```text
/tmp/ipykernel_108922/3723679228.py:6: DeprecationWarning: HookedTransformer.from_pretrained is deprecated and will be removed in a future major release. Use TransformerBridge.boot_transformers(...) instead, then call enable_compatibility_mode() for HookedTransformer-equivalent numerics. See docs/source/content/migrating_to_v3.md.
  model = HookedTransformer.from_pretrained("gpt2-small", device=device)
```

```text
Loading weights:   0%|          | 0/148 [00:00<?, ?it/s]
```

```text
Loaded pretrained model gpt2-small into HookedTransformer
Model loaded on CPU with 12 layers and 12 heads per layer.
```


### Cell 4 — code (execution count: 3)

```python
import random
import numpy as np
import torch
import transformers
import transformer_lens
import importlib.metadata

# Set fixed random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Safely fetch TransformerLens version
try:
    tl_version = importlib.metadata.version("transformer_lens")
except importlib.metadata.PackageNotFoundError:
    tl_version = "unknown"

# Log library versions for reproducibility
print(f"Environment Setup:")
print(f" - Python: 3.10+")
print(f" - PyTorch: {torch.__version__}")
print(f" - Transformers: {transformers.__version__}")
print(f" - TransformerLens: {tl_version}")
print(f" - NumPy: {np.__version__}")
```

#### Recorded output

```text
Environment Setup:
 - Python: 3.10+
 - PyTorch: 2.11.0+cpu
 - Transformers: 5.18.0
 - TransformerLens: 3.6.0
 - NumPy: 2.1.3
```


### Cell 5 — markdown

Installs above. Now import libraries and load the model


### Cell 6 — markdown

Initial dataset definition (zero-shot baseline, no few-shot prefix)


### Cell 7 — code (execution count: 4)

```python
dataset = [
    {"prompt": "3 + 5 =", "target": " 8", "corrupt": " 9", "difficulty": "Easy (1-step)"},
    {"prompt": "14 + 27 =", "target": " 41", "corrupt": " 31", "difficulty": "Medium (2-step w/ carry)"},
    {"prompt": "158 + 274 =", "target": " 432", "corrupt": " 422", "difficulty": "Hard (3-step w/ multi-carry)"}
]
```


### Cell 8 — code (execution count: 5)

```python
def evaluate_prompts(model, data):
    results = []
    for item in data:
        logits, cache = model.run_with_cache(item["prompt"])
        last_logit = logits[0, -1, :]
        pred_token_id = torch.argmax(last_logit).item()
        pred_str = model.to_string(pred_token_id)
        target_id = model.to_single_token(item["target"])
        corrupt_id = model.to_single_token(item["corrupt"])
        logit_diff = (last_logit[target_id] - last_logit[corrupt_id]).item()
        is_correct = (pred_str.strip() == item["target"].strip())
        results.append({
            "prompt": item["prompt"],
            "difficulty": item["difficulty"],
            "logit_diff": logit_diff,
            "correct": is_correct
        })
    return results

results = evaluate_prompts(model, dataset)
for r in results:
    print(r)
```

#### Recorded output

```text
{'prompt': '3 + 5 =', 'difficulty': 'Easy (1-step)', 'logit_diff': 0.6443347930908203, 'correct': False}
{'prompt': '14 + 27 =', 'difficulty': 'Medium (2-step w/ carry)', 'logit_diff': -0.6933021545410156, 'correct': False}
{'prompt': '158 + 274 =', 'difficulty': 'Hard (3-step w/ multi-carry)', 'logit_diff': 0.09389686584472656, 'correct': False}
```


### Cell 9 — code (execution count: 6)

```python

few_shot_prefix = "1 + 1 = 2\n2 + 2 = 4\n5 + 3 = 8\n"

dataset = [
    {"prompt": few_shot_prefix + "3 + 5 =", "target": " 8", "corrupt": " 9", "difficulty": "Easy (1-step)"},
    {"prompt": few_shot_prefix + "14 + 27 =", "target": " 41", "corrupt": " 31", "difficulty": "Medium (2-step w/ carry)"},
    {"prompt": few_shot_prefix + "158 + 274 =", "target": " 432", "corrupt": " 422", "difficulty": "Hard (3-step w/ multi-carry)"}
]

results = evaluate_prompts(model, dataset)
for r in results:
    print(r)
```

#### Recorded output

```text
{'prompt': '1 + 1 = 2\n2 + 2 = 4\n5 + 3 = 8\n3 + 5 =', 'difficulty': 'Easy (1-step)', 'logit_diff': -1.2596702575683594, 'correct': False}
{'prompt': '1 + 1 = 2\n2 + 2 = 4\n5 + 3 = 8\n14 + 27 =', 'difficulty': 'Medium (2-step w/ carry)', 'logit_diff': -0.5094718933105469, 'correct': False}
{'prompt': '1 + 1 = 2\n2 + 2 = 4\n5 + 3 = 8\n158 + 274 =', 'difficulty': 'Hard (3-step w/ multi-carry)', 'logit_diff': 0.6092090606689453, 'correct': False}
```


### Cell 10 — markdown

### Few-shot prediction: `158 + 274 =`

The model still predicts `' 1'` even with few-shot examples.
We record this output here, but we will run a clean prompt for
the actual DLA analysis below.

> **Warning:** Do not use the `logits` or `cache` variables from this
> loop as input to later analyses.


### Cell 11 — code (execution count: 7)

```python
for item in dataset:
    logits, cache = model.run_with_cache(item["prompt"])
    pred_id = torch.argmax(logits[0, -1, :]).item()
    print(f"Prompt: {item['prompt']!r}")
    print(f"  Predicted token: {model.to_string(pred_id)!r}")
```

#### Recorded output

```text
Prompt: '1 + 1 = 2\n2 + 2 = 4\n5 + 3 = 8\n3 + 5 ='
  Predicted token: ' 12'
Prompt: '1 + 1 = 2\n2 + 2 = 4\n5 + 3 = 8\n14 + 27 ='
  Predicted token: ' 32'
Prompt: '1 + 1 = 2\n2 + 2 = 4\n5 + 3 = 8\n158 + 274 ='
  Predicted token: ' 1'
```


### Cell 12 — markdown

After few shot prefix run it still predicts wrong answer .


### Cell 13 — code (execution count: 8)

```python
top_tokens = torch.topk(logits[0, -1], k=5).indices
print([model.to_string(t) for t in top_tokens])
```

#### Recorded output

```text
[' 1', ' 4', ' 5', ' 8', ' 6']
```


### Cell 14 — markdown

Fetching the top 5 predictions of the model


### Cell 15 — code (execution count: 9)

```python
print("Type of dataset item:", type(dataset[0]))
print("Content:", dataset[0])
```

#### Recorded output

```text
Type of dataset item: <class 'dict'>
Content: {'prompt': '1 + 1 = 2\n2 + 2 = 4\n5 + 3 = 8\n3 + 5 =', 'target': ' 8', 'corrupt': ' 9', 'difficulty': 'Easy (1-step)'}
```


### Cell 16 — code (execution count: 10)

```python
for item in dataset:
    target_str = item.get("target", "")
    corrupt_str = item.get("corrupt", "")

    # Prepend leading space required by model tokenization if prompt ends in '='
    target_id = model.to_single_token(" " + target_str.strip())
    corrupt_id = model.to_single_token(" " + corrupt_str.strip())
```


### Cell 17 — markdown

 Added a space before in both corrupt and right answer bcs gpt 2 small reads both things as different.


### Cell 18 — code (execution count: 11)

```python
last_logits = logits[0, -1]
probs = torch.softmax(last_logits, dim=-1)
top_k = torch.topk(last_logits, k=5)

for token_id, logit in zip(top_k.indices, top_k.values):
    token_str = model.to_string(token_id)
    prob = probs[token_id].item()
    print(f"ID: {token_id.item():<6} | Token: {repr(token_str):<8} | Prob: {prob*100:.2f}% | Logit: {logit.item():.2f}")
```

#### Recorded output

```text
ID: 352    | Token: ' 1'     | Prob: 5.59% | Logit: 15.65
ID: 604    | Token: ' 4'     | Prob: 4.65% | Logit: 15.47
ID: 642    | Token: ' 5'     | Prob: 4.65% | Logit: 15.47
ID: 807    | Token: ' 8'     | Prob: 4.61% | Logit: 15.46
ID: 718    | Token: ' 6'     | Prob: 4.30% | Logit: 15.39
```


### Cell 19 — markdown

Again calculating the top 5 tokens and with their probability and there is no single strong favorite of model therefore, we will run a check to see wether this is a correct DLA run by using ATTRIBUTION SUM


### Cell 20 — code (execution count: 12)

```python
import torch

target_id = 807  # Token ID for ' 8'
top_id = 352     # Token ID for ' 1'

# 1. Compute unembedding direction for Target minus Top prediction
unembed_diff = model.W_U[:, target_id] - model.W_U[:, top_id]

# 2. Extract normalized residual stream accumulated across layers
# 'cache' comes from model.run_with_cache(prompt)
accumulated_resid, labels = cache.get_full_resid_decomposition(
    layer=-1, expand_neurons=False, return_labels=True
)

# 3. Project each layer/head onto the unembedding direction
# Shape: [components, seq_len, d_model] -> project last sequence position
last_pos_resid = accumulated_resid[:, 0, -1, :]
normalized_resid= cache.apply_ln_to_stack(last_pos_resid, layer=-1, pos_slice=-1)
#

attributions = torch.einsum("c d, d -> c", normalized_resid, unembed_diff)

for label, attr in zip(labels, attributions):
    print(f"{label:<25} | Attribution: {attr.item():+.4f}")
```

#### Recorded output

```text
L0H0                      | Attribution: +0.0178
L0H1                      | Attribution: -0.0335
L0H2                      | Attribution: -0.0071
L0H3                      | Attribution: +0.0045
L0H4                      | Attribution: -0.0081
L0H5                      | Attribution: -0.0340
L0H6                      | Attribution: -0.0591
L0H7                      | Attribution: +0.0150
L0H8                      | Attribution: +0.0267
L0H9                      | Attribution: -0.0086
L0H10                     | Attribution: +0.0533
L0H11                     | Attribution: +0.0280
L1H0                      | Attribution: -0.0145
L1H1                      | Attribution: -0.0021
L1H2                      | Attribution: -0.0207
L1H3                      | Attribution: +0.0068
L1H4                      | Attribution: +0.0108
L1H5                      | Attribution: +0.0669
L1H6                      | Attribution: +0.0202
L1H7                      | Attribution: -0.0408
L1H8                      | Attribution: -0.0114
L1H9                      | Attribution: -0.0060
L1H10                     | Attribution: -0.0130
L1H11                     | Attribution: -0.0005
L2H0                      | Attribution: -0.0329
L2H1                      | Attribution: -0.0148
L2H2                      | Attribution: +0.0379
L2H3                      | Attribution: +0.0021
L2H4                      | Attribution: -0.0050
L2H5                      | Attribution: -0.0310
L2H6                      | Attribution: -0.0261
L2H7                      | Attribution: -0.0061
L2H8                      | Attribution: +0.0094
L2H9                      | Attribution: +0.0310
L2H10                     | Attribution: +0.0812
L2H11                     | Attribution: -0.0302
L3H0                      | Attribution: -0.0268
L3H1                      | Attribution: -0.0149
L3H2                      | Attribution: +0.0313
L3H3                      | Attribution: -0.0117
L3H4                      | Attribution: +0.0277
L3H5                      | Attribution: +0.0035
L3H6                      | Attribution: +0.0079
L3H7                      | Attribution: +0.0242
L3H8                      | Attribution: -0.0086
L3H9                      | Attribution: +0.0027
L3H10                     | Attribution: +0.0039
L3H11                     | Attribution: +0.0348
L4H0                      | Attribution: -0.0001
L4H1                      | Attribution: +0.0150
L4H2                      | Attribution: +0.0146
L4H3                      | Attribution: -0.0092
L4H4                      | Attribution: -0.0014
L4H5                      | Attribution: +0.0118
L4H6                      | Attribution: +0.0077
L4H7                      | Attribution: -0.0608
L4H8                      | Attribution: -0.0559
L4H9                      | Attribution: +0.0315
L4H10                     | Attribution: -0.0196
L4H11                     | Attribution: +0.0038
L5H0                      | Attribution: -0.0088
L5H1                      | Attribution: +0.0137
L5H2                      | Attribution: +0.0159
L5H3                      | Attribution: +0.0111
L5H4                      | Attribution: -0.0024
L5H5                      | Attribution: +0.0188
L5H6                      | Attribution: -0.0111
L5H7                      | Attribution: +0.0015
L5H8                      | Attribution: +0.0503
L5H9                      | Attribution: -0.0310
L5H10                     | Attribution: -0.0120
L5H11                     | Attribution: +0.0090
L6H0                      | Attribution: +0.0183
L6H1                      | Attribution: +0.0175
L6H2                      | Attribution: +0.0062
L6H3                      | Attribution: +0.0466
L6H4                      | Attribution: -0.0095
L6H5                      | Attribution: -0.0176
L6H6                      | Attribution: -0.0181
L6H7                      | Attribution: +0.0139
L6H8                      | Attribution: +0.0168
L6H9                      | Attribution: -0.0134
L6H10                     | Attribution: +0.0035
L6H11                     | Attribution: -0.0054
L7H0                      | Attribution: +0.0219
L7H1                      | Attribution: +0.0040
L7H2                      | Attribution: +0.0145
L7H3                      | Attribution: -0.0561
L7H4                      | Attribution: -0.0063
L7H5                      | Attribution: -0.0010
L7H6                      | Attribution: -0.0122
L7H7                      | Attribution: +0.0042
L7H8                      | Attribution: +0.0133
L7H9                      | Attribution: -0.0083
L7H10                     | Attribution: +0.0358
L7H11                     | Attribution: +0.1214
L8H0                      | Attribution: +0.0100
L8H1                      | Attribution: +0.0938
L8H2                      | Attribution: -0.0497
L8H3                      | Attribution: +0.0457
L8H4                      | Attribution: -0.0036
L8H5                      | Attribution: +0.0188
L8H6                      | Attribution: +0.0090
L8H7                      | Attribution: +0.0125
L8H8                      | Attribution: -0.0658
L8H9                      | Attribution: -0.0054
L8H10                     | Attribution: -0.0839
L8H11                     | Attribution: -0.0156
L9H0                      | Attribution: -0.0050
L9H1                      | Attribution: +0.1748
L9H2                      | Attribution: -0.0054
L9H3                      | Attribution: -0.0078
L9H4                      | Attribution: -0.0128
L9H5                      | Attribution: +0.0225
L9H6                      | Attribution: -0.0287
L9H7                      | Attribution: +0.0069
L9H8                      | Attribution: -0.0058
L9H9                      | Attribution: +0.0152
L9H10                     | Attribution: -0.0005
L9H11                     | Attribution: -0.0017
L10H0                     | Attribution: +0.0090
L10H1                     | Attribution: -0.0174
L10H2                     | Attribution: -0.0136
L10H3                     | Attribution: -0.0010
L10H4                     | Attribution: +0.0139
L10H5                     | Attribution: +0.0044
L10H6                     | Attribution: +0.0243
L10H7                     | Attribution: -0.1120
L10H8                     | Attribution: -0.0053
L10H9                     | Attribution: -0.0012
L10H10                    | Attribution: +0.0021
L10H11                    | Attribution: +0.0032
L11H0                     | Attribution: -0.0654
L11H1                     | Attribution: +0.0074
L11H2                     | Attribution: -0.0174
L11H3                     | Attribution: -0.0055
L11H4                     | Attribution: +0.0070
L11H5                     | Attribution: +0.0110
L11H6                     | Attribution: -0.0167
L11H7                     | Attribution: -0.0234
L11H8                     | Attribution: +0.0425
L11H9                     | Attribution: +0.0289
L11H10                    | Attribution: -0.0228
L11H11                    | Attribution: +0.0086
0_mlp_out                 | Attribution: -0.1519
1_mlp_out                 | Attribution: +0.0540
2_mlp_out                 | Attribution: +0.0532
3_mlp_out                 | Attribution: -0.0501
4_mlp_out                 | Attribution: -0.0182
5_mlp_out                 | Attribution: +0.0090
6_mlp_out                 | Attribution: -0.0147
7_mlp_out                 | Attribution: -0.0753
8_mlp_out                 | Attribution: +0.1963
9_mlp_out                 | Attribution: +0.0922
10_mlp_out                | Attribution: +0.3441
11_mlp_out                | Attribution: -0.0806
embed                     | Attribution: -0.0147
pos_embed                 | Attribution: +0.0097
bias                      | Attribution: +0.2436
```


### Cell 21 — code (execution count: 13)

```python
import torch

target_id = 807  # Token ID for ' 8'
top_id = 352     # Token ID for ' 1'

# 1. Compute unembedding direction for Target minus Top prediction
unembed_diff = model.W_U[:, target_id] - model.W_U[:, top_id]

# 2. Extract full decomposed residual stream (Shape: [components, batch, pos, d_model])
accumulated_resid, labels = cache.get_full_resid_decomposition(
    layer=-1, expand_neurons=False, return_labels=True
)

# 3. Apply LayerNorm to the FULL stack first
# TransformerLens uses pos_slice=-1 to align the correct stored scale factor.
# This returns a tensor of shape [components, batch, d_model]
normalized_resid_full= cache.apply_ln_to_stack(accumulated_resid, layer=-1, pos_slice=-1)

# 4. Now slice out batch 0 (Shape: [components, d_model])
normalized_resid = normalized_resid_full[:, 0, -1, :]

# 5. Project each component onto the unembedding difference direction
attributions = torch.einsum("c d, d -> c", normalized_resid, unembed_diff)

for label, attr in zip(labels, attributions):
    print(f"{label:<25} | Attribution: {attr.item():+.4f}")
```

#### Recorded output

```text
WARNING:root:Tried to compute head results when they were already cached
```

```text
L0H0                      | Attribution: +0.0178
L0H1                      | Attribution: -0.0335
L0H2                      | Attribution: -0.0071
L0H3                      | Attribution: +0.0045
L0H4                      | Attribution: -0.0081
L0H5                      | Attribution: -0.0340
L0H6                      | Attribution: -0.0591
L0H7                      | Attribution: +0.0150
L0H8                      | Attribution: +0.0267
L0H9                      | Attribution: -0.0086
L0H10                     | Attribution: +0.0533
L0H11                     | Attribution: +0.0280
L1H0                      | Attribution: -0.0145
L1H1                      | Attribution: -0.0021
L1H2                      | Attribution: -0.0207
L1H3                      | Attribution: +0.0068
L1H4                      | Attribution: +0.0108
L1H5                      | Attribution: +0.0669
L1H6                      | Attribution: +0.0202
L1H7                      | Attribution: -0.0408
L1H8                      | Attribution: -0.0114
L1H9                      | Attribution: -0.0060
L1H10                     | Attribution: -0.0130
L1H11                     | Attribution: -0.0005
L2H0                      | Attribution: -0.0329
L2H1                      | Attribution: -0.0148
L2H2                      | Attribution: +0.0379
L2H3                      | Attribution: +0.0021
L2H4                      | Attribution: -0.0050
L2H5                      | Attribution: -0.0310
L2H6                      | Attribution: -0.0261
L2H7                      | Attribution: -0.0061
L2H8                      | Attribution: +0.0094
L2H9                      | Attribution: +0.0310
L2H10                     | Attribution: +0.0812
L2H11                     | Attribution: -0.0302
L3H0                      | Attribution: -0.0268
L3H1                      | Attribution: -0.0149
L3H2                      | Attribution: +0.0313
L3H3                      | Attribution: -0.0117
L3H4                      | Attribution: +0.0277
L3H5                      | Attribution: +0.0035
L3H6                      | Attribution: +0.0079
L3H7                      | Attribution: +0.0242
L3H8                      | Attribution: -0.0086
L3H9                      | Attribution: +0.0027
L3H10                     | Attribution: +0.0039
L3H11                     | Attribution: +0.0348
L4H0                      | Attribution: -0.0001
L4H1                      | Attribution: +0.0150
L4H2                      | Attribution: +0.0146
L4H3                      | Attribution: -0.0092
L4H4                      | Attribution: -0.0014
L4H5                      | Attribution: +0.0118
L4H6                      | Attribution: +0.0077
L4H7                      | Attribution: -0.0608
L4H8                      | Attribution: -0.0559
L4H9                      | Attribution: +0.0315
L4H10                     | Attribution: -0.0196
L4H11                     | Attribution: +0.0038
L5H0                      | Attribution: -0.0088
L5H1                      | Attribution: +0.0137
L5H2                      | Attribution: +0.0159
L5H3                      | Attribution: +0.0111
L5H4                      | Attribution: -0.0024
L5H5                      | Attribution: +0.0188
L5H6                      | Attribution: -0.0111
L5H7                      | Attribution: +0.0015
L5H8                      | Attribution: +0.0503
L5H9                      | Attribution: -0.0310
L5H10                     | Attribution: -0.0120
L5H11                     | Attribution: +0.0090
L6H0                      | Attribution: +0.0183
L6H1                      | Attribution: +0.0175
L6H2                      | Attribution: +0.0062
L6H3                      | Attribution: +0.0466
L6H4                      | Attribution: -0.0095
L6H5                      | Attribution: -0.0176
L6H6                      | Attribution: -0.0181
L6H7                      | Attribution: +0.0139
L6H8                      | Attribution: +0.0168
L6H9                      | Attribution: -0.0134
L6H10                     | Attribution: +0.0035
L6H11                     | Attribution: -0.0054
L7H0                      | Attribution: +0.0219
L7H1                      | Attribution: +0.0040
L7H2                      | Attribution: +0.0145
L7H3                      | Attribution: -0.0561
L7H4                      | Attribution: -0.0063
L7H5                      | Attribution: -0.0010
L7H6                      | Attribution: -0.0122
L7H7                      | Attribution: +0.0042
L7H8                      | Attribution: +0.0133
L7H9                      | Attribution: -0.0083
L7H10                     | Attribution: +0.0358
L7H11                     | Attribution: +0.1214
L8H0                      | Attribution: +0.0100
L8H1                      | Attribution: +0.0938
L8H2                      | Attribution: -0.0497
L8H3                      | Attribution: +0.0457
L8H4                      | Attribution: -0.0036
L8H5                      | Attribution: +0.0188
L8H6                      | Attribution: +0.0090
L8H7                      | Attribution: +0.0125
L8H8                      | Attribution: -0.0658
L8H9                      | Attribution: -0.0054
L8H10                     | Attribution: -0.0839
L8H11                     | Attribution: -0.0156
L9H0                      | Attribution: -0.0050
L9H1                      | Attribution: +0.1748
L9H2                      | Attribution: -0.0054
L9H3                      | Attribution: -0.0078
L9H4                      | Attribution: -0.0128
L9H5                      | Attribution: +0.0225
L9H6                      | Attribution: -0.0287
L9H7                      | Attribution: +0.0069
L9H8                      | Attribution: -0.0058
L9H9                      | Attribution: +0.0152
L9H10                     | Attribution: -0.0005
L9H11                     | Attribution: -0.0017
L10H0                     | Attribution: +0.0090
L10H1                     | Attribution: -0.0174
L10H2                     | Attribution: -0.0136
L10H3                     | Attribution: -0.0010
L10H4                     | Attribution: +0.0139
L10H5                     | Attribution: +0.0044
L10H6                     | Attribution: +0.0243
L10H7                     | Attribution: -0.1120
L10H8                     | Attribution: -0.0053
L10H9                     | Attribution: -0.0012
L10H10                    | Attribution: +0.0021
L10H11                    | Attribution: +0.0032
L11H0                     | Attribution: -0.0654
L11H1                     | Attribution: +0.0074
L11H2                     | Attribution: -0.0174
L11H3                     | Attribution: -0.0055
L11H4                     | Attribution: +0.0070
L11H5                     | Attribution: +0.0110
L11H6                     | Attribution: -0.0167
L11H7                     | Attribution: -0.0234
L11H8                     | Attribution: +0.0425
L11H9                     | Attribution: +0.0289
L11H10                    | Attribution: -0.0228
L11H11                    | Attribution: +0.0086
0_mlp_out                 | Attribution: -0.1519
1_mlp_out                 | Attribution: +0.0540
2_mlp_out                 | Attribution: +0.0532
3_mlp_out                 | Attribution: -0.0501
4_mlp_out                 | Attribution: -0.0182
5_mlp_out                 | Attribution: +0.0090
6_mlp_out                 | Attribution: -0.0147
7_mlp_out                 | Attribution: -0.0753
8_mlp_out                 | Attribution: +0.1963
9_mlp_out                 | Attribution: +0.0922
10_mlp_out                | Attribution: +0.3441
11_mlp_out                | Attribution: -0.0806
embed                     | Attribution: -0.0147
pos_embed                 | Attribution: +0.0097
bias                      | Attribution: +0.2436
```


### Cell 22 — markdown

the sums dont addup actually not anywhere near to the logit diff -0.19 rather it is +0.8382 now we will add bias diff to get the correct dla sum


### Cell 23 — code (execution count: 14)

```python
bias_diff = (model.b_U[target_id] - model.b_U[top_id]).item()
print(f"Bias Difference: {bias_diff:+.4f}")
```

#### Recorded output

```text
Bias Difference: -1.0303
```


### Cell 24 — code (execution count: 15)

```python
target_id = 807  # ' 8'
top_id = 352     # ' 1'

# 1. Get the static bias difference
bias_diff = (model.b_U[target_id] - model.b_U[top_id]).item()

# 2. Get actual logit difference directly from the model's output
# (Assuming 'logits' is your saved model output)
actual_logit_diff = (logits[0, -1, target_id] - logits[0, -1, top_id]).item()
dla_sum = attributions.sum().item()

print(f"Circuit's Math (DLA Sum): {dla_sum:+.4f}")
print(f"Static Bias Difference:   {bias_diff:+.4f}")
print("-" * 30)
print(f"Expected Final Logit:     {dla_sum + bias_diff:+.4f}")
print(f"Actual Logit Diff:        {actual_logit_diff:+.4f}")
```

#### Recorded output

```text
Circuit's Math (DLA Sum): +0.8382
Static Bias Difference:   -1.0303
------------------------------
Expected Final Logit:     -0.1920
Actual Logit Diff:        -0.1920
```


### Cell 25 — markdown

### DLA Result Interpretation

*(Note: This DLA run is based on the `158 + 274 =` prompt from earlier cells, contrasting `' 8'` vs `' 1'`.)*

The DLA sum for `' 8'` versus `' 1'` is positive (circuit favours `' 8'`),
but the final logit for `' 1'` is higher because the static unembedding
bias $\Delta b_U$ is large and negative for this direction.

**This does not mean the model computed the answer 8 and was then
overridden.** It means the model's circuit components collectively
pushed slightly toward `' 8'`, but the unembedding layer's fixed prior
is strong enough to flip the output to `' 1'` without any suppression
of an arithmetic result.

We keep this here to record the `' 8'` vs `' 1'` contrast specifically.
Do not generalise from this prompt to the full DLA analysis — that is
done on `3 + 5 =` below.


### Cell 26 — code (execution count: 16)

```python
import torch
from transformer_lens import HookedTransformer

# 2. Run clean forward pass
prompt = "3 + 5 ="
logits, cache = model.run_with_cache(prompt)

# 3. Inspect top-5 predictions at the final position
last_logits = logits[0, -1]
probs = torch.softmax(last_logits, dim=-1)
top_k = torch.topk(last_logits, k=5)

print(f"Prompt: {repr(prompt)}\n" + "-"*40)
for token_id, logit in zip(top_k.indices, top_k.values):
    token_str = model.to_string(token_id)
    prob = probs[token_id].item()
    print(f"ID: {token_id.item():<6} | Token: {repr(token_str):<8} | Prob: {prob*100:.2f}% | Logit: {logit.item():.2f}"
    )
    # Define target/foil tokens and print them
target_id = model.to_single_token(" 8")
corrupt_id = model.to_single_token(" 9")

print(f"\nTarget token: {repr(model.to_string(target_id))} (ID {target_id})")
print(f"Foil token:   {repr(model.to_string(corrupt_id))} (ID {corrupt_id})")
```

#### Recorded output

```text
Prompt: '3 + 5 ='
----------------------------------------
ID: 352    | Token: ' 1'     | Prob: 6.25% | Logit: 13.39
ID: 604    | Token: ' 4'     | Prob: 6.25% | Logit: 13.39
ID: 362    | Token: ' 2'     | Prob: 6.11% | Logit: 13.36
ID: 718    | Token: ' 6'     | Prob: 5.72% | Logit: 13.30
ID: 642    | Token: ' 5'     | Prob: 5.72% | Logit: 13.30

Target token: ' 8' (ID 807)
Foil token:   ' 9' (ID 860)
```


### Cell 27 — markdown

okay so now we ran top 5 predictions forthe model to see without few shot texting so i can accurately measure so i can actually measure what the intervention broke or fixed inside the black box


### Cell 28 — code (execution count: 17)

```python
target_id = model.to_single_token(" 8")
corrupt_id = model.to_single_token(" 9")

target_logit = last_logits[target_id].item()
corrupt_logit = last_logits[corrupt_id].item()
logit_diff = target_logit - corrupt_logit

# Find exact rank of ' 8' in the full vocabulary
sorted_indices = torch.argsort(last_logits, descending=True)
target_rank = (sorted_indices == target_id).nonzero().item() + 1

print(f"Token ' 8' Rank: {target_rank} / {model.cfg.d_vocab}")
print(f"Target Logit (' 8'): {target_logit:.4f}")
print(f"Corrupt Logit (' 9'): {corrupt_logit:.4f}")
print(f"Logit Diff (' 8' - ' 9'): {logit_diff:+.4f}")
```

#### Recorded output

```text
Token ' 8' Rank: 7 / 50257
Target Logit (' 8'): 13.0319
Corrupt Logit (' 9'): 12.3876
Logit Diff (' 8' - ' 9'): +0.6443
```


### Cell 29 — markdown

Here we sorted the whole vocabulary token wise and then got the rank for 8 ,logit for the corrupt answer 9 and their logit difference


### Cell 30 — code (execution count: 18)

```python
import torch

target_id = model.to_single_token(" 8")
corrupt_id = model.to_single_token(" 9")

# 1. Compute unembedding difference vector (Target minus Corrupt)
unembed_diff = model.W_U[:, target_id] - model.W_U[:, corrupt_id]

# 2. Extract decomposed residual stream from cache
accumulated_resid, labels = cache.get_full_resid_decomposition(
    layer=-1, expand_neurons=False, return_labels=True
)

# 3. Project the last token position ('=') onto the difference direction
last_pos_resid = accumulated_resid[:, 0, -1, :]
# 1. ADD THIS LINE: Apply final LayerNorm scaling via TransformerLens cache
normalized_resid = cache.apply_ln_to_stack(last_pos_resid, layer=-1, pos_slice=-1)

# 2. MODIFY THIS LINE: Swap 'last_pos_resid' for 'normalized_resid'
attributions = torch.einsum("c d, d -> c", normalized_resid, unembed_diff)


# 4. Print top components driving the +0.6443 logit diff
for label, attr in zip(labels, attributions):
    print(f"{label:<25} | Attribution: {attr.item():+.4f}")
```

#### Recorded output

```text
L0H0                      | Attribution: -0.0076
L0H1                      | Attribution: -0.0172
L0H2                      | Attribution: -0.0009
L0H3                      | Attribution: -0.0032
L0H4                      | Attribution: -0.0097
L0H5                      | Attribution: -0.0016
L0H6                      | Attribution: +0.0214
L0H7                      | Attribution: +0.0359
L0H8                      | Attribution: -0.0098
L0H9                      | Attribution: -0.0048
L0H10                     | Attribution: +0.0406
L0H11                     | Attribution: -0.0039
L1H0                      | Attribution: +0.0073
L1H1                      | Attribution: +0.0074
L1H2                      | Attribution: -0.0192
L1H3                      | Attribution: -0.0161
L1H4                      | Attribution: +0.0587
L1H5                      | Attribution: +0.0116
L1H6                      | Attribution: -0.0117
L1H7                      | Attribution: +0.0146
L1H8                      | Attribution: +0.0076
L1H9                      | Attribution: -0.0080
L1H10                     | Attribution: +0.0105
L1H11                     | Attribution: -0.0333
L2H0                      | Attribution: -0.0259
L2H1                      | Attribution: -0.0047
L2H2                      | Attribution: +0.0351
L2H3                      | Attribution: +0.0250
L2H4                      | Attribution: -0.0207
L2H5                      | Attribution: +0.0189
L2H6                      | Attribution: -0.0147
L2H7                      | Attribution: -0.0014
L2H8                      | Attribution: +0.0055
L2H9                      | Attribution: +0.0127
L2H10                     | Attribution: +0.0057
L2H11                     | Attribution: +0.0082
L3H0                      | Attribution: -0.0119
L3H1                      | Attribution: -0.0282
L3H2                      | Attribution: +0.0024
L3H3                      | Attribution: -0.0134
L3H4                      | Attribution: +0.0026
L3H5                      | Attribution: +0.0012
L3H6                      | Attribution: -0.0012
L3H7                      | Attribution: +0.0372
L3H8                      | Attribution: -0.0007
L3H9                      | Attribution: -0.0119
L3H10                     | Attribution: -0.0067
L3H11                     | Attribution: +0.0010
L4H0                      | Attribution: +0.0066
L4H1                      | Attribution: -0.0120
L4H2                      | Attribution: +0.0080
L4H3                      | Attribution: +0.0156
L4H4                      | Attribution: +0.0090
L4H5                      | Attribution: -0.0004
L4H6                      | Attribution: -0.0413
L4H7                      | Attribution: -0.0135
L4H8                      | Attribution: -0.0212
L4H9                      | Attribution: -0.0086
L4H10                     | Attribution: -0.0056
L4H11                     | Attribution: +0.0345
L5H0                      | Attribution: +0.0160
L5H1                      | Attribution: -0.0209
L5H2                      | Attribution: +0.0320
L5H3                      | Attribution: -0.0326
L5H4                      | Attribution: -0.0053
L5H5                      | Attribution: -0.0088
L5H6                      | Attribution: +0.0019
L5H7                      | Attribution: +0.0072
L5H8                      | Attribution: +0.0543
L5H9                      | Attribution: +0.0216
L5H10                     | Attribution: -0.0065
L5H11                     | Attribution: +0.0011
L6H0                      | Attribution: -0.0149
L6H1                      | Attribution: +0.0228
L6H2                      | Attribution: +0.0046
L6H3                      | Attribution: +0.0174
L6H4                      | Attribution: +0.0161
L6H5                      | Attribution: -0.0285
L6H6                      | Attribution: +0.0090
L6H7                      | Attribution: +0.0024
L6H8                      | Attribution: +0.0051
L6H9                      | Attribution: +0.0161
L6H10                     | Attribution: -0.0134
L6H11                     | Attribution: -0.0131
L7H0                      | Attribution: +0.0053
L7H1                      | Attribution: +0.0020
L7H2                      | Attribution: -0.0025
L7H3                      | Attribution: -0.0053
L7H4                      | Attribution: +0.0079
L7H5                      | Attribution: -0.0103
L7H6                      | Attribution: +0.0270
L7H7                      | Attribution: -0.0041
L7H8                      | Attribution: +0.0149
L7H9                      | Attribution: +0.0017
L7H10                     | Attribution: +0.0134
L7H11                     | Attribution: +0.0003
L8H0                      | Attribution: -0.0110
L8H1                      | Attribution: -0.0033
L8H2                      | Attribution: -0.0069
L8H3                      | Attribution: -0.0113
L8H4                      | Attribution: +0.0052
L8H5                      | Attribution: +0.0242
L8H6                      | Attribution: +0.0288
L8H7                      | Attribution: +0.0062
L8H8                      | Attribution: +0.0326
L8H9                      | Attribution: +0.0124
L8H10                     | Attribution: -0.0263
L8H11                     | Attribution: -0.0205
L9H0                      | Attribution: -0.0133
L9H1                      | Attribution: +0.0300
L9H2                      | Attribution: +0.0064
L9H3                      | Attribution: -0.0042
L9H4                      | Attribution: -0.0051
L9H5                      | Attribution: -0.0117
L9H6                      | Attribution: -0.0139
L9H7                      | Attribution: +0.0184
L9H8                      | Attribution: -0.0070
L9H9                      | Attribution: -0.0348
L9H10                     | Attribution: +0.0077
L9H11                     | Attribution: +0.0035
L10H0                     | Attribution: -0.0188
L10H1                     | Attribution: -0.0152
L10H2                     | Attribution: +0.0086
L10H3                     | Attribution: +0.0036
L10H4                     | Attribution: +0.0123
L10H5                     | Attribution: -0.0198
L10H6                     | Attribution: +0.0267
L10H7                     | Attribution: +0.0138
L10H8                     | Attribution: +0.0060
L10H9                     | Attribution: +0.0113
L10H10                    | Attribution: +0.0239
L10H11                    | Attribution: -0.0133
L11H0                     | Attribution: -0.0448
L11H1                     | Attribution: +0.0509
L11H2                     | Attribution: +0.0028
L11H3                     | Attribution: +0.0116
L11H4                     | Attribution: -0.0084
L11H5                     | Attribution: -0.0213
L11H6                     | Attribution: -0.0089
L11H7                     | Attribution: -0.0059
L11H8                     | Attribution: +0.0160
L11H9                     | Attribution: +0.0020
L11H10                    | Attribution: -0.0105
L11H11                    | Attribution: +0.0008
0_mlp_out                 | Attribution: -0.0227
1_mlp_out                 | Attribution: +0.0117
2_mlp_out                 | Attribution: -0.0068
3_mlp_out                 | Attribution: +0.0134
4_mlp_out                 | Attribution: +0.0031
5_mlp_out                 | Attribution: -0.0069
6_mlp_out                 | Attribution: -0.0266
7_mlp_out                 | Attribution: +0.1164
8_mlp_out                 | Attribution: +0.0598
9_mlp_out                 | Attribution: +0.0423
10_mlp_out                | Attribution: -0.1174
11_mlp_out                | Attribution: +0.1368
embed                     | Attribution: -0.0033
pos_embed                 | Attribution: -0.0023
bias                      | Attribution: +0.0511
```


### Cell 31 — markdown

Again getting the sum roughly 0.4658 not near to Logit_difference= 0.64, So we will calculate unembed_bias diff which added to the sum should be equal to actual logit_diff


### Cell 32 — code (execution count: 19)

```python
_# Extract the raw, un-decomposed final residual vector at the last position
raw_last_resid = cache["resid_post", -1][0, -1, :]  # shape: [d_model]

# Pass through official final LN
ln_final_out = model.ln_final(raw_last_resid)

# Calculate exact manual logit difference
manual_logit_diff = (ln_final_out @ model.W_U[:, target_id] - ln_final_out @ model.W_U[:, corrupt_id]).item()
actual_cache_diff = (logits[0, -1, target_id] - logits[0, -1, corrupt_id]).item()

print(f"Manual LN Logit Diff:  {manual_logit_diff:+.4f}")
print(f"Actual Cache Logit Diff: {actual_cache_diff:+.4f}")
```

#### Recorded output

```text
Manual LN Logit Diff:  +0.4662
Actual Cache Logit Diff: +0.6443
```


### Cell 33 — code (execution count: 20)

```python
bias_diff = (model.b_U[target_id] - model.b_U[corrupt_id]).item()
print(f"Bias Difference: {bias_diff:+.4f}")
```

#### Recorded output

```text
Bias Difference: +0.1782
```


### Cell 34 — code (execution count: 21)

```python
# 1. Target head index 0 specifically in hook_z (shape: [batch, pos, n_heads, d_head])
def zero_suppressor_hook(value, hook):
    value[:, -1, 0, :] = 0.0  # Zero out head 0 (L11H0) at position '='
    return value

# 2. Pass fwd_hooks into model.hooks()
model.reset_hooks()
with model.hooks(fwd_hooks=[("blocks.11.attn.hook_z", zero_suppressor_hook)]):
    patched_logits = model(prompt)
    top_token_id = patched_logits[0, -1].argmax().item()

    print("New Top Token:", repr(model.to_string(top_token_id)))
```

#### Recorded output

```text
New Top Token: ' 2'
```


### Cell 35 — code (execution count: 22)

```python
import torch

# 1. Target the actual object measured in DLA: hook_z (shape: [batch, pos, n_heads, d_head])
def z_ablation_hook(value, hook):
    value[:, -1, 0, :] = 0.0  # Zero out L11H0's output vector at the final position
    return value

# 2. Run clean forward pass vs. ablated forward pass
model.reset_hooks()
with torch.no_grad():
    clean_logits = model(prompt)

model.reset_hooks()
with model.hooks(fwd_hooks=[("blocks.11.attn.hook_z", z_ablation_hook)]):
    with torch.no_grad():
        ablated_logits = model(prompt)

# 3. Evaluate the quantitative impact on the specific axis of interest
target_id = model.to_single_token(" 8")
corrupt_id = model.to_single_token(" 1") # or ' 9'

clean_diff = (clean_logits[0, -1, target_id] - clean_logits[0, -1, corrupt_id]).item()
ablated_diff = (ablated_logits[0, -1, target_id] - ablated_logits[0, -1, corrupt_id]).item()

print(f"Clean Logit Diff (' 8' vs baseline):   {clean_diff:+.4f}")
print(f"Ablated Logit Diff (' 8' vs baseline): {ablated_diff:+.4f}")
print(f"Delta (Shift toward target):           {ablated_diff - clean_diff:+.4f}")

# Check what the new top-1 token actually is
new_top_id = ablated_logits[0, -1].argmax().item()
print(f"New Top-1 Token: {repr(model.to_string(new_top_id))}")
```

#### Recorded output

```text
Clean Logit Diff (' 8' vs baseline):   -0.3539
Ablated Logit Diff (' 8' vs baseline): -0.3238
Delta (Shift toward target):           +0.0301
New Top-1 Token: ' 2'
```


### Cell 36 — markdown

#### Isolated Zero-Ablation on `L11H0`

* **Objective:** Test whether zero-ablating `L11H0` changes the target-versus-foil
logit difference on `3 + 5 =`.

* **Intervention:** Zeroed `blocks.11.attn.hook_z` at the final position.

* **Initial observation:** Zero-ablation changed the measured logit difference and
also changed the top-ranked token.

* **Important control:** Because zero-ablation substantially changed the final
LayerNorm scale, the raw intervention is not a clean estimate of the head's
prompt-dependent causal contribution.

The apparent effect therefore cannot be interpreted from the zero-ablation result
alone. The subsequent mean-ablation experiment provides the scale-preserving
causal check.


### Cell 37 — code (execution count: 23)

```python
model.reset_hooks()
```


### Cell 38 — code (execution count: 24)

```python
import torch

prompt = "3 + 5 ="
target_id  = model.to_single_token(" 8")
corrupt_id = model.to_single_token(" 9")
token_2_id = model.to_single_token(" 2")
token_1_id = model.to_single_token(" 1")  # Clean Top-1 token

model.reset_hooks()
if hasattr(model, 'clear_cache'): model.clear_cache()

# 1. Clean Pass with LayerNorm Scale Capture
with torch.no_grad():
    clean_logits, clean_cache = model.run_with_cache(prompt)

# 2. Ablated Pass with LayerNorm Scale Capture
def z_ablation_hook(value, hook):
    value[:, -1, 0, :] = 0.0
    return value

with model.hooks(fwd_hooks=[("blocks.11.attn.hook_z", z_ablation_hook)]):
    with torch.no_grad():
        ablated_logits, ablated_cache = model.run_with_cache(prompt)

# Extract final LayerNorm scale factors (std dev multiplier)
clean_scale = clean_cache["ln_final.hook_scale"][0, -1, 0].item()
ablated_scale = ablated_cache["ln_final.hook_scale"][0, -1, 0].item()

tokens = {
    "Target (' 8')": target_id,
    "Corrupt (' 9')": corrupt_id,
    "Token (' 2')": token_2_id,
    "Token (' 1')": token_1_id,
}

print("=== RAW LOGIT SHIFTS (INCLUDING ' 1') ===")
for name, t_id in tokens.items():
    c_val = clean_logits[0, -1, t_id].item()
    a_val = ablated_logits[0, -1, t_id].item()
    print(f"{name:<15} | Clean: {c_val:+.4f} -> Ablated: {a_val:+.4f} | Shift: {a_val - c_val:+.4f}")

print("\n=== LAYERNORM SCALE SHIFT ===")
print(f"Clean LN Scale (sigma):   {clean_scale:.4f}")
print(f"Ablated LN Scale (sigma): {ablated_scale:.4f}")
print(f"Scale Delta:              {ablated_scale - clean_scale:+.4f}")
```

#### Recorded output

```text
=== RAW LOGIT SHIFTS (INCLUDING ' 1') ===
Target (' 8')   | Clean: +13.0319 -> Ablated: +12.0324 | Shift: -0.9995
Corrupt (' 9')  | Clean: +12.3876 -> Ablated: +11.3223 | Shift: -1.0653
Token (' 2')    | Clean: +13.3621 -> Ablated: +12.3633 | Shift: -0.9988
Token (' 1')    | Clean: +13.3858 -> Ablated: +12.3561 | Shift: -1.0297

=== LAYERNORM SCALE SHIFT ===
Clean LN Scale (sigma):   19.2002
Ablated LN Scale (sigma): 23.1286
Scale Delta:              +3.9283
```


### Cell 39 — markdown

###  Raw Logit & LayerNorm Scale ($\sigma$) Tracking
* **Objective:** Determine whether the $+0.0658$ delta and `' 2'` argmax flip reflect targeted circuit suppression or a global LayerNorm artifact.
* **Method:** Measured raw token logits alongside `ln_final.hook_scale` before and after zero-ablation.

**Findings:**
1. **LayerNorm Scale Explosion:** $\sigma$ jumped from **19.2002** to **23.1286** (+20.46%). Because LayerNorm divides activations by $\sigma$, all downstream logits dropped roughly uniformly by ~1.0 (`' 8'`: -0.9995, `' 9'`: -1.0653, `' 2'`: -0.9988, `' 1'`: -1.0297).
2. **Knife-Edge Flip:** Clean `' 1'` (+13.3858) led `' 2'` (+13.3621) by a microscopic margin of **0.0237 logits**. The differential decay rate (0.0309) crossed this gap, letting `' 2'` take top spot.
3. **Conclusion:** Zero-ablation causes a severe global norm collapse. The argmax flip is a rank-order illusion acting on near-tied tokens.


### Cell 40 — code (execution count: 25)

```python
import torch

# 1. Setup Prompt & Tokens
prompt = "3 + 5 ="
target_id  = model.to_single_token(" 8")
corrupt_id = model.to_single_token(" 9")
token_2_id = model.to_single_token(" 2")
token_1_id = model.to_single_token(" 1")

# 2. Compute Reference Mean Activation for L11H0
# (Mean-ablation replaces prompt-specific information while preserving baseline scale)
reference_prompts = [
    "1 + 1 =", "2 + 3 =", "4 + 4 =", "6 + 2 =",
    "The capital of France is", "The sky is blue and the", "Output the final result:"
]

z_activations = []
for ref_p in reference_prompts:
    with torch.no_grad():
        _, ref_cache = model.run_with_cache(ref_p)
        # Store L11H0 output vector at the final position
        z_activations.append(ref_cache["blocks.11.attn.hook_z"][0, -1, 0, :])

mean_z_vector = torch.stack(z_activations).mean(dim=0)

# 3. Clean Forward Pass
model.reset_hooks()
if hasattr(model, 'clear_cache'):
    model.clear_cache()

with torch.no_grad():
    clean_logits, clean_cache = model.run_with_cache(prompt)

clean_scale = clean_cache["ln_final.hook_scale"][0, -1, 0].item()

# 4. Mean-Ablation Hook
def mean_ablation_hook(value, hook):
    value[:, -1, 0, :] = mean_z_vector  # Inject mean vector instead of 0.0
    return value

with model.hooks(fwd_hooks=[("blocks.11.attn.hook_z", mean_ablation_hook)]):
    with torch.no_grad():
        ablated_logits, ablated_cache = model.run_with_cache(prompt)

ablated_scale = ablated_cache["ln_final.hook_scale"][0, -1, 0].item()

# 5. Output Results
clean_diff = (clean_logits[0, -1, target_id] - clean_logits[0, -1, corrupt_id]).item()
ablated_diff = (ablated_logits[0, -1, target_id] - ablated_logits[0, -1, corrupt_id]).item()

print("=== GROUND-TRUTH PAIR EVALUATION (' 8' vs ' 9') ===")
print(f"Clean Logit Diff (' 8' - ' 9'):   {clean_diff:+.4f}")
print(f"Ablated Logit Diff (' 8' - ' 9'): {ablated_diff:+.4f}")
print(f"Causal Delta (' 8' vs ' 9'):      {ablated_diff - clean_diff:+.4f}")

tokens = {
    "Target (' 8')": target_id,
    "Corrupt (' 9')": corrupt_id,
    "Token (' 2')": token_2_id,
    "Token (' 1')": token_1_id,
}

print("\n=== RAW LOGIT SHIFTS (MEAN-ABLATION) ===")
for name, t_id in tokens.items():
    c_val = clean_logits[0, -1, t_id].item()
    a_val = ablated_logits[0, -1, t_id].item()
    print(f"{name:<15} | Clean: {c_val:+.4f} -> Ablated: {a_val:+.4f} | Shift: {a_val - c_val:+.4f}")

print("\n=== LAYERNORM SCALE SHIFT ===")
print(f"Clean LN Scale (sigma):   {clean_scale:.4f}")
print(f"Ablated LN Scale (sigma): {ablated_scale:.4f}")
print(f"Scale Delta:              {ablated_scale - clean_scale:+.4f}")

clean_top_id = clean_logits[0, -1].argmax().item()
ablated_top_id = ablated_logits[0, -1].argmax().item()
print("\n=== TOP PREDICTION SHIFT ===")
print(f"Clean Top-1 Token:   {repr(model.to_string(clean_top_id))}")
print(f"Ablated Top-1 Token: {repr(model.to_string(ablated_top_id))}")
```

#### Recorded output

```text
=== GROUND-TRUTH PAIR EVALUATION (' 8' vs ' 9') ===
Clean Logit Diff (' 8' - ' 9'):   +0.6443
Ablated Logit Diff (' 8' - ' 9'): +0.6456
Causal Delta (' 8' vs ' 9'):      +0.0013

=== RAW LOGIT SHIFTS (MEAN-ABLATION) ===
Target (' 8')   | Clean: +13.0319 -> Ablated: +13.0615 | Shift: +0.0296
Corrupt (' 9')  | Clean: +12.3876 -> Ablated: +12.4159 | Shift: +0.0283
Token (' 2')    | Clean: +13.3621 -> Ablated: +13.3629 | Shift: +0.0007
Token (' 1')    | Clean: +13.3858 -> Ablated: +13.3663 | Shift: -0.0195

=== LAYERNORM SCALE SHIFT ===
Clean LN Scale (sigma):   19.2002
Ablated LN Scale (sigma): 19.0148
Scale Delta:              -0.1854

=== TOP PREDICTION SHIFT ===
Clean Top-1 Token:   ' 1'
Ablated Top-1 Token: ' 4'
```


### Cell 41 — markdown

### Mean-Ablation Control: `L11H0`

The scale-preserving mean-ablation intervention reduced the apparent causal shift
from the zero-ablation result to approximately +0.0013 in the `8 vs 9` logit
difference.

Under this controlled intervention, the evidence does not support `L11H0` as an
active prompt-dependent suppressor for this target-versus-foil metric.

The zero-ablation result is therefore treated as a methodological false positive
caused primarily by the off-distribution perturbation and resulting LayerNorm
change.


### Cell 42 — code (execution count: 26)

```python
with model.hooks(fwd_hooks=[("blocks.11.attn.hook_z", zero_suppressor_hook)]):
    patched_logits = model(prompt)[0, -1]
    sorted_indices = torch.argsort(patched_logits, descending=True)

    target_id = model.to_single_token(" 8")
    new_rank = (sorted_indices == target_id).nonzero().item() + 1
    print(f"New Rank of ' 8': {new_rank} | Logit: {patched_logits[target_id].item():.4f}")
```

#### Recorded output

```text
New Rank of ' 8': 7 | Logit: 12.0324
```


### Cell 43 — code (execution count: 27)

```python
def zero_mlp10_hook(value, hook):
    value[:, -1, :] = 0.0  # Zero out final token output for MLP 10
    return value

model.reset_hooks()
with model.hooks(fwd_hooks=[("blocks.10.hook_mlp_out", zero_mlp10_hook)]):
    patched_logits = model(prompt)[0, -1]

    target_logit = patched_logits[target_id].item()
    corrupt_logit = patched_logits[corrupt_id].item()

    sorted_indices = torch.argsort(patched_logits, descending=True)
    rank = (sorted_indices == target_id).nonzero().item() + 1
    top_token = model.to_string(patched_logits.argmax().item())

    print(f"Top Token: {repr(top_token)}")
    print(f"Rank of ' 8': {rank}")
    print(f"Target Logit (' 8'): {target_logit:.4f}")
    print(f"Logit Diff (' 8' - ' 9'): {target_logit - corrupt_logit:+.4f}")
```

#### Recorded output

```text
Top Token: ' 1'
Rank of ' 8': 9
Target Logit (' 8'): 11.7080
Logit Diff (' 8' - ' 9'): +0.9142
```


### Cell 44 — code (execution count: 28)

```python
model.reset_hooks()
hooks = [
    ("blocks.10.hook_mlp_out", zero_mlp10_hook),
    ("blocks.11.attn.hook_z", zero_suppressor_hook)
]

with model.hooks(fwd_hooks=hooks):
    patched_logits = model(prompt)[0, -1]
    sorted_indices = torch.argsort(patched_logits, descending=True)
    rank = (sorted_indices == target_id).nonzero().item() + 1
    print(f"Double Ablation Rank of ' 8': {rank} | Top Token: {repr(model.to_string(patched_logits.argmax().item()))}")
```

#### Recorded output

```text
Double Ablation Rank of ' 8': 8 | Top Token: ' 1'
```


### Cell 45 — markdown

###  `MLP10` Single and Double Ablation Analysis
* **Objective:** Evaluate the individual and joint causal roles of `MLP10` and `L11H0` on token `' 8'` logit, rank, and target logit difference ($\text{Logit}(' 8') - \text{Logit}(' 9')$).

**Key Results:**
1. **`MLP10` Ablation:** Drains significant residual norm. Token `' 8'` logit drops to **11.7080** (Rank 9), but the logit difference widens to **+0.9142**, indicating `MLP10` contributed more baseline positive logit to `' 9'` than to `' 8'`.
2. **Double Ablation (`L11H0` + `MLP10`):** Token `' 8'` settles at Rank 8. Top-1 prediction reverts from `' 2'` back to `' 1'`.
3. **Takeaway:** Compounded zero-ablations exacerbate LayerNorm scale shifts. The reversal of the Top-1 token back to `' 1'` confirms that individual token flips in high-entropy logit regions under zero-ablation are non-linear scale artifacts rather than isolated circuit controls.


### Cell 46 — code (execution count: 29)

```python
import torch

prompt = "3 + 5 ="
target_id  = model.to_single_token(" 8")
corrupt_id = model.to_single_token(" 9")
token_1_id = model.to_single_token(" 1")

reference_prompts = [
    "1 + 1 =", "2 + 3 =", "4 + 4 =", "6 + 2 =",
    "The capital of France is", "The sky is blue and the", "Output the final result:"
]

# 1. Compute Mean Vectors for L11H0 and MLP10
z_activations = []
mlp_activations = []

for ref_p in reference_prompts:
    with torch.no_grad():
        _, ref_cache = model.run_with_cache(ref_p)
        z_activations.append(ref_cache["blocks.11.attn.hook_z"][0, -1, 0, :])
        mlp_activations.append(ref_cache["blocks.10.hook_mlp_out"][0, -1, :])

mean_z_vector = torch.stack(z_activations).mean(dim=0)
mean_mlp_vector = torch.stack(mlp_activations).mean(dim=0)

# 2. Hooks for Single and Double Mean-Ablation
def mean_z_hook(value, hook):
    value[:, -1, 0, :] = mean_z_vector
    return value

def mean_mlp_hook(value, hook):
    value[:, -1, :] = mean_mlp_vector
    return value

# 3. Execution Across Conditions
model.reset_hooks()
if hasattr(model, 'clear_cache'): model.clear_cache()

# Clean Pass
with torch.no_grad():
    clean_logits, clean_cache = model.run_with_cache(prompt)

# Single MLP10 Mean-Ablation
with model.hooks(fwd_hooks=[("blocks.10.hook_mlp_out", mean_mlp_hook)]):
    with torch.no_grad():
        mlp_logits, mlp_cache = model.run_with_cache(prompt)

# Double Mean-Ablation (L11H0 + MLP10)
double_hooks = [
    ("blocks.11.attn.hook_z", mean_z_hook),
    ("blocks.10.hook_mlp_out", mean_mlp_hook)
]
with model.hooks(fwd_hooks=double_hooks):
    with torch.no_grad():
        double_logits, double_cache = model.run_with_cache(prompt)

# 4. Extract Metrics
def evaluate_condition(name, logits, cache):
    t_logit = logits[0, -1, target_id].item()
    c_logit = logits[0, -1, corrupt_id].item()
    diff = t_logit - c_logit
    sigma = cache["ln_final.hook_scale"][0, -1, 0].item()
    top_tok = repr(model.to_string(logits[0, -1].argmax().item()))
    rank_8 = (logits[0, -1].argsort(descending=True) == target_id).nonzero().item() + 1
    print(f"{name:<22} | Logit(' 8'): {t_logit:+.4f} | Diff: {diff:+.4f} | σ: {sigma:.4f} | Rank(' 8'): {rank_8} | Top: {top_tok}")

print("=== CONTROLLED MEAN-ABLATION INTERACTION MATRIX ===")
evaluate_condition("Clean Baseline", clean_logits, clean_cache)
evaluate_condition("MLP10 Mean-Ablated", mlp_logits, mlp_cache)
evaluate_condition("Double Mean-Ablated", double_logits, double_cache)
```

#### Recorded output

```text
=== CONTROLLED MEAN-ABLATION INTERACTION MATRIX ===
Clean Baseline         | Logit(' 8'): +13.0319 | Diff: +0.6443 | σ: 19.2002 | Rank(' 8'): 7 | Top: ' 1'
MLP10 Mean-Ablated     | Logit(' 8'): +12.3468 | Diff: +0.8118 | σ: 17.8064 | Rank(' 8'): 7 | Top: ' 1'
Double Mean-Ablated    | Logit(' 8'): +12.3900 | Diff: +0.8127 | σ: 17.5919 | Rank(' 8'): 7 | Top: ' 1'
```


### Cell 47 — markdown

### Controlled Mean-Ablation Interaction

Mean-ablation of `MLP10` increased the `8 vs 9` logit difference on this prompt,
indicating a prompt-local causal contribution to this output metric.

However, this single-prompt effect does not establish `MLP10` as an arithmetic
circuit component. The result must be replicated across an independent cohort and
tested against matched non-arithmetic controls before assigning a task-specific
mechanistic role.

Double mean-ablation produced an approximately additive result, with no detectable
interaction between `MLP10` and `L11H0` under this intervention.


### Cell 48 — code (execution count: 30)

```python
import torch
import plotly.express as px

clean_prompt = "3 + 5 ="
corrupt_prompt = "3 + 9 ="

target_id = model.to_single_token(" 8")
corrupt_id = model.to_single_token(" 9")

# 1. Run clean and corrupt passes
clean_logits, clean_cache = model.run_with_cache(clean_prompt)
corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_prompt)

clean_diff = (clean_logits[0, -1, target_id] - clean_logits[0, -1, corrupt_id]).item()
corrupt_diff = (corrupt_logits[0, -1, target_id] - corrupt_logits[0, -1, corrupt_id]).item()

# 2. Patch residual stream position by position across layers
n_layers = model.cfg.n_layers
seq_len = clean_cache["resid_pre", 0].shape[1]
patch_results = torch.zeros(n_layers, seq_len)

for layer in range(n_layers):
    for pos in range(seq_len):
        def patch_hook(corrupt_activation, hook):
            corrupt_activation[:, pos, :] = clean_cache[hook.name][:, pos, :]
            return corrupt_activation

        hook_name = f"blocks.{layer}.hook_resid_post"
        patched_logits = model.run_with_hooks(
            corrupt_prompt,
            fwd_hooks=[(hook_name, patch_hook)]
        )

        patched_diff = (patched_logits[0, -1, target_id] - patched_logits[0, -1, corrupt_id]).item()
        # Normalized metric: 0 = corrupt performance, 1 = clean recovery
        patch_results[layer, pos] = (patched_diff - corrupt_diff) / (clean_diff - corrupt_diff)

# 3. Display causal heatmap
tokens = [model.to_string(t) for t in model.to_tokens(clean_prompt)[0]]
fig = px.imshow(
    patch_results.numpy(),
    x=tokens,
    y=[f"L{i}" for i in range(n_layers)],
    labels=dict(x="Sequence Position", y="Layer", color="Normalized Recovery"),
    title="Residual Stream Activation Patching (3 + 5 =)"
)
fig.show()
```

#### Recorded output

```text
[Rich HTML output omitted; no text representation was present.]
```


### Cell 49 — code (execution count: 31)

```python
import torch
import plotly.express as px

# Patch individual head outputs (hook_z) at position '='
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
head_results = torch.zeros(n_layers, n_heads)

for layer in range(n_layers):
    for head in range(n_heads):
        def patch_head_hook(corrupt_activation, hook):
            # Patch only the specific head at the last token position ('=')
            corrupt_activation[:, -1, head, :] = clean_cache[hook.name][:, -1, head, :]
            return corrupt_activation

        hook_name = f"blocks.{layer}.attn.hook_z"
        patched_logits = model.run_with_hooks(
            corrupt_prompt,
            fwd_hooks=[(hook_name, patch_head_hook)]
        )

        patched_diff = (patched_logits[0, -1, target_id] - patched_logits[0, -1, corrupt_id]).item()
        head_results[layer, head] = (patched_diff - corrupt_diff) / (clean_diff - corrupt_diff)

fig = px.imshow(
    head_results.numpy(),
    x=[f"H{i}" for i in range(n_heads)],
    y=[f"L{i}" for i in range(n_layers)],
    labels=dict(x="Head", y="Layer", color="Normalized Recovery"),
    title="Attention Head Activation Patching at Position '='"
)
fig.show()
```

#### Recorded output

```text
[Rich HTML output omitted; no text representation was present.]
```


### Cell 50 — code (execution count: 32)

```python
import plotly.express as px

# Extract attention patterns for the prompt
tokens = [model.to_string(t) for t in model.to_tokens(clean_prompt)[0]]

# Extract attention matrix for Layer 10, Head 2 and copy to cuda first
pattern_l10h2 = clean_cache["blocks.10.attn.hook_pattern"][0, 2].cpu().numpy()

fig = px.imshow(
    pattern_l10h2,
    x=tokens,
    y=tokens,
    labels=dict(x="Key (Attended To)", y="Query (Attending From)", color="Attention Weight"),
    title="L10H2 Attention Pattern (3 + 5 =)"
)
fig.show()
```

#### Recorded output

```text
[Rich HTML output omitted; no text representation was present.]
```


### Cell 51 — code (execution count: 33)

```python
import torch
import plotly.express as px

clean_prompt = "3 + 5 ="
tokens = [model.to_string(t) for t in model.to_tokens(clean_prompt)[0]]

# 1. Run forward pass and extract cache
with torch.no_grad():
    _, clean_cache = model.run_with_cache(clean_prompt)

# 2. Extract L9H9 attention pattern matrix [query_pos, key_pos]
pattern_l9h9 = clean_cache["blocks.9.attn.hook_pattern"][0, 9]

# 3. Print exact attention weights originating from query '=' (index -1)
print("=== L9H9 ATTENTION FROM TERMINAL '=' POSITION ===")
for tok, weight in zip(tokens, pattern_l9h9[-1, :]):
    print(f"Key Token: {repr(tok):<15} | Weight: {weight.item():.4f}")

# 4. Render full attention heatmap
fig = px.imshow(
    pattern_l9h9.cpu().numpy(),
    x=tokens,
    y=tokens,
    labels=dict(x="Key (Attended To)", y="Query (Attending From)", color="Attention Weight"),
    title="L9H9 Attention Pattern (3 + 5 =)",
    color_continuous_scale="Viridis",
    text_auto=".2f"
)
fig.show()
```

#### Recorded output

```text
=== L9H9 ATTENTION FROM TERMINAL '=' POSITION ===
Key Token: '<|endoftext|>' | Weight: 0.8810
Key Token: '3'             | Weight: 0.0384
Key Token: ' +'            | Weight: 0.0097
Key Token: ' 5'            | Weight: 0.0674
Key Token: ' ='            | Weight: 0.0035
```

```text
[Rich HTML output omitted; no text representation was present.]
```


### Cell 52 — code (execution count: 34)

```python
# 1. Get L10H2 internal activations at final position '='
# hook_v shape: [batch, pos, head, d_head]
v = cache["blocks.10.attn.hook_v"][0, :, 2, :]  # All key positions for Head 2
pattern = cache["blocks.10.attn.hook_pattern"][0, 2, -1, :]  # Query '=' attending to all keys

W_O = model.blocks[10].attn.W_O[2]  # Output projection for Head 2
unembed_diff = model.W_U[:, target_id] - model.W_U[:, corrupt_id]

# 2. Measure contribution of EACH source token through L10H2
tokens = [model.to_string(t) for t in model.to_tokens(clean_prompt)[0]]
print(f"{'Source Token':<15} | {'Attn Weight':<12} | {'Logit Diff Contribution':<25}")
print("-" * 55)

for pos, token in enumerate(tokens):
    attn_w = pattern[pos].item()
    # Individual head output vector from this specific key position
    head_out_pos = (v[pos] @ W_O) * attn_w
    contribution = torch.dot(head_out_pos, unembed_diff).item()
    print(f"{repr(token):<15} | {attn_w:<12.4f} | {contribution:+.4f}")
```

#### Recorded output

```text
Source Token    | Attn Weight  | Logit Diff Contribution
-------------------------------------------------------
'<|endoftext|>' | 0.7334       | -0.1334
'3'             | 0.0189       | -0.0214
' +'            | 0.1202       | +0.2346
' 5'            | 0.0792       | +0.2270
' ='            | 0.0483       | -0.1408
```


### Cell 53 — code (execution count: 35)

```python
# 1. Get L9h9 internal activations at final position '='
# hook_v shape: [batch, pos, head, d_head]
v = cache["blocks.9.attn.hook_v"][0, :, 9, :]  # All key positions for Head 2
pattern = cache["blocks.9.attn.hook_pattern"][0, 9, -1, :]  # Query '=' attending to all keys

W_O = model.blocks[9].attn.W_O[9]  # Output projection for Head 2
unembed_diff = model.W_U[:, target_id] - model.W_U[:, corrupt_id]

# 2. Measure contribution of EACH source token through L10H2
tokens = [model.to_string(t) for t in model.to_tokens(clean_prompt)[0]]
print(f"{'Source Token':<15} | {'Attn Weight':<12} | {'Logit Diff Contribution':<25}")
print("-" * 55)

for pos, token in enumerate(tokens):
    attn_w = pattern[pos].item()
    # Individual head output vector from this specific key position
    head_out_pos = (v[pos] @ W_O) * attn_w
    contribution = torch.dot(head_out_pos, unembed_diff).item()
    print(f"{repr(token):<15} | {attn_w:<12.4f} | {contribution:+.4f}")
```

#### Recorded output

```text
Source Token    | Attn Weight  | Logit Diff Contribution
-------------------------------------------------------
'<|endoftext|>' | 0.8810       | +0.0980
'3'             | 0.0384       | -0.4393
' +'            | 0.0097       | -0.0204
' 5'            | 0.0674       | -0.2998
' ='            | 0.0035       | -0.0060
```


### Cell 54 — markdown

### Attention-Pattern Diagnostic

The heads allocate substantial attention to the initial/end-of-text positions in
the tested prompt. This is descriptive evidence about attention allocation only.

Attention weight alone does not establish that the attended token carries the
task-relevant information or that the head participates causally in arithmetic.
The following intervention and control analyses are therefore required before
assigning a mechanistic role.


### Cell 55 — code (execution count: 36)

```python
import torch

clean_p = "3 + 5 ="
corrupt_p = "1 + 5 ="

target_id  = model.to_single_token(" 8")
corrupt_id = model.to_single_token(" 6")

# 1. Run baseline passes
clean_logits, clean_cache = model.run_with_cache(clean_p)
corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_p)

base_diff = (corrupt_logits[0, -1, target_id] - corrupt_logits[0, -1, corrupt_id]).item()

# Compute L9H9 delta in residual space (d_model = 768)
clean_z_l9h9 = clean_cache["blocks.9.attn.hook_z"][0, -1, 9, :]
corrupt_z_l9h9 = corrupt_cache["blocks.9.attn.hook_z"][0, -1, 9, :]
delta_l9h9 = (clean_z_l9h9 - corrupt_z_l9h9) @ model.blocks[9].attn.W_O[9]

# 2. Targeted Path Patching Hooks
def patch_mlp10(val, hook):
    val[0, -1, :] += delta_l9h9
    return val

def patch_mlp11(val, hook):
    val[0, -1, :] += delta_l9h9
    return val

def patch_l10h2_v(val, hook):
    # Project delta_l9h9 into L10H2's value space (d_head = 64)
    delta_v_l10h2 = delta_l9h9 @ model.blocks[10].attn.W_V[2]
    val[0, -1, 2, :] += delta_v_l10h2
    return val

receivers = [
    ("MLP 10", "blocks.10.hook_mlp_in", patch_mlp10),
    ("MLP 11", "blocks.11.hook_mlp_in", patch_mlp11),
    ("L10H2",  "blocks.10.attn.hook_v", patch_l10h2_v),
]

print("\n=== L9H9 PATH PATCHING TO DOWNSTREAM RECEIVERS ===")
for name, hook_name, hook_fn in receivers:
    model.reset_hooks()
    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        patched_logits = model.run_with_hooks(corrupt_p)
        p_diff = (patched_logits[0, -1, target_id] - patched_logits[0, -1, corrupt_id]).item()
        recovery = p_diff - base_diff
        print(f"Path L9H9 -> {name:<10} | Patched Logit Diff: {p_diff:+.4f} | Delta Recovery: {recovery:+.4f}")
```

#### Recorded output

```text

=== L9H9 PATH PATCHING TO DOWNSTREAM RECEIVERS ===
Path L9H9 -> MLP 10     | Patched Logit Diff: -0.2795 | Delta Recovery: +0.0000
Path L9H9 -> MLP 11     | Patched Logit Diff: -0.2795 | Delta Recovery: +0.0000
Path L9H9 -> L10H2      | Patched Logit Diff: -0.2801 | Delta Recovery: -0.0006
```


### Cell 56 — markdown

### L9H9 Path-Patching Status

The tested L9H9 downstream paths produced negligible recovery relative to the
corrupt condition.

Because the path-patching implementation was not sufficiently validated against
a complete causal baseline, these measurements are not used as positive or
negative evidence for the final circuit claim.

They are retained as an audited but unresolved analysis.


### Cell 57 — code (execution count: 37)

```python
import torch

# Trace which early layer (L0-L7) moves '3' into position '5'
corrupt_prompt = "3 + 9 ="
clean_prompt = "3 + 5 ="

# Patching position ' 5' across early layers
for layer in range(8):
    def patch_pos_hook(corrupt_act, hook):
        corrupt_act[:, 3, :] = clean_cache[hook.name][:, 3, :] # pos 3 is ' 5'
        return corrupt_act

    patched_logits = model.run_with_hooks(
        corrupt_prompt,
        fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch_pos_hook)]
    )
    diff = (patched_logits[0, -1, target_id] - patched_logits[0, -1, corrupt_id]).item()
    print(f"Layer {layer} Patch at pos ' 5' -> Logit Diff: {diff:+.4f}")
```

#### Recorded output

```text
Layer 0 Patch at pos ' 5' -> Logit Diff: -0.2645
Layer 1 Patch at pos ' 5' -> Logit Diff: -0.2605
Layer 2 Patch at pos ' 5' -> Logit Diff: -0.2448
Layer 3 Patch at pos ' 5' -> Logit Diff: -0.2209
Layer 4 Patch at pos ' 5' -> Logit Diff: -0.2404
Layer 5 Patch at pos ' 5' -> Logit Diff: -0.2473
Layer 6 Patch at pos ' 5' -> Logit Diff: -0.2448
Layer 7 Patch at pos ' 5' -> Logit Diff: -0.2444
```


### Cell 58 — code (execution count: 38)

```python
clean_prompt = "3 + 5 ="
corrupt_prompt = "1 + 5 ="

# Clean target is ' 8' (3+5), Corrupt target is ' 6' (1+5)
target_id = model.to_single_token(" 8")
corrupt_id = model.to_single_token(" 6")

clean_logits, clean_cache = model.run_with_cache(clean_prompt)
corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_prompt)

clean_diff = (clean_logits[0, -1, target_id] - clean_logits[0, -1, corrupt_id]).item()
corrupt_diff = (corrupt_logits[0, -1, target_id] - corrupt_logits[0, -1, corrupt_id]).item()

print(f"Clean Diff (' 8' - ' 6'): {clean_diff:+.4f}")
print(f"Corrupt Diff (' 8' - ' 6'): {corrupt_diff:+.4f}\n")

# Patch position ' 5' (index 3) across layers 0 to 8
for layer in range(9):
    def patch_second_operand(corrupt_act, hook):
        corrupt_act[:, 3, :] = clean_cache[hook.name][:, 3, :]
        return corrupt_act

    patched_logits = model.run_with_hooks(
        corrupt_prompt,
        fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch_second_operand)]
    )
    diff = (patched_logits[0, -1, target_id] - patched_logits[0, -1, corrupt_id]).item()
    norm_recovery = (diff - corrupt_diff) / (clean_diff - corrupt_diff)
    print(f"Layer {layer} Patch at pos ' 5' -> Logit Diff: {diff:+.4f} | Recovery: {norm_recovery*100:.1f}%")
```

#### Recorded output

```text
Clean Diff (' 8' - ' 6'): -0.2643
Corrupt Diff (' 8' - ' 6'): -0.2795

Layer 0 Patch at pos ' 5' -> Logit Diff: -0.2860 | Recovery: -43.1%
Layer 1 Patch at pos ' 5' -> Logit Diff: -0.2851 | Recovery: -37.0%
Layer 2 Patch at pos ' 5' -> Logit Diff: -0.2805 | Recovery: -6.9%
Layer 3 Patch at pos ' 5' -> Logit Diff: -0.2712 | Recovery: 54.5%
Layer 4 Patch at pos ' 5' -> Logit Diff: -0.2838 | Recovery: -28.5%
Layer 5 Patch at pos ' 5' -> Logit Diff: -0.2628 | Recovery: 110.0%
Layer 6 Patch at pos ' 5' -> Logit Diff: -0.3126 | Recovery: -217.7%
Layer 7 Patch at pos ' 5' -> Logit Diff: -0.3066 | Recovery: -178.4%
Layer 8 Patch at pos ' 5' -> Logit Diff: -0.2808 | Recovery: -8.9%
```


### Cell 59 — markdown

**Assumption** :Any recovery gained by patching position ' 5' will explicitly prove that information about '3' was moved into position ' 5'"

**Result/Interpretation** :The correct reading of this already-obtained result is the negative case: there is essentially no recovery, so this data does not support '3''s information having been relayed into position ' 5', at least not in layers 0–8, at least not in a form recoverable by patching the full residual stream there.


### Cell 60 — code (execution count: 39)

```python
clean_prompt = "3 + 5 ="
corrupt_prompt = "1 + 5 ="

target_id = model.to_single_token(" 8")
corrupt_id = model.to_single_token(" 6")

clean_logits, clean_cache = model.run_with_cache(clean_prompt)
corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_prompt)

# Patch position '=' (-1) across layers
for layer in range(model.cfg.n_layers):
    def patch_final_pos(corrupt_act, hook):
        corrupt_act[:, -1, :] = clean_cache[hook.name][:, -1, :]
        return corrupt_act

    patched_logits = model.run_with_hooks(
        corrupt_prompt,
        fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch_final_pos)]
    )
    diff = (patched_logits[0, -1, target_id] - patched_logits[0, -1, corrupt_id]).item()
    print(f"Layer {layer:<2} Patch at '=' -> Logit Diff (' 8' - ' 6'): {diff:+.4f}")
```

#### Recorded output

```text
Layer 0  Patch at '=' -> Logit Diff (' 8' - ' 6'): -0.2665
Layer 1  Patch at '=' -> Logit Diff (' 8' - ' 6'): -0.2663
Layer 2  Patch at '=' -> Logit Diff (' 8' - ' 6'): -0.2611
Layer 3  Patch at '=' -> Logit Diff (' 8' - ' 6'): -0.2564
Layer 4  Patch at '=' -> Logit Diff (' 8' - ' 6'): -0.2526
Layer 5  Patch at '=' -> Logit Diff (' 8' - ' 6'): -0.2788
Layer 6  Patch at '=' -> Logit Diff (' 8' - ' 6'): -0.2170
Layer 7  Patch at '=' -> Logit Diff (' 8' - ' 6'): -0.2177
Layer 8  Patch at '=' -> Logit Diff (' 8' - ' 6'): -0.2240
Layer 9  Patch at '=' -> Logit Diff (' 8' - ' 6'): -0.2447
Layer 10 Patch at '=' -> Logit Diff (' 8' - ' 6'): -0.2557
Layer 11 Patch at '=' -> Logit Diff (' 8' - ' 6'): -0.2643
```


### Cell 61 — markdown

8' vs ' 6' carries whatever static bias sits between those two specific tokens so,we will isolate operand-binding dynamics without that bias contaminating the signal, deliberately measuring against an unrelated, uncontaminated third token (' 9') while still corrupting the first operand in a sensible design choice.


### Cell 62 — code (execution count: 40)

```python
# --- Check (a): Static unembed bias for ' 8' vs ' 6' ---
bias_diff_86 = (model.b_U[target_id] - model.b_U[corrupt_id]).item()
print(f"Unembed Bias Diff (' 8' - ' 6'): {bias_diff_86:+.4f}")
```

#### Recorded output

```text
Unembed Bias Diff (' 8' - ' 6'): -0.1003
```


### Cell 63 — markdown

This cell proved our assumption from earlier about Large Static Bias Difference


### Cell 64 — code (execution count: 41)

```python
# --- Check (b): Raw individual logits, clean vs corrupt (not just the diff) ---
clean_target_logit  = clean_logits[0, -1, target_id].item()
clean_corrupt_logit = clean_logits[0, -1, corrupt_id].item()
corr_target_logit   = corrupt_logits[0, -1, target_id].item()
corr_corrupt_logit  = corrupt_logits[0, -1, corrupt_id].item()

print(f"' 8' logit  | Clean: {clean_target_logit:+.4f} -> Corrupt: {corr_target_logit:+.4f} | Shift: {corr_target_logit - clean_target_logit:+.4f}")
print(f"' 6' logit  | Clean: {clean_corrupt_logit:+.4f} -> Corrupt: {corr_corrupt_logit:+.4f} | Shift: {corr_corrupt_logit - clean_corrupt_logit:+.4f}")
```

#### Recorded output

```text
' 8' logit  | Clean: +13.0319 -> Corrupt: +13.2861 | Shift: +0.2542
' 6' logit  | Clean: +13.2962 -> Corrupt: +13.5655 | Shift: +0.2694
```


### Cell 65 — code (execution count: 42)

```python
# --- Check (c): Raw logit shift for every digit token, clean vs corrupt ---
digit_tokens = [str(d) for d in range(10)]  # includes '0' for completeness
digit_ids = {d: model.to_single_token(f" {d}") for d in digit_tokens}

results = []
for d, tid in digit_ids.items():
    clean_l = clean_logits[0, -1, tid].item()
    corr_l  = corrupt_logits[0, -1, tid].item()
    shift   = corr_l - clean_l
    results.append((d, clean_l, corr_l, shift))

# Sort by shift magnitude, largest first
results.sort(key=lambda x: -abs(x[3]))

print(f"{'Digit':<6} | {'Clean Logit':>12} | {'Corrupt Logit':>14} | {'Shift':>8}")
print("-" * 48)
for d, cl, cr, sh in results:
    marker = " <-- target/corrupt" if d in ("8", "6") else ""
    print(f"' {d}'    | {cl:>+12.4f} | {cr:>+14.4f} | {sh:>+8.4f}{marker}")

mean_shift = sum(r[3] for r in results) / len(results)
print(f"\nMean shift across all digits: {mean_shift:+.4f}")
print(f"Std dev of shifts: {(sum((r[3]-mean_shift)**2 for r in results)/len(results))**0.5:.4f}")
```

#### Recorded output

```text
Digit  |  Clean Logit |  Corrupt Logit |    Shift
------------------------------------------------
' 1'    |     +13.3858 |       +13.8655 |  +0.4797
' 0'    |     +12.5790 |       +13.0529 |  +0.4739
' 5'    |     +13.2961 |       +13.6616 |  +0.3655
' 2'    |     +13.3621 |       +13.7263 |  +0.3641
' 9'    |     +12.3876 |       +12.7415 |  +0.3539
' 7'    |     +12.8529 |       +13.1484 |  +0.2955
' 6'    |     +13.2962 |       +13.5655 |  +0.2694 <-- target/corrupt
' 8'    |     +13.0319 |       +13.2861 |  +0.2542 <-- target/corrupt
' 3'    |     +13.2515 |       +13.4189 |  +0.1673
' 4'    |     +13.3855 |       +13.4830 |  +0.0975

Mean shift across all digits: +0.3121
Std dev of shifts: 0.1158
```


### Cell 66 — code (execution count: 43)

```python
clean_prompt = "3 + 5 ="
corrupt_prompt = "1 + 5 ="

# Measure target ' 8' against an unrelated corrupt digit ' 9' to avoid ' 6' bias
target_id = model.to_single_token(" 8")
corrupt_id = model.to_single_token(" 9")

clean_logits, clean_cache = model.run_with_cache(clean_prompt)
corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_prompt)

# Patch operator position '+' (index 2)
for layer in range(8):
    def patch_op_pos(corrupt_act, hook):
        corrupt_act[:, 2, :] = clean_cache[hook.name][:, 2, :]
        return corrupt_act

    patched_logits = model.run_with_hooks(
        corrupt_prompt,
        fwd_hooks=[(f"blocks.{layer}.hook_resid_post", patch_op_pos)]
    )
    diff = (patched_logits[0, -1, target_id] - patched_logits[0, -1, corrupt_id]).item()
    print(f"Layer {layer} Patch at '+' -> Logit Diff (' 8' - ' 9'): {diff:+.4f}")
```

#### Recorded output

```text
Layer 0 Patch at '+' -> Logit Diff (' 8' - ' 9'): +0.5550
Layer 1 Patch at '+' -> Logit Diff (' 8' - ' 9'): +0.5720
Layer 2 Patch at '+' -> Logit Diff (' 8' - ' 9'): +0.6034
Layer 3 Patch at '+' -> Logit Diff (' 8' - ' 9'): +0.6208
Layer 4 Patch at '+' -> Logit Diff (' 8' - ' 9'): +0.6084
Layer 5 Patch at '+' -> Logit Diff (' 8' - ' 9'): +0.6034
Layer 6 Patch at '+' -> Logit Diff (' 8' - ' 9'): +0.6154
Layer 7 Patch at '+' -> Logit Diff (' 8' - ' 9'): +0.5939
```


### Cell 67 — markdown

### Operator-Position Patching: Initial Hypothesis

Earlier experiments suggested that information might be routed through the
operator position before reaching the final prediction.

The purpose of the following analysis is to test that hypothesis rather than
assume it.

Recovery after operator-position patching is not, by itself, sufficient to prove
that an operator-specific representation was copied or routed through that
position. A valid causal interpretation requires clean and corrupt baselines,
matched interventions, and controls for positional and token-specific effects.


### Cell 68 — code (execution count: 44)

```python
import torch

# Inspect attention weights from Query=' +' (index 2) to Key='3' (index 1)
tokens = [model.to_string(t) for t in model.to_tokens(clean_prompt)[0]]

print(f"{'Head':<10} | {'Attn Weight ( \'+\' -> \'3\' )':<25}")
print("-" * 40)

for layer in range(0, 4):
    pattern = clean_cache[f"blocks.{layer}.attn.hook_pattern"][0] # [n_heads, seq, seq]
    for head in range(model.cfg.n_heads):
        attn_to_first_op = pattern[head, 2, 1].item() # pos 2 attending to pos 1
        if attn_to_first_op > 0.10:  # Filter for notable attention edges
            print(f"L{layer}H{head:<6} | {attn_to_first_op:.4f}")
```

#### Recorded output

```text
Head       | Attn Weight ( '+' -> '3' )
----------------------------------------
L0H7      | 0.3400
L0H9      | 0.1630
L0H10     | 0.1161
L0H11     | 0.1845
L1H0      | 0.2494
L1H3      | 0.1651
L1H6      | 0.1139
L1H7      | 0.1171
L1H10     | 0.4588
L2H0      | 0.4873
L2H2      | 0.9717
L2H3      | 0.1231
L2H4      | 0.1171
L2H5      | 0.2595
L2H8      | 0.3797
L2H9      | 0.4858
L3H1      | 0.1777
L3H2      | 0.6242
L3H3      | 0.5595
L3H6      | 0.8336
L3H7      | 0.5497
L3H8      | 0.2189
L3H9      | 0.2727
L3H11     | 0.1639
```


### Cell 69 — code (execution count: 45)

```python
import torch
from transformer_lens import HookedTransformer

clean_prompt = "3 + 5 ="
tokens = model.to_str_tokens(clean_prompt)
print("Token Alignment:", list(enumerate(tokens)))

_, cache = model.run_with_cache(clean_prompt)

# 2. Complete 12-Layer Stack Attention Sweep
print("\n" + "="*55)
print(f"{'Head':<8} | {'Attn Weight ( \'+\' [pos 2] -> \'3\' [pos 1] )':<35}")
print("="*55)

for layer in range(model.cfg.n_layers):
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0]
    for head in range(model.cfg.n_heads):
        attn_val = pattern[head, 2, 1].item()
        if attn_val > 0.15:
            print(f"L{layer:<2}H{head:<4} | {attn_val:.4f}")

# 3. Positional Control Check
control_prompt = "The red dog ran"
_, ctrl_cache = model.run_with_cache(control_prompt)
ctrl_pattern = ctrl_cache["blocks.2.attn.hook_pattern"][0]
l2h2_ctrl_attn = ctrl_pattern[2, 2, 1].item()

print("\n" + "="*55)
print(f"L2H2 Positional Control Attn (' red' -> 'The'): {l2h2_ctrl_attn:.4f}")

# 4. Compute Head Outputs directly via hook_z @ W_O (Bulletproof against KeyError)
# cache["blocks.2.attn.hook_z"] shape: [batch, seq, n_heads, d_head]
# model.W_O[2, 2] shape: [d_head, d_model]
z_pos2 = cache["blocks.2.attn.hook_z"][0, 2, 2]
l2h2_out_pos2 = z_pos2 @ model.W_O[2, 2]

z_pos4 = cache["blocks.2.attn.hook_z"][0, 4, 2]
l2h2_out_pos4 = z_pos4 @ model.W_O[2, 2]

# 5. Target Direction Setup
target_id = model.to_single_token(" 8")
foil_id = model.to_single_token(" 6")
unembed_direction = model.W_U[:, target_id] - model.W_U[:, foil_id]

# 6. Position 2 Diagnostic: Subspace Representation Projection (Logit Lens)
proj_pos2 = torch.dot(l2h2_out_pos2, unembed_direction).item()

# 7. Position 4 Measurement: True Direct Logit Attribution (LN-Corrected)
ln_scaled_pos4 = cache.apply_ln_to_stack(l2h2_out_pos4.unsqueeze(0).unsqueeze(0), layer=-1, pos_slice=-1).squeeze()
dla_pos4_corrected = torch.dot(ln_scaled_pos4, unembed_direction).item()

bias_diff = (model.b_U[target_id] - model.b_U[foil_id]).item()

print("\n" + "="*55)
print(f"L2H2 Projected Content Vector at '+' (pos 2) [Logit Lens] : {proj_pos2:+.4f}")
print(f"L2H2 Direct Logit Contribution at '=' (pos 4) [LN-Corrected] : {dla_pos4_corrected:+.4f}")
print(f"Static Unembedding Bias Offset (b_U[' 8'] - b_U[' 6'])      : {bias_diff:+.4f}")
print("="*55)
```

#### Recorded output

```text
Token Alignment: [(0, '<|endoftext|>'), (1, '3'), (2, ' +'), (3, ' 5'), (4, ' =')]

=======================================================
Head     | Attn Weight ( '+' [pos 2] -> '3' [pos 1] )
=======================================================
L0 H7    | 0.3400
L0 H9    | 0.1630
L0 H11   | 0.1845
L1 H0    | 0.2494
L1 H3    | 0.1651
L1 H10   | 0.4588
L2 H0    | 0.4873
L2 H2    | 0.9717
L2 H5    | 0.2595
L2 H8    | 0.3797
L2 H9    | 0.4858
L3 H1    | 0.1777
L3 H2    | 0.6242
L3 H3    | 0.5595
L3 H6    | 0.8336
L3 H7    | 0.5497
L3 H8    | 0.2189
L3 H9    | 0.2727
L3 H11   | 0.1639
L4 H0    | 0.1720
L4 H1    | 0.2113
L4 H3    | 0.4390
L4 H5    | 0.2734
L4 H11   | 1.0000
L5 H3    | 0.2282
L5 H4    | 0.3909
L5 H10   | 0.1618
L6 H0    | 0.2084
L6 H8    | 0.5273
L6 H11   | 0.1824
L7 H0    | 0.4044
L8 H11   | 0.1502
L11H0    | 0.1528
L11H8    | 0.4513

=======================================================
L2H2 Positional Control Attn (' red' -> 'The'): 0.7076

=======================================================
L2H2 Projected Content Vector at '+' (pos 2) [Logit Lens] : +0.3827
L2H2 Direct Logit Contribution at '=' (pos 4) [LN-Corrected] : +0.0279
Static Unembedding Bias Offset (b_U[' 8'] - b_U[' 6'])      : -0.1003
=======================================================
```


### Cell 70 — code (execution count: 46)

```python
import torch
from transformer_lens import HookedTransformer

clean_prompt = "3 + 5 ="
tokens = model.to_str_tokens(clean_prompt)
print("Token Alignment:", list(enumerate(tokens)))

_, cache = model.run_with_cache(clean_prompt)

# 2. Positional Control Check for L4H11 (' red' [pos 2] -> 'The' [pos 1])
control_prompt = "The red dog ran"
_, ctrl_cache = model.run_with_cache(control_prompt)
ctrl_pattern = ctrl_cache["blocks.4.attn.hook_pattern"][0]
l4h11_ctrl_attn = ctrl_pattern[11, 2, 1].item()  # Layer 4, Head 11

# 3. Compute L4H11 Head Outputs via hook_z @ W_O[4, 11]
z_pos2 = cache["blocks.4.attn.hook_z"][0, 2, 11]
l4h11_out_pos2 = z_pos2 @ model.W_O[4, 11]

z_pos4 = cache["blocks.4.attn.hook_z"][0, 4, 11]
l4h11_out_pos4 = z_pos4 @ model.W_O[4, 11]

# 4. Target Direction Setup (' 8' vs ' 6')
target_id = model.to_single_token(" 8")
foil_id = model.to_single_token(" 6")
unembed_direction = model.W_U[:, target_id] - model.W_U[:, foil_id]

# 5. Position 2 Diagnostic: Subspace Representation Projection (Logit Lens)
proj_pos2 = torch.dot(l4h11_out_pos2, unembed_direction).item()

# 6. Position 4 Measurement: Direct Logit Attribution (LN-Corrected)
ln_scaled_pos4 = cache.apply_ln_to_stack(l4h11_out_pos4.unsqueeze(0).unsqueeze(0), layer=-1, pos_slice=-1).squeeze()
dla_pos4_corrected = torch.dot(ln_scaled_pos4, unembed_direction).item()

bias_diff = (model.b_U[target_id] - model.b_U[foil_id]).item()

print("\n" + "="*60)
print(f"L4H11 Arithmetic Attn (' +' -> '3')                       : 1.0000")
print(f"L4H11 Positional Control Attn (' red' -> 'The')          : {l4h11_ctrl_attn:.4f}")
print(f"L4H11 Projected Content Vector at '+' (pos 2) [Logit Lens]: {proj_pos2:+.4f}")
print(f"L4H11 Direct Logit Contribution at '=' (pos 4) [LN-Corr]  : {dla_pos4_corrected:+.4f}")
print(f"Static Unembedding Bias Offset (b_U[' 8'] - b_U[' 6'])      : {bias_diff:+.4f}")
print("="*60)
```

#### Recorded output

```text
Token Alignment: [(0, '<|endoftext|>'), (1, '3'), (2, ' +'), (3, ' 5'), (4, ' =')]

============================================================
L4H11 Arithmetic Attn (' +' -> '3')                       : 1.0000
L4H11 Positional Control Attn (' red' -> 'The')          : 1.0000
L4H11 Projected Content Vector at '+' (pos 2) [Logit Lens]: +0.0085
L4H11 Direct Logit Contribution at '=' (pos 4) [LN-Corr]  : -0.0027
Static Unembedding Bias Offset (b_U[' 8'] - b_U[' 6'])      : -0.1003
============================================================
```


### Cell 71 — code (execution count: 47)

```python
import torch
from transformer_lens import HookedTransformer

model.set_use_attn_result(True)  # Populates 'blocks.L.attn.hook_result' in cache

# 2. Complete unified candidate set from Phase 4 postmortem
candidate_heads = [
    (2, 2), (4, 11),  # First tier
    (2, 9), (3, 2), (3, 3), (3, 6), (3, 7)  # Next tier
]

# Standardized 5-token prompts for 1:1 positional indexing
arithmetic_prompt = "3 + 5 ="
control_prompt = "cat dog bird ="

target_id = model.to_single_token(" 8")
foil_id = model.to_single_token(" 6")

# 3. Forward passes with activation caching
_, cache_arith = model.run_with_cache(arithmetic_prompt)
_, cache_ctrl = model.run_with_cache(control_prompt)

print("Arithmetic Tokens:", model.to_str_tokens(arithmetic_prompt))
print("Control Tokens:   ", model.to_str_tokens(control_prompt))
print("=" * 72)

# Unembedding direction vector and static bias difference
unembed_direction = model.W_U[:, target_id] - model.W_U[:, foil_id]
b_U_diff = (model.b_U[target_id] - model.b_U[foil_id]).item() if model.b_U is not None else 0.0
print(f"Static Unembed Bias Diff (b_U[' 8'] - b_U[' 6']): {b_U_diff:+.4f}\n")

# 4. Diagnostic Battery Execution
print(f"{'Head':<8} | {'LN-Corrected DLA':<18} | {'Attn (Arithmetic)':<20} | {'Attn (Control)':<18}")
print("-" * 72)

for layer, head in candidate_heads:
    # Retrieve head output vector at position '=' (Pos -1)
    head_out = cache_arith["result", layer][0, -1, head, :]  # [d_model]

    # Apply exact LN scaling using built-in API method
    ln_corrected = cache_arith.apply_ln_to_stack(
        head_out.unsqueeze(0), layer=-1, pos_slice=-1
    )
    dla_contribution = torch.dot(ln_corrected.squeeze(), unembed_direction).item()

    # Attention from '=' (Pos -1) to First Token (Pos 1)
    attn_arith = cache_arith["pattern", layer][0, head, -1, 1].item()
    attn_ctrl = cache_ctrl["pattern", layer][0, head, -1, 1].item()

    print(f"L{layer}H{head:<3} | {dla_contribution:^+18.4f} | {attn_arith:^+20.1%} | {attn_ctrl:^+18.1%}")
```

#### Recorded output

```text
Arithmetic Tokens: ['<|endoftext|>', '3', ' +', ' 5', ' =']
Control Tokens:    ['<|endoftext|>', 'cat', ' dog', ' bird', ' =']
========================================================================
Static Unembed Bias Diff (b_U[' 8'] - b_U[' 6']): -0.1003

Head     | LN-Corrected DLA   | Attn (Arithmetic)    | Attn (Control)
------------------------------------------------------------------------
L2H2   |      +0.0279       |        +3.1%         |       +1.6%
L4H11  |      -0.0027       |        +0.0%         |       +0.0%
L2H9   |      +0.0427       |        +12.2%        |       +8.5%
L3H2   |      +0.0074       |        +7.6%         |       +18.3%
L3H3   |      -0.0247       |        +2.1%         |       +2.0%
L3H6   |      +0.0139       |        +12.8%        |       +17.4%
L3H7   |      +0.0251       |        +7.0%         |       +1.1%
```


### Cell 72 — code (execution count: 48)

```python
 def ablate_l2h2_hook(value, hook):
    # value shape: [batch, pos, head, d_head]
    value[:, 2, 2, :] = 0.0  # Zero out Head 2 at position '+'
    return value

model.reset_hooks()
ablated_logits = model.run_with_hooks(
    clean_prompt,
    fwd_hooks=[("blocks.2.attn.hook_z", ablate_l2h2_hook)]
)

orig_diff = (clean_logits[0, -1, target_id] - clean_logits[0, -1, corrupt_id]).item()
ablated_diff = (ablated_logits[0, -1, target_id] - ablated_logits[0, -1, corrupt_id]).item()

print(f"Original Logit Diff (' 8' - ' 9'): {orig_diff:+.4f}")
print(f"Ablated L2H2 Logit Diff:          {ablated_diff:+.4f}")
print(f"Direct Signal Loss:                {ablated_diff - orig_diff:+.4f}")
```

#### Recorded output

```text
Original Logit Diff (' 8' - ' 9'): +0.6443
Ablated L2H2 Logit Diff:          +0.6295
Direct Signal Loss:                -0.0149
```


### Cell 73 — markdown

L2H2 was tested under the more generous (effect-inflating) ablation method and still showed only a 2.3% drop — a genuinely clean negative result, not one that needs a mean-ablation follow-up to be trustworthy, unlike L11H0 and MLP10 earlier, where the zero-ablation numbers actively misled until mean-ablation corrected them.


### Cell 74 — code (execution count: 49)

```python
import torch
from transformer_lens import HookedTransformer

prompt = "3 + 5 ="
target_id = model.to_single_token(" 8")
foil_id = model.to_single_token(" 9")

# --- Baseline (clean) run ---
clean_logits, clean_cache = model.run_with_cache(prompt)
clean_diff = (clean_logits[0, -1, target_id] - clean_logits[0, -1, foil_id]).item()
print(f"Original Logit Diff (' 8' - ' 9'): {clean_diff:+.4f}")

# --- Zero-ablate L3H7's output (hook_z) at the last position ---
LAYER, HEAD = 3, 7

def zero_ablate_head(z, hook):
    z[:, -1, HEAD, :] = 0.0
    return z

ablated_logits = model.run_with_hooks(
    prompt,
    fwd_hooks=[(f"blocks.{LAYER}.attn.hook_z", zero_ablate_head)]
)
ablated_diff = (ablated_logits[0, -1, target_id] - ablated_logits[0, -1, foil_id]).item()
signal_loss = ablated_diff - clean_diff

print(f"Ablated L3H7 Logit Diff:          {ablated_diff:+.4f}")
print(f"Direct Signal Loss:                {signal_loss:+.4f}")
```

#### Recorded output

```text
Original Logit Diff (' 8' - ' 9'): +0.6443
Ablated L3H7 Logit Diff:          +0.6718
Direct Signal Loss:                +0.0274
```


### Cell 75 — code (execution count: 50)

```python
import torch
from transformer_lens import HookedTransformer

prompt = "3 + 5 ="
target_id = model.to_single_token(" 8")
foil_id = model.to_single_token(" 9")   # matches the ablation test's pair, NOT the ' 6' pair
LAYER, HEAD = 3, 7

model.set_use_attn_result(True)
clean_logits, clean_cache = model.run_with_cache(prompt)

clean_diff = (clean_logits[0, -1, target_id] - clean_logits[0, -1, foil_id]).item()
print(f"Original Logit Diff (' 8' - ' 9'): {clean_diff:+.4f}")

unembed_direction = model.W_U[:, target_id] - model.W_U[:, foil_id]

# L3H7's output AT POSITION 4 ('=') — matches the ablation's intervention point exactly
head_out_pos4 = clean_cache["result", LAYER][0, -1, HEAD, :]

ln_corrected = clean_cache.apply_ln_to_stack(
    head_out_pos4.unsqueeze(0), layer=-1, pos_slice=-1
)
dla_pos4_89 = torch.dot(ln_corrected.squeeze(), unembed_direction).item()

print(f"L3H7 Direct Logit Contribution at '=' (pos 4), pair (' 8'-' 9'), LN-corrected: {dla_pos4_89:+.4f}")
```

#### Recorded output

```text
Original Logit Diff (' 8' - ' 9'): +0.6443
L3H7 Direct Logit Contribution at '=' (pos 4), pair (' 8'-' 9'), LN-corrected: +0.0372
```


### Cell 76 — markdown

## Candidate-Head Falsification

The original single-prompt analysis identified several apparently promising heads,
including `L2H2` and `L4H11`.

The subsequent audit tested these candidates using:

1. arithmetic versus matched control attention patterns,
2. LayerNorm-corrected DLA,
3. direct ablation at the relevant intervention position.

Most candidates did not show arithmetic-specific attention or a consistent causal
effect. `L3H7` showed a modest arithmetic-versus-control attention difference, but
its measured causal effect did not align with its direct attribution sign.

Therefore, these candidates are not treated as components of an established
addition circuit.

This analysis is retained as evidence that the initial single-prompt circuit
search produced false-positive candidates when attention, attribution, and causal
tests were considered separately.


### Cell 77 — code (execution count: 51)

```python
# Target heads identified in the attention scan
target_heads = [
    (2, 0), (2, 2), (2, 9),  # Layer 2 drivers
    (3, 2), (3, 3), (3, 6)   # Layer 3 drivers
]

def ablate_cluster_hook(value, hook):
    layer_idx = hook.layer()
    for l, h in target_heads:
        if l == layer_idx:
            value[:, 2, h, :] = 0.0  # Zero out target heads at position '+'
    return value

hooks = [(f"blocks.{l}.attn.hook_z", ablate_cluster_hook) for l in [2, 3]]

model.reset_hooks()
cluster_logits = model.run_with_hooks(clean_prompt, fwd_hooks=hooks)

cluster_diff = (cluster_logits[0, -1, target_id] - cluster_logits[0, -1, corrupt_id]).item()
print(f"Original Logit Diff (' 8' - ' 9'): {orig_diff:+.4f}")
print(f"Cluster Ablated Logit Diff:          {cluster_diff:+.4f}")
print(f"Total Cluster Signal Loss:           {cluster_diff - orig_diff:+.4f}")
```

#### Recorded output

```text
Original Logit Diff (' 8' - ' 9'): +0.6443
Cluster Ablated Logit Diff:          +0.5936
Total Cluster Signal Loss:           -0.0508
```


### Cell 78 — markdown

### Multi-Head Cluster Ablation

Ablating the tested head cluster changed the target-versus-foil logit difference
only modestly.

This result shows that the tested cluster is not necessary for preserving the full
measured output signal under this intervention.

The result alone cannot distinguish among redundancy, compensation, parallel
pathways, or an initially incorrect candidate set. No specific circuit
architecture is inferred from this experiment.


### Cell 79 — code (execution count: 52)

```python
import torch

print(f"{'Head':<10} | {'Attn Weight ( \'=\' -> \'3\' )':<25}")
print("-" * 40)

for layer in range(model.cfg.n_layers):
    pattern = clean_cache[f"blocks.{layer}.attn.hook_pattern"][0]  # [n_heads, seq, seq]
    for head in range(model.cfg.n_heads):
        # Query at pos -1 ('='), Key at pos 1 ('3')
        attn_to_first_op = pattern[head, -1, 1].item()
        if attn_to_first_op > 0.05:  # Filter for active attention edges (>5%)
            print(f"L{layer:<2}H{head:<5} | {attn_to_first_op:.4f}")
```

#### Recorded output

```text
Head       | Attn Weight ( '=' -> '3' )
----------------------------------------
L0 H0     | 0.0750
L0 H2     | 0.1209
L0 H4     | 0.0603
L0 H7     | 0.1020
L0 H8     | 0.0545
L0 H9     | 0.1238
L0 H10    | 0.1113
L0 H11    | 0.1050
L1 H0     | 0.0536
L1 H1     | 0.0558
L1 H3     | 0.0825
L1 H4     | 0.0770
L1 H6     | 0.0578
L1 H7     | 0.1109
L1 H8     | 0.0752
L1 H9     | 0.0930
L1 H10    | 0.1921
L2 H0     | 0.1789
L2 H5     | 0.0524
L2 H6     | 0.0616
L2 H8     | 0.0601
L2 H9     | 0.1219
L2 H10    | 0.0944
L3 H1     | 0.1444
L3 H2     | 0.0763
L3 H5     | 0.0925
L3 H6     | 0.1281
L3 H7     | 0.0702
L3 H8     | 0.0582
L3 H9     | 0.2218
L3 H11    | 0.1286
L4 H3     | 0.2722
L4 H5     | 0.1569
L4 H6     | 0.1587
L4 H9     | 0.0508
L5 H2     | 0.1800
L5 H3     | 0.1637
L5 H4     | 0.1286
L6 H0     | 0.1513
L6 H1     | 0.0863
L6 H5     | 0.0740
L6 H7     | 0.1668
L6 H8     | 0.0758
L6 H11    | 0.1245
L7 H3     | 0.0970
L7 H4     | 0.0640
L7 H8     | 0.0671
L7 H9     | 0.0568
L8 H5     | 0.0648
L8 H6     | 0.1514
L8 H7     | 0.0589
L8 H10    | 0.0588
L8 H11    | 0.0697
L9 H1     | 0.0626
L10H4     | 0.0672
L10H7     | 0.0653
L10H9     | 0.0555
L11H0     | 0.1949
L11H3     | 0.1476
L11H8     | 0.0837
```


### Cell 80 — markdown

 ### Direct Path Scan

The attention scan identifies heads that allocate substantial attention from the
final position toward the first operand.

This identifies candidate information-access pathways, not established causal
edges. Candidate heads are therefore subjected to intervention before any
mechanistic interpretation is assigned.


### Cell 81 — code (execution count: 53)

```python
# Target heads performing direct reads from '3' to '='
direct_heads = [
    (3, 9),   # Early-mid direct reader
    (4, 3),   # Primary mid-layer reader
    (11, 0),  # Late direct reader
    (11, 3)   # Late direct reader
]

def ablate_direct_readers(value, hook):
    layer_idx = hook.layer()
    for l, h in direct_heads:
        if l == layer_idx:
            # Zero out direct head outputs at the final token pos '='
            value[:, -1, h, :] = 0.0
    return value

hooks = [(f"blocks.{l}.attn.hook_z", ablate_direct_readers) for l in [3, 4, 11]]

model.reset_hooks()
direct_ablated_logits = model.run_with_hooks(clean_prompt, fwd_hooks=hooks)

direct_diff = (direct_ablated_logits[0, -1, target_id] - direct_ablated_logits[0, -1, corrupt_id]).item()
print(f"Original Logit Diff (' 8' - ' 9'): {orig_diff:+.4f}")
print(f"Direct Readers Ablated Diff:       {direct_diff:+.4f}")
print(f"Signal Loss:                      {direct_diff - orig_diff:+.4f}")
```

#### Recorded output

```text
Original Logit Diff (' 8' - ' 9'): +0.6443
Direct Readers Ablated Diff:       +0.6895
Signal Loss:                      +0.0452
```


### Cell 82 — markdown

### Direct Reader Ablation

Ablating the selected direct-reader heads did not reduce the tested target-versus-
foil metric; the metric increased slightly.

Within this intervention, these heads were therefore not necessary for the measured
output difference.

This does not establish that the first operand is irrelevant or that no alternative
information pathway exists.


### Cell 83 — code (execution count: 54)

```python
def block_pos1_attention(pattern, hook):
    # pattern shape: [batch, n_heads, seq_q, seq_k]
    # Zero out all attention paid to Key position 1 ('3') by any Query position
    pattern[:, :, :, 1] = 0.0
    return pattern

# Apply hook across all early layers (L0-L5)
hooks = [(f"blocks.{l}.attn.hook_pattern", block_pos1_attention) for l in range(6)]

model.reset_hooks()
blocked_logits = model.run_with_hooks(clean_prompt, fwd_hooks=hooks)

blocked_diff = (blocked_logits[0, -1, target_id] - blocked_logits[0, -1, corrupt_id]).item()
print(f"Original Logit Diff (' 8' - ' 9'): {orig_diff:+.4f}")
print(f"Pos 1 Attention Blocked Diff:     {blocked_diff:+.4f}")
print(f"Total Signal Loss:                {blocked_diff - orig_diff:+.4f}")
```

#### Recorded output

```text
Original Logit Diff (' 8' - ' 9'): +0.6443
Pos 1 Attention Blocked Diff:     +0.5449
Total Signal Loss:                -0.0994
```


### Cell 84 — markdown

### Early-Layer Position-1 Attention Blocking

Blocking attention to the first operand in early layers produced a measurable
change in the target-versus-foil metric.

This is evidence that early processing of the first-operand position can influence
the measured output.

It does not by itself establish an "early routing circuit" or identify the
specific information carried by those attention edges.


### Cell 85 — code (execution count: 55)

```python
def block_pos1_all_layers(pattern, hook):
    # Zero out all attention paid to Key position 1 ('3') by ANY layer/query
    pattern[:, :, :, 1] = 0.0
    return pattern

# Block Key position 1 across ALL 12 layers
hooks = [(f"blocks.{l}.attn.hook_pattern", block_pos1_all_layers) for l in range(model.cfg.n_layers)]

model.reset_hooks()
full_blocked_logits = model.run_with_hooks(clean_prompt, fwd_hooks=hooks)

full_blocked_diff = (full_blocked_logits[0, -1, target_id] - full_blocked_logits[0, -1, corrupt_id]).item()
print(f"Original Logit Diff (' 8' - ' 9'):     {orig_diff:+.4f}")
print(f"L0-L5 Blocked Logit Diff:              +0.5449")
print(f"All-Layer Blocked Logit Diff ('3'):    {full_blocked_diff:+.4f}")
print(f"Total Circuit Signal Loss:             {full_blocked_diff - orig_diff:+.4f}")
```

#### Recorded output

```text
Original Logit Diff (' 8' - ' 9'):     +0.6443
L0-L5 Blocked Logit Diff:              +0.5449
All-Layer Blocked Logit Diff ('3'):    +0.4753
Total Circuit Signal Loss:             -0.1690
```


### Cell 86 — code (execution count: 56)

```python
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Tuple, Literal
from transformer_lens import HookedTransformer

@dataclass
class PromptItem:
    tier: str                      # 'Easy', 'Medium', 'Hard'
    clean_prompt: str              # e.g., "3 + 5 ="
    corrupt_prompt: str            # Baseline for activation patching, e.g., "1 + 5 ="
    target_token: str              # Correct answer token
    foil_tokens: List[str]         # Symmetric contrast set
    target_id: int
    foil_ids: List[int]

class ArithmeticDatasetBuilder:
    def __init__(self, model: HookedTransformer):
        self.model = model

    def _to_id(self, tok_str: str) -> int:
        return self.model.to_single_token(tok_str)

    def build_dataset(self) -> List[PromptItem]:
        dataset: List[PromptItem] = []

        # --- TIER 1: Easy (No Carry: a + b < 10) ---
        for a in range(1, 9):
            for b in range(1, 9):
                if a + b < 10:
                    clean = f"{a} + {b} ="
                    # Parity-preserving corruption: shift by 2, wrap within 1-8
                    corrupt_a = ((a - 1 + 2) % 8) + 1
                    corrupt = f"{corrupt_a} + {b} ="
                    ans = f" {a + b}"
                    foils = [f" {a+b-1}", f" {a+b+1}"]
                    dataset.append(PromptItem(
                        tier="Easy", clean_prompt=clean, corrupt_prompt=corrupt,
                        target_token=ans, foil_tokens=foils,
                        target_id=self._to_id(ans),
                        foil_ids=[self._to_id(f) for f in foils]
                    ))

        # --- TIER 2: Medium (Single Carry: a + b >= 10) ---
        for a in range(2, 10):
            for b in range(2, 10):
                if a + b >= 10:
                    clean = f"{a} + {b} ="
                    corrupt_a = ((a - 2 + 2) % 8) + 2  # wrap within 2-9
                    corrupt = f"{corrupt_a} + {b} ="
                    ans = f" {a + b}"
                    foils = [f" {a+b-1}", f" {a+b+1}"]
                    dataset.append(PromptItem(
                        tier="Medium", clean_prompt=clean, corrupt_prompt=corrupt,
                        target_token=ans, foil_tokens=foils,
                        target_id=self._to_id(ans),
                        foil_ids=[self._to_id(f) for f in foils]
                    ))

        # --- TIER 3: Hard (Multi-Carry) ---
        # Already parity-preserving (+10), unchanged — now consistent with Easy/Medium.
        hard_pairs = [(18, 25), (28, 37), (46, 78), (39, 85), (57, 68)]
        for ab, cd in hard_pairs:
            clean = f"{ab} + {cd} ="
            corrupt = f"{ab + 10} + {cd} ="
            ans = f" {ab + cd}"
            foils = [f" {ab+cd-1}", f" {ab+cd+1}", f" {ab+cd-10}"]
            dataset.append(PromptItem(
                tier="Hard", clean_prompt=clean, corrupt_prompt=corrupt,
                target_token=ans, foil_tokens=foils,
                target_id=self._to_id(ans),
                foil_ids=[self._to_id(f) for f in foils]
            ))

        return dataset
def compute_symmetric_logit_diff(
    logits: torch.Tensor,
    target_id: int,
    foil_ids: List[int],
    pos: int = -1,
    metric_type: Literal["mean_foils", "all_single_digits"] = "mean_foils",
    model: HookedTransformer = None
) -> torch.Tensor:
    """
    Computes unbiased logit difference: Logit(target) - Mean(Logits(foils)).
    Eliminates unigram bias artifacts caused by single arbitrary foils.
    """
    target_logit = logits[0, pos, target_id]

    if metric_type == "mean_foils":
        foil_logits = logits[0, pos, foil_ids]
        baseline_logit = torch.mean(foil_logits)
    elif metric_type == "all_single_digits":
        assert model is not None, "Model reference required for all_single_digits evaluation."
        digit_tokens = [model.to_single_token(f" {d}") for d in range(10)]
        digit_tokens = [t for t in digit_tokens if t != target_id]
        baseline_logit = torch.mean(logits[0, pos, digit_tokens])
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")

    return target_logit - baseline_logit

# Reuse existing model instance if already loaded in memory
if 'model' not in locals() and 'model' not in globals():
    device = "cpu" if torch.cpu.is_available() else "cuda"
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
else:
    print("Using existing 'model' instance from session memory.")

# Initialize Dataset
builder = ArithmeticDatasetBuilder(model)
prompt_matrix = builder.build_dataset()

# Print Tier Distribution Summary
tier_counts = {t: 0 for t in ["Easy", "Medium", "Hard"]}
for item in prompt_matrix:
    tier_counts[item.tier] += 1

print(f"Dataset Initialized across {len(prompt_matrix)} total prompts.")
print(f"Tiers: Easy (No Carry) = {tier_counts['Easy']} | Medium (Single Carry) = {tier_counts['Medium']} | Hard (Multi-Carry) = {tier_counts['Hard']}")

# Sample Metric Test on First Item
sample = prompt_matrix[0]
logits, _ = model.run_with_cache(sample.clean_prompt)
sym_diff = compute_symmetric_logit_diff(
    logits, sample.target_id, sample.foil_ids, metric_type="mean_foils"
)

print(f"\nSample Prompt: '{sample.clean_prompt}' -> Target: '{sample.target_token}' | Foils: {sample.foil_tokens}")
print(f"Symmetric Logit Difference: {sym_diff.item():+.4f}")
```

#### Recorded output

```text
Using existing 'model' instance from session memory.
Dataset Initialized across 84 total prompts.
Tiers: Easy (No Carry) = 36 | Medium (Single Carry) = 43 | Hard (Multi-Carry) = 5

Sample Prompt: '1 + 1 =' -> Target: ' 2' | Foils: [' 1', ' 3']
Symmetric Logit Difference: +0.2485
```


### Cell 87 — markdown

The starting point. You'd already built a big test set: 84 different addition problems ("3 + 5 =", "18 + 25 =", etc.), split into three difficulty groups — Easy (no carrying, like 3+5), Medium (needs carrying, like 7+8), and Hard (two-digit numbers). The idea was to check whether things you found on one example ("3 + 5 =") actually held up across many examples, or were just a fluke of that one prompt.


### Cell 88 — code (execution count: 57)

```python
import torch

def evaluate_dataset_baseline(model, prompt_matrix, metric_type="mean_foils"):
    tier_results = {"Easy": [], "Medium": [], "Hard": []}

    model.reset_hooks()

    with torch.no_grad():
        for item in prompt_matrix:
            logits, _ = model.run_with_cache(item.clean_prompt)
            diff = compute_symmetric_logit_diff(
                logits,
                target_id=item.target_id,
                foil_ids=item.foil_ids,
                metric_type=metric_type,
                model=model
            )
            tier_results[item.tier].append(diff.item())

    print(f"{'Tier':<10} | {'Count':<6} | {'Mean Logit Diff':<18} | {'Std Dev':<10}")
    print("-" * 52)

    for tier, values in tier_results.items():
        val_tensor = torch.tensor(values)
        mean_val = val_tensor.mean().item()
        std_val = val_tensor.std().item() if len(values) > 1 else 0.0
        print(f"{tier:<10} | {len(values):<6} | {mean_val:^+18.4f} | {std_val:<10.4f}")

    return tier_results

# Execute Dataset Baseline
baseline_metrics = evaluate_dataset_baseline(model, prompt_matrix)
```

#### Recorded output

```text
Tier       | Count  | Mean Logit Diff    | Std Dev
----------------------------------------------------
Easy       | 36     |      +0.0018       | 0.3174
Medium     | 43     |      +0.0083       | 0.2542
Hard       | 5      |      -0.1810       | 0.2182
```


### Cell 89 — code (execution count: 58)

```python
import torch

def run_factorial_patching(model, prompt_matrix):
    # Track mean metric recovery by tier and position
    results = {
        tier: {"pos_op": [], "pos_b": []}
        for tier in ["Easy", "Medium", "Hard"]
    }

    for item in prompt_matrix:
        clean_logits, clean_cache = model.run_with_cache(item.clean_prompt)

        # Tokenize positions dynamically
        clean_tokens = model.to_str_tokens(item.clean_prompt)
        # Find index of '+' and position 'b' (penultimate prompt token before '=')
        op_idx = clean_tokens.index("+") if "+" in clean_tokens else 1
        b_idx = len(clean_tokens) - 2

        # 1. Patch Operator Position '+' across Layers 0-3
        def patch_op_hook(corrupt_act, hook):
            corrupt_act[:, op_idx, :] = clean_cache[hook.name][:, op_idx, :]
            return corrupt_act

        # 2. Patch Second Operand Position 'b' across Layers 0-3
        def patch_b_hook(corrupt_act, hook):
            corrupt_act[:, b_idx, :] = clean_cache[hook.name][:, b_idx, :]
            return corrupt_act

        with torch.no_grad():
            # Run patch at '+'
            op_hooks = [(f"blocks.{l}.hook_resid_post", patch_op_hook) for l in range(4)]
            op_logits = model.run_with_hooks(item.corrupt_prompt, fwd_hooks=op_hooks)
            op_diff = compute_symmetric_logit_diff(op_logits, item.target_id, item.foil_ids)

            # Run patch at 'b'
            b_hooks = [(f"blocks.{l}.hook_resid_post", patch_b_hook) for l in range(4)]
            b_logits = model.run_with_hooks(item.corrupt_prompt, fwd_hooks=b_hooks)
            b_diff = compute_symmetric_logit_diff(b_logits, item.target_id, item.foil_ids)

        results[item.tier]["pos_op"].append(op_diff.item())
        results[item.tier]["pos_b"].append(b_diff.item())

    print(f"\n{'Tier':<8} | {'Patched Pos':<12} | {'Mean Logit Diff':<18} | {'Std Dev':<10}")
    print("-" * 56)
    for tier in ["Easy", "Medium", "Hard"]:
        for pos_key, pos_label in [("pos_op", "'+' (Operator)"), ("pos_b", "'b' (Operand 2)")]:
            vals = torch.tensor(results[tier][pos_key])
            print(f"{tier:<8} | {pos_label:<12} | {vals.mean().item():^+18.4f} | {vals.std().item():<10.4f}")

# Execute Factorial Patching
run_factorial_patching(model, prompt_matrix)
```

#### Recorded output

```text

Tier     | Patched Pos  | Mean Logit Diff    | Std Dev
--------------------------------------------------------
Easy     | '+' (Operator) |      -0.0006       | 0.3138
Easy     | 'b' (Operand 2) |      -0.0041       | 0.2847
Medium   | '+' (Operator) |      +0.0079       | 0.2510
Medium   | 'b' (Operand 2) |      -0.0008       | 0.2380
Hard     | '+' (Operator) |      -0.1809       | 0.2169
Hard     | 'b' (Operand 2) |      -0.1725       | 0.1691
```


### Cell 90 — code (execution count: 59)

```python
import torch
def evaluate_dataset_head_dla(model, prompt_matrix):
    head_dla_sum = torch.zeros((12, 12), device=model.cfg.device)
    head_dla_sq_sum = torch.zeros((12, 12), device=model.cfg.device)
    N = len(prompt_matrix)

    model.reset_hooks()
    with torch.no_grad():
        for item in prompt_matrix:
            _, cache = model.run_with_cache(item.clean_prompt)

            head_outs = torch.stack([
                torch.einsum(
                    "hd,hdm->hm",
                    cache[f"blocks.{l}.attn.hook_z"][0, -1, :, :],
                    model.blocks[l].attn.W_O,
                )
                for l in range(12)
            ])  # [12 layers, 12 heads, d_model]

            # Apply THIS item's own LN scale before projecting — critical fix
            ln_corrected = cache.apply_ln_to_stack(
                head_outs.reshape(144, -1).unsqueeze(0), layer=-1, pos_slice=-1
            ).reshape(12, 12, -1)

            target_u = model.W_U[:, item.target_id]
            foil_u = torch.mean(model.W_U[:, item.foil_ids], dim=-1)
            direction = target_u - foil_u

            dla = torch.einsum("lhd,d->lh", ln_corrected, direction)

            head_dla_sum += dla
            head_dla_sq_sum += dla ** 2

    mean_dla = head_dla_sum / N
    std_dla = torch.sqrt((head_dla_sq_sum / N) - (mean_dla ** 2))

    flat_means = mean_dla.flatten()
    top_indices = torch.topk(flat_means, k=5).indices

    print(f"{'Head':<10} | {'Mean Dataset DLA':<18} | {'Std Dev':<10}")
    print("-" * 46)
    for idx in top_indices:
        l, h = idx.item() // 12, idx.item() % 12
        print(f"L{l}H{h:<7} | {mean_dla[l,h].item():+18.4f} | {std_dla[l,h].item():<10.4f}")

    return mean_dla, std_dla

# Execute Dataset-Wide Head DLA
mean_dla, std_dla = evaluate_dataset_head_dla(model, prompt_matrix)
```

#### Recorded output

```text
Head       | Mean Dataset DLA   | Std Dev
----------------------------------------------
L9H1       |            +0.0064 | 0.0279
L10H7       |            +0.0043 | 0.0332
L5H8       |            +0.0030 | 0.0238
L6H8       |            +0.0025 | 0.0202
L8H8       |            +0.0020 | 0.0414
```


### Cell 91 — code (execution count: 60)

```python
import torch, gc

def evaluate_dataset_pos1_ablation(model, prompt_matrix):
    clean_diffs = []
    blocked_diffs = []
    paired_deltas = []

    with torch.no_grad():  # stops autograd graph retention
        for item in prompt_matrix:
            model.reset_hooks()

            # Only need logits here — no cache required at all, so don't build one
            tokens = model.to_tokens(item.clean_prompt)
            clean_logits = model(tokens)  # plain forward pass, no cache
            clean_diff = compute_symmetric_logit_diff(
                clean_logits, item.target_id, item.foil_ids
            ).item()
            clean_diffs.append(clean_diff)

            def block_pos1_pattern(pattern, hook):
                pattern[:, :, :, 1] = 0.0
                pattern[:, :, :, :] = pattern / pattern.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                return pattern

            hooks = [(f"blocks.{l}.attn.hook_pattern", block_pos1_pattern) for l in range(model.cfg.n_layers)]
            blocked_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)  # also no cache — fine, hooks don't need it
            blocked_diff = compute_symmetric_logit_diff(
                blocked_logits, item.target_id, item.foil_ids
            ).item()
            blocked_diffs.append(blocked_diff)
            paired_deltas.append(clean_diff - blocked_diff)

            model.reset_hooks()
            del tokens, clean_logits, blocked_logits
            gc.collect()

    clean_tensor = torch.tensor(clean_diffs)
    blocked_tensor = torch.tensor(blocked_diffs)
    paired_tensor = torch.tensor(paired_deltas)

    print(f"{'Condition':<32} | {'Mean Logit Diff':<18} | {'Std Dev':<10}")
    print("-" * 66)
    print(f"{'Clean Baseline':<32} | {clean_tensor.mean().item():+18.4f} | {clean_tensor.std().item():<10.4f}")
    print(f"{'Pos 1 Blocked (renormalized)':<32} | {blocked_tensor.mean().item():+18.4f} | {blocked_tensor.std().item():<10.4f}")
    print(f"\nPaired per-item delta (clean - blocked): mean={paired_tensor.mean().item():+.4f}, "
          f"SD={paired_tensor.std().item():.4f}, "
          f"SE={paired_tensor.std().item()/len(paired_tensor)**0.5:.4f}")

    return clean_tensor, blocked_tensor, paired_tensor

clean_tensor, blocked_tensor, paired_tensor = evaluate_dataset_pos1_ablation(model, prompt_matrix)
```

#### Recorded output

```text
Condition                        | Mean Logit Diff    | Std Dev
------------------------------------------------------------------
Clean Baseline                   |            -0.0058 | 0.2819
Pos 1 Blocked (renormalized)     |            -0.0457 | 0.3492

Paired per-item delta (clean - blocked): mean=+0.0399, SD=0.2078, SE=0.0227
```


### Cell 92 — markdown

### Position-1 ablation — overall result

The overall paired test gives p ≈ 0.08 (marginal). The tier-stratified
analysis below shows the effect is concentrated in the **Easy**
tier: p = .0116. Medium and Hard tiers are not significant.


### Cell 93 — code (execution count: 61)

```python
import torch
from scipy import stats

def tier_stratified_pos1_analysis(prompt_matrix, paired_deltas):
    """
    paired_deltas: list/tensor of per-item (clean - blocked) deltas,
    in the same order as prompt_matrix.
    """
    tier_deltas = {"Easy": [], "Medium": [], "Hard": []}

    for item, delta in zip(prompt_matrix, paired_deltas):
        tier_deltas[item.tier].append(float(delta))

    print(f"{'Tier':<8} | {'N':<4} | {'Mean Delta':<12} | {'Median':<10} | {'SE':<8} | {'t':<6} | {'p':<8} | {'Pos/N (%)':<12} | {'Sign p':<8}")
    print("-" * 90)

    for tier, vals in tier_deltas.items():
        n = len(vals)
        if n < 2:
            print(f"{tier:<8} | {n:<4} | (too few for stats)")
            continue

        t = torch.tensor(vals)
        mean_d = t.mean().item()
        median_d = t.median().item()
        se = t.std().item() / (n ** 0.5)
        t_stat, p_val = stats.ttest_1samp(vals, 0.0)

        pos_count = sum(1 for v in vals if v > 0)
        sign_p = stats.binomtest(pos_count, n, 0.5).pvalue

        print(f"{tier:<8} | {n:<4} | {mean_d:+12.4f} | {median_d:+10.4f} | {se:<8.4f} | "
              f"{t_stat:<6.2f} | {p_val:<8.4f} | {pos_count}/{n} ({100*pos_count/n:.1f}%) | {sign_p:<8.4f}")

    return tier_deltas

# Run using the already-computed paired_tensor from the previous cell
tier_deltas = tier_stratified_pos1_analysis(prompt_matrix, paired_tensor)
```

#### Recorded output

```text
Tier     | N    | Mean Delta   | Median     | SE       | t      | p        | Pos/N (%)    | Sign p
------------------------------------------------------------------------------------------
Easy     | 36   |      +0.0852 |    +0.0723 | 0.0320   | 2.66   | 0.0116   | 24/36 (66.7%) | 0.0652
Medium   | 43   |      -0.0114 |    -0.0165 | 0.0320   | -0.36  | 0.7240   | 19/43 (44.2%) | 0.5424
Hard     | 5    |      +0.1540 |    +0.0690 | 0.0978   | 1.58   | 0.1902   | 3/5 (60.0%) | 1.0000
```


### Cell 94 — markdown

### Tier-Stratified Interpretation

The position-1 blocking effect differs across difficulty tiers in the present
dataset, with the strongest evidence occurring in the Easy tier.

The Medium tier does not show the same effect, while the Hard tier contains only a
small number of examples.

These results motivate a hypothesis that arithmetic difficulty may interact with
information routing, but they do not establish that the model uses completely
different computational strategies.

A formal interaction test on a larger balanced dataset would be required to support
that stronger claim..


### Cell 95 — code (execution count: 62)

```python
# Quick distribution & non-parametric check
positive_shifts = (paired_tensor > 0).sum().item()
total_items = len(paired_tensor)
median_delta = paired_tensor.median().item()

print(f"Positive Deltas (Clean > Blocked): {positive_shifts}/{total_items} ({positive_shifts/total_items*100:.1f}%)")
print(f"Median Delta: {median_delta:+.4f}")
```

#### Recorded output

```text
Positive Deltas (Clean > Blocked): 46/84 (54.8%)
Median Delta: +0.0212
```


### Cell 96 — code (execution count: 63)

```python
import torch, gc
from typing import Tuple

def evaluate_dataset_head_dla(model: HookedTransformer, dataset: list) -> Tuple[torch.Tensor, torch.Tensor]:
    head_dla_sum = torch.zeros((12, 12))
    head_dla_sq_sum = torch.zeros((12, 12))
    N = len(dataset)

    z_hooks = [f"blocks.{l}.attn.hook_z" for l in range(12)]
    keep = set(z_hooks + ["ln_final.hook_scale"])

    with torch.no_grad():
        for item in dataset:
            tokens = model.to_tokens(item.clean_prompt)
            _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in keep)

            ln_scale = cache["ln_final.hook_scale"][0, -1, 0]

            head_outs = torch.stack([
                cache[f"blocks.{l}.attn.hook_z"][0, -1, :, :] @ model.blocks[l].attn.W_O
                for l in range(12)
            ])  # [12, 12, d_model]

            head_resid_ln = head_outs / ln_scale

            target_u = model.W_U[:, item.target_id]
            foil_u = torch.mean(model.W_U[:, item.foil_ids], dim=-1)
            direction = target_u - foil_u

            dla = torch.einsum("lhd,d->lh", head_resid_ln, direction)
            head_dla_sum += dla
            head_dla_sq_sum += dla ** 2

            del cache, tokens
            gc.collect()

    mean_dla = head_dla_sum / N
    std_dla = torch.sqrt((head_dla_sq_sum / N) - (mean_dla ** 2))

    flat_means = mean_dla.flatten()
    top_indices = torch.topk(flat_means, k=5).indices
    print(f"{'Head':<10} | {'Mean Dataset DLA':<18} | {'Std Dev':<10}")
    print("-" * 46)
    for idx in top_indices:
        l, h = idx.item() // 12, idx.item() % 12
        print(f"L{l}H{h:<7} | {mean_dla[l,h].item():+18.4f} | {std_dla[l,h].item():<10.4f}")

    return mean_dla, std_dla
```


### Cell 97 — code (execution count: 64)

```python
import torch
import pandas as pd
import numpy as np

def analyze_per_operand_breakdown(model, prompt_matrix):
    records = []

    model.reset_hooks()

    with torch.no_grad():
        for item in prompt_matrix:
            # Parse operands from clean prompt (e.g., "3 + 5 =")
            tokens_str = item.clean_prompt.split()
            a_val = tokens_str[0]
            b_val = tokens_str[2]

            # 1. Baseline Logit Difference
            clean_logits, clean_cache = model.run_with_cache(item.clean_prompt)
            base_diff = compute_symmetric_logit_diff(
                clean_logits, item.target_id, item.foil_ids
            ).item()

            # 2. Operator '+' Patching across Layers 0-3
            clean_tokens = model.to_str_tokens(item.clean_prompt)
            op_idx = clean_tokens.index(" +") if " +" in clean_tokens else 1

            def patch_op_hook(corrupt_act, hook):
                corrupt_act[:, op_idx, :] = clean_cache[hook.name][:, op_idx, :]
                return corrupt_act

            op_hooks = [(f"blocks.{l}.hook_resid_post", patch_op_hook) for l in range(4)]
            patched_logits = model.run_with_hooks(item.corrupt_prompt, fwd_hooks=op_hooks)
            patched_diff = compute_symmetric_logit_diff(
                patched_logits, item.target_id, item.foil_ids
            ).item()

            # Patching Delta (Recovery above baseline)
            patch_delta = patched_diff - base_diff

            records.append({
                "Tier": item.tier,
                "a": a_val,
                "b": b_val,
                "Prompt": item.clean_prompt,
                "Target": item.target_token.strip(),
                "Baseline_Diff": base_diff,
                "Patched_Diff": patched_diff,
                "Patch_Delta": patch_delta
            })

    df = pd.DataFrame(records)

    # Display full itemized breakdown sorted by Baseline Logit Diff descending
    print("=== PER-OPERAND-PAIR BREAKDOWN (Top 15 & Bottom 15 Prompts) ===")
    df_sorted = df.sort_values(by="Baseline_Diff", ascending=False)
    print("\n--- Top 10 Highest Baseline Performers ---")
    print(df_sorted[['Tier', 'Prompt', 'Target', 'Baseline_Diff', 'Patch_Delta']].head(10).to_string(index=False))

    print("\n--- Bottom 10 Lowest Baseline Performers ---")
    print(df_sorted[['Tier', 'Prompt', 'Target', 'Baseline_Diff', 'Patch_Delta']].tail(10).to_string(index=False))

    # Create 2D Matrix Grid for Single-Digit Prompts (Easy & Medium)
    df_single = df[df['Tier'].isin(['Easy', 'Medium'])].copy()
    df_single['a'] = df_single['a'].astype(int)
    df_single['b'] = df_single['b'].astype(int)

    matrix_base = df_single.pivot(index='a', columns='b', values='Baseline_Diff')
    matrix_delta = df_single.pivot(index='a', columns='b', values='Patch_Delta')

    print("\n=== BASELINE LOGIT DIFF MATRIX (Rows: Operand 'a', Cols: Operand 'b') ===")
    print(matrix_base.round(3).fillna("-"))

    print("\n=== OPERATOR PATCHING DELTA MATRIX (Rows: Operand 'a', Cols: Operand 'b') ===")
    print(matrix_delta.round(3).fillna("-"))

    # Specific Subgroup Hypothesis Testing
    print("\n=== SUBGROUP HYPOTHESIS TESTS ===")

    # 1. Identical Operands (a = b) vs Non-Identical (a != b)
    doubles = df_single[df_single['a'] == df_single['b']]
    non_doubles = df_single[df_single['a'] != df_single['b']]
    print(f"Doubles (a=a) Mean Baseline : {doubles['Baseline_Diff'].mean():+.4f} (n={len(doubles)})")
    print(f"Non-Doubles Mean Baseline   : {non_doubles['Baseline_Diff'].mean():+.4f} (n={len(non_doubles)})")

    # 2. Target = 10 (Base-10 Completion Hotspot)
    target_10 = df_single[df_single['Target'] == '10']
    print(f"Target = 10 Mean Baseline   : {target_10['Baseline_Diff'].mean():+.4f} (n={len(target_10)})")

    return df

# Execute Analysis
df_per_pair = analyze_per_operand_breakdown(model, prompt_matrix)
```

#### Recorded output

```text
=== PER-OPERAND-PAIR BREAKDOWN (Top 15 & Bottom 15 Prompts) ===

--- Top 10 Highest Baseline Performers ---
  Tier  Prompt Target  Baseline_Diff  Patch_Delta
Medium 8 + 8 =     16       0.719002    -0.141879
  Easy 1 + 2 =      3       0.676641    -0.613883
Medium 5 + 5 =     10       0.647227    -0.358732
  Easy 4 + 4 =      8       0.605705    -0.214654
  Easy 2 + 6 =      8       0.428913     0.021371
  Easy 3 + 5 =      8       0.411659    -0.038223
Medium 6 + 6 =     12       0.401203    -0.079903
  Easy 2 + 4 =      6       0.366152    -0.018232
Medium 8 + 4 =     12       0.362541     0.063424
Medium 4 + 8 =     12       0.358138     0.004354

--- Bottom 10 Lowest Baseline Performers ---
  Tier    Prompt Target  Baseline_Diff  Patch_Delta
  Easy   1 + 6 =      7      -0.327970     0.011306
Medium   8 + 9 =     17      -0.329903     0.119144
  Easy   2 + 5 =      7      -0.330213    -0.075989
  Easy   2 + 3 =      5      -0.330858     0.086849
  Easy   4 + 3 =      7      -0.333618     0.034485
Medium   3 + 8 =     11      -0.345842    -0.032701
Medium   5 + 6 =     11      -0.389825     0.205171
  Hard 18 + 25 =     43      -0.395320     0.011059
  Easy   4 + 5 =      9      -0.437211     0.234550
  Easy   3 + 4 =      7      -0.495322     0.167361

=== BASELINE LOGIT DIFF MATRIX (Rows: Operand 'a', Cols: Operand 'b') ===
b      1      2      3      4      5      6      7      8      9
a
1  0.248  0.677  0.321 -0.250  0.161 -0.328  0.263 -0.286      -
2 -0.156  0.251 -0.331  0.366 -0.330  0.429 -0.193  0.223 -0.234
3  0.084 -0.195  0.230 -0.495  0.412 -0.256  0.093 -0.346  0.188
4 -0.167  0.263 -0.334  0.606 -0.437  0.110 -0.178  0.358 -0.167
5  0.063 -0.298  0.304 -0.210  0.647 -0.390  0.222 -0.162 -0.297
6 -0.318  0.275 -0.102  0.036 -0.232  0.401 -0.164 -0.179  0.113
7  0.207 -0.140  0.090 -0.196  0.162 -0.057 -0.238  0.158  0.081
8 -0.269  0.154 -0.292  0.363 -0.110 -0.214  0.144  0.719  -0.33
9      - -0.120  0.126 -0.035 -0.238  0.153 -0.008 -0.052  0.052

=== OPERATOR PATCHING DELTA MATRIX (Rows: Operand 'a', Cols: Operand 'b') ===
b      1      2      3      4      5      6      7      8      9
a
1 -0.222 -0.614  0.001  0.025  0.029  0.011  0.036 -0.001      -
2 -0.071  0.037  0.087 -0.018 -0.076  0.021  0.006 -0.053  0.015
3 -0.156  0.206 -0.060  0.167 -0.038 -0.062  0.107 -0.033  0.011
4 -0.025  0.114  0.034 -0.215  0.235  0.012 -0.052  0.004  0.049
5   -0.0  0.202 -0.055  0.050 -0.359  0.205 -0.123  0.027   0.04
6 -0.079  0.209 -0.044  0.089 -0.110 -0.080  0.050  0.014  0.043
7  0.077 -0.043 -0.132  0.059 -0.032  0.006  0.066 -0.044  0.037
8 -0.002  0.053 -0.061  0.063 -0.038  0.046 -0.009 -0.142  0.119
9      - -0.015  0.055 -0.169 -0.110  0.017  0.104 -0.213  0.061

=== SUBGROUP HYPOTHESIS TESTS ===
Doubles (a=a) Mean Baseline : +0.3241 (n=9)
Non-Doubles Mean Baseline   : -0.0357 (n=70)
Target = 10 Mean Baseline   : +0.1933 (n=7)
```


### Cell 98 — code (execution count: 65)

```python
from scipy import stats

def tier_stratified_significance(df, delta_col="Patch_Delta"):
    print(f"{'Tier':<8} | {'N':<4} | {'Mean Delta':<12} | {'Median':<10} | {'SE':<8} | {'t':<6} | {'p':<8} | {'Pos/N (%)':<14} | {'Sign p':<8}")
    print("-" * 95)

    for tier in ["Easy", "Medium", "Hard"]:
        vals = df.loc[df["Tier"] == tier, delta_col].dropna().values
        n = len(vals)
        if n < 2:
            print(f"{tier:<8} | {n:<4} | (too few for stats)")
            continue

        mean_d = vals.mean()
        median_d = float(np.median(vals))
        se = vals.std(ddof=1) / (n ** 0.5)
        t_stat, p_val = stats.ttest_1samp(vals, 0.0)

        pos_count = int((vals > 0).sum())
        sign_p = stats.binomtest(pos_count, n, 0.5).pvalue

        print(f"{tier:<8} | {n:<4} | {mean_d:+12.4f} | {median_d:+10.4f} | {se:<8.4f} | "
              f"{t_stat:<6.2f} | {p_val:<8.4f} | {pos_count}/{n} ({100*pos_count/n:.1f}%){'':<3} | {sign_p:<8.4f}")

# Run on the operator-patching delta column from your already-corrected data
tier_stratified_significance(df_per_pair, delta_col="Patch_Delta")
```

#### Recorded output

```text
Tier     | N    | Mean Delta   | Median     | SE       | t      | p        | Pos/N (%)      | Sign p
-----------------------------------------------------------------------------------------------
Easy     | 36   |      -0.0065 |    +0.0002 | 0.0248   | -0.26  | 0.7942   | 18/36 (50.0%)    | 1.0000
Medium   | 43   |      -0.0098 |    +0.0115 | 0.0152   | -0.65  | 0.5220   | 25/43 (58.1%)    | 0.3604
Hard     | 5    |      +0.0022 |    +0.0111 | 0.0461   | 0.05   | 0.9649   | 3/5 (60.0%)    | 1.0000
```


### Cell 99 — code (execution count: 66)

```python
import torch
import pandas as pd
from typing import List, Dict, Tuple
from transformer_lens import HookedTransformer



# 2. Prompt Dataset Generator (N=36 Single-Digit Addition Cohort)
def generate_addition_dataset() -> Tuple[List[str], List[str], List[str]]:
    prompts, targets, foils = [], [], []
    for a in range(1, 10):
        for b in range(1, 10):
            ans = a + b
            if ans < 10:  # Single-digit targets
                prompts.append(f"{a} + {b} =")
                targets.append(f" {ans}")
                foil_ans = ans + 2 if ans <= 7 else ans - 2
                foils.append(f" {foil_ans}")
    return prompts, targets, foils

# 3. Filtering High-Accuracy Subpopulation (D_acc)
def run_subpopulation_filtering(
    prompts: List[str],
    targets: List[str],
    foils: List[str],
    tau: float = 1.0
) -> Tuple[List[str], List[str], List[str], torch.Tensor]:
    valid_prompts, valid_targets, valid_foils, valid_diffs = [], [], [], []

    for p, t, f in zip(prompts, targets, foils):
        t_id = model.to_single_token(t)
        f_id = model.to_single_token(f)

        with torch.no_grad():
            logits = model(p)[:, -1, :]
            top1_id = torch.argmax(logits, dim=-1).item()
            diff = (logits[0, t_id] - logits[0, f_id]).item()

        if top1_id == t_id and diff >= tau:
            valid_prompts.append(p)
            valid_targets.append(t)
            valid_foils.append(f)
            valid_diffs.append(diff)

    return valid_prompts, valid_targets, valid_foils, torch.tensor(valid_diffs)

# 4. Corrected LayerNorm-Scaled DLA Decomposition
def run_dla_analysis(
    prompts: List[str],
    targets: List[str],
    foils: List[str]
) -> pd.DataFrame:
    dla_records = []

    for p, t, f in zip(prompts, targets, foils):
        t_id = model.to_single_token(t)
        f_id = model.to_single_token(f)
        tokens = model.to_tokens(p)

        with torch.no_grad():
            # Include hook_resid_pre and hook_attn_out in names_filter
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: "hook_resid_pre" in n or "hook_attn_out" in n or "hook_mlp_out" in n or "ln_final" in n
            )
            ground_truth_diff = (logits[0, -1, t_id] - logits[0, -1, f_id]).item()

            # Extract 4D scale tensor and slice at position -1
            scale = cache["ln_final.hook_scale"][:, -1, :] # [batch, pos, 1]
            W_U_diff = model.W_U[:, t_id] - model.W_U[:, f_id]
            b_U_diff = (model.b_U[t_id] - model.b_U[f_id]).item()

            row = {"prompt": p, "ground_truth_diff": ground_truth_diff, "b_U_diff": b_U_diff}

            # A. Initial Embeddings (Token + Position)
            embed_contrib = cache["blocks.0.hook_resid_pre"][:, -1, :]
            scaled_embed = embed_contrib / scale[0]
            row["embeddings"] = torch.dot(scaled_embed.squeeze(0), W_U_diff).item()

            # B. Layer-wise Attention & MLP outputs
            for l in range(model.cfg.n_layers):
                attn_out = cache[f"blocks.{l}.hook_attn_out"][:, -1, :]
                mlp_out = cache[f"blocks.{l}.hook_mlp_out"][:, -1, :]

                row[f"L{l}_attn"] = torch.dot((attn_out / scale[0]).squeeze(0), W_U_diff).item()
                row[f"L{l}_mlp"] = torch.dot((mlp_out / scale[0]).squeeze(0), W_U_diff).item()

            # Full residual stream sum check
            accumulated_dla = row["embeddings"] + sum(
                row[f"L{l}_attn"] + row[f"L{l}_mlp"] for l in range(model.cfg.n_layers)
            )
            reconstructed = accumulated_dla + b_U_diff

            # Verified equivalence check (|sum - ground_truth| < 1e-3)
            diff_error = abs(reconstructed - ground_truth_diff)
            assert diff_error < 1e-3, f"Mismatch: reconstructed {reconstructed:.4f} vs GT {ground_truth_diff:.4f}"

            dla_records.append(row)

    return pd.DataFrame(dla_records)

# 5. Pipeline Execution
prompts, targets, foils = generate_addition_dataset()
v_prompts, v_targets, v_foils, v_diffs = run_subpopulation_filtering(prompts, targets, foils, tau=1.0)

print(f"=== SUBPOPULATION FILTERING RESULTS ===")
print(f"Total Evaluated Prompts: {len(prompts)}")
print(f"High-Accuracy Subset (D_acc) Count: {len(v_prompts)}")

if len(v_prompts) > 0:
    print(f"Mean Logit Difference (D_acc): {v_diffs.mean().item():.4f}\n")
    df_dla = run_dla_analysis(v_prompts, v_targets, v_foils)

    print(f"=== COMPLETE DLA ATTRIBUTION BREAKDOWN ON D_acc (Mean Values) ===")
    mean_dla = df_dla.drop(columns=["prompt"]).mean()
    print(mean_dla.to_string())
else:
    print("No prompts met the threshold tau >= 1.0.")
```

#### Recorded output

```text
=== SUBPOPULATION FILTERING RESULTS ===
Total Evaluated Prompts: 36
High-Accuracy Subset (D_acc) Count: 1
Mean Logit Difference (D_acc): 1.5536

=== COMPLETE DLA ATTRIBUTION BREAKDOWN ON D_acc (Mean Values) ===
ground_truth_diff    1.553580
b_U_diff             0.360159
embeddings           0.014525
L0_attn              0.056416
L0_mlp               0.057748
L1_attn              0.041161
L1_mlp              -0.050510
L2_attn             -0.039748
L2_mlp              -0.014546
L3_attn              0.045852
L3_mlp               0.044943
L4_attn              0.014951
L4_mlp               0.021308
L5_attn              0.161524
L5_mlp              -0.057230
L6_attn              0.092429
L6_mlp              -0.067750
L7_attn              0.066679
L7_mlp              -0.066070
L8_attn              0.108864
L8_mlp              -0.031087
L9_attn              0.367650
L9_mlp               0.253661
L10_attn             0.131179
L10_mlp              0.148027
L11_attn            -0.005161
L11_mlp             -0.101396
```


### Cell 100 — markdown

### D_acc Statistical Scope

The high-accuracy filtering procedure leaves only one zero-shot example in the
current cohort.

The resulting DLA decomposition is therefore useful as a case study of what the
filter selects, but it is not sufficient for population-level mechanistic
inference.

No circuit-level conclusion is drawn from the `N=1` subset.


### Cell 101 — code (execution count: 67)

```python
# ── COHORT A: Earlier few-shot cohort (5/33 result) ──
# PREFIX excludes (1,1), (2,3), (4,1). n=33 after single-sum<10 filter.
import torch, pandas as pd

PREFIX  = "1 + 1 = 2\n2 + 3 = 5\n4 + 1 = 5\n"
EXCLUDE = {(1, 1), (2, 3), (4, 1)}          # prefix examples are never evaluated

def build(fewshot):
    rows = []
    for a in range(1, 10):
        for b in range(1, 10):
            s = a + b
            if s >= 10 or (fewshot and (a, b) in EXCLUDE):
                continue
            foil = s + 2 if s <= 7 else s - 2
            rows.append(dict(a=a, b=b, t=f" {s}", f=f" {foil}",
                             prompt=(PREFIX if fewshot else "") + f"{a} + {b} ="))
    return rows

def screen(rows):
    out = []
    with torch.no_grad():
        for r in rows:
            lg = model(r["prompt"])[0, -1]
            t, f = model.to_single_token(r["t"]), model.to_single_token(r["f"])
            out.append({**r,
                        "rank": (lg > lg[t]).sum().item() + 1,
                        "diff": (lg[t] - lg[f]).item()})
    df = pd.DataFrame(out)
    df["top1"], df["top3"] = df["rank"] == 1, df["rank"] <= 3
    return df

for name, fs in [("zero-shot", False), ("few-shot", True)]:
    df = screen(build(fs))
    strict = {tau: int((df.top1 & (df["diff"] >= tau)).sum()) for tau in (0.0, 0.5, 1.0)}
    print(f"{name}: n={len(df)} | top1={int(df.top1.sum())} | top3={int(df.top3.sum())} | strict(tau)={strict}")
    print(df[df.top1][["a", "b", "t", "diff"]].to_string(index=False), "\n")
```

#### Recorded output

```text
zero-shot: n=36 | top1=2 | top3=5 | strict(tau)={0.0: 2, 0.5: 2, 1.0: 1}
 a  b  t     diff
 1  2  3 1.553580
 1  3  4 0.731577

few-shot: n=33 | top1=5 | top3=15 | strict(tau)={0.0: 5, 0.5: 5, 1.0: 5}
 a  b  t     diff
 1  5  6 1.036507
 1  6  7 1.702593
 2  4  6 1.517866
 3  3  6 1.537722
 4  2  6 1.472296
```


### Cell 102 — code (execution count: 68)

```python
# ── COHORT B: Held-out few-shot cohort (current result) ──
# Excludes prefix pairs + twin swap. n=32.
import numpy as np, torch

EXCLUDED = {(1, 1), (2, 2), (5, 3), (3, 5)}          # prefix pairs + twin
FEWSHOT  = "1 + 1 = 2\n2 + 2 = 4\n5 + 3 = 8\n"

cohort = [(a, b, a + b) for a in range(1, 10) for b in range(1, 10)
          if a + b < 10 and (a, b) not in EXCLUDED]
ans  = np.array([s for _, _, s in cohort])
digs = sorted(set(ans.tolist()))                      # answer digits that occur
ids  = {d: model.to_single_token(f" {d}") for d in digs}

def run(prefix, label, n_perm=5000, seed=0):
    L, top_full = [], []
    with torch.no_grad():
        for a, b, _ in cohort:
            lg = model(prefix + f"{a} + {b} =")[0, -1]
            L.append([lg[ids[d]].item() for d in digs])
            top_full.append(model.tokenizer.decode(lg.argmax().item()).strip())
    L = np.array(L)                                   # [n_prompts, n_digits]

    def contrast(labels):
        out = []
        for j, d in enumerate(digs):
            m = labels == d
            if m.any() and (~m).any():
                out.append(L[m, j].mean() - L[~m, j].mean())
        return float(np.mean(out))

    obs  = contrast(ans)
    rng  = np.random.default_rng(seed)
    null = np.array([contrast(rng.permutation(ans)) for _ in range(n_perm)])
    p    = (1 + (null >= obs).sum()) / (1 + n_perm)

    per_digit = {d: round(L[ans == d, j].mean() - L[ans != d, j].mean(), 3)
                 for j, d in enumerate(digs) if (ans == d).any()}

    pred = np.array(digs)[L.argmax(1)]                # top-1 among answer digits
    acc_digit = (pred == ans).mean()
    counts = {d: int((ans == d).sum()) for d in digs}
    best_const = max(counts.values()) / len(ans)
    acc_full = np.mean([t == str(s) for t, s in zip(top_full, ans)])

    print(f"=== {label} (n={len(ans)}) ===")
    print(f"own-answer logit contrast: {obs:+.3f} | permutation p = {p:.4f} "
          f"(null mean {null.mean():+.3f}, sd {null.std():.3f})")
    print(f"per-digit contrast: {per_digit}")
    print(f"top-1 full vocab: {acc_full:.3f} | top-1 among digits {digs}: {acc_digit:.3f} "
          f"| chance {1/len(digs):.3f} | best constant guess {best_const:.3f}")
    print(f"answer counts: {counts}\n")

run("",      "zero-shot")
run(FEWSHOT, "few-shot")
```

#### Recorded output

```text
=== zero-shot (n=32) ===
own-answer logit contrast: +0.123 | permutation p = 0.0248 (null mean +0.000, sd 0.060)
per-digit contrast: {3: np.float64(0.206), 4: np.float64(-0.066), 5: np.float64(-0.107), 6: np.float64(-0.01), 7: np.float64(0.143), 8: np.float64(0.268), 9: np.float64(0.43)}
top-1 full vocab: 0.062 | top-1 among digits [3, 4, 5, 6, 7, 8, 9]: 0.125 | chance 0.143 | best constant guess 0.250
answer counts: {3: 2, 4: 2, 5: 4, 6: 5, 7: 6, 8: 5, 9: 8}

=== few-shot (n=32) ===
own-answer logit contrast: +0.812 | permutation p = 0.0002 (null mean -0.001, sd 0.187)
per-digit contrast: {3: np.float64(2.997), 4: np.float64(1.924), 5: np.float64(0.979), 6: np.float64(0.391), 7: np.float64(-0.071), 8: np.float64(-0.112), 9: np.float64(-0.426)}
top-1 full vocab: 0.062 | top-1 among digits [3, 4, 5, 6, 7, 8, 9]: 0.312 | chance 0.143 | best constant guess 0.250
answer counts: {3: 2, 4: 2, 5: 4, 6: 5, 7: 6, 8: 5, 9: 8}
```


### Cell 103 — code (execution count: 69)

```python
# ── COHORT C: Reconciliation — old vs current ──
# Runs same evaluation logic on both cohorts side-by-side.
import numpy as np, torch

EXCLUDED = {(1, 1), (2, 2), (5, 3), (3, 5)}
FEWSHOT  = "1 + 1 = 2\n2 + 2 = 4\n5 + 3 = 8\n"
cohort = [(a, b, a + b) for a in range(1, 10) for b in range(1, 10)
          if a + b < 10 and (a, b) not in EXCLUDED]
ans  = np.array([s for *_, s in cohort])
vals = list(range(2, 11))                         # includes both neighbours of every answer
ids  = [model.to_single_token(f" {v}") for v in vals]
col  = {v: j for j, v in enumerate(vals)}

def contrast(L, lab, weighted=False):
    out, w = [], []
    for d in range(3, 10):
        m = lab == d
        if m.any() and (~m).any():
            out.append(L[m, col[d]].mean() - L[~m, col[d]].mean()); w.append(m.sum())
    return float(np.average(out, weights=w if weighted else None))

def neighbour(L, lab):
    return float(np.mean([L[i, col[a]] - 0.5 * (L[i, col[a-1]] + L[i, col[a+1]])
                          for i, a in enumerate(lab)]))

def analyze(prefix, name, n_perm=5000, seed=0):
    texts, rows, top = [], [], []
    with torch.no_grad():
        for a, b, _ in cohort:
            p = prefix + f"{a} + {b} ="
            lg = model(p)[0, -1]
            texts.append(p); rows.append(lg[ids].cpu().numpy())
            top.append(model.tokenizer.decode(lg.argmax().item()).strip())
    L = np.array(rows)
    dig = [col[d] for d in range(3, 10)]
    Lc = L.copy(); Lc[:, dig] = L[:, dig] - L[:, dig].mean(1, keepdims=True)  # remove per-prompt level

    stats = {"raw contrast":            lambda lab: contrast(L, lab),
             "count-weighted contrast": lambda lab: contrast(L, lab, True),
             "row-centred contrast":    lambda lab: contrast(Lc, lab),
             "neighbour specificity":   lambda lab: neighbour(L, lab)}
    rng = np.random.default_rng(seed)
    print(f"=== {name} === first prompt: {texts[0]!r}")
    for k, f in stats.items():
        obs  = f(ans)
        null = np.array([f(rng.permutation(ans)) for _ in range(n_perm)])
        p = (1 + (null >= obs).sum()) / (1 + n_perm)
        print(f"{k:26s} obs={obs:+.3f}  null sd={null.std():.3f}  p={p:.4f}")
    hits = [(c[0], c[1]) for c, t in zip(cohort, top) if t == str(c[2])]
    print(f"full-vocab top-1 hits: {len(hits)} {hits}\n")

analyze("", "zero-shot")
analyze(FEWSHOT, "few-shot")
```

#### Recorded output

```text
=== zero-shot === first prompt: '1 + 2 ='
raw contrast               obs=+0.123  null sd=0.060  p=0.0248
count-weighted contrast    obs=+0.170  null sd=0.041  p=0.0004
row-centred contrast       obs=+0.224  null sd=0.054  p=0.0004
neighbour specificity      obs=-0.036  null sd=0.023  p=0.2470
full-vocab top-1 hits: 2 [(1, 2), (1, 3)]

=== few-shot === first prompt: '1 + 1 = 2\n2 + 2 = 4\n5 + 3 = 8\n1 + 2 ='
raw contrast               obs=+0.812  null sd=0.187  p=0.0002
count-weighted contrast    obs=+0.354  null sd=0.131  p=0.0024
row-centred contrast       obs=+0.521  null sd=0.140  p=0.0004
neighbour specificity      obs=-0.041  null sd=0.029  p=0.0154
full-vocab top-1 hits: 2 [(2, 1), (3, 1)]
```


### Cell 104 — code (execution count: 70)

```python
# ============================================================
# M1 — OLD vs CURRENT COHORT RECONCILIATION
# Purpose:
#   Determine exactly why the earlier result was 5/33 while the
#   current result is 2/32, using the SAME model and SAME
#   top-1 evaluation logic.
#
# IMPORTANT:
#   Do not change the model or evaluation code before running.
# ============================================================

import torch

# -----------------------------
# OLD analysis definition
# -----------------------------
OLD_EXCLUDE = {
    (1, 1),
    (2, 3),
    (4, 1),
}

OLD_PREFIX = (
    "1 + 1 = 2\n"
    "2 + 3 = 5\n"
    "4 + 1 = 5\n"
)

# -----------------------------
# CURRENT analysis definition
# -----------------------------
CURRENT_EXCLUDE = {
    (1, 1),
    (2, 2),
    (5, 3),
    (3, 5),
}

CURRENT_PREFIX = (
    "1 + 1 = 2\n"
    "2 + 2 = 4\n"
    "5 + 3 = 8\n"
)


# ============================================================
# Deterministic cohort builder
# ============================================================

def build_cohort(exclude_set, prefix=""):
    rows = []

    for a in range(1, 10):
        for b in range(1, 10):

            ans = a + b

            # Same single-digit-answer rule used in M1
            if ans >= 10:
                continue

            if (a, b) in exclude_set:
                continue

            prompt = prefix + f"{a} + {b} ="
            target = f" {ans}"

            rows.append({
                "a": a,
                "b": b,
                "answer": ans,
                "prompt": prompt,
                "target": target,
            })

    return rows


# ============================================================
# Top-1 evaluator
# Uses TOKEN ID equality, not decoded string equality.
# This avoids formatting/string-decoding ambiguity.
# ============================================================

def evaluate_top1_exact(model, rows):

    results = []

    for item in rows:

        logits = model(item["prompt"])

        # logits shape: [batch, seq, vocab]
        last_logits = logits[0, -1, :]

        pred_id = int(torch.argmax(last_logits).item())
        target_id = int(model.to_single_token(item["target"]))

        hit = (pred_id == target_id)

        results.append({
            **item,
            "pred_id": pred_id,
            "target_id": target_id,
            "hit": hit,
            "pred_str": model.to_string(pred_id),
        })

    return results


# ============================================================
# Build OLD and CURRENT cohorts
# ============================================================

old_rows = build_cohort(
    OLD_EXCLUDE,
    OLD_PREFIX
)

current_rows = build_cohort(
    CURRENT_EXCLUDE,
    CURRENT_PREFIX
)

print("=" * 70)
print("COHORT SIZES")
print("=" * 70)

print(f"OLD cohort     : n = {len(old_rows)}")
print(f"CURRENT cohort : n = {len(current_rows)}")


# ============================================================
# Evaluate both using the exact same model/evaluator
# ============================================================

old_results = evaluate_top1_exact(model, old_rows)
current_results = evaluate_top1_exact(model, current_rows)

old_hits = [
    (r["a"], r["b"], r["answer"], r["pred_str"])
    for r in old_results
    if r["hit"]
]

current_hits = [
    (r["a"], r["b"], r["answer"], r["pred_str"])
    for r in current_results
    if r["hit"]
]


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 70)
print("TOP-1 RESULTS")
print("=" * 70)

print(
    f"OLD     : {len(old_hits)}/{len(old_results)} "
    f"= {len(old_hits)/len(old_results):.3f}"
)

print(
    f"CURRENT : {len(current_hits)}/{len(current_results)} "
    f"= {len(current_hits)/len(current_results):.3f}"
)


# ============================================================
# Exact hit lists
# ============================================================

print("\n" + "=" * 70)
print("OLD HIT LIST")
print("=" * 70)

for h in old_hits:
    print(
        f"{h[0]} + {h[1]} = {h[2]} "
        f"| predicted = {repr(h[3])}"
    )

print("\n" + "=" * 70)
print("CURRENT HIT LIST")
print("=" * 70)

for h in current_hits:
    print(
        f"{h[0]} + {h[1]} = {h[2]} "
        f"| predicted = {repr(h[3])}"
    )


# ============================================================
# Compare the actual operand sets
# ============================================================

old_pairs = {(r["a"], r["b"]) for r in old_results}
current_pairs = {(r["a"], r["b"]) for r in current_results}

old_only = sorted(old_pairs - current_pairs)
current_only = sorted(current_pairs - old_pairs)
common_pairs = sorted(old_pairs & current_pairs)

print("\n" + "=" * 70)
print("COHORT MEMBERSHIP DIFFERENCES")
print("=" * 70)

print(f"Common operand pairs : {len(common_pairs)}")
print(f"OLD-only pairs       : {len(old_only)}")
print(f"CURRENT-only pairs   : {len(current_only)}")

print("\nOLD-only:")
print(old_only)

print("\nCURRENT-only:")
print(current_only)


# ============================================================
# Compare hits as operand pairs
# ============================================================

old_hit_pairs = {(r["a"], r["b"]) for r in old_results if r["hit"]}
current_hit_pairs = {(r["a"], r["b"]) for r in current_results if r["hit"]}

print("\n" + "=" * 70)
print("HIT-LIST DIFFERENCE")
print("=" * 70)

print("OLD hits:")
print(sorted(old_hit_pairs))

print("\nCURRENT hits:")
print(sorted(current_hit_pairs))

print("\nOLD hits not in CURRENT:")
print(sorted(old_hit_pairs - current_hit_pairs))

print("\nCURRENT hits not in OLD:")
print(sorted(current_hit_pairs - old_hit_pairs))


# ============================================================
# Show the exact prompts for every hit
# ============================================================

print("\n" + "=" * 70)
print("EXACT OLD HIT PROMPTS")
print("=" * 70)

for r in old_results:
    if r["hit"]:
        print(repr(r["prompt"]))

print("\n" + "=" * 70)
print("EXACT CURRENT HIT PROMPTS")
print("=" * 70)

for r in current_results:
    if r["hit"]:
        print(repr(r["prompt"]))


# ============================================================
# Final diagnostic
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC")
print("=" * 70)

print("OLD prefix:")
print(repr(OLD_PREFIX))

print("\nCURRENT prefix:")
print(repr(CURRENT_PREFIX))

print("\nOLD exclusions:")
print(sorted(OLD_EXCLUDE))

print("\nCURRENT exclusions:")
print(sorted(CURRENT_EXCLUDE))

print("\nInterpretation:")
print(
    "The OLD and CURRENT experiments are not the same experiment: "
    "both the few-shot prefix and the excluded operand pairs changed."
)
print(
    "Therefore, 5/33 vs 2/32 cannot be treated as a contradiction "
    "until the OLD definition is reproduced under the current model."
)
```

#### Recorded output

```text
======================================================================
COHORT SIZES
======================================================================
OLD cohort     : n = 33
CURRENT cohort : n = 32

======================================================================
TOP-1 RESULTS
======================================================================
OLD     : 5/33 = 0.152
CURRENT : 2/32 = 0.062

======================================================================
OLD HIT LIST
======================================================================
1 + 5 = 6 | predicted = ' 6'
1 + 6 = 7 | predicted = ' 7'
2 + 4 = 6 | predicted = ' 6'
3 + 3 = 6 | predicted = ' 6'
4 + 2 = 6 | predicted = ' 6'

======================================================================
CURRENT HIT LIST
======================================================================
2 + 1 = 3 | predicted = ' 3'
3 + 1 = 4 | predicted = ' 4'

======================================================================
COHORT MEMBERSHIP DIFFERENCES
======================================================================
Common operand pairs : 30
OLD-only pairs       : 3
CURRENT-only pairs   : 2

OLD-only:
[(2, 2), (3, 5), (5, 3)]

CURRENT-only:
[(2, 3), (4, 1)]

======================================================================
HIT-LIST DIFFERENCE
======================================================================
OLD hits:
[(1, 5), (1, 6), (2, 4), (3, 3), (4, 2)]

CURRENT hits:
[(2, 1), (3, 1)]

OLD hits not in CURRENT:
[(1, 5), (1, 6), (2, 4), (3, 3), (4, 2)]

CURRENT hits not in OLD:
[(2, 1), (3, 1)]

======================================================================
EXACT OLD HIT PROMPTS
======================================================================
'1 + 1 = 2\n2 + 3 = 5\n4 + 1 = 5\n1 + 5 ='
'1 + 1 = 2\n2 + 3 = 5\n4 + 1 = 5\n1 + 6 ='
'1 + 1 = 2\n2 + 3 = 5\n4 + 1 = 5\n2 + 4 ='
'1 + 1 = 2\n2 + 3 = 5\n4 + 1 = 5\n3 + 3 ='
'1 + 1 = 2\n2 + 3 = 5\n4 + 1 = 5\n4 + 2 ='

======================================================================
EXACT CURRENT HIT PROMPTS
======================================================================
'1 + 1 = 2\n2 + 2 = 4\n5 + 3 = 8\n2 + 1 ='
'1 + 1 = 2\n2 + 2 = 4\n5 + 3 = 8\n3 + 1 ='

======================================================================
DIAGNOSTIC
======================================================================
OLD prefix:
'1 + 1 = 2\n2 + 3 = 5\n4 + 1 = 5\n'

CURRENT prefix:
'1 + 1 = 2\n2 + 2 = 4\n5 + 3 = 8\n'

OLD exclusions:
[(1, 1), (2, 3), (4, 1)]

CURRENT exclusions:
[(1, 1), (2, 2), (3, 5), (5, 3)]

Interpretation:
The OLD and CURRENT experiments are not the same experiment: both the few-shot prefix and the excluded operand pairs changed.
Therefore, 5/33 vs 2/32 cannot be treated as a contradiction until the OLD definition is reproduced under the current model.
```


### Cell 105 — code (execution count: 71)

```python
# ── Sanity checks: cohort sizes must match expected counts ──

# Extract the actual cohorts using the logic defined in your earlier cells
zero_shot_all_pairs     = build(False)  # From Cohort A cell
few_shot_earlier_cohort = build(True)   # From Cohort A cell
zero_shot_held_out      = cohort        # From Cohort B/C cell
few_shot_held_out       = cohort        # From Cohort B/C cell

assert len(zero_shot_all_pairs) == 36, f"Expected 36, got {len(zero_shot_all_pairs)}"
assert len(zero_shot_held_out)  == 32, f"Expected 32, got {len(zero_shot_held_out)}"
assert len(few_shot_held_out)   == 32, f"Expected 32, got {len(few_shot_held_out)}"
assert len(few_shot_earlier_cohort) == 33, f"Expected 33, got {len(few_shot_earlier_cohort)}"

print("All-pairs zero-shot: ", len(zero_shot_all_pairs))
print("Held-out zero-shot:  ", len(zero_shot_held_out))
print("Held-out few-shot:   ", len(few_shot_held_out))
print("Earlier few-shot:    ", len(few_shot_earlier_cohort))
```

#### Recorded output

```text
All-pairs zero-shot:  36
Held-out zero-shot:   32
Held-out few-shot:    32
Earlier few-shot:     33
```


### Cell 106 — markdown

### M1 Evaluation Results

| Cohort | Correct | N | Accuracy |
|---|---|---|---|
| All-pairs zero-shot | 2 | 36 | 5.56% |
| Held-out zero-shot | 2 | 32 | 6.25% |
| Held-out few-shot | 2 | 32 | 6.25% |
| Earlier few-shot | 5 | 33 | 15.15% |
| Best constant guess (held-out) | 8 | 32 | 25.00% |

> **Do not report 6.2% alongside N=36.** The 6.2% figure belongs to the
> 2/32 held-out cohort. The 2/36 all-pairs result is 5.56%.


### Cell 107 — markdown

### M1 Final Interpretation

The M1 benchmark does not provide evidence for a general, reliably functioning
addition mechanism under the evaluated task distribution.

The model shows some correct-answer-token preference, particularly under few-shot
prompting, but full-vocabulary top-1 arithmetic performance remains low and is
sensitive to token-specific and neighborhood controls.

The result therefore constrains the addition-circuit hypothesis without implying
that GPT-2 Small contains no arithmetic-related representations anywhere in the
network.


### Cell 108 — code (execution count: 72)

```python
passed = df[df.top1 & (df["diff"] >= 1.0)]
failed = df[~(df.top1 & (df["diff"] >= 1.0))]
dla_pass = run_dla_analysis(list(passed.prompt), list(passed.t), list(passed.f)).drop(columns="prompt").mean()
dla_fail = run_dla_analysis(list(failed.prompt), list(failed.t), list(failed.f)).drop(columns="prompt").mean()
print(pd.DataFrame({"passed": dla_pass, "failed": dla_fail, "gap": dla_pass - dla_fail}))
```

#### Recorded output

```text
                     passed    failed       gap
ground_truth_diff  1.453397 -0.608653  2.062050
b_U_diff           0.124364  0.040586  0.083778
embeddings        -0.002520  0.000135 -0.002654
L0_attn            0.010694  0.005149  0.005545
L0_mlp            -0.035097  0.026036 -0.061133
L1_attn            0.008521  0.004291  0.004230
L1_mlp            -0.000044 -0.008251  0.008207
L2_attn           -0.015232  0.007359 -0.022591
L2_mlp            -0.009114 -0.011404  0.002290
L3_attn            0.080524 -0.030819  0.111342
L3_mlp            -0.021147  0.021900 -0.043047
L4_attn            0.009132  0.005663  0.003470
L4_mlp             0.007263  0.002678  0.004585
L5_attn            0.090259 -0.019090  0.109349
L5_mlp             0.003277 -0.002156  0.005433
L6_attn            0.044830 -0.011610  0.056440
L6_mlp            -0.003308  0.004016 -0.007324
L7_attn            0.194117 -0.055696  0.249814
L7_mlp            -0.032798  0.004286 -0.037084
L8_attn            0.185231 -0.088726  0.273956
L8_mlp             0.132067 -0.077966  0.210033
L9_attn            0.461339 -0.229202  0.690541
L9_mlp             0.266083 -0.140714  0.406796
L10_attn           0.026301  0.027161 -0.000859
L10_mlp           -0.172613 -0.042770 -0.129843
L11_attn           0.125430 -0.025317  0.150747
L11_mlp           -0.024160 -0.014192 -0.009968
```


### Cell 109 — code (execution count: 73)

```python
import torch
import pandas as pd
import numpy as np

def run_control_diagnostics(model, prompt_matrix):
    # -------------------------------------------------------------------------
    # DIAGNOSTIC 1: Non-Arithmetic Control Prompts (Unigram / Parity Bias)
    # -------------------------------------------------------------------------
    non_math_templates = [
        "The object in the box =",

        "The word on the page =",
        "Yesterday at the store =",
        "The item sequence number ="
    ]

    control_records = []

    model.reset_hooks()
    with torch.no_grad():
        for item in prompt_matrix:
            target_val = int(item.target_token.strip())
            is_even = (target_val % 2 == 0)

            # Compute baseline symmetric diff over arithmetic prompt
            arith_logits, _ = model.run_with_cache(item.clean_prompt)
            arith_diff = compute_symmetric_logit_diff(
                arith_logits, item.target_id, item.foil_ids
            ).item()

            # Compute symmetric diff over non-arithmetic filler templates
            filler_diffs = []
            for tmpl in non_math_templates:
                filler_logits, _ = model.run_with_cache(tmpl)
                f_diff = compute_symmetric_logit_diff(
                    filler_logits, item.target_id, item.foil_ids
                ).item()
                filler_diffs.append(f_diff)

            mean_filler_diff = float(np.mean(filler_diffs))

            control_records.append({
                "Prompt": item.clean_prompt,
                "Target": target_val,
                "Parity": "Even" if is_even else "Odd",
                "Tier": item.tier,
                "Arith_Baseline_Diff": arith_diff,
                "Filler_Control_Diff": mean_filler_diff,
                "Net_Arithmetic_Signal": arith_diff - mean_filler_diff
            })

    df_ctrl = pd.DataFrame(control_records)

    print("=== DIAGNOSTIC 1: NON-ARITHMETIC CONTROL vs ARITHMETIC BASELINE ===")
    parity_summary = df_ctrl.groupby("Parity").agg(
        Count=("Target", "count"),
        Arith_Baseline_Mean=("Arith_Baseline_Diff", "mean"),
        Arith_Baseline_Std=("Arith_Baseline_Diff", "std"),
        Filler_Control_Mean=("Filler_Control_Diff", "mean"),
        Filler_Control_Std=("Filler_Control_Diff", "std"),
        Net_Signal_Mean=("Net_Arithmetic_Signal", "mean")
    ).reset_index()

    print(parity_summary.to_string(index=False))

    # -------------------------------------------------------------------------
    # DIAGNOSTIC 2: Parity-Controlled Subgroup Analysis (Doubles vs Non-Doubles)
    # -------------------------------------------------------------------------
    # Parse single-digit subset (Easy & Medium)
    records_subgroup = []

    with torch.no_grad():
        for item in prompt_matrix:
            if item.tier in ["Easy", "Medium"]:
                tokens_str = item.clean_prompt.split()
                a_val = int(tokens_str[0])
                b_val = int(tokens_str[2])
                target_val = int(item.target_token.strip())
                is_double = (a_val == b_val)
                is_target_10 = (target_val == 10)
                parity = "Even" if (target_val % 2 == 0) else "Odd"

                # Baseline arithmetic diff
                clean_logits, _ = model.run_with_cache(item.clean_prompt)
                base_diff = compute_symmetric_logit_diff(
                    clean_logits, item.target_id, item.foil_ids
                ).item()

                # Non-math filler baseline for this item
                filler_diffs = []
                for tmpl in non_math_templates:
                    f_logits, _ = model.run_with_cache(tmpl)
                    filler_diffs.append(
                        compute_symmetric_logit_diff(f_logits, item.target_id, item.foil_ids).item()
                    )
                mean_filler = float(np.mean(filler_diffs))

                # Assign strict parity-controlled category
                if is_double:
                    cat = "1. Doubles (a=b) [All Even Sums]"
                elif is_target_10:
                    cat = "2. Target-10 Non-Doubles [Even Sum]"
                elif parity == "Even":
                    cat = "3. Non-Doubles (Even Sum Control)"
                else:
                    cat = "4. Non-Doubles (Odd Sum Control)"

                records_subgroup.append({
                    "Category": cat,
                    "Prompt": item.clean_prompt,
                    "Target": target_val,
                    "Parity": parity,
                    "Baseline_Diff": base_diff,
                    "Filler_Diff": mean_filler,
                    "Net_Signal": base_diff - mean_filler
                })

    df_sub = pd.DataFrame(records_subgroup)

    print("\n=== DIAGNOSTIC 2: PARITY-CONTROLLED SUBGROUP BREAKDOWN ===")
    subgroup_summary = df_sub.groupby("Category").agg(
        Count=("Target", "count"),
        Raw_Baseline_Mean=("Baseline_Diff", "mean"),
        Filler_Control_Mean=("Filler_Diff", "mean"),
        Net_Arithmetic_Signal=("Net_Signal", "mean"),
        Std_Dev=("Net_Signal", "std")
    ).reset_index()

    print(subgroup_summary.to_string(index=False))

    return df_ctrl, df_sub

# Execute Diagnostics
df_ctrl_results, df_sub_results = run_control_diagnostics(model, prompt_matrix)
```

#### Recorded output

```text
=== DIAGNOSTIC 1: NON-ARITHMETIC CONTROL vs ARITHMETIC BASELINE ===
Parity  Count  Arith_Baseline_Mean  Arith_Baseline_Std  Filler_Control_Mean  Filler_Control_Std  Net_Signal_Mean
  Even     41             0.176473            0.235988             0.079970            0.418308         0.096503
   Odd     43            -0.179560            0.201615            -0.155533            0.325510        -0.024027

=== DIAGNOSTIC 2: PARITY-CONTROLLED SUBGROUP BREAKDOWN ===
                           Category  Count  Raw_Baseline_Mean  Filler_Control_Mean  Net_Arithmetic_Signal  Std_Dev
   1. Doubles (a=b) [All Even Sums]      9           0.324127             0.055802               0.268325 0.394308
2. Target-10 Non-Doubles [Even Sum]      6           0.117616             0.538263              -0.420647 0.063844
  3. Non-Doubles (Even Sum Control)     24           0.154637             0.067207               0.087430 0.281372
   4. Non-Doubles (Odd Sum Control)     40          -0.172868            -0.160970              -0.011898 0.305815
```


### Cell 110 — code (execution count: 74)

```python
import torch

# 1. Verify BPE Token IDs
tokens = [" 8", " 9", " 10", " 11"]
token_ids = {tok: model.to_single_token(tok) for tok in tokens}
print("BPE Token Mapping:", token_ids)

# 2. Inspect Unembedding Bias (b_U) Asymmetry
b_U = model.b_U if hasattr(model, 'b_U') else torch.zeros(model.cfg.d_vocab)
print("\n--- Unembedding Bias (b_U) Comparison ---")
for tok, tid in token_ids.items():
    print(f"Token '{tok}' (ID {tid:<4}): b_U = {b_U[tid].item():+.4f}")

# 3. Exact Target-10 Filler Control Inspection
filler_prompt = "The object in the box ="
logits, _ = model.run_with_cache(filler_prompt)
last_logits = logits[0, -1, :]

logit_10 = last_logits[token_ids[" 10"]].item()
logit_9 = last_logits[token_ids[" 9"]].item()
logit_11 = last_logits[token_ids[" 11"]].item()

print(f"\n--- Filler Prompt: '{filler_prompt}' ---")
print(f"Logit(' 10'): {logit_10:+.4f}")
print(f"Logit(' 9') : {logit_9:+.4f} (Single-digit foil)")
print(f"Logit(' 11'): {logit_11:+.4f} (Two-digit foil)")
print(f"Filler Logit Diff (' 10' vs [' 9', ' 11']): {logit_10 - (logit_9 + logit_11)/2:+.4f}")
```

#### Recorded output

```text
BPE Token Mapping: {' 8': 807, ' 9': 860, ' 10': 838, ' 11': 1367}

--- Unembedding Bias (b_U) Comparison ---
Token ' 8' (ID 807 ): b_U = +3.3653
Token ' 9' (ID 860 ): b_U = +3.1871
Token ' 10' (ID 838 ): b_U = +3.7600
Token ' 11' (ID 1367): b_U = +3.0021

--- Filler Prompt: 'The object in the box =' ---
Logit(' 10'): +8.2909
Logit(' 9') : +7.8045 (Single-digit foil)
Logit(' 11'): +6.6474 (Two-digit foil)
Filler Logit Diff (' 10' vs [' 9', ' 11']): +1.0649
```


### Cell 111 — code (execution count: 75)

```python

import torch
import pandas as pd

def evaluate_doubles_vs_matched_controls(model):
    # Matched pairs sharing EXACT target and foil tokens
    matched_pairs = [
        {"target": " 6", "foils": [" 5", " 7"], "double": "3 + 3 =", "non_double": "4 + 2 ="},
        {"target": " 8", "foils": [" 7", " 9"], "double": "4 + 4 =", "non_double": "5 + 3 ="},
        {"target": " 10", "foils": [" 9", " 11"], "double": "5 + 5 =", "non_double": "6 + 4 ="},
        {"target": " 12", "foils": [" 11", " 13"], "double": "6 + 6 =", "non_double": "7 + 5 ="},
        {"target": " 14", "foils": [" 13", " 15"], "double": "7 + 7 =", "non_double": "8 + 6 ="},
        {"target": " 16", "foils": [" 15", " 17"], "double": "8 + 8 =", "non_double": "9 + 7 ="},
    ]

    filler_templates = [
        "The object in the box =",
        "The word on the page =",
        "Yesterday at the store =",
        "The item sequence number ="
    ]

    results = []

    model.reset_hooks()
    with torch.no_grad():
        for p in matched_pairs:
            target_id = model.to_single_token(p["target"])
            foil_ids = [model.to_single_token(f) for f in p["foils"]]

            # 1. Output-side Unembedding Bias Check (b_U)
            b_U = model.b_U if hasattr(model, 'b_U') else torch.zeros(model.cfg.d_vocab)
            bias_diff = (b_U[target_id] - torch.mean(b_U[foil_ids])).item()

            # 2. Filler Baseline Logit Diff (Target preference in non-math context)
            filler_diffs = []
            for tmpl in filler_templates:
                f_logits, _ = model.run_with_cache(tmpl)
                f_diff = compute_symmetric_logit_diff(f_logits, target_id, foil_ids).item()
                filler_diffs.append(f_diff)
            mean_filler = sum(filler_diffs) / len(filler_diffs)

            # 3. Double Prompt Logit Diff (e.g., "4 + 4 =")
            dbl_logits, _ = model.run_with_cache(p["double"])
            dbl_raw = compute_symmetric_logit_diff(dbl_logits, target_id, foil_ids).item()

            # 4. Matched Non-Double Prompt Logit Diff (e.g., "5 + 3 =")
            ndbl_logits, _ = model.run_with_cache(p["non_double"])
            ndbl_raw = compute_symmetric_logit_diff(ndbl_logits, target_id, foil_ids).item()

            # 5. Net Arithmetic Signals (Arithmetic Logit Diff - Filler Baseline)
            dbl_net = dbl_raw - mean_filler
            ndbl_net = ndbl_raw - mean_filler
            doubles_effect = dbl_raw - ndbl_raw

            results.append({
                "Target": p["target"].strip(),
                "Double_Prompt": p["double"],
                "NonDouble_Prompt": p["non_double"],
                "b_U_Bias_Diff": bias_diff,
                "Filler_Baseline": mean_filler,
                "Double_Raw": dbl_raw,
                "NonDouble_Raw": ndbl_raw,
                "Double_Net": dbl_net,
                "NonDouble_Net": ndbl_net,
                "Doubles_Advantage": doubles_effect
            })

    df = pd.DataFrame(results)

    print("=== MATCHED TARGET/FOIL DOUBLES CONTROL EXPERIMENT ===")
    print(df[["Target", "Double_Prompt", "NonDouble_Prompt", "Filler_Baseline",
              "Double_Raw", "NonDouble_Raw", "Double_Net", "NonDouble_Net", "Doubles_Advantage"]].to_string(index=False))

    print("\n--- Summary Statistics Across Matched Pairs ---")
    print(f"Mean Filler Baseline Logit Diff : {df['Filler_Baseline'].mean():+.4f}")
    print(f"Mean Double Raw Logit Diff     : {df['Double_Raw'].mean():+.4f}")
    print(f"Mean Non-Double Raw Logit Diff : {df['NonDouble_Raw'].mean():+.4f}")
    print(f"Mean Double Net Signal         : {df['Double_Net'].mean():+.4f}")
    print(f"Mean Non-Double Net Signal     : {df['NonDouble_Net'].mean():+.4f}")
    print(f"Mean Doubles Advantage (Delta) : {df['Doubles_Advantage'].mean():+.4f}")

    return df

# Execute Matched Pair Diagnostics
df_doubles_check = evaluate_doubles_vs_matched_controls(model)
```

#### Recorded output

```text
=== MATCHED TARGET/FOIL DOUBLES CONTROL EXPERIMENT ===
Target Double_Prompt NonDouble_Prompt  Filler_Baseline  Double_Raw  NonDouble_Raw  Double_Net  NonDouble_Net  Doubles_Advantage
     6       3 + 3 =          4 + 2 =        -0.214199    0.230024       0.263328    0.444223       0.477526          -0.033303
     8       4 + 4 =          5 + 3 =         0.120319    0.605705       0.304052    0.485386       0.183733           0.301653
    10       5 + 5 =          6 + 4 =         0.538263    0.647227       0.036099    0.108964      -0.502164           0.611128
    12       6 + 6 =          7 + 5 =         0.521429    0.401203       0.161645   -0.120226      -0.359784           0.239558
    14       7 + 7 =          8 + 6 =        -0.231127   -0.238291      -0.214467   -0.007164       0.016660          -0.023824
    16       8 + 8 =          9 + 7 =         0.003147    0.719002      -0.007687    0.715854      -0.010834           0.726688

--- Summary Statistics Across Matched Pairs ---
Mean Filler Baseline Logit Diff : +0.1230
Mean Double Raw Logit Diff     : +0.3941
Mean Non-Double Raw Logit Diff : +0.0905
Mean Double Net Signal         : +0.2712
Mean Non-Double Net Signal     : -0.0325
Mean Doubles Advantage (Delta) : +0.3037
```


### Cell 112 — markdown

## Transition: From Addition Circuit to Equal-Operand Behavior

The preceding analyses began with a search for a general addition circuit.

That hypothesis did not survive the complete audit.

However, a different phenomenon remained reproducible: prompts with equal operands
such as `4 + 4 =` tended to produce a larger target-token preference than matched
non-equal operands such as `5 + 3 =`, while preserving the same target and foil
tokens.

This changes the mechanistic question.

Rather than asking:

> Where is the addition circuit?

we now ask:

> What internal mechanism produces the equal-operand advantage?

The remaining experiments are therefore designed specifically to reverse-engineer
this surviving behavior and to determine whether it corresponds to an identifiable,
causal circuit.


### Cell 113 — code (execution count: 76)

```python
import torch
import pandas as pd

def run_target12_and_filtered_doubles_diagnostic(model):
    filler_templates = [
        "The object in the box =",
        "The word on the page =",
        "Yesterday at the store =",
        "The item sequence number ="
    ]

    # 1. Target=12 BPE & Filler Baseline Diagnostic
    tokens_12 = [" 11", " 12", " 13"]
    token_ids_12 = {tok: model.to_single_token(tok) for tok in tokens_12}
    b_U = model.b_U if hasattr(model, 'b_U') else torch.zeros(model.cfg.d_vocab)

    print("=== DIAGNOSTIC 1: TARGET 12 TOKEN & UNEMBEDDING BIAS (b_U) CHECK ===")
    print("BPE Token Mapping:", token_ids_12)
    for tok, tid in token_ids_12.items():
        print(f"Token '{tok}' (ID {tid:<5}): b_U = {b_U[tid].item():+.4f}")

    target_12_id = token_ids_12[" 12"]
    foil_ids_12 = [token_ids_12[" 11"], token_ids_12[" 13"]]

    filler_diffs_12 = [
        compute_symmetric_logit_diff(model.run_with_cache(tmpl)[0], target_12_id, foil_ids_12).item()
        for tmpl in filler_templates
    ]
    mean_filler_12 = sum(filler_diffs_12) / len(filler_diffs_12)
    print(f"\nTarget 12 Mean Filler Baseline Diff (' 12' vs [' 11', ' 13']): {mean_filler_12:+.4f}")

    # 2. Matched Doubles Control (Excluding Target 10)
    matched_pairs = [
        {"target": " 6", "foils": [" 5", " 7"], "double": "3 + 3 =", "non_double": "4 + 2 ="},
        {"target": " 8", "foils": [" 7", " 9"], "double": "4 + 4 =", "non_double": "5 + 3 ="},
        {"target": " 12", "foils": [" 11", " 13"], "double": "6 + 6 =", "non_double": "7 + 5 ="},
        {"target": " 14", "foils": [" 13", " 15"], "double": "7 + 7 =", "non_double": "8 + 6 ="},
        {"target": " 16", "foils": [" 15", " 17"], "double": "8 + 8 =", "non_double": "9 + 7 ="},
    ]

    results = []
    with torch.no_grad():
        for p in matched_pairs:
            target_id = model.to_single_token(p["target"])
            foil_ids = [model.to_single_token(f) for f in p["foils"]]

            f_diffs = [
                compute_symmetric_logit_diff(model.run_with_cache(tmpl)[0], target_id, foil_ids).item()
                for tmpl in filler_templates
            ]
            mean_filler = sum(f_diffs) / len(f_diffs)

            dbl_raw = compute_symmetric_logit_diff(model.run_with_cache(p["double"])[0], target_id, foil_ids).item()
            ndbl_raw = compute_symmetric_logit_diff(model.run_with_cache(p["non_double"])[0], target_id, foil_ids).item()

            results.append({
                "Target": p["target"].strip(),
                "Double_Prompt": p["double"],
                "NonDouble_Prompt": p["non_double"],
                "Filler_Baseline": mean_filler,
                "Double_Raw": dbl_raw,
                "NonDouble_Raw": ndbl_raw,
                "Double_Net": dbl_raw - mean_filler,
                "NonDouble_Net": ndbl_raw - mean_filler,
                "Doubles_Advantage": dbl_raw - ndbl_raw
            })

    df = pd.DataFrame(results)

    print("\n=== MATCHED DOUBLES CONTROL (EXCLUDING TARGET 10) ===")
    print(df[["Target", "Double_Prompt", "NonDouble_Prompt", "Filler_Baseline",
              "Double_Raw", "NonDouble_Raw", "Double_Net", "NonDouble_Net", "Doubles_Advantage"]].to_string(index=False))

    print("\n--- Summary Statistics (Excluding Target 10) ---")
    print(f"Mean Filler Baseline Logit Diff : {df['Filler_Baseline'].mean():+.4f}")
    print(f"Mean Doubles Advantage (Delta) : {df['Doubles_Advantage'].mean():+.4f}")

    return df

# Execute Diagnostic
df_filtered = run_target12_and_filtered_doubles_diagnostic(model)
```

#### Recorded output

```text
=== DIAGNOSTIC 1: TARGET 12 TOKEN & UNEMBEDDING BIAS (b_U) CHECK ===
BPE Token Mapping: {' 11': 1367, ' 12': 1105, ' 13': 1511}
Token ' 11' (ID 1367 ): b_U = +3.0021
Token ' 12' (ID 1105 ): b_U = +3.2965
Token ' 13' (ID 1511 ): b_U = +2.7944

Target 12 Mean Filler Baseline Diff (' 12' vs [' 11', ' 13']): +0.5214

=== MATCHED DOUBLES CONTROL (EXCLUDING TARGET 10) ===
Target Double_Prompt NonDouble_Prompt  Filler_Baseline  Double_Raw  NonDouble_Raw  Double_Net  NonDouble_Net  Doubles_Advantage
     6       3 + 3 =          4 + 2 =        -0.214199    0.230024       0.263328    0.444223       0.477526          -0.033303
     8       4 + 4 =          5 + 3 =         0.120319    0.605705       0.304052    0.485386       0.183733           0.301653
    12       6 + 6 =          7 + 5 =         0.521429    0.401203       0.161645   -0.120226      -0.359784           0.239558
    14       7 + 7 =          8 + 6 =        -0.231127   -0.238291      -0.214467   -0.007164       0.016660          -0.023824
    16       8 + 8 =          9 + 7 =         0.003147    0.719002      -0.007687    0.715854      -0.010834           0.726688

--- Summary Statistics (Excluding Target 10) ---
Mean Filler Baseline Logit Diff : +0.0399
Mean Doubles Advantage (Delta) : +0.2422
```


### Cell 114 — code (execution count: 77)

```python
import torch
import einops
import pandas as pd
import numpy as np
from transformer_lens import HookedTransformer
from transformer_lens.utils import test_prompt
from torch.utils.data import DataLoader

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================
DEVICE = "cpu" if torch.cpu.is_available() else "cuda"
print(f"Using device: {DEVICE}")

model.eval()

# ==========================================
# 2. CONTROLLED DATASET GENERATION
# ==========================================
# Define multiple templates to test format-dependence
TEMPLATES = [
    "{op1}+{op2}=",
    "{op1} + {op2} =",
    "{op1} plus {op2} equals"
]

def generate_evaluation_suite():
    """
    Generates structured pairs of:
    - True arithmetic prompts (e.g., '3+5=')
    - Position- and token-matched non-arithmetic filler controls (e.g., '3x5xy')
    - Answer-fixed groups (e.g., targets that resolve to 8: 3+5, 2+6, 4+4)
    """
    suite = []

    # Example equation bank with answer-fixed groupings targeting '8' and '16'
    equation_groups = {
        8: [("3", "5"), ("2", "6"), ("1", "7"), ("4", "4")],
        16: [("8", "8"), ("7", "9")]
    }

    foil_token = "9" # Standardized foil for contrast

    for target_val, pairs in equation_groups.items():
        for op1, op2 in pairs:
            for template in TEMPLATES:
                # 1. True Arithmetic Prompt
                arith_prompt = template.format(op1=op1, op2=op2)

                # 2. Position- & Token-Matched Filler Control (replaces operator/symbols with neutral tokens)
                # Ensures exact token length and embedding geometry match minus arithmetic semantics
                filler_template = template.replace("+", "x").replace("=", "y").replace("plus", "and").replace("equals", "gives")
                filler_prompt = filler_template.format(op1=op1, op2=op2)

                suite.append({
                    "target": str(target_val),
                    "foil": foil_token,
                    "arith_prompt": arith_prompt,
                    "filler_prompt": filler_prompt,
                    "is_double": (op1 == op2),
                    "template": template
                })

    return pd.DataFrame(suite)

df_eval = generate_evaluation_suite()
print(f"Generated {len(df_eval)} controlled evaluation rows across templates.")
```

#### Recorded output

```text
Using device: cpu
Generated 18 controlled evaluation rows across templates.
```

```text
/tmp/ipykernel_108922/2684954333.py:6: DeprecationWarning:

The 'utils' module has been deprecated. Please use 'transformer_lens.utilities' instead. Importing from utils.py will be removed in TransformerLens 4.0.
```


### Cell 115 — code (execution count: 78)

```python
def compute_true_did(model, row):
    """
    Computes the true Difference-in-Differences estimator:
    DiD = (Logit(Target) - Logit(Foil))_Arithmetic - (Logit(Target) - Logit(Foil))_Filler
    """
    target_str = row["target"]
    foil_str = row["foil"]

    target_id = model.to_single_token(target_str)
    foil_id = model.to_single_token(foil_str)

    # --- 1. Run Arithmetic Prompt ---
    logits_arith, _ = model.run_with_cache(row["arith_prompt"])
    last_pos_arith = logits_arith[0, -1, :]
    delta_arith = last_pos_arith[target_id] - last_pos_arith[foil_id]

    # --- 2. Run Matched Filler Prompt ---
    logits_filler, _ = model.run_with_cache(row["filler_prompt"])
    last_pos_filler = logits_filler[0, -1, :]
    delta_filler = last_pos_filler[target_id] - last_pos_filler[foil_id]

    # --- 3. True Difference-in-Differences ---
    did_value = delta_arith - delta_filler

    return {
        "delta_arith": delta_arith.item(),
        "delta_filler": delta_filler.item(),
        "did_signal": did_value.item()
    }

# Re-running execution over the evaluation suite
results = []
for idx, row in df_eval.iterrows():
    metrics = compute_true_did(model, row)
    results.append({**row.to_dict(), **metrics})

df_did_results = pd.DataFrame(results)

print("\n--- TRUE DiD ANSWER-FIXED SWEEP (Target = 8) ---")
print(df_did_results[df_did_results["target"] == "8"][["arith_prompt", "is_double", "did_signal"]])
```

#### Recorded output

```text

--- TRUE DiD ANSWER-FIXED SWEEP (Target = 8) ---
       arith_prompt  is_double  did_signal
0              3+5=      False   -0.372133
1           3 + 5 =      False   -0.935297
2   3 plus 5 equals      False    0.288593
3              2+6=      False   -0.237706
4           2 + 6 =      False   -1.267207
5   2 plus 6 equals      False    0.372419
6              1+7=      False   -0.029224
7           1 + 7 =      False   -1.206789
8   1 plus 7 equals      False    0.112541
9              4+4=       True   -0.944210
10          4 + 4 =       True   -1.401953
11  4 plus 4 equals       True    0.559604
```


### Cell 116 — code (execution count: 79)

```python
import pandas as pd, torch

TEMPLATES = {
    "{op1}+{op2}=":            {"sp": "",  "filler": "cat+dog="},
    "{op1} + {op2} =":         {"sp": " ", "filler": "cat + dog ="},
    "{op1} plus {op2} equals": {"sp": " ", "filler": "cat plus dog equals"},
}
GROUPS = {8: [("3","5"),("2","6"),("1","7"),("4","4")],
          16: [("8","8"),("7","9")]}

def generate_evaluation_suite():
    rows = []
    for tgt, pairs in GROUPS.items():
        for a, b in pairs:
            for tmpl, cfg in TEMPLATES.items():
                sp = cfg["sp"]
                rows.append({
                    "target": tgt, "template": tmpl,
                    "target_tok": f"{sp}{tgt}",
                    "foil_toks": [f"{sp}{tgt-1}", f"{sp}{tgt+1}"],
                    "arith_prompt": tmpl.format(op1=a, op2=b),
                    "filler_prompt": cfg["filler"],
                    "is_double": a == b,
                })
    return pd.DataFrame(rows)

def compute_true_did(model, row):
    t = model.to_single_token(row["target_tok"])
    f = [model.to_single_token(x) for x in row["foil_toks"]]
    def diff(prompt):
        with torch.no_grad():
            lg = model(prompt)[0, -1]
        return (lg[t] - lg[f].mean()).item()
    da, dfil = diff(row["arith_prompt"]), diff(row["filler_prompt"])
    return {"delta_arith": da, "delta_filler": dfil, "did_signal": da - dfil}

df_eval = generate_evaluation_suite()
df_did = pd.DataFrame([{**r, **compute_true_did(model, r)} for _, r in df_eval.iterrows()])
print(df_did[["target","template","arith_prompt","is_double","did_signal"]].to_string(index=False))
```

#### Recorded output

```text
 target                template    arith_prompt  is_double  did_signal
      8            {op1}+{op2}=            3+5=      False    0.154325
      8         {op1} + {op2} =         3 + 5 =      False    0.275396
      8 {op1} plus {op2} equals 3 plus 5 equals      False   -0.277866
      8            {op1}+{op2}=            2+6=      False    0.263095
      8         {op1} + {op2} =         2 + 6 =      False    0.292650
      8 {op1} plus {op2} equals 2 plus 6 equals      False   -0.081479
      8            {op1}+{op2}=            1+7=      False   -0.071034
      8         {op1} + {op2} =         1 + 7 =      False    0.126455
      8 {op1} plus {op2} equals 1 plus 7 equals      False   -0.401561
      8            {op1}+{op2}=            4+4=       True    0.449819
      8         {op1} + {op2} =         4 + 4 =       True    0.469442
      8 {op1} plus {op2} equals 4 plus 4 equals       True    0.350301
     16            {op1}+{op2}=            8+8=       True    1.038495
     16         {op1} + {op2} =         8 + 8 =       True    0.895722
     16 {op1} plus {op2} equals 8 plus 8 equals       True    1.316249
     16            {op1}+{op2}=            7+9=      False    0.323347
     16         {op1} + {op2} =         7 + 9 =      False    0.258118
     16 {op1} plus {op2} equals 7 plus 9 equals      False   -0.046011
```


### Cell 117 — code (execution count: 80)

```python
import pandas as pd
import torch

TEMPLATES = {
    "{op1}+{op2}=":            {"sp": "",  "filler": "cat+dog="},
    "{op1} + {op2} =":         {"sp": " ", "filler": "cat + dog ="},
    "{op1} plus {op2} equals": {"sp": " ", "filler": "cat plus dog equals"},
}

# The previously observed "null" targets
GROUPS = {
    6:  {"doubles": [("3","3")], "controls": [("4","2"), ("5","1")]},
    14: {"doubles": [("7","7")], "controls": [("8","6"), ("9","5")]}
}

def generate_evaluation_suite():
    rows = []
    for tgt, pairs in GROUPS.items():
        # Process doubles
        for a, b in pairs["doubles"]:
            for tmpl, cfg in TEMPLATES.items():
                sp = cfg["sp"]
                rows.append({
                    "target": tgt,
                    "template": tmpl,
                    "target_tok": f"{sp}{tgt}",
                    "foil_toks": [f"{sp}{tgt-1}", f"{sp}{tgt+1}"],
                    "arith_prompt": tmpl.format(op1=a, op2=b),
                    "filler_prompt": cfg["filler"],
                    "is_double": True,
                    "pair_label": f"{a}+{b}"
                })
        # Process controls
        for a, b in pairs["controls"]:
            for tmpl, cfg in TEMPLATES.items():
                sp = cfg["sp"]
                rows.append({
                    "target": tgt,
                    "template": tmpl,
                    "target_tok": f"{sp}{tgt}",
                    "foil_toks": [f"{sp}{tgt-1}", f"{sp}{tgt+1}"],
                    "arith_prompt": tmpl.format(op1=a, op2=b),
                    "filler_prompt": cfg["filler"],
                    "is_double": False,
                    "pair_label": f"{a}+{b}"
                })
    return pd.DataFrame(rows)

def compute_true_did(model, row):
    t = model.to_single_token(row["target_tok"])
    f = [model.to_single_token(x) for x in row["foil_toks"]]

    def diff(prompt):
        with torch.no_grad():
            lg = model(prompt)[0, -1]
        return (lg[t] - lg[f].mean()).item()

    da = diff(row["arith_prompt"])
    dfil = diff(row["filler_prompt"])
    return {"delta_arith": da, "delta_filler": dfil, "did_signal": da - dfil}

# Execute evaluation on null targets
df_eval_nulls = generate_evaluation_suite()
df_did_nulls = pd.DataFrame([{**r, **compute_true_did(model, r)} for _, r in df_eval_nulls.iterrows()])

# Detailed output
print(df_did_nulls[["target", "template", "arith_prompt", "is_double", "did_signal"]].to_string(index=False))
```

#### Recorded output

```text
 target                template    arith_prompt  is_double  did_signal
      6            {op1}+{op2}=            3+3=       True    0.156927
      6         {op1} + {op2} =         3 + 3 =       True    0.474819
      6 {op1} plus {op2} equals 3 plus 3 equals       True    0.330229
      6            {op1}+{op2}=            4+2=      False    0.325162
      6         {op1} + {op2} =         4 + 2 =      False    0.508122
      6 {op1} plus {op2} equals 4 plus 2 equals      False    0.503551
      6            {op1}+{op2}=            5+1=      False    0.065456
      6         {op1} + {op2} =         5 + 1 =      False    0.307637
      6 {op1} plus {op2} equals 5 plus 1 equals      False   -0.093138
     14            {op1}+{op2}=            7+7=       True    0.265025
     14         {op1} + {op2} =         7 + 7 =       True    0.058463
     14 {op1} plus {op2} equals 7 plus 7 equals       True    0.036342
     14            {op1}+{op2}=            8+6=      False    0.313908
     14         {op1} + {op2} =         8 + 6 =      False    0.082287
     14 {op1} plus {op2} equals 8 plus 6 equals      False    0.190995
     14            {op1}+{op2}=            9+5=      False    0.224055
     14         {op1} + {op2} =         9 + 5 =      False    0.059179
     14 {op1} plus {op2} equals 9 plus 5 equals      False    0.001247
```


### Cell 118 — code (execution count: 81)

```python
import numpy as np
import pandas as pd

# The df_did_results DataFrame from a previous cell already contains the necessary data.
# We will use this DataFrame directly and rename its columns to match the expected structure.
did_df = df_did_results.copy()

# Add a 'pair_id' column for consistency with the original logic's expectation
did_df['pair_id'] = did_df.index

# Rename columns to match the expected 'arithmetic_signal' and 'filler_signal'
did_df = did_df.rename(columns={'delta_arith': 'arithmetic_signal', 'delta_filler': 'filler_signal'})

# Ensure all expected columns are present, dropping any that are not needed by the audit.
# The original code's validation of 'filler_df["net_signal"].isna().any()' is now implicitly handled
# as delta_filler (now filler_signal) is always computed if the row exists in df_did_results.

audit = did_df[
    [
        "pair_id",
        "arith_prompt",
        "filler_prompt",
        "arithmetic_signal",
        "filler_signal",
        "did_signal",
    ]
]

print(audit.to_string(index=False))
```

#### Recorded output

```text
 pair_id    arith_prompt filler_prompt  arithmetic_signal  filler_signal  did_signal
       0            3+5=          3x5y           0.560959       0.933092   -0.372133
       1         3 + 5 =       3 x 5 y           0.638235       1.573532   -0.935297
       2 3 plus 5 equals 3 and 5 gives           0.188052      -0.100541    0.288593
       3            2+6=          2x6y           0.705994       0.943700   -0.237706
       4         2 + 6 =       2 x 6 y           0.654894       1.922101   -1.267207
       5 2 plus 6 equals 2 and 6 gives           0.343659      -0.028759    0.372419
       6            1+7=          1x7y           0.308802       0.338026   -0.029224
       7         1 + 7 =       1 x 7 y           0.406257       1.613046   -1.206789
       8 1 plus 7 equals 1 and 7 gives          -0.016519      -0.129060    0.112541
       9            4+4=          4x4y           0.879041       1.823251   -0.944210
      10         4 + 4 =       4 x 4 y           0.800325       2.202278   -1.401953
      11 4 plus 4 equals 4 and 4 gives           0.713203       0.153599    0.559604
      12            8+8=          8x8y          -0.306602      -0.070627   -0.235975
      13         8 + 8 =       8 x 8 y           0.040417       1.948869   -1.908452
      14 8 plus 8 equals 8 and 8 gives           0.348700       0.188349    0.160351
      15            7+9=          7x9y          -1.164497      -1.127398   -0.037099
      16         7 + 9 =       7 x 9 y          -0.612493       1.732522   -2.345015
      17 7 plus 9 equals 7 and 9 gives          -1.564977      -0.393870   -1.171107
```


### Cell 119 — code (execution count: 82)

```python
print(audit[["pair_id", "filler_prompt", "did_signal"]].to_string(index=False))
```

#### Recorded output

```text
 pair_id filler_prompt  did_signal
       0          3x5y   -0.372133
       1       3 x 5 y   -0.935297
       2 3 and 5 gives    0.288593
       3          2x6y   -0.237706
       4       2 x 6 y   -1.267207
       5 2 and 6 gives    0.372419
       6          1x7y   -0.029224
       7       1 x 7 y   -1.206789
       8 1 and 7 gives    0.112541
       9          4x4y   -0.944210
      10       4 x 4 y   -1.401953
      11 4 and 4 gives    0.559604
      12          8x8y   -0.235975
      13       8 x 8 y   -1.908452
      14 8 and 8 gives    0.160351
      15          7x9y   -0.037099
      16       7 x 9 y   -2.345015
      17 7 and 9 gives   -1.171107
```


### Cell 120 — code (execution count: 83)

```python
audit = did_df[
    [
        "pair_id",
        "arith_prompt",
        "filler_prompt",
        "arithmetic_signal",
        "filler_signal",
        "did_signal",
    ]
].copy()

print(audit.to_string(index=False))

print("\nAre all filler signals zero?",
      np.allclose(audit["filler_signal"], 0.0))

print("Does DiD equal the arithmetic signal?",
      np.allclose(audit["did_signal"], audit["arithmetic_signal"]))

print("\nMaximum absolute filler signal:",
      audit["filler_signal"].abs().max())
```

#### Recorded output

```text
 pair_id    arith_prompt filler_prompt  arithmetic_signal  filler_signal  did_signal
       0            3+5=          3x5y           0.560959       0.933092   -0.372133
       1         3 + 5 =       3 x 5 y           0.638235       1.573532   -0.935297
       2 3 plus 5 equals 3 and 5 gives           0.188052      -0.100541    0.288593
       3            2+6=          2x6y           0.705994       0.943700   -0.237706
       4         2 + 6 =       2 x 6 y           0.654894       1.922101   -1.267207
       5 2 plus 6 equals 2 and 6 gives           0.343659      -0.028759    0.372419
       6            1+7=          1x7y           0.308802       0.338026   -0.029224
       7         1 + 7 =       1 x 7 y           0.406257       1.613046   -1.206789
       8 1 plus 7 equals 1 and 7 gives          -0.016519      -0.129060    0.112541
       9            4+4=          4x4y           0.879041       1.823251   -0.944210
      10         4 + 4 =       4 x 4 y           0.800325       2.202278   -1.401953
      11 4 plus 4 equals 4 and 4 gives           0.713203       0.153599    0.559604
      12            8+8=          8x8y          -0.306602      -0.070627   -0.235975
      13         8 + 8 =       8 x 8 y           0.040417       1.948869   -1.908452
      14 8 plus 8 equals 8 and 8 gives           0.348700       0.188349    0.160351
      15            7+9=          7x9y          -1.164497      -1.127398   -0.037099
      16         7 + 9 =       7 x 9 y          -0.612493       1.732522   -2.345015
      17 7 plus 9 equals 7 and 9 gives          -1.564977      -0.393870   -1.171107

Are all filler signals zero? False
Does DiD equal the arithmetic signal? False

Maximum absolute filler signal: 2.202277660369873
```


### Cell 121 — code (execution count: 84)

```python
import pandas as pd
import torch

def compute_true_did(model, row):
    """
    DiD = [logit(target) - mean(logit(foils))]_arithmetic
        - [logit(target) - mean(logit(foils))]_filler
    """
    # 1. Verify required prompt columns
    required = ["arith_prompt", "filler_prompt", "target_tok"]
    missing = [col for col in required if col not in row or pd.isna(row[col])]
    if missing:
        raise ValueError(f"Missing required column(s) {missing} in row {row.name}.")

    # 2. Extract single target token ID and foil token IDs
    target_id = model.to_single_token(row["target_tok"])

    # Handle both single 'foil' string or list 'foil_toks'
    if "foil_toks" in row and isinstance(row["foil_toks"], list):
        foil_ids = [model.to_single_token(f) for f in row["foil_toks"]]
    elif "foil" in row and pd.notna(row["foil"]):
        foil_ids = [model.to_single_token(str(row["foil"]))]
    else:
        raise KeyError("Row must contain either a 'foil' column or a 'foil_toks' list.")

    # 3. Compute logits without cache
    with torch.no_grad():
        logits_arith = model(row["arith_prompt"])[0, -1]
        logits_filler = model(row["filler_prompt"])[0, -1]

    # 4. Calculate deltas (target logit minus mean foil logit)
    delta_arith = logits_arith[target_id] - logits_arith[foil_ids].mean()
    delta_filler = logits_filler[target_id] - logits_filler[foil_ids].mean()
    did_value = delta_arith - delta_filler

    return {
        "delta_arith": delta_arith.item(),
        "delta_filler": delta_filler.item(),
        "did_signal": did_value.item(),
    }

# Ensure df_eval has required columns before iterating
if "filler_prompt" not in df_eval.columns or df_eval["filler_prompt"].isna().any():
    raise ValueError(
        "df_eval has missing filler prompts. Define and score the matched "
        "non-arithmetic control for every row before calculating DiD."
    )

# Compute metrics
results = []
for _, row in df_eval.iterrows():
    metrics = compute_true_did(model, row)
    results.append({**row.to_dict(), **metrics})

df_did_results = pd.DataFrame(results)

print("\n--- TRUE DiD ANSWER-FIXED SWEEP ---")

audit_columns = [
    "arith_prompt",
    "filler_prompt",
    "target",
    "is_double",
    "delta_arith",
    "delta_filler",
    "did_signal",
]
if "foil" in df_did_results.columns:
    audit_columns.insert(4, "foil")

# Match target whether stored as string or int
target_8 = df_did_results[df_did_results["target"].astype(str) == "8"]
print(target_8[audit_columns].to_string(index=False))

# Sanity check
max_diff = torch.tensor(
    (target_8["did_signal"] - target_8["delta_arith"]).to_numpy()
).abs().max().item()
print(f"\nDid DiD accidentally equal the arithmetic delta? {max_diff < 1e-7}")
```

#### Recorded output

```text

--- TRUE DiD ANSWER-FIXED SWEEP ---
   arith_prompt       filler_prompt  target  is_double  delta_arith  delta_filler  did_signal
           3+5=            cat+dog=       8      False     0.432576      0.278252    0.154325
        3 + 5 =         cat + dog =       8      False     0.411659      0.136263    0.275396
3 plus 5 equals cat plus dog equals       8      False     0.082077      0.359943   -0.277866
           2+6=            cat+dog=       8      False     0.541347      0.278252    0.263095
        2 + 6 =         cat + dog =       8      False     0.428913      0.136263    0.292650
2 plus 6 equals cat plus dog equals       8      False     0.278464      0.359943   -0.081479
           1+7=            cat+dog=       8      False     0.207217      0.278252   -0.071034
        1 + 7 =         cat + dog =       8      False     0.262718      0.136263    0.126455
1 plus 7 equals cat plus dog equals       8      False    -0.041617      0.359943   -0.401561
           4+4=            cat+dog=       8       True     0.728070      0.278252    0.449819
        4 + 4 =         cat + dog =       8       True     0.605705      0.136263    0.469442
4 plus 4 equals cat plus dog equals       8       True     0.710244      0.359943    0.350301

Did DiD accidentally equal the arithmetic delta? False
```


### Cell 122 — code (execution count: 85)

```python
import pandas as pd
import numpy as npf
import torch

def generate_multi_family_suite():
    """
    Generates an evaluation suite with multiple distinct filler families
    to test the robustness of the arithmetic-minus-filler effect.
    """
    equation_groups = {
        8: [("3", "5"), ("2", "6"), ("1", "7"), ("4", "4")],
        16: [("8", "8"), ("7", "9")]
    }
    foil_token = "9"

    suite = []

    # Define templates and their respective family-specific fillers
    templates_and_fillers = [
        # Base Symbol Family
        {
            "template": "{op1}+{op2}=",
            "family_name": "Symbolic_Operator_Swap",
            "filler_template": "{op1}x{op2}y"
        },
        {
            "template": "{op1} + {op2} =",
            "family_name": "Symbolic_Operator_Swap",
            "filler_template": "{op1} x {op2} y"
        },
        # Neutral Lexical Family
        {
            "template": "{op1} plus {op2} equals",
            "family_name": "Neutral_Lexical_Swap",
            "filler_template": "{op1} with {op2} yields"
        }
    ]

    for target_val, pairs in equation_groups.items():
        for op1, op2 in pairs:
            for item in templates_and_fillers:
                arith_prompt = item["template"].format(op1=op1, op2=op2)
                filler_prompt = item["filler_template"].format(op1=op1, op2=op2)

                suite.append({
                    "target": str(target_val),
                    "foil": foil_token,
                    "arith_prompt": arith_prompt,
                    "filler_prompt": filler_prompt,
                    "filler_family": item["family_name"],
                    "is_double": (op1 == op2),
                    "template": item["template"]
                })

    return pd.DataFrame(suite)

df_multi_family = generate_multi_family_suite()

# --- Execution Loop for Multi-Family DiD ---
results = []
for _, row in df_multi_family.iterrows():
    target_id = model.to_single_token(str(row["target"]))
    foil_id = model.to_single_token(str(row["foil"]))

    with torch.no_grad():
        logits_arith = model(row["arith_prompt"])
        logits_filler = model(row["filler_prompt"])

    delta_arith = logits_arith[0, -1, target_id] - logits_arith[0, -1, foil_id]
    delta_filler = logits_filler[0, -1, target_id] - logits_filler[0, -1, foil_id]
    did_value = delta_arith - delta_filler

    results.append({
        **row.to_dict(),
        "delta_arith": delta_arith.item(),
        "delta_filler": delta_filler.item(),
        "did_signal": did_value.item()
    })

df_results_family = pd.DataFrame(results)

# --- Reporting DiD Separately by Family ---
print("\n--- DiD REPLICATION ACROSS FILLER FAMILIES ---")
family_summary = df_results_family.groupby("filler_family").agg(
    mean_did=("did_signal", "mean"),
    std_did=("did_signal", "std"),
    count=("did_signal", "count")
).reset_index()

print(family_summary.to_string(index=False))
```

#### Recorded output

```text

--- DiD REPLICATION ACROSS FILLER FAMILIES ---
         filler_family  mean_did  std_did  count
  Neutral_Lexical_Swap -0.061409 0.496252      6
Symbolic_Operator_Swap -0.910088 0.753718     12
```


### Cell 123 — code (execution count: 86)

```python
# ==========================================
# PAIRED INTERSECTION COMPARISON ACROSS FAMILIES
# ==========================================

# 1. Create a unique identifier for each equation pair + answer combination
df_results_family["pair_key"] = df_results_family["target"] + "_" + df_results_family["arith_prompt"].str.extract(r'(\d+[\+\splus]+\d+)')[0]

# 2. Pivot the dataframe so each row represents a unique equation pair,
# and filler families become columns (comparing base symbolic vs lexical)
# Let's normalize by looking at the base structural mapping:
pivot_did = df_results_family.pivot_table(
    index=["target", "is_double"],
    columns="filler_family",
    values="did_signal"
).dropna()

print("\n--- PAIRED INTERSECTION ACROSS FAMILIES ---")
print(pivot_did.to_string())

# 3. Compute paired difference between families on the exact same pairs
if "Symbolic_Operator_Swap" in pivot_did.columns and "Neutral_Lexical_Swap" in pivot_did.columns:
    paired_diff = pivot_did["Symbolic_Operator_Swap"] - pivot_did["Neutral_Lexical_Swap"]
    print("\nPaired Difference (Symbolic minus Lexical) per equation pair:")
    print(paired_diff)
    print(f"\nMean Paired Difference: {paired_diff.mean():.4f} (std: {paired_diff.std():.4f})")
```

#### Recorded output

```text

--- PAIRED INTERSECTION ACROSS FAMILIES ---
filler_family     Neutral_Lexical_Swap  Symbolic_Operator_Swap
target is_double
16     False                 -0.853986               -1.191057
       True                   0.578861               -1.072213
8      False                 -0.127497               -0.674726
       True                   0.289162               -1.173081

Paired Difference (Symbolic minus Lexical) per equation pair:
target  is_double
16      False       -0.337071
        True        -1.651075
8       False       -0.547229
        True        -1.462243
dtype: float64

Mean Paired Difference: -0.9994 (std: 0.6537)
```


### Cell 124 — code (execution count: 87)

```python
df = df_results_family.copy()

# The same arithmetic prompt must be present once per filler family.
df["pair_key"] = (
    df["target"].astype(str)
    + " || "
    + df["arith_prompt"].astype(str)
)

# Refuse silent averaging: each pair must occur exactly once per family.
duplicates = (
    df.groupby(["pair_key", "filler_family"])
    .size()
    .reset_index(name="n")
)

if (duplicates["n"] != 1).any():
    print(duplicates[duplicates["n"] != 1])
    raise ValueError(
        "Duplicate pair/filler-family rows found. "
        "Add a control-variant column to the pairing key; do not average silently."
    )

# Exact prompt-level intersection across the two filler families.
wide = df.pivot(
    index=["pair_key", "target", "arith_prompt", "is_double"],
    columns="filler_family",
    values="did_signal",
).reset_index()

required = ["Neutral_Lexical_Swap", "Symbolic_Operator_Swap"]

paired = wide.dropna(subset=required).copy()

paired["family_difference"] = (
    paired["Symbolic_Operator_Swap"]
    - paired["Neutral_Lexical_Swap"]
)

print("\n--- TRUE PAIRED INTERSECTION ---")
print(
    paired[
        [
            "target",
            "arith_prompt",
            "is_double",
            "Neutral_Lexical_Swap",
            "Symbolic_Operator_Swap",
            "family_difference",
        ]
    ].to_string(index=False)
)

print("\nNumber of exactly matched prompts:", len(paired))
print("Mean paired difference:", paired["family_difference"].mean())
print("SD paired difference:", paired["family_difference"].std(ddof=1))
```

#### Recorded output

```text

--- TRUE PAIRED INTERSECTION ---
Empty DataFrame
Columns: [target, arith_prompt, is_double, Neutral_Lexical_Swap, Symbolic_Operator_Swap, family_difference]
Index: []

Number of exactly matched prompts: 0
Mean paired difference: nan
SD paired difference: nan
```


### Cell 125 — code (execution count: 88)

```python
families = ["Neutral_Lexical_Swap", "Symbolic_Operator_Swap"]

for family in families:
    keys = set(
        df.loc[df["filler_family"] == family, "pair_key"]
        .dropna()
        .unique()
    )
    print(f"\n{family}: {len(keys)} prompt pairs")
    for key in sorted(keys):
        print(" ", key)

lexical_keys = set(
    df.loc[df["filler_family"] == "Neutral_Lexical_Swap", "pair_key"]
)

symbolic_keys = set(
    df.loc[df["filler_family"] == "Symbolic_Operator_Swap", "pair_key"]
)

print("\nOnly lexical:")
print(sorted(lexical_keys - symbolic_keys))

print("\nOnly symbolic:")
print(sorted(symbolic_keys - lexical_keys))
```

#### Recorded output

```text

Neutral_Lexical_Swap: 6 prompt pairs
  16 || 7 plus 9 equals
  16 || 8 plus 8 equals
  8 || 1 plus 7 equals
  8 || 2 plus 6 equals
  8 || 3 plus 5 equals
  8 || 4 plus 4 equals

Symbolic_Operator_Swap: 12 prompt pairs
  16 || 7 + 9 =
  16 || 7+9=
  16 || 8 + 8 =
  16 || 8+8=
  8 || 1 + 7 =
  8 || 1+7=
  8 || 2 + 6 =
  8 || 2+6=
  8 || 3 + 5 =
  8 || 3+5=
  8 || 4 + 4 =
  8 || 4+4=

Only lexical:
['16 || 7 plus 9 equals', '16 || 8 plus 8 equals', '8 || 1 plus 7 equals', '8 || 2 plus 6 equals', '8 || 3 plus 5 equals', '8 || 4 plus 4 equals']

Only symbolic:
['16 || 7 + 9 =', '16 || 7+9=', '16 || 8 + 8 =', '16 || 8+8=', '8 || 1 + 7 =', '8 || 1+7=', '8 || 2 + 6 =', '8 || 2+6=', '8 || 3 + 5 =', '8 || 3+5=', '8 || 4 + 4 =', '8 || 4+4=']
```


### Cell 126 — code (execution count: 89)

```python
import re

# 1. Function to extract a canonical equation signature (e.g., operands and target)
def get_canonical_key(prompt_str):
    # Extracts numbers from the prompt string
    nums = re.findall(r'\d+', prompt_str)
    if len(nums) >= 2:
        return f"{nums[0]}_{nums[1]}"
    return prompt_str

# 2. Add canonical keys to your results dataframe
df_results_family["canonical_key"] = df_results_family["arith_prompt"].apply(get_canonical_key)

# 3. Subgroup symbolic templates by formatting style
def classify_format(prompt):
    if "+" in prompt and " " in prompt:
        return "Symbolic_Spaced"
    elif "+" in prompt and " " not in prompt:
        return "Symbolic_Compact"
    elif "plus" in prompt:
        return "Lexical_Spaced"
    return "Other"

df_results_family["format_subgroup"] = df_results_family["arith_prompt"].apply(classify_format)

# 4. Pivot to compare Lexical_Spaced directly against Symbolic_Spaced (true structural match)
pivot_clean = df_results_family.pivot_table(
    index=["target", "canonical_key"],
    columns="format_subgroup",
    values="did_signal"
).dropna(subset=["Lexical_Spaced", "Symbolic_Spaced"])

print("\n--- PAIRED INTERSECTION (Spaced Lexical vs. Spaced Symbolic) ---")
print(pivot_clean[["Lexical_Spaced", "Symbolic_Spaced"]].to_string())

# 5. Compute the clean paired difference
clean_paired_diff = pivot_clean["Symbolic_Spaced"] - pivot_clean["Lexical_Spaced"]
print(f"\nMean Paired Difference (Symbolic Spaced minus Lexical Spaced): {clean_paired_diff.mean():.4f} (std: {clean_paired_diff.std():.4f})")
```

#### Recorded output

```text

--- PAIRED INTERSECTION (Spaced Lexical vs. Spaced Symbolic) ---
format_subgroup       Lexical_Spaced  Symbolic_Spaced
target canonical_key
16     7_9                 -0.853986        -2.345015
       8_8                  0.578861        -1.908452
8      1_7                 -0.322716        -1.206789
       2_6                 -0.025598        -1.267207
       3_5                 -0.034178        -0.935297
       4_4                  0.289162        -1.401953

Mean Paired Difference (Symbolic Spaced minus Lexical Spaced): -1.4494 (std: 0.6001)
```


### Cell 127 — code (execution count: 90)

```python
# ============================================================
# DIAGNOSTIC: 7 / 14 TOKEN + UNEMBEDDING BIAS CHECK
# Purpose:
# Test whether the 7+7=14 reversal could be explained by
# tokenization or static output-space / unembedding bias.
# ============================================================

import torch
import pandas as pd

# ------------------------------------------------------------
# 1. Tokens involved in the 7 and 14 comparisons
# ------------------------------------------------------------

diagnostic_tokens = [
    " 7",
    " 13",
    " 14",
    " 15",
]

print("=" * 70)
print("DIAGNOSTIC 1: TOKEN IDS + STATIC UNEMBEDDING BIAS")
print("=" * 70)

token_info = []

for tok in diagnostic_tokens:
    # Verify that each string is exactly one GPT-2 token
    ids = model.to_tokens(tok, prepend_bos=False)[0]

    token_count = len(ids)
    token_id = model.to_single_token(tok) if token_count == 1 else None

    # b_U is the static unembedding bias
    if hasattr(model, "b_U"):
        b_u = model.b_U[token_id].item() if token_id is not None else float("nan")
    else:
        b_u = float("nan")

    # Unembedding-vector norm
    if token_id is not None:
        w_u_norm = model.W_U[:, token_id].norm().item()
    else:
        w_u_norm = float("nan")

    token_info.append({
        "token": tok,
        "token_id": token_id,
        "token_count": token_count,
        "b_U": b_u,
        "W_U_norm": w_u_norm,
    })

token_df = pd.DataFrame(token_info)

print(token_df.to_string(index=False))

print("\nToken strings according to GPT-2:")
for tok in diagnostic_tokens:
    ids = model.to_tokens(tok, prepend_bos=False)[0]
    print(f"{tok!r} -> IDs {ids.tolist()} -> "
          f"{model.to_str_tokens(ids)}")


# ------------------------------------------------------------
# 2. Check the target-vs-adjacent-foil metric for target 14
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSTIC 2: TARGET 14 FILLER BASELINE")
print("=" * 70)

target_14 = model.to_single_token(" 14")

foil_14 = [
    model.to_single_token(" 13"),
    model.to_single_token(" 15"),
]

filler_templates = [
    "The object in the box =",
    "The word on the page =",
    "Yesterday at the store =",
    "The item sequence number =",
]

filler_results_14 = []

for tmpl in filler_templates:

    logits, _ = model.run_with_cache(tmpl)

    diff = compute_symmetric_logit_diff(
        logits,
        target_14,
        foil_14
    ).item()

    filler_results_14.append({
        "template": tmpl,
        "symmetric_diff_14": diff
    })

filler_df_14 = pd.DataFrame(filler_results_14)

print(filler_df_14.to_string(index=False))

mean_filler_14 = filler_df_14["symmetric_diff_14"].mean()

print(
    f"\nTarget 14 mean filler baseline "
    f"(14 vs 13/15): {mean_filler_14:+.4f}"
)


# ------------------------------------------------------------
# 3. Check target 7 against adjacent single-digit foils
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSTIC 3: TARGET 7 FILLER BASELINE")
print("=" * 70)

target_7 = model.to_single_token(" 7")

foil_7 = [
    model.to_single_token(" 6"),
    model.to_single_token(" 8"),
]

filler_results_7 = []

for tmpl in filler_templates:

    logits, _ = model.run_with_cache(tmpl)

    diff = compute_symmetric_logit_diff(
        logits,
        target_7,
        foil_7
    ).item()

    filler_results_7.append({
        "template": tmpl,
        "symmetric_diff_7": diff
    })

filler_df_7 = pd.DataFrame(filler_results_7)

print(filler_df_7.to_string(index=False))

mean_filler_7 = filler_df_7["symmetric_diff_7"].mean()

print(
    f"\nTarget 7 mean filler baseline "
    f"(7 vs 6/8): {mean_filler_7:+.4f}"
)


# ------------------------------------------------------------
# 4. Direct comparison of the two relevant targets
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSTIC 4: 7 vs 14 OUTPUT-SPACE COMPARISON")
print("=" * 70)

comparison = pd.DataFrame([
    {
        "target": "7",
        "target_id": model.to_single_token(" 7"),
        "b_U": model.b_U[model.to_single_token(" 7")].item(),
        "mean_filler_diff": mean_filler_7,
    },
    {
        "target": "14",
        "target_id": model.to_single_token(" 14"),
        "b_U": model.b_U[model.to_single_token(" 14")].item(),
        "mean_filler_diff": mean_filler_14,
    }
])

print(comparison.to_string(index=False))


# ------------------------------------------------------------
# 5. Simple interpretation flags
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSTIC FLAGS")
print("=" * 70)

print(
    f"14 b_U = "
    f"{model.b_U[model.to_single_token(' 14')].item():+.4f}"
)

print(
    f"7 b_U  = "
    f"{model.b_U[model.to_single_token(' 7')].item():+.4f}"
)

print(
    f"14 filler baseline = "
    f"{mean_filler_14:+.4f}"
)

print(
    f"7 filler baseline = "
    f"{mean_filler_7:+.4f}"
)

print("\nInterpretation:")
print(
    "- A large 14 filler baseline would indicate substantial "
    "task-independent output preference for token ' 14'."
)
print(
    "- A large 7 filler baseline would indicate analogous "
    "task-independent preference for token ' 7'."
)
print(
    "- This diagnostic alone does NOT establish that the "
    "7+7=14 reversal is an artifact."
)
```

#### Recorded output

```text
======================================================================
DIAGNOSTIC 1: TOKEN IDS + STATIC UNEMBEDDING BIAS
======================================================================
token  token_id  token_count      b_U  W_U_norm
    7       767            1 3.407630  3.150503
   13      1511            1 2.794446  3.231750
   14      1478            1 2.769754  3.240118
   15      1315            1 3.056077  3.206985

Token strings according to GPT-2:
' 7' -> IDs [767] -> [' 7']
' 13' -> IDs [1511] -> [' 13']
' 14' -> IDs [1478] -> [' 14']
' 15' -> IDs [1315] -> [' 15']

======================================================================
DIAGNOSTIC 2: TARGET 14 FILLER BASELINE
======================================================================
                  template  symmetric_diff_14
   The object in the box =          -0.452552
    The word on the page =          -0.296002
  Yesterday at the store =           0.188878
The item sequence number =          -0.364831

Target 14 mean filler baseline (14 vs 13/15): -0.2311

======================================================================
DIAGNOSTIC 3: TARGET 7 FILLER BASELINE
======================================================================
                  template  symmetric_diff_7
   The object in the box =         -0.189260
    The word on the page =         -0.116718
  Yesterday at the store =          0.009834
The item sequence number =         -0.050679

Target 7 mean filler baseline (7 vs 6/8): -0.0867

======================================================================
DIAGNOSTIC 4: 7 vs 14 OUTPUT-SPACE COMPARISON
======================================================================
target  target_id      b_U  mean_filler_diff
     7        767 3.407630         -0.086706
    14       1478 2.769754         -0.231127

======================================================================
DIAGNOSTIC FLAGS
======================================================================
14 b_U = +2.7698
7 b_U  = +3.4076
14 filler baseline = -0.2311
7 filler baseline = -0.0867

Interpretation:
- A large 14 filler baseline would indicate substantial task-independent output preference for token ' 14'.
- A large 7 filler baseline would indicate analogous task-independent preference for token ' 7'.
- This diagnostic alone does NOT establish that the 7+7=14 reversal is an artifact.
```


### Cell 128 — code (execution count: 91)

```python
import numpy as np, torch

def score(prompt, T):
    ids = [model.to_single_token(f" {x}") for x in (T, T - 1, T + 1)]
    with torch.no_grad():
        lg = model(prompt)[0, -1]
    return (lg[ids[0]] - 0.5 * (lg[ids[1]] + lg[ids[2]])).item()

rows = []
for T in range(4, 17, 2):
    d = T // 2
    dbl = score(f"{d} + {d} =", T)
    ctl = [(a, T - a) for a in range(1, 10) if 1 <= T - a <= 9 and a != T - a]
    cs = np.array([score(f"{a} + {b} =", T) for a, b in ctl])
    sd = cs.std(ddof=1)
    rows.append((T, len(cs), dbl, cs.mean(), dbl - cs.mean(), (dbl - cs.mean()) / sd, 1 + int((cs > dbl).sum())))

print(f"{'T':>3} {'n_ctl':>5} {'double':>8} {'ctl_mean':>9} {'adv':>7} {'adv/ctl_sd':>10} {'rank':>8}")
for T, n, dbl, cm, adv, z, rk in rows:
    print(f"{T:>3} {n:>5} {dbl:>+8.3f} {cm:>+9.3f} {adv:>+7.3f} {z:>+10.2f} {rk:>4}/{n + 1}")

adv = np.array([r[4] for r in rows])
print(f"\nmean adv {adv.mean():+.3f}  SE {adv.std(ddof=1) / np.sqrt(len(adv)):.3f}  positive {int((adv > 0).sum())}/{len(adv)}")
 # ── Save per-prompt scores
import pandas as pd

rows_export = []
for T in range(4, 17, 2):
    d = T // 2
    dbl = score(f"{d} + {d} =", T)
    ctl = [(a, T - a) for a in range(1, 10) if 1 <= T - a <= 9 and a != T - a]

    # Save all individual control prompt scores
    for a, b in ctl:
        rows_export.append({"T": T, "type": "control", "prompt": f"{a} + {b} =", "score": score(f"{a} + {b} =", T)})

    # Save the double prompt score
    rows_export.append({"T": T, "type": "double", "prompt": f"{d} + {d} =", "score": dbl})

df_doubles = pd.DataFrame(rows_export)
df_doubles.to_csv("doubles_scan_scores.csv", index=False)
print("Saved doubles_scan_scores.csv (per-prompt breakdown)")
```

#### Recorded output

```text
  T n_ctl   double  ctl_mean     adv adv/ctl_sd     rank
  4     2   +0.251    +0.203  +0.049      +0.29    2/3
  6     4   +0.230    +0.213  +0.017      +0.13    3/5
  8     6   +0.606    +0.315  +0.291      +3.31    1/7
 10     8   +0.647    +0.113  +0.535      +9.54    1/9
 12     6   +0.401    +0.236  +0.165      +1.63    1/7
 14     4   -0.238    -0.232  -0.006      -0.12    4/5
 16     2   +0.719    +0.037  +0.682     +10.83    1/3

mean adv +0.247  SE 0.102  positive 6/7
Saved doubles_scan_scores.csv (per-prompt breakdown)
```


### Cell 129 — markdown

## Formal validation of the equal-operand effect:


### Cell 130 — code (execution count: 92)

```python
#  PRIMARY EQUAL-OPERAND METRIC

import numpy as np
import pandas as pd

# Work only with the digit+digit '+' experiment already exported
# by the preceding doubles_scan_scores cell.
df_primary = df_doubles.copy()

# Sanity checks
required_cols = {"T", "type", "prompt", "score"}
missing = required_cols - set(df_primary.columns)
if missing:
    raise ValueError(f"df_doubles is missing required columns: {missing}")

# Keep only the primary '+' digit+digit analysis
# (df_doubles was generated from prompts like "a + b =")
df_primary = df_primary[
    df_primary["prompt"].str.contains(r"\+", regex=True, na=False)
].copy()

df_primary["type"] = df_primary["type"].astype(str)

# Compute one matched effect for each target:
#   double score - mean score of all non-double controls for that target
target_rows = []

for T in sorted(df_primary["T"].unique()):

    target_data = df_primary[df_primary["T"] == T]

    doubles = target_data[target_data["type"] == "double"]["score"].to_numpy()
    controls = target_data[target_data["type"] == "control"]["score"].to_numpy()

    if len(doubles) != 1:
        raise ValueError(
            f"T={T}: expected exactly 1 double prompt, found {len(doubles)}"
        )

    if len(controls) < 2:
        raise ValueError(
            f"T={T}: too few control observations ({len(controls)})"
        )

    double_score = float(doubles[0])
    control_mean = float(np.mean(controls))
    advantage = double_score - control_mean

    target_rows.append({
        "T": int(T),
        "double_score": double_score,
        "control_mean": control_mean,
        "advantage": advantage,
        "n_controls": len(controls),
    })

equal_operand_summary = pd.DataFrame(target_rows)

print("=" * 72)
print("PRIMARY EQUAL-OPERAND EFFECT")
print("=" * 72)

print(equal_operand_summary.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

advantages = equal_operand_summary["advantage"].to_numpy()

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)

print(f"Target levels                : {len(advantages)}")
print(f"Mean equal-operand advantage : {advantages.mean():+.6f}")
print(f"SD across target levels      : {advantages.std(ddof=1):+.6f}")
print(f"Min target-level advantage   : {advantages.min():+.6f}")
print(f"Max target-level advantage   : {advantages.max():+.6f}")

# Save the target-level analysis for the repository/paper
equal_operand_summary.to_csv(
    "equal_operand_primary_target_summary.csv",
    index=False
)

print("\nSaved: equal_operand_primary_target_summary.csv")
```

#### Recorded output

```text
========================================================================
PRIMARY EQUAL-OPERAND EFFECT
========================================================================
 T  double_score  control_mean  advantage  n_controls
 4       +0.2515       +0.2028    +0.0486           2
 6       +0.2300       +0.2132    +0.0168           4
 8       +0.6057       +0.3147    +0.2910           6
10       +0.6472       +0.1126    +0.5346           8
12       +0.4012       +0.2365    +0.1647           6
14       -0.2383       -0.2321    -0.0062           4
16       +0.7190       +0.0369    +0.6821           2

========================================================================
SUMMARY
========================================================================
Target levels                : 7
Mean equal-operand advantage : +0.247385
SD across target levels      : +0.269948
Min target-level advantage   : -0.006201
Max target-level advantage   : +0.682146

Saved: equal_operand_primary_target_summary.csv
```


### Cell 131 — code (execution count: 93)

```python
# EXACT PAIRED SIGN-PERMUTATION TEST + BOOTSTRAP CI

import itertools

advantages = equal_operand_summary["advantage"].to_numpy(dtype=float)
n_targets = len(advantages)

observed_mean = advantages.mean()

# ------------------------------------------------------------
# 1. Exact sign-flip permutation test
# ------------------------------------------------------------
# Under H0: no systematic equal-operand advantage.
# With n target-level observations, enumerate all 2^n sign flips.

permuted_means = []

for signs in itertools.product([-1, 1], repeat=n_targets):
    signs = np.asarray(signs, dtype=float)
    permuted_means.append(np.mean(advantages * signs))

permuted_means = np.asarray(permuted_means)

# Two-sided exact p-value
p_exact = np.mean(
    np.abs(permuted_means) >= abs(observed_mean)
)

# ------------------------------------------------------------
# 2. Bootstrap 95% CI over target levels
# ------------------------------------------------------------

rng = np.random.default_rng(20261008)
n_boot = 10000

bootstrap_means = np.empty(n_boot)

for i in range(n_boot):
    sample = rng.choice(
        advantages,
        size=n_targets,
        replace=True
    )
    bootstrap_means[i] = np.mean(sample)

ci_low, ci_high = np.percentile(
    bootstrap_means,
    [2.5, 97.5]
)

# ------------------------------------------------------------
# 3. Descriptive effect size
# ------------------------------------------------------------

target_sd = advantages.std(ddof=1)

# This is deliberately NOT called a t-statistic.
adv_over_sd_target = (
    observed_mean / target_sd
    if target_sd > 0 else np.nan
)

# ------------------------------------------------------------
# 4. Report
# ------------------------------------------------------------


print("FORMAL INFERENCE FOR EQUAL-OPERAND EFFECT")


print(f"Number of target levels       : {n_targets}")
print(f"Observed mean advantage       : {observed_mean:+.6f}")
print(f"Exact sign-flip p-value      : {p_exact:.6f}")
print(f"Bootstrap 95% CI              : [{ci_low:+.6f}, {ci_high:+.6f}]")
print(f"Target-level advantage / SD   : {adv_over_sd_target:+.4f}")

print("\nInterpretation:")
if p_exact < 0.05 and ci_low > 0:
    print(
        "The equal-operand advantage is positive across target levels, "
        "with the exact paired sign-flip test rejecting a zero-mean "
        "target-level effect at alpha=0.05."
    )
elif p_exact < 0.05:
    print(
        "The exact sign-flip test rejects a zero-mean target-level effect, "
        "but the bootstrap interval crosses zero; interpret cautiously."
    )
else:
    print(
        "The target-level evidence does not reject a zero-mean "
        "equal-operand effect under the exact sign-flip test."
    )

# Save inferential summary
inference_summary = pd.DataFrame([{
    "n_target_levels": n_targets,
    "mean_advantage": observed_mean,
    "exact_signflip_p": p_exact,
    "bootstrap_ci_low": ci_low,
    "bootstrap_ci_high": ci_high,
    "advantage_over_target_sd": adv_over_sd_target,
}])

inference_summary.to_csv(
    "equal_operand_primary_inference.csv",
    index=False
)

print("\nSaved: equal_operand_primary_inference.csv")
```

#### Recorded output

```text
FORMAL INFERENCE FOR EQUAL-OPERAND EFFECT
Number of target levels       : 7
Observed mean advantage       : +0.247385
Exact sign-flip p-value      : 0.031250
Bootstrap 95% CI              : [+0.073815, +0.437425]
Target-level advantage / SD   : +0.9164

Interpretation:
The equal-operand advantage is positive across target levels, with the exact paired sign-flip test rejecting a zero-mean target-level effect at alpha=0.05.

Saved: equal_operand_primary_inference.csv
```


### Cell 132 — code (execution count: 94)

```python
# PARITY-CONTROLLED EQUAL-OPERAND TEST
#
# Primary comparison:
#   double:   d + d
#   controls: a + b, where a+b=T and a!=b
#
# Difference from the original metric:
#   target T is compared against T-2 and T+2,
#   rather than T-1 and T+1.
#
# This keeps target and foil parity matched

import numpy as np
import pandas as pd
import torch
import itertools

PARITY_TARGETS = list(range(4, 17, 2))


def single_token_id(token_string):
    """Return GPT-2 token ID and fail loudly if representation is not single-token."""
    ids = model.to_tokens(token_string, prepend_bos=False)

    if ids.numel() != 1:
        raise ValueError(
            f"{token_string!r} is not a single GPT-2 token. "
            f"Token IDs: {ids.tolist()}"
        )

    return int(ids.flatten()[0].item())


def score_parity_controlled(prompt, T):
    """
    Target-vs-foil score using parity-preserving neighboring targets:
        target = T
        foils  = T-2, T+2
    """
    target_id = single_token_id(f" {T}")
    foil_minus_id = single_token_id(f" {T-2}")
    foil_plus_id = single_token_id(f" {T+2}")

    with torch.no_grad():
        logits = model(prompt)[0, -1]

    return (
        logits[target_id]
        - 0.5 * (logits[foil_minus_id] + logits[foil_plus_id])
    ).item()


parity_rows = []

for T in PARITY_TARGETS:

    d = T // 2

    # All ordered non-double controls with the same target sum.
    controls = []

    for a in range(1, 10):
        b = T - a

        if 1 <= b <= 9 and a != b:
            controls.append((a, b))

    double_prompt = f"{d} + {d} ="

    double_score = score_parity_controlled(
        double_prompt,
        T
    )

    control_scores = [
        score_parity_controlled(
            f"{a} + {b} =",
            T
        )
        for a, b in controls
    ]

    control_mean = float(np.mean(control_scores))
    advantage = double_score - control_mean

    parity_rows.append({
        "T": T,
        "double_prompt": double_prompt,
        "double_score": double_score,
        "control_mean": control_mean,
        "advantage": advantage,
        "n_controls": len(control_scores),
    })


df_parity_control = pd.DataFrame(parity_rows)

advantages = df_parity_control["advantage"].to_numpy(dtype=float)


print("PARITY-CONTROLLED EQUAL-OPERAND TEST")


print(
    df_parity_control.to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}"
    )
)

print("\nSummary")
print("-" * 78)

print(f"Targets tested        : {len(advantages)}")
print(f"Mean advantage        : {advantages.mean():+.6f}")
print(f"SD across targets     : {advantages.std(ddof=1):+.6f}")
print(f"Positive targets      : {(advantages > 0).sum()}/{len(advantages)}")

# ------------------------------------------------------------
# Exact paired sign-flip test
# ------------------------------------------------------------

permuted_means = np.array([
    np.mean(
        advantages * np.asarray(signs, dtype=float)
    )
    for signs in itertools.product([-1, 1], repeat=len(advantages))
])

p_signflip = np.mean(
    np.abs(permuted_means) >= abs(advantages.mean())
)

print(f"Exact sign-flip p    : {p_signflip:.6f}")

# ------------------------------------------------------------
# Bootstrap confidence interval
# ------------------------------------------------------------

rng = np.random.default_rng(20261008)
N_BOOT = 10000

boot_means = np.empty(N_BOOT)

for i in range(N_BOOT):
    sample = rng.choice(
        advantages,
        size=len(advantages),
        replace=True
    )
    boot_means[i] = sample.mean()

ci_low, ci_high = np.percentile(
    boot_means,
    [2.5, 97.5]
)

print(
    f"Bootstrap 95% CI    : "
    f"[{ci_low:+.6f}, {ci_high:+.6f}]"
)

df_parity_control.to_csv(
    "equal_operand_parity_control.csv",
    index=False
)

pd.DataFrame([{
    "n_targets": len(advantages),
    "mean_advantage": advantages.mean(),
    "signflip_p": p_signflip,
    "bootstrap_ci_low": ci_low,
    "bootstrap_ci_high": ci_high,
}]).to_csv(
    "equal_operand_parity_control_inference.csv",
    index=False
)

print("\nSaved:")
print("  equal_operand_parity_control.csv")
print("  equal_operand_parity_control_inference.csv")
```

#### Recorded output

```text
PARITY-CONTROLLED EQUAL-OPERAND TEST
 T double_prompt  double_score  control_mean  advantage  n_controls
 4       2 + 2 =     +0.125508     +0.236757  -0.111248           2
 6       3 + 3 =     -0.163944     +0.015317  -0.179262           4
 8       4 + 4 =     +0.316150     +0.167482  +0.148668           6
10       5 + 5 =     +0.302339     -0.161769  +0.464108           8
12       6 + 6 =     +0.261011     +0.101276  +0.159735           6
14       7 + 7 =     -0.211317     -0.398112  +0.186795           4
16       8 + 8 =     +0.815971     +0.179035  +0.636936           2

Summary
------------------------------------------------------------------------------
Targets tested        : 7
Mean advantage        : +0.186533
SD across targets     : +0.289983
Positive targets      : 5/7
Exact sign-flip p    : 0.140625
Bootstrap 95% CI    : [-0.006814, +0.386278]

Saved:
  equal_operand_parity_control.csv
  equal_operand_parity_control_inference.csv
```


### Cell 133 — code (execution count: 95)

```python
# ============================================================
# FORMAL FORMAT COMPARISON
#
# Formats:
#   digit + digit
#   digit + word
#   word + word
#
# Statistical unit:
#   target level
#
# Global test:
#   Friedman repeated-measures test
#
# Pairwise tests:
#   exact paired sign-flip
#
# Multiple comparisons:
#   Holm correction
# ============================================================

import numpy as np
import pandas as pd
import itertools
from scipy import stats

WORDS_N4 = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}

FORMAT_FUNCTIONS_N4 = {
    "digit+digit": lambda a, b: f"{a} + {b} =",
    "digit+word":  lambda a, b: f"{a} + {WORDS_N4[b]} =",
    "word+word":   lambda a, b: f"{WORDS_N4[a]} + {WORDS_N4[b]} =",
}


def score_standard_n4(prompt, T):
    target_id = model.to_single_token(f" {T}")
    foil_minus = model.to_single_token(f" {T-1}")
    foil_plus = model.to_single_token(f" {T+1}")

    with torch.no_grad():
        logits = model(prompt)[0, -1]

    return (
        logits[target_id]
        - 0.5 * (
            logits[foil_minus]
            + logits[foil_plus]
        )
    ).item()


def format_advantage(T, formatter):
    d = T // 2

    pairs = [
        (a, T-a)
        for a in range(1, 10)
        if 1 <= T-a <= 9 and a != T-a and a < T-a
    ]

    double_score = score_standard_n4(
        formatter(d, d),
        T
    )

    control_scores = []

    for a, b in pairs:
        score_ab = score_standard_n4(
            formatter(a, b),
            T
        )

        score_ba = score_standard_n4(
            formatter(b, a),
            T
        )

        control_scores.append(
            0.5 * (score_ab + score_ba)
        )

    return (
        double_score
        - np.mean(control_scores)
    )


N4_TARGETS = list(range(4, 17, 2))

format_matrix = []

for T in N4_TARGETS:

    row = {"T": T}

    for name, formatter in FORMAT_FUNCTIONS_N4.items():

        row[name] = format_advantage(
            T,
            formatter
        )

    format_matrix.append(row)


df_format_adv = pd.DataFrame(format_matrix)

print("FORMAT-LEVEL EQUAL-OPERAND ADVANTAGE")


print(
    df_format_adv.to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}"
    )
)

# ------------------------------------------------------------
# Friedman omnibus test
# ------------------------------------------------------------

format_columns = list(FORMAT_FUNCTIONS_N4.keys())

friedman_result = stats.friedmanchisquare(
    *[
        df_format_adv[name].to_numpy(dtype=float)
        for name in format_columns
    ]
)

print("\nFriedman repeated-measures test")
print("-" * 78)
print(f"Statistic : {friedman_result.statistic:.6f}")
print(f"p-value   : {friedman_result.pvalue:.6f}")


# ------------------------------------------------------------
# Exact paired sign-flip test
# ------------------------------------------------------------

def exact_signflip_two_sided(values):
    values = np.asarray(values, dtype=float)

    null_means = np.array([
        np.mean(values * np.asarray(signs))
        for signs in itertools.product([-1, 1], repeat=len(values))
    ])

    observed = np.mean(values)

    return np.mean(
        np.abs(null_means) >= abs(observed)
    )


pairwise_results = []

pairs = list(
    itertools.combinations(
        format_columns,
        2
    )
)

for fmt_a, fmt_b in pairs:

    difference = (
        df_format_adv[fmt_a]
        - df_format_adv[fmt_b]
    ).to_numpy(dtype=float)

    p = exact_signflip_two_sided(
        difference
    )

    pairwise_results.append({
        "format_a": fmt_a,
        "format_b": fmt_b,
        "mean_difference": difference.mean(),
        "p_raw": p,
    })


df_format_pairwise = pd.DataFrame(
    pairwise_results
)


# ------------------------------------------------------------
# Holm correction
# ------------------------------------------------------------

def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)

    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)

    m = len(p_values)

    running_max = 0.0

    for rank, idx in enumerate(order):
        value = (
            p_values[idx]
            * (m - rank)
        )

        running_max = max(
            running_max,
            value
        )

        adjusted[idx] = min(
            running_max,
            1.0
        )

    return adjusted


df_format_pairwise["p_holm"] = holm_adjust(
    df_format_pairwise["p_raw"].to_numpy()
)

print("\nPairwise exact sign-flip comparisons")
print("-" * 78)

print(
    df_format_pairwise.to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}"
    )
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df_format_adv.to_csv(
    "equal_operand_format_advantages.csv",
    index=False
)

df_format_pairwise.to_csv(
    "equal_operand_format_pairwise_tests.csv",
    index=False
)

print("\nSaved:")
print("  equal_operand_format_advantages.csv")
print("  equal_operand_format_pairwise_tests.csv")
```

#### Recorded output

```text
FORMAT-LEVEL EQUAL-OPERAND ADVANTAGE
 T  digit+digit  digit+word  word+word
 4    +0.048633   +0.176977  +0.099550
 6    +0.016807   +0.053717  -0.289831
 8    +0.290962   +0.541647  +1.206146
10    +0.534628   +0.932795  +1.591604
12    +0.164717   +0.436063  +0.721015
14    -0.006201   +0.031726  -0.154784
16    +0.682146   +1.091484  +1.879133

Friedman repeated-measures test
------------------------------------------------------------------------------
Statistic : 5.428571
p-value   : 0.066252

Pairwise exact sign-flip comparisons
------------------------------------------------------------------------------
   format_a   format_b  mean_difference     p_raw    p_holm
digit+digit digit+word        -0.218959 +0.015625 +0.046875
digit+digit  word+word        -0.474448 +0.109375 +0.218750
 digit+word  word+word        -0.255489 +0.187500 +0.218750

Saved:
  equal_operand_format_advantages.csv
  equal_operand_format_pairwise_tests.csv
```


### Cell 134 — code (execution count: 96)

```python
# FORMAL OPERATOR COMPARISON
#
# The target token remains T for every operator.
#
# This is intentional:
# we are testing whether the equal-operand target preference
# persists when the operator/connector changes.
#
# This is NOT an arithmetic-correctness test for -, *, and.
#
# Statistical unit:
#   target level
#
# Global:
#   Friedman repeated-measures test
#
# Pairwise:
#   exact paired sign-flip


import numpy as np
import pandas as pd
import itertools
from scipy import stats

OPERATOR_FORMS_N5 = {
    "plus":  lambda a, b: f"{a} + {b} =",
    "minus": lambda a, b: f"{a} − {b} =",
    "times": lambda a, b: f"{a} × {b} =",
    "and":   lambda a, b: f"{a} and {b} =",
    "then":  lambda a, b: f"{a} then {b} =",
}

N5_TARGETS = list(range(4, 17, 2))


def score_operator(prompt, T):
    target_id = model.to_single_token(f" {T}")
    foil_minus = model.to_single_token(f" {T-1}")
    foil_plus = model.to_single_token(f" {T+1}")

    with torch.no_grad():
        logits = model(prompt)[0, -1]

    return (
        logits[target_id]
        - 0.5 * (
            logits[foil_minus]
            + logits[foil_plus]
        )
    ).item()


def operator_advantage(T, formatter):

    d = T // 2

    pairs = [
        (a, T-a)
        for a in range(1, 10)
        if 1 <= T-a <= 9
        and a != T-a
        and a < T-a
    ]

    double_score = score_operator(
        formatter(d, d),
        T
    )

    control_scores = []

    for a, b in pairs:

        score_ab = score_operator(
            formatter(a, b),
            T
        )

        score_ba = score_operator(
            formatter(b, a),
            T
        )

        control_scores.append(
            0.5 * (
                score_ab
                + score_ba
            )
        )

    return (
        double_score
        - np.mean(control_scores)
    )


operator_rows = []

for T in N5_TARGETS:

    row = {"T": T}

    for name, formatter in OPERATOR_FORMS_N5.items():

        row[name] = operator_advantage(
            T,
            formatter
        )

    operator_rows.append(row)


df_operator_adv = pd.DataFrame(
    operator_rows
)

print("OPERATOR-LEVEL EQUAL-OPERAND ADVANTAGE")

print(
    df_operator_adv.to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}"
    )
)

# ------------------------------------------------------------
# Friedman omnibus test
# ------------------------------------------------------------

operator_columns = list(
    OPERATOR_FORMS_N5.keys()
)

friedman_operator = stats.friedmanchisquare(
    *[
        df_operator_adv[name].to_numpy(dtype=float)
        for name in operator_columns
    ]
)

print("\nFriedman repeated-measures test")
print("-" * 78)
print(
    f"Statistic : "
    f"{friedman_operator.statistic:.6f}"
)
print(
    f"p-value   : "
    f"{friedman_operator.pvalue:.6f}"
)


# ------------------------------------------------------------
# Pairwise exact sign-flip
# ------------------------------------------------------------

def exact_signflip(values):
    values = np.asarray(values, dtype=float)

    observed = np.mean(values)

    null_distribution = np.array([
        np.mean(
            values
            * np.asarray(signs)
        )
        for signs in itertools.product(
            [-1, 1],
            repeat=len(values)
        )
    ])

    return np.mean(
        np.abs(null_distribution)
        >= abs(observed)
    )


pairwise_operator = []

for op_a, op_b in itertools.combinations(
    operator_columns,
    2
):

    diff = (
        df_operator_adv[op_a]
        - df_operator_adv[op_b]
    ).to_numpy(dtype=float)

    pairwise_operator.append({
        "operator_a": op_a,
        "operator_b": op_b,
        "mean_difference": diff.mean(),
        "p_raw": exact_signflip(diff),
    })


df_operator_pairwise = pd.DataFrame(
    pairwise_operator
)

df_operator_pairwise["p_holm"] = holm_adjust(
    df_operator_pairwise["p_raw"].to_numpy()
)

print("\nPairwise operator comparisons")
print("-" * 78)

print(
    df_operator_pairwise.to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}"
    )
)


# ------------------------------------------------------------
# Explicit addition-vs-all comparison
# ------------------------------------------------------------

addition_comparisons = []

for op in [
    "minus",
    "times",
    "and",
    "then",
]:

    diff = (
        df_operator_adv["plus"]
        - df_operator_adv[op]
    ).to_numpy(dtype=float)

    addition_comparisons.append({
        "comparison": f"plus_vs_{op}",
        "mean_difference_+_-_other": diff.mean(),
        "p_exact": exact_signflip(diff),
    })


df_plus_comparisons = pd.DataFrame(
    addition_comparisons
)

df_plus_comparisons["p_holm"] = holm_adjust(
    df_plus_comparisons["p_exact"].to_numpy()
)

print("\nAddition vs other operators")
print("-" * 78)

print(
    df_plus_comparisons.to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}"
    )
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df_operator_adv.to_csv(
    "equal_operand_operator_advantages.csv",
    index=False
)

df_operator_pairwise.to_csv(
    "equal_operand_operator_pairwise.csv",
    index=False
)

df_plus_comparisons.to_csv(
    "equal_operand_plus_vs_others.csv",
    index=False
)

print("\nSaved:")
print("  equal_operand_operator_advantages.csv")
print("  equal_operand_operator_pairwise.csv")
print("  equal_operand_plus_vs_others.csv")
```

#### Recorded output

```text
OPERATOR-LEVEL EQUAL-OPERAND ADVANTAGE
 T      plus     minus     times       and      then
 4 +0.048633 -0.150930 +0.121164 -0.048032 -0.153273
 6 +0.016807 -0.003621 -0.009409 +0.100839 +0.074206
 8 +0.290962 +0.179569 +0.090690 +0.321472 +0.343901
10 +0.534628 +0.552476 +0.376308 +0.533761 +0.703466
12 +0.164717 +0.176290 +0.129516 +0.275036 +0.226648
14 -0.006201 +0.009374 +0.054056 +0.060165 +0.152642
16 +0.682146 +0.382793 +0.297930 +0.744923 +0.922419

Friedman repeated-measures test
------------------------------------------------------------------------------
Statistic : 10.514286
p-value   : 0.032601

Pairwise operator comparisons
------------------------------------------------------------------------------
operator_a operator_b  mean_difference     p_raw    p_holm
      plus      minus        +0.083677 +0.187500 +1.000000
      plus      times        +0.095920 +0.187500 +1.000000
      plus        and        -0.036639 +0.234375 +1.000000
      plus       then        -0.076902 +0.171875 +1.000000
     minus      times        +0.012242 +0.843750 +1.000000
     minus        and        -0.120316 +0.031250 +0.312500
     minus       then        -0.160580 +0.031250 +0.312500
     times        and        -0.132558 +0.140625 +1.000000
     times       then        -0.172822 +0.140625 +1.000000
       and       then        -0.040264 +0.375000 +1.000000

Addition vs other operators
------------------------------------------------------------------------------
   comparison  mean_difference_+_-_other   p_exact    p_holm
plus_vs_minus                  +0.083677 +0.187500 +0.687500
plus_vs_times                  +0.095920 +0.187500 +0.687500
  plus_vs_and                  -0.036639 +0.234375 +0.687500
 plus_vs_then                  -0.076902 +0.171875 +0.687500

Saved:
  equal_operand_operator_advantages.csv
  equal_operand_operator_pairwise.csv
  equal_operand_plus_vs_others.csv
```


### Cell 135 — markdown

## Mechanistic analysis of the equal-operand effect


### Cell 136 — code (execution count: 97)

```python
import numpy as np, torch

W = {1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine"}
FORMATS = {"digit+digit": lambda a, b: f"{a} + {b} =",
           "digit+word":  lambda a, b: f"{a} + {W[b]} =",
           "word+word":   lambda a, b: f"{W[a]} + {W[b]} ="}

def score(prompt, T):
    ids = [model.to_single_token(f" {x}") for x in (T, T - 1, T + 1)]
    with torch.no_grad():
        lg = model(prompt)[0, -1]
    return (lg[ids[0]] - 0.5 * (lg[ids[1]] + lg[ids[2]])).item()

adv = {}
print(f"{'T':>3} {'ctl':>3} | " + " | ".join(f"{k:>12}" for k in FORMATS))
for T in range(4, 17, 2):
    d = T // 2
    pairs = [(a, T - a) for a in range(1, d) if 1 <= T - a <= 9]      # unique splits
    row = []
    for f in FORMATS.values():
        ctl = [0.5 * (score(f(a, b), T) + score(f(b, a), T)) for a, b in pairs]
        row.append(score(f(d, d), T) - np.mean(ctl))
    adv[T] = row
    print(f"{T:>3} {len(pairs):>3} | " + " | ".join(f"{x:>+12.3f}" for x in row))

for name, ts in [("targets 8/10/12/16", [8, 10, 12, 16]), ("targets 4/6/14", [4, 6, 14])]:
    m = np.mean([adv[t] for t in ts], axis=0)
    print(f"{name}: " + " | ".join(f"{k} {v:+.3f}" for k, v in zip(FORMATS, m)))
```

#### Recorded output

```text
  T ctl |  digit+digit |   digit+word |    word+word
  4   1 |       +0.049 |       +0.177 |       +0.100
  6   2 |       +0.017 |       +0.054 |       -0.290
  8   3 |       +0.291 |       +0.542 |       +1.206
 10   4 |       +0.535 |       +0.933 |       +1.592
 12   3 |       +0.165 |       +0.436 |       +0.721
 14   2 |       -0.006 |       +0.032 |       -0.155
 16   1 |       +0.682 |       +1.091 |       +1.879
targets 8/10/12/16: digit+digit +0.418 | digit+word +0.750 | word+word +1.349
targets 4/6/14: digit+digit +0.020 | digit+word +0.087 | word+word -0.115
```


### Cell 137 — code (execution count: 98)

```python
for f_name, f in FORMATS.items():
    dbl = score(f(d, d), T)
    ctl = np.mean([0.5 * (score(f(a, b), T) + score(f(b, a), T)) for a, b in pairs])
    print(f"T={T:>2} {f_name:>12}: double {dbl:+.3f} | controls {ctl:+.3f}")
```

#### Recorded output

```text
T=16  digit+digit: double +0.719 | controls +0.037
T=16   digit+word: double +1.255 | controls +0.164
T=16    word+word: double +1.950 | controls +0.071
```


### Cell 138 — code (execution count: 99)

```python
for T in (8, 10, 12):
    d = T // 2
    pairs = [(a, T - a) for a in range(1, d) if 1 <= T - a <= 9]
    for name, f in FORMATS.items():
        dbl = score(f(d, d), T)
        ctl = np.mean([0.5 * (score(f(a, b), T) + score(f(b, a), T)) for a, b in pairs])
        print(f"T={T:>2} {name:>12}: double {dbl:+.3f} | controls {ctl:+.3f}")
```

#### Recorded output

```text
T= 8  digit+digit: double +0.606 | controls +0.315
T= 8   digit+word: double +0.996 | controls +0.454
T= 8    word+word: double +1.542 | controls +0.336
T=10  digit+digit: double +0.647 | controls +0.113
T=10   digit+word: double +1.329 | controls +0.396
T=10    word+word: double +2.057 | controls +0.465
T=12  digit+digit: double +0.401 | controls +0.236
T=12   digit+word: double +0.935 | controls +0.499
T=12    word+word: double +1.180 | controls +0.459
```


### Cell 139 — code (execution count: 100)

```python
# reuses score(), FORMATS from the earlier cells
for name, f in FORMATS.items():
    resid, advs = [], []
    for T in (8, 10, 12, 16):
        d = T // 2
        pairs = [(a, T - a) for a in range(1, d) if 1 <= T - a <= 9]
        ctl = np.array([score(f(a, b), T) for a, b in pairs] + [score(f(b, a), T) for a, b in pairs])
        resid += list(ctl - ctl.mean())
        advs.append(score(f(d, d), T) - ctl.mean())
    sd = np.sqrt(np.sum(np.square(resid)) / (len(resid) - 4))      # pooled SD, 18 df
    print(f"{name:>12}: pooled control SD {sd:.3f} | mean adv {np.mean(advs):+.3f} | adv/SD {np.mean(advs)/sd:+.2f}")
    # ── Save per-prompt scores for repo ──
import pandas as pd

rows_export = []
for T in range(4, 17, 2):
    d = T // 2
    dbl = score(f"{d} + {d} =", T)
    ctl = [(a, T - a) for a in range(1, 10) if 1 <= T - a <= 9 and a != T - a]

    # Save all individual control prompt scores
    for a, b in ctl:
        rows_export.append({"T": T, "type": "control", "prompt": f"{a} + {b} =", "score": score(f"{a} + {b} =", T)})

    # Save the double prompt score
    rows_export.append({"T": T, "type": "double", "prompt": f"{d} + {d} =", "score": dbl})

df_doubles = pd.DataFrame(rows_export)
df_doubles.to_csv("doubles_scan_scores.csv", index=False)
print("Saved doubles_scan_scores.csv (per-prompt breakdown)")
```

#### Recorded output

```text
 digit+digit: pooled control SD 0.080 | mean adv +0.418 | adv/SD +5.22
  digit+word: pooled control SD 0.171 | mean adv +0.750 | adv/SD +4.40
   word+word: pooled control SD 0.273 | mean adv +1.349 | adv/SD +4.93
Saved doubles_scan_scores.csv (per-prompt breakdown)
```


### Cell 140 — code (execution count: 101)

```python
import numpy as np

FORMS = {"plus": "{a} + {b} =", "times": "{a} × {b} =", "minus": "{a} − {b} =",
         "and": "{a} and {b} =", "then": "{a} then {b} ="}

def run_form(f, targets):
    resid, advs = [], {}
    for T in targets:
        d = T // 2
        pairs = [(a, T - a) for a in range(1, d) if 1 <= T - a <= 9]
        ctl = np.array([score(f.format(a=a, b=b), T) for a, b in pairs] +
                       [score(f.format(a=b, b=a), T) for a, b in pairs])
        resid += list(ctl - ctl.mean())
        advs[T] = score(f.format(a=d, b=d), T) - ctl.mean()
    sd = np.sqrt(np.sum(np.square(resid)) / (len(resid) - len(targets)))
    return advs, sd

print(f"{'form':>6} | tokens(4,4) | {'pos adv':>8} {'SD':>6} {'adv/SD':>7} | {'null adv':>8} {'SD':>6} {'adv/SD':>7}")
for name, f in FORMS.items():
    ap, sp = run_form(f, (8, 10, 12, 16))
    an, sn = run_form(f, (6, 14))
    mp, mn = np.mean(list(ap.values())), np.mean(list(an.values()))
    ntok = model.to_tokens(f.format(a=4, b=4)).shape[1]
    print(f"{name:>6} | {ntok:>11} | {mp:+8.3f} {sp:6.3f} {mp/sp:+7.2f} | {mn:+8.3f} {sn:6.3f} {mn/sn:+7.2f}")
```

#### Recorded output

```text
  form | tokens(4,4) |  pos adv     SD  adv/SD | null adv     SD  adv/SD
  plus |           5 |   +0.418  0.080   +5.22 |   +0.005  0.099   +0.05
 times |           5 |   +0.224  0.092   +2.43 |   +0.022  0.113   +0.20
 minus |           5 |   +0.323  0.052   +6.15 |   +0.003  0.056   +0.05
   and |           5 |   +0.469  0.095   +4.95 |   +0.081  0.094   +0.86
  then |           5 |   +0.549  0.120   +4.56 |   +0.113  0.082   +1.38
```


### Cell 141 — markdown


"


### Cell 142 — markdown

### Legacy exploratory corpus analysis — do not rerun as confirmatory evidence

**Status:** Historical output preserved before runtime reset.

**Original method:** Corpus n-gram queries and target filtering using `r[2] + r[3] >= 20`.

**Historical output:**
Corpus Exact Matches:
             5 + 5 =: 9,848
   five plus five is: 159
             6 + 4 =: 5,179
    six plus four is: 53
             7 + 3 =: 3,500
 seven plus three is: 183

**Interpretation:** Exploratory only. The eligibility threshold has not been independently justified, so these results must not be cited as confirmatory evidence.

**Provenance:** Output copied from the original notebook execution. Not independently revalidated.


### Cell 143 — code (execution count: 102)

```python
import time, requests, numpy as np
from scipy.stats import spearmanr

URL   = "https://api.infini-gram.io/"
INDEX = "v4_dolma-v1_7_llama"
P     = ""   # ← CORRECTED: empty string. API adds '▁' automatically.

def count(q, tries=3):
    err = None
    for i in range(tries):
        try:
            r = requests.post(
                URL,
                json={"index": INDEX, "query_type": "count", "query": q},
                timeout=30
            ).json()
            if "error" in r:
                raise RuntimeError(r["error"])
            time.sleep(0.15)
            return r["count"], r.get("tokens")
        except Exception as e:
            err = e
            time.sleep(1.0 * (i + 1))
    raise RuntimeError(f"{q!r}: {err}")  # fail loudly

# Verify tokenisation first
print("Token check:", count(P + "5 + 5 = 10")[1])

ADV = {4: +0.049, 6: +0.017, 8: +0.291, 10: +0.535,
       12: +0.165, 14: -0.006, 16: +0.682}

rows = []   # ← Clear rows before running
for T, adv in ADV.items():
    d     = T // 2
    pairs = [(a, T - a) for a in range(1, d) if 1 <= T - a <= 9]
    J  = lambda a, b: count(f"{P}{a} + {b} = {T}")[0]
    C  = lambda a, b: count(f"{P}{a} + {b} =")[0]

    dj, dc = J(d, d), C(d, d)
    cj = np.mean([J(a, b) for a, b in pairs] + [J(b, a) for a, b in pairs])
    cc = np.mean([C(a, b) for a, b in pairs] + [C(b, a) for a, b in pairs])

    lj = np.log10((dj + 1) / (cj + 1))
    lc = np.log10(((dj + 1) / (dc + 1)) / ((cj + 1) / (cc + 1)))
    rows.append((T, adv, dj, cj, lj, lc))

    print(f"T={T:>2}: dbl_joint={dj:>6}  ctrl_joint={cj:>8.1f}  "
          f"log_ratio_joint={lj:+.3f}  log_ratio_cond={lc:+.3f}")
```

#### Recorded output

```text
Token check: ['5', '▁+', '▁', '5', '▁=', '▁', '1', '0']
T= 4: dbl_joint= 28489  ctrl_joint=  4305.0  log_ratio_joint=+0.821  log_ratio_cond=+0.040
T= 6: dbl_joint=  5076  ctrl_joint=  2506.8  log_ratio_joint=+0.306  log_ratio_cond=+0.133
T= 8: dbl_joint=  3191  ctrl_joint=  1907.2  log_ratio_joint=+0.223  log_ratio_cond=-0.031
T=10: dbl_joint=  3812  ctrl_joint=  1490.9  log_ratio_joint=+0.408  log_ratio_cond=+0.054
T=12: dbl_joint=  1547  ctrl_joint=  1151.2  log_ratio_joint=+0.128  log_ratio_cond=-0.031
T=14: dbl_joint=   854  ctrl_joint=   696.5  log_ratio_joint=+0.088  log_ratio_cond=+0.080
T=16: dbl_joint=  1083  ctrl_joint=   420.5  log_ratio_joint=+0.410  log_ratio_cond=+0.123
```


### Cell 144 — code (execution count: 103)

```python
import numpy as np
import pandas as pd
import torch

# Ensure clean, fresh definition of mappings
WORDS = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}

FORMATS = {
    'digit+digit': lambda a, b: f"{a} + {b} =",
    'digit+word':  lambda a, b: f"{a} + {WORDS[b]} =",
    'word+word':   lambda a, b: f"{WORDS[a]} + {WORDS[b]} ="
}

def score_evaluation(prompt, T):
    # standard target, foil-1, foil+1
    target_id = model.to_single_token(f" {T}")
    foil_minus = model.to_single_token(f" {T-1}")
    foil_plus = model.to_single_token(f" {T+1}")

    with torch.no_grad():
        logits = model(prompt)[0, -1]
    return (logits[target_id] - 0.5 * (logits[foil_minus] + logits[foil_plus])).item()

# Audit sweep to regenerate control and double values cleanly
records = []
for T in range(4, 17, 2):
    d = T // 2
    # All unique valid single-digit summand combinations that sum to T (a != b)
    pairs = [(a, T - a) for a in range(1, 10) if 1 <= T - a <= 9 and a < T - a]

    for fmt_name, prompt_fn in FORMATS.items():
        # 1. Evaluate double
        double_prompt = prompt_fn(d, d)
        double_score = score_evaluation(double_prompt, T)

        # 2. Evaluate all valid controls symmetric orders
        control_scores = []
        for a, b in pairs:
            s1 = score_evaluation(prompt_fn(a, b), T)
            s2 = score_evaluation(prompt_fn(b, a), T)
            control_scores.append(0.5 * (s1 + s2))

        mean_control = np.mean(control_scores) if control_scores else 0.0
        advantage = double_score - mean_control

        records.append({
            "Target": T,
            "Format": fmt_name,
            "Double_Prompt": double_prompt,
            "Double_Score": double_score,
            "Mean_Control_Score": mean_control,
            "Advantage": advantage
        })

df_variance_scaling = pd.DataFrame(records)

print("=== CLEAN GENERATED FORMAT VARIANCE DATA ===")
display(df_variance_scaling)

# Save to disk to satisfy the integrity audit source requirements
df_variance_scaling.to_csv("variance_scaling.csv", index=False)
print("Saved accurate, non-duplicated control evaluation to 'variance_scaling.csv'.")
```

#### Recorded output

```text
=== CLEAN GENERATED FORMAT VARIANCE DATA ===
```

```text
    Target       Format    Double_Prompt  Double_Score  Mean_Control_Score  \
0        4  digit+digit          2 + 2 =      0.251480            0.202847
1        4   digit+word        2 + two =      0.532180            0.355203
2        4    word+word      two + two =      0.465640            0.366090
3        6  digit+digit          3 + 3 =      0.230024            0.213217
4        6   digit+word      3 + three =      0.237948            0.184231
5        6    word+word  three + three =     -0.082720            0.207111
6        8  digit+digit          4 + 4 =      0.605705            0.314744
7        8   digit+word       4 + four =      0.995697            0.454050
8        8    word+word    four + four =      1.542169            0.336023
9       10  digit+digit          5 + 5 =      0.647227            0.112599
10      10   digit+word       5 + five =      1.328588            0.395792
11      10    word+word    five + five =      2.056720            0.465116
12      12  digit+digit          6 + 6 =      0.401203            0.236486
13      12   digit+word        6 + six =      0.934820            0.498757
14      12    word+word      six + six =      1.179948            0.458933
15      14  digit+digit          7 + 7 =     -0.238291           -0.232090
16      14   digit+word      7 + seven =     -0.161430           -0.193156
17      14    word+word  seven + seven =     -0.315639           -0.160854
18      16  digit+digit          8 + 8 =      0.719002            0.036856
19      16   digit+word      8 + eight =      1.255064            0.163580
20      16    word+word  eight + eight =      1.950054            0.070921

    Advantage
0    0.048633
1    0.176977
2    0.099550
3    0.016807
4    0.053717
5   -0.289831
6    0.290962
7    0.541647
8    1.206146
9    0.534628
10   0.932795
11   1.591604
12   0.164717
13   0.436063
14   0.721015
15  -0.006201
16   0.031726
17  -0.154784
18   0.682146
19   1.091484
20   1.879133
```

```text
Saved accurate, non-duplicated control evaluation to 'variance_scaling.csv'.
```


### Cell 145 — code (execution count: 104)

```python
# CANONICAL EQUAL-OPERAND CIRCUIT DATASET


import pandas as pd
import numpy as np

DISCOVERY_TARGETS = [4, 6, 10, 12, 16]

# These were already examined in the earlier doubles analysis.
# They are not a blind holdout.
TRANSFER_TARGETS = [8, 14]
HOLDOUT_TARGETS = TRANSFER_TARGETS  # compatibility with unedited legacy cells
ALL_CIRCUIT_TARGETS = DISCOVERY_TARGETS + TRANSFER_TARGETS

def build_circuit_dataset():

    rows = []

    for T in ALL_CIRCUIT_TARGETS:

        split = (
            "discovery"
            if T in DISCOVERY_TARGETS
            else "exploratory_transfer"
        )

        d = T // 2

        # Matched corrupt prompt for activation patching.
        corrupt_a = d - 1
        corrupt_b = d + 1

        if not (
            1 <= corrupt_a <= 9
            and 1 <= corrupt_b <= 9
        ):
            raise ValueError(
                f"Invalid matched corruption for T={T}"
            )

        matched_corrupt = (
            f"{corrupt_a} + {corrupt_b} ="
        )

        # ----------------------------------------------------
        # Double prompt
        # ----------------------------------------------------

        rows.append({
            "target": T,
            "a": d,
            "b": d,
            "condition": "double",
            "is_double": True,
            "split": split,
            "operator": "+",
            "prompt": f"{d} + {d} =",
            "matched_corrupt_prompt": matched_corrupt,
        })

        # ----------------------------------------------------
        # All ordered non-double controls
        # ----------------------------------------------------

        for a in range(1, 10):

            b = T - a

            if (
                1 <= b <= 9
                and a != b
            ):

                rows.append({
                    "target": T,
                    "a": a,
                    "b": b,
                    "condition": "control",
                    "is_double": False,
                    "split": split,
                    "operator": "+",
                    "prompt": f"{a} + {b} =",
                    "matched_corrupt_prompt": matched_corrupt,
                })

    df = pd.DataFrame(rows)

    # Safety checks
    assert df["target"].isin(
        ALL_CIRCUIT_TARGETS
    ).all()

    assert (
        df[
            df["condition"] == "double"
        ]
        .groupby("target")
        .size()
        .eq(1)
        .all()
    )

    print("=" * 78)
    print("CANONICAL CIRCUIT DATASET")
    print("=" * 78)

    print(
        df.groupby(
            ["split", "target", "condition"]
        ).size().to_string()
    )

    print("\nTotal rows:", len(df))

    df.to_csv(
        "equal_operand_circuit_dataset.csv",
        index=False
    )

    print(
        "\nSaved: "
        "equal_operand_circuit_dataset.csv"
    )

    return df


df_circuit = build_circuit_dataset()
```

#### Recorded output

```text
==============================================================================
CANONICAL CIRCUIT DATASET
==============================================================================
split                 target  condition
discovery             4       control      2
                              double       1
                      6       control      4
                              double       1
                      10      control      8
                              double       1
                      12      control      6
                              double       1
                      16      control      2
                              double       1
exploratory_transfer  8       control      6
                              double       1
                      14      control      4
                              double       1

Total rows: 39

Saved: equal_operand_circuit_dataset.csv
```


### Cell 146 — code (execution count: 105)

```python
# ============================================================
# TOKENIZATION CONTRACT AND CANONICAL SCORER
# ============================================================

import numpy as np
import pandas as pd
import torch


def encode_prompt(prompt):
    """Encode one prompt with an explicit BOS token."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    tokens = model.to_tokens(prompt, prepend_bos=True)
    tokens = tokens.to(model.W_E.device)
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError(f"Unexpected token tensor shape: {tuple(tokens.shape)}")
    return tokens


def answer_token_id(number):
    """ID of the exact leading-space answer token; fail if it is multi-token."""
    text = f" {int(number)}"
    ids = model.to_tokens(text, prepend_bos=False)
    if ids.shape[1] != 1:
        pieces = model.to_str_tokens(text, prepend_bos=False)
        raise ValueError(f"{text!r} is not a single token: {pieces}")
    return int(ids[0, 0].item())


def score_from_last_logits(last_logits, target):
    """Target-versus-neighbor contrast, using true vocabulary IDs."""
    target_id = answer_token_id(target)
    minus_id = answer_token_id(target - 1)
    plus_id = answer_token_id(target + 1)
    return last_logits[target_id] - 0.5 * (last_logits[minus_id] + last_logits[plus_id])


def primary_score(prompt, target):
    """Canonical score; same BOS policy and metric in every downstream cell."""
    tokens = encode_prompt(prompt)
    with torch.no_grad():
        logits = model(tokens, return_type="logits")
    return float(score_from_last_logits(logits[0, -1], int(target)).item())


if "df_circuit" not in globals():
    raise RuntimeError("Run the canonical circuit dataset cell first.")

rows = []
for _, row in df_circuit.iterrows():
    prompt = row["prompt"]
    target = int(row["target"])
    tokens = encode_prompt(prompt)
    token_strings = model.to_str_tokens(tokens[0])
    token_ids = tokens[0].detach().cpu().tolist()

    rows.append({
        "split": row["split"],
        "target": target,
        "condition": row["condition"],
        "prompt": prompt,
        "n_prompt_tokens_with_bos": int(tokens.shape[1]),
        "tokens": repr(token_strings),
        "token_ids": repr(token_ids),
        "target_text": f" {target}",
        "target_id": answer_token_id(target),
        "foil_minus_text": f" {target - 1}",
        "foil_minus_id": answer_token_id(target - 1),
        "foil_plus_text": f" {target + 1}",
        "foil_plus_id": answer_token_id(target + 1),
    })

df_token_audit = pd.DataFrame(rows)
print("TOKENIZATION AUDIT — BOS INCLUDED EXPLICITLY")
display(df_token_audit)

lengths = sorted(df_token_audit["n_prompt_tokens_with_bos"].unique().tolist())
print("Unique sequence lengths:", lengths)
print("Token ID for ' 4':", answer_token_id(4))

if len(lengths) != 1:
    raise RuntimeError("Prompt sequence lengths differ; do not run positional analyses.")

if answer_token_id(4) != 604:
    print("WARNING: tokenizer reports a different ID for ' 4' than the audit report's 604.")
    print("Check the loaded GPT-2 model/tokenizer before proceeding.")
    raise RuntimeError("Token-ID spot check failed.")

df_token_audit.to_csv("equal_operand_tokenization_audit.csv", index=False)
print("Saved equal_operand_tokenization_audit.csv")
```

#### Recorded output

```text
TOKENIZATION AUDIT — BOS INCLUDED EXPLICITLY
```

```text
                   split  target condition   prompt  n_prompt_tokens_with_bos  \
0              discovery       4    double  2 + 2 =                         5
1              discovery       4   control  1 + 3 =                         5
2              discovery       4   control  3 + 1 =                         5
3              discovery       6    double  3 + 3 =                         5
4              discovery       6   control  1 + 5 =                         5
5              discovery       6   control  2 + 4 =                         5
6              discovery       6   control  4 + 2 =                         5
7              discovery       6   control  5 + 1 =                         5
8              discovery      10    double  5 + 5 =                         5
9              discovery      10   control  1 + 9 =                         5
10             discovery      10   control  2 + 8 =                         5
11             discovery      10   control  3 + 7 =                         5
12             discovery      10   control  4 + 6 =                         5
13             discovery      10   control  6 + 4 =                         5
14             discovery      10   control  7 + 3 =                         5
15             discovery      10   control  8 + 2 =                         5
16             discovery      10   control  9 + 1 =                         5
17             discovery      12    double  6 + 6 =                         5
18             discovery      12   control  3 + 9 =                         5
19             discovery      12   control  4 + 8 =                         5
20             discovery      12   control  5 + 7 =                         5
21             discovery      12   control  7 + 5 =                         5
22             discovery      12   control  8 + 4 =                         5
23             discovery      12   control  9 + 3 =                         5
24             discovery      16    double  8 + 8 =                         5
25             discovery      16   control  7 + 9 =                         5
26             discovery      16   control  9 + 7 =                         5
27  exploratory_transfer       8    double  4 + 4 =                         5
28  exploratory_transfer       8   control  1 + 7 =                         5
29  exploratory_transfer       8   control  2 + 6 =                         5
30  exploratory_transfer       8   control  3 + 5 =                         5
31  exploratory_transfer       8   control  5 + 3 =                         5
32  exploratory_transfer       8   control  6 + 2 =                         5
33  exploratory_transfer       8   control  7 + 1 =                         5
34  exploratory_transfer      14    double  7 + 7 =                         5
35  exploratory_transfer      14   control  5 + 9 =                         5
36  exploratory_transfer      14   control  6 + 8 =                         5
37  exploratory_transfer      14   control  8 + 6 =                         5
38  exploratory_transfer      14   control  9 + 5 =                         5

                                      tokens                    token_ids  \
0   ['<|endoftext|>', '2', ' +', ' 2', ' =']  [50256, 17, 1343, 362, 796]
1   ['<|endoftext|>', '1', ' +', ' 3', ' =']  [50256, 16, 1343, 513, 796]
2   ['<|endoftext|>', '3', ' +', ' 1', ' =']  [50256, 18, 1343, 352, 796]
3   ['<|endoftext|>', '3', ' +', ' 3', ' =']  [50256, 18, 1343, 513, 796]
4   ['<|endoftext|>', '1', ' +', ' 5', ' =']  [50256, 16, 1343, 642, 796]
5   ['<|endoftext|>', '2', ' +', ' 4', ' =']  [50256, 17, 1343, 604, 796]
6   ['<|endoftext|>', '4', ' +', ' 2', ' =']  [50256, 19, 1343, 362, 796]
7   ['<|endoftext|>', '5', ' +', ' 1', ' =']  [50256, 20, 1343, 352, 796]
8   ['<|endoftext|>', '5', ' +', ' 5', ' =']  [50256, 20, 1343, 642, 796]
9   ['<|endoftext|>', '1', ' +', ' 9', ' =']  [50256, 16, 1343, 860, 796]
10  ['<|endoftext|>', '2', ' +', ' 8', ' =']  [50256, 17, 1343, 807, 796]
11  ['<|endoftext|>', '3', ' +', ' 7', ' =']  [50256, 18, 1343, 767, 796]
12  ['<|endoftext|>', '4', ' +', ' 6', ' =']  [50256, 19, 1343, 718, 796]
13  ['<|endoftext|>', '6', ' +', ' 4', ' =']  [50256, 21, 1343, 604, 796]
14  ['<|endoftext|>', '7', ' +', ' 3', ' =']  [50256, 22, 1343, 513, 796]
15  ['<|endoftext|>', '8', ' +', ' 2', ' =']  [50256, 23, 1343, 362, 796]
16  ['<|endoftext|>', '9', ' +', ' 1', ' =']  [50256, 24, 1343, 352, 796]
17  ['<|endoftext|>', '6', ' +', ' 6', ' =']  [50256, 21, 1343, 718, 796]
18  ['<|endoftext|>', '3', ' +', ' 9', ' =']  [50256, 18, 1343, 860, 796]
19  ['<|endoftext|>', '4', ' +', ' 8', ' =']  [50256, 19, 1343, 807, 796]
20  ['<|endoftext|>', '5', ' +', ' 7', ' =']  [50256, 20, 1343, 767, 796]
21  ['<|endoftext|>', '7', ' +', ' 5', ' =']  [50256, 22, 1343, 642, 796]
22  ['<|endoftext|>', '8', ' +', ' 4', ' =']  [50256, 23, 1343, 604, 796]
23  ['<|endoftext|>', '9', ' +', ' 3', ' =']  [50256, 24, 1343, 513, 796]
24  ['<|endoftext|>', '8', ' +', ' 8', ' =']  [50256, 23, 1343, 807, 796]
25  ['<|endoftext|>', '7', ' +', ' 9', ' =']  [50256, 22, 1343, 860, 796]
26  ['<|endoftext|>', '9', ' +', ' 7', ' =']  [50256, 24, 1343, 767, 796]
27  ['<|endoftext|>', '4', ' +', ' 4', ' =']  [50256, 19, 1343, 604, 796]
28  ['<|endoftext|>', '1', ' +', ' 7', ' =']  [50256, 16, 1343, 767, 796]
29  ['<|endoftext|>', '2', ' +', ' 6', ' =']  [50256, 17, 1343, 718, 796]
30  ['<|endoftext|>', '3', ' +', ' 5', ' =']  [50256, 18, 1343, 642, 796]
31  ['<|endoftext|>', '5', ' +', ' 3', ' =']  [50256, 20, 1343, 513, 796]
32  ['<|endoftext|>', '6', ' +', ' 2', ' =']  [50256, 21, 1343, 362, 796]
33  ['<|endoftext|>', '7', ' +', ' 1', ' =']  [50256, 22, 1343, 352, 796]
34  ['<|endoftext|>', '7', ' +', ' 7', ' =']  [50256, 22, 1343, 767, 796]
35  ['<|endoftext|>', '5', ' +', ' 9', ' =']  [50256, 20, 1343, 860, 796]
36  ['<|endoftext|>', '6', ' +', ' 8', ' =']  [50256, 21, 1343, 807, 796]
37  ['<|endoftext|>', '8', ' +', ' 6', ' =']  [50256, 23, 1343, 718, 796]
38  ['<|endoftext|>', '9', ' +', ' 5', ' =']  [50256, 24, 1343, 642, 796]

   target_text  target_id foil_minus_text  foil_minus_id foil_plus_text  \
0            4        604               3            513              5
1            4        604               3            513              5
2            4        604               3            513              5
3            6        718               5            642              7
4            6        718               5            642              7
5            6        718               5            642              7
6            6        718               5            642              7
7            6        718               5            642              7
8           10        838               9            860             11
9           10        838               9            860             11
10          10        838               9            860             11
11          10        838               9            860             11
12          10        838               9            860             11
13          10        838               9            860             11
14          10        838               9            860             11
15          10        838               9            860             11
16          10        838               9            860             11
17          12       1105              11           1367             13
18          12       1105              11           1367             13
19          12       1105              11           1367             13
20          12       1105              11           1367             13
21          12       1105              11           1367             13
22          12       1105              11           1367             13
23          12       1105              11           1367             13
24          16       1467              15           1315             17
25          16       1467              15           1315             17
26          16       1467              15           1315             17
27           8        807               7            767              9
28           8        807               7            767              9
29           8        807               7            767              9
30           8        807               7            767              9
31           8        807               7            767              9
32           8        807               7            767              9
33           8        807               7            767              9
34          14       1478              13           1511             15
35          14       1478              13           1511             15
36          14       1478              13           1511             15
37          14       1478              13           1511             15
38          14       1478              13           1511             15

    foil_plus_id
0            642
1            642
2            642
3            767
4            767
5            767
6            767
7            767
8           1367
9           1367
10          1367
11          1367
12          1367
13          1367
14          1367
15          1367
16          1367
17          1511
18          1511
19          1511
20          1511
21          1511
22          1511
23          1511
24          1596
25          1596
26          1596
27           860
28           860
29           860
30           860
31           860
32           860
33           860
34          1315
35          1315
36          1315
37          1315
38          1315
```

```text
Unique sequence lengths: [5]
Token ID for ' 4': 604
Saved equal_operand_tokenization_audit.csv
```


### Cell 147 — code (execution count: 106)

```python
# ============================================================
# CLEAN BEHAVIORAL MAP
# This table is the score reference for all later interventions.
# ============================================================

import pandas as pd
import torch

model.reset_hooks()
rows = []

for _, row in df_circuit.iterrows():
    prompt = row["prompt"]
    target = int(row["target"])
    tokens = encode_prompt(prompt)

    with torch.no_grad():
        logits = model(tokens, return_type="logits")[0, -1]

    target_id = answer_token_id(target)
    minus_id = answer_token_id(target - 1)
    plus_id = answer_token_id(target + 1)
    score = float(score_from_last_logits(logits, target).item())
    rank = int((torch.argsort(logits, descending=True) == target_id).nonzero().item()) + 1

    rows.append({
        **row.to_dict(),
        "target_score": score,
        "target_logit": float(logits[target_id].item()),
        "foil_minus_logit": float(logits[minus_id].item()),
        "foil_plus_logit": float(logits[plus_id].item()),
        "target_rank": rank,
        "top_token": model.to_string(int(logits.argmax().item())),
        "sequence_length_with_bos": int(tokens.shape[1]),
    })

df_behavior = pd.DataFrame(rows)
display(df_behavior)
df_behavior.to_csv("equal_operand_behavioral_map.csv", index=False)
print("Saved equal_operand_behavioral_map.csv")
```

#### Recorded output

```text
    target  a  b condition  is_double                 split operator   prompt  \
0        4  2  2    double       True             discovery        +  2 + 2 =
1        4  1  3   control      False             discovery        +  1 + 3 =
2        4  3  1   control      False             discovery        +  3 + 1 =
3        6  3  3    double       True             discovery        +  3 + 3 =
4        6  1  5   control      False             discovery        +  1 + 5 =
5        6  2  4   control      False             discovery        +  2 + 4 =
6        6  4  2   control      False             discovery        +  4 + 2 =
7        6  5  1   control      False             discovery        +  5 + 1 =
8       10  5  5    double       True             discovery        +  5 + 5 =
9       10  1  9   control      False             discovery        +  1 + 9 =
10      10  2  8   control      False             discovery        +  2 + 8 =
11      10  3  7   control      False             discovery        +  3 + 7 =
12      10  4  6   control      False             discovery        +  4 + 6 =
13      10  6  4   control      False             discovery        +  6 + 4 =
14      10  7  3   control      False             discovery        +  7 + 3 =
15      10  8  2   control      False             discovery        +  8 + 2 =
16      10  9  1   control      False             discovery        +  9 + 1 =
17      12  6  6    double       True             discovery        +  6 + 6 =
18      12  3  9   control      False             discovery        +  3 + 9 =
19      12  4  8   control      False             discovery        +  4 + 8 =
20      12  5  7   control      False             discovery        +  5 + 7 =
21      12  7  5   control      False             discovery        +  7 + 5 =
22      12  8  4   control      False             discovery        +  8 + 4 =
23      12  9  3   control      False             discovery        +  9 + 3 =
24      16  8  8    double       True             discovery        +  8 + 8 =
25      16  7  9   control      False             discovery        +  7 + 9 =
26      16  9  7   control      False             discovery        +  9 + 7 =
27       8  4  4    double       True  exploratory_transfer        +  4 + 4 =
28       8  1  7   control      False  exploratory_transfer        +  1 + 7 =
29       8  2  6   control      False  exploratory_transfer        +  2 + 6 =
30       8  3  5   control      False  exploratory_transfer        +  3 + 5 =
31       8  5  3   control      False  exploratory_transfer        +  5 + 3 =
32       8  6  2   control      False  exploratory_transfer        +  6 + 2 =
33       8  7  1   control      False  exploratory_transfer        +  7 + 1 =
34      14  7  7    double       True  exploratory_transfer        +  7 + 7 =
35      14  5  9   control      False  exploratory_transfer        +  5 + 9 =
36      14  6  8   control      False  exploratory_transfer        +  6 + 8 =
37      14  8  6   control      False  exploratory_transfer        +  8 + 6 =
38      14  9  5   control      False  exploratory_transfer        +  9 + 5 =

   matched_corrupt_prompt  target_score  target_logit  foil_minus_logit  \
0                 1 + 3 =      0.251480     13.300371         13.458366
1                 1 + 3 =      0.321500     13.991928         13.939170
2                 1 + 3 =      0.084194     12.458035         12.651694
3                 2 + 4 =      0.230024     12.779467         12.869018
4                 2 + 4 =      0.160547     13.565547         13.661588
5                 2 + 4 =      0.366152     13.627018         13.509815
6                 2 + 4 =      0.263328     13.015692         13.014518
7                 2 + 4 =      0.062842     11.968349         12.330737
8                 4 + 6 =      0.647227     12.190495         11.788294
9                 4 + 6 =      0.120125     12.872520         12.988550
10                4 + 6 =      0.222570     12.793678         12.851210
11                4 + 6 =      0.092882     12.607094         12.773340
12                4 + 6 =      0.110022     12.566038         12.681242
13                4 + 6 =      0.036099     12.681029         12.902577
14                4 + 6 =      0.090355     12.437704         12.622171
15                4 + 6 =      0.153770     12.384809         12.564263
16                4 + 6 =      0.074967     11.766494         12.055997
17                5 + 7 =      0.401203     11.689165         11.491422
18                5 + 7 =      0.188457     11.996735         11.953621
19                5 + 7 =      0.358138     12.538576         12.325848
20                5 + 7 =      0.222425     12.126093         12.073707
21                5 + 7 =      0.161645     12.034288         12.057436
22                5 + 7 =      0.362541     12.616251         12.399569
23                5 + 7 =      0.125708     12.345938         12.377241
24                7 + 9 =      0.719002     11.901329         11.472034
25                7 + 9 =      0.081398     11.385910         11.577675
26                7 + 9 =     -0.007687     12.224802         12.397388
27                3 + 5 =      0.605705     13.062444         12.627987
28                3 + 5 =      0.262718     13.530373         13.396792
29                3 + 5 =      0.428913     13.448914         13.198854
30                3 + 5 =      0.411659     13.031895         12.852909
31                3 + 5 =      0.304052     12.999474         12.887085
32                3 + 5 =      0.274593     12.840955         12.749542
33                3 + 5 =      0.206526     12.036761         12.126982
34                6 + 8 =     -0.238291     11.023294         11.191963
35                6 + 8 =     -0.297479     11.329855         11.515271
36                6 + 8 =     -0.178841     11.677770         11.809566
37                6 + 8 =     -0.214467     11.805688         11.973024
38                6 + 8 =     -0.237575     11.599623         11.820952

    foil_plus_logit  target_rank top_token  sequence_length_with_bos
0         12.639416            4         2                         5
1         13.401687            1         4                         5
2         12.095987            4         1                         5
3         12.229867            6         3                         5
4         13.148412            4         1                         5
5         13.011918            5         4                         5
6         12.490211            5         2                         5
7         11.480274            7         1                         5
8         11.298241            9         5                         5
9         12.516241            7         1                         5
10        12.291006           11         2                         5
11        12.255083           11         4                         5
12        12.230791           11         4                         5
13        12.387283           11         6                         5
14        12.072527           11         1                         5
15        11.897814           11         2                         5
16        11.327057           11         1                         5
17        11.084501           12         6                         5
18        11.662933           12         1                         5
19        12.035028           12         4                         5
20        11.733627           12         6                         5
21        11.687851           13         6                         5
22        12.107852           12         8                         5
23        12.063220           13         1                         5
24        10.892620           13         8                         5
25        11.031349           17         8                         5
26        12.067589           17         6                         5
27        12.285490            7         4                         5
28        13.138515            2         1                         5
29        12.841146            6         2                         5
30        12.387563            7         1                         5
31        12.503757            7         2                         5
32        12.383181            8         2                         5
33        11.533489            9         1                         5
34        11.331208           17         7                         5
35        11.739396           20         5                         5
36        11.903655           19         6                         5
37        12.067286           19         8                         5
38        11.853442           18         6                         5
```

```text
Saved equal_operand_behavioral_map.csv
```


### Cell 148 — code (execution count: 107)

```python
# ============================================================
# CORPUS QUERY COLLECTION — RAW LOGGING ONLY
# No eligibility floor or correlation is chosen in this cell.
# ============================================================

import json
import time
import requests
import numpy as np
import pandas as pd

URL = "https://api.infini-gram.io/"
INDEX = "v4_dolma-v1_7_llama"
PREFIX = ""
CORPUS_TARGETS = [4, 6, 8, 10, 12, 14, 16]

query_log = []


def count_query_logged(query, target, role, a=None, b=None, max_tries=3):
    payload = {"index": INDEX, "query_type": "count", "query": query}
    last_error = None

    for attempt in range(1, max_tries + 1):
        status = None
        raw_text = ""
        try:
            response = requests.post(URL, json=payload, timeout=30)
            status = response.status_code
            raw_text = response.text
            response.raise_for_status()
            parsed = response.json()
            if "error" in parsed:
                raise RuntimeError(str(parsed["error"]))
            if "count" not in parsed:
                raise ValueError("Response contained no 'count' field")

            record = {
                "target": target, "role": role, "a": a, "b": b,
                "query": query, "request_payload": json.dumps(payload),
                "attempt": attempt, "http_status": status,
                "count": int(parsed["count"]),
                "tokens": json.dumps(parsed.get("tokens"), ensure_ascii=False),
                "raw_response": raw_text, "error": "",
            }
            query_log.append(record)
            time.sleep(0.15)
            return int(parsed["count"])

        except Exception as exc:
            last_error = repr(exc)
            query_log.append({
                "target": target, "role": role, "a": a, "b": b,
                "query": query, "request_payload": json.dumps(payload),
                "attempt": attempt, "http_status": status,
                "count": np.nan, "tokens": "null",
                "raw_response": raw_text, "error": last_error,
            })
            time.sleep(float(attempt))

    # Preserve a missing count, not a fabricated zero. This lets the caller
    # save the partial raw log and mark the target as incomplete.
    return np.nan


# API/tokenization check; response is kept in the log.
count_query_logged("5 + 5 = 10", target=10, role="tokenization_check")

query_summary_rows = []
for target in CORPUS_TARGETS:
    d = target // 2
    control_pairs = [
        (a, target - a)
        for a in range(1, 10)
        if 1 <= target - a <= 9 and a != target - a
    ]

    double_joint = count_query_logged(
        f"{PREFIX}{d} + {d} = {target}", target, "double_joint", d, d
    )
    double_prefix = count_query_logged(
        f"{PREFIX}{d} + {d} =", target, "double_prefix", d, d
    )

    control_joint_counts = []
    control_prefix_counts = []
    for a, b in control_pairs:
        control_joint_counts.append(count_query_logged(
            f"{PREFIX}{a} + {b} = {target}", target, "control_joint", a, b
        ))
        control_prefix_counts.append(count_query_logged(
            f"{PREFIX}{a} + {b} =", target, "control_prefix", a, b
        ))

    mean_control_joint = float(np.mean(control_joint_counts)) if control_joint_counts else np.nan
    mean_control_prefix = float(np.mean(control_prefix_counts)) if control_prefix_counts else np.nan
    log_ratio_joint = float(np.log10((double_joint + 1) / (mean_control_joint + 1))) if np.isfinite(mean_control_joint) else np.nan
    log_ratio_conditional = (
        float(np.log10(
            ((double_joint + 1) / (double_prefix + 1))
            / ((mean_control_joint + 1) / (mean_control_prefix + 1))
        ))
        if np.isfinite(mean_control_joint) and np.isfinite(mean_control_prefix)
        else np.nan
    )

    query_summary_rows.append({
        "target": target,
        "double_joint_count": double_joint,
        "double_prefix_count": double_prefix,
        "n_ordered_controls": len(control_pairs),
        "mean_control_joint_count": mean_control_joint,
        "mean_control_prefix_count": mean_control_prefix,
        "log_ratio_joint": log_ratio_joint,
        "log_ratio_conditional": log_ratio_conditional,
        "floor_decision": "NOT_APPLIED_PENDING_PRESPECIFICATION",
    })


df_corpus_query_log = pd.DataFrame(query_log)
df_corpus_raw_summary = pd.DataFrame(query_summary_rows)
df_corpus_query_log.to_csv("corpus_raw_api_query_log.csv", index=False)
df_corpus_query_log.to_json("corpus_raw_api_query_log.jsonl", orient="records", lines=True, force_ascii=False)
df_corpus_raw_summary.to_csv("corpus_raw_target_summary_no_floor.csv", index=False)

display(df_corpus_raw_summary)
print("Query records:", len(df_corpus_query_log))
print("Failed attempts:", int(df_corpus_query_log["error"].ne("").sum()))
print("Saved raw query log, JSONL log, and target summary without eligibility filtering.")
```

#### Recorded output

```text
   target  double_joint_count  double_prefix_count  n_ordered_controls  \
0       4               28489                64946                   2
1       6                5076                11432                   4
2       8                3191                 9251                   6
3      10                3812                 9848                   8
4      12                1547                 5074                   6
5      14                 854                 2759                   4
6      16                1083                 3787                   2

   mean_control_joint_count  mean_control_prefix_count  log_ratio_joint  \
0               4305.000000               10762.500000         0.820618
1               2506.750000                7661.750000         0.306323
2               1907.166667                5151.333333         0.223447
3               1490.875000                4364.375000         0.407534
4               1151.166667                3517.166667         0.128256
5                696.500000                2703.500000         0.088422
6                420.500000                1952.000000         0.410232

   log_ratio_conditional                        floor_decision
0               0.040013  NOT_APPLIED_PENDING_PRESPECIFICATION
1               0.132547  NOT_APPLIED_PENDING_PRESPECIFICATION
2              -0.030785  NOT_APPLIED_PENDING_PRESPECIFICATION
3               0.054164  NOT_APPLIED_PENDING_PRESPECIFICATION
4              -0.030864  NOT_APPLIED_PENDING_PRESPECIFICATION
5               0.079600  NOT_APPLIED_PENDING_PRESPECIFICATION
6               0.122524  NOT_APPLIED_PENDING_PRESPECIFICATION
```

```text
Query records: 79
Failed attempts: 0
Saved raw query log, JSONL log, and target summary without eligibility filtering.
```


### Cell 149 — code (execution count: 108)

```python
# ============================================================
# POPULATION DLA FOR THE EQUAL-OPERAND EFFECT
#
# DLA is computed at the final prediction position with
# explicit final LayerNorm scaling.
#
# Output:
#   component-level DLA for every prompt
# ============================================================

import torch
import pandas as pd
import numpy as np
import gc

model.set_use_attn_result(True)

dla_rows = []

for _, row in df_circuit.iterrows():

    prompt = row["prompt"]
    T = int(row["target"])

    tokens = encode_prompt(prompt)

    with torch.no_grad():

        logits, cache = model.run_with_cache(
            tokens
        )

        target_id = model.to_single_token(
            f" {T}"
        )

        foil_minus_id = model.to_single_token(
            f" {T-1}"
        )

        foil_plus_id = model.to_single_token(
            f" {T+1}"
        )

        direction = (
            model.W_U[:, target_id]
            - 0.5 * (
                model.W_U[:, foil_minus_id]
                + model.W_U[:, foil_plus_id]
            )
        )

        # Complete residual decomposition.
        resid_stack, labels = (
            cache.get_full_resid_decomposition(
                layer=-1,
                expand_neurons=False,
                return_labels=True
            )
        )

        # Apply final-LN scaling BEFORE projection.
        resid_stack_ln = (
            cache.apply_ln_to_stack(
                resid_stack,
                layer=-1,
                pos_slice=-1
            )
        )

        last_pos = (
            resid_stack_ln[:, 0, -1, :]
        )

        component_dla = torch.einsum(
            "cd,d->c",
            last_pos,
            direction
        )

        bias_diff = (
            model.b_U[target_id]
            - 0.5 * (
                model.b_U[foil_minus_id]
                + model.b_U[foil_plus_id]
            )
        ).item()

        for label, value in zip(
            labels,
            component_dla
        ):

            dla_rows.append({
                "split": row["split"],
                "target": T,
                "condition": row["condition"],
                "prompt": prompt,
                "component": str(label),
                "dla": value.item(),
                "bias_diff": bias_diff,
            })

    del cache
    del tokens
    gc.collect()


df_population_dla = pd.DataFrame(
    dla_rows
)


print("LPOPULATION DLA COMPLETE")

print(
    "Rows:",
    len(df_population_dla)
)

print(
    "Components:",
    df_population_dla["component"].nunique()
)

df_population_dla.to_csv(
    "equal_operand_population_dla.csv",
    index=False
)

print(
    "\nSaved: "
    "equal_operand_population_dla.csv"
)
```

#### Recorded output

```text
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
WARNING:root:Tried to compute head results when they were already cached
```

```text
LPOPULATION DLA COMPLETE
Rows: 6201
Components: 159

Saved: equal_operand_population_dla.csv
```


### Cell 150 — code (execution count: 109)

```python
# ============================================================
# COMPONENT-LEVEL EQUAL-OPERAND DLA CONTRAST
#
# Contrast:
#
#   DLA(double, T)
#   -
#   mean[DLA(control, T)]
#
# calculated separately at each target level.
#
# Formal inference:
#   exact sign-flip across discovery targets
#
# Multiple comparisons:
#   Benjamini-Hochberg FDR
# ============================================================

import re
import itertools
import numpy as np
import pandas as pd

HEAD_PATTERN = re.compile(
    r"^L\d+H\d+$"
)

MLP_PATTERN = re.compile(
    r"^\d+_mlp_out$"
)


def is_component_candidate(label):
    label = str(label)

    return (
        bool(HEAD_PATTERN.match(label))
        or bool(MLP_PATTERN.match(label))
    )


def exact_signflip(values):

    values = np.asarray(
        values,
        dtype=float
    )

    observed = values.mean()

    null_distribution = np.array([
        np.mean(
            values
            * np.asarray(signs)
        )
        for signs in itertools.product(
            [-1, 1],
            repeat=len(values)
        )
    ])

    return np.mean(
        np.abs(null_distribution)
        >= abs(observed)
    )


def bh_fdr(p_values):
    """Benjamini-Hochberg adjusted q-values, preserving input order."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if np.any(~np.isfinite(p_values)) or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p_values must be finite and between 0 and 1")

    n = len(p_values)
    if n == 0:
        return np.asarray([], dtype=float)

    order = np.argsort(p_values)
    sorted_p = p_values[order]
    ranks = np.arange(1, n + 1, dtype=float)
    sorted_q = sorted_p * n / ranks
    sorted_q = np.minimum.accumulate(sorted_q[::-1])[::-1]

    q_values = np.empty(n, dtype=float)
    q_values[order] = np.clip(sorted_q, 0.0, 1.0)
    return q_values


component_labels = sorted(
    c
    for c in
    df_population_dla["component"].unique()
    if is_component_candidate(c)
)


component_stats = []

for component in component_labels:

    contrasts = []

    for T in DISCOVERY_TARGETS:

        target_data = df_population_dla[
            (df_population_dla["split"] == "discovery")
            &
            (df_population_dla["target"] == T)
            &
            (df_population_dla["component"] == component)
        ]

        double_vals = target_data[
            target_data["condition"] == "double"
        ]["dla"].to_numpy()

        control_vals = target_data[
            target_data["condition"] == "control"
        ]["dla"].to_numpy()

        if len(double_vals) != 1:
            raise ValueError(
                f"{component}, T={T}: "
                f"expected one double value."
            )

        if len(control_vals) < 2:
            raise ValueError(
                f"{component}, T={T}: "
                f"too few controls."
            )

        contrast = (
            double_vals[0]
            - control_vals.mean()
        )

        contrasts.append(
            contrast
        )

    contrasts = np.asarray(
        contrasts,
        dtype=float
    )

    component_stats.append({
        "component": component,
        "mean_contrast": contrasts.mean(),
        "sd_contrast": contrasts.std(ddof=1),
        "positive_targets": int(
            (contrasts > 0).sum()
        ),
        "negative_targets": int(
            (contrasts < 0).sum()
        ),
        "p_exact_signflip": exact_signflip(
            contrasts
        ),
    })


df_component_stats = pd.DataFrame(
    component_stats
)

df_component_stats["q_fdr"] = bh_fdr(
    df_component_stats[
        "p_exact_signflip"
    ].to_numpy()
)

df_component_stats["abs_mean_contrast"] = (
    df_component_stats["mean_contrast"]
    .abs()
)

df_component_stats = (
    df_component_stats
    .sort_values(
        "abs_mean_contrast",
        ascending=False
    )
    .reset_index(drop=True)
)

print("COMPONENT DLA CONTRAST")


print(
    df_component_stats.head(30).to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}"
    )
)

df_component_stats.to_csv(
    "equal_operand_component_dla_stats.csv",
    index=False
)

print(
    "\nSaved: "
    "equal_operand_component_dla_stats.csv"
)
```

#### Recorded output

```text
COMPONENT DLA CONTRAST
 component  mean_contrast  sd_contrast  positive_targets  negative_targets  p_exact_signflip     q_fdr  abs_mean_contrast
      L9H9      +0.078350    +0.083077                 4                 1         +0.125000 +0.513158          +0.078350
10_mlp_out      +0.062766    +0.071974                 5                 0         +0.062500 +0.513158          +0.062766
     L10H2      +0.052577    +0.042061                 4                 1         +0.125000 +0.513158          +0.052577
     L8H11      +0.041092    +0.042521                 4                 1         +0.125000 +0.513158          +0.041092
     L10H7      -0.031791    +0.040796                 1                 4         +0.187500 +0.609375          +0.031791
 9_mlp_out      +0.024537    +0.044503                 3                 2         +0.375000 +0.722222          +0.024537
      L6H9      +0.014111    +0.010640                 5                 0         +0.062500 +0.513158          +0.014111
11_mlp_out      -0.012700    +0.022690                 2                 3         +0.375000 +0.722222          +0.012700
 5_mlp_out      +0.012690    +0.009286                 4                 1         +0.125000 +0.513158          +0.012690
 8_mlp_out      +0.012589    +0.013065                 4                 1         +0.125000 +0.513158          +0.012589
      L6H7      +0.010766    +0.009230                 5                 0         +0.062500 +0.513158          +0.010766
     L11H2      +0.010418    +0.015465                 3                 2         +0.250000 +0.609375          +0.010418
     L8H10      -0.009854    +0.010336                 0                 5         +0.062500 +0.513158          +0.009854
      L5H8      -0.008942    +0.012067                 1                 4         +0.187500 +0.609375          +0.008942
     L4H11      -0.007012    +0.010212                 1                 4         +0.187500 +0.609375          +0.007012
     L11H4      +0.006920    +0.005987                 4                 1         +0.125000 +0.513158          +0.006920
     L11H8      +0.006884    +0.013259                 3                 2         +0.375000 +0.722222          +0.006884
      L7H9      -0.006694    +0.008549                 1                 4         +0.250000 +0.609375          +0.006694
 7_mlp_out      +0.006645    +0.008798                 4                 1         +0.187500 +0.609375          +0.006645
     L11H3      +0.006582    +0.006971                 4                 1         +0.125000 +0.513158          +0.006582
    L11H10      -0.006404    +0.015206                 2                 3         +0.500000 +0.757282          +0.006404
      L7H5      +0.006364    +0.003553                 5                 0         +0.062500 +0.513158          +0.006364
     L7H10      +0.005694    +0.010070                 3                 2         +0.250000 +0.609375          +0.005694
 6_mlp_out      -0.005511    +0.021875                 3                 2         +0.937500 +0.949675          +0.005511
      L9H6      +0.005429    +0.006616                 5                 0         +0.062500 +0.513158          +0.005429
      L8H6      +0.005353    +0.013785                 2                 3         +0.500000 +0.757282          +0.005353
      L9H1      -0.005015    +0.035664                 2                 3         +0.812500 +0.931985          +0.005015
      L7H4      +0.004737    +0.001324                 5                 0         +0.062500 +0.513158          +0.004737
      L4H6      -0.004684    +0.004809                 1                 4         +0.125000 +0.513158          +0.004684
     L10H1      +0.004487    +0.014465                 3                 2         +0.500000 +0.757282          +0.004487

Saved: equal_operand_component_dla_stats.csv
```


### Cell 151 — code (execution count: 110)

```python
# ============================================================
# EXPLORATORY CANDIDATE SET
#
# Select the strongest absolute DLA contrasts.
#
# IMPORTANT:
# These are candidates for causal testing,
# not established circuit components.
# ============================================================

TOP_K_DLA = 12

candidate_components = (
    df_component_stats
    .head(TOP_K_DLA)
    ["component"]
    .tolist()
)


print("DLA CANDIDATE COMPONENTS")


for rank, component in enumerate(
    candidate_components,
    start=1
):
    row = df_component_stats[
        df_component_stats["component"]
        == component
    ].iloc[0]

    print(
        f"{rank:>2}. "
        f"{component:<12} "
        f"mean DLA contrast="
        f"{row['mean_contrast']:+.5f} "
        f"p="
        f"{row['p_exact_signflip']:.5f} "
        f"q="
        f"{row['q_fdr']:.5f}"
    )

pd.DataFrame({
    "rank": np.arange(
        1,
        len(candidate_components) + 1
    ),
    "component": candidate_components,
}).to_csv(
    "equal_operand_dla_candidates.csv",
    index=False
)

print(
    "\nSaved: "
    "equal_operand_dla_candidates.csv"
)
```

#### Recorded output

```text
DLA CANDIDATE COMPONENTS
 1. L9H9         mean DLA contrast=+0.07835 p=0.12500 q=0.51316
 2. 10_mlp_out   mean DLA contrast=+0.06277 p=0.06250 q=0.51316
 3. L10H2        mean DLA contrast=+0.05258 p=0.12500 q=0.51316
 4. L8H11        mean DLA contrast=+0.04109 p=0.12500 q=0.51316
 5. L10H7        mean DLA contrast=-0.03179 p=0.18750 q=0.60938
 6. 9_mlp_out    mean DLA contrast=+0.02454 p=0.37500 q=0.72222
 7. L6H9         mean DLA contrast=+0.01411 p=0.06250 q=0.51316
 8. 11_mlp_out   mean DLA contrast=-0.01270 p=0.37500 q=0.72222
 9. 5_mlp_out    mean DLA contrast=+0.01269 p=0.12500 q=0.51316
10. 8_mlp_out    mean DLA contrast=+0.01259 p=0.12500 q=0.51316
11. L6H7         mean DLA contrast=+0.01077 p=0.06250 q=0.51316
12. L11H2        mean DLA contrast=+0.01042 p=0.25000 q=0.60938

Saved: equal_operand_dla_candidates.csv
```


### Cell 152 — code (execution count: 111)

```python
# ============================================================
# L7 — MEAN-ABLATION CAUSAL SCREEN (EXPLORATORY)
# Correct target IDs, explicit BOS, clean-score validation,
# and measured final LayerNorm scale diagnostics.
# ============================================================

import gc
import re
import numpy as np
import pandas as pd
import torch

model.reset_hooks()

if not candidate_components:
    raise RuntimeError("The DLA candidate set is empty; stop here.")

discovery_df = df_circuit.loc[df_circuit["split"] == "discovery"].copy().reset_index(drop=True)
discovery_prompts = discovery_df["prompt"].tolist()
discovery_targets = discovery_df["target"].astype(int).tolist()
token_rows = [encode_prompt(p) for p in discovery_prompts]
lengths = {int(t.shape[1]) for t in token_rows}
if len(lengths) != 1:
    raise RuntimeError(f"L7 requires equal sequence lengths; got {sorted(lengths)}")
discovery_tokens = torch.cat(token_rows, dim=0)

print("L7 discovery prompts:", len(discovery_prompts))
print("L7 batch shape:", tuple(discovery_tokens.shape))
print("Exploratory components screened:", len(candidate_components))


def batch_primary_scores(logits, targets):
    targets = [int(t) for t in targets]
    if logits.shape[0] != len(targets):
        raise ValueError("Number of logits rows does not match targets.")
    device = logits.device
    batch_idx = torch.arange(len(targets), device=device)
    target_ids = torch.tensor([answer_token_id(t) for t in targets], device=device)
    minus_ids = torch.tensor([answer_token_id(t - 1) for t in targets], device=device)
    plus_ids = torch.tensor([answer_token_id(t + 1) for t in targets], device=device)
    final_logits = logits[:, -1, :]
    return final_logits[batch_idx, target_ids] - 0.5 * (
        final_logits[batch_idx, minus_ids] + final_logits[batch_idx, plus_ids]
    )


def parse_component_l7(component):
    component = str(component)
    match = re.fullmatch(r"L(\d+)H(\d+)", component)
    if match:
        return {"type": "head", "layer": int(match.group(1)), "head": int(match.group(2))}
    match = re.fullmatch(r"(\d+)_mlp_out", component)
    if match:
        return {"type": "mlp", "layer": int(match.group(1))}
    raise ValueError(f"Unsupported component: {component}")


def component_hook_name_l7(component):
    spec = parse_component_l7(component)
    if spec["type"] == "head":
        return f"blocks.{spec['layer']}.attn.hook_z"
    return f"blocks.{spec['layer']}.hook_mlp_out"


def make_mean_ablation_hook_l7(component, reference_vector):
    spec = parse_component_l7(component)
    hook_name = component_hook_name_l7(component)

    if spec["type"] == "head":
        head = spec["head"]
        def hook_fn(value, hook):
            value = value.clone()
            value[:, -1, head, :] = reference_vector.to(device=value.device, dtype=value.dtype)
            return value
    else:
        def hook_fn(value, hook):
            value = value.clone()
            value[:, -1, :] = reference_vector.to(device=value.device, dtype=value.dtype)
            return value

    return hook_name, hook_fn


def run_and_capture_final_ln_scale(tokens, extra_hooks=None):
    captured = {}

    def scale_hook(value, hook):
        # Supports common TransformerLens scale shapes [batch, pos, 1]
        # and [batch, pos]. The returned tensor is left unmodified.
        last = value[:, -1].reshape(value.shape[0], -1).mean(dim=1)
        captured["scale"] = last.detach().float().cpu().numpy()
        return value

    hooks = list(extra_hooks or []) + [("ln_final.hook_scale", scale_hook)]
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)

    if "scale" not in captured:
        raise RuntimeError("Did not capture ln_final.hook_scale; inspect the installed TransformerLens hook names.")
    return logits, captured["scale"]


# Clean pass and measured clean scales.
clean_logits, clean_ln_scale = run_and_capture_final_ln_scale(discovery_tokens)
clean_scores = batch_primary_scores(clean_logits, discovery_targets).detach().cpu().numpy()

# Validate against the canonical behavioral map before any ablation.
lookup = {(str(r["prompt"]), int(r["target"])): float(r["target_score"]) for _, r in df_behavior.iterrows()}
reference_scores = np.asarray([lookup[(str(p), int(t))] for p, t in zip(discovery_prompts, discovery_targets)])
max_diff = float(np.max(np.abs(clean_scores - reference_scores)))
print(f"Maximum clean-score discrepancy vs behavioral map: {max_diff:.8f}")
if max_diff > 5e-4:
    raise RuntimeError("L7 clean-score validation failed; do not interpret causal results.")

# Mean reference activations: same BOS-prefixed discovery batch, same final position.
candidate_reference_vectors = {}
for component in candidate_components:
    hook_name = component_hook_name_l7(component)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            discovery_tokens,
            names_filter=lambda name, wanted=hook_name: name == wanted
        )
    activation = cache[hook_name]
    spec = parse_component_l7(component)
    if spec["type"] == "head":
        reference = activation[:, -1, spec["head"], :].mean(dim=0).detach()
    else:
        reference = activation[:, -1, :].mean(dim=0).detach()
    candidate_reference_vectors[component] = reference
    del cache, activation
    gc.collect()

print("Reference vectors built:", len(candidate_reference_vectors))

# Run one joint forward per candidate and record paired prompt scores.
causal_rows = []
for component in candidate_components:
    print("Ablation screen:", component)
    hook_name, component_hook = make_mean_ablation_hook_l7(
        component, candidate_reference_vectors[component]
    )
    ablated_logits, ablated_ln_scale = run_and_capture_final_ln_scale(
        discovery_tokens,
        extra_hooks=[(hook_name, component_hook)]
    )
    ablated_scores = batch_primary_scores(ablated_logits, discovery_targets).detach().cpu().numpy()

    for i, row in discovery_df.iterrows():
        ln_delta = float(ablated_ln_scale[i] - clean_ln_scale[i])
        ln_relative = ln_delta / max(abs(float(clean_ln_scale[i])), 1e-8)
        causal_rows.append({
            "component": component,
            "target": int(row["target"]),
            "condition": row["condition"],
            "prompt": row["prompt"],
            "clean_score": float(clean_scores[i]),
            "ablated_score": float(ablated_scores[i]),
            "causal_delta": float(ablated_scores[i] - clean_scores[i]),
            "clean_ln_scale": float(clean_ln_scale[i]),
            "ablated_ln_scale": float(ablated_ln_scale[i]),
            "ln_scale_delta": ln_delta,
            "ln_scale_relative_delta": float(ln_relative),
        })
    del ablated_logits
    gc.collect()


df_causal_screen = pd.DataFrame(causal_rows)
if df_causal_screen.empty or df_causal_screen["ln_scale_delta"].isna().any():
    raise RuntimeError("L7 did not produce complete scale diagnostics.")
df_causal_screen.to_csv("equal_operand_mean_ablation_screen.csv", index=False)
display(df_causal_screen.head(20))
print("Saved equal_operand_mean_ablation_screen.csv")
print("Scale changes are diagnostics, not evidence that the intervention is on-distribution.")
```

#### Recorded output

```text
L7 discovery prompts: 27
L7 batch shape: (27, 5)
Exploratory components screened: 12
Maximum clean-score discrepancy vs behavioral map: 0.00000477
Reference vectors built: 12
Ablation screen: L9H9
Ablation screen: 10_mlp_out
Ablation screen: L10H2
Ablation screen: L8H11
Ablation screen: L10H7
Ablation screen: 9_mlp_out
Ablation screen: L6H9
Ablation screen: 11_mlp_out
Ablation screen: 5_mlp_out
Ablation screen: 8_mlp_out
Ablation screen: L6H7
Ablation screen: L11H2
```

```text
   component  target condition   prompt  clean_score  ablated_score  \
0       L9H9       4    double  2 + 2 =     0.251475       0.242887
1       L9H9       4   control  1 + 3 =     0.321500       0.348428
2       L9H9       4   control  3 + 1 =     0.084194       0.113064
3       L9H9       6    double  3 + 3 =     0.230020       0.243445
4       L9H9       6   control  1 + 5 =     0.160547       0.196177
5       L9H9       6   control  2 + 4 =     0.366150       0.353354
6       L9H9       6   control  4 + 2 =     0.263329       0.252364
7       L9H9       6   control  5 + 1 =     0.062846       0.130741
8       L9H9      10    double  5 + 5 =     0.647227       0.580715
9       L9H9      10   control  1 + 9 =     0.120123       0.155250
10      L9H9      10   control  2 + 8 =     0.222569       0.184076
11      L9H9      10   control  3 + 7 =     0.092884       0.092921
12      L9H9      10   control  4 + 6 =     0.110023       0.099161
13      L9H9      10   control  6 + 4 =     0.036098       0.025167
14      L9H9      10   control  7 + 3 =     0.090352       0.091690
15      L9H9      10   control  8 + 2 =     0.153768       0.115917
16      L9H9      10   control  9 + 1 =     0.074964       0.153681
17      L9H9      12    double  6 + 6 =     0.401201       0.338675
18      L9H9      12   control  3 + 9 =     0.188456       0.200515
19      L9H9      12   control  4 + 8 =     0.358142       0.334909

    causal_delta  clean_ln_scale  ablated_ln_scale  ln_scale_delta  \
0      -0.008589       20.642012         20.607826       -0.034185
1       0.026928       20.199615         20.192186       -0.007429
2       0.028870       18.926306         18.914911       -0.011395
3       0.013426       20.469273         20.499943        0.030670
4       0.035630       18.300924         18.270706       -0.030218
5      -0.012795       20.786850         20.811697        0.024847
6      -0.010964       20.156878         20.141298       -0.015579
7       0.067895       17.781740         17.746351       -0.035389
8      -0.066512       17.938320         17.912916       -0.025404
9       0.035127       20.138668         20.097948       -0.040720
10     -0.038492       20.900604         20.893436       -0.007168
11      0.000037       21.237133         21.298319        0.061186
12     -0.010861       19.515268         19.521498        0.006229
13     -0.010931       20.177826         20.192894        0.015068
14      0.001338       20.559181         20.597662        0.038481
15     -0.037850       20.418070         20.345781       -0.072289
16      0.078717       18.917061         18.875927       -0.041134
17     -0.062526       19.649988         19.712421        0.062433
18      0.012059       19.860172         19.855467       -0.004705
19     -0.023232       19.889332         19.876024       -0.013308

    ln_scale_relative_delta
0                 -0.001656
1                 -0.000368
2                 -0.000602
3                  0.001498
4                 -0.001651
5                  0.001195
6                 -0.000773
7                 -0.001990
8                 -0.001416
9                 -0.002022
10                -0.000343
11                 0.002881
12                 0.000319
13                 0.000747
14                 0.001872
15                -0.003540
16                -0.002174
17                 0.003177
18                -0.000237
19                -0.000669
```

```text
Saved equal_operand_mean_ablation_screen.csv
Scale changes are diagnostics, not evidence that the intervention is on-distribution.
```


### Cell 153 — code (execution count: 112)

```python
# ============================================================
# L8 — TARGET-LEVEL CAUSAL CONTRASTS (EXPLORATORY)
# Exact sign-flip tests and FDR correction across screened components.
# ============================================================

import itertools
import numpy as np
import pandas as pd


def exact_signflip_p(values, alternative="two-sided"):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a nonempty finite 1-D array")
    observed = float(values.mean())
    null = np.asarray([
        np.mean(values * np.asarray(signs, dtype=float))
        for signs in itertools.product([-1, 1], repeat=len(values))
    ])
    if alternative == "two-sided":
        return float(np.mean(np.abs(null) >= abs(observed)))
    if alternative == "less":
        return float(np.mean(null <= observed))
    if alternative == "greater":
        return float(np.mean(null >= observed))
    raise ValueError("alternative must be two-sided, less, or greater")


def bh_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(p_values)) or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p-values must be finite and in [0, 1]")
    n = len(p_values)
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    ranks = np.arange(1, n + 1, dtype=float)
    sorted_q = sorted_p * n / ranks
    sorted_q = np.minimum.accumulate(sorted_q[::-1])[::-1]
    q_values = np.empty(n, dtype=float)
    q_values[order] = np.clip(sorted_q, 0.0, 1.0)
    return q_values


summary_rows = []
for component_index, component in enumerate(candidate_components):
    comp = df_causal_screen[df_causal_screen["component"] == component]
    target_changes = []

    for T in DISCOVERY_TARGETS:
        data = comp[comp["target"] == T]
        doubles = data[data["condition"] == "double"]
        controls = data[data["condition"] == "control"]
        if len(doubles) != 1 or len(controls) < 2:
            raise ValueError(f"Bad trial structure for {component}, target {T}")

        clean_adv = float(doubles["clean_score"].iloc[0] - controls["clean_score"].mean())
        ablated_adv = float(doubles["ablated_score"].iloc[0] - controls["ablated_score"].mean())
        target_changes.append(ablated_adv - clean_adv)

    changes = np.asarray(target_changes, dtype=float)
    rng = np.random.default_rng(20261009 + component_index)
    boot = np.asarray([
        rng.choice(changes, size=len(changes), replace=True).mean()
        for _ in range(20000)
    ])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    comp_scale = comp["ln_scale_relative_delta"].abs()

    summary_rows.append({
        "component": component,
        "mean_advantage_change": float(changes.mean()),
        "sd_advantage_change": float(changes.std(ddof=1)),
        "targets_reduced": int((changes < 0).sum()),
        "targets_total": len(changes),
        "p_two_sided": exact_signflip_p(changes, "two-sided"),
        "p_reduce_one_sided": exact_signflip_p(changes, "less"),
        "bootstrap_ci_low": float(lo),
        "bootstrap_ci_high": float(hi),
        "mean_abs_relative_ln_scale_change": float(comp_scale.mean()),
        "max_abs_relative_ln_scale_change": float(comp_scale.max()),
    })

df_causal_summary = pd.DataFrame(summary_rows)
df_causal_summary["q_fdr_two_sided"] = bh_adjust(df_causal_summary["p_two_sided"].to_numpy())
df_causal_summary["q_fdr_reduce_one_sided"] = bh_adjust(df_causal_summary["p_reduce_one_sided"].to_numpy())
df_causal_summary = df_causal_summary.sort_values("mean_advantage_change").reset_index(drop=True)

display(df_causal_summary)
df_causal_summary.to_csv("equal_operand_causal_specificity.csv", index=False)
print("Saved equal_operand_causal_specificity.csv")
print("Inference unit is the target level (only five discovery targets); treat this as exploratory.")
```

#### Recorded output

```text
     component  mean_advantage_change  sd_advantage_change  targets_reduced  \
0   10_mlp_out              -0.078840             0.094255                5
1         L9H9              -0.070940             0.062652                5
2        L10H2              -0.058887             0.042789                5
3    9_mlp_out              -0.045290             0.071313                3
4        L8H11              -0.036078             0.026870                5
5    8_mlp_out              -0.028950             0.034392                4
6         L6H7              -0.016824             0.016705                4
7         L6H9              -0.010845             0.010251                4
8        L11H2              -0.008926             0.014472                3
9   11_mlp_out               0.010015             0.029559                2
10   5_mlp_out               0.010997             0.006436                0
11       L10H7               0.033515             0.035447                1

    targets_total  p_two_sided  p_reduce_one_sided  bootstrap_ci_low  \
0               5       0.0625             0.03125         -0.159220
1               5       0.0625             0.03125         -0.124993
2               5       0.0625             0.03125         -0.091854
3               5       0.3750             0.18750         -0.101085
4               5       0.0625             0.03125         -0.056799
5               5       0.1250             0.06250         -0.057729
6               5       0.1250             0.06250         -0.031865
7               5       0.1250             0.06250         -0.018808
8               5       0.2500             0.12500         -0.020860
9               5       0.4375             0.81250         -0.012770
10              5       0.0625             1.00000          0.005372
11              5       0.1250             0.96875          0.005393

    bootstrap_ci_high  mean_abs_relative_ln_scale_change  \
0           -0.011958                           0.020246
1           -0.030931                           0.001496
2           -0.026021                           0.001966
3            0.009059                           0.018212
4           -0.015357                           0.002991
5           -0.003321                           0.019164
6           -0.005455                           0.005303
7           -0.002882                           0.001768
8            0.000742                           0.000182
9            0.032800                           0.024798
10           0.015109                           0.006547
11           0.061961                           0.000980

    max_abs_relative_ln_scale_change  q_fdr_two_sided  q_fdr_reduce_one_sided
0                           0.066421         0.150000                0.093750
1                           0.003540         0.150000                0.093750
2                           0.005249         0.150000                0.093750
3                           0.052599         0.409091                0.250000
4                           0.007777         0.150000                0.093750
5                           0.055043         0.166667                0.107143
6                           0.028169         0.166667                0.107143
7                           0.009025         0.166667                0.107143
8                           0.000588         0.300000                0.187500
9                           0.084328         0.437500                0.975000
10                          0.012855         0.150000                1.000000
11                          0.002298         0.166667                1.000000
```

```text
Saved equal_operand_causal_specificity.csv
Inference unit is the target level (only five discovery targets); treat this as exploratory.
```


### Cell 154 — code (execution count: 113)

```python
model.reset_hooks()
```


### Cell 155 — code (execution count: 114)

```python
# ============================================================
# L9 — ACTIVATION PATCHING OF CANDIDATE COMPONENTS
#
# Question:
#   Can activation from a clean equal-operand prompt restore
#   the target-vs-foil score in a matched non-equal prompt?
#
# Clean:
#   d + d =
#
# Corrupt:
#   (d-1) + (d+1) =
#
# Candidate components come from the preceding causal screen.
# No variable called `sparse_component` is used.
# ============================================================

import numpy as np
import pandas as pd
import torch


# ------------------------------------------------------------
# 1. Check required variables
# ------------------------------------------------------------

required_variables = [
    "candidate_components",
    "df_circuit",
    "DISCOVERY_TARGETS",
    "primary_score",
    "parse_component_l7",
]

missing_variables = [
    name
    for name in required_variables
    if name not in globals()
]

if missing_variables:
    raise RuntimeError(
        "L9 is missing required variables: "
        f"{missing_variables}. "
        "Run L1-L8 in order."
    )


# ------------------------------------------------------------
# 2. Use candidates from the causal screen
#
# Prefer the strongest candidates that actually have a
# measured causal effect.
# ------------------------------------------------------------

if "df_causal_summary" not in globals():
    raise RuntimeError(
        "df_causal_summary is missing. "
        "Run L8 first."
    )

patch_candidates = (
    df_causal_summary[
        df_causal_summary["component"].isin(
            candidate_components
        )
    ]
    .sort_values(
        "mean_advantage_change",
        ascending=True
    )
    .head(5)
    ["component"]
    .tolist()
)

if len(patch_candidates) == 0:
    raise RuntimeError(
        "No valid patching candidates were found."
    )

print("=" * 78)
print("L9 — ACTIVATION PATCHING")
print("=" * 78)

print(
    "Candidates:"
)

for component in patch_candidates:
    print(
        f"  {component}"
    )


# ------------------------------------------------------------
# 3. Helper for a clean -> corrupt activation patch
# ------------------------------------------------------------

def make_clean_to_corrupt_patch_l9(
    component,
    clean_cache
):

    spec = parse_component_l7(
        component
    )

    # --------------------------------------------------------
    # Attention head
    # --------------------------------------------------------

    if spec["type"] == "head":

        layer = spec["layer"]
        head = spec["head"]

        hook_name = (
            f"blocks.{layer}.attn.hook_z"
        )

        def patch_hook(
            value,
            hook
        ):

            value = value.clone()

            value[
                :,
                -1,
                head,
                :
            ] = clean_cache[
                hook_name
            ][:,
            -1,
            head,
            :].to(
                value.device,
                dtype=value.dtype
            )

            return value

        return (
            hook_name,
            patch_hook
        )

    # --------------------------------------------------------
    # MLP output
    # --------------------------------------------------------

    elif spec["type"] == "mlp":

        layer = spec["layer"]

        hook_name = (
            f"blocks.{layer}.hook_mlp_out"
        )

        def patch_hook(
            value,
            hook
        ):

            value = value.clone()

            value[
                :,
                -1,
                :
            ] = clean_cache[
                hook_name
            ][:,
            -1,
            :].to(
                value.device,
                dtype=value.dtype
            )

            return value

        return (
            hook_name,
            patch_hook
        )

    else:

        raise ValueError(
            f"Unsupported component: {component}"
        )


# ------------------------------------------------------------
# 4. Run clean -> corrupt -> patched experiments
# ------------------------------------------------------------

patch_rows = []


for component in patch_candidates:

    print(
        f"\nTesting {component}..."
    )

    for T in DISCOVERY_TARGETS:

        d = T // 2

        clean_prompt = (
            f"{d} + {d} ="
        )

        corrupt_prompt = (
            f"{d-1} + {d+1} ="
        )

        # ----------------------------------------------------
        # Clean cache
        # ----------------------------------------------------

        with torch.no_grad():

           clean_logits, clean_cache = model.run_with_cache(encode_prompt(clean_prompt))

        # ----------------------------------------------------
        # Clean score
        # ----------------------------------------------------

        clean_metric = primary_score(
            clean_prompt,
            T
        )

        # ----------------------------------------------------
        # Corrupt score
        # ----------------------------------------------------

        corrupt_metric = primary_score(
            corrupt_prompt,
            T
        )

        # ----------------------------------------------------
        # Build patch hook
        # ----------------------------------------------------

        hook_name, patch_hook = (
            make_clean_to_corrupt_patch_l9(
                component,
                clean_cache
            )
        )

        # ----------------------------------------------------
        # Patch clean activation into corrupt prompt
        # ----------------------------------------------------

        with torch.no_grad():

            patched_logits = model.run_with_hooks(
    encode_prompt(corrupt_prompt),
    fwd_hooks=[(hook_name, patch_hook)]
)

        # ----------------------------------------------------
        # Calculate patched score
        # ----------------------------------------------------

        target_id = model.to_single_token(
            f" {T}"
        )

        foil_minus_id = model.to_single_token(
            f" {T-1}"
        )

        foil_plus_id = model.to_single_token(
            f" {T+1}"
        )

        patched_metric = (
            patched_logits[
                0,
                -1,
                target_id
            ]
            - 0.5 * (
                patched_logits[
                    0,
                    -1,
                    foil_minus_id
                ]
                + patched_logits[
                    0,
                    -1,
                    foil_plus_id
                ]
            )
        ).item()

        # ----------------------------------------------------
        # Recovery
        # ----------------------------------------------------

        denominator = (
            clean_metric
            - corrupt_metric
        )

        if abs(denominator) < 1e-8:

            recovery = np.nan

        else:

            recovery = (
                patched_metric
                - corrupt_metric
            ) / denominator

        patch_rows.append({
            "component": component,
            "target": int(T),
            "clean_prompt": clean_prompt,
            "corrupt_prompt": corrupt_prompt,
            "clean_metric": clean_metric,
            "corrupt_metric": corrupt_metric,
            "patched_metric": patched_metric,
            "recovery": recovery,
        })

        del clean_cache


# ------------------------------------------------------------
# 5. Results table
# ------------------------------------------------------------

df_patch = pd.DataFrame(
    patch_rows
)

print("\n" + "=" * 78)
print("PATCHING RESULTS")
print("=" * 78)

print(
    df_patch.to_string(
        index=False,
        float_format=lambda x:
            f"{x:+.6f}"
    )
)


# ------------------------------------------------------------
# 6. Candidate-level summary
# ------------------------------------------------------------

patch_summary = (
    df_patch
    .groupby("component")
    .agg(
        mean_recovery=(
            "recovery",
            "mean"
        ),

        median_recovery=(
            "recovery",
            "median"
        ),

        positive_recoveries=(
            "recovery",
            lambda x:
                int(
                    (x > 0).sum()
                )
        ),

        n_valid=(
            "recovery",
            lambda x:
                int(
                    x.notna().sum()
                )
        ),
    )
    .reset_index()
    .sort_values(
        "mean_recovery",
        ascending=False
    )
)

print("\n" + "=" * 78)
print("PATCH SUMMARY")
print("=" * 78)

print(
    patch_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:+.6f}"
    )
)


# ------------------------------------------------------------
# 7. Save
# ------------------------------------------------------------

df_patch.to_csv(
    "equal_operand_activation_patch.csv",
    index=False
)

patch_summary.to_csv(
    "equal_operand_activation_patch_summary.csv",
    index=False
)

print("\nSaved:")
print(
    "  equal_operand_activation_patch.csv"
)

print(
    "  equal_operand_activation_patch_summary.csv"
)
model.reset_hooks()
```

#### Recorded output

```text
==============================================================================
L9 — ACTIVATION PATCHING
==============================================================================
Candidates:
  10_mlp_out
  L9H9
  L10H2
  9_mlp_out
  L8H11

Testing 10_mlp_out...

Testing L9H9...

Testing L10H2...

Testing 9_mlp_out...

Testing L8H11...

==============================================================================
PATCHING RESULTS
==============================================================================
 component  target clean_prompt corrupt_prompt  clean_metric  corrupt_metric  patched_metric  recovery
10_mlp_out       4      2 + 2 =        1 + 3 =     +0.251480       +0.321500       +0.334983 -0.192561
10_mlp_out       6      3 + 3 =        2 + 4 =     +0.230024       +0.366152       +0.356499 +0.070912
10_mlp_out      10      5 + 5 =        4 + 6 =     +0.647227       +0.110022       +0.248698 +0.258144
10_mlp_out      12      6 + 6 =        5 + 7 =     +0.401203       +0.222425       +0.223994 +0.008775
10_mlp_out      16      8 + 8 =        7 + 9 =     +0.719002       +0.081398       +0.183592 +0.160278
      L9H9       4      2 + 2 =        1 + 3 =     +0.251480       +0.321500       +0.341348 -0.283461
      L9H9       6      3 + 3 =        2 + 4 =     +0.230024       +0.366152       +0.329456 +0.269567
      L9H9      10      5 + 5 =        4 + 6 =     +0.647227       +0.110022       +0.122733 +0.023662
      L9H9      12      6 + 6 =        5 + 7 =     +0.401203       +0.222425       +0.316327 +0.525242
      L9H9      16      8 + 8 =        7 + 9 =     +0.719002       +0.081398       +0.239374 +0.247765
     L10H2       4      2 + 2 =        1 + 3 =     +0.251480       +0.321500       +0.366118 -0.637229
     L10H2       6      3 + 3 =        2 + 4 =     +0.230024       +0.366152       +0.336089 +0.220842
     L10H2      10      5 + 5 =        4 + 6 =     +0.647227       +0.110022       +0.173741 +0.118613
     L10H2      12      6 + 6 =        5 + 7 =     +0.401203       +0.222425       +0.288916 +0.371915
     L10H2      16      8 + 8 =        7 + 9 =     +0.719002       +0.081398       +0.159035 +0.121763
 9_mlp_out       4      2 + 2 =        1 + 3 =     +0.251480       +0.321500       +0.245206 +1.089607
 9_mlp_out       6      3 + 3 =        2 + 4 =     +0.230024       +0.366152       +0.347802 +0.134798
 9_mlp_out      10      5 + 5 =        4 + 6 =     +0.647227       +0.110022       +0.167957 +0.107846
 9_mlp_out      12      6 + 6 =        5 + 7 =     +0.401203       +0.222425       +0.215998 -0.035954
 9_mlp_out      16      8 + 8 =        7 + 9 =     +0.719002       +0.081398       +0.153546 +0.113155
     L8H11       4      2 + 2 =        1 + 3 =     +0.251480       +0.321500       +0.329011 -0.107272
     L8H11       6      3 + 3 =        2 + 4 =     +0.230024       +0.366152       +0.346482 +0.144493
     L8H11      10      5 + 5 =        4 + 6 =     +0.647227       +0.110022       +0.130193 +0.037548
     L8H11      12      6 + 6 =        5 + 7 =     +0.401203       +0.222425       +0.249885 +0.153594
     L8H11      16      8 + 8 =        7 + 9 =     +0.719002       +0.081398       +0.132214 +0.079698

==============================================================================
PATCH SUMMARY
==============================================================================
 component  mean_recovery  median_recovery  positive_recoveries  n_valid
 9_mlp_out      +0.281890        +0.113155                    4        5
      L9H9      +0.156555        +0.247765                    4        5
     L8H11      +0.061612        +0.079698                    4        5
10_mlp_out      +0.061110        +0.070912                    4        5
     L10H2      +0.039181        +0.121763                    4        5

Saved:
  equal_operand_activation_patch.csv
  equal_operand_activation_patch_summary.csv
```


### Cell 156 — code (execution count: 115)

```python
# ============================================================
# L10 — POSITION-WISE ACTIVATION PATCHING
#
# Question:
# At which sequence positions does patching a candidate
# component change the clean-vs-corrupt equal-operand contrast?
#
# Clean:   d + d =
# Corrupt: (d-1) + (d+1) =
#
# This is an exploratory localization test, not proof that
# a specific circuit has been identified.
# ============================================================

import gc
import re
import numpy as np
import pandas as pd
import torch

model.reset_hooks()

# ------------------------------------------------------------
# 1. Check prerequisites and select exploratory candidates
# ------------------------------------------------------------

required_objects = [
    "df_causal_summary",
    "df_behavior",
    "DISCOVERY_TARGETS",
    "model",
]

missing_objects = [
    name for name in required_objects
    if name not in globals()
]

if missing_objects:
    raise RuntimeError(
        "Run the corrected upstream cells first. Missing: "
        + ", ".join(missing_objects)
    )

required_columns = {"component", "mean_advantage_change"}

if not required_columns.issubset(df_causal_summary.columns):
    raise ValueError(
        "df_causal_summary must contain: "
        "'component' and 'mean_advantage_change'."
    )

if not {"prompt", "target", "target_score"}.issubset(
    df_behavior.columns
):
    raise ValueError(
        "df_behavior must contain prompt, target, and target_score. "
        "Rerun the corrected behavioral-map cell."
    )

position_candidates = (
    df_causal_summary.loc[
        df_causal_summary["mean_advantage_change"] < 0
    ]
    .sort_values("mean_advantage_change", ascending=True)
    .head(3)["component"]
    .tolist()
)

if not position_candidates:
    raise RuntimeError(
        "No negative-effect candidates were selected by the "
        "exploratory L8 screen. Do not force a component into L10."
    )

print("=" * 72)
print("L10 — POSITION-WISE ACTIVATION PATCHING")
print("=" * 72)
print("Exploratory candidates:")
for component in position_candidates:
    print(" ", component)


# ------------------------------------------------------------
# 2. Shared helpers
# ------------------------------------------------------------

def encode_prompt_l10(prompt):
    """Encode a prompt with exactly one explicit BOS convention."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    return tokens.to(model.W_E.device)


def answer_id_l10(number):
    """Get the vocabulary ID for the exact leading-space answer."""
    text = f" {int(number)}"
    token_ids = model.to_tokens(text, prepend_bos=False)

    if token_ids.shape[1] != 1:
        pieces = model.to_str_tokens(
            text, prepend_bos=False
        )
        raise ValueError(
            f"{text!r} is not a single token: {pieces}. "
            "Do not continue with this target until its scoring "
            "representation is resolved."
        )

    return int(token_ids[0, 0].item())


def score_logits_l10(logits, target):
    """Target logit minus the mean of the adjacent numeric foils."""
    tid = answer_id_l10(target)
    minus_id = answer_id_l10(target - 1)
    plus_id = answer_id_l10(target + 1)

    return (
        logits[tid]
        - 0.5 * (logits[minus_id] + logits[plus_id])
    )


def primary_score_l10(prompt, target):
    tokens = encode_prompt_l10(prompt)

    with torch.no_grad():
        logits = model(tokens, return_type="logits")[0, -1]

    return float(score_logits_l10(logits, target).item())


def parse_component_l10(component):
    """Parse names such as L10H7 and 8_mlp_out."""
    component = str(component)

    match = re.fullmatch(r"L(\d+)H(\d+)", component)
    if match:
        return {
            "type": "head",
            "layer": int(match.group(1)),
            "head": int(match.group(2)),
        }

    match = re.fullmatch(r"(\d+)_mlp_out", component)
    if match:
        return {
            "type": "mlp",
            "layer": int(match.group(1)),
        }

    raise ValueError(
        f"Unsupported candidate component: {component}"
    )


# ------------------------------------------------------------
# 3. Validate the behavioral reference map
# ------------------------------------------------------------

baseline_lookup = {
    (str(row["prompt"]), int(row["target"])):
        float(row["target_score"])
    for _, row in df_behavior.iterrows()
}

position_rows = []
reference_tolerance = 5e-4

# ------------------------------------------------------------
# 4. Sweep components, targets, and token positions
# ------------------------------------------------------------

for component in position_candidates:

    spec = parse_component_l10(component)
    layer = spec["layer"]

    if spec["type"] == "head":
        head = spec["head"]
        hook_name = f"blocks.{layer}.attn.hook_z"
    else:
        head = None
        hook_name = f"blocks.{layer}.hook_mlp_out"

    print(f"\nTesting {component}")

    for T in DISCOVERY_TARGETS:

        T = int(T)
        d = T // 2

        clean_prompt = f"{d} + {d} ="
        corrupt_prompt = f"{d-1} + {d+1} ="

        # ----------------------------------------------------
        # 4a. Encode each prompt with explicit BOS.
        #     These tensors are the actual model inputs.
        # ----------------------------------------------------

        clean_input = encode_prompt_l10(clean_prompt)
        corrupt_input = encode_prompt_l10(corrupt_prompt)

        clean_tokens = model.to_str_tokens(clean_input[0])
        corrupt_tokens = model.to_str_tokens(corrupt_input[0])

        if len(clean_tokens) != len(corrupt_tokens):
            raise ValueError(
                f"Token-length mismatch at T={T}: "
                f"clean={clean_tokens}; corrupt={corrupt_tokens}"
            )

        n_positions = len(clean_tokens)

        # ----------------------------------------------------
        # 4b. Capture actual clean-run activations.
        #
        # Important: clean_cache is an activation cache,
        # NOT the tensor returned by encode_prompt_l10().
        # ----------------------------------------------------

        model.reset_hooks()

        with torch.no_grad():
            _, clean_cache = model.run_with_cache(
                clean_input,
                names_filter=lambda name, wanted=hook_name:
                    name == wanted
            )

        if hook_name not in clean_cache:
            raise RuntimeError(
                f"Activation {hook_name!r} was not captured. "
                "Check the component name and hook."
            )

        clean_component_acts = clean_cache[hook_name]

        if clean_component_acts.shape[1] != n_positions:
            raise RuntimeError(
                f"Activation/token length mismatch for {component}, "
                f"T={T}: activation shape={tuple(clean_component_acts.shape)}, "
                f"tokens={n_positions}"
            )

        # Validate token IDs before scoring.
        for number in (T - 1, T, T + 1):
            answer_id_l10(number)

        # ----------------------------------------------------
        # 4c. Get baseline metrics and check the clean score.
        # ----------------------------------------------------

        clean_metric = primary_score_l10(clean_prompt, T)
        corrupt_metric = primary_score_l10(corrupt_prompt, T)

        baseline_key = (clean_prompt, T)

        if baseline_key not in baseline_lookup:
            raise RuntimeError(
                f"No canonical behavioral baseline for {baseline_key}. "
                "Check df_behavior and rerun the upstream map."
            )

        reference_score = baseline_lookup[baseline_key]
        difference = abs(clean_metric - reference_score)

        if difference > reference_tolerance:
            raise RuntimeError(
                f"Clean-score validation failed for T={T}: "
                f"L10={clean_metric:+.6f}, "
                f"behavioral map={reference_score:+.6f}, "
                f"absolute difference={difference:.6f}. "
                "Stop and resolve the scoring/BOS mismatch."
            )

        denominator = clean_metric - corrupt_metric

        print(
            f"  T={T:2d} | clean={clean_metric:+.5f} "
            f"| corrupt={corrupt_metric:+.5f} "
            f"| positions={n_positions}"
        )

        # ----------------------------------------------------
        # 4d. Patch each position from clean into corrupt.
        # ----------------------------------------------------

        for position in range(n_positions):

            # Bind the source activation and index in the
            # closure's default arguments. This avoids accidental
            # reuse of values from another loop iteration.
            source_acts = clean_component_acts

            if spec["type"] == "head":

                def patch_hook(
                    value,
                    hook,
                    pos=position,
                    h=head,
                    source=source_acts,
                ):
                    value = value.clone()
                    value[:, pos, h, :] = source[
                        :, pos, h, :
                    ].to(
                        device=value.device,
                        dtype=value.dtype
                    )
                    return value

            else:

                def patch_hook(
                    value,
                    hook,
                    pos=position,
                    source=source_acts,
                ):
                    value = value.clone()
                    value[:, pos, :] = source[
                        :, pos, :
                    ].to(
                        device=value.device,
                        dtype=value.dtype
                    )
                    return value

            # The corrupt run uses the same BOS convention.
            model.reset_hooks()

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    corrupt_input,
                    fwd_hooks=[
                        (hook_name, patch_hook)
                    ]
                )

            patched_metric = float(
                score_logits_l10(
                    patched_logits[0, -1],
                    T
                ).item()
            )

            if abs(denominator) < 1e-8:
                recovery = np.nan
            else:
                recovery = (
                    patched_metric - corrupt_metric
                ) / denominator

            # The token label comes from THIS target's clean prompt.
            position_rows.append({
                "component": component,
                "target": T,
                "position": position,
                "token": clean_tokens[position],
                "corrupt_token": corrupt_tokens[position],
                "clean_metric": clean_metric,
                "canonical_clean_metric": reference_score,
                "corrupt_metric": corrupt_metric,
                "patched_metric": patched_metric,
                "recovery": (
                    float(recovery)
                    if np.isfinite(recovery)
                    else np.nan
                ),
            })

        del clean_cache
        del clean_component_acts
        gc.collect()


# ------------------------------------------------------------
# 5. Assemble and save results
# ------------------------------------------------------------

model.reset_hooks()

df_position_patch = pd.DataFrame(position_rows)

if df_position_patch.empty:
    raise RuntimeError("L10 produced no position-patching results.")

print("\n" + "=" * 72)
print("POSITION PATCHING RESULTS")
print("=" * 72)

display(df_position_patch)

position_summary = (
    df_position_patch
    .groupby(
        ["component", "position", "token"],
        as_index=False
    )
    .agg(
        mean_recovery=("recovery", "mean"),
        median_recovery=("recovery", "median"),
        sd_recovery=("recovery", "std"),
        n_valid=("recovery", lambda x: int(x.notna().sum())),
    )
    .sort_values("mean_recovery", ascending=False)
)

print("\nPOSITION SUMMARY")
display(position_summary)

df_position_patch.to_csv(
    "equal_operand_position_patch.csv",
    index=False
)

position_summary.to_csv(
    "equal_operand_position_patch_summary.csv",
    index=False
)

print("\nSaved:")
print("  equal_operand_position_patch.csv")
print("  equal_operand_position_patch_summary.csv")
print(
    "\nInterpretation note: position-wise recovery is exploratory. "
    "A large recovery at a position does not, by itself, prove that "
    "the corresponding token stores or computes the effect."
)
```

#### Recorded output

```text
========================================================================
L10 — POSITION-WISE ACTIVATION PATCHING
========================================================================
Exploratory candidates:
  10_mlp_out
  L9H9
  L10H2

Testing 10_mlp_out
  T= 4 | clean=+0.25148 | corrupt=+0.32150 | positions=5
  T= 6 | clean=+0.23002 | corrupt=+0.36615 | positions=5
  T=10 | clean=+0.64723 | corrupt=+0.11002 | positions=5
  T=12 | clean=+0.40120 | corrupt=+0.22243 | positions=5
  T=16 | clean=+0.71900 | corrupt=+0.08140 | positions=5

Testing L9H9
  T= 4 | clean=+0.25148 | corrupt=+0.32150 | positions=5
  T= 6 | clean=+0.23002 | corrupt=+0.36615 | positions=5
  T=10 | clean=+0.64723 | corrupt=+0.11002 | positions=5
  T=12 | clean=+0.40120 | corrupt=+0.22243 | positions=5
  T=16 | clean=+0.71900 | corrupt=+0.08140 | positions=5

Testing L10H2
  T= 4 | clean=+0.25148 | corrupt=+0.32150 | positions=5
  T= 6 | clean=+0.23002 | corrupt=+0.36615 | positions=5
  T=10 | clean=+0.64723 | corrupt=+0.11002 | positions=5
  T=12 | clean=+0.40120 | corrupt=+0.22243 | positions=5
  T=16 | clean=+0.71900 | corrupt=+0.08140 | positions=5

========================================================================
POSITION PATCHING RESULTS
========================================================================
```

```text
     component  target  position          token  corrupt_token  clean_metric  \
0   10_mlp_out       4         0  <|endoftext|>  <|endoftext|>      0.251480
1   10_mlp_out       4         1              2              1      0.251480
2   10_mlp_out       4         2              +              +      0.251480
3   10_mlp_out       4         3              2              3      0.251480
4   10_mlp_out       4         4              =              =      0.251480
..         ...     ...       ...            ...            ...           ...
70       L10H2      16         0  <|endoftext|>  <|endoftext|>      0.719002
71       L10H2      16         1              8              7      0.719002
72       L10H2      16         2              +              +      0.719002
73       L10H2      16         3              8              9      0.719002
74       L10H2      16         4              =              =      0.719002

    canonical_clean_metric  corrupt_metric  patched_metric  recovery
0                 0.251480        0.321500        0.321500 -0.000000
1                 0.251480        0.321500        0.315338  0.087999
2                 0.251480        0.321500        0.320492  0.014396
3                 0.251480        0.321500        0.318913  0.036951
4                 0.251480        0.321500        0.334983 -0.192561
..                     ...             ...             ...       ...
70                0.719002        0.081398        0.081398  0.000000
71                0.719002        0.081398        0.081589  0.000299
72                0.719002        0.081398        0.082301  0.001416
73                0.719002        0.081398        0.082045  0.001014
74                0.719002        0.081398        0.159035  0.121763

[75 rows x 10 columns]
```

```text

POSITION SUMMARY
```

```text
     component  position          token  mean_recovery  median_recovery  \
38        L9H9         4              =       0.156555         0.247765
1   10_mlp_out         1              2       0.087999         0.087999
12  10_mlp_out         4              =       0.061110         0.070912
25       L10H2         4              =       0.039181         0.121763
7   10_mlp_out         3              2       0.036951         0.036951
8   10_mlp_out         3              3       0.012736         0.012736
23       L10H2         3              6       0.006775         0.006775
10  10_mlp_out         3              6       0.004524         0.004524
6   10_mlp_out         2              +       0.002974         0.002382
32        L9H9         2              +       0.002687         0.003027
2   10_mlp_out         1              3       0.002074         0.002074
24       L10H2         3              8       0.001014         0.001014
17       L10H2         1              6       0.000987         0.000987
3   10_mlp_out         1              5       0.000534         0.000534
22       L10H2         3              5       0.000423         0.000423
18       L10H2         1              8       0.000299         0.000299
19       L10H2         2              +       0.000294         0.000810
37        L9H9         3              8       0.000203         0.000203
15       L10H2         1              3       0.000140         0.000140
28        L9H9         1              3       0.000105         0.000105
31        L9H9         1              8       0.000076         0.000076
16       L10H2         1              5       0.000053         0.000053
29        L9H9         1              5       0.000048         0.000048
34        L9H9         3              3       0.000035         0.000035
30        L9H9         1              6       0.000005         0.000005
0   10_mlp_out         0  <|endoftext|>       0.000000         0.000000
13       L10H2         0  <|endoftext|>       0.000000         0.000000
26        L9H9         0  <|endoftext|>       0.000000         0.000000
35        L9H9         3              5      -0.000103        -0.000103
27        L9H9         1              2      -0.000204        -0.000204
36        L9H9         3              6      -0.000571        -0.000571
14       L10H2         1              2      -0.000872        -0.000872
5   10_mlp_out         1              8      -0.001053        -0.001053
11  10_mlp_out         3              8      -0.001412        -0.001412
9   10_mlp_out         3              5      -0.002171        -0.002171
21       L10H2         3              3      -0.002403        -0.002403
20       L10H2         3              2      -0.004222        -0.004222
4   10_mlp_out         1              6      -0.004273        -0.004273
33        L9H9         3              2      -0.006401        -0.006401

    sd_recovery  n_valid
38     0.303437        5
1           NaN        1
12     0.170177        5
25     0.391910        5
7           NaN        1
8           NaN        1
23          NaN        1
10          NaN        1
6      0.007547        5
32     0.001831        5
2           NaN        1
24          NaN        1
17          NaN        1
3           NaN        1
22          NaN        1
18          NaN        1
19     0.004315        5
37          NaN        1
15          NaN        1
28          NaN        1
31          NaN        1
16          NaN        1
29          NaN        1
34          NaN        1
30          NaN        1
0      0.000000        5
13     0.000000        5
26     0.000000        5
35          NaN        1
27          NaN        1
36          NaN        1
14          NaN        1
5           NaN        1
11          NaN        1
9           NaN        1
21          NaN        1
20          NaN        1
4           NaN        1
33          NaN        1
```

```text

Saved:
  equal_operand_position_patch.csv
  equal_operand_position_patch_summary.csv

Interpretation note: position-wise recovery is exploratory. A large recovery at a position does not, by itself, prove that the corresponding token stores or computes the effect.
```


### Cell 157 — code (execution count: 116)

```python
# ============================================================
# L11 — ATTENTION-WEIGHT EDGE PERTURBATIONS (EXPLORATORY)
# ============================================================

import numpy as np
import pandas as pd
import torch

model.reset_hooks()

if "position_candidates" not in globals():
    raise RuntimeError("Run corrected L10 first.")

head_candidates = []
for component in position_candidates:
    spec = parse_component_l10(component)
    if spec["type"] == "head":
        head_candidates.append((component, spec["layer"], spec["head"]))

print("Attention-head candidates from corrected L10:", [x[0] for x in head_candidates])

if not head_candidates:
    print("No attention-head candidate was selected; L11 is skipped rather than forcing L10H7.")
    df_attention_edges = pd.DataFrame()
    edge_summary = pd.DataFrame()
    df_attention_edges.to_csv("equal_operand_attention_edge_intervention.csv", index=False)
    edge_summary.to_csv("equal_operand_attention_edge_intervention_summary.csv", index=False)
else:
    results = []

    def make_edge_knockout_hook(head, source_pos):
        def hook_fn(pattern, hook):
            # pattern shape: [batch, head, destination, source]
            pattern = pattern.clone()
            row = pattern[:, head, -1, :].clone()
            row[:, source_pos] = 0.0
            denom = row.sum(dim=-1, keepdim=True)
            row = row / denom.clamp_min(1e-8)
            pattern[:, head, -1, :] = row
            return pattern
        return hook_fn

    for component, layer, head in head_candidates:
        hook_name = f"blocks.{layer}.attn.hook_pattern"
        for T in DISCOVERY_TARGETS:
            d = T // 2
            clean_prompt = f"{d} + {d} ="
            corrupt_prompt = f"{d-1} + {d+1} ="
            clean_tokens = encode_prompt(clean_prompt)
            corrupt_tokens = encode_prompt(corrupt_prompt)

            clean_token_strings = model.to_str_tokens(clean_tokens[0])
            corrupt_token_strings = model.to_str_tokens(corrupt_tokens[0])
            if clean_tokens.shape[1] != corrupt_tokens.shape[1]:
                raise RuntimeError(f"Length mismatch at target {T}.")

            with torch.no_grad():
                clean_logits, clean_cache = model.run_with_cache(clean_tokens)
                corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)

            clean_score = float(score_from_last_logits(clean_logits[0, -1], T).item())
            corrupt_score = float(score_from_last_logits(corrupt_logits[0, -1], T).item())
            base_contrast = clean_score - corrupt_score
            clean_pattern = clean_cache[hook_name][0, head, -1, :].detach().cpu().numpy()
            corrupt_pattern = corrupt_cache[hook_name][0, head, -1, :].detach().cpu().numpy()

            for source_pos in range(int(clean_tokens.shape[1])):
                # Clean prompt with one incoming edge reweighted out.
                model.reset_hooks()
                clean_hook = make_edge_knockout_hook(head, source_pos)
                with torch.no_grad():
                    clean_knockout_logits = model.run_with_hooks(
                        clean_tokens,
                        fwd_hooks=[(hook_name, clean_hook)]
                    )
                clean_knockout_score = float(score_from_last_logits(clean_knockout_logits[0, -1], T).item())

                # Corrupt prompt with corresponding position's edge reweighted out.
                model.reset_hooks()
                corrupt_hook = make_edge_knockout_hook(head, source_pos)
                with torch.no_grad():
                    corrupt_knockout_logits = model.run_with_hooks(
                        corrupt_tokens,
                        fwd_hooks=[(hook_name, corrupt_hook)]
                    )
                corrupt_knockout_score = float(score_from_last_logits(corrupt_knockout_logits[0, -1], T).item())

                new_contrast = clean_knockout_score - corrupt_knockout_score
                results.append({
                    "component": component,
                    "layer": layer,
                    "head": head,
                    "target": T,
                    "source_position": source_pos,
                    "clean_source_token": clean_token_strings[source_pos],
                    "corrupt_source_token": corrupt_token_strings[source_pos],
                    "clean_attention_weight": float(clean_pattern[source_pos]),
                    "corrupt_attention_weight": float(corrupt_pattern[source_pos]),
                    "base_contrast": base_contrast,
                    "contrast_after_knockout": new_contrast,
                    "contrast_change_due_to_knockout": base_contrast - new_contrast,
                })

            del clean_cache, corrupt_cache
            model.reset_hooks()

    df_attention_edges = pd.DataFrame(results)
    if not df_attention_edges.empty:
        edge_summary = (
            df_attention_edges.groupby(["component", "source_position"], as_index=False)
            .agg(
                mean_clean_attention=("clean_attention_weight", "mean"),
                mean_corrupt_attention=("corrupt_attention_weight", "mean"),
                mean_contrast_change=("contrast_change_due_to_knockout", "mean"),
                median_contrast_change=("contrast_change_due_to_knockout", "median"),
                n_targets=("contrast_change_due_to_knockout", "count"),
                n_positive=("contrast_change_due_to_knockout", lambda s: int((s > 0).sum())),
                n_negative=("contrast_change_due_to_knockout", lambda s: int((s < 0).sum())),
            )
            .sort_values("mean_contrast_change", ascending=False)
        )
        display(df_attention_edges)
        display(edge_summary)
    else:
        edge_summary = pd.DataFrame()

    df_attention_edges.to_csv("equal_operand_attention_edge_intervention.csv", index=False)
    edge_summary.to_csv("equal_operand_attention_edge_intervention_summary.csv", index=False)
    model.reset_hooks()
    print("Saved attention-edge tables.")

print("Interpretation: this is an exploratory attention-weight perturbation; a large attention weight alone is not causal evidence.")
```

#### Recorded output

```text
Attention-head candidates from corrected L10: ['L9H9', 'L10H2']
```

```text
   component  layer  head  target  source_position clean_source_token  \
0       L9H9      9     9       4                0      <|endoftext|>
1       L9H9      9     9       4                1                  2
2       L9H9      9     9       4                2                  +
3       L9H9      9     9       4                3                  2
4       L9H9      9     9       4                4                  =
5       L9H9      9     9       6                0      <|endoftext|>
6       L9H9      9     9       6                1                  3
7       L9H9      9     9       6                2                  +
8       L9H9      9     9       6                3                  3
9       L9H9      9     9       6                4                  =
10      L9H9      9     9      10                0      <|endoftext|>
11      L9H9      9     9      10                1                  5
12      L9H9      9     9      10                2                  +
13      L9H9      9     9      10                3                  5
14      L9H9      9     9      10                4                  =
15      L9H9      9     9      12                0      <|endoftext|>
16      L9H9      9     9      12                1                  6
17      L9H9      9     9      12                2                  +
18      L9H9      9     9      12                3                  6
19      L9H9      9     9      12                4                  =
20      L9H9      9     9      16                0      <|endoftext|>
21      L9H9      9     9      16                1                  8
22      L9H9      9     9      16                2                  +
23      L9H9      9     9      16                3                  8
24      L9H9      9     9      16                4                  =
25     L10H2     10     2       4                0      <|endoftext|>
26     L10H2     10     2       4                1                  2
27     L10H2     10     2       4                2                  +
28     L10H2     10     2       4                3                  2
29     L10H2     10     2       4                4                  =
30     L10H2     10     2       6                0      <|endoftext|>
31     L10H2     10     2       6                1                  3
32     L10H2     10     2       6                2                  +
33     L10H2     10     2       6                3                  3
34     L10H2     10     2       6                4                  =
35     L10H2     10     2      10                0      <|endoftext|>
36     L10H2     10     2      10                1                  5
37     L10H2     10     2      10                2                  +
38     L10H2     10     2      10                3                  5
39     L10H2     10     2      10                4                  =
40     L10H2     10     2      12                0      <|endoftext|>
41     L10H2     10     2      12                1                  6
42     L10H2     10     2      12                2                  +
43     L10H2     10     2      12                3                  6
44     L10H2     10     2      12                4                  =
45     L10H2     10     2      16                0      <|endoftext|>
46     L10H2     10     2      16                1                  8
47     L10H2     10     2      16                2                  +
48     L10H2     10     2      16                3                  8
49     L10H2     10     2      16                4                  =

   corrupt_source_token  clean_attention_weight  corrupt_attention_weight  \
0         <|endoftext|>                0.809362                  0.891826
1                     1                0.096853                  0.026713
2                     +                0.020896                  0.010882
3                     3                0.068676                  0.067837
4                     =                0.004213                  0.002743
5         <|endoftext|>                0.766426                  0.850446
6                     2                0.096101                  0.042411
7                     +                0.022503                  0.009887
8                     4                0.110344                  0.093814
9                     =                0.004626                  0.003443
10        <|endoftext|>                0.806902                  0.876022
11                    4                0.105196                  0.054824
12                    +                0.022109                  0.010480
13                    6                0.061042                  0.055329
14                    =                0.004752                  0.003344
15        <|endoftext|>                0.756682                  0.829248
16                    5                0.144825                  0.077140
17                    +                0.018312                  0.014596
18                    7                0.075841                  0.075111
19                    =                0.004340                  0.003905
20        <|endoftext|>                0.715161                  0.811050
21                    7                0.173529                  0.089136
22                    +                0.010050                  0.010083
23                    9                0.097581                  0.086152
24                    =                0.003679                  0.003578
25        <|endoftext|>                0.722968                  0.698934
26                    1                0.029819                  0.012742
27                    +                0.112745                  0.130323
28                    3                0.096138                  0.112311
29                    =                0.038329                  0.045690
30        <|endoftext|>                0.717713                  0.717036
31                    2                0.023902                  0.018008
32                    +                0.126433                  0.118027
33                    4                0.092234                  0.091046
34                    =                0.039717                  0.055882
35        <|endoftext|>                0.689366                  0.756302
36                    4                0.045121                  0.027319
37                    +                0.147772                  0.107625
38                    6                0.077212                  0.066904
39                    =                0.040529                  0.041850
40        <|endoftext|>                0.739668                  0.694666
41                    5                0.040684                  0.033982
42                    +                0.105723                  0.151430
43                    7                0.075266                  0.077627
44                    =                0.038660                  0.042296
45        <|endoftext|>                0.790880                  0.757674
46                    7                0.041744                  0.038977
47                    +                0.077485                  0.103394
48                    9                0.054969                  0.058078
49                    =                0.034922                  0.041878

    base_contrast  contrast_after_knockout  contrast_change_due_to_knockout
0       -0.070020                 0.011945                        -0.081964
1       -0.070020                -0.075661                         0.005641
2       -0.070020                -0.068457                        -0.001563
3       -0.070020                -0.088642                         0.018622
4       -0.070020                -0.069534                        -0.000485
5       -0.136127                -0.145909                         0.009782
6       -0.136127                -0.126346                        -0.009782
7       -0.136127                -0.140121                         0.003994
8       -0.136127                -0.129234                        -0.006893
9       -0.136127                -0.136877                         0.000750
10       0.537206                 0.682384                        -0.145179
11       0.537206                 0.507927                         0.029279
12       0.537206                 0.539796                        -0.002590
13       0.537206                 0.526489                         0.010716
14       0.537206                 0.538458                        -0.001252
15       0.178778                 0.445922                        -0.267144
16       0.178778                 0.141941                         0.036837
17       0.178778                 0.181993                        -0.003216
18       0.178778                 0.135736                         0.043041
19       0.178778                 0.179402                        -0.000625
20       0.637604                 1.022183                        -0.384580
21       0.637604                 0.539090                         0.098514
22       0.637604                 0.640410                        -0.002807
23       0.637604                 0.594014                         0.043590
24       0.637604                 0.638744                        -0.001141
25      -0.070020                 0.002584                        -0.072603
26      -0.070020                -0.073479                         0.003459
27      -0.070020                -0.074989                         0.004970
28      -0.070020                -0.081262                         0.011242
29      -0.070020                -0.073978                         0.003959
30      -0.136127                -0.200523                         0.064396
31      -0.136127                -0.135083                        -0.001044
32      -0.136127                -0.136754                         0.000627
33      -0.136127                -0.117864                        -0.018264
34      -0.136127                -0.135236                        -0.000892
35       0.537206                 0.764230                        -0.227024
36       0.537206                 0.524809                         0.012397
37       0.537206                 0.511021                         0.026185
38       0.537206                 0.500332                         0.036874
39       0.537206                 0.538636                        -0.001431
40       0.178778                 0.373538                        -0.194760
41       0.178778                 0.163815                         0.014963
42       0.178778                 0.187119                        -0.008341
43       0.178778                 0.129721                         0.049057
44       0.178778                 0.179894                        -0.001117
45       0.637604                 0.977075                        -0.339471
46       0.637604                 0.603223                         0.034381
47       0.637604                 0.629570                         0.008034
48       0.637604                 0.603037                         0.034567
49       0.637604                 0.637184                         0.000420
```

```text
  component  source_position  mean_clean_attention  mean_corrupt_attention  \
6      L9H9                1              0.123301                0.058045
3     L10H2                3              0.079164                0.081193
8      L9H9                3              0.082697                0.075649
1     L10H2                1              0.036254                0.026205
2     L10H2                2              0.114032                0.122160
4     L10H2                4              0.038432                0.045519
9      L9H9                4              0.004322                0.003403
7      L9H9                2              0.018774                0.011186
0     L10H2                0              0.732119                0.724922
5      L9H9                0              0.770906                0.851718

   mean_contrast_change  median_contrast_change  n_targets  n_positive  \
6              0.032098                0.029279          5           4
3              0.022695                0.034567          5           4
8              0.021815                0.018622          5           4
1              0.012831                0.012397          5           4
2              0.006295                0.004970          5           4
4              0.000188               -0.000892          5           2
9             -0.000551               -0.000625          5           1
7             -0.001236               -0.002590          5           1
0             -0.153893               -0.194760          5           1
5             -0.173817               -0.145179          5           1

   n_negative
6           1
3           1
8           1
1           1
2           1
4           3
9           4
7           4
0           4
5           4
```

```text
Saved attention-edge tables.
Interpretation: this is an exploratory attention-weight perturbation; a large attention weight alone is not causal evidence.
```


### Cell 158 — code (execution count: 117)

```python
# ============================================================
# SHARED CIRCUIT HELPERS — CANONICAL BOS AND SCORING CONTRACT
# ============================================================

import numpy as np
import pandas as pd
import torch


def parse_component(component):
    return parse_component_l7(component)


def make_mean_ablation_hook(component, reference_vector):
    return make_mean_ablation_hook_l7(component, reference_vector)


def evaluate_prompt_with_ablation(prompt, target, components):
    tokens = encode_prompt(prompt)
    hooks = []
    for component in components:
        if component not in candidate_reference_vectors:
            raise KeyError(f"No reference vector for {component}")
        hook_name, hook_fn = make_mean_ablation_hook(
            component, candidate_reference_vectors[component]
        )
        hooks.append((hook_name, hook_fn))

    with torch.no_grad():
        if hooks:
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        else:
            logits = model(tokens, return_type="logits")

    return float(score_from_last_logits(logits[0, -1], int(target)).item())


def evaluate_target_level_advantage(raw_df, components):
    required = {"target", "condition", "prompt"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"Expected raw prompt-level data; missing {sorted(missing)}")

    rows = []
    for target in sorted(raw_df["target"].astype(int).unique()):
        data = raw_df[raw_df["target"].astype(int) == target]
        doubles = data[data["condition"] == "double"]
        controls = data[data["condition"] == "control"]
        if len(doubles) != 1 or len(controls) < 2:
            raise ValueError(f"Target {target}: invalid double/control structure")

        double_score = evaluate_prompt_with_ablation(
            str(doubles.iloc[0]["prompt"]), int(target), components
        )
        control_scores = [
            evaluate_prompt_with_ablation(str(r["prompt"]), int(target), components)
            for _, r in controls.iterrows()
        ]
        control_mean = float(np.mean(control_scores))
        rows.append({
            "target": int(target),
            "double_score": float(double_score),
            "control_mean": control_mean,
            "advantage": float(double_score - control_mean),
            "n_controls": len(control_scores),
        })
    return pd.DataFrame(rows)

print("Shared circuit helpers ready; explicit BOS and true target token IDs are enforced.")
```

#### Recorded output

```text
Shared circuit helpers ready; explicit BOS and true target token IDs are enforced.
```


### Cell 159 — code (execution count: 118)

```python
# ============================================================
# Mechanistic analysis of the equal-operand effect
# L12 — Proposed circuit ablation
#
# Purpose:
#   Build a candidate circuit ONLY from components that were
#   already identified in the preceding DLA + causal screen,
#   then measure the effect of jointly ablating them.
#
# IMPORTANT DATA NAMING:
#   df_circuit_discovery_raw
#       = individual discovery prompts
#
#   df_circuit_discovery_results
#       = target-level aggregated circuit-ablation results
#
#   These variables are kept separate and are NEVER overwritten.
# ============================================================

import numpy as np
import pandas as pd
import torch


# ------------------------------------------------------------
# 1. Recover the RAW discovery dataframe
# ------------------------------------------------------------

if "df_circuit" not in globals():
    raise RuntimeError(
        "df_circuit does not exist. "
        "Run the canonical circuit dataset cell first."
    )

df_circuit_discovery_raw = (
    df_circuit[
        df_circuit["split"] == "discovery"
    ]
    .copy()
    .reset_index(drop=True)
)


required_raw_columns = {
    "target",
    "condition",
    "prompt",
}

missing_raw = (
    required_raw_columns
    - set(df_circuit_discovery_raw.columns)
)

if missing_raw:
    raise ValueError(
        "Raw discovery dataframe is missing columns: "
        f"{sorted(missing_raw)}"
    )


# ------------------------------------------------------------
# 2. Check that the preceding causal analysis exists
# ------------------------------------------------------------

if "df_causal_summary" not in globals():
    raise RuntimeError(
        "df_causal_summary does not exist. "
        "Run L8 before L12."
    )

required_causal_columns = {
    "component",
    "mean_advantage_change",
}

missing_causal = (
    required_causal_columns
    - set(df_causal_summary.columns)
)

if missing_causal:
    raise ValueError(
        "df_causal_summary is missing columns: "
        f"{sorted(missing_causal)}"
    )


# ------------------------------------------------------------
# 3. Check that candidate reference vectors exist
# ------------------------------------------------------------

if "candidate_reference_vectors" not in globals():
    raise RuntimeError(
        "candidate_reference_vectors does not exist. "
        "Run L7 before L12."
    )


# ------------------------------------------------------------
# 4. Select candidate components
#
# We require the component to have:
#   - a measured causal effect on the equal-operand contrast
#   - a NEGATIVE mean advantage change
#
# Negative means that ablation reduced the equal-operand
# advantage, which is the direction expected for a component
# supporting the phenomenon.
#
# We do NOT select candidates from raw DLA alone.
# ------------------------------------------------------------

candidate_table = (
    df_causal_summary[
        df_causal_summary["component"].isin(
            candidate_reference_vectors.keys()
        )
    ]
    .copy()
)

candidate_table = candidate_table[
    np.isfinite(
        candidate_table["mean_advantage_change"]
    )
].copy()

negative_candidates = (
    candidate_table[
        candidate_table[
            "mean_advantage_change"
        ] < 0
    ]
    .sort_values(
        "mean_advantage_change",
        ascending=True
    )
)

# ------------------------------------------------------------
# Require at least 3 negative candidates.
#
# If fewer exist, STOP rather than constructing an arbitrary
# three-component circuit.
# ------------------------------------------------------------

if len(negative_candidates) < 3:
    print("=" * 78)
    print("L12 — CANDIDATE CIRCUIT ABORTED")
    print("=" * 78)

    print(
        f"Only {len(negative_candidates)} candidate(s) "
        "show a negative mean causal change."
    )

    print(
        "\nNo proposed three-component circuit will be "
        "constructed automatically."
    )

    print(
        "\nThis is scientifically preferable to selecting "
        "components whose causal effect does not support "
        "the proposed mechanism."
    )

    display(
        candidate_table[
            [
                "component",
                "mean_advantage_change",
            ]
        ]
        .sort_values(
            "mean_advantage_change"
        )
    )

    raise RuntimeError(
        "Insufficient negative causal candidates "
        "for an automatic three-component circuit."
    )


# ------------------------------------------------------------
# 5. Define the proposed circuit
# ------------------------------------------------------------

N_CIRCUIT_COMPONENTS = 3

PROPOSED_CIRCUIT = (
    negative_candidates
    .head(N_CIRCUIT_COMPONENTS)
    ["component"]
    .tolist()
)


print("=" * 78)
print("L12 — PROPOSED CIRCUIT")
print("=" * 78)

for rank, component in enumerate(
    PROPOSED_CIRCUIT,
    start=1
):

    effect = float(
        candidate_table.loc[
            candidate_table["component"]
            == component,
            "mean_advantage_change"
        ].iloc[0]
    )

    print(
        f"{rank}. {component:<12} "
        f"mean causal change = {effect:+.6f}"
    )


# ------------------------------------------------------------
# 6. Helper: evaluate one prompt with a joint mean ablation
# ------------------------------------------------------------

def evaluate_prompt_with_joint_ablation(prompt, T, components):
    tokens = encode_prompt(prompt)
    hooks = []
    for component in components:
        if component not in candidate_reference_vectors:
            raise KeyError(f"No reference vector available for {component}")
        hook_name, hook_fn = make_mean_ablation_hook_l7(
            component, candidate_reference_vectors[component]
        )
        hooks.append((hook_name, hook_fn))

    with torch.no_grad():
        if hooks:
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        else:
            logits = model(tokens, return_type="logits")

    return float(score_from_last_logits(logits[0, -1], int(T)).item())

# ------------------------------------------------------------
# 7. Compute FULL-MODEL discovery effects
#
# This is kept separate from the ablated result dataframe.
# ------------------------------------------------------------

full_model_rows = []

for T in sorted(
    df_circuit_discovery_raw[
        "target"
    ].unique()
):

    target_data = (
        df_circuit_discovery_raw[
            df_circuit_discovery_raw[
                "target"
            ] == T
        ]
    )

    double_rows = target_data[
        target_data["condition"]
        == "double"
    ]

    control_rows = target_data[
        target_data["condition"]
        == "control"
    ]

    if len(double_rows) != 1:
        raise ValueError(
            f"T={T}: expected exactly one double "
            f"prompt, found {len(double_rows)}."
        )

    if len(control_rows) < 2:
        raise ValueError(
            f"T={T}: too few control prompts "
            f"({len(control_rows)})."
        )

    double_prompt = (
        double_rows.iloc[0]["prompt"]
    )

    double_score = primary_score(
        double_prompt,
        int(T)
    )

    control_scores = [
        primary_score(
            row["prompt"],
            int(T)
        )
        for _, row in control_rows.iterrows()
    ]

    control_mean = float(
        np.mean(control_scores)
    )

    full_model_rows.append({
        "target": int(T),
        "double_score": float(
            double_score
        ),
        "control_mean": control_mean,
        "advantage": (
            float(double_score)
            - control_mean
        ),
    })


df_full_discovery = pd.DataFrame(
    full_model_rows
)


# ------------------------------------------------------------
# 8. Jointly ablate the proposed circuit
# ------------------------------------------------------------

circuit_rows = []

for T in sorted(
    df_circuit_discovery_raw[
        "target"
    ].unique()
):

    target_data = (
        df_circuit_discovery_raw[
            df_circuit_discovery_raw[
                "target"
            ] == T
        ]
    )

    double_rows = target_data[
        target_data["condition"]
        == "double"
    ]

    control_rows = target_data[
        target_data["condition"]
        == "control"
    ]

    double_prompt = (
        double_rows.iloc[0]["prompt"]
    )

    # --------------------------------------------------------
    # Ablated double
    # --------------------------------------------------------

    double_ablated = (
        evaluate_prompt_with_joint_ablation(
            double_prompt,
            int(T),
            PROPOSED_CIRCUIT
        )
    )

    # --------------------------------------------------------
    # Ablated controls
    # --------------------------------------------------------

    control_ablated_scores = []

    for _, control_row in (
        control_rows.iterrows()
    ):

        control_score = (
            evaluate_prompt_with_joint_ablation(
                control_row["prompt"],
                int(T),
                PROPOSED_CIRCUIT
            )
        )

        control_ablated_scores.append(
            control_score
        )

    control_ablated_mean = float(
        np.mean(
            control_ablated_scores
        )
    )

    ablated_advantage = (
        double_ablated
        - control_ablated_mean
    )

    circuit_rows.append({
        "target": int(T),
        "double_score": double_ablated,
        "control_mean": control_ablated_mean,
        "advantage": ablated_advantage,
    })


# IMPORTANT:
# This is the aggregated target-level result.
df_circuit_discovery_results = (
    pd.DataFrame(
        circuit_rows
    )
)


# ------------------------------------------------------------
# 9. Align full and ablated effects
# ------------------------------------------------------------

comparison = (
    df_full_discovery[
        [
            "target",
            "advantage"
        ]
    ]
    .rename(
        columns={
            "advantage":
                "full_effect"
        }
    )
    .merge(
        df_circuit_discovery_results[
            [
                "target",
                "advantage"
            ]
        ].rename(
            columns={
                "advantage":
                    "circuit_ablated_effect"
            }
        ),
        on="target",
        how="inner"
    )
)

if comparison.empty:
    raise RuntimeError(
        "No overlapping target levels between "
        "full-model and circuit-ablated results."
    )


# ------------------------------------------------------------
# 10. Calculate effect removed
# ------------------------------------------------------------

comparison[
    "effect_removed"
] = (
    comparison["full_effect"]
    - comparison["circuit_ablated_effect"]
)

comparison[
    "fraction_removed"
] = np.where(
    comparison["full_effect"].abs() > 1e-8,

    comparison["effect_removed"]
    / comparison["full_effect"],

    np.nan
)


# ------------------------------------------------------------
# 11. Summary statistics
# ------------------------------------------------------------

full_mean = (
    comparison["full_effect"]
    .mean()
)

circuit_mean = (
    comparison["circuit_ablated_effect"]
    .mean()
)

mean_effect_removed = (
    comparison["effect_removed"]
    .mean()
)

mean_fraction_removed = (
    comparison["fraction_removed"]
    .mean()
)
removal_fraction = mean_fraction_removed

# ------------------------------------------------------------
# 12. Display
# ------------------------------------------------------------

print("\n" + "=" * 78)
print("L12 — PROPOSED CIRCUIT ABLATION RESULTS")
print("=" * 78)

print(
    "\nTarget-level comparison:"
)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x:
            f"{x:+.6f}"
    )
)

print("\n" + "-" * 78)

print(
    f"Full-model mean advantage       : "
    f"{full_mean:+.6f}"
)

print(
    f"Circuit-ablated mean advantage  : "
    f"{circuit_mean:+.6f}"
)

print(
    f"Mean effect removed              : "
    f"{mean_effect_removed:+.6f}"
)

print(
    f"Mean fraction removed            : "
    f"{mean_fraction_removed:+.6f}"
)


# ------------------------------------------------------------
# 13. Save results
# ------------------------------------------------------------

comparison.to_csv(
    "equal_operand_proposed_circuit_effects.csv",
    index=False
)

df_circuit_discovery_results.to_csv(
    "equal_operand_proposed_circuit_discovery.csv",
    index=False
)

pd.DataFrame({
    "component": PROPOSED_CIRCUIT
}).to_csv(
    "equal_operand_proposed_circuit_components.csv",
    index=False
)


print("\nSaved:")
print(
    "  equal_operand_proposed_circuit_effects.csv"
)
print(
    "  equal_operand_proposed_circuit_discovery.csv"
)
print(
    "  equal_operand_proposed_circuit_components.csv"
)
```

#### Recorded output

```text
==============================================================================
L12 — PROPOSED CIRCUIT
==============================================================================
1. 10_mlp_out   mean causal change = -0.078840
2. L9H9         mean causal change = -0.070940
3. L10H2        mean causal change = -0.058887

==============================================================================
L12 — PROPOSED CIRCUIT ABLATION RESULTS
==============================================================================

Target-level comparison:
 target  full_effect  circuit_ablated_effect  effect_removed  fraction_removed
      4    +0.048633               -0.026336       +0.074969         +1.541528
      6    +0.016807               +0.002678       +0.014129         +0.840681
     10    +0.534628               +0.174941       +0.359687         +0.672779
     12    +0.164717               +0.020338       +0.144380         +0.876530
     16    +0.682146               +0.315137       +0.367009         +0.538021

------------------------------------------------------------------------------
Full-model mean advantage       : +0.289386
Circuit-ablated mean advantage  : +0.097351
Mean effect removed              : +0.192035
Mean fraction removed            : +0.893908

Saved:
  equal_operand_proposed_circuit_effects.csv
  equal_operand_proposed_circuit_discovery.csv
  equal_operand_proposed_circuit_components.csv
```


### Cell 160 — code (execution count: 119)

```python
if "PROPOSED_CIRCUIT" not in globals() or not PROPOSED_CIRCUIT:
    raise RuntimeError("L12 did not produce a candidate set; no transfer test can run.")

FROZEN_TRANSFER_CIRCUIT = tuple(PROPOSED_CIRCUIT)
pd.DataFrame({
    "component": list(FROZEN_TRANSFER_CIRCUIT),
    "status": "frozen_before_prospective_range_transfer_test",
}).to_csv("equal_operand_frozen_transfer_circuit.csv", index=False)
print("Frozen circuit:", FROZEN_TRANSFER_CIRCUIT)
```

#### Recorded output

```text
Frozen circuit: ('10_mlp_out', 'L9H9', 'L10H2')
```


### Cell 161 — code (execution count: 120)

```python
# ============================================================
# PROSPECTIVE RANGE-TRANSFER TEST — NOT INDEPENDENT CONFIRMATION
# ============================================================

import numpy as np
import pandas as pd
import torch

NEW_RANGE_TARGETS = [24, 26, 28, 30, 32, 34, 36]
OPERANDS = range(10, 20)

if "FROZEN_TRANSFER_CIRCUIT" not in globals():
    raise RuntimeError("Run the candidate-freezing cell first.")

# Fail before scoring if any outcome/foil is not exactly one token.
for target in NEW_RANGE_TARGETS:
    for number in (target - 1, target, target + 1):
        answer_token_id(number)


def score_prompt_list(prompts, targets, components=()):
    """Scores variable-length prompts safely by batching equal lengths."""
    prompts = list(prompts)
    targets = [int(t) for t in targets]
    if len(prompts) != len(targets):
        raise ValueError("Prompt/target lengths differ.")

    encoded = [encode_prompt(p) for p in prompts]
    groups = {}
    for i, tokens in enumerate(encoded):
        groups.setdefault(int(tokens.shape[1]), []).append(i)

    hooks = []
    for component in components:
        if component not in candidate_reference_vectors:
            raise KeyError(f"Frozen component has no reference vector: {component}")
        name, fn = make_mean_ablation_hook_l7(
            component, candidate_reference_vectors[component]
        )
        hooks.append((name, fn))

    out = np.empty(len(prompts), dtype=float)
    for _, indices in groups.items():
        batch = torch.cat([encoded[i] for i in indices], dim=0)
        group_targets = [targets[i] for i in indices]
        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(batch, fwd_hooks=hooks)
            else:
                logits = model(batch, return_type="logits")
        scores = batch_primary_scores(logits, group_targets).detach().cpu().numpy()
        for j, original_i in enumerate(indices):
            out[original_i] = float(scores[j])
    return out


prompt_rows = []
summary_rows = []

for target in NEW_RANGE_TARGETS:
    d = target // 2
    double_prompt = f"{d} + {d} ="
    controls = [
        (a, target - a)
        for a in OPERANDS
        if 10 <= target - a <= 19 and a != target - a
    ]
    if len(controls) < 2:
        raise RuntimeError(f"Too few matched non-double controls for target {target}.")

    prompts = [double_prompt] + [f"{a} + {b} =" for a, b in controls]
    targets = [target] * len(prompts)
    conditions = ["double"] + ["control"] * len(controls)

    clean_scores = score_prompt_list(prompts, targets, components=())
    ablated_scores = score_prompt_list(
        prompts, targets, components=FROZEN_TRANSFER_CIRCUIT
    )

    for i, prompt in enumerate(prompts):
        token_ids = encode_prompt(prompt)[0].detach().cpu().tolist()
        prompt_rows.append({
            "target": target,
            "condition": conditions[i],
            "prompt": prompt,
            "token_ids_with_bos": repr(token_ids),
            "token_strings_with_bos": repr(model.to_str_tokens(encode_prompt(prompt)[0])),
            "clean_score": float(clean_scores[i]),
            "ablated_score": float(ablated_scores[i]),
            "score_change": float(ablated_scores[i] - clean_scores[i]),
        })

    clean_adv = float(clean_scores[0] - clean_scores[1:].mean())
    ablated_adv = float(ablated_scores[0] - ablated_scores[1:].mean())
    summary_rows.append({
        "target": target,
        "n_controls": len(controls),
        "clean_advantage": clean_adv,
        "ablated_advantage": ablated_adv,
        "advantage_change": ablated_adv - clean_adv,
        "fraction_removed": (clean_adv - ablated_adv) / clean_adv if abs(clean_adv) > 1e-8 else np.nan,
        "evidence_status": "prospective_range_transfer_not_independent_confirmation",
    })

df_newrange_prompts = pd.DataFrame(prompt_rows)
df_newrange_summary = pd.DataFrame(summary_rows)
display(df_newrange_prompts)
display(df_newrange_summary)
df_newrange_prompts.to_csv("equal_operand_new_range_prompt_results.csv", index=False)
df_newrange_summary.to_csv("equal_operand_new_range_target_summary.csv", index=False)
print("Saved the prospective range-transfer prompt and summary tables.")
```

#### Recorded output

```text
    target condition     prompt              token_ids_with_bos  \
0       24    double  12 + 12 =  [50256, 1065, 1343, 1105, 796]
1       24   control  10 + 14 =   [50256, 940, 1343, 1478, 796]
2       24   control  11 + 13 =  [50256, 1157, 1343, 1511, 796]
3       24   control  13 + 11 =  [50256, 1485, 1343, 1367, 796]
4       24   control  14 + 10 =   [50256, 1415, 1343, 838, 796]
5       26    double  13 + 13 =  [50256, 1485, 1343, 1511, 796]
6       26   control  10 + 16 =   [50256, 940, 1343, 1467, 796]
7       26   control  11 + 15 =  [50256, 1157, 1343, 1315, 796]
8       26   control  12 + 14 =  [50256, 1065, 1343, 1478, 796]
9       26   control  14 + 12 =  [50256, 1415, 1343, 1105, 796]
10      26   control  15 + 11 =  [50256, 1314, 1343, 1367, 796]
11      26   control  16 + 10 =   [50256, 1433, 1343, 838, 796]
12      28    double  14 + 14 =  [50256, 1415, 1343, 1478, 796]
13      28   control  10 + 18 =   [50256, 940, 1343, 1248, 796]
14      28   control  11 + 17 =  [50256, 1157, 1343, 1596, 796]
15      28   control  12 + 16 =  [50256, 1065, 1343, 1467, 796]
16      28   control  13 + 15 =  [50256, 1485, 1343, 1315, 796]
17      28   control  15 + 13 =  [50256, 1314, 1343, 1511, 796]
18      28   control  16 + 12 =  [50256, 1433, 1343, 1105, 796]
19      28   control  17 + 11 =  [50256, 1558, 1343, 1367, 796]
20      28   control  18 + 10 =   [50256, 1507, 1343, 838, 796]
21      30    double  15 + 15 =  [50256, 1314, 1343, 1315, 796]
22      30   control  11 + 19 =   [50256, 1157, 1343, 678, 796]
23      30   control  12 + 18 =  [50256, 1065, 1343, 1248, 796]
24      30   control  13 + 17 =  [50256, 1485, 1343, 1596, 796]
25      30   control  14 + 16 =  [50256, 1415, 1343, 1467, 796]
26      30   control  16 + 14 =  [50256, 1433, 1343, 1478, 796]
27      30   control  17 + 13 =  [50256, 1558, 1343, 1511, 796]
28      30   control  18 + 12 =  [50256, 1507, 1343, 1105, 796]
29      30   control  19 + 11 =  [50256, 1129, 1343, 1367, 796]
30      32    double  16 + 16 =  [50256, 1433, 1343, 1467, 796]
31      32   control  13 + 19 =   [50256, 1485, 1343, 678, 796]
32      32   control  14 + 18 =  [50256, 1415, 1343, 1248, 796]
33      32   control  15 + 17 =  [50256, 1314, 1343, 1596, 796]
34      32   control  17 + 15 =  [50256, 1558, 1343, 1315, 796]
35      32   control  18 + 14 =  [50256, 1507, 1343, 1478, 796]
36      32   control  19 + 13 =  [50256, 1129, 1343, 1511, 796]
37      34    double  17 + 17 =  [50256, 1558, 1343, 1596, 796]
38      34   control  15 + 19 =   [50256, 1314, 1343, 678, 796]
39      34   control  16 + 18 =  [50256, 1433, 1343, 1248, 796]
40      34   control  18 + 16 =  [50256, 1507, 1343, 1467, 796]
41      34   control  19 + 15 =  [50256, 1129, 1343, 1315, 796]
42      36    double  18 + 18 =  [50256, 1507, 1343, 1248, 796]
43      36   control  17 + 19 =   [50256, 1558, 1343, 678, 796]
44      36   control  19 + 17 =  [50256, 1129, 1343, 1596, 796]

                        token_strings_with_bos  clean_score  ablated_score  \
0   ['<|endoftext|>', '12', ' +', ' 12', ' =']     0.443065       0.276733
1   ['<|endoftext|>', '10', ' +', ' 14', ' =']     0.037534       0.054862
2   ['<|endoftext|>', '11', ' +', ' 13', ' =']    -0.067835       0.020296
3   ['<|endoftext|>', '13', ' +', ' 11', ' =']    -0.124887      -0.029182
4   ['<|endoftext|>', '14', ' +', ' 10', ' =']     0.144678       0.105232
5   ['<|endoftext|>', '13', ' +', ' 13', ' =']    -0.243073      -0.203610
6   ['<|endoftext|>', '10', ' +', ' 16', ' =']    -0.238747      -0.264857
7   ['<|endoftext|>', '11', ' +', ' 15', ' =']    -0.190052      -0.184752
8   ['<|endoftext|>', '12', ' +', ' 14', ' =']    -0.063668      -0.135645
9   ['<|endoftext|>', '14', ' +', ' 12', ' =']    -0.080343      -0.152803
10  ['<|endoftext|>', '15', ' +', ' 11', ' =']    -0.137741      -0.159072
11  ['<|endoftext|>', '16', ' +', ' 10', ' =']    -0.043846      -0.167096
12  ['<|endoftext|>', '14', ' +', ' 14', ' =']     0.424971       0.217639
13  ['<|endoftext|>', '10', ' +', ' 18', ' =']     0.188051       0.112452
14  ['<|endoftext|>', '11', ' +', ' 17', ' =']     0.050157       0.041383
15  ['<|endoftext|>', '12', ' +', ' 16', ' =']     0.248533       0.195189
16  ['<|endoftext|>', '13', ' +', ' 15', ' =']     0.134798       0.148516
17  ['<|endoftext|>', '15', ' +', ' 13', ' =']     0.055504       0.046241
18  ['<|endoftext|>', '16', ' +', ' 12', ' =']     0.217598       0.164791
19  ['<|endoftext|>', '17', ' +', ' 11', ' =']     0.078145       0.027960
20  ['<|endoftext|>', '18', ' +', ' 10', ' =']     0.285995       0.148325
21  ['<|endoftext|>', '15', ' +', ' 15', ' =']     1.101715       0.877142
22  ['<|endoftext|>', '11', ' +', ' 19', ' =']     0.293061       0.400987
23  ['<|endoftext|>', '12', ' +', ' 18', ' =']     0.491315       0.523088
24  ['<|endoftext|>', '13', ' +', ' 17', ' =']     0.384706       0.476063
25  ['<|endoftext|>', '14', ' +', ' 16', ' =']     0.452418       0.491105
26  ['<|endoftext|>', '16', ' +', ' 14', ' =']     0.311951       0.353922
27  ['<|endoftext|>', '17', ' +', ' 13', ' =']     0.230803       0.332507
28  ['<|endoftext|>', '18', ' +', ' 12', ' =']     0.387423       0.434187
29  ['<|endoftext|>', '19', ' +', ' 11', ' =']     0.098639       0.229359
30  ['<|endoftext|>', '16', ' +', ' 16', ' =']     1.046242       0.741693
31  ['<|endoftext|>', '13', ' +', ' 19', ' =']     0.285569       0.330588
32  ['<|endoftext|>', '14', ' +', ' 18', ' =']     0.318117       0.328321
33  ['<|endoftext|>', '15', ' +', ' 17', ' =']     0.272956       0.349359
34  ['<|endoftext|>', '17', ' +', ' 15', ' =']     0.308711       0.348351
35  ['<|endoftext|>', '18', ' +', ' 14', ' =']     0.290656       0.304272
36  ['<|endoftext|>', '19', ' +', ' 13', ' =']     0.197693       0.308785
37  ['<|endoftext|>', '17', ' +', ' 17', ' =']    -0.060437      -0.049438
38  ['<|endoftext|>', '15', ' +', ' 19', ' =']    -0.097310      -0.115515
39  ['<|endoftext|>', '16', ' +', ' 18', ' =']     0.067560      -0.077886
40  ['<|endoftext|>', '18', ' +', ' 16', ' =']     0.147935      -0.005853
41  ['<|endoftext|>', '19', ' +', ' 15', ' =']    -0.095463      -0.136557
42  ['<|endoftext|>', '18', ' +', ' 18', ' =']     0.965268       0.673116
43  ['<|endoftext|>', '17', ' +', ' 19', ' =']     0.403783       0.415344
44  ['<|endoftext|>', '19', ' +', ' 17', ' =']     0.279374       0.312194

    score_change
0      -0.166331
1       0.017328
2       0.088131
3       0.095704
4      -0.039446
5       0.039462
6      -0.026111
7       0.005301
8      -0.071977
9      -0.072460
10     -0.021331
11     -0.123250
12     -0.207332
13     -0.075600
14     -0.008774
15     -0.053345
16      0.013718
17     -0.009263
18     -0.052807
19     -0.050185
20     -0.137670
21     -0.224573
22      0.107925
23      0.031774
24      0.091357
25      0.038687
26      0.041971
27      0.101705
28      0.046764
29      0.130720
30     -0.304549
31      0.045019
32      0.010204
33      0.076403
34      0.039639
35      0.013616
36      0.111093
37      0.010999
38     -0.018205
39     -0.145446
40     -0.153788
41     -0.041094
42     -0.292152
43      0.011561
44      0.032820
```

```text
   target  n_controls  clean_advantage  ablated_advantage  advantage_change  \
0      24           4         0.445692           0.238931         -0.206761
1      26           6        -0.117340          -0.026240          0.091100
2      28           8         0.267623           0.107032         -0.160591
3      30           8         0.770426           0.471990         -0.298436
4      32           6         0.767291           0.413413         -0.353878
5      34           4        -0.066118           0.034514          0.100632
6      36           2         0.623690           0.309347         -0.314343

   fraction_removed                                    evidence_status
0          0.463909  prospective_range_transfer_not_independent_con...
1          0.776379  prospective_range_transfer_not_independent_con...
2          0.600064  prospective_range_transfer_not_independent_con...
3          0.387365  prospective_range_transfer_not_independent_con...
4          0.461204  prospective_range_transfer_not_independent_con...
5          1.522007  prospective_range_transfer_not_independent_con...
6          0.504005  prospective_range_transfer_not_independent_con...
```

```text
Saved the prospective range-transfer prompt and summary tables.
```


### Cell 162 — code (execution count: 128)

```python
# ============================================================
# L13 — EFFECT-LEVEL CIRCUIT COMPLETENESS
#
# Compares:
#
#   FULL MODEL equal-operand effect
#       vs
#   EFFECT AFTER PROPOSED-CIRCUIT ABLATION
#
# The unit is target level.
# ============================================================

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. Safety checks
# ------------------------------------------------------------



if "df_circuit_discovery_results" not in globals():
    raise RuntimeError(
        "df_circuit_discovery_results does not exist. "
        "Run the corrected L12 cell first."
    )


required_full = {
    "target",
    "advantage",
}

required_circuit = {
    "target",
    "advantage",
}

missing_full = (
    required_full
    - set(df_full_discovery.columns)
)

missing_circuit = (
    required_circuit
    - set(df_circuit_discovery_results.columns)
)

if missing_full:
    raise ValueError(
        "df_full_discovery is missing columns: "
        f"{missing_full}"
    )

if missing_circuit:
    raise ValueError(
        "df_circuit_discovery_results is missing columns: "
        f"{missing_circuit}"
    )


# ------------------------------------------------------------
# 2. Align full-model and circuit-ablated effects
# ------------------------------------------------------------

full_effects = (
    df_full_discovery
    .set_index("target")["advantage"]
)

circuit_effects = (
    df_circuit_discovery_results
    .set_index("target")["advantage"]
)


common_targets = sorted(
    set(full_effects.index)
    &
    set(circuit_effects.index)
)

if not common_targets:
    raise ValueError(
        "No common target levels between full-model "
        "and circuit-ablated results."
    )


# ------------------------------------------------------------
# 3. Calculate completeness for each target
# ------------------------------------------------------------

completeness_rows = []

for T in common_targets:

    full_effect = float(
        full_effects.loc[T]
    )

    circuit_effect = float(
        circuit_effects.loc[T]
    )

    effect_removed = (
        full_effect
        - circuit_effect
    )

    if abs(full_effect) < 1e-8:
        completeness = np.nan
    else:
        completeness = (
            effect_removed
            / full_effect
        )

    completeness_rows.append({
        "target": T,
        "full_effect": full_effect,
        "circuit_ablated_effect": circuit_effect,
        "effect_removed": effect_removed,
        "effect_level_completeness": completeness,
    })


df_completeness = pd.DataFrame(
    completeness_rows
)


# ------------------------------------------------------------
# 4. Display
# ------------------------------------------------------------

print("=" * 78)
print("L13 — EFFECT-LEVEL CIRCUIT COMPLETENESS")
print("=" * 78)

print(
    df_completeness.to_string(
        index=False,
        float_format=lambda x:
            f"{x:+.6f}"
    )
)


mean_completeness = (
    df_completeness[
        "effect_level_completeness"
    ]
    .mean()
)

print("\n" + "-" * 78)

print("The mean ratio is unstable because the full effects for some targets are small; target-level values are shown descriptively."

)


# ------------------------------------------------------------
# 5. Save
# ------------------------------------------------------------

df_completeness.to_csv(
    "equal_operand_circuit_completeness.csv",
    index=False
)

print(
    "\nSaved: "
    "equal_operand_circuit_completeness.csv"
)
```

#### Recorded output

```text
==============================================================================
L13 — EFFECT-LEVEL CIRCUIT COMPLETENESS
==============================================================================
 target  full_effect  circuit_ablated_effect  effect_removed  effect_level_completeness
      4    +0.048633               -0.026336       +0.074969                  +1.541528
      6    +0.016807               +0.002678       +0.014129                  +0.840681
     10    +0.534628               +0.174941       +0.359687                  +0.672779
     12    +0.164717               +0.020338       +0.144380                  +0.876530
     16    +0.682146               +0.315137       +0.367009                  +0.538021

------------------------------------------------------------------------------
The mean ratio is unstable because the full effects for some targets are small; target-level values are shown descriptively.

Saved: equal_operand_circuit_completeness.csv
```


### Cell 163 — code (execution count: 123)

```python
# ============================================================
# L14 — CIRCUIT MINIMALITY / LEAVE-ONE-OUT
#
# Uses the RAW discovery dataframe.
#
# Full proposed circuit:
#   all proposed components ablated
#
# Leave-one-out:
#   all proposed components except one are ablated
#
# The difference tells us how much retaining one component
# restores the equal-operand effect.
# ============================================================

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Make sure we are using the RAW trial-level dataframe.
# ------------------------------------------------------------

df_circuit_discovery_raw = (
    df_circuit[
        df_circuit["split"] == "discovery"
    ]
    .copy()
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------

required_columns = {
    "target",
    "condition",
    "prompt",
}

missing_columns = (
    required_columns
    - set(df_circuit_discovery_raw.columns)
)

if missing_columns:
    raise ValueError(
        "Raw discovery dataframe is missing: "
        f"{missing_columns}"
    )


# ------------------------------------------------------------
# Evaluate each leave-one-out circuit
# ------------------------------------------------------------

minimality_rows = []


for omitted_component in PROPOSED_CIRCUIT:

    # Keep every proposed component except the omitted one.
    remaining_components = [
        component
        for component in PROPOSED_CIRCUIT
        if component != omitted_component
    ]

    # Run the actual target-level evaluation on the
    # RAW trial-level data.
    df_loo = (
        evaluate_target_level_advantage(
            df_circuit_discovery_raw,
            remaining_components
        )
    )

    loo_mean = (
        df_loo["advantage"]
        .mean()
    )

    minimality_rows.append({
        "omitted_component":
            omitted_component,

        "remaining_ablation_set":
            repr(remaining_components),

        "mean_advantage":
            loo_mean,

        "restoration_vs_full_circuit":
            loo_mean - circuit_mean,
    })


# ------------------------------------------------------------
# Assemble results
# ------------------------------------------------------------

df_minimality = (
    pd.DataFrame(
        minimality_rows
    )
    .sort_values(
        "restoration_vs_full_circuit",
        ascending=False
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

print("=" * 78)
print("L14 — CIRCUIT MINIMALITY")
print("=" * 78)

print(
    df_minimality.to_string(
        index=False,
        float_format=lambda x:
            f"{x:+.6f}"
    )
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df_minimality.to_csv(
    "equal_operand_circuit_minimality.csv",
    index=False
)

print(
    "\nSaved: "
    "equal_operand_circuit_minimality.csv"
)
```

#### Recorded output

```text
==============================================================================
L14 — CIRCUIT MINIMALITY
==============================================================================
omitted_component  remaining_ablation_set  mean_advantage  restoration_vs_full_circuit
             L9H9 ['10_mlp_out', 'L10H2']       +0.160792                    +0.063440
       10_mlp_out       ['L9H9', 'L10H2']       +0.159495                    +0.062144
            L10H2  ['10_mlp_out', 'L9H9']       +0.147634                    +0.050283

Saved: equal_operand_circuit_minimality.csv
```


### Cell 164 — code (execution count: 124)

```python
# ============================================================
# L15 —EXPLORATORY TRANSFER CHECK
#
# Circuit was selected only using discovery targets:
#   4, 6, 10, 12, 16
#
# It is now tested on:
#   8, 14
#
# No circuit selection occurs here.
# ============================================================

import pandas as pd
import numpy as np

transfer_df = df_circuit[df_circuit["split"] == "exploratory_transfer"].copy()

# Full-model target-level advantages
holdout_full_rows = []

for T in TRANSFER_TARGETS:

    data = transfer_df[
        transfer_df["target"] == T
    ]

    double_row = data[
        data["condition"] == "double"
    ].iloc[0]

    controls = data[
        data["condition"] == "control"
    ]

    double_score = primary_score(
        double_row["prompt"],
        T
    )

    control_scores = [
        primary_score(
            r["prompt"],
            T
        )
        for _, r in controls.iterrows()
    ]

    holdout_full_rows.append({
        "target": T,
        "full_advantage": (
            double_score
            - np.mean(control_scores)
        )
    })


df_holdout_full = pd.DataFrame(
    holdout_full_rows
)


# Circuit-ablated target-level advantages
df_holdout_circuit = (
    evaluate_target_level_advantage(
        transfer_df,
        PROPOSED_CIRCUIT
    )
    .rename(
        columns={
            "advantage":
                "circuit_ablated_advantage"
        }
    )
)


df_holdout = (
    df_holdout_full
    .merge(
        df_holdout_circuit[
            [
                "target",
                "circuit_ablated_advantage"
            ]
        ],
        on="target"
    )
)

df_holdout["advantage_change"] = (
    df_holdout[
        "circuit_ablated_advantage"
    ]
    - df_holdout[
        "full_advantage"
    ]
)

print("=" * 78)
print("L15 —EXPLORATORY TRANSFER CHECK")
print("=" * 78)

print(
    df_holdout.to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}"
    )
)

print(
    "\nImportant:"
)


df_holdout.to_csv(
    "equal_operand_circuit_holdout.csv",
    index=False
)

print(
    "\nSaved: "
    "equal_operand_circuit_transfer_check.csv"
)
```

#### Recorded output

```text
==============================================================================
L15 —EXPLORATORY TRANSFER CHECK
==============================================================================
 target  full_advantage  circuit_ablated_advantage  advantage_change
      8       +0.290962                  +0.102645         -0.188317
     14       -0.006201                  +0.029561         +0.035761

Important:

Saved: equal_operand_circuit_transfer_check.csv
```


### Cell 165 — code (execution count: 125)

```python
# ============================================================
# L16 — OPERATOR TRANSFER OF THE TARGET-TOKEN CONTRAST
#
# The circuit was selected using:
#   '+' digit+digit discovery prompts only.
#
# We now apply the exact same circuit to:
#   +
#   -
#   ×
#   and
#   then
#
# using exploratory transfer targets 8 and 14.
# ============================================================

OPERATOR_FORMS_L16 = {
    "plus":  lambda a, b: f"{a} + {b} =",
    "minus": lambda a, b: f"{a} − {b} =",
    "times": lambda a, b: f"{a} × {b} =",
    "and":   lambda a, b: f"{a} and {b} =",
    "then":  lambda a, b: f"{a} then {b} =",
}


def operator_target_level_advantage_with_circuit(
    T,
    formatter
):

    d = T // 2

    pairs = [
        (a, T-a)
        for a in range(1, 10)
        if 1 <= T-a <= 9
        and a != T-a
        and a < T-a
    ]

    double_prompt = formatter(
        d,
        d
    )

    double_score = primary_score(
        double_prompt,
        T
    )

    control_scores = []

    for a, b in pairs:

        p1 = formatter(a, b)
        p2 = formatter(b, a)

        control_scores.append(
            0.5 * (
                primary_score(p1, T)
                +
                primary_score(p2, T)
            )
        )

    # Circuit-ablated scores
    double_ablated = (
        evaluate_prompt_with_ablation(
            double_prompt,
            T,
            PROPOSED_CIRCUIT
        )
    )

    ablated_controls = []

    for a, b in pairs:

        p1 = formatter(a, b)
        p2 = formatter(b, a)

        ablated_controls.append(
            0.5 * (
                evaluate_prompt_with_ablation(
                    p1,
                    T,
                    PROPOSED_CIRCUIT
                )
                +
                evaluate_prompt_with_ablation(
                    p2,
                    T,
                    PROPOSED_CIRCUIT
                )
            )
        )

    return {
        "target": T,
        "double_clean": double_score,
        "control_clean": np.mean(
            control_scores
        ),
        "advantage_clean": (
            double_score
            - np.mean(control_scores)
        ),
        "double_ablated": double_ablated,
        "control_ablated": np.mean(
            ablated_controls
        ),
        "advantage_ablated": (
            double_ablated
            - np.mean(ablated_controls)
        ),
    }


operator_transfer_rows = []

for operator_name, formatter in (
    OPERATOR_FORMS_L16.items()
):

    for T in TRANSFER_TARGETS:

        result = (
            operator_target_level_advantage_with_circuit(
                T,
                formatter
            )
        )

        result["operator"] = operator_name

        result["advantage_change"] = (
            result["advantage_ablated"]
            - result["advantage_clean"]
        )

        operator_transfer_rows.append(
            result
        )


df_operator_transfer = pd.DataFrame(
    operator_transfer_rows
)

print("=" * 78)
print("L16 — OPERATOR TRANSFER OF THE TARGET-TOKEN CONTRAST")
print("=" * 78)

print(
    df_operator_transfer[
        [
            "operator",
            "target",
            "advantage_clean",
            "advantage_ablated",
            "advantage_change",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}"
    )
)
df_operator_transfer["metric_scope"] = (
    "Target-versus-neighbor numeric logit contrast; not arithmetic correctness"
)
df_operator_transfer.to_csv(
    "equal_operand_circuit_operator_transfer.csv",
    index=False
)

print(
    "\nSaved: "
    "equal_operand_circuit_operator_transfer.csv"
)
```

#### Recorded output

```text
==============================================================================
L16 — OPERATOR TRANSFER OF THE TARGET-TOKEN CONTRAST
==============================================================================
operator  target  advantage_clean  advantage_ablated  advantage_change
    plus       8        +0.290962          +0.102645         -0.188317
    plus      14        -0.006201          +0.029561         +0.035761
   minus       8        +0.179569          +0.007978         -0.171592
   minus      14        +0.009374          +0.034811         +0.025436
   times       8        +0.090690          -0.056223         -0.146913
   times      14        +0.054056          +0.042768         -0.011288
     and       8        +0.321472          +0.061786         -0.259686
     and      14        +0.060165          +0.087242         +0.027077
    then       8        +0.343901          +0.046014         -0.297887
    then      14        +0.152642          +0.163703         +0.011062

Saved: equal_operand_circuit_operator_transfer.csv
```


### Cell 166 — code (execution count: 126)

```python
# ============================================================
# L17 — NON-ARITHMETIC FILLER CONTROLS
#
# The same proposed circuit is applied to unrelated prompts.
#
# If the circuit causes equally large changes on fillers,
# it is probably not an arithmetic/equal-operand-specific
# mechanism.
# ============================================================

FILLER_PROMPTS_L17 = [
    "cat dog bird =",
    "The object in the box =",
    "The word on the page =",
    "Yesterday at the store =",
]

filler_rows = []

for T in TRANSFER_TARGETS:

    for filler in FILLER_PROMPTS_L17:

        clean_score = primary_score(
            filler,
            T
        )

        ablated_score = (
            evaluate_prompt_with_ablation(
                filler,
                T,
                PROPOSED_CIRCUIT
            )
        )

        filler_rows.append({
            "target": T,
            "filler_prompt": filler,
            "clean_score": clean_score,
            "ablated_score": ablated_score,
            "causal_shift": (
                ablated_score
                - clean_score
            ),
        })


df_filler_controls = pd.DataFrame(
    filler_rows
)

print("=" * 78)
print("L17 — FILLER SPECIFICITY")
print("=" * 78)

print(
    df_filler_controls.to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}"
    )
)

print("\nSome filler controls also show effects, so these results do not establish that the candidate components are specific to the equal-operand contrast.")

print("\nAggregate filler shift")
print("-" * 78)

print(
    df_filler_controls[
        "causal_shift"
    ].describe()
)

df_filler_controls.to_csv(
    "equal_operand_filler_controls.csv",
    index=False
)

print(
    "\nSaved: "
    "equal_operand_filler_controls.csv"
)
```

#### Recorded output

```text
==============================================================================
L17 — FILLER SPECIFICITY
==============================================================================
 target            filler_prompt  clean_score  ablated_score  causal_shift
      8           cat dog bird =    -0.154968      -0.221392     -0.066424
      8  The object in the box =    +0.165780      +0.094228     -0.071552
      8   The word on the page =    +0.252596      +0.236347     -0.016249
      8 Yesterday at the store =    +0.086393      +0.023756     -0.062637
     14           cat dog bird =    -0.104176      -0.016116     +0.088059
     14  The object in the box =    -0.452552      -0.271688     +0.180863
     14   The word on the page =    -0.296002      -0.176094     +0.119908
     14 Yesterday at the store =    +0.188878      +0.176530     -0.012348

Some filler controls also show effects, so these results do not establish that the candidate components are specific to the equal-operand contrast.

Aggregate filler shift
------------------------------------------------------------------------------
count    8.000000
mean     0.019953
std      0.096755
min     -0.071552
25%     -0.063584
50%     -0.014298
75%      0.096022
max      0.180863
Name: causal_shift, dtype: float64

Saved: equal_operand_filler_controls.csv
```


### Cell 167 — code (execution count: 127)

```python
# ============================================================
# L18 — FINAL MECHANISTIC AUDIT
#
# This cell does NOT automatically certify a circuit.
#
# It summarizes:
#   A. DLA localization
#   B. causal ablation
#   C. activation patching
#   D. position localization
#   E. circuit removal
#   F. completeness
#   G. minimality
#   H. exploratory transfer check
#   I. operator transfer
#   J. filler specificity
# ============================================================

import pandas as pd
import numpy as np

print("=" * 82)
print("FINAL MECHANISTIC AUDIT")
print("=" * 82)

print("\nCANDIDATE COMPONENTS FOR FOLLOW-UP")
print("-" * 82)
print("  Refer to 10_mlp_out, L9H9, and L10H2 as exploratory candidates, not a confirmed circuit.")

for component in PROPOSED_CIRCUIT:
    print(" ", component)


# ------------------------------------------------------------
# A. DLA
# ------------------------------------------------------------

print("\n[A] DLA LOCALIZATION")
print("-" * 82)

print(
    df_component_stats[
        df_component_stats["component"]
        .isin(PROPOSED_CIRCUIT)
    ][
        [
            "component",
            "mean_contrast",
            "positive_targets",
            "p_exact_signflip",
            "q_fdr",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}"
    )
)


# ------------------------------------------------------------
# B. Causal ablation
# ------------------------------------------------------------

print("\n[B] CAUSAL ABLATION")
print("-" * 82)

print(
    df_causal_summary[
        df_causal_summary["component"]
        .isin(PROPOSED_CIRCUIT)
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}"
    )
)


# ------------------------------------------------------------
# C. Patching
# ------------------------------------------------------------

print("\n[C] ACTIVATION PATCHING")
print("-" * 82)

print(
    patch_summary[
        patch_summary["component"]
        .isin(PROPOSED_CIRCUIT)
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}"
    )
)


# ------------------------------------------------------------
# D. Position localization
# ------------------------------------------------------------

print("\n[D] POSITION LOCALIZATION")
print("-" * 82)

print(
    position_summary[
        position_summary["component"]
        .isin(PROPOSED_CIRCUIT)
    ].head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}"
    )
)


# ------------------------------------------------------------
# E. Circuit removal
# ------------------------------------------------------------

print("\n[E] CIRCUIT-LEVEL REMOVAL")
print("-" * 82)

print(
    f"Full model mean advantage : "
    f"{full_mean:+.6f}"
)

print(
    f"Circuit-ablated mean      : "
    f"{circuit_mean:+.6f}"
)

print(
    f"Removal fraction           : "
    f"{removal_fraction:+.6f}"
)


# ------------------------------------------------------------
# F. Completeness
# ------------------------------------------------------------

print("\n[F] EFFECT-LEVEL COMPLETENESS")
print("-" * 82)

print( "The mean ratio is unstable because the full effects for some targets are small; target-level values are shown descriptively."
)

# ------------------------------------------------------------
# G. Minimality
# ------------------------------------------------------------

print("\n[G] MINIMALITY")
print("-" * 82)

print(
    df_minimality.to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}"
    )
)


# ------------------------------------------------------------
# H. EXPLORATORY TRANSFER CHECK
# ------------------------------------------------------------

print("\n[H] EXPLORATORY TRANSFER CHECK")
print("-" * 82)

print(
    df_holdout.to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}"
    )
)


# ------------------------------------------------------------
# I. Operator transfer
# ------------------------------------------------------------

print("\n[I] OPERATOR TRANSFER")
print("-" * 82)

print(
    df_operator_transfer.to_string(
        index=False,
        float_format=lambda x: f"{x:+.6f}"
    )
)


# ------------------------------------------------------------
# J. Filler specificity
# ------------------------------------------------------------

print("\n[J] FILLER SPECIFICITY")
print("-" * 82)

print(
    f"Mean absolute filler shift: "
    f"{df_filler_controls['causal_shift'].abs().mean():+.6f}"
)

print(
    f"Mean signed filler shift: "
    f"{df_filler_controls['causal_shift'].mean():+.6f}"
)


# ------------------------------------------------------------
# Manual claim checklist
# ------------------------------------------------------------

print("\n" + "=" * 82)
print("CLAIM CHECKLIST")
print("=" * 82)

print(
    """
1. LOCALIZATION
   Do candidate components show reproducible equal-operand DLA contrast?

2. CAUSALITY
   Does mean ablation reduce the equal-operand advantage?

3. SPECIFICITY
   Is the causal effect larger on equal-operand comparisons than
   on matched non-double controls?

4. PATCHING
   Does transferring the clean candidate activation into the corrupt
   prompt recover the behavior?

5. POSITION
   Does the patch effect localize to a meaningful token position?

6. CIRCUIT LEVEL
   Does joint circuit ablation remove a substantial portion of the
   full equal-operand effect?

7. COMPLETENESS
   Does the proposed circuit account for a substantial fraction of
   the measured effect?

8. MINIMALITY
   Does removing individual circuit components substantially alter
   the circuit-level effect?

9. GENERALIZATION
   Does the SAME circuit retain its predicted causal role on held-out
   examples?

10. SPECIFICITY TO TASK
    Is the effect smaller on matched non-arithmetic filler prompts?

Only if these pieces form a coherent chain should the paper make a
specific circuit-level claim. Otherwise report the result as a
component-level causal finding or distributed mechanism.
"""
)


# ------------------------------------------------------------
# Manual Summary export
# ------------------------------------------------------------

final_summary = pd.DataFrame([{
    "proposed_circuit": repr(PROPOSED_CIRCUIT),
    "full_discovery_advantage": full_mean,
    "circuit_ablated_advantage": circuit_mean,
    "removal_fraction": removal_fraction,
    "mean_completeness": (
        df_completeness[
            "effect_level_completeness"
        ].mean()
    ),
    "mean_patch_recovery": (
        patch_summary[
            patch_summary["component"]
            .isin(PROPOSED_CIRCUIT)
        ]["mean_recovery"].mean()
    ),
    "mean_holdout_advantage_change": (
        df_holdout[
            "advantage_change"
        ].mean()
    ),
    "mean_filler_absolute_shift": (
        df_filler_controls[
            "causal_shift"
        ].abs().mean()
    ),
}])


final_summary.to_csv(
    "equal_operand_final_mechanistic_summary.csv",
    index=False
)

print(
    "\nSaved: "
    "equal_operand_final_mechanistic_summary.csv"
)
```

#### Recorded output

```text
==================================================================================
FINAL MECHANISTIC AUDIT
==================================================================================

CANDIDATE COMPONENTS FOR FOLLOW-UP
----------------------------------------------------------------------------------
  Refer to 10_mlp_out, L9H9, and L10H2 as exploratory candidates, not a confirmed circuit.
  10_mlp_out
  L9H9
  L10H2

[A] DLA LOCALIZATION
----------------------------------------------------------------------------------
 component  mean_contrast  positive_targets  p_exact_signflip     q_fdr
      L9H9      +0.078350                 4         +0.125000 +0.513158
10_mlp_out      +0.062766                 5         +0.062500 +0.513158
     L10H2      +0.052577                 4         +0.125000 +0.513158

[B] CAUSAL ABLATION
----------------------------------------------------------------------------------
 component  mean_advantage_change  sd_advantage_change  targets_reduced  targets_total  p_two_sided  p_reduce_one_sided  bootstrap_ci_low  bootstrap_ci_high  mean_abs_relative_ln_scale_change  max_abs_relative_ln_scale_change  q_fdr_two_sided  q_fdr_reduce_one_sided
10_mlp_out              -0.078840            +0.094255                5              5    +0.062500           +0.031250         -0.159220          -0.011958                          +0.020246                         +0.066421        +0.150000               +0.093750
      L9H9              -0.070940            +0.062652                5              5    +0.062500           +0.031250         -0.124993          -0.030931                          +0.001496                         +0.003540        +0.150000               +0.093750
     L10H2              -0.058887            +0.042789                5              5    +0.062500           +0.031250         -0.091854          -0.026021                          +0.001966                         +0.005249        +0.150000               +0.093750

[C] ACTIVATION PATCHING
----------------------------------------------------------------------------------
 component  mean_recovery  median_recovery  positive_recoveries  n_valid
      L9H9      +0.156555        +0.247765                    4        5
10_mlp_out      +0.061110        +0.070912                    4        5
     L10H2      +0.039181        +0.121763                    4        5

[D] POSITION LOCALIZATION
----------------------------------------------------------------------------------
 component  position token  mean_recovery  median_recovery  sd_recovery  n_valid
      L9H9         4     =      +0.156555        +0.247765    +0.303437        5
10_mlp_out         1     2      +0.087999        +0.087999          NaN        1
10_mlp_out         4     =      +0.061110        +0.070912    +0.170177        5
     L10H2         4     =      +0.039181        +0.121763    +0.391910        5
10_mlp_out         3     2      +0.036951        +0.036951          NaN        1
10_mlp_out         3     3      +0.012736        +0.012736          NaN        1
     L10H2         3     6      +0.006775        +0.006775          NaN        1
10_mlp_out         3     6      +0.004524        +0.004524          NaN        1
10_mlp_out         2     +      +0.002974        +0.002382    +0.007547        5
      L9H9         2     +      +0.002687        +0.003027    +0.001831        5
10_mlp_out         1     3      +0.002074        +0.002074          NaN        1
     L10H2         3     8      +0.001014        +0.001014          NaN        1
     L10H2         1     6      +0.000987        +0.000987          NaN        1
10_mlp_out         1     5      +0.000534        +0.000534          NaN        1
     L10H2         3     5      +0.000423        +0.000423          NaN        1

[E] CIRCUIT-LEVEL REMOVAL
----------------------------------------------------------------------------------
Full model mean advantage : +0.289386
Circuit-ablated mean      : +0.097351
Removal fraction           : +0.893908

[F] EFFECT-LEVEL COMPLETENESS
----------------------------------------------------------------------------------
The mean ratio is unstable because the full effects for some targets are small; target-level values are shown descriptively.

[G] MINIMALITY
----------------------------------------------------------------------------------
omitted_component  remaining_ablation_set  mean_advantage  restoration_vs_full_circuit
             L9H9 ['10_mlp_out', 'L10H2']       +0.160792                    +0.063440
       10_mlp_out       ['L9H9', 'L10H2']       +0.159495                    +0.062144
            L10H2  ['10_mlp_out', 'L9H9']       +0.147634                    +0.050283

[H] EXPLORATORY TRANSFER CHECK
----------------------------------------------------------------------------------
 target  full_advantage  circuit_ablated_advantage  advantage_change
      8       +0.290962                  +0.102645         -0.188317
     14       -0.006201                  +0.029561         +0.035761

[I] OPERATOR TRANSFER
----------------------------------------------------------------------------------
 target  double_clean  control_clean  advantage_clean  double_ablated  control_ablated  advantage_ablated operator  advantage_change                                                              metric_scope
      8     +0.605705      +0.314744        +0.290962       +0.469328        +0.366683          +0.102645     plus         -0.188317 Target-versus-neighbor numeric logit contrast; not arithmetic correctness
     14     -0.238291      -0.232090        -0.006201       -0.223401        -0.252962          +0.029561     plus         +0.035761 Target-versus-neighbor numeric logit contrast; not arithmetic correctness
      8     +0.360562      +0.180993        +0.179569       +0.264278        +0.256301          +0.007978    minus         -0.171592 Target-versus-neighbor numeric logit contrast; not arithmetic correctness
     14     -0.241978      -0.251352        +0.009374       -0.241793        -0.276603          +0.034811    minus         +0.025436 Target-versus-neighbor numeric logit contrast; not arithmetic correctness
      8     +0.509995      +0.419304        +0.090690       +0.349352        +0.405575          -0.056223    times         -0.146913 Target-versus-neighbor numeric logit contrast; not arithmetic correctness
     14     -0.296125      -0.350181        +0.054056       -0.241048        -0.283816          +0.042768    times         -0.011288 Target-versus-neighbor numeric logit contrast; not arithmetic correctness
      8     +0.588014      +0.266542        +0.321472       +0.282081        +0.220295          +0.061786      and         -0.259686 Target-versus-neighbor numeric logit contrast; not arithmetic correctness
     14     -0.143995      -0.204160        +0.060165       +0.031454        -0.055788          +0.087242      and         +0.027077 Target-versus-neighbor numeric logit contrast; not arithmetic correctness
      8     +0.438091      +0.094190        +0.343901       +0.139807        +0.093793          +0.046014     then         -0.297887 Target-versus-neighbor numeric logit contrast; not arithmetic correctness
     14     -0.196564      -0.349205        +0.152642       -0.060388        -0.224091          +0.163703     then         +0.011062 Target-versus-neighbor numeric logit contrast; not arithmetic correctness

[J] FILLER SPECIFICITY
----------------------------------------------------------------------------------
Mean absolute filler shift: +0.077255
Mean signed filler shift: +0.019953

==================================================================================
CLAIM CHECKLIST
==================================================================================

1. LOCALIZATION
   Do candidate components show reproducible equal-operand DLA contrast?

2. CAUSALITY
   Does mean ablation reduce the equal-operand advantage?

3. SPECIFICITY
   Is the causal effect larger on equal-operand comparisons than
   on matched non-double controls?

4. PATCHING
   Does transferring the clean candidate activation into the corrupt
   prompt recover the behavior?

5. POSITION
   Does the patch effect localize to a meaningful token position?

6. CIRCUIT LEVEL
   Does joint circuit ablation remove a substantial portion of the
   full equal-operand effect?

7. COMPLETENESS
   Does the proposed circuit account for a substantial fraction of
   the measured effect?

8. MINIMALITY
   Does removing individual circuit components substantially alter
   the circuit-level effect?

9. GENERALIZATION
   Does the SAME circuit retain its predicted causal role on held-out
   examples?

10. SPECIFICITY TO TASK
    Is the effect smaller on matched non-arithmetic filler prompts?

Only if these pieces form a coherent chain should the paper make a
specific circuit-level claim. Otherwise report the result as a
component-level causal finding or distributed mechanism.


Saved: equal_operand_final_mechanistic_summary.csv
```
