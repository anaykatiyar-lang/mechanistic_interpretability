# Research Methodology and Experimental Design

This document details the rigorous experimental design, mathematical formalisms, control protocols, and metric specifications developed to investigate the GPT-2 Small arithmetic doubling anomaly.

---

## 1. Model and Architectural Environment

- **Target Architecture**: GPT-2 Small (124M parameters, 12 layers, 12 attention heads per layer, $d_{\text{model}} = 768$, $d_{\text{head}} = 64$, $d_{\text{vocab}} = 50,257$).
- **Tooling**: TransformerLens library (`HookedTransformer.from_pretrained("gpt2-small")`).
- **Weight Folding**: Executed under default `fold_ln=True`, whereby LayerNorm gain and bias are folded directly into adjacent projection weights. Consequently, `ln_final` has no learnable weight, but dynamic scaling factor `hook_scale` is actively extracted at run time.
- **Tokenization Conventions**: Target and foil strings include explicit leading spaces (for example, `' 8'`). Whitespace is part of the GPT-2 byte-pair tokenization and must be preserved. Exact token IDs for all targets/foils should be regenerated from the selected model; the originating token-diagnostic output is not committed.

---

## 2. Experimental Cohorts and Prompt Design

### 2.1 Single-Prompt Intensive Cohort
- **Prompt**: `"3 + 5 ="` (tokens: `[BOS, "3", " +", " 5", " ="]`, positions 0 to 4).
- **Evaluation Targets**: Target `' 8'`, foil `' 9'` (or `' 6'`).
- **Few-Shot Prompt**: 23 tokens with prefix `"1 + 1 = 2\n2 + 2 = 4\n5 + 3 = 8\n"`.

### 2.2 Population-Scale Cohort ($N=84$)
Designed to test whether single-prompt mechanistic observations generalize across the arithmetic space:
- **Easy Tier ($N=36$)**: Single-digit addends with sum $< 10$ ($a, b \in [1, 8]$, no carry).
- **Medium Tier ($N=43$)**: Single-digit addends with sum $\ge 10$ ($a, b \in [2, 9]$, single carry).
- **Hard Tier ($N=5$)**: Two-digit addends with multi-digit carries (`18+25`, `28+37`, `46+78`, `39+85`, `57+68`).
- **79-Cell Addition Grid**: Systematic evaluation across single-digit pairs $a \times b \in [1, 9]^2$ (excluding edge pairs `1+9` and `9+1`).

### 2.3 Non-Arithmetic Filler Controls
Used to decouple true mathematical computation from static surface heuristics:
1. `"The object in the box ="`
2. `"The word on the page ="`
3. `"Yesterday at the store ="`
4. `"The item sequence number ="`

### 2.4 Doubles Dense-Scan Cohort
Evaluates equal addends (`a + a =`) against all valid single-digit non-double split pairs ($a + b = T$, $a \ne b$) for even targets $T \in [4, 16]$:
- Both operand orders ($a+b$ and $b+a$) are evaluated to eliminate ordering biases.
- Target tokens: `' 4'`, `' 6'`, `' 8'`, `' 10'`, `' 12'`, `' 14'`, `' 16'`.
- Symmetric foils: $T - 1$ and $T + 1$.

---

## 3. Mathematical Definitions of Metrics

### 3.1 Logit Difference and Symmetric Logit Difference
For output logits $L \in \mathbb{R}^{d_{\text{vocab}}}$ at final position:
$$\text{logit\\_diff} = L[\text{target}] - L[\text{foil}]$$

For symmetric evaluations with target $\pm 1$ neighbors:
$$\text{sym\\_diff} = L[T] - \frac{L[T-1] + L[T+1]}{2}$$

### 3.2 Mean-of-Neighbors Doubles Score and Advantage
For any prompt evaluating target $T$:
$$\text{Score}(P, T) = L_P[T] - \frac{L_P[T-1] + L_P[T+1]}{2}$$

The raw doubles advantage is defined as:
$$\text{Advantage}(T) = \text{Score}(d+d, T) - \frac{1}{|C_T|} \sum_{c \in C_T} \text{Score}(c, T)$$
where $C_T$ is the complete set of matched split controls yielding sum $T$.

### 3.3 Pooled Control Variance and Normalized Advantage ($adv/\text{SD}$)

`Adv/SD` is a project-defined normalized effect-size statistic. It should not be interpreted as a conventional Student's t-statistic, a p-value, or formal evidence of statistical significance. Because observations are structured and partially repeated across targets, operands, and operators, formal inferential claims require an appropriate paired/permutation/bootstrap analysis rather than relying on `adv/SD` alone.
Because raw advantage scales with vocabulary variance across presentation formats (digits vs words), normalized advantage is computed relative to pooled control standard deviation:
$$s_{\text{pooled}} = \sqrt{\frac{\sum_{k} (n_k - 1) s_k^2}{\sum_{k} (n_k - 1)}}$$
where $k \in \{8, 10, 12, 16\}$ represents positive target sums ($df = 18$).
$$\text{adv}/\text{SD} = \frac{\overline{\text{Advantage}}}{s_{\text{pooled}}}$$

---

## 4. Mechanistic Attribution and Intervention Protocols

### 4.1 Mathematically Rigorous Direct Logit Attribution (DLA)
The raw/corrected component pairs shown in the manuscript imply a scale factor near $19.2$. The broader $14\times$–$40\times$ range in earlier wording is not supported by those displayed pairs. Corrected attribution enforces:
$$\text{DLA}_i = \left( \frac{x_i}{\sigma_{\text{final}}} \right) \cdot \left( W_U[:, \text{target}] - W_U[:, \text{foil}] \right)$$
where $\sigma_{\text{final}} = \text{ln\\_final.hook\\_scale}$ is the standard deviation across residual dimensions at the final position.

The exact residual identity must be satisfied to within numerical tolerance ($< 10^{-3}$):
$$\sum_{i=1}^{159} \text{DLA}_i + \left( b_U[\text{target}] - b_U[\text{foil}] \right) = \text{logit\\_diff}$$
(Decomposed over 144 attention heads, 12 MLP blocks, embedding, positional embedding, and folded LayerNorm bias).

### 4.2 Mean-Ablation vs Zero-Ablation Protocol
- **Zero-Ablation Invalidation**: Setting head output $z \leftarrow 0$ drives total residual norm $\sigma$ off-distribution ($19.20 \to 23.13$), uniformly depressing all tracked logits by $\approx 1.0$.
- **Mean-Ablation Standard**: Causal interventions replace activation $z$ with its mean activation $\mu_z$ computed across 5 diverse non-arithmetic reference sentences:
$$z_{\text{ablated}} = \frac{1}{M} \sum_{m=1}^M z(\text{ref}_m)$$
Under mean-ablation, residual variance remains on-distribution ($\sigma = 19.0148$).

### 4.3 Attention-Pattern Blocking With Renormalization
To test causal information routing from operand positions without zeroing self-attention mass:
1. Extract attention pattern $A_{l, h} \in \mathbb{R}^{S \times S}$.
2. Zero column corresponding to operand 1 (position 1): $A_{l, h}[:, 1] \leftarrow 0$.
3. Renormalize remaining attention rows:
$$A_{l, h}[i, j] \leftarrow \frac{A_{l, h}[i, j]}{\sum_{k \ne 1} A_{l, h}[i, k]}$$

---

## 5. Operator and Corpus Comparisons

### 5.1 Operator-Swap Battery
To test whether the reported equal-operand preference appears only with addition, addends are coupled via five strings: `+`, `−`, `×`, `and`, and `then`. The notebook defines Unicode multiplication and subtraction explicitly and uses ordered unequal splits; its saved summaries match the operator aggregate to rounding. Evaluation tracks logits on token $2d$. Prompt-level operator scores are unavailable, and the corrected analysis has not been rerun as a model evaluation. The script uses Unicode forms by default and can run ASCII forms as a separate sensitivity condition. These summaries do not establish universal operator blindness.

### 5.2 Corpus N-Gram Frequency Audit Protocol

The corpus analysis is intended as contextual evidence rather than a direct measurement of GPT-2's training exposure. Dolma v1.7 is a proxy corpus, not the exact GPT-2 training corpus. Sparse exact-match counts therefore limit any conclusion about memorization.
- **API**: Infini-gram (`https://api.infini-gram.io/`).
- **Index**: `v4_dolma-v1_7_llama`. Corpus-size claims require a source citation and are not used here.
- **Queries**: Joint exact string (`"a + b = T"`) and prompt prefix (`"a + b ="`).
- **Corrected query**: The query uses the `index` request field, no extra leading-space anchor, retry and explicit failure handling, a fresh results list, and a tokenization check. Its selected query-total floor gives seven usable targets. The exploratory associations are joint $\rho=+0.46$, $p=.294$ and conditional $\rho=-0.14$, $p=.760$ ($n=7$); neither is significant. Rounded per-target summaries are in `data/corpus_frequency_requery_summary.csv`. Individual control-query counts and raw API responses are not committed.
- **Reliability and scope**: The historical files disagree on the count-floor rule. The audit script reports candidate floor bases separately and marks API failures explicitly. Dolma is a proxy rather than GPT-2's exact training data, so these counts cannot establish or exclude broader memorization or distributional effects.


---

## 6. Research Flow and Evidence Labels

The study's final reasoning path is behavioral baseline → matched controls → token and unembedding baselines → corrected DLA → candidate interventions → doubles comparison → operator/connector substitutions → corpus proxy audit. Causal validation of the surviving doubles effect remains proposed work, not a completed result. The full stage record uses the same fields at each step: question, prediction, measurement, result, interpretation, limitation, and next step. See [RESEARCH_FLOW.md](RESEARCH_FLOW.md).

Use the labels consistently:
- **[VERIFIED RESULT]** for a measurement tied to an available data artifact or explicitly reported audit result.
- **[INTERPRETATION]** for what that measurement supports within the tested design.
- **[HYPOTHESIS]** for an untested proposed mechanism.
- **[LIMITATION]** for scope, design, tokenization, corpus, or evidence boundaries.

Literature is methodological context; it is not a source of this project's experimental values. For source-to-claim boundaries and the records for headline numbers, see [LITERATURE_MAP.md](LITERATURE_MAP.md). Cite [audit_trail.md](audit_trail.md) for the detailed correction record and verification status. The notebook is an executed source artifact; its saved output is not a substitute for independent reruns, and the individual corpus responses and formatted prompt-level scores remain unavailable.
