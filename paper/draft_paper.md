# Arithmetic Prompt Processing in GPT-2 Small: Auditing Circuit Claims and an Equal-Operand Preference

**Anay Katiyar**  
Independent Researcher  

---

## Abstract

Mechanistic interpretability studies often search for localized circuits responsible for algorithmic behavior. This research draft documents an audit of arithmetic-shaped prompts in GPT-2 Small (124M parameters). The tested suite does not provide evidence for a general, reliably functioning addition mechanism. The M1 notebook distinguishes the all-pairs zero-shot result (2/36, 5.56%) from a held-out result (2/32, 6.25%) with a same-cohort constant-guess baseline of 8/32 (25%). The committed 79-cell grid has parity-sign agreement in 68/79 cells (86.1%); the pattern is concentrated at sums ≤13 (63/64) and is much weaker at sums ≥14 (5/15). These cells are not independent observations. Displayed raw/corrected DLA pairs imply a scale factor near 19.2; the previously stated 14×–40× range is not supported by those pairs. A source notebook and a row-level digit-addition doubles scan are now included. The corrected corpus query returns no significant correlation for either the joint-count measure (ρ=+0.46, p=.294) or conditional measure (ρ=−0.14, p=.760; n=7); the earlier ρ=+0.82 result remains retracted. These bounded results do not establish universal operator blindness or identify the mechanism behind the equal-operand preference.

---

## 1. Introduction

A central objective of mechanistic interpretability is reverse-engineering the neural circuits responsible for algorithmic reasoning in transformer language models (Elhage et al., 2021; Olsson et al., 2022; Wang et al., 2022). Arithmetic tasks—such as single-digit addition—have frequently served as model organisms for studying representation learning, numerical binding, and modular arithmetic circuits (Nanda et al., 2023; Zhong et al., 2023). In small, pretrained language models like GPT-2 Small (Radford et al., 2019), preliminary probing frequently suggests identifiable circuits: attention heads attending to operand tokens, late-layer multi-layer perceptrons (MLPs) promoting correct digits, and apparent head-level suppressors regulating logit outputs.

Interpreting model internals without statistical controls and numerical error accounting carries substantial risk. This draft records the available trajectory of an investigation into GPT-2 Small's responses to addition prompts of the form `a + b =`. The source notebook and selected supporting data exports are included; some run-level and API response records remain unavailable.

Our initial investigations appeared to support a structured heuristic addition pathway. Yet, systematic methodological auditing revealed that several headline observations were artifacts of:
1. **Unscaled Direct Logit Attribution (DLA)**: Neglecting final LayerNorm scaling inflated the displayed component attributions. The raw/corrected pairs in Table 2 imply a factor near $19.2$; the broader $14\times$–$40\times$ range in earlier wording is not supported by those pairs.
2. **Zero-Ablation Off-Distribution Shifts**: Forcing activation tensors to zero, which inflates residual stream standard deviation $\sigma$ from $19.20$ to $23.13$ and uniformly depresses all tracked logits by $\approx 1.0$, creating spurious "suppressor" heads.
3. **Token Geometry and Parity Confounds**: Conflating task-specific computation with intrinsic token-level biases, such as numerical parity preferences and static unembedding biases ($b_U$).

After these checks, a reported equal-operand preference remained: prompts of the form `a + a =` had higher target-token scores than reported matched split controls for selected sums. The archived aggregate results also report persistence across prompt formats and operator substitutions. Because several run-level outputs are absent and the committed operator script did not match the reported controls, those cross-condition summaries are provisional. They do not identify the mechanism or rule out broader operator-sensitive explanations.

---

## Research Question and Evidence Path

This study asks whether GPT-2 Small's apparent preference for correct answers on simple addition prompts reflects a general, addition-specific internal mechanism. The competing explanations include token-level output preferences, prompt structure, repeated operands, and other learned associations. The experiments were intended to distinguish these accounts rather than to treat every positive logit difference as evidence of a circuit.

The investigation began with an apparent addition-related signal. Correcting the attribution calculation changed its scale; separating the unembedding bias changed how some net outputs were interpreted; broader controls weakened several component-specific explanations. A target-dependent equal-operand advantage remained, but it also appeared under the tested non-addition operators and connectors. That result narrows the addition-specific interpretation without identifying the mechanism behind the remaining effect.

The experimental order was: behavioral baseline, matched controls, token and unembedding baselines, corrected Direct Logit Attribution (DLA), candidate interventions, doubles comparison, operator and connector substitutions, and a corpus-proxy audit. The final causal validation step remains proposed work. The question, prediction, measurement, result, interpretation, limitation, and next step for each stage are summarized in [docs/RESEARCH_FLOW.md](../docs/RESEARCH_FLOW.md). The paper-to-claim and data-provenance mapping is in [docs/LITERATURE_MAP.md](../docs/LITERATURE_MAP.md); the detailed correction record is in [docs/audit_trail.md](../docs/audit_trail.md), with a condensed chronology in [docs/RESEARCH_LOG.md](../docs/RESEARCH_LOG.md).

The closest published comparison is Hanna et al.'s analysis of a different mathematical behavior in GPT-2 Small: greater-than prediction in year-like contexts. Work on trained modular-addition transformers supplies useful methodological and conceptual comparisons, but those settings differ from this pretrained decimal-addition benchmark. These sources motivate careful task definition and validation; they do not supply evidence for the measurements reported here.

---

## 2. Baseline Competence: The M1 Benchmark

Before attributing internal representations to an "addition circuit", one must establish whether the model actually solves the task. We evaluated GPT-2 Small across a standardized cohort of single-digit addition prompts ($N=36$, $a, b \in [1, 9]$, $a + b < 10$).

```
```

The notebook includes the top-five token output for the clean `3 + 5 =` prompt. It is a single-prompt diagnostic, not a cohort-wide token-ranking result. The broader cohort-level M1 tables are reported separately below.

Although the logit difference between target (` 8`) and foil (` 9`) is positive ($+0.6443$), the model fails to output `' 8'` in its top predictions. Table 1 summarizes the model's accuracy across full vocabulary and subset selections.

**Table 1: Task performance across zero-shot and few-shot addition cohorts.**

| Metric | Zero-Shot, all pairs ($N=36$) | Zero-Shot, held-out ($N=32$) | Few-Shot, held-out ($N=32$) | Reference |
|---|---|---|---|---|
| **Top-1 Full Vocabulary** | 2/36 (5.56%) | 2/32 (6.25%) | 2/32 (6.25%) | — |
| **Top-3 Full Vocabulary** | 5/36 (13.89%) | — | — | — |
| **Top-1 Among Answer Digits** | — | 12.5% | 31.2% | 14.3% (uniform over 7 digits) |
| **Best Constant Guess** | — | 8/32 (25%; answer 9) | 8/32 (25%; answer 9) | 25% on held-out cohort |
| **Strict Correctness Filter ($\tau=1.0$)** | 1/36 (2.78%) | — | — | Earlier few-shot cohort: 5/33 |

The notebook separates the all-pairs and held-out zero-shot cohorts, resolving the earlier apparent mismatch: 2/36 is 5.56%, whereas the held-out result is 2/32 (6.25%, previously displayed as 6.2%). The 25% constant-guess rate is 8/32 on the held-out cohort. Few-shot 2/32 uses the held-out prompt definition; the earlier few-shot 5/33 result uses a different prefix and exclusions. The notebook contains the cohort construction and saved output. The strict-filtered subset is outcome-selected, so any attribution on that subset is descriptive. These task-specific results do not establish that GPT-2 Small contains no addition-related representations or computations.

---

## 3. Forensic Deconstruction of the "Addition Circuit"

### 3.1 Direct Logit Attribution and the LayerNorm Scaling Correction
Direct Logit Attribution (DLA) projects component activations onto the unembedding direction $W_U[:, \text{target}] - W_U[:, \text{foil}]$. In the reported evaluation, component projections summed to $\approx +8.95$ against a reported logit difference of $+0.6443$. The source notebook is included; the activation cache itself is not exported as a standalone artifact.

This discrepancy arose because component activations were extracted prior to final LayerNorm. Under `fold_ln=True`, final LayerNorm computes:
$$x_{\text{normalized}} = \frac{x - \mu}{\sigma_{\text{final}}}$$
where $\sigma_{\text{final}} = \text{ln\\_final.hook\\_scale}$. The displayed raw/corrected component pairs imply a scale factor near $19.2$ (e.g., MLP 11 raw attribution $+2.6263 \to$ corrected $+0.1368$). The earlier $14\times$–$40\times$ range is not supported by the pairs shown here.

Furthermore, the static unembedding bias term $b_U[\text{target}] - b_U[\text{foil}]$ sits outside residual decomposition. Incorporating both corrections satisfies the exact sum identity to within $< 10^{-3}$:

$$\sum_{i=1}^{159} \text{DLA}_i + \left( b_U[\text{target}] - b_U[\text{foil}] \right) = \text{logit\\_diff}$$

**Table 2: Corrected DLA decomposition on prompt `"3 + 5 ="` (Target `' 8'`, Foil `' 9'`).**

| Component | Unscaled DLA (Superseded) | Corrected DLA | Verification Identity |
|---|---|---|---|
| **MLP 11** | +2.6263 | +0.1368 | Sum Heads: +0.2172 |
| **MLP 7** | +2.2342 | +0.1164 | Sum MLPs: +0.2031 |
| **MLP 8** | +1.1474 | +0.0598 | Embed + Pos + Bias: +0.0455 |
| **MLP 10** | -2.2538 | -0.1174 | **Residual Sum: +0.4658** |
| **Head L11H1** | +0.9763 | +0.0509 | $b_U[\text{8}] - b_U[\text{9}]$: +0.1782 |
| **Head L11H0** | -0.8608 | -0.0448 | **Reconstructed: +0.6440** |
| **Head L9H1** | +0.5762 | +0.0300 | **Measured Logit Diff: +0.6443** |

### 3.2 The Static-Bias Override
For the few-shot Run 2 comparison on `158 + 274 =` (target `' 8'`, foil `' 1'`), the dynamic contribution was $+0.8382$ and the logit difference was $-0.1920$. The static bias term, $b_U[\text{8}] - b_U[\text{1}] = -1.0303$, accounts for the sign difference algebraically. Since the arithmetic answer is 432, this is a token-contrast identity, not evidence that an addition circuit computed 8 and was overridden. The notebook now reruns this prompt explicitly in its DLA cells to avoid depending on variables left by the earlier prediction loop.

### 3.3 Zero-Ablation Artifacts vs. Mean-Ablation Additivity
Zeroing the output of head L11H0 caused logit difference to rise by $+0.0658$, leading to its early characterization as an active "suppressor head". Tracking the residual stream standard deviation $\sigma$, however, revealed that zero-ablation forced $\sigma$ from $19.2002$ to $23.1286$, inducing an unphysiological shift that depressed all vocabulary logits uniformly by $\approx 1.0$.

When replaced with mean-ablation across seven reference prompts (four arithmetic and three generic text), the reported L11H0 shift was $+0.0013$ ($\sigma = 19.0148$). Joint ablation of L11H0 and MLP10 yielded a reported logit difference of $+0.8127$, compared with an additive prediction of $+0.8131$; their difference is $-0.0004$ on this prompt. This result does not support the earlier non-additive suppressor interpretation for this prompt. The source code and saved output are in the included notebook.

---

## 4. The Doubling Anomaly

Because the tested benchmark provides no positive evidence for a general, reliably functioning addition mechanism, we examined structured patterns in the addition grid. The committed grid and the 39-row digit+digit score file report a target-dependent equal-operand preference: it is positive for selected sums, while the measured advantage is near zero or negative at other sums. The prompt-level double/control scores are inspectable and reproduce the displayed dense-scan table. The three-format means are also committed; per-prompt format-specific controls remain unavailable.

![Figure 1: 79-Cell Addition Grid Symmetric Logit Difference Heatmap](figures/addition_grid_heatmap.svg)
*Figure 1: Heatmap of symmetric logit differences across the 79 committed grid rows. Parity sign agreement is 68/79 overall, 63/64 for sums ≤13, and 5/15 for sums ≥14 (0/9 for sums 14–15). Repeated target sums mean grid cells are not independent; the high-sum pattern differs from the lower-sum pattern.*

### 4.1 Dense Scan Evaluation
We conducted a dense scan across all even target sums $T \in [4, 16]$, evaluating double prompts ($d + d = T$) against all unique non-double single-digit splits ($a + b = T$, $a \ne b$) in both operand orders.

**Table 3: Dense scan metrics across even target sums.**

| Target ($T$) | Double Prompt | Ordered Control Prompts ($n$) | Double Score | Control Mean | Advantage | Normalized $adv/\text{SD}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **4** | `2 + 2 =` | 2 | +0.251 | +0.203 | +0.049 | +0.29 |
| **6** | `3 + 3 =` | 4 | +0.230 | +0.213 | +0.017 | +0.13 |
| **8** | `4 + 4 =` | 6 | +0.606 | +0.315 | **+0.291** | **+3.31** |
| **10** | `5 + 5 =` | 8 | +0.647 | +0.113 | **+0.535** | **+9.54** |
| **12** | `6 + 6 =` | 6 | +0.401 | +0.236 | **+0.165** | **+1.63** |
| **14** | `7 + 7 =` | 4 | -0.238 | -0.232 | -0.006 | -0.12 |
| **16** | `8 + 8 =` | 2 | +0.719 | +0.037 | **+0.682** | **+10.83** |

These digit+digit values can be recomputed from [`data/doubles_scan_scores.csv`](../data/doubles_scan_scores.csv), which contains all 39 double and ordered-control prompt scores. The format-level means are in [`data/variance_scaling.csv`](../data/variance_scaling.csv); its earlier values are preserved in [`data/variance_scaling_legacy.csv`](../data/variance_scaling_legacy.csv). In the digit+digit summary, advantages are positive at 8, 10, 12, and 16 and near zero at 4, 6, and 14; this does not establish null effects across all formats. Format-specific mean double/control scores changed for targets 4, 6, and 14 while their advantages remained effectively unchanged. The formatted prompt-level scores needed to independently recompute all three formats' pooled SDs are not exported.

### 4.2 Invariance to Lexical and Format Variations
To test whether the doubling effect is driven by token-level repetition of identical surface strings, we evaluated cross-format variants: digit+word (`4 + four =`) and word+word (`four + four =`).

**Table 4: Variance scaling across prompt presentation formats (Targets 8, 10, 12, 16).**

| Format | Mean Double Score | Mean Control Score | Raw Advantage | Pooled Control SD | Normalized $adv/\text{SD}$ |
|---|---|---|---|---|---|
| **digit + digit** (`4 + 4 =`) | +0.593 | +0.175 | +0.418 | 0.080 | **5.22** |
| **digit + word** (`4 + four =`) | +1.129 | +0.378 | +0.750 | 0.171 | **4.40** |
| **word + word** (`four + four =`) | +1.682 | +0.333 | +1.349 | 0.273 | **4.93** |

While the reported raw advantage grows with verbalization ($+0.418 \to +1.349$), the reported normalized values are similar ($5.22$, $4.40$, $4.93$). They are descriptive ratios, not conventional Student's t-statistics, p-values, or evidence of statistical significance. The reported `4 + four =` condition lacks a repeated surface token, so literal repetition of the same written digit is not required for the preference. This does not rule out lexical equivalence, latent repetition, or string-familiarity effects. The notebook's pooled word+word control SD is 0.273; the uploaded means file does not contain the prompt-level formatted controls needed to recompute it.

---

## 5. Operator and Connector Substitutions

Does the equal-operand preference appear only with addition? An addition-specific account predicts that it should weaken when the operator changes. The comparison is limited to the tested prompt strings and target tokens.

![Figure 2: Operator Swaps and Normalized adv/SD Ratio](figures/operator_swap_heatmap.svg)
*Figure 2: Reported normalized advantage ($adv/\text{SD}$) on token $2a$ across operator strings. The largest reported raw advantage is under “then” (+0.549); the largest reported normalized ratio is under subtraction (6.15). The included notebook's executed code and saved output reproduce these aggregates for the Unicode `×` and `−` strings. Per-prompt operator scores are not exported, and these descriptive ratios do not establish universal operator blindness.*

We evaluated all double and control prompts across 5 operator configurations, tracking logits strictly on the sum token $2a$:
1. Addition (`+`): `d + d =`
2. Multiplication (reported as `×` in the research log): `d × d =`
3. Subtraction (reported as `−` in the research log): `d − d =`
4. Conjunction (`and`): `d and d =`
5. Temporal Sequence (`then`): `d then d =`

**Table 5: Operator swap results across positive targets (8, 10, 12, 16) and null targets (6, 14).**

| Operator / Connector | Positive Adv | Control SD | Positive $adv/\text{SD}$ | Null Adv | Control SD | Null $adv/\text{SD}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **plus (`+`)** | +0.418 | 0.080 | **+5.22** | +0.005 | 0.099 | +0.05 |
| **times (`×`)** | +0.224 | 0.092 | **+2.43** | +0.022 | 0.113 | +0.20 |
| **minus (`−`)** | +0.323 | 0.052 | **+6.15** | +0.003 | 0.056 | +0.05 |
| **and** | +0.469 | 0.095 | **+4.95** | +0.081 | 0.094 | +0.86 |
| **then** | +0.549 | 0.120 | **+4.56** | +0.113 | 0.082 | +1.38 |

### The Subtraction Result
The archived JSON labels are incomplete/ASCII, but the included notebook's source cell defines `×` and `−` explicitly, uses ordered unequal controls, and prints the same aggregate values to rounding. The notebook therefore resolves the glyph identity for the source run. The prompt-level operator scores are not exported, so this is a source-notebook check rather than an independent fresh rerun. The corrected repository script retains Unicode as the reported form and supports an ASCII sensitivity condition.

For the reported subtraction condition, a double prompt such as `"4 − 4 ="` is compared with split controls while scoring token `' 8'`; mathematically, $4 - 4 = 0$, not $8$. The reported normalized ratio is the largest among the five summaries (**$adv/\text{SD} = 6.15$**), but the largest reported raw advantage is under “then” (+0.549). The advantage appears under all five reported strings, which weakens a simple addition-only account within this limited design.

The supplied notebook code and saved output resolve the old control-generation and glyph mismatch at the aggregate level. The JSON still lacks operator prompt-level scores, and no separate fresh model rerun was performed for this repository update. Operator-position patching (A19) is a separate experiment and does not validate or invalidate these operator-swap aggregates.

---

## 6. Pretraining Corpus Frequency Audit

The external corpus-frequency analysis is inconclusive. Sparse joint-query counts and the mismatch between the available proxy corpus and GPT-2's exact training distribution prevent strong conclusions about memorization. The results should therefore not be used either to establish or to rule out a memorization-based explanation.

We investigated whether the doubling anomaly is driven by verbatim co-occurrence frequencies in pretraining text. Using the Infini-gram API, we audited the open 3-trillion-token Dolma v1.7 corpus (`v4_dolma-v1_7_llama`) across all 7 target sums.

**Methodological Retraction**: An early preliminary query indicated a correlation between joint prompt-answer counts and model advantage ($\rho = +0.82$, $p = 0.023$). Audit inspection revealed that query tokenization formatting had produced 0 joint matches for higher targets, collapsing the ratio into a prompt-frequency artifact.

**Repository status**: The included notebook reran the corrected Infini-gram request with `index`, an empty extra anchor, and a reset results list. All seven targets passed its stated count floor. The resulting exploratory correlations were $\rho=+0.46$, $p=0.294$ for joint counts and $\rho=-0.14$, $p=0.760$ for conditional counts ($n=7$); neither is significant. The earlier $\rho=+0.82$, $p=0.023$ remains retracted. [`data/corpus_frequency_requery_summary.csv`](../data/corpus_frequency_requery_summary.csv) records the rounded per-target summary from the notebook output; the individual API responses and each control query count are not archived. Dolma remains a proxy, not GPT-2's exact training corpus, so this audit neither establishes nor rules out memorization.

---

## 7. Discussion and Related Work

The cited literature is used as methodological context, not as evidence for the present measurements. In particular, previous circuit studies show examples of mechanistic explanations on other behaviors; they do not establish that the same mechanism is present in this addition benchmark. See [the literature-to-claim map](../docs/LITERATURE_MAP.md) for source-by-source scope.

### 7.1 Static Heuristics vs. Algorithmic Circuits
The committed addition grid shows parity-sign agreement in 68/79 cells (86.1%) overall, but the rate is 63/64 for sums ≤13 and 5/15 for sums ≥14. This pattern does not support a uniform parity account across the grid. The audit reports an elevated target-10 baseline on one filler prompt; the four-template mean is recomputed by the diagnostic script and is not available as a committed output.

### 7.2 Methodological Lessons for Mechanistic Interpretability
This investigation illustrates several safeguards for this task:
1. **Account for final LayerNorm scaling** when projecting residual components into unembedding space.
2. **Compare interventions with distribution-preserving baselines**; zero-ablation changed residual-stream scale in the reported prompt.
3. **Use operator and connector controls**, while treating the result as evidence about only the tested strings and controls.

---

## 8. Conclusion

The reported evaluation suite does not establish a general, reliably functioning addition mechanism in GPT-2 Small. Several early mechanistic interpretations were weakened after methodological corrections. A target-dependent equal-operand preference is present in the committed grid and prompt-level digit-addition scan; the operator and connector summaries persist across the tested strings but remain bounded to their design. The corrected corpus audit shows no significant frequency correlation in its seven-target proxy analysis. Missing formatted prompt-level scores, individual corpus API responses, and the single-operand control limit further conclusions. These experiments do not establish the absence of arithmetic-related representations or operator-sensitive mechanisms elsewhere in the model.

---

## AI Use Disclosure

Claude Sonnet 5.5, ChatGPT, and Gemini 3.1 Pro were used as research-assistance tools during the development of this project for literature exploration, code assistance, and drafting or revising text. Suggestions were treated as provisional. The author reviewed reported data, calculations, citations, and claims against available project records, code, audit trail, and cited sources; this review identified missing source runs and artifact mismatches that are documented in the repository. Not every reported result could be independently regenerated from the committed files. The author made the final research decisions and is responsible for the data, analyses, claims, citations, and manuscript.

---

## References

This bibliography provides methodological context, not the provenance for project measurements. See [the literature-to-claim map](../docs/LITERATURE_MAP.md) for each source's relevance and scope, [the provenance table](../docs/LITERATURE_MAP.md#data-provenance-and-project-evidence) for project evidence, and [the audit trail](../docs/audit_trail.md) for method corrections and result status.

- Bolukbasi, T., et al. (2021). An Interpretability Illusion for BERT. *research draft arXiv:2104.07143*.
- Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. *Transformer Circuits Thread*.
- Hase, P., Bansal, M., Kim, B., and Ghandeharioun, A. (2023). Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models. *NeurIPS 2023*.
- Hanna, M., Liu, O., and Variengien, A. (2023). How does GPT-2 compute greater-than? Interpreting mathematical abilities in a pre-trained language model. *NeurIPS 2023*.
- Conmy, A., et al. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. *NeurIPS 2023*.
- Nanda, N., Chan, L., Lieberum, T., Smith, J., and Steinhardt, J. (2023). Progress Measures for Grokking via Mechanistic Interpretability. *ICLR 2023*.
- Olsson, C., et al. (2022). In-context Learning and Induction Heads. *Transformer Circuits Thread*.
- Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Technical Report*.
- Wang, K., et al. (2023). Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small. *ICLR 2023*.
- Zhong, Z., et al. (2023). The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks. *NeurIPS*.
