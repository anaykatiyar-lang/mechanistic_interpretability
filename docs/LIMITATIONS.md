# Methodological Limitations and Boundary Conditions

This document catalogs the exact scope limitations, tokenization artifacts, statistical noise bounds, and unresolved questions that constrain the conclusions of this project.

---

## 1. Scope and Architectural Constraints

1. **Single Model Architecture**:
   All empirical findings are derived from GPT-2 Small (124M parameters, 12 layers). No inferences are made regarding larger model scales (GPT-2 Medium/XL, Llama, Gemma) or models trained with modern code/math synthesis data.
2. **Restricted Arithmetic Domain**:
   Most experiments focus on single-digit operands and relatively small sums. This domain is substantially narrower than general arithmetic. The results therefore characterize GPT-2 Small's behavior on the tested arithmetic distribution rather than establishing a general theory of arithmetic computation in the model.
3. **Single-Prompt Attribution Caveats**:
   Several initial mechanistic insights (e.g., L11H0/MLP10 additivity, static-bias override) were derived on isolated prompts (`"3 + 5 ="`). While algebraically exact on those prompts, they do not constitute universal circuit motifs across the model.

---

## 2. Token Geometry and Representation Quirks

1. **Leading Whitespace Sensitivity**:
   In GPT-2's byte-pair encoding (BPE), leading whitespace is part of the tokenized string. The DiD v1 analysis used unspaced targets in at least one comparison, but the exact per-token IDs and originating run outputs are not committed. The reported format swing therefore cannot be attributed solely to tokenization without separating the other simultaneous design changes recorded in the audit trail.
2. **Single vs Multi-Token Representations**:
   While single-digit and early two-digit answers (`' 10'`, `' 12'`, `' 14'`, `' 16'`) are single BPE tokens in GPT-2, external tokenizers (such as Llama used in Dolma indexing) segment multi-digit numbers into individual digits. This creates token geometry mismatches during external corpus frequency matching.
3. **Round-Number and Static Bias Skew ($b_U$)**:
   - `' 10'` exhibits an abnormally high static unembedding bias ($b_U = +3.7600$) and is generically favored by $+1.0649$ logits on non-arithmetic filler prompts.
   - Target `' 14'` exhibits an abnormally low baseline ($-0.2311$), explaining why $7 + 7 = 14$ appears as an anomaly in raw metrics. These are properties of vocabulary embedding geometry, not arithmetic computation failures.

---

## 3. Statistical Noise-Floor Bounds and the Winner's Curse

1. **Dataset-Wide DLA Winner's Curse**:
   In uncorrected dataset-wide DLA sweeps, head L9H1 appeared prominent ($+0.1199$). After proper LayerNorm scaling, its mean contribution shrank to $+0.0064$ ($\text{SD} = 0.0279$), yielding a descriptive test statistic of $t \approx 2.1$. For the maximum of 144 independent standard normal draws, an extreme value of $t \approx 2.6$ is expected purely by chance. Thus, L9H1 cannot be designated a causal arithmetic driver.
2. **79-Cell Grid Non-Independence**:
   The committed grid has 79 cells and repeated target sums (for example, target 10 has five operand pairs). Parity predicts sign in 68/79 cells (86.1%) overall, 63/64 for sums ≤13, and 5/15 for sums ≥14. The cells are not independent; the lower-sum agreement should not be generalized across all sums. The earlier ~87% summary used a different row count and is retained only as history.
3. **Dense Scan Sample Limitations**:
   Because each target sum has only one true double ($d+d$), target 4 ($2+2$) and target 16 ($8+8$) have only one unique control pair (evaluated in two operand orders). These estimates therefore have greater sampling uncertainty and should not be interpreted as equally precise across all target values.

---

## 4. Unanchored Metrics and Experimental Blindspots

1. **Unanchored Operator-Position Patching (A19)**:
   In the separate operator-position patching experiment, the metric evaluated was:
   $$\Delta_{\text{patch}} = \text{diff}_{\text{patched}} - \text{diff}_{\text{clean}}$$
   Because the corrupted prompt's baseline logit difference was not included as an anchor, a delta near zero cannot distinguish between "full recovery of corrupt state" and "no corruption effect". Operator binding therefore remains untested by this metric. This limitation is separate from the operator-swap comparison.
2. **Invalid L9H9 Path Patching (A12)**:
   Early attempts to demonstrate indirect routing from L9H9 into MLP10 and L10H2 failed to report delta norms or confirm hook execution on CPU runtimes. The hypothesis that L9H9 acts as an information router remains unverified.
3. **Easy-Tier Attention Blocking Mechanism**:
   Position-1 attention blocking produced a statistically significant reduction in target logit diff on Easy-tier prompts ($p = 0.0116$), but Medium-tier prompts showed no effect ($p = 0.724$). The exact internal mechanism mediating this difference remains unidentified.

---

## 5. Corpus Frequency Audit Boundaries

The external corpus-frequency analysis was inconclusive. Sparse joint-query counts and the mismatch between the available corpus and GPT-2's exact training distribution prevent strong conclusions about memorization. The results therefore should not be used either to establish or to rule out a memorization-based explanation.

1. **Corpus Proxy Discrepancy**:
   Dolma v1.7 is used as a proxy for WebText. Because WebText is proprietary and unreleased, true pretraining co-occurrence statistics cannot be measured directly. The corpus-size figure is omitted until supported by a verified source.
2. **Unresolved Count Floor and Query Provenance**:
   The committed corpus JSON lacks per-control counts and does not match the committed query script's schema. Historical files also differ on the count-floor rule. The report is retained as inconclusive, but no specific usable-target list or exact count is independently verified from the repository.
3. **Run 1 Artifact Retraction**:
   An initial analysis reported a correlation ($\rho = +0.82$, $p = 0.023$) between corpus frequency and model advantage. The statistic was retracted after the audit identified a query-format artifact. It must not be used as a finding.

4. **Multi-Token Representation Limitation**: Some external arithmetic examples produce answers that do not correspond cleanly to a single GPT-2 token. Analyses that rely on `model.to_single_token` are restricted to examples whose target and foil representations satisfy the required single-token condition. Results from this restricted subset should not be generalized to arbitrary multi-token arithmetic answers.

5. **Parity Confound**: Digit-level output comparisons can inherit parity-related token biases. Because neighboring foil tokens can differ in parity from the target, part of the measured target-vs-neighbor preference can reflect static or contextual parity effects rather than arithmetic computation. Parity-preserving controls and filler prompts are therefore necessary when interpreting these measurements.

6. **Round-Number / Target-Token Bias**: Target-specific baseline effects are substantial for some numbers, particularly round-number tokens such as `10`. Target-specific baseline controls are therefore treated as necessary for interpreting mechanistic measurements.


---

## 6. Literature and Data-Provenance Boundaries

The cited mechanistic studies use different tasks and, in several cases, different training regimes. They motivate questions about task definition, controls, and causal validation; they do not independently corroborate this project's results. In particular, trained modular-addition transformers and GPT-2 Small greater-than results should not be presented as direct evidence for pretrained GPT-2 Small decimal addition. See [LITERATURE_MAP.md](LITERATURE_MAP.md) for source-by-source boundaries.

The repository's [audit_trail.md](audit_trail.md) and [RESEARCH_LOG.md](RESEARCH_LOG.md) preserve headline measurements and corrections. Some originating notebook cells and run-level outputs are not included in the repository, so the exact row-level link from every headline number to its original execution is not fully verified. A committed data artifact is the primary measurement record; where no run-level artifact exists, the reported value remains documented in the manuscript and audit trail pending a traceable source.


The manuscript's M1 table pairs a zero-shot top-1 value of 6.2% with a count of 2/36. That count is about 5.6%, so the percentage/count pair is internally inconsistent. Both reported entries are preserved pending the originating run or a verified correction; this integration does not infer which entry should change.

The reported 25% constant-guess baseline is 8/32, while the zero-shot header states N=36. The source M1 prompt/results file is absent, so same-cohort baseline comparisons and the reported top-five token/rank details are not independently verified. Few-shot top-1 2/32 and strict-filter 5/33 are reported from separate runs and remain unreconciled.

## 7. Reproduction Gaps Identified in the Committed Repository

- The operator-swap script previously omitted valid single-digit splits for positive targets (4/4/4/2 controls instead of 6/8/6/2), while the legacy JSON reports 18 positive-control degrees of freedom. The script now builds all ordered splits, but the new model run has not been executed here.
- The legacy operator JSON lacks per-prompt scores and uses ASCII operator labels; the research log describes Unicode multiplication/subtraction glyphs. The exact historical prompt strings are unresolved.
- The single-operand comparison (`4 + 3` and `4 + 5`, scoring token `' 8'`) has no committed run output and has not been run by the reproduction code.
- `data/variance_scaling.csv` contains aggregate summaries without raw control scores. The displayed word+word SDs do not exactly reconcile after rounding.
- The N=84, M1, DLA, and ablation tables have no originating notebook or table-generation code in this repository.
- The full multiplication control set includes `1 × 7 = 7` at target 8 and `1 × 9 = 9` at target 10. Those answers equal the lower neighbor foil (`T−1`) for their target, so the control construction may mechanically lower the control mean; this needs to be reported and assessed explicitly.
- Equality, lexical equivalence, and string familiarity are not separated by the current comparisons. There is one double prompt per target.
