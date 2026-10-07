# Research Audit Log

This document summarizes the research stages and methodological corrections that affect the validity or interpretation of the GPT-2 Small arithmetic addition and doubling analyses. Operational issues and third-party review history are not included.

Each entry details: **Initial State**, **Methodological Flaw**, **Applied Correction**, **Impact on Empirical Conclusion**, and **Exact Verified Numbers**.

---

## 1. Timeline of Interpretive Phases

| Phase | Working Hypothesis / Initial Interpretation | Empirical Discovery / What Changed It | Status |
|---|---|---|---|
| **Phase A** | Single prompt (`3 + 5 =`) suggested a dedicated addition circuit: late MLPs (L11, L7, L8), L11H1, L10H2, and L2H2 binding head (97% attention). | Direct logit attribution sums failed to match true logits. | Superseded |
| **Phase B** | Unscaled direct logit attribution (DLA) seemed to indicate enormous component contributions (early records used a ~14×–40× range). | Applying proper final LayerNorm scaling (`cache.apply_ln_to_stack`) and unembedding bias ($b_U$) closed the residual identity to $< 10^{-3}$. Displayed raw/corrected pairs imply a scale factor near 19.2; the wider range is not supported by those pairs. | Corrected; wider range historical and unsupported |
| **Phase C** | Head L11H0 was deemed an active "suppressor head" and MLP10 $\times$ L11H0 was claimed to have a non-additive interaction. | Zero-ablation was found to take the residual off-distribution ($\sigma$ jumped from 19.20 to 23.13); the reported mean-ablation value for L11H0 is near zero ($+0.0013$). The reported double-ablation value (+0.8127) is close to the additive prediction (+0.8131) on that prompt. | Early interpretation retired; single-prompt result only |
| **Phase D** | Early attention heads (L2H2, L4H11) were designated "arithmetic binding heads". | Non-arithmetic control prompts revealed L4H11 is a generic previous-token head (1.000 on control text) and L2H2 is a general syntactic head (70.8% on control). | Falsified |
| **Phase E** | Output logits were thought to exhibit "first-operand invariance". | Full digit-shift sweeps revealed all digits co-move ($+0.0975$ to $+0.4797$), invalidating invariance. | Retired |
| **Phase F** | Scaled population testing ($N=84$) was deployed across Easy, Medium, and Hard cohorts. | Population baseline collapsed to $-0.0058 \pm 0.2819$; top head L9H1 was consistent with null draws ($t \approx 2.1$). Parity confound identified and fixed. | Falsified |
| **Phase G** | Attention-pattern blocking at position 1 appeared null when pooled. | Tier stratification showed Easy tier (no-carry) has a statistically significant drop ($+0.0852$, $p=0.0116$), while Medium is null ($p=0.724$). | Nuanced |
| **Phase H** | Output preferences were attributed to arithmetic reasoning. | The committed 79-cell grid has parity-sign agreement in 68/79 cells (86.1%); the distribution is 63/64 for sums ≤13 and 5/15 for sums ≥14. Separate filler results are reported in the audit trail, but their run output is not committed. | Grid recomputed; causal explanation not established |
| **Phase I** | Prompts of the form `a + a =` showed higher logits for the sum token than matched non-doubles. | Legacy summaries report an equal-operand preference across formats and operator strings, with a reported subtraction $adv/\text{SD}=6.15$. Prompt-level scores are absent; the prior script omitted controls and used ASCII glyphs where the log records Unicode. | Reported legacy result; not independently reproduced |
| **Phase J** | M1 benchmark test: evaluated if GPT-2 Small answers addition correctly. | Zero-shot top-1 is reported as 6.2% (2/36; the count implies 5.6%); the 25% constant-guess baseline is reported for N=32. The originating run is absent, so the comparison is unresolved. | Reported limitation; cohort mismatch unresolved |
| **Phase K** | External corpus n-gram frequency audit (Infini-gram / Dolma v1.7). | The legacy corpus artifact has no per-control counts, and its floor rule and result cannot be recomputed from committed files. The corrected query script has not been run here. | Inconclusive; legacy counts unverified |

---

## 2. Selected Methodological Audit Entries

### A01. Direct Logit Attribution Without Final LayerNorm Scaling
- **Original**: Component activations from `get_full_resid_decomposition` projected directly onto $W_U[:, \text{target}] - W_U[:, \text{foil}]$.
- **Problem**: Activations are pre-LayerNorm; LN divides by standard deviation $\sigma$ of the full residual stream. Raw projections overshoot by $14\times$ to $40\times$.
- **Correction**: Divided by `ln_final.hook_scale` or applied `cache.apply_ln_to_stack(..., layer=-1, pos_slice=-1)`.
- **Effect**: Raw sum $\approx +8.95$ shrunk to corrected $+0.4658$. Component rankings remained stable, but all raw magnitudes are superseded.

### A02. Missing Unembedding Bias Term ($b_U$)
- **Original**: DLA component sum was compared directly to total logit difference.
- **Problem**: The term $b_U[\text{target}] - b_U[\text{foil}]$ sits outside residual stream decomposition.
- **Correction**: Explicitly added $b_U[\text{target}] - b_U[\text{foil}]$.
- **Effect**: Closed residual identity: Run 1 ($+0.4658 + 0.1782 = +0.6440$ vs measured $+0.6443$). In Run 2, dynamic circuit favoured target ($+0.8382$) but static bias ($-1.0303$) flipped the output to foil ($-0.1921$).

### A03. LayerNorm Correction Applied to Pre-Sliced Tensor
- **Original**: Sliced `accumulated_resid[:, 0, -1, :]` before calling `apply_ln_to_stack`.
- **Problem**: Mismatched tensor ranks caused buggy computation on Run 2 ($+0.6761$, creating a $-0.1642$ gap).
- **Correction**: Passed full 4D tensor stack and sliced position afterward.
- **Effect**: Fixed Run 2 DLA sum to $+0.8382$, closing residual gap to $0.0001$.

### A04. Stale Variables and Mislabeled Quantities
- **Original**: Typo `arget_id`; stale corrupt ID reused; ungrounded baselines ($-0.3539$ and $-2.7162$).
- **Problem**: Interactive notebook state pollution across sessions.
- **Correction**: Decode and reprint token IDs prior to forward pass; fresh restart of kernel.
- **Effect**: Erroneous baseline claims cleared from experimental record.

### A06. Zero-Ablation Off-Distribution Artifacts
- **Original**: Zero-ablation of L11H0 caused $\Delta +0.0658$ logit diff boost; L11H0 labeled a "circuit suppressor". Double ablation of L11H0 + MLP10 claimed non-additive.
- **Problem**: Setting activations to zero forces residual variance $\sigma$ off-distribution ($19.20 \to 23.13$), uniformly depressing all tracked logits by $\approx 1.0$.
- **Correction**: Replaced zero-ablation with mean-ablation using reference sentences.
- **Effect**: L11H0 mean-ablation $\Delta$ was reported as $+0.0013$ ($50\times$ smaller); double ablation measured $+0.8127$ vs predicted $+0.8131$ on that prompt. This is consistent with additivity in that measurement; it does not establish exact or general additivity. The suppressor interpretation was retired.

### A07. Wrong Ablation Object and Indexing
- **Original**: Ablated `hook_v` instead of `hook_z`; omitted head indexing on layer 11.
- **Correction**: Targeted `hook_z` with explicit head index `[batch, pos, head, :]`.
- **Effect**: Resolved inconsistent head-level intervention metrics.

### A08. Attention-Pattern Blocking Without Renormalization
- **Original**: Zeroed attention pattern entries without renormalizing remaining weights.
- **Correction**: Renormalized attention matrix rows to sum to 1.0.
- **Effect**: Pooled blocked mean shifted from $-0.0357$ to $-0.0457$.

### A09. Recovery Percentages With Near-Zero Denominator
- **Original**: Reported percentage recoveries on $3 + 5 =$ vs $1 + 5 =$ where clean $-$ corrupt was only $0.0152$.
- **Correction**: Retired percentage recovery on small denominators; relied exclusively on raw logit differentials.

### A10. Position and Indexing Confusion in Activation Patching
- **Original**: Confused token value with position index; patched position identical between prompts.
- **Correction**: Strictly aligned sequence positions: Position 0 (BOS), 1 (Op 1), 2 (`+`), 3 (Op 2), 4 (`=`).
- **Effect**: Clarified that only `=` position (L10 to L11) carries meaningful residual transfer.

### A11. Inconsistent Foil Token Selection
- **Original**: Shifted foil token between `6` and `9` across experimental phases.
- **Correction**: Standardized on symmetric target $\pm 1$ and parity-matched target $\pm 2$ foils.
- **Effect**: Harmonized evaluation suite; cross-phase comparisons properly controlled.

### A12. Invalid Path Patching (L9H9)
- **Original**: Injected L9H9 delta into downstream components without delta norm checks.
- **Correction / Status**: Marked invalid and unproven; indirect routing hypothesis retired as unresolved.

### A13. Cross-Pair and Cross-Position Inconsistencies
- **Original**: Compared DLA from one pair against ablation on a different pair.
- **Correction**: Recomputed metrics strictly within identical prompt-target pairs.

### A14. Conflating High Attention Weight With Task Specificity
- **Original**: Designated L2H2 (97.2%) and L4H11 (100.0%) as arithmetic binders based on attention weight.
- **Correction**: Ran non-arithmetic control prompts (`The red dog ran`, `cat dog bird =`).
- **Effect**: L4H11 attended 100% on controls (previous-token head); L2H2 attended 70.8% on controls. Specificity falsified.

### A15. Corrupt-Prompt Parity Flip Confound
- **Original**: Easy/Medium corruption rule $a \to (a \pmod 8) + 1$ flipped numerical parity, confounding patching with parity bias.
- **Correction**: Implemented parity-preserving rule $a \to a + 2$.
- **Effect**: Spurious positive patching recoveries on operand position collapsed ($+0.0197 \to -0.0041$).

### A16. Winner's Curse in Dataset-Wide Head DLA
- **Original**: Top head L9H1 reported with unscaled DLA $+0.1199$.
- **Correction**: Applied per-item LayerNorm scaling; corrected mean $+0.0064$ ($\text{SD}=0.0279$).
- **Effect**: Top head $t \approx 2.1$, entirely consistent with the maximum of 144 null normal draws ($\approx 2.6$). No single head drives the task.

### A18. Pooled Null Masking Tier-Stratified Effect
- **Original**: Pooled position-1 attention blocking test showed null ($p \approx 0.08$).
- **Correction**: Stratified analysis by arithmetic difficulty tiers.
- **Effect**: Easy tier (no carry) revealed significant effect ($+0.0852$, $p=0.0116$), whereas Medium (carry) was null ($p=0.724$).

### A19. Operator-Patching Metric Unanchored to Corrupt Baseline
- **Original**: Computed patch delta as $\text{patched} - \text{clean}$ without corrupt baseline reference.
- **Correction / Status**: Documented that operator-binding cannot be claimed as "falsified" nor "confirmed" by this metric alone; status classified as unproven limitation.

### A20. Parity Bias Across the 79-Cell Grid
- **Original**: Evaluated 79 single-digit cells without accounting for repeated targets and foil parity.
- **Correction**: Recomputed the committed CSV directly: parity-sign agreement is 68/79 overall, 63/64 for sums ≤13, and 5/15 for sums ≥14. The earlier ~87% summary and filler values (even net $+0.0965$, odd net $-0.0240$) remain historical audit-trail reports; the filler run output is not committed. No causal claim that parity alone drives the grid is made.

### A21. Round-Number and Low-Baseline Artifacts
- **Original**: Target 10 deficit ($-0.4206$) and $7 + 7 = 14$ failure interpreted as arithmetic computation errors.
- **Correction**: Measured baseline biases: `' 10'` has generic filler preference of $+1.0649$ ($b_U = +3.7600$); Target 14 has low baseline ($-0.2311$). Both are vocabulary artifacts.

### A22. First-Operand Invariance Rebuttal
- **Original**: Difference of $0.0152$ between $3 + 5 =$ and $1 + 5 =$ read as operand invariance.
- **Correction**: Digit-shift sweep demonstrated systemic upward shift across all digit logits (mean $+0.3121$).

### A23. Doubles Experimental Design Refinements
- **Original**: Single control prompt per target; noisy control comparisons.
- **Correction**: The legacy data file reports a dense scan over single-digit splits (both operand orders) and normalized advantage ($adv/\text{SD}$), but its prompt-level values are absent. The former committed operator script omitted valid splits for several targets; corrected generation logic is now in `src/config.py` and `src/operator_swap.py`.
- **Effect / Status**: Target-dependent advantages are reported by the legacy aggregate (positive at 8, 10, 12, 16; near-zero or negative at 4, 6, 14). These values are not independently reproduced by the replacement script in this checkout.

### A24. Tokenization and Multi-Token Pitfalls
- **Original**: Stripped leading whitespace in DiD v1 (`"8"` vs `" 8"`), producing format artifacts.
- **Correction**: Enforced single-token representation with explicit leading space across all targets and foils.

### A25. M1 Benchmark: Baseline Accuracy Assessment
- **Original**: Proposed subset filtering to correctly answered prompts.
- **Correction**: Evaluated model across full vocabulary and constant-guess baselines.
- **Effect / Status**: Zero-shot top-1 is reported as $\approx 6.2\%$ with 2/36, although 2/36 is about 5.6%; the constant-guess 25% baseline is reported on N=32. The originating M1 run is absent, so the percentage/count and cohort differences remain unresolved. The benchmark does not support a general, reliably functioning addition mechanism under the tested conditions.

### A26. External Frequency Audit Methodology Fixes
- **Original**: Used incorrect API payload (`"corpus"` instead of `"index"`), anchored queries with leading space producing spurious $\rho = +0.82$.
- **Correction**: The replacement API client uses the documented request shape, reports alternative historical floor bases, and preserves query failures instead of substituting zero. It has not been executed in this checkout; the legacy JSON lacks per-control counts and cannot verify the reported floor outcome.
- **Effect / Status**: The prior $\rho = +0.82, p=0.023$ remains retracted. The corpus analysis is inconclusive; higher-sum sparsity is a legacy report, not independently reproduced here.

### A28. Deconstruction of Premature Framing Overreach
- Retired early theoretical concepts ("Sign Inversion Paradox", "Attention Sinks as Information Movers") that lacked causal grounding.

### A29. Cross-file value and labeling discrepancies
- The reported `' 8'` rank differs between Run 2 / a table and the controlled matrix. The DLA target/foil label differs between the research log and session record. L11H0 values −0.0448 (Run 1) and −0.1188 (Run 2) refer to different runs.
- These source-label discrepancies remain unresolved until the corresponding run outputs are available; they are not silently reconciled.


## 30. Final Claim-Strength Harmonization

The final interpretation is intentionally narrower than several early formulations. The corrected experiments preserve the numerical results and audit history while distinguishing measured results from interpretations and hypotheses.

- **[REPORTED RESULT]** Legacy aggregates report a target-dependent equal-operand/doubles advantage across selected operators and connectors; the operator result has not been reproduced from prompt-level data in this checkout.
- **[INTERPRETATION]** The observed effect is not sufficient evidence for an addition-specific circuit under the tested conditions.
- **[LIMITATION]** The M1 benchmark is scoped to the evaluated task distribution, prompt formats, tokenization scheme, and target range; it does not establish the absence of every arithmetic-related representation or mechanism in GPT-2 Small.
- **[LIMITATION]** Adv/SD is a project-defined descriptive normalized effect-size measure, not a conventional Student's t-statistic, p-value, or formal significance test.
- **[LIMITATION]** The operator-swap experiment uses a limited set of target values and matched control structures, so cross-operator persistence is not a universal statement about all operators or contexts.
- **[LIMITATION]** The external corpus-frequency analysis is inconclusive and uses Dolma v1.7 only as a proxy rather than GPT-2's exact training corpus.
- **[LIMITATION]** The 79-cell grid contains repeated target sums and related prompt structures, so cells are not fully independent replications.

These edits do not change measurements in the CSV or legacy result artifacts. Where the repository contradicts a historical summary or lacks row-level records, the discrepancy is stated rather than reconciled.


---

## 31. Literature, Narrative, and Provenance Integration

The final research flow is recorded separately from the chronological correction history in [RESEARCH_FLOW.md](RESEARCH_FLOW.md). It links each stage's question and prediction to the measurement, reported result, interpretation, limitation, and next step. The final causal validation of the equal-operand effect is listed as proposed work, not as a completed experiment.

Source relevance and source-to-claim boundaries are mapped in [LITERATURE_MAP.md](LITERATURE_MAP.md). It distinguishes literature-derived methodological context from project data and identifies unverified run-level provenance. The detailed method and code history is recorded in [audit_trail.md](audit_trail.md).

No numerical measurements were changed in this integration. The numbered entries retained in [audit_trail.md](audit_trail.md) are the detailed audit record; this log summarizes the research-relevant chronology. The literature map points from headline value families to relevant artifacts and audit entries without replacing or reconciling their values.

## 32. Repository artifact and reproduction audit

- `src/operator_swap.py` had a syntax error and incomplete positive-target controls. The replacement compiles and uses all ordered unequal splits. The legacy JSON is preserved; no TransformerLens model run was possible in this environment, so no replacement scores are claimed.
- `src/frequency_audit.py` previously converted failed queries to zero counts. The replacement retains errors, saves per-query records, and reports alternative floor calculations. It was not run against the external API here; the legacy corpus counts remain unverified.
- `data/addition_grid_79cell.csv` has 79 data rows and includes the `(9, 5)` pair; no row was added. Its parity statistics were recomputed from the committed values.
- The operator figure uses legacy aggregates, not a fresh model run; no corpus-frequency figure is committed. The operator labels distinguish raw advantage from normalized advantage and avoid a global-maximum or falsification claim.
- M1, N=84, DLA, ablation, and token-diagnostic run outputs are not present. Their reported values remain source-reported; the M1 percentage/count and cohort mismatch is unresolved. The single-operand control has not been run.
- The committed grid was checked directly and includes `(9, 5)`; no row was added. The README status badge says “Research draft,” and no “Published” badge is present.
