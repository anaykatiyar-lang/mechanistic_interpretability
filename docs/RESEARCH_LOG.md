# Comprehensive Chronological Research Audit Log

This document records the complete chronological audit trail of investigations, corrections, and falsification milestones for the GPT-2 Small arithmetic addition and doubling anomaly research.

Each entry details: **Initial State**, **Methodological Flaw**, **Applied Correction**, **Impact on Empirical Conclusion**, and **Exact Verified Numbers**.

---

## 1. Timeline of Interpretive Phases

| Phase | Working Hypothesis / Initial Interpretation | Empirical Discovery / What Changed It | Status |
|---|---|---|---|
| **Phase A** | Single prompt (`3 + 5 =`) suggested a dedicated addition circuit: late MLPs (L11, L7, L8), L11H1, L10H2, and L2H2 binding head (97% attention). | Direct logit attribution sums failed to match true logits. | Superseded |
| **Phase B** | Unscaled direct logit attribution (DLA) seemed to indicate enormous component contributions (~14×–40× too large). | Applying proper final LayerNorm scaling (`cache.apply_ln_to_stack`) and unembedding bias ($b_U$) closed the residual identity to $< 10^{-3}$. | Corrected |
| **Phase C** | Head L11H0 was deemed an active "suppressor head" and MLP10 $\times$ L11H0 was claimed to have a non-additive interaction. | Zero-ablation was found to take the residual off-distribution ($\sigma$ jumped from 19.20 to 23.13); mean-ablation revealed L11H0 has near-zero causal effect ($+0.0013$) and the interaction is strictly additive. | Falsified |
| **Phase D** | Early attention heads (L2H2, L4H11) were designated "arithmetic binding heads". | Non-arithmetic control prompts revealed L4H11 is a generic previous-token head (1.000 on control text) and L2H2 is a general syntactic head (70.8% on control). | Falsified |
| **Phase E** | Output logits were thought to exhibit "first-operand invariance". | Full digit-shift sweeps revealed all digits co-move ($+0.0975$ to $+0.4797$), invalidating invariance. | Retired |
| **Phase F** | Scaled population testing ($N=84$) was deployed across Easy, Medium, and Hard cohorts. | Population baseline collapsed to $-0.0058 \pm 0.2819$; top head L9H1 was consistent with null draws ($t \approx 2.1$). Parity confound identified and fixed. | Falsified |
| **Phase G** | Attention-pattern blocking at position 1 appeared null when pooled. | Tier stratification showed Easy tier (no-carry) has a statistically significant drop ($+0.0852$, $p=0.0116$), while Medium is null ($p=0.724$). | Nuanced |
| **Phase H** | Output preferences were attributed to arithmetic reasoning. | Parity grid (87% predictable by parity) and filler prompts proved baseline preferences are driven by static token bias ($b_U$, parity, round numbers). | Verified |
| **Phase I** | Prompts of the form `a + a =` showed higher logits for the sum token than matched non-doubles. | Effect persisted across formats (`4 + four =`) and operator swaps ($\times$, $-$, 'and', 'then'), peaking under subtraction ($adv/\text{SD} = 6.15$). This supports a cross-operator persistence interpretation; it does not establish universal operator blindness. | Corroborated |
| **Phase J** | M1 benchmark test: evaluated if GPT-2 Small answers addition correctly. | Zero-shot top-1 accuracy is $\approx 6\%$, below the constant-guess baseline (25%). The tested benchmark provides no positive evidence for a general, reliably functioning addition mechanism. | Limitation Closed |
| **Phase K** | External corpus n-gram frequency audit (Infini-gram / Dolma v1.7). | Higher target sums lacked sufficient occurrences ($<20$ floor); the test concluded inconclusive. The operator-swap results argue against a simple addition-specific surface-form explanation, but do not rule out broader memorization or distributional explanations. | Inconclusive |

---

## 2. Chronological Ledger of Methodological Audits (A01–A29)

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

### A05. Hook and TransformerLens API Issues
- **Original**: `hook_result` missing unless `set_use_attn_result(True)` enabled; `ln_final.weight` returning `None` under `fold_ln=True`.
- **Problem**: API errors under default HookedTransformer configurations.
- **Correction**: Explicit flags set; utilized `hook_scale` directly.
- **Effect**: Numerical stability guaranteed without crashes.

### A06. Zero-Ablation Off-Distribution Artifacts
- **Original**: Zero-ablation of L11H0 caused $\Delta +0.0658$ logit diff boost; L11H0 labeled a "circuit suppressor". Double ablation of L11H0 + MLP10 claimed non-additive.
- **Problem**: Setting activations to zero forces residual variance $\sigma$ off-distribution ($19.20 \to 23.13$), uniformly depressing all tracked logits by $\approx 1.0$.
- **Correction**: Replaced zero-ablation with mean-ablation using reference sentences.
- **Effect**: L11H0 mean-ablation $\Delta$ dropped to $+0.0013$ ($50\times$ smaller); double ablation measured $+0.8127$ vs predicted $+0.8131$ (strictly additive). Suppressor hypothesis falsified.

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

### A17. Colab CPU RAM Crashes
- **Original**: Full cache collection on every prompt without gradient disabling.
- **Correction**: Applied `torch.no_grad()`, `names_filter`, and explicit garbage collection.

### A18. Pooled Null Masking Tier-Stratified Effect
- **Original**: Pooled position-1 attention blocking test showed null ($p \approx 0.08$).
- **Correction**: Stratified analysis by arithmetic difficulty tiers.
- **Effect**: Easy tier (no carry) revealed significant effect ($+0.0852$, $p=0.0116$), whereas Medium (carry) was null ($p=0.724$).

### A19. Operator-Patching Metric Unanchored to Corrupt Baseline
- **Original**: Computed patch delta as $\text{patched} - \text{clean}$ without corrupt baseline reference.
- **Correction / Status**: Documented that operator-binding cannot be claimed as "falsified" nor "confirmed" by this metric alone; status classified as unproven limitation.

### A20. Parity Bias Across the 79-Cell Grid
- **Original**: Evaluated 79 single-digit cells without accounting for repeated targets and foil parity.
- **Correction**: Tested non-arithmetic filler controls; verified that 87% of signs match parity bias (even net $+0.0965$, odd net $-0.0240$).

### A21. Round-Number and Low-Baseline Artifacts
- **Original**: Target 10 deficit ($-0.4206$) and $7 + 7 = 14$ failure interpreted as arithmetic computation errors.
- **Correction**: Measured baseline biases: `' 10'` has generic filler preference of $+1.0649$ ($b_U = +3.7600$); Target 14 has low baseline ($-0.2311$). Both are vocabulary artifacts.

### A22. First-Operand Invariance Rebuttal
- **Original**: Difference of $0.0152$ between $3 + 5 =$ and $1 + 5 =$ read as operand invariance.
- **Correction**: Digit-shift sweep demonstrated systemic upward shift across all digit logits (mean $+0.3121$).

### A23. Doubles Experimental Design Refinements
- **Original**: Single control prompt per target; noisy control comparisons.
- **Correction**: Dense scan over all valid single-digit splits (both operand orders); normalized advantage ($adv/\text{SD}$).
- **Effect**: Confirmed advantage is target-dependent (positive at 8, 10, 12, 16; null at 4, 6, 14).

### A24. Tokenization and Multi-Token Pitfalls
- **Original**: Stripped leading whitespace in DiD v1 (`"8"` vs `" 8"`), producing format artifacts.
- **Correction**: Enforced single-token representation with explicit leading space across all targets and foils.

### A25. M1 Benchmark: Baseline Accuracy Assessment
- **Original**: Proposed subset filtering to correctly answered prompts.
- **Correction**: Evaluated model across full vocabulary and constant-guess baselines.
- **Effect**: Zero-shot top-1 accuracy is $\approx 6.2\%$, well below constant guess of $25\%$. General addition mechanism hypothesis not supported under the tested benchmark.

### A26. External Frequency Audit Methodology Fixes
- **Original**: Used incorrect API payload (`"corpus"` instead of `"index"`), anchored queries with leading space producing spurious $\rho = +0.82$.
- **Correction**: Rewrote API query client with correct parameters and count floor ($\ge 20$).
- **Effect**: Established that Dolma v1.7 counts are too sparse for higher sums; test formally classified as inconclusive.

### A27. Rebuttal of Third-Party External Audit Errors
- Corrected misstatements in external AI audits (e.g., claims of single filler template when four were run; mislabeled logit values).

### A28. Deconstruction of Premature Framing Overreach
- Retired early theoretical concepts ("Sign Inversion Paradox", "Attention Sinks as Information Movers") that lacked causal grounding.

### A29. Audit Trail Harmonization and Provenance Archiving
- Synchronized all empirical logs, ensuring every preserved figure is tied to reproducible notebook cells.


## 30. Final Claim-Strength Harmonization

The final interpretation is intentionally narrower than several early formulations. The corrected experiments preserve the numerical results and audit history while distinguishing measured results from interpretations and hypotheses.

- **[VERIFIED RESULT]** The corrected experiments measure an equal-operand/doubles advantage that persists across the tested operators and connectors.
- **[INTERPRETATION]** The observed effect is not sufficient evidence for an addition-specific circuit under the tested conditions.
- **[LIMITATION]** The M1 benchmark is scoped to the evaluated task distribution, prompt formats, tokenization scheme, and target range; it does not establish the absence of every arithmetic-related representation or mechanism in GPT-2 Small.
- **[LIMITATION]** Adv/SD is a project-defined descriptive normalized effect-size measure, not a conventional Student's t-statistic, p-value, or formal significance test.
- **[LIMITATION]** The operator-swap experiment uses a limited set of target values and matched control structures, so cross-operator persistence is not a universal statement about all operators or contexts.
- **[LIMITATION]** The external corpus-frequency analysis is inconclusive and uses Dolma v1.7 only as a proxy rather than GPT-2's exact training corpus.
- **[LIMITATION]** The 79-cell grid contains repeated target sums and related prompt structures, so cells are not fully independent replications.

These changes correct claim strength and interpretation only; experimental numerical results, DLA values, operator-swap measurements, parity measurements, ablation measurements, corpus counts, token norms, and retired-result documentation remain unchanged.


---

## 31. Literature, Narrative, and Provenance Integration

The final research flow is recorded separately from the chronological correction history in [RESEARCH_FLOW.md](RESEARCH_FLOW.md). It links each stage's question and prediction to the measurement, reported result, interpretation, limitation, and next step. The final causal validation of the equal-operand effect is listed as proposed work, not as a completed experiment.

Source relevance and source-to-claim boundaries are mapped in [LITERATURE_MAP.md](LITERATURE_MAP.md). That map covers the manuscript bibliography and source recommendations in the supplied project materials. It distinguishes literature-derived methodological context from this project's own data, records the Hase et al. reference correction, and identifies unverified run-level provenance.

No numerical measurements were changed in this integration. The existing entries A01–A29 remain the chronological audit record; the new map points from headline value families to the relevant artifacts and audit entries without replacing or reconciling their values.
