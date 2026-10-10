Audit trail for GPT-2 Small arithmetic-prompt analyses
Methodological and code-audit history of the GPT-2 Small addition project.
Each entry: Original, Problem, Correction, Effect on conclusion, Numbers.
Entries focus on issues that affect result validity, interpretation, or unresolved evidence. Numbered identifiers are retained for cross-document references.
Tags: [VERIFIED RESULT] [INTERPRETATION] [HYPOTHESIS] [LIMITATION].
Project status: supplied clean-rerun exports and raw corpus logs are archived under `data/current/`; see A32 and the data manifest.
## A01. Direct Logit Attribution without final LayerNorm scaling
- Original: component outputs from `get_full_resid_decomposition` projected directly onto W_U[:,target] − W_U[:,foil].
- Problem: components are pre-LayerNorm; LN divides by σ of the full summed stream. Raw sums overshoot.
- Correction: apply `cache.apply_ln_to_stack(..., layer=-1, pos_slice=-1)` (or divide by `ln_final.hook_scale`).
- Effect: magnitudes changed ~14× to ~40× (`11_mlp_out` +2.6263 → +0.1368; `10_mlp_out` −2.2538 → −0.1174). Rankings and signs of the largest components held. Raw magnitudes are SUPERSEDED.
- Numbers: raw Run 1 sum ≈ +8.95 vs +0.6443; corrected +0.4658 (before b_U).
## A02. Missing unembedding bias term
- Original: DLA sum compared directly to logit_diff.
- Problem: b_U[target] − b_U[foil] sits outside the residual decomposition.
- Correction: add it manually.
- Effect: closed the residual. Run 1: +0.4658 + 0.1782 = +0.6440 vs +0.6443. Run 2: +0.8382 − 1.0303 = −0.1921 vs −0.1920. Other pairs: ` 8`/` 6` −0.1003.
- [HYPOTHESIS] The non-zero b_U comes from the folded final-LayerNorm bias under `fold_ln=True`. Never verified.
- Changed the conclusion for Run 2: the circuit favors ` 8` but the static term flips the net output.
## A03. LayerNorm correction applied to a pre-sliced tensor
- Original: `accumulated_resid[:, 0, -1, :]` sliced first, then `apply_ln_to_stack(..., pos_slice=-1)`; a code comment claimed the output was `[components, batch, d_model]`.
- Problem: call-order and shape assumption. Debug shapes showed input and output both `[159, 1, 23, 768]` (4D), so the comment was wrong.
- Correction: pass the full 4D stack, slice after.
- Effect: Run 2 DLA sum +0.6761 (buggy) → +0.8382 (fixed), closing a −0.1642 gap to residual 0.0001. Run 1 was nearly unaffected (short prompt).
## A04. Stale state and untraced values
- Problem: stale token IDs and baselines produced mislabeled outputs. The −0.3539 value was a buggy Run 2 expectation, not a measurement; −2.7162 was never traced. An identical +0.1782 bias was also assigned to different pairs.
- Correction: regenerate token IDs and baselines from fresh run state before comparison.
- Effect: no final conclusion depends on the invalid −0.3539 analysis. The −2.7162 value remains unexplained.
## A06. Zero-ablation artifacts
- Original: zero-ablation of L11H0 and MLP10; double ablation judged "non-additive" (rank 8). L11H0 declared a suppressor and later a "key circuit controller."
- Problem: zero-ablation moves the residual off-distribution; σ rose from 19.2002 to 23.1286 and every tracked logit fell by ≈1.0 (' 8' −0.9995, ' 9' −1.0653, ' 2' −0.9988, ' 1' −1.0297). The top-1 flip landed on different tokens by method (' 2' zero, ' 4' mean) among near-tied tokens.
- Correction: mean-ablation using five unrelated reference sentences.
- Effect: changed the conclusion. L11H0: +0.0658 → +0.0013 (~50×; σ → 19.0148). MLP10: +0.2699 (rank 9) → +0.1675 (rank 7). Double: +0.1684 vs +0.1688 predicted, residual 0.0004 (additive). A residual σ super-additivity of +0.0291 was noted and not pursued.
## A07. Wrong ablation object and indexing
- Original (ARCHIVE): zeroing `hook_v` instead of the head output `hook_z`; an early script zeroed all layer-11 heads due to a missing head index.
- Problem: DLA decomposes the head output, not its value vector.
- Correction: `hook_z` with explicit head index.
- Effect: the dramatic top-1 flips were not interpretable regardless; resolved by A06.
## A08. Attention-pattern blocking without renormalization
- Original: `pattern[:, :, :, 1] = 0.0`, which leaves rows summing to <1.
- Problem: removes content and also total attention mass; it also zeroes position 1's self-attention.
- Correction: renormalize rows. [LIMITATION] The self-attention removal remains.
- Effect: pooled blocked mean −0.0357 → −0.0457; paired Δ +0.0399 (t≈1.76). Original single-prompt blocks: layers 0–5 −0.0994; all 12 layers −0.1690 (26.2%).
## A09. Recovery percentages with a tiny denominator
- Original: normalized recovery on `3 + 5 =` vs `1 + 5 =` (` 8`/` 6`), where clean − corrupt = 0.0152.
- Problem: noise amplified (−217.7% to +110.0%).
- Correction: rely on raw diffs; no recovery claims for that pair.
- Effect: removed any recovery claim from that sweep.
## A10. Position and pair confusion in patching
- Items: (i) "position `'5'`" (a sequence index) confused with the token ` 5` as a comparison target; (ii) in `3 + 5 =` vs `1 + 5 =` the ` 5` position is identical in both prompts, so patching it tests whether first-operand information was relayed into it; (iii) the differing position is index 1, not 0 (BOS) as once stated; (iv) the layer-0 near-full recovery on the `3 + 9 =` pair is trivial (the corrupted token's own position is overwritten); (v) the ` +` (index 2) patch was run on a pair with ` 9` as an "unrelated" foil with no printed baselines.
- Effect: the position-3 and ` +` sweeps carry no recovery claims. The `=`-position sweep (L10 to L11 movement) is the only interpretable one.
## A11. Foil changes and metric inconsistency
- Original (LOG): the foil changed from ` 6` to ` 9` between phases; Phase 4 compared with Phase 3 invalid.
- Correction: symmetric ±1 foils (N=84), parity-matched ±2 foils (D_acc), fixed T±1 neighbors in doubles work. A 79-cell finding that ±1 foils are opposite parity is a metric-sensitivity point (see A20).
- Effect: invalidated cross-phase comparisons; the final doubles design is internally consistent.
## A12. Invalid path patching (L9H9)
- Original: delta from L9H9 `hook_z` (clean − corrupt) @ W_O[9] injected into MLP10, MLP11 (`hook_mlp_in`) and L10H2 (`hook_v` via W_V[2]).
- Problem: built on the ` 8`/` 6` pair (`3 + 5 =` vs `1 + 5 =`) where L9H9 attends 3.84% to the first operand; delta norm never printed; execution of the `run_with_hooks` calls not confirmed. Output: deltas +0.0000, +0.0000, −0.0006.
- Correction: not performed. [LIMITATION] H8 (L9H9 indirect routing) unresolved. The MLP hook choice (`hook_mlp_in`) was confirmed to be architecturally sensible.
- Effect: no conclusion about L9H9 beyond a raw direct total of −0.6675 (uncorrected) vs positive patching recovery (ARCHIVE).
## A13. Cross-pair and cross-position comparisons for L3H7 and L2H2
- Original: L3H7 DLA (+0.0251, ` 8`/` 6`, from the six-head table) compared against an ablation on ` 8`/` 9`; the DLA of the position-2 logit lens used as a "sign."
- Correction: DLA recomputed at position 4 on ` 8`/` 9`: +0.0372. Ablation +0.0274 (zero) / +0.0231 (mean).
- Effect: the L3H7 mismatch survived re-derivation. [LIMITATION] L2H2's DLA (` 8`/` 6`, +0.0279) vs ablation (` 8`/` 9`, −0.0149) still compares two pairs, so "consistent" is weak.
## A14. Attention weight is not task specificity
- Original: L2H2 (0.9717) and L4H11 (1.0000) treated as binding candidates; later six-head battery.
- Problem: control prompts gave 0.7076 (L2H2) and 1.0000 (L4H11); L3H2 and L3H6 flipped sign across tokenization variants; a stray leading space in one prompt changed several values (acknowledged paste artifact).
- Correction: control prompts and LayerNorm-corrected DLA at the final position.
- Effect: changed the conclusion: five or six of seven candidates are generic; none explains the baseline (sum ≈ +0.090, wrong sign for ` 8`/` 6` −0.2643).
## A15. Corrupt-prompt parity flip and tier inconsistency
- Original: Easy/Medium corrupt rule `a → (a%8)+1` (always a+1, flipping parity); Hard used `ab → ab+10` (parity-preserving).
- Problem: confounded patching with the parity effect; tiers not comparable.
- Correction: `a → a+2` with wrap.
- Effect: 'b'-position means moved Easy +0.0197 → −0.0041, Medium +0.0033 → −0.0008; ' +' means barely moved (Easy −0.0003 → −0.0006, Medium +0.0084 → +0.0079). Qualitative conclusion unchanged (no recovery) but the positive 'b' readings were leakage.
## A16. Unscaled dataset-wide DLA; winner's-curse reasoning
- Original: top head L9H1 +0.1199 (SD 0.5370), without LayerNorm correction.
- Correction: per-item LayerNorm scale. Result +0.0064 (SD 0.0279). Mean/SD ratio unchanged (~0.23), so the shrink alone is not evidence of noise.
- Effect: conclusion unchanged ("no reliably dominant head"); the sound argument is that t ≈ 2.1 (CALC) for the best of 144 heads is what null draws give (~2.6). [LIMITATION] Approximate; prompts not independent; dataset DLA has no b_U (valid for ranking only).
## A18. Pooled null masking a real effect
- Original: pooled position-1 test (t≈1.76, p≈0.08; sign test p≈0.44) called "closed."
- Problem: sign tests are weaker; pooling mixed tiers.
- Correction: tier-stratified paired tests.
- Effect: changed the conclusion: Easy tier real (+0.0852, p=0.0116), Medium null, Hard inconclusive. An earlier claim that Hard tier drove the skew was wrong. [LIMITATION] No mean-ablation cross-check; mechanism unknown.
## A19. Operator-patching metric not anchored to a corrupt baseline (identified at archive compilation)
- Original: `patch_delta = patched_diff − base_diff`, where `base_diff` is the clean prompt's diff; all four hooks at layers 0 to 3 applied together; the earlier pooled "net recovery +0.0000". The corrupt prompt's own diff is never computed.
- Problem: patched ≈ clean is expected both under full recovery and under no corruption effect. Pooled clean means are ≈0, so a null here does not distinguish "no recovery." The same applies to the 'b'-position test.
- Correction: not performed (project closed). A valid test would use patched − corrupt (or a normalized recovery) per item, and ideally a per-item relation between recovery and the corruption effect.
- Effect: "Operator Binding Falsified" is retained historically but is NOT established. The earlier single-prompt ` +` sweep (+0.5550 → +0.6208) also had no printed baselines.
- Retraction: inspection of the source confirms that `op_idx` included the leading space in `" +"`.
## A20. Parity grid, filler breadth and foil parity
- Items: (i) the 79 grid cells are not independent (repeated target sums); (ii) filler control was described as one template; it was four (box, page, store, sequence); (iii) ±1 foils have the opposite parity of the target, so the symmetric metric directly senses parity bias (the D_acc script used ±2 foils).
- Effect: parity-bias conclusion (H10) stands on the filler reproduction (even net +0.0965, odd −0.0240) but the ~87% figure should be read with the independence caveat. No permutation test on the grid was run.
## A21. Round-number and low-baseline token effects
- Original: target-10 net −0.4206 read as an arithmetic deficit; 7+7=14 reversal unexplained.
- Correction: b_U and filler diagnostics: ' 10' b_U +3.7600, filler +1.0649; target-14 baseline −0.2311 (b_U(14) minus mean(b_U(13), b_U(15)) ≈ −0.155, CALC); tokenization of 7/13/14/15 confirmed single-token.
- Effect: target-10 and 7+7 anomalies explained as baseline artifacts. [LIMITATION] Doesn't explain why 14 shows no arithmetic signal.
## A22. First-operand "invariance" and a mislabeled logit
- Original: gap 0.0152 read as invariance; also an earlier output labeled ' 6' +0.3539.
- Problem: the digit sweep showed all digits shifted (+0.0975 to +0.4797); +0.3539 belonged to ' 9' (' 6' was +0.2694).
- Correction: digit-shift sweep.
- Effect: invariance claim retired; co-movement across digits.
## A23. Doubles analyses
- Items: (i) one control per target and noisy controls (two controls for one target disagreed by ~0.6); (ii) the DiD v1 sweep (A24) wrongly showed no doubles advantage; (iii) the LOG's §12 equation labels (6+6=12 … 16+16=32) mismatch the values (3+3=6 … 8+8=16); (iv) the dense scan counts both operand orders, so unique controls are half the printed n; T16's z=10.83 is the gap between two orders; (v) the format-sweep "replication" re-tested pre-identified targets (forking-paths concern); (vi) raw advantage growing with word forms was initially read as a stronger effect.
- Correction: unique splits, both orders averaged; normalized adv/SD; scale check; operator swaps; explicit forking-paths caveat.
- Effect: the doubles effect was kept as corroborated and target-dependent, not "confirmed." The word-form growth is mostly scale (adv/SD 5.22 / 4.40 / 4.93). Weak-baseline explanation ruled out (controls do not fall).

### Retired legacy format table

The superseded `data/variance_scaling_legacy.csv` was reviewed before retirement. Its rounded records are transcribed below for audit context; the current analysis uses `data/current/variance_scaling.csv`. Missing prompt-level source rows still prevent independent recomputation of pooled control SDs.

| target | format | double_score | control_mean | advantage | control_sd | adv_over_sd | cohort |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | digit+digit | 0.251 | 0.203 | 0.049 | 0.169 | 0.29 | null |
| 6 | digit+digit | 0.230 | 0.213 | 0.017 | 0.131 | 0.13 | null |
| 8 | digit+digit | 0.606 | 0.315 | 0.291 | 0.088 | 3.31 | positive |
| 10 | digit+digit | 0.647 | 0.113 | 0.535 | 0.056 | 9.54 | positive |
| 12 | digit+digit | 0.401 | 0.236 | 0.165 | 0.101 | 1.63 | positive |
| 14 | digit+digit | -0.238 | -0.232 | -0.006 | 0.050 | -0.12 | null |
| 16 | digit+digit | 0.719 | 0.037 | 0.682 | 0.063 | 10.83 | positive |
| 4 | digit+word | 0.380 | 0.203 | 0.177 | 0.210 | 0.84 | null |
| 6 | digit+word | 0.267 | 0.213 | 0.054 | 0.180 | 0.30 | null |
| 8 | digit+word | 0.996 | 0.454 | 0.542 | 0.155 | 3.50 | positive |
| 10 | digit+word | 1.329 | 0.396 | 0.933 | 0.172 | 5.42 | positive |
| 12 | digit+word | 0.935 | 0.499 | 0.436 | 0.184 | 2.37 | positive |
| 14 | digit+word | -0.200 | -0.232 | 0.032 | 0.160 | 0.20 | null |
| 16 | digit+word | 1.255 | 0.164 | 1.091 | 0.175 | 6.23 | positive |
| 4 | word+word | 0.303 | 0.203 | 0.100 | 0.310 | 0.32 | null |
| 6 | word+word | -0.077 | 0.213 | -0.290 | 0.295 | -0.98 | null |
| 8 | word+word | 1.542 | 0.336 | 1.206 | 0.260 | 4.64 | positive |
| 10 | word+word | 2.057 | 0.465 | 1.592 | 0.278 | 5.73 | positive |
| 12 | word+word | 1.180 | 0.459 | 0.721 | 0.285 | 2.53 | positive |
| 14 | word+word | -0.387 | -0.232 | -0.155 | 0.250 | -0.62 | null |
| 16 | word+word | 1.950 | 0.071 | 1.879 | 0.268 | 7.01 | positive |
| mean_positive_T8_10_12_16 | digit+digit | 0.593 | 0.175 | 0.418 | 0.080 | 5.22 | summary_positive |
| mean_positive_T8_10_12_16 | digit+word | 1.129 | 0.378 | 0.750 | 0.171 | 4.40 | summary_positive |
| mean_positive_T8_10_12_16 | word+word | 1.682 | 0.333 | 1.349 | 0.273 | 4.93 | summary_positive |
| mean_null_T4_6_14 | digit+digit | 0.081 | 0.061 | 0.020 | 0.117 | 0.17 | summary_null |
| mean_null_T4_6_14 | digit+word | 0.149 | 0.061 | 0.087 | 0.183 | 0.48 | summary_null |
| mean_null_T4_6_14 | word+word | -0.054 | 0.061 | -0.115 | 0.285 | -0.40 | summary_null |

### Retired prior corpus summary

The earlier summary below is historical only. Its per-query response files were not archived with the repository, and it is not a result from the final raw-query collector.

| target | addition_advantage | double_joint_count | mean_control_joint_count | log10_joint_ratio | log10_conditional_ratio |
| --- | --- | --- | --- | --- | --- |
| 4 | 0.049 | 28489 | 4305.000000 | 0.820618 | 0.040013 |
| 6 | 0.017 | 5076 | 2506.750000 | 0.306323 | 0.132547 |
| 8 | 0.291 | 3191 | 1907.166667 | 0.223447 | -0.030785 |
| 10 | 0.535 | 3812 | 1490.875000 | 0.407534 | 0.054164 |
| 12 | 0.165 | 1547 | 1151.200000 | 0.128256 | -0.030864 |
| 14 | -0.006 | 854 | 696.500000 | 0.088422 | 0.079600 |
| 16 | 0.682 | 1083 | 420.500000 | 0.410232 | 0.122524 |

## A24. Tokenization and multi-token answers
- Items: (i) the DiD v1 sweep scored tokens without the leading space (`"8"` vs `" 8"`), producing a spurious format effect; (ii) a fixed foil `"9"` for all targets (for target 16, a one-digit vs two-digit contrast); (iii) a "neutral" filler (`3 x 5 y`) that kept the operands and changed the final token; (iv) Hard-tier answers were suspected of being multi-token. A summary (EXT) reports ` 41` (6073) and ` 432` (46393) are single tokens, but `to_single_token` for every Hard target/foil was not confirmed; (v) the Llama tokenizer used by infini-gram splits numbers into single digits.
- Correction: leading-space targets, T±1 foils, a neutral filler; tokenization audit for 7/13/14/15; `to_single_token` errors would flag multi-token strings.
- Effect: DiD v1 (SUPERSEDED) invalidated; v2 restored the doubles advantage at 8 and 16. [LIMITATION] The Hard-tier single-token status is unverified (the `_to_id` code was not seen).
## A25. M1: baseline accuracy, subset filtering and metric mixing
- Original: M1 was "no general addition circuit"; the proposed fix was filtering to correctly answered prompts.
- Problems: (i) the strict filter left 1 of 36 prompts, and DLA on one prompt is descriptive; (ii) DLA on outcome-filtered prompts is biased by construction (the filter forces a large positive total); (iii) the −1.260 value was the few-shot prompt with foil ` 9`, not comparable to the zero-shot −0.2643; (iv) target>foil is not the same as top-1 correctness (EXT: `158 + 274`: 432 beat 422 by +0.609 yet was not in the top-10); (v) hits clustered on answer 6, but a constant-6 guess scores 5/33, equal to the model; the best constant guess (always 9) scores 8/33; (vi) the few-shot prefix contains twins (e.g. `5 + 3 = 8` vs the test `3 + 5 =`).
- Correction: top-1/top-3/strict tables; constant-guess baselines; own-answer contrast with permutation nulls.
- Effect: M1 closed as a limitation: top-1 at or below constant-guess baseline; contrasts are weak but reliable (row-centred p=0.0004 both settings); neighbour specificity not significant zero-shot. [LIMITATION] Few-shot top-1: 5/33 vs 2/32 across runs, not reconciled. The level-effect interpretation of the few-shot contrast was unsupported.
## A26. Frequency test: API, anchoring, statistic
- Original: `requests.post(..., {"corpus": ...})`; counted only the prompt `a + b =`; T=10 only; `.get("count", 0)` hid errors.
- Problems: the field is `index`, so an error would have silently returned 0; counting the prompt alone ignores whether the answer follows; first corrected run's leading-space anchor produced `▁▁` queries with tiny counts; with zero joint counts the conditional ratio reduces to a ratio of prompt counts, which produced an artifactual ρ = +0.82 (p=0.023) that crossed the pre-set threshold.
- Correction: `index` field, errors raised, joint and conditional counts, no extra anchor, a minimum-count floor (≥20), joint ratio as the primary statistic.
- Result: usable targets [4, 6], so **inconclusive**. Reasoning kept: frequency is neither supported nor excluded; the minus-form result and the target pattern (8/10/12/16, not 4/6/14) are inference from other experiments.
## A28. Manuscript framing overreach
- Original: a literature and novelty outline built around the "Sign Inversion Paradox" and "late-layer attention sinks masquerading as information movers."
- Problems: "sign inversion" named two different phenomena (static-bias override; local-vs-causal mismatch); the sink claim rests on L9H9, whose indirect-routing path patch was invalid (A12); the single-prompt basis was later superseded by the N=84 collapse and token-bias controls.
- Effect: neither is a supported headline finding (see [RESEARCH_FLOW.md](RESEARCH_FLOW.md) and [LIMITATIONS.md](LIMITATIONS.md)). [LITERATURE_MAP.md](LITERATURE_MAP.md) describes each cited source, its relevance, and its scope; literature remains methodological context, not evidence for project measurements.
## A29. Cross-record value and label discrepancies
- The reported rank for `' 8'` differs across Run 2 and a table (rank 4) versus the controlled matrix (rank 7). One DLA table labels the contrast as `' 8'`/`' 6'`, while the controlled matrix uses `' 8'`/`' 9'`. L11H0 values −0.0448 (Run 1) and −0.1188 (Run 2) refer to different runs and must remain separate.
- Effect: these source-label discrepancies remain unresolved pending the corresponding run outputs; they are not silently reconciled.

## A30. Exploratory mechanism screen and original corpus-log status
- **Source:** Final full-record notebook analysis cells 145–167 and their saved output tables.
- **Candidate screen:** `10_mlp_out`, L9H9, and L10H2 are exploratory candidates. The component DLA screen reports q_FDR = 0.513; the two-sided causal screen reports q_FDR = 0.150. Neither passes q < .05.
- **Transfer and specificity:** At targets 8 and 14, joint ablation lowers the target-level advantage at 8 and raises it at 14. In the prospective range 24–36, the signed advantage change is negative at five targets and positive at two; clean contrasts at 26 and 34 are negative. This remains exploratory transfer, not confirmation. Filler prompts also show effects.
- **Completeness:** The target-level completeness ratios vary from 0.54 to 1.54; the largest ratio occurs where the full effect is small. Do not summarize these ratios as a stable circuit-level removal fraction.
- **Original corpus log status:** At the time of this record, raw per-request files were not part of the repository. The supplied raw CSV/JSONL and reconstructed no-floor summary are now archived under `data/current/` (A32). No eligibility floor or association test was applied to this export; prior correlation results are historical. See [CORPUS_AUDIT.md](CORPUS_AUDIT.md).

## A32. Supplied clean-rerun exports integrated into canonical data
- **Source:** Clean-rerun CSV/JSONL outputs. Canonical outputs are stored once under `data/current/`; row counts and SHA-256 digests are in `data/MANIFEST.csv`.
- **Byte checks:** `doubles_scan_scores.csv` (39 rows) and `variance_scaling.csv` (21 rows) byte-match the existing canonical files. The two supplied final-summary aliases are identical and stored once. No experimental numbers were edited or merged.
- **Primary equal-operand result:** Mean advantage +0.2473847 across seven target levels; exact target-level sign-flip p = 0.03125; 10,000-resample target bootstrap 95% percentile interval [+0.0738152, +0.4374255], seed 20261008. The effect-size ratio is descriptive. Interpretation is bounded to these seven target levels.
- **Format inference:** Digit+digit versus digit+word paired difference −0.2189595, raw p = 0.015625, Holm p = 0.046875. Other format contrasts have Holm p = 0.21875. The contrast is target-level and does not establish general invariance.
- **Operator inference:** Plus-versus-other paired Holm p = 0.6875 for each comparison. This is failure to detect a difference, not evidence of equivalence.
- **Candidate causal screen:** For nominated components, one-sided raw p = 0.03125 each, but one-sided FDR q = 0.09375; two-sided FDR q = 0.15. No nominated component passes q < 0.05. Prospective-range signed advantage change decreases at five targets and increases at two; clean advantages at targets 26 and 34 are already negative. These outputs do not establish a circuit.
- **Inference sensitivity added to manuscript:** The seven-target primary sign-flip result (mean +0.2473847, p = 0.03125) is accompanied by the sign test (6/7 positive, p = 0.125), leave-one-target-out tests (p = 0.03125 or 0.0625), and the neighbor-operand exclusion check (six eligible targets, mean +0.256077, exact sign-flip p = 0.125; target 4 has no retained controls). These checks are sensitivity analyses and do not change the saved measurements.
- **Candidate-screen scope clarified:** The discovery cohort contains five targets; the FDR adjustment family contains 12 nominated components. The prospective transfer cohort is even targets 24, 26, 28, 30, 32, 34, and 36. Its selected-component results are exploratory and not independent confirmation.
- **Corpus export:** Raw CSV and JSONL contain 79 request attempts, 0 errors, and 79 HTTP 200 responses. A seven-target no-floor summary is reconstructed from the raw log using notebook cell 147. Double joint counts exceed mean ordered-control joint counts for all seven targets; conditional log ratios vary in sign. No association test or eligibility floor was applied. Historical correlations remain historical and `ρ=+0.82, p=0.023` remains retracted.
- **Scope:** The supplied output exports do not include every format/operator prompt score. The single-operand control remains untested. No data values were silently reconciled.

## Summary of research-relevant corrections
| Issue | Changed conclusion? |
|---|---|
| A01, A02, A03 (DLA math) | Yes for magnitudes; Run 2 reinterpreted as static-bias override |
| A06 (zero-ablation) | Yes (L11H0 suppressor retired; interaction retired) |
| A14 (attention vs specificity) | Yes (generic heads) |
| A15, A16 (N=84 parity, scaling) | Qualitatively no; numbers changed |
| A18 (pooled null) | Yes (Easy-tier effect found) |
| A19 (operator-patching anchor) | Yes for the status of "falsified" (not established) |
| A21 (round numbers) | Yes (target-10 and 7+7 anomalies explained) |
| A22 (invariance) | Yes (retired) |
| A23, A24 (doubles design and tokenization) | Yes (v1 artifact retired; effect corroborated, not confirmed) |
| A25 (M1) | Closed as a scoped limitation; cohort definitions separated |
| A26, A30, and A32 (frequency) | Earlier ρ = +0.82 discarded; the 79-request raw log is archived; no association is reported and the relationship remains unresolved |
| A04, A07–A10, A12–A13, A20, A29 | No change to the surviving claim; data-integrity limits retained |
