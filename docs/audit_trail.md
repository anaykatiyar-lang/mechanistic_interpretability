AUDIT_TRAIL.md
Methodological and code-audit history of the GPT-2 Small addition project.
Each entry: Original, Problem, Correction, Effect on conclusion, Numbers.
Entries focus on issues that affect result validity, interpretation, or unresolved evidence. Numbered identifiers are retained for cross-document references.
Tags: [VERIFIED RESULT] [INTERPRETATION] [HYPOTHESIS] [LIMITATION].
Project status: CLOSED.
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
- Retraction: a screenshot-based suspicion that `op_idx` lacked the leading space in `" +"` was a misreading; the code was correct.
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
## A24. Tokenization and multi-token answers
- Items: (i) the DiD v1 sweep scored tokens without the leading space (`"8"` vs `" 8"`), producing a spurious format effect; (ii) a fixed foil `"9"` for all targets (for target 16, a one-digit vs two-digit contrast); (iii) a "neutral" filler (`3 x 5 y`) that kept the operands and changed the final token; (iv) Hard-tier answers were suspected of being multi-token. A summary (EXT) reports ` 41` (6073) and ` 432` (46393) are single tokens, but `to_single_token` for every Hard target/foil was not confirmed; (v) the Llama tokenizer used by infini-gram splits numbers into single digits.
- Correction: leading-space targets, T±1 foils, a neutral filler; tokenization audit for 7/13/14/15; `to_single_token` errors would flag multi-token strings.
- Effect: DiD v1 (SUPERSEDED) invalidated; v2 restored the doubles advantage at 8 and 16. [LIMITATION] The Hard-tier single-token status is unverified (the `_to_id` code was not seen).
## A25. M1: baseline accuracy, subset filtering and metric mixing
- Original: M1 was "no general addition circuit"; the proposed fix was filtering to correctly answered prompts.
- Problems: (i) the strict filter left 1 of 36 prompts, and DLA on one prompt is descriptive; (ii) DLA on outcome-filtered prompts is biased by construction (the filter forces a large positive total); (iii) the −1.260 value was the few-shot prompt with foil ` 9`, not comparable to the zero-shot −0.2643; (iv) target>foil is not the same as top-1 correctness (EXT: `158 + 274`: 432 beat 422 by +0.609 yet was not in the top-10); (v) hits clustered on answer 6, but a constant-6 guess scores 5/33, equal to the model; the best constant guess (always 9) scores 8/33; (vi) the few-shot prefix contains twins (e.g. `5 + 3 = 8` vs the test `3 + 5 =`).
- Correction: top-1/top-3/strict tables; constant-guess baselines; own-answer contrast with permutation nulls.
- Effect: M1 closed as a limitation: top-1 at or below constant-guess baseline; contrasts are weak but reliable (row-centred p=0.0004 both settings); neighbour specificity not significant zero-shot. [LIMITATION] Few-shot top-1: 5/33 vs 2/32 across runs, not reconciled. My earlier "level effect" explanation of the few-shot contrast was wrong.
## A26. Frequency test: API, anchoring, statistic
- Original: `requests.post(..., {"corpus": ...})`; counted only the prompt `a + b =`; T=10 only; `.get("count", 0)` hid errors.
- Problems: the field is `index`, so an error would have silently returned 0; counting the prompt alone ignores whether the answer follows; first corrected run's leading-space anchor produced `▁▁` queries with tiny counts; with zero joint counts the conditional ratio reduces to a ratio of prompt counts, which produced an artifactual ρ = +0.82 (p=0.023) that crossed the pre-set threshold.
- Correction: `index` field, errors raised, joint and conditional counts, no extra anchor, a minimum-count floor (≥20), joint ratio as the primary statistic.
- Result: usable targets [4, 6], so **inconclusive**. Reasoning kept: frequency is neither supported nor excluded; the minus-form result and the target pattern (8/10/12/16, not 4/6/14) are inference from other experiments.
## A28. Manuscript framing overreach
- Original: a literature and novelty outline built around the "Sign Inversion Paradox" and "late-layer attention sinks masquerading as information movers."
- Problems: "sign inversion" named two different phenomena (static-bias override; local-vs-causal mismatch); the sink claim rests on L9H9, whose indirect-routing path patch was invalid (A12); the single-prompt basis was later superseded by the N=84 collapse and token-bias controls.
- Effect: neither is a supported headline finding (see FINAL_RESEARCH_STATE). Related literature surfaced in searches (greater-than circuit, IOI, copy suppression, attention-sink papers) was not systematically verified.
## A29. Cross-file value and labeling discrepancies
- The reported rank for `' 8'` differs across Run 2 and a table (rank 4) versus the controlled matrix (rank 7). The original DLA table is labeled `' 8'`/`' 6'` in the research log but `' 8'`/`' 9'` in a session record. L11H0 values −0.0448 (Run 1) and −0.1188 (Run 2) refer to different runs and must remain separate.
- Effect: these source-label discrepancies remain unresolved pending the corresponding run outputs; they are not silently reconciled.
## Summary table: did the correction change the conclusion?
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
| A25 (M1) | Closed as limitation |
| A26 (frequency) | Inconclusive; ρ = +0.82 discarded |
| A04, A07–A10, A12–A13, A20, A29 | No change to the surviving claim; data-integrity limits retained |
