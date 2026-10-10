# Audit index

[`audit_trail.md`](audit_trail.md) is the single detailed audit record. This page is a reader index; it does not repeat the underlying audit entries.

| Audit | Main correction | Effect on the claim |
|---|---|---|
| [A01–A03](audit_trail.md#a01-direct-logit-attribution-without-final-layernorm-scaling) | Correct final LayerNorm scaling, tensor order, and output-bias accounting in DLA. | Corrects attribution magnitudes and reconciles selected logit differences. |
| [A06–A09](audit_trail.md#a06-zero-ablation-artifacts) | Separate off-distribution zero-ablation shifts from mean-ablation results; retire tiny-denominator recovery ratios. | Weakens early suppressor and recovery claims. |
| [A12–A16](audit_trail.md#a12-invalid-path-patching-l9h9) | Audit invalid path patches, controls, parity confounds, and multiple candidate selection. | Attention and localization observations do not establish task-specific causal routes. |
| [A18](audit_trail.md#a18-pooled-null-masking-a-real-effect) | Stratify the pooled position-1 analysis by difficulty tier. | A scoped Easy-tier effect remains; other tiers are not equivalent evidence. |
| [A19](audit_trail.md#a19-operator-patching-metric-not-anchored-to-a-corrupt-baseline-identified-at-archive-compilation) | Identify the missing corrupt baseline in operator-position patching. | “Operator binding falsified” is not established. |
| [A23–A25](audit_trail.md#a23-doubles-analyses) | Refine doubles controls and tokenization; distinguish benchmark cohorts. | The equal-operand effect is target-dependent and corroborated, not confirmed as a mechanism. |
| [A26](audit_trail.md#a26-frequency-test-api-anchoring-statistic) | Correct corpus API/query construction; retire the `ρ=+0.82` result. | Corpus-frequency conclusion is unresolved/inconclusive. |
| [A30](audit_trail.md#a30-final-notebook-mechanism-screen-and-corpus-log-status) | Record the exploratory component screen and former raw-log gap; current raw logs are archived under `data/current/`. | Candidate components remain exploratory; no circuit-level claim is supported. |
| [A32](audit_trail.md#a32-supplied-clean-rerun-exports-integrated-into-canonical-data) | Integrate supplied current-run exports and update target-level inference, intervention, and corpus provenance. | Primary target-level effect is nominally significant; no validated circuit or corpus-frequency association is established. |

The audit trail preserves historical values and labels them as superseded, retracted, or unresolved where appropriate. It is the source of truth for correction history.
