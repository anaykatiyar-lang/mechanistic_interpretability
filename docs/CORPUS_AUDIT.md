# Corpus audit

The corpus collector queried exact double prompts and matched unequal-operand controls in the Dolma v1.7 proxy corpus (`v4_dolma-v1_7_llama`). The raw collector applied no eligibility floor and did not calculate a correlation. Dolma is a proxy and does not establish GPT-2 Small's training-data exposure.

The repository notebook records 79 query attempts with no failures. The later supplied notebook output records 80 attempts with one failure. Both display the same seven-target summary, but the per-attempt logs were not supplied, so the difference cannot be reconciled. These are separate saved outputs; neither is silently substituted for the other.

**Finding:** the available corpus output does not establish whether corpus frequency explains the equal-operand preference. No association test was reported by the raw collector. The raw response logs are needed before the run discrepancy or a frequency relationship can be assessed further.

See [audit trail entries A26, A30, and A31](audit_trail.md) for the run discrepancy and provenance limits, and the [data manifest](../data/MANIFEST.csv) for missing exports.
