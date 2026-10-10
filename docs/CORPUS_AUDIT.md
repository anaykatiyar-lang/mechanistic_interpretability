# Corpus audit

The corpus collector queried exact double prompts and matched ordered unequal-operand controls in the Dolma v1.7 proxy index (`v4_dolma-v1_7_llama`). The supplied clean-rerun export is archived in `data/current/corpus_raw_api_query_log.csv` and `.jsonl`. It contains 79 logged attempts, all HTTP 200, with zero recorded errors. The seven-target no-floor summary is archived at `data/current/corpus_raw_target_summary_no_floor.csv` and reconstructed from those request/response records using the notebook collector's cell 147 aggregation.

For all seven targets, the raw joint count for the double string exceeds the mean count for ordered controls (positive joint $\log_{10}$ ratio, not a correlation coefficient). Conditional $\log_{10}$ ratios are mixed in sign. These are descriptive corpus counts only. No eligibility floor was applied and no association test between corpus counts and model advantage was run. Dolma is a proxy, not GPT-2's training corpus; these results cannot establish actual training exposure or explain the model's equal-operand effect.

| Target | Double joint | Double prefix | Ordered controls | Mean control joint | Mean control prefix | Joint $\log_{10}$ ratio | Conditional $\log_{10}$ ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 28489 | 64946 | 2 | 4305.000 | 10762.500 | +0.8206 | +0.0400 |
| 6 | 5076 | 11432 | 4 | 2506.750 | 7661.750 | +0.3063 | +0.1325 |
| 8 | 3191 | 9251 | 6 | 1907.167 | 5151.333 | +0.2234 | -0.0308 |
| 10 | 3812 | 9848 | 8 | 1490.875 | 4364.375 | +0.4075 | +0.0542 |
| 12 | 1547 | 5074 | 6 | 1151.167 | 3517.167 | +0.1283 | -0.0309 |
| 14 | 854 | 2759 | 4 | 696.500 | 2703.500 | +0.0884 | +0.0796 |
| 16 | 1083 | 3787 | 2 | 420.500 | 1952.000 | +0.4102 | +0.1225 |

The supplied raw export resolves the former absence of per-attempt logs. The historical `ρ=+0.82, p=0.023` result remains retracted. The earlier `ρ=+0.46` / `ρ=−0.14` correlations are retained as notebook analyses, separate from the descriptive summary reconstructed from this raw log.

See [audit trail A26, A30, and A32](audit_trail.md) and the [data manifest](../data/MANIFEST.csv) for provenance.
