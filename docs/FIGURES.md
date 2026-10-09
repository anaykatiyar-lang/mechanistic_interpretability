# Figures

## Grid and operator summaries

| File | Data source | Description |
|---|---|---|
| [`../figures/static/addition_grid_heatmap.svg`](../figures/static/addition_grid_heatmap.svg) | `data/current/addition_grid_79cell.csv` | Symmetric logit-difference heatmap for the 79-row addition grid. |
| [`../figures/static/operator_swap_heatmap.svg`](../figures/static/operator_swap_heatmap.svg) | `data/current/operator_swap_results.json` | Summary of normalized advantages across the reported operator and connector conditions. This is an aggregate visualization, not prompt-level data or a fresh model run. |

The SVGs can be regenerated with `python -m src.plot_heatmaps`. The manuscript links directly to the canonical copies in `figures/`.

## Activation patching and attention

The following interactive figures are saved HTML artifacts from the analysis. They are included for inspection; the final full-record notebook does not currently reproduce these exact files.

| File | Measurement |
|---|---|
| [`../figures/interactive/residual_stream_patching.html`](../figures/interactive/residual_stream_patching.html) | Normalized recovery by layer and sequence position for the `3 + 5 =` clean and `3 + 9 =` corrupt prompts. |
| [`../figures/interactive/attention_head_patching.html`](../figures/interactive/attention_head_patching.html) | Normalized recovery by layer and attention head at the final prompt position. |
| [`../figures/interactive/l10h2_attention_pattern.html`](../figures/interactive/l10h2_attention_pattern.html) | Attention weights across prompt tokens for layer 10, head 2. |
| [`../figures/interactive/l9h9_attention_pattern.html`](../figures/interactive/l9h9_attention_pattern.html) | Attention weights across prompt tokens for layer 9, head 9. |

The patching heatmaps show normalized recovery and can contain values below zero or above one. Interpret them using the clean/corrupt baselines and metric definition in the notebook. Attention patterns describe where a head attends; they do not alone demonstrate causal routing or arithmetic specificity. In particular, the L9H9 path-patching result remains unverified (see [audit trail A12](audit_trail.md#a12-invalid-path-patching-l9h9)).
