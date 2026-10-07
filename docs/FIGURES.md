# Figures

## Grid and operator summaries

| File | Data source | Description |
|---|---|---|
| [`../figures/addition_grid_heatmap.svg`](../figures/addition_grid_heatmap.svg) | `data/addition_grid_79cell.csv` | Symmetric logit-difference heatmap for the 79-row addition grid. |
| [`../figures/operator_swap_heatmap.svg`](../figures/operator_swap_heatmap.svg) | `data/operator_swap_results.json` | Summary of normalized advantages across the reported operator and connector conditions. This is an aggregate visualization, not prompt-level data or a fresh model run. |

The SVGs can be regenerated with `python -m src.plot_heatmaps`. The manuscript uses copies in `paper/figures/`.

## Activation patching and attention

The following interactive figures correspond to the saved notebook outputs. Notebook cells 47–50 write the HTML files into `figures/colab/` when executed from the repository root.

| File | Notebook cell | Measurement |
|---|---:|---|
| [`../figures/colab/residual_stream_patching.html`](../figures/colab/residual_stream_patching.html) | 47 | Normalized recovery by layer and sequence position for the `3 + 5 =` clean and `3 + 9 =` corrupt prompts. |
| [`../figures/colab/attention_head_patching.html`](../figures/colab/attention_head_patching.html) | 48 | Normalized recovery by layer and attention head at the final prompt position. |
| [`../figures/colab/l10h2_attention_pattern.html`](../figures/colab/l10h2_attention_pattern.html) | 49 | Attention weights across prompt tokens for layer 10, head 2. |
| [`../figures/colab/l9h9_attention_pattern.html`](../figures/colab/l9h9_attention_pattern.html) | 50 | Attention weights across prompt tokens for layer 9, head 9. |

The patching heatmaps show normalized recovery and can contain values below zero or above one. Interpret them using the clean/corrupt baselines and metric definition in the notebook. Attention patterns describe where a head attends; they do not alone demonstrate causal routing or arithmetic specificity. In particular, the L9H9 path-patching result remains unverified (see [audit trail A12](audit_trail.md#a12-invalid-path-patching-l9h9)).
