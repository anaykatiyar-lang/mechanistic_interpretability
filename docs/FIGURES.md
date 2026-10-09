# Figures

## Grid and operator summaries

| File | Data source | Description |
|---|---|---|
| [`../figures/static/addition_grid_heatmap.svg`](../figures/static/addition_grid_heatmap.svg) | `data/current/addition_grid_79cell.csv` | Symmetric logit-difference heatmap for the 79-row addition grid. |
| [`../figures/static/operator_swap_heatmap.svg`](../figures/static/operator_swap_heatmap.svg) | `data/current/equal_operand_operator_advantages.csv` | Seven-target matched equal-operand advantages by operator/connector. Paired plus-versus-other Holm p-values are 0.6875; lack of detection is not equivalence. |
| [`../paper/figures/addition_grid_heatmap.pdf`](../paper/figures/addition_grid_heatmap.pdf) | `data/current/addition_grid_79cell.csv` | Publication-format vector rendering of Figure 1. |
| [`../paper/figures/operator_target_advantages.pdf`](../paper/figures/operator_target_advantages.pdf) | `data/current/equal_operand_operator_advantages.csv` | Publication-format target-by-operator heatmap of Figure 2. |

Regenerate the SVG figures with `python src/plot_heatmaps.py`; the manuscript vector assets are generated from the same canonical CSVs with `python src/render_paper_figures.py`.

## Activation patching and attention

Interactive figures are saved HTML artifacts for inspection. Attention patterns describe where a head attends; they do not alone demonstrate causal routing or arithmetic specificity. In particular, the L9H9 path-patching result remains unverified (see [audit trail A12](audit_trail.md#a12-invalid-path-patching-l9h9)).
