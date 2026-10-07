# Figure Inventory and Provenance

`figures/` is the canonical repository location for visual artifacts. Each figure has one of two sources: a repository plotting script over committed data, or a saved Colab notebook result. These sources are labelled separately so that a figure is not mistaken for an independent run or for evidence beyond its underlying experiment.

## Repository-generated figures

| File | Source | Meaning and scope |
|---|---|---|
| [`../figures/addition_grid_heatmap.svg`](../figures/addition_grid_heatmap.svg) | `src/plot_heatmaps.py` over `data/addition_grid_79cell.csv` | Symmetric logit-difference grid. Regenerate with `python -m src.plot_heatmaps`. |
| [`../figures/operator_swap_heatmap.svg`](../figures/operator_swap_heatmap.svg) | `src/plot_heatmaps.py` over the legacy `data/operator_swap_results.json` aggregate | Operator-swap summary, not prompt-level observations or a fresh model run. See the README and audit trail for the aggregate's limitations. |

The manuscript keeps publication copies of those two SVGs in `paper/figures/`. Update both copies together when regenerating them.

## Colab figure exports

The following HTML files were supplied as Colab exports and copied into the repository without changing their contents. Their Plotly specifications were compared with the corresponding saved notebook outputs: titles, axes, coordinates, and complete value matrices match. Each currently embeds the Plotly JavaScript bundle, so it is self-contained and can be opened in a browser without a network connection.

| File | Notebook cell | Figure title | Measurement shown |
|---|---:|---|---|
| [`../figures/colab/residual_stream_patching.html`](../figures/colab/residual_stream_patching.html) | 47 | Residual Stream Activation Patching (3 + 5 =) | Normalized recovery by layer and sequence position for the stated clean/corrupt prompt pair. |
| [`../figures/colab/attention_head_patching.html`](../figures/colab/attention_head_patching.html) | 48 | Attention Head Activation Patching at Position '=' | Normalized recovery by layer and attention head at the final prompt position. |
| [`../figures/colab/l10h2_attention_pattern.html`](../figures/colab/l10h2_attention_pattern.html) | 49 | L10H2 Attention Pattern (3 + 5 =) | Attention weights across the five prompt tokens for layer 10, head 2. |
| [`../figures/colab/l9h9_attention_pattern.html`](../figures/colab/l9h9_attention_pattern.html) | 50 | L9H9 Attention Pattern (3 + 5 =) | Attention weights across the five prompt tokens for layer 9, head 9. This visualization alone does not establish causal routing or arithmetic specificity. |

These are visual exports of the saved notebook results, not fresh model reruns. The patching heatmaps display normalized recovery and may include values below zero or above one; those values should be read using the metric and controls in notebook cells 47–48. They do not by themselves establish a general arithmetic circuit or validate the L9H9 path-patching claim, which the audit trail marks unverified.

## Keeping Colab and the repository aligned

1. Open or clone the repository in Colab and set the working directory to the repository root. An uploaded notebook running in an unrelated `/content` directory cannot write files back to GitHub automatically.
2. Run the notebook cells in order. Cells 47–50 now save their Plotly figures into `figures/colab/` and display them.
3. Compare each exported figure's prompt, metric, and cell with this inventory; keep distinct experimental conditions as distinct files.
4. Commit updated notebook source/outputs and the matching figure exports together. Keep `figures/` as the canonical set; `paper/figures/` is a manuscript copy of the two script-generated SVGs only.

On rerun, the notebook writes compact HTML files that load Plotly 2.35.2 from its CDN. The supplied files remain self-contained; if they are regenerated, the newer files require a network connection to load Plotly.
