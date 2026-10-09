# Arithmetic Prompt Processing in GPT-2 Small

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research draft](https://img.shields.io/badge/status-research%20draft-blue.svg)]()

This project traces an apparent addition-circuit signal through behavioral baselines, measurement audits, prompt controls, and causal screens. The tested suite does not establish a general addition-specific mechanism; it does retain a target-dependent equal-operand preference as a measured result and a sharper target for follow-up. This manuscript is a research draft and has not been peer reviewed.

**Explore:** [Claims ledger](docs/FINDINGS.md) · [Research flow](#research-flow) · [Interactive research explorer](index.html) · [Run the notebook](#reproduce) · [Audits](docs/audits.md)

## Research flow

**Circuit hypothesis** → behavior baseline → DLA audit → intervention controls → prompt/parity map  
**Equal-operand pattern** (+0.247; 6/7) → boundary checks → corpus audit → no circuit established → causal follow-up

Follow the links for the [methods](docs/METHODOLOGY.md), [limitations](docs/LIMITATIONS.md), [research flow](docs/RESEARCH_FLOW.md), and [audit trail](docs/audit_trail.md).

## Findings

- **Behavioral baseline:** top-1 accuracy was 2/36 (5.56%) on the all-pairs zero-shot set and 2/32 (6.25%) on the held-out zero-shot set. The best constant guess on the held-out set scored 8/32 (25%). These are separate cohorts.
- **Addition grid:** parity predicted the sign of the measured logit difference in 68/79 rows overall, 63/64 rows for sums ≤13, and 5/15 rows for sums ≥14. Rows share target sums and are not independent.
- **Equal operands:** the digit+digit scan reports a mean advantage of +0.247 (SE 0.102), positive at 6 of 7 targets. The effect is target-dependent; the mechanism is not identified. The normalized `adv/SD` ratio is descriptive, not a significance test.
- **Candidate-component screen:** the final notebook labels `10_mlp_out`, L9H9, and L10H2 as exploratory candidates. DLA q-values are 0.513; two-sided causal-screen q-values are 0.150. Transfer is mixed, so no circuit is established.
- **Corpus comparison:** unresolved. The repository notebook records 79 query attempts and no failures; a later uploaded notebook output records 80 attempts and one failure. Both display the same seven-target summary, but their raw per-attempt logs were not supplied and neither raw-collection cell reports a correlation. Earlier correlations are historical, not current findings. See the [corpus audit](docs/CORPUS_AUDIT.md).

These findings apply to the tested model, prompts, targets, and controls. They do not show that GPT-2 Small lacks all arithmetic-related representations or computations.

## Interactive figures

The script-generated figures summarize the addition grid and operator comparison:

![Addition grid symmetric logit-difference heatmap](figures/static/addition_grid_heatmap.svg)

![Operator-swap normalized advantage heatmap](figures/static/operator_swap_heatmap.svg)

Open the saved interactive HTML figures:

- [Residual-stream patching](figures/interactive/residual_stream_patching.html)
- [Attention-head patching](figures/interactive/attention_head_patching.html)
- [L9H9 attention pattern](figures/interactive/l9h9_attention_pattern.html)
- [L10H2 attention pattern](figures/interactive/l10h2_attention_pattern.html)

The final full-record notebook can also be opened in Colab:

[![Open notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anaykatiyar-lang/mechanistic_interpretability/blob/main/notebooks/00_full_record.ipynb)

The [interactive research explorer](index.html) groups the figures and evidence on one page. To publish it as a live website, select **Settings → Pages → Deploy from a branch → `main` / `(root)`** in the GitHub repository.

Figure sources and measurement scope are described in [docs/FIGURES.md](docs/FIGURES.md). Attention views are descriptive; patching displays should be read with the notebook's clean and corrupt baselines.

<details>
<summary>What the figures can and cannot show</summary>

The patching figures show normalized recovery for the tested prompts. Values below zero or above one are possible and should be interpreted with their baselines. Attention patterns show where a head attends; by themselves, they do not establish causal routing or arithmetic specificity. The L9H9 path-patching result remains unverified ([audit A12](docs/audit_trail.md#a12-invalid-path-patching-l9h9)).

</details>

## Reproduce

The notebook prints `Python: 3.10+` and records TransformerLens 3.6.0, PyTorch 2.11.0+cpu, Transformers 5.18.0, and NumPy 2.1.3. The ranges in `requirements.txt` are not a complete lockfile.

```bash
git clone https://github.com/anaykatiyar-lang/mechanistic_interpretability.git
cd mechanistic_interpretability
python -m pip install -r requirements.txt
```

Regenerate the two SVG figures from the committed grid and operator summary:

```bash
python -m src.plot_heatmaps
```

The operator comparison and corpus query scripts are documented in their module help and write separate output files by default. The corpus audit requires network access to Infini-gram. The notebook uses GPT-2 Small through TransformerLens and may require a GPU for practical execution.

## Evidence and scope

The final notebook is preserved at [`notebooks/00_full_record.ipynb`](notebooks/00_full_record.ipynb), with a [Markdown export](notebooks/00_full_record.md). Its saved outputs document the repository's uploaded run; this repository did not rerun the notebook. A later supplied notebook has a corpus collector output with one additional logged attempt and one failure; the per-attempt logs are unavailable, so this is recorded as a separate run rather than reconciled. Formatted prompt-level control scores and corpus query logs are not available here. The single-operand comparison remains unverified. See the [claims ledger](docs/FINDINGS.md), [study limitations](docs/LIMITATIONS.md), [research flow](docs/RESEARCH_FLOW.md), [literature-to-claim map](docs/LITERATURE_MAP.md), [corpus audit](docs/CORPUS_AUDIT.md), and [audit trail](docs/audit_trail.md). File-level provenance is in [`data/MANIFEST.csv`](data/MANIFEST.csv).

<details>
<summary>Audit highlights</summary>

| Audit | Correction or current scope |
|---|---|
| DLA scaling and unembedding bias | Apply final LayerNorm scaling and include the output bias term when reconciling logit differences. |
| Zero-ablation | Treat off-distribution zero-ablation cautiously; the mean-ablation control reduced the L11H0 shift from +0.0658 to +0.0013. |
| Position-1 control | Corrected the operand corruption rule; the earlier positive reading was leakage. |
| Operator patching | The original metric lacked a corrupt baseline, so the strong falsification claim is not established. |
| Corpus-frequency result | The earlier ρ=+0.82 result is retracted; raw-collection outputs differ by one logged attempt and report no correlation, so the relationship remains unresolved. |
| Candidate components | `10_mlp_out`, L9H9, and L10H2 remain exploratory; they are not a confirmed circuit. |

See the [full audit trail](docs/audit_trail.md) for the entries and source values.

</details>

## Repository contents

- `data/` — committed prompt scores, aggregate tables, and the artifact manifest.
- `docs/` — claims ledger, methodology, limitations, literature map, research flow, and canonical audit record.
- `index.html` — interactive research explorer for GitHub Pages.
- `figures/` — generated SVGs and interactive notebook figures.
- `notebooks/` — full-record notebook and Markdown export.
- `paper/draft_paper.md` — manuscript draft.
- `src/` — metric definitions, plotting, operator comparison, and corpus audit scripts.

## AI assistance

Claude Sonnet 5.5, ChatGPT, and Gemini 3.1 Pro were used to stress-test the analysis by questioning assumptions and identifying potential weaknesses. Their suggestions were treated as prompts for review, not as evidence; the author is responsible for the data, analysis, and conclusions.
