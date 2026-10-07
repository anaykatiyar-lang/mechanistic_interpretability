# Arithmetic Prompt Processing in GPT-2 Small

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research draft](https://img.shields.io/badge/status-research%20draft-blue.svg)]()

This project examines whether GPT-2 Small (124M parameters) uses a general addition-specific mechanism on simple decimal addition prompts. The tested experiments do not establish such a mechanism. They identify a target-dependent preference for equal-operand prompts under selected conditions, but its cause remains unresolved. This manuscript is a research draft and has not been peer reviewed.

## Findings

- **Behavioral baseline:** top-1 accuracy was 2/36 (5.56%) on the all-pairs zero-shot set and 2/32 (6.25%) on the held-out zero-shot set. The best constant guess on the held-out set scored 8/32 (25%). These are separate cohorts.
- **Addition grid:** parity predicted the sign of the measured logit difference in 68/79 rows overall, 63/64 rows for sums ≤13, and 5/15 rows for sums ≥14. Rows share target sums and are not independent.
- **Equal operands:** the digit+digit scan reports a mean advantage of +0.247 (SE 0.102), positive at 6 of 7 targets. The effect is target-dependent; the mechanism is not identified. The normalized `adv/SD` ratio is descriptive, not a significance test.
- **Corpus comparison:** the corrected seven-target analysis found no significant association for joint counts (ρ=+0.46, p=.294) or conditional counts (ρ=−0.14, p=.760). The earlier ρ=+0.82 result is retracted.

These findings apply to the tested model, prompts, targets, and controls. They do not show that GPT-2 Small lacks all arithmetic-related representations or computations.

## Figures

The script-generated figures summarize the addition grid and operator comparison:

![Addition grid symmetric logit-difference heatmap](figures/addition_grid_heatmap.svg)

![Operator-swap normalized advantage heatmap](figures/operator_swap_heatmap.svg)

Interactive activation-patching and attention figures are available as [HTML files](figures/colab/). Figure sources and measurement scope are described in [docs/FIGURES.md](docs/FIGURES.md).

## Reproduce

The dependency ranges in `requirements.txt` are not a lockfile. The original runtime environment was not recorded, so exact historical package versions cannot be asserted.

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

The notebook contains saved outputs for the behavioral benchmark, DLA, interventions, operator comparisons, and corpus query. These outputs document the reported runs but have not all been independently regenerated from a clean environment. The committed digit+digit scores support the dense-scan table; formatted prompt-level control scores and individual corpus API responses are not available. The single-operand comparison remains unverified. See [Study limitations](docs/LIMITATIONS.md), the [research flow](docs/RESEARCH_FLOW.md), the [literature-to-claim map](docs/LITERATURE_MAP.md), and the [audit trail](docs/audit_trail.md) for the detailed evidence and correction record.

## Repository contents

- `data/` — prompt scores, aggregate tables, and corpus summaries.
- `docs/` — methodology, study limitations, literature mapping, research flow, and audit records.
- `figures/` — generated SVGs and interactive notebook figures.
- `notebooks/` — analysis notebook with saved outputs.
- `paper/draft_paper.md` — manuscript draft.
- `src/` — metric definitions, plotting, operator comparison, and corpus audit scripts.

## AI-use disclosure

The manuscript records the AI research-assistance tools used and the author's responsibility for the data, analyses, claims, and citations. AI assistance does not replace source records or independent reproduction.
