# Arithmetic Prompt Processing in GPT-2 Small

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research draft](https://img.shields.io/badge/status-research%20draft-blue.svg)]()

This repository contains a research draft and a partial set of analysis artifacts for an investigation of arithmetic-shaped prompts in GPT-2 Small (124M parameters). It is not a peer-reviewed publication. Several headline results remain documented without the originating notebook, raw outputs, or table-generation code; see the provenance notes below before treating the numbers as independently reproduced.

## Research question and scoped result

The project asks whether GPT-2 Small's apparent preference for addition answers supports a general, addition-specific mechanism. The tested suite does not establish such a mechanism. A target-dependent equal-operand preference is reported in several prompt formats and operator substitutions, but its mechanism is unresolved. Operator persistence weakens a simple addition-specific surface-form account; it does not establish universal operator blindness.

The committed 79-row addition grid can be recomputed from its CSV: parity predicts the sign in 68/79 cells (86.1%) overall, 63/64 cells with sums at most 13, and 5/15 cells with sums of at least 14. The high-sum pattern differs sharply from the lower-sum pattern. These cells share target sums and are not independent observations.

## Important provenance limits

- The manuscript reports zero-shot top-1 accuracy as 6.2% (2/36). The count 2/36 is approximately 5.6%, and the reported 25% constant-guess baseline uses an N=32 cohort. The originating M1 results and prompt file are not committed, so these values are retained as reported and the direct comparison remains unresolved.
- `data/operator_swap_results.json` contains legacy aggregates without per-prompt scores. Its df metadata matches full controls, but the previous committed script used incomplete controls and ASCII `*`/`-`; the research log records `×`/`−`. The corrected script writes a separate reproduction artifact and supports an ASCII sensitivity run. The legacy aggregates have not been regenerated in this checkout.
- `data/variance_scaling.csv` contains reported aggregates without raw prompt scores. Means recompute from the displayed per-target rows to rounding. The stored word+word pooled SD (0.273) does not recompute from the displayed rounded per-target SDs (approximately 0.275); raw scores are absent, so neither value is silently replaced.
- `data/corpus_frequencies.json` is a legacy aggregate with no per-control counts. Its floor flags and verdict cannot be regenerated from that file. The corrected query script reports alternative floor definitions and preserves API failures as errors; the historical count artifact is not overwritten.
- The repository does not contain the originating notebook or row-level outputs for the M1 benchmark, DLA tables, ablations, N=84 analyses, or the single-operand control. These remain reported findings, not independently reproducible results from this repository.
- DLA raw/corrected pairs shown in the manuscript imply a scale factor near 19.2. The broader “14×–40×” range is not supported by the displayed pairs. The audit trail preserves that earlier wording as history and marks the unsupported range.

The source-to-claim and literature map is in [`docs/LITERATURE_MAP.md`](docs/LITERATURE_MAP.md). The experiment narrative is in [`docs/RESEARCH_FLOW.md`](docs/RESEARCH_FLOW.md), methodological details are in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md), and scope/provenance limits are in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md). The detailed correction record is [`docs/audit_trail.md`](docs/audit_trail.md).

## Repository contents

```text
mechanistic_interpretability/
├── README.md
├── requirements.txt
├── data/
│   ├── addition_grid_79cell.csv
│   ├── operator_swap_results.json          # legacy aggregate; provenance caveat above
│   ├── variance_scaling.csv                 # reported aggregates; raw prompt scores absent
│   └── corpus_frequencies.json              # legacy aggregate; provenance caveat above
├── docs/
│   ├── audit_trail.md
│   ├── LITERATURE_MAP.md
│   ├── LIMITATIONS.md
│   ├── METHODOLOGY.md
│   ├── RESEARCH_FLOW.md
│   └── RESEARCH_LOG.md
├── figures/
├── paper/
│   └── draft_paper.md
└── src/
    ├── config.py
    ├── frequency_audit.py
    ├── metrics.py
    ├── operator_swap.py
    ├── plot_heatmaps.py
    └── token_diagnostics.py
```

## Setup and commands

The dependency ranges in `requirements.txt` are not a lockfile. The environment used for the original reported runs was not captured, so exact historical package versions cannot be asserted.

```bash
git clone https://github.com/anaykatiyar-lang/mechanistic_interpretability.git
cd mechanistic_interpretability
python -m pip install -r requirements.txt
```

Regenerate the SVG figures from the committed grid and legacy operator summary:

```bash
python -m src.plot_heatmaps
```

Run the operator comparison with all valid ordered unequal splits. By default, this tests the Unicode `×` and `−` glyphs documented in the research log and writes a new detailed file without overwriting the legacy aggregate:

```bash
python -m src.operator_swap --operator-variant reported_unicode --output data/operator_swap_reproduction.json
```

Run an additional ASCII tokenization-sensitivity condition:

```bash
python -m src.operator_swap --operator-variant both --output data/operator_swap_reproduction.json
```

The corpus audit queries Infini-gram and writes every queried prompt/count. It reports all three historical floor definitions rather than selecting one without source evidence. Failed API queries make the run fail and are recorded; they are never treated as zero counts.

```bash
python -m src.frequency_audit --output data/corpus_frequency_audit_reproduction.json
```

`src.token_diagnostics` requires GPT-2 Small through TransformerLens. Its token-bias and filler outputs are computed by the script; the static-bias override reconstruction includes an archived DLA value and is not a fresh DLA calculation.

## Claim boundaries

- **Reported result:** an equal-operand advantage appears for selected target sums and persists in the archived operator/connector summaries.
- **Interpretation:** this weakens a simple addition-specific explanation for that measured preference.
- **Not established:** a general arithmetic algorithm, a universal operator-blind mechanism, or a causal explanation for the equal-operand effect.
- **Unresolved:** several run-level values and corpus counts lack their originating files; the audit trail identifies these gaps.

The manuscript is a research draft. AI-use disclosure and responsibility statements are in the manuscript; AI assistance does not substitute for the missing source runs or independent reproduction.
