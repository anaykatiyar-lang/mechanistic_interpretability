<p align="center">
  <a href="https://colab.research.google.com/github/anaykatiyar-lang/mechanistic_interpretability/blob/main/notebooks/00_full_record.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Google Colab">
  </a>
</p>
</p>

<h1 align="center">Arithmetic Probes in GPT-2 Small</h1>

<p align="center">
  <em>When GPT-2 sees <code>3 + 3 =</code>, is it adding—or responding to a familiar pattern?</em>
</p>

<p align="center">
  A mechanistic interpretability investigation of arithmetic-like behavior, failed circuit hypotheses, and the equal-operand effect.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Research%20Draft-blue" alt="Research draft">
  <img src="https://img.shields.io/badge/Model-GPT--2%20Small-purple" alt="GPT-2 Small">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
</p>

---

## The question
What would it take to show that GPT-2 is adding—not just matching patterns?

This project began by investigating a possible addition circuit in GPT-2 Small. As baselines, attribution methods, and causal controls were scrutinized, several early circuit interpretations weakened. The investigation then shifted toward a more specific observation: **a target-dependent advantage associated with equal operands.**

The mechanism behind that effect remains an open question.

## What the results show

- **Weak zero-shot addition performance:** the recorded all-pairs evaluation achieved top-1 accuracy of 2/36. A separate held-out evaluation achieved 2/32, compared with 8/32 for its best constant-guess baseline.
- **An equal-operand pattern:** the saved digit-plus-digit scores have a mean target-level advantage of +0.247 across seven sums (exact sign-flip p = 0.03125; 95% target-bootstrap interval [+0.0738, +0.4374]). The estimate is sensitive: a sign test gives p = 0.125, and excluding controls whose operands equal the target's neighboring values gives p = 0.125 across six eligible targets. Treat this as an exploratory pattern, not a robust or mechanistic finding. The digit+digit versus digit+word contrast has Holm p = 0.046875 within its three-comparison family and was not adjusted across all project analyses.
- **Operator comparison:** at target 8, the saved `and` (+0.321) and `then` (+0.344) advantages are at least as large as `plus` (+0.291). This is descriptive: the same sum token is scored for every string, and the paired tests do not detect a difference.
- **No confirmed circuit:** candidate-component screens remain exploratory (two-sided FDR q = 0.150 across 12 nominated components); transfer effects are mixed, and the five-target discovery screen has limited p-value resolution.
- **Corpus audit:** the archived no-floor summary has positive double-versus-control joint log ratios at all seven targets, but mixed conditional ratios. No association test was prespecified, so the relationship to the model effect remains unresolved.

These conclusions apply to the tested model, prompts, targets, and controls. They do not imply that GPT-2 Small lacks every arithmetic-related representation or computation.

## Follow the investigation

**Circuit hypothesis → behavioral baselines → attribution audits → controlled comparisons → equal-operand effect → exploratory causal tests**

Explore the evidence and methods:

- [Claims and findings](docs/FINDINGS.md)
- [Methodology](docs/METHODOLOGY.md)
- [Limitations](docs/LIMITATIONS.md)
- [Research flow](docs/RESEARCH_FLOW.md)
- [Audit trail](docs/Audit_Trail.md)
- [Interactive research explorer](index.html)

## Visual evidence

![Addition-grid symmetric logit-difference heatmap](figures/static/addition_grid_heatmap.svg)

![Target-level equal-operand advantages by operator string](figures/static/operator_swap_heatmap.svg)

Explore the [interactive research explorer](index.html) for additional figures and comparisons.

Attribution and attention visualizations are not, by themselves, proof of a causal circuit. Interpret intervention results alongside their clean and corrupt baselines.

## Reproduce

**Run the notebook:** click the Colab badge at the top of this page.

To work locally:

```bash
git clone https://github.com/anaykatiyar-lang/mechanistic_interpretability.git
cd mechanistic_interpretability
python -m pip install -r requirements.txt
```

Regenerate the committed static figures:

```bash
python src/plot_heatmaps.py
```

See the notebook, [current data exports](data/current/), [data manifest](data/MANIFEST.csv), and methodology for experimental details. The corpus scripts require network access to Infini-gram; archived raw responses permit inspection of the supplied query run.

## Project structure

- `notebooks/` — full research record and Markdown export.
- `src/` — metrics, visualizations, operator comparisons, and corpus scripts.
- `data/` — committed results and artifact manifest.
- `figures/` — static and interactive figures.
- `docs/` — methods, findings, limitations, and audit history.
- `paper/` — manuscript draft.

## AI assistance

AI tools assisted with code review, debugging, methodological critique, and exploration of possible explanations. Their suggestions were treated as prompts for verification, not as evidence. Responsibility for the experiments, interpretation, and final claims remains with the author.
