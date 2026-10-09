<p align="center">
  <a href="https://colab.research.google.com/github/anaykatiyar-lang/mechanistic_interpretability/blob/main/notebooks/00_full_record.ipynb">
    <img
      src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Colaboratory_SVG_Logo.svg"
      alt="Open notebook in Google Colab"
      width="100"
    />
  </a>
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

A model can produce arithmetic-like answers without necessarily implementing a general arithmetic algorithm.

This project began by investigating a possible addition circuit in GPT-2 Small. As baselines, attribution methods, and causal controls were scrutinized, several early circuit interpretations weakened. The investigation then shifted toward a more specific observation: **a target-dependent advantage associated with equal operands.**

The mechanism behind that effect remains an open question.

## What the results show

- **Weak zero-shot addition performance:** the recorded all-pairs evaluation achieved top-1 accuracy of 2/36. A separate held-out evaluation achieved 2/32, compared with 8/32 for its best constant-guess baseline.
- **An equal-operand effect:** the committed digit-plus-digit scan reports a mean advantage of +0.247, positive at 6 of 7 target levels. This is a descriptive result, not a significance test.
- **No confirmed circuit:** the component analyses remain exploratory, with mixed transfer results. The evidence does not establish a general addition-specific mechanism.
- **An unresolved corpus comparison:** the current corpus evidence is insufficient to establish a reliable relationship between corpus co-occurrence and the measured model effect.

These conclusions apply to the tested model, prompts, targets, and controls. They do not imply that GPT-2 Small lacks every arithmetic-related representation or computation.

## Follow the investigation

**Circuit hypothesis → behavioral baselines → attribution audits → controlled comparisons → equal-operand effect → exploratory causal tests**

Explore the evidence and methods:

- [Claims and findings](docs/FINDINGS.md)
- [Methodology](docs/METHODOLOGY.md)
- [Limitations](docs/LIMITATIONS.md)
- [Research flow](docs/RESEARCH_FLOW.md)
- [Audit trail](docs/audit_trail.md)
- [Interactive research explorer](index.html)

## Visual evidence

![Addition-grid symmetric logit-difference heatmap](figures/static/addition_grid_heatmap.svg)

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
python -m src.plot_heatmaps
```

See the notebook and methodology documentation for experimental details and environment requirements. The corpus scripts require network access to Infini-gram.

## Project structure

- `notebooks/` — full research record and Markdown export.
- `src/` — metrics, visualizations, operator comparisons, and corpus scripts.
- `data/` — committed results and artifact manifest.
- `figures/` — static and interactive figures.
- `docs/` — methods, findings, limitations, and audit history.
- `paper/` — manuscript draft.

## AI assistance

AI tools assisted with code review, debugging, methodological critique, and exploration of possible explanations. Their suggestions were treated as prompts for verification, not as evidence. Responsibility for the experiments, interpretation, and final claims remains with the author.
