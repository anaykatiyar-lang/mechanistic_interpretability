# 🔬 Transformer Interpretability Research

<p align="center">
  <img src="https://img.shields.io/badge/Model-GPT--2%20Small-blue?style=flat-square" alt="Model">
  <img src="https://img.shields.io/badge/Focus-Mechanistic%20Interpretability-purple?style=flat-square" alt="Focus">
  <img src="https://img.shields.io/badge/Python-3.8%2B-green?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License">
</p>

## 📌 Overview

This project explores the interpretability of Transformer models, focusing on understanding how these models process and generate information. We utilize advanced techniques such as **activation patching**, logit lens, and circuit tracing to investigate the internal workings of pre-trained language models [cite: topic-Structured a Mechanistic Interpretability research project analyzing failure taxonomies, attention drift, activation patching, logit lens, and circuit tracing in language models such as GPT-2 Small and Phi-3-mini-4k-instruct, setting up an arXiv pre-print and GitHub repository], specifically **GPT-2 Small**.

---

## 📂 Repository Structure

The repository is organized to enhance clarity, reproducibility, and collaboration:

```text
├── src/           # Python source code files (reusable functions, classes, and scripts)
├── data/          # Raw and processed datasets
├── notebooks/     # Jupyter & Google Colab notebooks for exploratory analysis
├── results/       # Generated outputs, figures, tables, and model checkpoints
└── docs/          # Additional project documentation and notes
# 🔬 Mechanistic Interpretability: Transformer Internal Representations

<p align="center">
  <img src="[https://img.shields.io/badge/Models-GPT--2%20Small%20%7C%20Phi--3-blue?style=flat-square](https://img.shields.io/badge/Models-GPT--2%20Small%20%7C%20Phi--3-blue?style=flat-square)" alt="Models">
  <img src="[https://img.shields.io/badge/Focus-Mechanistic%20Interpretability-purple?style=flat-square](https://img.shields.io/badge/Focus-Mechanistic%20Interpretability-purple?style=flat-square)" alt="Focus">
  <img src="[https://img.shields.io/badge/Toolkit-TransformerLens-green?style=flat-square](https://img.shields.io/badge/Toolkit-TransformerLens-green?style=flat-square)" alt="TransformerLens">
  <img src="[https://img.shields.io/badge/Status-Active%20Research-orange?style=flat-square](https://img.shields.io/badge/Status-Active%20Research-orange?style=flat-square)" alt="Status">
</p>

## 📌 Project Overview

This repository houses a mechanistic interpretability research project aimed at reverse-engineering the internal mechanisms of pre-trained language models. Moving beyond black-box evaluations, this work investigates how models process information internally, tracking specific computational sub-graphs and failure modes.

---

## 🔍 Core Research Areas

* **Activation Patching:** Causal intervention experiments to localize where specific information is computed and transmitted across layers and heads.
* **Logit Lens & Circuit Tracing:** Mapping intermediate residual stream states directly to vocabulary space to observe how predictions evolve across network depth.
* **Attention Drift & Failure Taxonomies:** Analyzing attention pattern anomalies and routing failures under distribution shifts or specific prompt structures.
* **Target Models:** Primary experimentation on `GPT-2 Small` with extension to `Phi-3-mini-4k-instruct`.

---

## 📂 Repository Structure

The repository is organized to enhance clarity, reproducibility, and collaboration:

```text
├── src/           # Modular Python scripts (patching hooks, metrics, utils)
├── data/          # Evaluation datasets and prompt templates
├── notebooks/     # Exploratory analysis & interactive experiment notebooks
├── results/       # Generated figures, attention heatmaps, and metrics logs
└── docs/          # Extended write-ups, failure taxonomies, and pre-print drafts
```

---

## ⚙️ Setup & Installation

To set up the local environment and install the necessary dependencies, please follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install dependencies:** Make sure you have Python 3.8+ installed. It is recommended to use a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### Running Notebooks
All exploratory analysis and experiments are conducted within Jupyter/Colab notebooks located in the `notebooks/` directory. To run them:
1. Navigate to the `notebooks/` directory.
2. Open the desired notebook in Jupyter or upload it to Google Colab with a GPU runtime.
3. Execute the cells sequentially.

### Running Scripts
For reusable functions, validation scripts, or specific analyses, Python scripts can be found in the `src/` directory. To run a script:
```bash
python src/your_script_name.py
```

---

## 📈 Reproducibility

To reproduce the results presented in this project:
1. Ensure you have followed the Setup instructions to install all dependencies.
2. Run the notebooks in the `notebooks/` directory. Each notebook is designed to be self-contained for its specific analysis.
3. Refer to the `results/` directory for generated outputs and figures that might be referenced in any documentation or papers.

Inspect generated figures, evaluation tables, and outputs under the results/ directory.

This structured approach aims to make the research process transparent, modular, and verifiable for upcoming arXiv pre-print submissions [cite: topic-Structured a Mechanistic Interpretability research project analyzing failure taxonomies, attention drift, activation patching, logit lens, and circuit tracing in language models such as GPT-2 Small and Phi-3-mini-4k-instruct, setting up an arXiv pre-print and GitHub repository].
