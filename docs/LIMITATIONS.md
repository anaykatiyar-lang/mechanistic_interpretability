# Methodological Limitations and Boundary Conditions

This document catalogs the exact scope limitations, tokenization artifacts, statistical noise bounds, and unresolved questions that constrain the conclusions of this project.

---

## 1. Scope and Architectural Constraints

1. **Single Model Architecture**:
   All empirical findings are derived from GPT-2 Small (124M parameters, 12 layers). No inferences are made regarding larger model scales (GPT-2 Medium/XL, Llama, Gemma) or models trained with modern code/math synthesis data.
2. **Restricted Arithmetic Domain**:
   Evaluations focus primarily on single-digit addends ($a, b \in [1, 9]$) producing sums between 2 and 18. Hard-tier multi-digit prompts ($N=5$) provide suggestive evidence but remain statistically underpowered.
3. **Single-Prompt Attribution Caveats**:
   Several initial mechanistic insights (e.g., L11H0/MLP10 additivity, static-bias override) were derived on isolated prompts (`"3 + 5 ="`). While algebraically exact on those prompts, they do not constitute universal circuit motifs across the model.

---

## 2. Token Geometry and Representation Quirks

1. **Leading Whitespace Sensitivity**:
   In GPT-2's byte-pair encoding (BPE), tokens with and without a leading space occupy completely different vocabulary indices:
   - Target `' 8'` = token ID 807
   - Target `'8'` = token ID 23
   Evaluating targets without leading spaces (as in DiD v1) produces spurious format swings of over 1.4 logits that are purely tokenization artifacts.
2. **Single vs Multi-Token Representations**:
   While single-digit and early two-digit answers (`' 10'`, `' 12'`, `' 14'`, `' 16'`) are single BPE tokens in GPT-2, external tokenizers (such as Llama used in Dolma indexing) segment multi-digit numbers into individual digits. This creates token geometry mismatches during external corpus frequency matching.
3. **Round-Number and Static Bias Skew ($b_U$)**:
   - `' 10'` exhibits an abnormally high static unembedding bias ($b_U = +3.7600$) and is generically favored by $+1.0649$ logits on non-arithmetic filler prompts.
   - Target `' 14'` exhibits an abnormally low baseline ($-0.2311$), explaining why $7 + 7 = 14$ appears as an anomaly in raw metrics. These are properties of vocabulary embedding geometry, not arithmetic computation failures.

---

## 3. Statistical Noise-Floor Bounds and the Winner's Curse

1. **Dataset-Wide DLA Winner's Curse**:
   In uncorrected dataset-wide DLA sweeps, head L9H1 appeared prominent ($+0.1199$). After proper LayerNorm scaling, its mean contribution shrank to $+0.0064$ ($\text{SD} = 0.0279$), yielding a test statistic of $t \approx 2.1$. For the maximum of 144 independent standard normal draws, an extreme value of $t \approx 2.6$ is expected purely by chance. Thus, L9H1 cannot be designated a causal arithmetic driver.
2. **79-Cell Grid Non-Independence**:
   The 79 single-digit grid cells feature repeated target sums (e.g., target 10 produced by $1+9, 2+8, 3+7, 4+6, 5+5$). While the 87% parity alignment is corroborated on independent filler controls, the grid trials are not statistically independent.
3. **Dense Scan Sample Limitations**:
   Because each target sum has only one true double ($d+d$), target 4 ($2+2$) and target 16 ($8+8$) have only one unique control pair (evaluated in two operand orders).

---

## 4. Unanchored Metrics and Experimental Blindspots

1. **Unanchored Operator-Position Patching (A19)**:
   In the operator-position patching experiment, the metric evaluated was:
   $$\Delta_{\text{patch}} = \text{diff}_{\text{patched}} - \text{diff}_{\text{clean}}$$
   Because the corrupted prompt's baseline logit difference was not included as an anchor, a delta near zero cannot distinguish between "full recovery of corrupt state" and "no corruption effect". Consequently, "Operator Binding Falsified" cannot be claimed as conclusively proven, and remains an open question.
2. **Invalid L9H9 Path Patching (A12)**:
   Early attempts to demonstrate indirect routing from L9H9 into MLP10 and L10H2 failed to report delta norms or confirm hook execution on CPU runtimes. The hypothesis that L9H9 acts as an information router remains unverified.
3. **Easy-Tier Attention Blocking Mechanism**:
   Position-1 attention blocking produced a statistically significant reduction in target logit diff on Easy-tier prompts ($p = 0.0116$), but Medium-tier prompts showed no effect ($p = 0.724$). The exact internal mechanism mediating this difference remains unidentified.

---

## 5. Corpus Frequency Audit Boundaries

1. **Corpus Proxy Discrepancy**:
   Dolma v1.7 is an open 3-trillion-token corpus used as a proxy for WebText. Because WebText is proprietary and unreleased, true pretraining co-occurrence statistics cannot be measured directly.
2. **Sparsity Floor Violation**:
   Exact joint queries (`"4 + 4 = 8"`, `"8 + 8 = 16"`) yielded fewer than 20 occurrences in Dolma. Consequently, corpus frequency could neither be confirmed nor rejected based on n-gram counts alone.
3. **Run 1 Artifact Retraction**:
   An initial analysis reported a strong correlation ($\rho = +0.82$, $p = 0.023$) between corpus frequency and model advantage. Subsequent investigation revealed that leading-space query formatting had produced zero joint matches, causing the ratio to degenerate into a trivial prompt-count artifact. This correlation was formally retracted.
