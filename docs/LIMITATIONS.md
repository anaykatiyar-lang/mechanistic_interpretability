# Methodological Limitations and Boundary Conditions

This document catalogs the exact scope limitations, tokenization artifacts, statistical noise bounds, and unresolved questions that constrain the conclusions of this project.

---

## 1. Scope and Architectural Constraints

1. **Single Model Architecture**:
   All empirical findings are derived from GPT-2 Small (124M parameters, 12 layers). No inferences are made regarding larger model scales (GPT-2 Medium/XL, Llama, Gemma) or models trained with modern code/math synthesis data.
2. **Restricted Arithmetic Domain**:
   Most experiments focus on single-digit operands and relatively small sums. This domain is substantially narrower than general arithmetic. The results therefore characterize GPT-2 Small's behavior on the tested arithmetic distribution rather than establishing a general theory of arithmetic computation in the model.
3. **Single-Prompt Attribution Caveats**:
   Several initial mechanistic insights (e.g., L11H0/MLP10 additivity, static-bias override) were derived on isolated prompts (`"3 + 5 ="`). While algebraically exact on those prompts, they do not constitute universal circuit motifs across the model.

---

## 2. Token Geometry and Representation Quirks

1. **Leading Whitespace Sensitivity**:
   In GPT-2's byte-pair encoding (BPE), leading whitespace is part of the tokenized string. The historical DiD v1 format contrast used unspaced targets in at least one comparison and cannot be isolated from its other design changes. The clean-rerun equal-operand token IDs are now archived in `data/current/equal_operand_tokenization_audit.csv`; this audit does not retroactively repair the v1 result.
2. **Single vs Multi-Token Representations**:
   While single-digit and early two-digit answers (`' 10'`, `' 12'`, `' 14'`, `' 16'`) are single BPE tokens in GPT-2, external tokenizers (such as Llama used in Dolma indexing) segment multi-digit numbers into individual digits. This creates token geometry mismatches during external corpus frequency matching.
3. **Round-Number and Static Bias Skew ($b_U$)**:
   - `' 10'` exhibits an abnormally high static unembedding bias ($b_U = +3.7600$) and is generically favored by $+1.0649$ logits on non-arithmetic filler prompts.
   - Target `' 14'` exhibits an abnormally low baseline ($-0.2311$), explaining why $7 + 7 = 14$ appears as an anomaly in raw metrics. These are properties of vocabulary embedding geometry, not arithmetic computation failures.

---

## 3. Statistical Noise-Floor Bounds and the Winner's Curse

1. **Dataset-Wide DLA Winner's Curse**:
   In uncorrected dataset-wide DLA sweeps, head L9H1 appeared prominent ($+0.1199$). After proper LayerNorm scaling, its mean contribution shrank to $+0.0064$ ($\text{SD} = 0.0279$), yielding a descriptive test statistic of $t \approx 2.1$. For the maximum of 144 independent standard normal draws, an extreme value of $t \approx 2.6$ is expected purely by chance. Thus, L9H1 cannot be designated a causal arithmetic driver.
2. **79-Cell Grid Non-Independence**:
   The committed grid has 79 cells and repeated target sums (for example, target 10 has five operand pairs). Parity predicts sign in 68/79 cells (86.1%) overall, 63/64 for sums ≤13, and 5/15 for sums ≥14. The cells are not independent; the lower-sum agreement should not be generalized across all sums. The earlier ~87% summary used a different row count and is retained only as history.
3. **Dense Scan Sample Limitations**:
   Because each target sum has only one true double ($d+d$), target 4 ($2+2$) and target 16 ($8+8$) have only one unique control pair (evaluated in two operand orders). These estimates therefore have greater sampling uncertainty and should not be interpreted as equally precise across all target values.
4. **Equal-Operand Inference Sensitivity**:
   The primary mean is +0.2474 with exact sign-flip $p=0.03125$, but six of seven targets are positive (two-sided sign test $p=0.125$), leave-one-target-out sign-flip values are $0.03125$ or $0.0625$, and excluding unequal controls containing a target-neighbor operand leaves six targets with mean +0.2561 and $p=0.125$. The result is sensitive to the target/control definition and should be treated as exploratory.
5. **Candidate-Screen Resolution**:
   The causal screen uses five discovery targets and adjusts over 12 nominated components. The minimum attainable one-sided exact sign-flip $p$ is $1/32=0.03125$ before family correction. Thus, failure to pass the FDR threshold does not distinguish no component effect from limited power.

---

## 4. Unanchored Metrics and Experimental Blindspots

1. **Unanchored Operator-Position Patching (A19)**:
   In the separate operator-position patching experiment, the metric evaluated was:
   $$\Delta_{\text{patch}} = \text{diff}_{\text{patched}} - \text{diff}_{\text{clean}}$$
   Because the corrupted prompt's baseline logit difference was not included as an anchor, a delta near zero cannot distinguish between "full recovery of corrupt state" and "no corruption effect". Operator binding therefore remains untested by this metric. This limitation is separate from the operator-swap comparison.
2. **Invalid L9H9 Path Patching (A12)**:
   Early attempts to demonstrate indirect routing from L9H9 into MLP10 and L10H2 failed to report delta norms or confirm hook execution on CPU runtimes. The hypothesis that L9H9 acts as an information router remains unverified.
3. **Easy-Tier Attention Blocking Mechanism**:
   Position-1 attention blocking produced a statistically significant reduction in target logit diff on Easy-tier prompts ($p = 0.0116$), but Medium-tier prompts showed no effect ($p = 0.724$). The exact internal mechanism mediating this difference remains unidentified.

---

## 5. Corpus Frequency Audit Boundaries

The external corpus-frequency analysis remains inconclusive. The seven-target count comparison and the mismatch between the available corpus and GPT-2's exact training distribution prevent strong conclusions about memorization. The results therefore should not be used either to establish or to rule out a memorization-based explanation.

1. **Corpus Proxy Discrepancy**:
   Dolma v1.7 is used as a proxy for WebText. Because WebText is proprietary and unreleased, true pretraining co-occurrence statistics cannot be measured directly. The corpus-size figure is omitted until supported by a verified source.
2. **Run distinction and analysis scope**:
   The supplied clean-rerun CSV/JSONL logs contain 79 attempts with zero recorded failures; the seven-target no-floor summary is archived. The archived raw collection applies no eligibility floor and reports no association test, so the relationship between corpus counts and model effects remains unresolved; see [CORPUS_AUDIT.md](CORPUS_AUDIT.md) and audit A30 and A32.
3. **Run 1 Artifact Retraction**:
   An initial analysis reported a correlation ($\rho = +0.82$, $p = 0.023$) between corpus frequency and model advantage. The statistic was retracted after the audit identified a query-format artifact. It must not be used as a finding.

4. **Earlier Exploratory Summary**:
   A previous summary reported joint-count Spearman $\rho=+0.46$, $p=.294$, and conditional-count $\rho=-0.14$, $p=.760$ ($n=7$). These values remain historical and are not results of the archived raw-query log. Raw responses are now available, but an eligibility rule and association statistic were not prespecified or applied; the frequency relationship therefore remains unresolved.

5. **Multi-Token Representation Limitation**: Some external arithmetic examples produce answers that do not correspond cleanly to a single GPT-2 token. Analyses that rely on `model.to_single_token` are restricted to examples whose target and foil representations satisfy the required single-token condition. Results from this restricted subset should not be generalized to arbitrary multi-token arithmetic answers.

6. **Parity Confound**: Digit-level output comparisons can inherit parity-related token biases. Because neighboring foil tokens can differ in parity from the target, part of the measured target-vs-neighbor preference can reflect static or contextual parity effects rather than arithmetic computation. Parity-preserving controls and filler prompts are therefore necessary when interpreting these measurements.

7. **Round-Number / Target-Token Bias**: Target-specific baseline effects are substantial for some numbers, particularly round-number tokens such as `10`. Target-specific baseline controls are therefore treated as necessary for interpreting mechanistic measurements.


---

## 6. Literature and Measurement Boundaries

The cited mechanistic studies use different tasks and, in several cases, different training regimes. They motivate questions about task definition, controls, and causal validation; they do not independently corroborate these measurements. Trained modular-addition transformers and GPT-2 Small greater-than results are not direct evidence about pretrained GPT-2 Small decimal addition. See [LITERATURE_MAP.md](LITERATURE_MAP.md) for source-specific scope.

The supplied clean-rerun exports include the 39-row digit+digit score file, target-level format and operator tables, intervention results, and individual corpus API responses. Format/operator exports remain target-level summaries rather than all underlying prompt scores. The single-operand comparison remains untested. The M1 cohorts are distinct: all-pairs zero-shot is 2/36 (5.56%); held-out zero-shot is 2/32 (6.25%) with an 8/32 (25%) best-constant baseline; held-out few-shot is 2/32; the earlier few-shot 5/33 uses another prefix and exclusion set. See [Audit_Trail.md](Audit_Trail.md) for the detailed correction record.

## 7. Unresolved Reproduction and Design Checks

- Operator comparisons now have seven matched target-level means per condition and paired target-level inference; prompt-level operator scores are not archived.
- The single-operand comparison (`4 + 3` and `4 + 5`, scoring token `' 8'`) has no run output.
- Format comparisons now have target-level means and paired target-level tests. The prompt-level formatted controls needed to recompute pooled SDs are still not included; the current format files are target-level summaries.
- The output bundle does not include standalone exports for every intermediate candidate-selection, filler-control, and operator-transfer table listed in the notebook manifest.
- In the multiplication controls, `1 × 7 = 7` at target 8 and `1 × 9 = 9` at target 10 match the lower-neighbor foil (`T−1`). This may mechanically lower the control mean and requires explicit assessment.
- The current comparisons do not separate equality, lexical equivalence, and string familiarity. There is one double prompt per target.
