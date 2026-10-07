"""
metrics.py — Logit-difference, advantage/SD, and pooled-variance functions.

These functions implement the metric definitions summarized in
``docs/METHODOLOGY.md``:
    - logit_diff = logit(target) − logit(foil)
    - symmetric logit diff = logit(target) − mean(logit(foils))
    - doubles score = logit(T) − mean(logit(T−1), logit(T+1))
    - doubles advantage = score(double) − mean(score(controls))
    - adv/SD = mean(advantage) / pooled_control_SD (a project-defined descriptive normalized effect measure)
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


def logit_diff(
    logits: torch.Tensor,
    target_id: int,
    foil_id: int,
    pos: int = -1,
) -> float:
    """
    Compute logit(target) − logit(foil) at a given sequence position.

    Args:
        logits: Model output logits, shape [batch, seq_len, vocab].
        target_id: Vocabulary index of the target token (must include leading space).
        foil_id: Vocabulary index of the foil token.
        pos: Sequence position to read (default: last token, i.e. the '=' position).

    Returns:
        Scalar logit difference (float).
    """
    return (logits[0, pos, target_id] - logits[0, pos, foil_id]).item()


def symmetric_logit_diff(
    logits: torch.Tensor,
    target_id: int,
    foil_ids: List[int],
    pos: int = -1,
) -> float:
    """
    Symmetric logit difference: target minus mean of foil logits.

    NOTE (A20, §4): When foils are target ±1, they have the opposite parity
    of the target, so this metric is directly sensitive to parity-linked bias.

    Args:
        logits: Shape [batch, seq_len, vocab].
        target_id: Vocabulary index of the target.
        foil_ids: List of foil vocabulary indices (typically target ±1).
        pos: Sequence position.

    Returns:
        Scalar symmetric logit difference.
    """
    target_logit = logits[0, pos, target_id].item()
    foil_mean = np.mean([logits[0, pos, f].item() for f in foil_ids])
    return float(target_logit - foil_mean)


def doubles_score(
    logits: torch.Tensor,
    target_id: int,
    neighbor_ids: Tuple[int, int],
    pos: int = -1,
) -> float:
    """
    Mean-of-neighbors score: logit(T) − mean(logit(T−1), logit(T+1)).

    This is the scoring function used in the dense-scan doubles experiment (E11).

    Args:
        logits: Shape [batch, seq_len, vocab].
        target_id: Token ID for the sum T.
        neighbor_ids: (token_id for T−1, token_id for T+1).
        pos: Sequence position.

    Returns:
        Scalar score.
    """
    t = logits[0, pos, target_id].item()
    n_mean = np.mean([logits[0, pos, n].item() for n in neighbor_ids])
    return float(t - n_mean)


def doubles_advantage(
    double_score: float,
    control_scores: List[float],
) -> float:
    """
    Advantage = score(double) − mean(score(controls)).

    Controls must be averaged over both operand orders per the dense-scan
    design (E11): unique controls are half the printed n.
    """
    return float(double_score - np.mean(control_scores))


def adv_over_sd(
    advantages: List[float],
    control_scores_by_target: Dict[int, List[float]],
) -> float:
    """
    Normalized advantage: mean(advantage) / pooled_control_SD.

    Pooled SD is computed from control scores centered by their per-target means.
    With the full ordered split controls on targets 8, 10, 12, and 16, this gives
    18 within-target degrees of freedom. With incomplete control lists the df is
    smaller; callers should record the actual control counts.

    Args:
        advantages: Per-target advantage values (positive targets only).
        control_scores_by_target: {target_sum: [score_1, score_2, ...]}

    Returns:
        Descriptive normalized advantage ratio. This is not a conventional Student's
        t-statistic, p-value, or standalone test of statistical significance.
    """
    mean_adv = float(np.mean(advantages))
    pooled_sd = pooled_control_sd(control_scores_by_target)
    if pooled_sd == 0:
        return float("inf")
    return float(mean_adv / pooled_sd)


def pooled_control_sd(
    control_scores_by_target: Dict[int, List[float]],
) -> float:
    """
    Pooled standard deviation of control scores around their per-target means.

    Uses Bessel correction within each group, then pools via the standard
    weighted-variance formula:
        s_pooled = sqrt( sum( (n_k - 1) * s_k^2 ) / sum( n_k - 1 ) )

    Args:
        control_scores_by_target: {target_sum: [score_1, score_2, ...]}

    Returns:
        Pooled SD (float).
    """
    ss_total = 0.0
    df_total = 0
    for scores in control_scores_by_target.values():
        arr = np.array(scores)
        if len(arr) < 2:
            continue
        ss_total += float(np.sum((arr - arr.mean()) ** 2))
        df_total += len(arr) - 1
    if df_total == 0:
        return 0.0
    return float(np.sqrt(ss_total / df_total))


def dla_component_projection(
    component_resid: torch.Tensor,
    W_U: torch.Tensor,
    target_id: int,
    foil_id: int,
) -> float:
    """
    Direct Logit Attribution for a single residual-stream component.

    Projects the component's (LayerNorm-scaled) output onto the unembedding
    difference direction W_U[:, target] − W_U[:, foil].

    IMPORTANT: The component must already be LayerNorm-scaled via
    `cache.apply_ln_to_stack(..., layer=-1, pos_slice=-1)` or division
    by `ln_final.hook_scale`. Raw (pre-LN) components overshoot by
    14x–40x (A01).

    Args:
        component_resid: Shape [d_model] — one component's contribution
                         at the final position, already LN-scaled.
        W_U: The unembedding matrix, shape [d_model, d_vocab].
        target_id: Vocabulary index of the target token.
        foil_id: Vocabulary index of the foil token.

    Returns:
        Scalar DLA contribution (float).
    """
    diff_direction = W_U[:, target_id] - W_U[:, foil_id]
    return float(torch.dot(component_resid.float(), diff_direction.float()).item())


def verify_dla_sum(
    component_dlas: List[float],
    b_U_target: float,
    b_U_foil: float,
    measured_logit_diff: float,
    atol: float = 1e-3,
) -> Tuple[bool, float]:
    """
    Verify the DLA sum identity:
        sum(component_DLAs) + (b_U[target] − b_U[foil]) ≈ logit_diff

    The residual must be < atol for the decomposition to be valid (E01).

    Args:
        component_dlas: List of DLA values for all 159 components
                        (144 heads + 12 MLPs + embed + pos_embed + bias).
        b_U_target: Unembedding bias for the target token.
        b_U_foil: Unembedding bias for the foil token.
        measured_logit_diff: The model's actual logit difference.
        atol: Absolute tolerance (default 1e-3).

    Returns:
        (passes: bool, residual: float)
    """
    reconstructed = sum(component_dlas) + (b_U_target - b_U_foil)
    residual = abs(reconstructed - measured_logit_diff)
    return residual < atol, float(residual)


def own_answer_contrast(
    logit_matrix: np.ndarray,
    answer_labels: np.ndarray,
    digits: Optional[List[int]] = None,
) -> float:
    """
    Row-centred own-answer logit contrast (E12).

    For each digit d, compute the mean logit of token ` d` on prompts whose
    answer is d minus its mean on other prompts, then average over digits.
    Row-centring removes per-prompt level effects.

    Args:
        logit_matrix: Shape [n_prompts, n_digits] — logits for each digit
                      token on each prompt, already row-centred by the caller.
        answer_labels: Shape [n_prompts] — true answer digit for each prompt.
        digits: List of digit values (default: 3..9 matching the 32-prompt cohort).

    Returns:
        Mean own-answer contrast (float). This helper does not itself center
        rows, construct a permutation null, or test neighbor specificity.
    """
    if digits is None:
        digits = list(range(3, 10))

    contrasts = []
    for i, d in enumerate(digits):
        mask = answer_labels == d
        if mask.sum() == 0:
            continue
        own = logit_matrix[mask, i].mean()
        other = logit_matrix[~mask, i].mean()
        contrasts.append(own - other)

    return float(np.mean(contrasts)) if contrasts else 0.0
