# src/__init__.py
"""
GPT-2 Small Doubling Anomaly — Falsification Toolkit.

Modules:
    metrics          – adv/SD, pooled variance, and logit-difference functions
    operator_swap    – TransformerLens script for operator-blindness checks
    frequency_audit  – Infini-gram API audit with rate-limiting & error handling
    token_diagnostics – Unembedding bias (b_U) & BPE vector norm diagnostic
"""

from .metrics import (
    logit_diff,
    symmetric_logit_diff,
    doubles_score,
    doubles_advantage,
    adv_over_sd,
    pooled_control_sd,
    dla_component_projection,
    verify_dla_sum,
    own_answer_contrast,
)

__version__ = "0.1.0"
__all__ = [
    "logit_diff",
    "symmetric_logit_diff",
    "doubles_score",
    "doubles_advantage",
    "adv_over_sd",
    "pooled_control_sd",
    "dla_component_projection",
    "verify_dla_sum",
    "own_answer_contrast",
]
