"""
operator_swap.py — TransformerLens test for operator blindness in GPT-2 Small.

Tests whether the "doubles advantage" (higher score for the doubled sum token on
a ? a than on matched split controls) is specific to addition or represents
an operator-blind associative phenomenon.

Prompts evaluated across 5 operators / connectors:
    "+", "-", "*", "and", "then"
for targets 4, 6, 8, 10, 12, 14, 16.

Empirical finding (PROJECT_MEMORY.md §7 E11, R13):
    The advantage persists across the tested operators and is strongest under minus
    (adv/SD = 6.15 vs 5.22 for plus). This is evidence against a simple addition-specific explanation,
    not a universal proof of operator blindness.
"""

Interpretation note: the reported adv/SD values are descriptive normalized effect
measures, not conventional t-statistics or p-values.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformer_lens import HookedTransformer

from src.metrics import doubles_score, doubles_advantage, pooled_control_sd


# Target configurations and matched splits
DOUBLES_CONFIG = {
    4: {"double": (2, 2), "controls": [(1, 3), (3, 1)]},
    6: {"double": (3, 3), "controls": [(2, 4), (4, 2), (1, 5), (5, 1)]},
    8: {"double": (4, 4), "controls": [(3, 5), (5, 3), (2, 6), (6, 2)]},
    10: {"double": (5, 5), "controls": [(4, 6), (6, 4), (3, 7), (7, 3)]},
    12: {"double": (6, 6), "controls": [(5, 7), (7, 5), (4, 8), (8, 4)]},
    14: {"double": (7, 7), "controls": [(6, 8), (8, 6), (5, 9), (9, 5)]},
    16: {"double": (8, 8), "controls": [(7, 9), (9, 7)]},
}

POSITIVE_TARGETS = [8, 10, 12, 16]
NULL_TARGETS = [6, 14]
OPERATORS = ["+", "-", "*", "and", "then"]


def format_prompt(a: int, op: str, b: int) -> str:
    """Format a 5-token prompt: e.g., '4 + 4 =', '4 - 4 =', '4 and 4 ='."""
    return f"{a} {op} {b} ="


def evaluate_operator_swaps(
    model: HookedTransformer,
    operators: List[str] = OPERATORS,
) -> Dict[str, dict]:
    """
    Evaluate the doubles advantage across operators and connectors.
    Always measures the logit difference for the sum token (T) relative
    to its neighbors (T-1, T+1) at the final '=' token position.
    """
    results_by_op = {}

    for op in operators:
        advantages = {}
        control_scores_by_target = {}

        for target_sum, config in DOUBLES_CONFIG.items():
            # Get token IDs with leading space (CRITICAL: A24)
            target_str = f" {target_sum}"
            target_id = model.to_single_token(target_str)
            foil_prev_id = model.to_single_token(f" {target_sum - 1}")
            foil_next_id = model.to_single_token(f" {target_sum + 1}")
            neighbors = (foil_prev_id, foil_next_id)

            # Evaluate double prompt
            d_a, d_b = config["double"]
            double_prompt = format_prompt(d_a, op, d_b)
            with torch.no_grad():
                d_logits = model(double_prompt)
            d_score = doubles_score(d_logits, target_id, neighbors, pos=-1)

            # Evaluate control prompts
            c_scores = []
            for c_a, c_b in config["controls"]:
                ctl_prompt = format_prompt(c_a, op, c_b)
                with torch.no_grad():
                    c_logits = model(ctl_prompt)
                c_score = doubles_score(c_logits, target_id, neighbors, pos=-1)
                c_scores.append(c_score)

            control_scores_by_target[target_sum] = c_scores
            advantages[target_sum] = doubles_advantage(d_score, c_scores)

        # Compute pooled variance and adv/SD for positive targets
        pos_advs = [advantages[t] for t in POSITIVE_TARGETS]
        pos_controls = {t: control_scores_by_target[t] for t in POSITIVE_TARGETS}
        pos_sd = pooled_control_sd(pos_controls)
        pos_adv_sd = float(np.mean(pos_advs) / pos_sd) if pos_sd > 0 else 0.0

        # Compute for null targets
        null_advs = [advantages[t] for t in NULL_TARGETS]
        null_controls = {t: control_scores_by_target[t] for t in NULL_TARGETS}
        null_sd = pooled_control_sd(null_controls)
        null_adv_sd = float(np.mean(null_advs) / null_sd) if null_sd > 0 else 0.0

        results_by_op[op] = {
            "operator": op,
            "positive_advantage": float(np.mean(pos_advs)),
            "positive_control_sd": float(pos_sd),
            "positive_adv_over_sd": float(pos_adv_sd),
            "null_advantage": float(np.mean(null_advs)),
            "null_control_sd": float(null_sd),
            "null_adv_over_sd": float(null_adv_sd),
            "per_target_advantages": {str(t): float(advantages[t]) for t in DOUBLES_CONFIG},
        }

    return results_by_op


def main():
    parser = argparse.ArgumentParser(description="Operator Swap Experiment in GPT-2 Small")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    parser.add_argument("--output", type=str, default="data/operator_swap_evaluated.json", help="Output path")
    args = parser.parse_args()

    print(f"Loading HookedTransformer 'gpt2-small' on {args.device}...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=args.device)

    print("Running operator swap battery...")
    results = evaluate_operator_swaps(model)

    print("\n--- OPERATOR SWAP SUMMARY ---")
    for op, data in results.items():
        print(f"Operator: {op:<5} | Pos Adv: {data['positive_advantage']:+.3f} | SD: {data['positive_control_sd']:.3f} | adv/SD: {data['positive_adv_over_sd']:+.2f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
