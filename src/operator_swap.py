"""Reproduce the equal-operand comparison across operator strings.

The default operator strings follow the Unicode glyphs recorded in
``docs/RESEARCH_LOG.md``. ASCII ``*`` and ``-`` are available as a separate
tokenization-sensitivity condition. The older aggregate JSON is preserved;
this script writes a new detailed output so an unverified legacy artifact is
not overwritten silently.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformer_lens import HookedTransformer

from src.config import DOUBLES_CONFIG, NULL_TARGETS, OPERATOR_VARIANTS, POSITIVE_TARGETS
from src.metrics import doubles_advantage, doubles_score, pooled_control_sd


def format_prompt(a: int, operator: str, b: int) -> str:
    """Format a prompt without normalizing or substituting its operator glyph."""
    return f"{a} {operator} {b} ="


def _single_token_id(model: HookedTransformer, token: str) -> int:
    token_id = model.to_single_token(token)
    if not isinstance(token_id, int):
        token_id = int(token_id)
    return token_id


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def evaluate_operator_set(
    model: HookedTransformer,
    operators: dict[str, str],
) -> dict[str, Any]:
    """Evaluate all doubles and all ordered unequal single-digit controls."""
    results: dict[str, Any] = {}

    for label, operator in operators.items():
        per_target: dict[str, Any] = {}
        control_scores_by_target: dict[int, list[float]] = {}
        advantages: dict[int, float] = {}

        for target, config in DOUBLES_CONFIG.items():
            target_text = f" {target}"
            foil_texts = (f" {target - 1}", f" {target + 1}")
            target_id = _single_token_id(model, target_text)
            foil_ids = tuple(_single_token_id(model, text) for text in foil_texts)

            double_a, double_b = config["double"]
            double_prompt = format_prompt(double_a, operator, double_b)
            with torch.no_grad():
                double_logits = model(double_prompt)
            double_score = doubles_score(double_logits, target_id, foil_ids, pos=-1)

            control_records = []
            control_scores = []
            for control_a, control_b in config["controls"]:
                prompt = format_prompt(control_a, operator, control_b)
                with torch.no_grad():
                    logits = model(prompt)
                score = doubles_score(logits, target_id, foil_ids, pos=-1)
                control_scores.append(score)
                control_records.append({"prompt": prompt, "score": score})

            advantage = doubles_advantage(double_score, control_scores)
            control_scores_by_target[target] = control_scores
            advantages[target] = advantage
            per_target[str(target)] = {
                "target_token": target_text,
                "foil_tokens": list(foil_texts),
                "double_prompt": double_prompt,
                "double_score": double_score,
                "controls": control_records,
                "control_mean": float(np.mean(control_scores)),
                "advantage": advantage,
            }

        positive_advantages = [advantages[target] for target in POSITIVE_TARGETS]
        positive_sd = pooled_control_sd(
            {target: control_scores_by_target[target] for target in POSITIVE_TARGETS}
        )
        null_advantages = [advantages[target] for target in NULL_TARGETS]
        null_sd = pooled_control_sd(
            {target: control_scores_by_target[target] for target in NULL_TARGETS}
        )

        results[label] = {
            "operator_string": operator,
            "positive_targets": list(POSITIVE_TARGETS),
            "positive_advantage": float(np.mean(positive_advantages)),
            "positive_control_sd": positive_sd,
            "positive_control_df": sum(
                len(control_scores_by_target[t]) - 1 for t in POSITIVE_TARGETS
            ),
            "positive_adv_over_sd": float(np.mean(positive_advantages) / positive_sd)
            if positive_sd > 0
            else None,
            "null_targets": list(NULL_TARGETS),
            "null_advantage": float(np.mean(null_advantages)),
            "null_control_sd": null_sd,
            "null_control_df": sum(
                len(control_scores_by_target[t]) - 1 for t in NULL_TARGETS
            ),
            "null_adv_over_sd": float(np.mean(null_advantages) / null_sd)
            if null_sd > 0
            else None,
            "per_target": per_target,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the doubles comparison across operator strings"
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--operator-variant",
        default="reported_unicode",
        choices=("reported_unicode", "ascii_sensitivity", "both"),
        help="Run the documented Unicode glyphs, ASCII sensitivity controls, or both.",
    )
    parser.add_argument(
        "--output",
        default="data/operator_swap_reproduction.json",
        help="Write a new detailed result file; legacy aggregates are not overwritten.",
    )
    args = parser.parse_args()

    variants = (
        list(OPERATOR_VARIANTS)
        if args.operator_variant == "both"
        else [args.operator_variant]
    )
    model = HookedTransformer.from_pretrained("gpt2-small", device=args.device)
    outputs = {
        variant: evaluate_operator_set(model, OPERATOR_VARIANTS[variant])
        for variant in variants
    }

    report = {
        "metadata": {
            "experiment": "Equal-operand advantage under operator-string substitutions",
            "model": "gpt2-small",
            "target_position": "final '=' token position",
            "score": "logit(T) - mean(logit(T-1), logit(T+1))",
            "controls": "all ordered unequal single-digit splits for each target",
            "positive_control_df": 18,
            "null_control_df": 6,
            "operator_variants": variants,
            "torch_version": _package_version("torch"),
            "transformer_lens_version": _package_version("transformer-lens"),
            "status": "fresh run output; compare with legacy aggregate before updating claims",
        },
        "results": outputs,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved detailed operator-swap output to {output_path}")


if __name__ == "__main__":
    main()
