"""Unembedding bias (b_U) and filler-prompt diagnostics.

The script recomputes token biases, vector norms, and the mean of the four
filler prompts listed below. The archived +1.0649 target-10 value refers to a
single "object in the box" prompt, not that four-prompt mean. The static-bias
override reconstruction uses a separately supplied archived DLA value; it is
not a fresh DLA calculation, and the originating prompt/run artifact is absent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformer_lens import HookedTransformer


DIGIT_TOKENS = [
    " 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9",
    " 10", " 11", " 12", " 13", " 14", " 15", " 16"
]

FILLER_PROMPTS = [
    "The object in the box =",
    "The word on the page =",
    "Yesterday at the store =",
    "The item sequence number ="
]


def extract_unembedding_properties(model: HookedTransformer) -> Dict[str, dict]:
    """
    Extract unembedding bias (b_U) and column L2 norms for digit tokens.
    """
    # model.b_U is shape [d_vocab]
    b_U = model.b_U.detach().cpu()
    # model.W_U is shape [d_model, d_vocab]
    W_U = model.W_U.detach().cpu()

    token_stats = {}
    for tok_str in DIGIT_TOKENS:
        tok_id = model.to_single_token(tok_str)
        bias_val = b_U[tok_id].item()
        norm_val = torch.norm(W_U[:, tok_id], p=2).item()
        token_stats[tok_str] = {
            "token_id": tok_id,
            "token_string": tok_str,
            "b_U": float(bias_val),
            "W_U_norm": float(norm_val),
        }

    return token_stats


def evaluate_filler_baselines(model: HookedTransformer) -> Dict[str, dict]:
    """
    Evaluate target-10 and target-14 preferences on four non-arithmetic fillers.
    """
    filler_results = {}
    t10_id = model.to_single_token(" 10")
    t9_id = model.to_single_token(" 9")
    t11_id = model.to_single_token(" 11")

    t14_id = model.to_single_token(" 14")
    t13_id = model.to_single_token(" 13")
    t15_id = model.to_single_token(" 15")

    for prompt in FILLER_PROMPTS:
        with torch.no_grad():
            logits = model(prompt)
        last_logits = logits[0, -1, :].cpu()

        score_10 = (last_logits[t10_id] - 0.5 * (last_logits[t9_id] + last_logits[t11_id])).item()
        score_14 = (last_logits[t14_id] - 0.5 * (last_logits[t13_id] + last_logits[t15_id])).item()

        filler_results[prompt] = {
            "prompt": prompt,
            "target_10_vs_neighbors": float(score_10),
            "target_14_vs_neighbors": float(score_14),
        }

    mean_10 = sum(r["target_10_vs_neighbors"] for r in filler_results.values()) / len(filler_results)
    mean_14 = sum(r["target_14_vs_neighbors"] for r in filler_results.values()) / len(filler_results)

    return {
        "prompts": filler_results,
        "mean_target_10_preference": float(mean_10),
        "mean_target_14_preference": float(mean_14),
    }


def analyze_static_bias_override(
    model: HookedTransformer,
    target_tok: str = " 8",
    foil_tok: str = " 1",
    circuit_dla_sum: float = 0.8382,
) -> dict:
    """
    Reconstruct the archived static-bias override reported for Run 2 (E01, A02):
    Circuit favors target by +0.8382, but static b_U diff is -1.0303,
    causing net logit difference to flip negative (-0.1921).
    """
    target_id = model.to_single_token(target_tok)
    foil_id = model.to_single_token(foil_tok)
    b_U = model.b_U.detach().cpu()

    delta_b_U = (b_U[target_id] - b_U[foil_id]).item()
    net_logit_diff = circuit_dla_sum + delta_b_U

    return {
        "target": target_tok,
        "foil": foil_tok,
        "circuit_dla_sum": float(circuit_dla_sum),
        "circuit_dla_source": "archived reported value; no source run output is committed",
        "delta_b_U": float(delta_b_U),
        "net_logit_diff": float(net_logit_diff),
        "override_occurred": (circuit_dla_sum > 0 and net_logit_diff < 0),
    }


def main():
    parser = argparse.ArgumentParser(description="GPT-2 Small Token & Unembedding Diagnostics")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="data/token_diagnostics_report.json")
    args = parser.parse_args()

    print(f"Loading HookedTransformer 'gpt2-small' on {args.device}...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=args.device)

    print("Computing unembedding biases (b_U) and norms...")
    tok_stats = extract_unembedding_properties(model)

    print("Evaluating non-arithmetic filler controls...")
    filler_stats = evaluate_filler_baselines(model)

    print("Evaluating static-bias override...")
    override_stats = analyze_static_bias_override(model)

    report = {
        "token_statistics": tok_stats,
        "filler_baselines": filler_stats,
        "static_bias_override_case": override_stats,
    }

    print("\n--- TOKEN DIAGNOSTICS SUMMARY ---")
    print(f"' 10' b_U: {tok_stats[' 10']['b_U']:+.4f} | ' 8' b_U: {tok_stats[' 8']['b_U']:+.4f} | ' 14' b_U: {tok_stats[' 14']['b_U']:+.4f}")
    print(f"Mean Target 10 Filler Preference: {filler_stats['mean_target_10_preference']:+.4f}")
    print(f"Mean Target 14 Filler Preference: {filler_stats['mean_target_14_preference']:+.4f}")
    print(f"Run 2 Static Override: Circuit={override_stats['circuit_dla_sum']:+.4f}, Delta_b_U={override_stats['delta_b_U']:+.4f} -> Net={override_stats['net_logit_diff']:+.4f}")

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved diagnostics to {out_p}")


if __name__ == "__main__":
    main()
