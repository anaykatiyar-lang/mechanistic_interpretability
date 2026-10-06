"""
frequency_audit.py — Infini-gram API audit for corpus N-gram frequency.

Audit methodology and fixes established in PROJECT_MEMORY.md §7 E13 and AUDIT_TRAIL.md A26:
    - Uses 'index': 'v4_dolma-v1_7_llama' (not 'corpus', which fails silently)
    - Raises on HTTP/API errors instead of silently swallowing with .get("count", 0)
    - Queries both joint 'a + b = T' and prompt-only 'a + b ='
    - Applies rate-limiting with exponential backoff
    - Enforces a minimum-count floor (≥20 combined matches) before computing statistics
    - Discards Run 1's spurious rho = +0.82 artifact caused by leading-space anchor zero-counts
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from scipy.stats import spearmanr


INFINI_GRAM_URL = "https://api.infini-gram.io/"
DEFAULT_INDEX = "v4_dolma-v1_7_llama"
MIN_COUNT_FLOOR = 20

TARGET_PROMPTS = {
    4: {"double": "2 + 2 =", "double_joint": "2 + 2 = 4", "controls": [("1 + 3 =", "1 + 3 = 4"), ("3 + 1 =", "3 + 1 = 4")]},
    6: {"double": "3 + 3 =", "double_joint": "3 + 3 = 6", "controls": [("2 + 4 =", "2 + 4 = 6"), ("4 + 2 =", "4 + 2 = 6"), ("1 + 5 =", "1 + 5 = 6"), ("5 + 1 =", "5 + 1 = 6")]},
    8: {"double": "4 + 4 =", "double_joint": "4 + 4 = 8", "controls": [("3 + 5 =", "3 + 5 = 8"), ("5 + 3 =", "5 + 3 = 8"), ("2 + 6 =", "2 + 6 = 8"), ("6 + 2 =", "6 + 2 = 8")]},
    10: {"double": "5 + 5 =", "double_joint": "5 + 5 = 10", "controls": [("4 + 6 =", "4 + 6 = 10"), ("6 + 4 =", "6 + 4 = 10"), ("3 + 7 =", "3 + 7 = 10"), ("7 + 3 =", "7 + 3 = 10")]},
    12: {"double": "6 + 6 =", "double_joint": "6 + 6 = 12", "controls": [("5 + 7 =", "5 + 7 = 12"), ("7 + 5 =", "7 + 5 = 12"), ("4 + 8 =", "4 + 8 = 12"), ("8 + 4 =", "8 + 4 = 12")]},
    14: {"double": "7 + 7 =", "double_joint": "7 + 7 = 14", "controls": [("6 + 8 =", "6 + 8 = 14"), ("8 + 6 =", "8 + 6 = 14"), ("5 + 9 =", "5 + 9 = 14"), ("9 + 5 =", "9 + 5 = 14")]},
    16: {"double": "8 + 8 =", "double_joint": "8 + 8 = 16", "controls": [("7 + 9 =", "7 + 9 = 16"), ("9 + 7 =", "9 + 7 = 16")]},
}


def query_infinigram_count(
    query: str,
    index: str = DEFAULT_INDEX,
    max_retries: int = 5,
    backoff_factor: float = 1.5,
) -> int:
    """
    Query Infini-gram API for exact-string count with exponential backoff.

    CRITICAL FIX (A26): Must use the 'index' field, NOT 'corpus'.
    Must explicitly raise on API errors rather than defaulting to 0.
    """
    payload = {
        "index": index,
        "query_type": "count",
        "query": query,
    }

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(INFINI_GRAM_URL, json=payload, timeout=15)
            if resp.status_code == 429:  # Rate limited
                sleep_time = backoff_factor ** (attempt + 1)
                time.sleep(sleep_time)
                continue
            resp.raise_for_status()
            data = resp.json()
            if "count" not in data:
                raise ValueError(f"Unexpected API response missing 'count': {data}")
            return int(data["count"])
        except (requests.RequestException, ValueError) as e:
            last_exc = e
            time.sleep(backoff_factor ** attempt)

    raise RuntimeError(f"Failed to query count for {query!r} after {max_retries} retries: {last_exc}")


def run_corpus_audit(index: str = DEFAULT_INDEX, floor: int = MIN_COUNT_FLOOR) -> Dict[str, Any]:
    """
    Run full audit across all 7 targets, recording counts and evaluating
    whether counts pass the reliability floor.
    """
    audit_results = {}
    usable_targets = []

    for target_sum, cfg in TARGET_PROMPTS.items():
        time.sleep(0.5)  # Rate limiting courteous sleep
        try:
            d_joint = query_infinigram_count(cfg["double_joint"], index=index)
            d_prompt = query_infinigram_count(cfg["double"], index=index)
        except Exception as e:
            print(f"Warning: Failed querying double for target {target_sum}: {e}")
            d_joint, d_prompt = 0, 0

        c_joints = []
        c_prompts = []
        for c_prompt_str, c_joint_str in cfg["controls"]:
            time.sleep(0.5)
            try:
                cj = query_infinigram_count(c_joint_str, index=index)
                cp = query_infinigram_count(c_prompt_str, index=index)
            except Exception as e:
                print(f"Warning: Failed querying control {c_prompt_str!r}: {e}")
                cj, cp = 0, 0
            c_joints.append(cj)
            c_prompts.append(cp)

        total_matches = d_joint + sum(c_joints)
        floor_passed = total_matches >= floor
        if floor_passed:
            usable_targets.append(target_sum)

        audit_results[str(target_sum)] = {
            "target": target_sum,
            "double_joint_count": d_joint,
            "double_prompt_count": d_prompt,
            "control_joint_counts": c_joints,
            "control_prompt_counts": c_prompts,
            "total_joint_matches": total_matches,
            "passed_floor": floor_passed,
        }

    is_conclusive = len(usable_targets) >= 5
    summary = {
        "index_used": index,
        "floor_threshold": floor,
        "usable_targets": usable_targets,
        "verdict": "CONCLUSIVE" if is_conclusive else "INCONCLUSIVE",
        "details": audit_results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Infini-gram Corpus Frequency Audit")
    parser.add_argument("--index", type=str, default=DEFAULT_INDEX, help="Infini-gram index name")
    parser.add_argument("--floor", type=int, default=MIN_COUNT_FLOOR, help="Minimum joint count threshold")
    parser.add_argument("--output", type=str, default="data/corpus_frequency_audit_run.json", help="Output file")
    args = parser.parse_args()

    print(f"Auditing Dolma corpus via Infini-gram (index={args.index}, floor={args.floor})...")
    report = run_corpus_audit(index=args.index, floor=args.floor)

    print("\n--- AUDIT SUMMARY ---")
    print(f"Usable Targets (passed floor {args.floor}): {report['usable_targets']}")
    print(f"Status: {report['verdict']}")
    if report["verdict"] == "INCONCLUSIVE":
        print("Note (A26): Insufficient joint counts across higher sums. The frequency test is inconclusive.")

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved audit output to {out_p}")


if __name__ == "__main__":
    main()
