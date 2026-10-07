"""Query Infini-gram for exact prompt and prompt-answer counts.

The historical corpus artifact and the earlier analysis disagree about how the
minimum-count floor was applied. This script records all three candidate floor
bases instead of selecting one silently. It also preserves failed queries as
errors; an API failure is never converted into a count of zero.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

import requests

from src.config import DOUBLES_CONFIG, TARGETS


INFINI_GRAM_URL = "https://api.infini-gram.io/"
DEFAULT_INDEX = "v4_dolma-v1_7_llama"
MIN_COUNT_FLOOR = 20


def query_infinigram_count(
    query: str,
    index: str = DEFAULT_INDEX,
    max_retries: int = 5,
    backoff_factor: float = 1.5,
) -> int:
    """Return one exact-string count or raise after retries; never return zero on failure."""
    payload = {"index": index, "query_type": "count", "query": query}
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = requests.post(INFINI_GRAM_URL, json=payload, timeout=30)
            if response.status_code == 429:
                time.sleep(backoff_factor ** (attempt + 1))
                continue
            response.raise_for_status()
            body = response.json()
            if "count" not in body:
                raise ValueError(f"API response did not include 'count': {body!r}")
            return int(body["count"])
        except (requests.RequestException, ValueError, TypeError) as error:
            last_error = error
            if attempt + 1 < max_retries:
                time.sleep(backoff_factor**attempt)

    raise RuntimeError(
        f"Could not retrieve count for {query!r} after {max_retries} attempts: {last_error}"
    )


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_corpus_audit(
    index: str = DEFAULT_INDEX,
    floor: int = MIN_COUNT_FLOOR,
    pause_seconds: float = 0.5,
) -> dict[str, Any]:
    """Query all double/control strings and report alternative count-floor rules."""
    targets: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []

    for target in TARGETS:
        config = DOUBLES_CONFIG[target]
        double_a, double_b = config["double"]
        double_prompt = f"{double_a} + {double_b} ="
        double_joint = f"{double_prompt} {target}"

        def query_record(query: str, label: str) -> int | None:
            time.sleep(pause_seconds)
            try:
                return query_infinigram_count(query, index=index)
            except Exception as error:  # Keep the failure visible in the artifact.
                errors.append(
                    {
                        "target": target,
                        "query_role": label,
                        "query": query,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                return None

        double_prompt_count = query_record(double_prompt, "double_prompt")
        double_joint_count = query_record(double_joint, "double_joint")
        controls = []
        for control_a, control_b in config["controls"]:
            prompt = f"{control_a} + {control_b} ="
            joint = f"{prompt} {target}"
            controls.append(
                {
                    "prompt": prompt,
                    "joint_query": joint,
                    "prompt_count": query_record(prompt, "control_prompt"),
                    "joint_count": query_record(joint, "control_joint"),
                }
            )

        joint_counts = [record["joint_count"] for record in controls]
        complete_controls = all(count is not None for count in joint_counts)
        controls_mean = (
            sum(joint_counts) / len(joint_counts)
            if complete_controls and joint_counts
            else None
        )
        controls_sum = sum(joint_counts) if complete_controls else None

        bases = {
            "double_joint_only": double_joint_count,
            "double_plus_mean_control_joint": (
                double_joint_count + controls_mean
                if double_joint_count is not None and controls_mean is not None
                else None
            ),
            "double_plus_sum_control_joint": (
                double_joint_count + controls_sum
                if double_joint_count is not None and controls_sum is not None
                else None
            ),
        }
        targets[str(target)] = {
            "target": target,
            "double_prompt": double_prompt,
            "double_joint_query": double_joint,
            "double_prompt_count": double_prompt_count,
            "double_joint_count": double_joint_count,
            "controls": controls,
            "floor_bases": {
                name: {
                    "value": value,
                    "passes": value >= floor if value is not None else None,
                }
                for name, value in bases.items()
            },
        }

    all_complete = not errors
    usable_by_basis = {
        basis: [
            int(target)
            for target, row in targets.items()
            if row["floor_bases"][basis]["passes"] is True
        ]
        for basis in (
            "double_joint_only",
            "double_plus_mean_control_joint",
            "double_plus_sum_control_joint",
        )
    }
    return {
        "metadata": {
            "index": index,
            "endpoint": INFINI_GRAM_URL,
            "minimum_count_floor": floor,
            "floor_bases_reported": list(usable_by_basis),
            "floor_basis_status": "unresolved in historical audit; all variants shown",
            "requests_version": _package_version("requests"),
            "status": "complete" if all_complete else "FAILED: one or more queries failed",
        },
        "usable_targets_by_floor_basis": usable_by_basis if all_complete else None,
        "targets": targets,
        "errors": errors,
        "statistical_conclusion": "not computed; the historical floor definition and rank-test inputs are unresolved",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run the Infini-gram frequency count audit")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--floor", type=int, default=MIN_COUNT_FLOOR)
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    parser.add_argument(
        "--output",
        default="data/corpus_frequency_audit_reproduction.json",
        help="Write a new artifact; do not overwrite the legacy counts.",
    )
    args = parser.parse_args()

    report = run_corpus_audit(args.index, args.floor, args.pause_seconds)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Audit status: {report['metadata']['status']}")
    print(f"Saved detailed corpus audit to {output_path}")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
