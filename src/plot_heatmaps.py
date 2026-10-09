"""Build the committed SVG figures directly from the data artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


def generate_addition_grid_svg(csv_path: str, output_path: str) -> None:
    """Render the 1–9 addition grid and report parity agreement by sum range."""
    grid = [[None for _ in range(9)] for _ in range(9)]
    rows: list[dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            a, b = int(row["a"]), int(row["b"])
            grid[a - 1][b - 1] = float(row["symmetric_logit_diff"])

    def agreement(selected: list[dict[str, str]]) -> tuple[int, int]:
        correct = sum(
            (float(row["symmetric_logit_diff"]) > 0) == (row["parity"] == "even")
            for row in selected
        )
        return correct, len(selected)

    overall = agreement(rows)
    lower = agreement([row for row in rows if int(row["target_sum"]) <= 13])
    upper = agreement([row for row in rows if int(row["target_sum"]) >= 14])

    # Leave a clear gap between the two-line summary and the x-axis title.
    cell_size, padding_x, padding_y = 56, 90, 120
    width, height = padding_x + 9 * cell_size + 60, padding_y + 9 * cell_size + 80
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background:#fff;font-family:Arial,sans-serif">',
        '<style>.title{font-size:16px;font-weight:bold;fill:#111827}.subtitle{font-size:11px;fill:#4b5563}.axis{font-size:13px;font-weight:600;fill:#374151}.tick{font-size:12px;fill:#6b7280}.value{font-size:11px;font-weight:600;text-anchor:middle;dominant-baseline:central}</style>',
        f'<text x="{width/2}" y="28" text-anchor="middle" class="title">GPT-2 Small Addition Grid ({len(rows)} cells): Symmetric Logit Difference</text>',
        f'<text x="{width/2}" y="48" text-anchor="middle" class="subtitle">Parity sign agreement: {overall[0]}/{overall[1]} overall; {lower[0]}/{lower[1]} for sums ≤13; {upper[0]}/{upper[1]} for sums ≥14.</text>',
        f'<text x="{width/2}" y="66" text-anchor="middle" class="subtitle">Repeated target sums mean cells are not independent; parity alignment is weaker and changes at higher sums.</text>',
        f'<text x="{padding_x + 4.5 * cell_size}" y="{padding_y - 30}" text-anchor="middle" class="axis">Second Operand (b)</text>',
        f'<text x="45" y="{padding_y + 4.5 * cell_size}" text-anchor="middle" transform="rotate(-90 45 {padding_y + 4.5 * cell_size})" class="axis">First Operand (a)</text>',
    ]
    for index in range(9):
        svg.append(f'<text x="{padding_x + index*cell_size + cell_size/2}" y="{padding_y-10}" text-anchor="middle" class="tick">{index+1}</text>')
        svg.append(f'<text x="{padding_x-15}" y="{padding_y + index*cell_size + cell_size/2}" text-anchor="end" class="tick">{index+1}</text>')

    for i in range(9):
        for j in range(9):
            value = grid[i][j]
            x, y = padding_x + j * cell_size, padding_y + i * cell_size
            if value is None:
                svg.append(f'<rect x="{x}" y="{y}" width="{cell_size-2}" height="{cell_size-2}" fill="#f3f4f6" rx="4"/>')
                svg.append(f'<text x="{x+cell_size/2}" y="{y+cell_size/2}" class="value" fill="#9ca3af">—</text>')
                continue
            norm = max(-1.0, min(1.0, value / 0.75))
            if norm >= 0:
                rgb = (int(255-norm*218), int(255-norm*156), int(255-norm*20))
                text_color = "#fff" if norm > 0.45 else "#1e3a8a"
            else:
                magnitude = abs(norm)
                rgb = (int(255-magnitude*35), int(255-magnitude*217), int(255-magnitude*217))
                text_color = "#fff" if magnitude > 0.45 else "#7f1d1d"
            fill = "#%02x%02x%02x" % rgb
            svg.append(f'<rect x="{x}" y="{y}" width="{cell_size-2}" height="{cell_size-2}" fill="{fill}" rx="4"/>')
            if i == j:
                svg.append(f'<rect x="{x}" y="{y}" width="{cell_size-2}" height="{cell_size-2}" stroke="#1e3a8a" stroke-width="2.5" fill="none" rx="4"/>')
            svg.append(f'<text x="{x+cell_size/2}" y="{y+cell_size/2}" class="value" fill="{text_color}">{value:+.3f}</text>')
    svg.append("</svg>")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(svg), encoding="utf-8")


def _operator_rows(data_path: str, variant: str) -> list[dict[str, Any]]:
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    results = data.get("results", {})
    if variant in results and isinstance(results[variant], dict):
        results = results[variant]
    rows = []
    for key, item in results.items():
        if not isinstance(item, dict) or "positive_adv_over_sd" not in item:
            continue
        adv_sd = item["positive_adv_over_sd"]
        raw_adv = item.get("positive_advantage")
        operator = item.get("operator_string", item.get("operator", key))
        if adv_sd is None or raw_adv is None:
            continue
        rows.append({"label": key, "operator": operator, "adv_sd": float(adv_sd), "raw_adv": float(raw_adv)})
    if not rows:
        raise ValueError(f"No compatible aggregate operator rows found in {data_path!r} for {variant!r}.")
    return rows


def generate_operator_swap_svg(data_path: str, output_path: str, variant: str) -> None:
    """Render operator summaries from their JSON source, not duplicated constants."""
    rows = _operator_rows(data_path, variant)
    raw_max = max(rows, key=lambda row: row["raw_adv"])
    normalized_max = max(rows, key=lambda row: row["adv_sd"])
    width, height = 720, 126 + 54 * len(rows)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background:#fff;font-family:Arial,sans-serif">',
        '<style>.title{font-size:16px;font-weight:bold;fill:#111827}.subtitle{font-size:11px;fill:#4b5563}.label{font-size:13px;fill:#1f2937}.bar{font-size:12px;font-weight:bold;fill:#fff}.value{font-size:11px;fill:#4b5563}</style>',
        f'<text x="{width/2}" y="28" text-anchor="middle" class="title">Reported Doubles Advantage Under Operator Substitutions</text>',
        f'<text x="{width/2}" y="48" text-anchor="middle" class="subtitle">Descriptive adv/SD summary; largest raw advantage: {html.escape(raw_max["label"])} ({raw_max["raw_adv"]:+.3f}).</text>',
        f'<text x="{width/2}" y="66" text-anchor="middle" class="subtitle">Largest adv/SD: {html.escape(normalized_max["label"])} ({normalized_max["adv_sd"]:.2f}).</text>',
        f'<text x="{width/2}" y="84" text-anchor="middle" class="subtitle">Legacy aggregate; not reproduced from prompt-level scores. adv/SD is not a significance test.</text>',
    ]
    start_y, bar_height, max_width = 100, 30, 340
    max_value = max(row["adv_sd"] for row in rows) or 1.0
    for index, row in enumerate(rows):
        y = start_y + index * 54
        label = f'{row["label"]} ({row["operator"]})'
        bar_width = row["adv_sd"] / max_value * max_width
        svg.append(f'<text x="115" y="{y+bar_height/2+4}" text-anchor="end" class="label">{html.escape(label)}</text>')
        svg.append(f'<rect x="130" y="{y}" width="{bar_width:.2f}" height="{bar_height}" fill="#426b9b" rx="5"/>')
        svg.append(f'<text x="{130+bar_width-8:.2f}" y="{y+bar_height/2+4}" text-anchor="end" class="bar">{row["adv_sd"]:.2f}</text>')
        svg.append(f'<text x="{140+bar_width:.2f}" y="{y+bar_height/2+4}" class="value">raw advantage {row["raw_adv"]:+.3f}</text>')
    svg.append("</svg>")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the repository SVG figures from committed artifacts")
    parser.add_argument("--grid-csv", default="data/current/addition_grid_79cell.csv")
    parser.add_argument("--operator-json", default="data/current/operator_swap_results.json")
    parser.add_argument("--operator-variant", default="reported_unicode")
    parser.add_argument("--grid-svg", default="figures/static/addition_grid_heatmap.svg")
    parser.add_argument("--operator-svg", default="figures/static/operator_swap_heatmap.svg")
    args = parser.parse_args()

    generate_addition_grid_svg(args.grid_csv, args.grid_svg)
    generate_operator_swap_svg(args.operator_json, args.operator_svg, args.operator_variant)

if __name__ == "__main__":
    main()
