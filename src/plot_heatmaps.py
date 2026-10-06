"""
plot_heatmaps.py — Generate publication-grade heatmaps for the addition grid
and operator swap falsification experiments.

Generates:
    1. figures/addition_grid_parity_heatmap.png / .svg
       (79-cell a x b grid showing parity bias and the doubling diagonal)
    2. figures/operator_swap_heatmap.png / .svg
       (Operator-blindness comparison across +, -, *, and, then)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_addition_grid_svg(csv_path: str, output_path: str):
    """
    Generate a standalone, high-precision SVG heatmap for the 79-cell addition grid.
    Colors cells using a diverging palette:
        Red/Warm (< 0, typically odd sums)
        Blue/Cool (> 0, typically even sums)
    """
    df = pd.read_csv(csv_path)
    grid = np.full((9, 9), np.nan)
    for _, row in df.iterrows():
        a = int(row["a"]) - 1
        b = int(row["b"]) - 1
        grid[a, b] = float(row["symmetric_logit_diff"])

    cell_size = 56
    padding_x = 90
    padding_y = 90
    width = padding_x + 9 * cell_size + 60
    height = padding_y + 9 * cell_size + 80

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background-color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif;">',
        '<style>',
        '  .title { font-size: 16px; font-weight: bold; fill: #111827; }',
        '  .subtitle { font-size: 12px; fill: #4b5563; }',
        '  .axis-label { font-size: 13px; font-weight: 600; fill: #374151; }',
        '  .tick-label { font-size: 12px; font-weight: 500; fill: #6b7280; }',
        '  .cell-text { font-size: 11px; font-weight: 600; text-anchor: middle; dominant-baseline: central; }',
        '  .diag-border { stroke: #1e3a8a; stroke-width: 2.5px; fill: none; }',
        '</style>',
        f'<text x="{width/2}" y="32" text-anchor="middle" class="title">GPT-2 Small Addition Grid (79 Cells): Symmetric Logit Difference</text>',
        f'<text x="{width/2}" y="52" text-anchor="middle" class="subtitle">Checkerboard pattern reveals parity bias (Blue = Even/Positive, Red = Odd/Negative). Bold outline = Doubles diagonal.</text>',
    ]

    # Column headers (Operand b)
    svg_parts.append(f'<text x="{padding_x + 4.5 * cell_size}" y="{padding_y - 30}" text-anchor="middle" class="axis-label">Second Operand (b)</text>')
    for j in range(9):
        x = padding_x + j * cell_size + cell_size / 2
        svg_parts.append(f'<text x="{x}" y="{padding_y - 10}" text-anchor="middle" class="tick-label">{j + 1}</text>')

    # Row headers (Operand a)
    svg_parts.append(f'<text x="25" y="{padding_y + 4.5 * cell_size}" text-anchor="middle" transform="rotate(-90 25 {padding_y + 4.5 * cell_size})" class="axis-label">First Operand (a)</text>')
    for i in range(9):
        y = padding_y + i * cell_size + cell_size / 2
        svg_parts.append(f'<text x="{padding_x - 15}" y="{y}" text-anchor="end" class="tick-label">{i + 1}</text>')

    # Draw cells
    for i in range(9):
        for j in range(9):
            val = grid[i, j]
            x = padding_x + j * cell_size
            y = padding_y + i * cell_size

            if np.isnan(val):
                # Empty cell (e.g. 1+9 or 9+1)
                svg_parts.append(f'<rect x="{x}" y="{y}" width="{cell_size-2}" height="{cell_size-2}" fill="#f3f4f6" rx="4" />')
                svg_parts.append(f'<text x="{x + cell_size/2}" y="{y + cell_size/2}" class="cell-text" fill="#9ca3af">—</text>')
                continue

            # Color mapping
            # Max expected magnitude ~ 0.72
            norm = np.clip(val / 0.75, -1.0, 1.0)
            if norm >= 0:
                # Cool Blue
                r = int(255 - norm * (255 - 37))
                g = int(255 - norm * (255 - 99))
                b = int(255 - norm * (255 - 235))
                text_color = "#ffffff" if norm > 0.45 else "#1e3a8a"
            else:
                # Warm Red
                anorm = abs(norm)
                r = int(255 - anorm * (255 - 220))
                g = int(255 - anorm * (255 - 38))
                b = int(255 - anorm * (255 - 38))
                text_color = "#ffffff" if anorm > 0.45 else "#7f1d1d"

            fill_hex = f"#{r:02x}{g:02x}{b:02x}"
            is_diag = (i == j)
            svg_parts.append(f'<rect x="{x}" y="{y}" width="{cell_size-2}" height="{cell_size-2}" fill="{fill_hex}" rx="4" />')

            if is_diag:
                svg_parts.append(f'<rect x="{x}" y="{y}" width="{cell_size-2}" height="{cell_size-2}" class="diag-border" rx="4" />')

            sign_str = f"{val:+.3f}"
            svg_parts.append(f'<text x="{x + cell_size/2}" y="{y + cell_size/2}" class="cell-text" fill="{text_color}">{sign_str}</text>')

    svg_parts.append('</svg>')

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))
    print(f"Generated Addition Grid Heatmap SVG at: {out_file}")


def generate_operator_swap_svg(output_path: str):
    """
    Generate an SVG heatmap comparing operators (+, -, *, and, then)
    across normalized adv/SD metrics.
    """
    data = [
        ("plus (+)", 5.22, 0.418, 0.080),
        ("times (*)", 2.43, 0.224, 0.092),
        ("minus (-)", 6.15, 0.323, 0.052),
        ("and", 4.95, 0.469, 0.095),
        ("then", 4.56, 0.549, 0.120),
    ]

    width = 680
    height = 360
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background-color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif;">',
        '<style>',
        '  .title { font-size: 16px; font-weight: bold; fill: #111827; }',
        '  .subtitle { font-size: 12px; fill: #4b5563; }',
        '  .header { font-size: 13px; font-weight: 600; fill: #374151; }',
        '  .row-label { font-size: 13px; font-weight: 500; fill: #1f2937; }',
        '  .bar-text { font-size: 12px; font-weight: bold; fill: #ffffff; }',
        '  .val-text { font-size: 12px; fill: #4b5563; }',
        '</style>',
        f'<text x="{width/2}" y="32" text-anchor="middle" class="title">Doubles Advantage Under Operator Swaps (adv/SD Ratio)</text>',
        f'<text x="{width/2}" y="52" text-anchor="middle" class="subtitle">Global peak occurs under subtraction (adv/SD = 6.15), falsifying mathematical addition.</text>',
    ]

    start_y = 90
    bar_height = 36
    max_adv_sd = 7.0
    bar_max_width = 320

    for idx, (op, adv_sd, raw_adv, sd) in enumerate(data):
        y = start_y + idx * (bar_height + 14)
        bar_w = (adv_sd / max_adv_sd) * bar_max_width

        color = "#2563eb"  # Blue
        if op.startswith("minus"):
            color = "#dc2626"  # Highlight red for subtraction peak!

        svg_parts.append(f'<text x="110" y="{y + bar_height/2 + 4}" text-anchor="end" class="row-label">{op}</text>')
        svg_parts.append(f'<rect x="130" y="{y}" width="{bar_w}" height="{bar_height}" fill="{color}" rx="6" />')
        svg_parts.append(f'<text x="{130 + bar_w - 12}" y="{y + bar_height/2 + 4}" text-anchor="end" class="bar-text">+{adv_sd:.2f} SD</text>')
        svg_parts.append(f'<text x="{140 + bar_w}" y="{y + bar_height/2 + 4}" class="val-text">(Adv: {raw_adv:+.3f}, SD: {sd:.3f})</text>')

    svg_parts.append('</svg>')

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))
    print(f"Generated Operator Swap Heatmap SVG at: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate Figures & Heatmaps")
    parser.add_argument("--grid-csv", type=str, default="data/addition_grid_79cell.csv")
    parser.add_argument("--grid-svg", type=str, default="figures/addition_grid_heatmap.svg")
    parser.add_argument("--operator-svg", type=str, default="figures/operator_swap_heatmap.svg")
    args = parser.parse_args()

    generate_addition_grid_svg(args.grid_csv, args.grid_svg)
    generate_operator_swap_svg(args.operator_svg)


if __name__ == "__main__":
    main()
