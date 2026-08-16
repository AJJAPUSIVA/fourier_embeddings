"""Generate dependency-free SVG plots for a Fourier dimension sweep."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def svg(title: str, subtitle: str, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="540" viewBox="0 0 900 540" role="img" aria-labelledby="title desc">
<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(subtitle)}</desc>
<rect width="900" height="540" fill="#fff" rx="16"/><text x="55" y="48" font-family="system-ui" font-size="25" font-weight="700">{html.escape(title)}</text>
<text x="55" y="76" font-family="system-ui" font-size="14" fill="#475569">{html.escape(subtitle)}</text>{body}</svg>\n'''


def scatter(report: dict, x_key: str, output: Path) -> None:
    rows = report["fourier_dimensions"]
    x_values = [row[x_key] for row in rows]
    y_values = [row["validation_perplexity"]["mean"] for row in rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min = min(y_values + [report["kronecker"]["validation_perplexity"]["mean"]])
    y_max = max(y_values + [report["kronecker"]["validation_perplexity"]["mean"]])
    y_pad = max((y_max - y_min) * 0.15, 1.0)
    y_min, y_max = y_min - y_pad, y_max + y_pad
    left, top, width, height = 85, 110, 750, 340
    xp = lambda value: left + width * (value - x_min) / max(x_max - x_min, 1)
    yp = lambda value: top + height * (y_max - value) / max(y_max - y_min, 1e-12)
    points = []
    for row, x, y in zip(rows, x_values, y_values):
        points.append(f'<circle cx="{xp(x):.2f}" cy="{yp(y):.2f}" r="7" fill="#2563eb"/>')
        points.append(f'<text x="{xp(x):.2f}" y="{yp(y)-13:.2f}" text-anchor="middle" font-family="system-ui" font-size="12">D={row["dimension"]}, {y:.1f}</text>')
    kron_y = report["kronecker"]["validation_perplexity"]["mean"]
    points.append(f'<line x1="{left}" y1="{yp(kron_y):.2f}" x2="{left+width}" y2="{yp(kron_y):.2f}" stroke="#f59e0b" stroke-width="2" stroke-dasharray="7 5"/>')
    points.append(f'<text x="{left+width-5}" y="{yp(kron_y)-8:.2f}" text-anchor="end" font-family="system-ui" font-size="12">Kronecker {kron_y:.1f}</text>')
    points.extend([
        f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" stroke="#94a3b8"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+height}" stroke="#94a3b8"/>',
        f'<text x="{left+width/2}" y="500" text-anchor="middle" font-family="system-ui" font-size="14">{"Fourier dimension" if x_key == "dimension" else "Embedding parameters"}</text>',
        f'<text x="24" y="{top+height/2}" transform="rotate(-90 24 {top+height/2})" text-anchor="middle" font-family="system-ui" font-size="14">Validation perplexity</text>',
    ])
    title = "Fourier dimension versus validation perplexity" if x_key == "dimension" else "Quality–compression frontier"
    output.write_text(svg(title, "Matched deterministic runs; lower perplexity and fewer parameters are better.", "".join(points)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scatter(report, "dimension", args.output_dir / "dimension_vs_perplexity.svg")
    scatter(report, "embedding_parameters", args.output_dir / "parameters_vs_perplexity.svg")


if __name__ == "__main__":
    main()
