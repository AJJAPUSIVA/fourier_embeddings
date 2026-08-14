"""Generate dependency-free SVG plots from committed training_results.json."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


COLORS = {"dense": "#64748b", "kronecker": "#f59e0b", "fourier": "#2563eb"}


def svg_document(title: str, subtitle: str, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="540" viewBox="0 0 900 540" role="img" aria-labelledby="title desc">
<title id="title">{title}</title><desc id="desc">{subtitle}</desc>
<rect width="900" height="540" fill="#ffffff" rx="16"/>
<text x="60" y="52" font-family="system-ui,sans-serif" font-size="26" font-weight="700" fill="#0f172a">{title}</text>
<text x="60" y="80" font-family="system-ui,sans-serif" font-size="14" fill="#475569">{subtitle}</text>
{body}
</svg>\n'''


def bar_plot(report: dict, metric: str, output: Path) -> None:
    arms = [name for name in ("dense", "kronecker", "fourier") if name in report["arms"]]
    if metric == "perplexity":
        values = [report["arms"][arm]["validation_perplexity"]["mean"] for arm in arms]
        errors = [report["arms"][arm]["validation_perplexity"]["std"] for arm in arms]
        title = "Validation perplexity by embedding"
        subtitle = "WikiText-2, 1,000 matched steps; lower is better; error bars show ±1 sample SD across 3 seeds."
        scale = lambda value: value
        label = lambda value: f"{value:.2f}"
        max_value = max(value + error for value, error in zip(values, errors)) * 1.12
    elif metric == "parameters":
        values = [report["arms"][arm]["embedding_parameters"] for arm in arms]
        errors = [0.0] * len(values)
        title = "Trainable embedding parameters"
        subtitle = "Logarithmic bar height; exact parameter counts are printed above each bar."
        scale = lambda value: math.log10(max(value, 1))
        label = lambda value: f"{value:,}"
        max_value = max(scale(value) for value in values) * 1.08
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    baseline_y, plot_height = 455, 320
    bar_width, gap, start_x = 150, 90, 105
    pieces = [
        '<line x1="70" y1="455" x2="850" y2="455" stroke="#cbd5e1" stroke-width="2"/>',
    ]
    for index, (arm, value, error) in enumerate(zip(arms, values, errors)):
        x = start_x + index * (bar_width + gap)
        plotted = scale(value)
        height = plot_height * plotted / max_value
        y = baseline_y - height
        pieces.append(f'<rect x="{x}" y="{y:.2f}" width="{bar_width}" height="{height:.2f}" rx="8" fill="{COLORS[arm]}"/>')
        pieces.append(f'<text x="{x + bar_width / 2}" y="{y - 12:.2f}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="16" font-weight="650" fill="#0f172a">{label(value)}</text>')
        pieces.append(f'<text x="{x + bar_width / 2}" y="490" text-anchor="middle" font-family="system-ui,sans-serif" font-size="17" fill="#334155">{arm.title()}</text>')
        if metric == "perplexity" and error:
            error_height = plot_height * error / max_value
            center = x + bar_width / 2
            pieces.extend([
                f'<line x1="{center}" y1="{y-error_height:.2f}" x2="{center}" y2="{y+error_height:.2f}" stroke="#0f172a" stroke-width="2"/>',
                f'<line x1="{center-12}" y1="{y-error_height:.2f}" x2="{center+12}" y2="{y-error_height:.2f}" stroke="#0f172a" stroke-width="2"/>',
                f'<line x1="{center-12}" y1="{y+error_height:.2f}" x2="{center+12}" y2="{y+error_height:.2f}" stroke="#0f172a" stroke-width="2"/>',
            ])
    output.write_text(svg_document(title, subtitle, "\n".join(pieces)), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/training_results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bar_plot(report, "perplexity", args.output_dir / "validation_perplexity.svg")
    bar_plot(report, "parameters", args.output_dir / "embedding_parameters.svg")


if __name__ == "__main__":
    main()
