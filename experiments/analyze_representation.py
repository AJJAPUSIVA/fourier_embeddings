"""Measure Fourier-code norms, nearest neighbors, distances, and margins."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

try:
    from experiments.collision_probe import (
        collision_groups,
        encode_vocabulary,
        load_vocabulary,
        nearest_distinct_neighbors,
    )
except ModuleNotFoundError:  # Direct execution without an editable install.
    from collision_probe import (  # type: ignore[no-redef]
        collision_groups,
        encode_vocabulary,
        load_vocabulary,
        nearest_distinct_neighbors,
    )


def summary(values: torch.Tensor) -> dict[str, object]:
    values = values.detach().cpu().to(torch.float64)
    if not values.numel():
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=True)) if values.numel() > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def quantiles(values: torch.Tensor, levels: tuple[float, ...]) -> dict[str, float]:
    values = values.detach().cpu().to(torch.float64)
    if not values.numel():
        return {}
    return {str(level): float(torch.quantile(values, level)) for level in levels}


def pearson_correlation(
    left: torch.Tensor,
    right: torch.Tensor,
    relative_constant_tolerance: float = 1e-6,
) -> dict[str, object]:
    """Return a correlation only when both variables have meaningful variance."""
    left = left.detach().cpu().to(torch.float64)
    right = right.detach().cpu().to(torch.float64)
    if left.numel() < 2 or right.numel() != left.numel():
        return {
            "value": None,
            "status": "unavailable_insufficient_observations",
            "relative_constant_tolerance": relative_constant_tolerance,
        }
    left_mean = left.mean()
    right_mean = right.mean()
    left_centered = left - left_mean
    right_centered = right - right_mean
    right_scale = max(abs(float(right_mean)), 1.0)
    if float(right.std(unbiased=False)) <= relative_constant_tolerance * right_scale:
        return {
            "value": None,
            "status": "unavailable_near_constant_right_variable",
            "relative_constant_tolerance": relative_constant_tolerance,
        }
    denominator = left_centered.norm() * right_centered.norm()
    if denominator == 0:
        return {
            "value": None,
            "status": "unavailable_zero_variance",
            "relative_constant_tolerance": relative_constant_tolerance,
        }
    return {
        "value": float((left_centered @ right_centered) / denominator),
        "status": "computed",
        "relative_constant_tolerance": relative_constant_tolerance,
    }


def norm_analysis(
    raw_bytes: list[bytes],
    stage_codes: dict[str, torch.Tensor],
) -> dict:
    lengths = torch.tensor([len(row) for row in raw_bytes], dtype=torch.long)
    norms = {name: codes.to(torch.float32).norm(dim=1) for name, codes in stage_codes.items()}
    groups = []
    for length in sorted(set(lengths.tolist())):
        mask = lengths == length
        item = {"byte_length": length, "count": int(mask.sum())}
        for name, values in norms.items():
            item[name] = summary(values[mask])
        groups.append(item)
    return {
        "byte_length": {
            "summary": summary(lengths.to(torch.float32)),
            "quantiles": quantiles(lengths.to(torch.float32), (0.0, 0.5, 0.9, 0.99, 1.0)),
        },
        "stages": {
            name: {
                "norm": summary(values),
                "norm_length_pearson": pearson_correlation(lengths, values),
            }
            for name, values in norms.items()
        },
        "by_byte_length": groups,
    }


def nearest_distinct_euclidean(
    codes: torch.Tensor,
    raw_bytes: list[bytes],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find each row's nearest Euclidean neighbor with different bytes."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    count = codes.size(0)
    if count < 2:
        return torch.full((count,), math.inf), torch.full((count,), -1, dtype=torch.long)
    codes = codes.detach().cpu().to(torch.float32)
    best_values = torch.full((count,), math.inf, dtype=torch.float32)
    best_indices = torch.full((count,), -1, dtype=torch.long)
    same_byte_rows = defaultdict(list)
    for index, raw in enumerate(raw_bytes):
        same_byte_rows[raw].append(index)
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        distances = torch.cdist(codes[start:stop], codes)
        for local_row, global_row in enumerate(range(start, stop)):
            distances[local_row, same_byte_rows[raw_bytes[global_row]]] = math.inf
        values, indices = distances.min(dim=1)
        best_values[start:stop] = values
        best_indices[start:stop] = indices
    return best_values, best_indices


def _neighbor_examples(
    metric: str,
    values: torch.Tensor,
    neighbors: torch.Tensor,
    token_ids: list[int],
    raw_bytes: list[bytes],
    token_text: list[str],
    *,
    descending: bool,
    limit: int = 20,
) -> list[dict]:
    order = torch.argsort(values, descending=descending)
    seen = set()
    examples = []
    for row_value in order:
        row = int(row_value)
        neighbor = int(neighbors[row])
        if neighbor < 0 or not math.isfinite(float(values[row])):
            continue
        pair = tuple(sorted((row, neighbor)))
        if pair in seen:
            continue
        seen.add(pair)
        examples.append({
            metric: float(values[row]),
            "left": {
                "token_id": token_ids[row],
                "text": token_text[row],
                "bytes_hex": raw_bytes[row].hex(),
                "byte_length": len(raw_bytes[row]),
            },
            "right": {
                "token_id": token_ids[neighbor],
                "text": token_text[neighbor],
                "bytes_hex": raw_bytes[neighbor].hex(),
                "byte_length": len(raw_bytes[neighbor]),
            },
        })
        if len(examples) >= limit:
            break
    return examples


def neighbor_analysis(
    codes: torch.Tensor,
    raw_bytes: list[bytes],
    token_ids: list[int],
    token_text: list[str],
    block_size: int,
) -> tuple[dict, torch.Tensor]:
    cosine, cosine_neighbors = nearest_distinct_neighbors(codes, raw_bytes, block_size)
    distance, distance_neighbors = nearest_distinct_euclidean(codes, raw_bytes, block_size)
    valid_cosine = torch.isfinite(cosine)
    valid_distance = torch.isfinite(distance)
    margins = 1.0 - cosine[valid_cosine]
    lengths = torch.tensor([len(row) for row in raw_bytes], dtype=torch.long)
    by_length = []
    for length in sorted(set(lengths.tolist())):
        mask = lengths == length
        cosine_mask = mask & valid_cosine
        distance_mask = mask & valid_distance
        by_length.append({
            "byte_length": length,
            "count": int(mask.sum()),
            "nearest_distinct_cosine": summary(cosine[cosine_mask]),
            "nearest_distinct_euclidean": summary(distance[distance_mask]),
            "cosine_self_retrieval_margin": summary(1.0 - cosine[cosine_mask]),
        })
    return {
        "analyzed_tokens": len(raw_bytes),
        "distinct_byte_strings": len(set(raw_bytes)),
        "nearest_distinct_cosine": {
            "summary": summary(cosine[valid_cosine]),
            "quantiles": quantiles(cosine[valid_cosine], (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)),
            "worst_pairs": _neighbor_examples(
                "cosine", cosine, cosine_neighbors, token_ids, raw_bytes, token_text,
                descending=True,
            ),
        },
        "nearest_distinct_euclidean": {
            "summary": summary(distance[valid_distance]),
            "quantiles": quantiles(distance[valid_distance], (0.0, 0.001, 0.01, 0.1, 0.5, 1.0)),
            "closest_pairs": _neighbor_examples(
                "euclidean", distance, distance_neighbors, token_ids, raw_bytes, token_text,
                descending=False,
            ),
        },
        "cosine_self_retrieval_margin": {
            "summary": summary(margins),
            "quantiles": quantiles(margins, (0.0, 0.001, 0.01, 0.1, 0.5, 1.0)),
        },
        "by_byte_length": by_length,
    }, cosine[valid_cosine]


def _svg(title: str, subtitle: str, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="560" viewBox="0 0 960 560" role="img" aria-labelledby="title desc">
<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(subtitle)}</desc>
<rect width="960" height="560" fill="#ffffff" rx="16"/>
<text x="60" y="48" font-family="system-ui,sans-serif" font-size="25" font-weight="700" fill="#0f172a">{html.escape(title)}</text>
<text x="60" y="76" font-family="system-ui,sans-serif" font-size="14" fill="#475569">{html.escape(subtitle)}</text>
{body}</svg>
'''


def write_norm_plot(analysis: dict, output: Path) -> None:
    groups = analysis["by_byte_length"]
    series = (
        ("raw_sum", "#dc2626", "Raw sum"),
        ("length_normalized", "#f59e0b", "Length normalized"),
        ("z_normalized", "#2563eb", "Z-normalized"),
    )
    points = [
        (item["byte_length"], name, item[name]["mean"])
        for item in groups for name, _, _ in series if item[name]["mean"] is not None
    ]
    max_length = max((item[0] for item in points), default=1)
    max_norm = max((item[2] for item in points), default=1.0)
    left, top, width, height = 80, 110, 820, 360
    pieces = [
        f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" stroke="#94a3b8"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+height}" stroke="#94a3b8"/>',
    ]
    for index, (name, color, label) in enumerate(series):
        coords = []
        for item in groups:
            value = item[name]["mean"]
            if value is None:
                continue
            x = left + width * item["byte_length"] / max(max_length, 1)
            y = top + height * (1 - value / max(max_norm, 1e-12))
            coords.append(f"{x:.2f},{y:.2f}")
        pieces.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="3"/>')
        legend_x = 105 + index * 245
        pieces.append(f'<line x1="{legend_x}" y1="515" x2="{legend_x+30}" y2="515" stroke="{color}" stroke-width="4"/>')
        pieces.append(f'<text x="{legend_x+38}" y="520" font-family="system-ui,sans-serif" font-size="14" fill="#334155">{label}</text>')
    pieces.extend([
        f'<text x="{left+width/2}" y="500" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14">Byte length</text>',
        f'<text x="22" y="{top+height/2}" transform="rotate(-90 22 {top+height/2})" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14">Mean L2 norm</text>',
        f'<text x="{left}" y="490" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12">0</text>',
        f'<text x="{left+width}" y="490" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12">{max_length}</text>',
    ])
    output.write_text(_svg(
        "Fourier-code norm by token byte length",
        "Means for the raw sum, 1/sqrt(L) length normalization, and final per-token z-normalization.",
        "\n".join(pieces),
    ), encoding="utf-8")


def write_cosine_histogram(values: torch.Tensor, output: Path, bins: int = 30) -> None:
    values = values.detach().cpu().to(torch.float32)
    minimum = float(values.min()) if values.numel() else 0.0
    maximum = float(values.max()) if values.numel() else 1.0
    if minimum == maximum:
        maximum = minimum + 1e-6
    counts = torch.histc(values, bins=bins, min=minimum, max=maximum) if values.numel() else torch.zeros(bins)
    max_count = max(float(counts.max()), 1.0)
    left, top, width, height = 80, 110, 820, 360
    bar_width = width / bins
    pieces = [
        f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" stroke="#94a3b8"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+height}" stroke="#94a3b8"/>',
    ]
    for index, count in enumerate(counts.tolist()):
        bar_height = height * count / max_count
        x = left + index * bar_width
        y = top + height - bar_height
        pieces.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width-1, 1):.2f}" height="{bar_height:.2f}" fill="#2563eb"/>')
    pieces.extend([
        f'<text x="{left}" y="492" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12">{minimum:.3f}</text>',
        f'<text x="{left+width}" y="492" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12">{maximum:.3f}</text>',
        f'<text x="{left+width/2}" y="520" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14">Nearest distinct cosine similarity</text>',
        f'<text x="22" y="{top+height/2}" transform="rotate(-90 22 {top+height/2})" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14">Token count</text>',
    ])
    output.write_text(_svg(
        "Nearest-neighbour cosine distribution",
        "Each token is compared with its closest token having a different byte string.",
        "\n".join(pieces),
    ), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=50257)
    parser.add_argument("--near-max-tokens", type=int, default=10000)
    parser.add_argument("--codec-batch-size", type=int, default=512)
    parser.add_argument("--frequency-chunk-size", type=int, default=16)
    parser.add_argument("--similarity-block-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("results/representation_analysis.json"))
    parser.add_argument("--plot-dir", type=Path, default=Path("results/plots"))
    return parser.parse_args()


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    if args.dimension <= 0 or args.dimension % 2:
        raise SystemExit("dimension must be a positive even integer")
    if args.max_tokens <= 0 or args.near_max_tokens <= 0:
        raise SystemExit("token limits must be positive")
    if args.codec_batch_size <= 0 or args.frequency_chunk_size <= 0:
        raise SystemExit("codec batch and frequency chunk sizes must be positive")
    if args.similarity_block_size <= 0:
        raise SystemExit("similarity block size must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    token_ids, raw_bytes, token_text = load_vocabulary(tokenizer, args.max_tokens)
    common = {
        "raw_bytes": raw_bytes,
        "dimension": args.dimension,
        "batch_size": args.codec_batch_size,
        "frequency_chunk_size": args.frequency_chunk_size,
    }
    stage_codes = {
        "raw_sum": encode_vocabulary(**common, length_normalize=False, z_normalize=False),
        "length_normalized": encode_vocabulary(**common, length_normalize=True, z_normalize=False),
        "z_normalized": encode_vocabulary(**common, length_normalize=True, z_normalize=True),
    }
    if any(not torch.isfinite(codes).all() for codes in stage_codes.values()):
        raise RuntimeError("non-finite codec output detected")

    near_count = min(len(raw_bytes), args.near_max_tokens)
    neighbors, nearest_cosines = neighbor_analysis(
        stage_codes["z_normalized"][:near_count],
        raw_bytes[:near_count], token_ids[:near_count], token_text[:near_count],
        args.similarity_block_size,
    )
    report = {
        "schema_version": 2,
        "tokenizer": args.tokenizer,
        "dimension": args.dimension,
        "analyzed_tokens": len(raw_bytes),
        "normalization_stages": {
            "raw_sum": {"length_normalize": False, "z_normalize": False},
            "length_normalized": {"length_normalize": True, "z_normalize": False},
            "z_normalized": {"length_normalize": True, "z_normalize": True},
        },
        "metric_relationships": {
            "euclidean_after_z_normalization": {
                "formula": "distance = sqrt(2 * norm^2 * (1 - cosine)) for equal norms",
                "interpretation": (
                    "Per-token z-normalization makes norms nearly constant, so the reported "
                    "Euclidean and cosine nearest-neighbor results are not independent evidence."
                ),
            },
        },
        "norm_analysis": norm_analysis(raw_bytes, stage_codes),
        "collisions": {
            "exact": collision_groups(stage_codes["z_normalized"], raw_bytes),
            "quantized_4dp": collision_groups(stage_codes["z_normalized"], raw_bytes, decimals=4),
        },
        "nearest_neighbors": neighbors,
        "scope": {
            "norm_and_collision_tokens": len(raw_bytes),
            "nearest_neighbor_tokens": near_count,
            "nearest_neighbor_selection": "deterministic tokenizer-ID prefix",
            "duplicate_byte_strings_excluded_from_neighbor_candidates": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_norm_plot(report["norm_analysis"], args.plot_dir / "norm_by_length.svg")
    write_cosine_histogram(nearest_cosines, args.plot_dir / "nearest_cosine_distribution.svg")
    print(f"wrote {args.output}")
    print(f"wrote {args.plot_dir / 'norm_by_length.svg'}")
    print(f"wrote {args.plot_dir / 'nearest_cosine_distribution.svg'}")


if __name__ == "__main__":
    main()
