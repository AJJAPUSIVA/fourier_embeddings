"""Vocabulary-scale exact and near-collision analysis for Fourier codes."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from fourier_embedding.codec import fourier_codec
from kronecker_embeddings.tokenizer_utils import token_id_to_bytes


def _row_keys(values: np.ndarray) -> np.ndarray:
    """Return one hashable fixed-width byte key per contiguous array row."""
    contiguous = np.ascontiguousarray(values)
    row_dtype = np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    return contiguous.view(row_dtype).reshape(-1)


def collision_groups(
    codes: torch.Tensor,
    raw_bytes: list[bytes],
    decimals: int | None = None,
    max_examples: int = 10,
) -> dict:
    """Count code collisions between *different* byte sequences.

    ``decimals=None`` compares exact fp32 bit patterns. Otherwise rows are
    rounded before comparison, providing a quantized-collision diagnostic.
    """
    array = codes.detach().cpu().to(torch.float32).numpy()
    if decimals is not None:
        array = np.round(array, decimals=decimals)
    keys = _row_keys(array)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)

    groups = []
    involved = set()
    for group_id in np.flatnonzero(counts > 1):
        indexes = np.flatnonzero(inverse == group_id).tolist()
        distinct = {}
        for index in indexes:
            distinct.setdefault(raw_bytes[index], index)
        if len(distinct) < 2:
            continue
        representatives = list(distinct.values())
        groups.append(representatives)
        involved.update(indexes)

    examples = [
        [
            {"row": index, "bytes_hex": raw_bytes[index].hex()}
            for index in group[:4]
        ]
        for group in groups[:max_examples]
    ]
    return {
        "rounding_decimals": decimals,
        "collision_groups": len(groups),
        "tokens_in_collision_groups": len(involved),
        "examples": examples,
    }


def nearest_distinct_neighbors(
    codes: torch.Tensor,
    raw_bytes: list[bytes],
    block_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find each row's closest cosine neighbor with a different byte string."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    count = codes.size(0)
    if count < 2:
        return torch.full((count,), -math.inf), torch.full((count,), -1, dtype=torch.long)

    normalized = F.normalize(codes.to(torch.float32), dim=1, eps=1e-12)
    best_values = torch.full((count,), -math.inf, dtype=torch.float32)
    best_indices = torch.full((count,), -1, dtype=torch.long)
    same_byte_rows = defaultdict(list)
    for index, raw in enumerate(raw_bytes):
        same_byte_rows[raw].append(index)

    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        similarities = normalized[start:stop] @ normalized.T
        for local_row, global_row in enumerate(range(start, stop)):
            similarities[local_row, same_byte_rows[raw_bytes[global_row]]] = -math.inf
        values, indices = similarities.max(dim=1)
        best_values[start:stop] = values
        best_indices[start:stop] = indices

    return best_values, best_indices


def near_collision_report(
    codes: torch.Tensor,
    raw_bytes: list[bytes],
    token_ids: list[int],
    token_text: list[str],
    thresholds: Iterable[float],
    block_size: int,
    worst_pairs: int = 20,
) -> dict:
    values, neighbors = nearest_distinct_neighbors(
        codes, raw_bytes, block_size=block_size
    )
    finite = torch.isfinite(values)
    valid_values = values[finite]
    quantiles = {}
    if valid_values.numel():
        for q in (0.5, 0.9, 0.99, 0.999, 1.0):
            quantiles[str(q)] = float(torch.quantile(valid_values, q))

    threshold_counts = {
        str(threshold): int((valid_values >= threshold).sum())
        for threshold in thresholds
    }

    order = torch.argsort(values, descending=True)
    seen_pairs = set()
    examples = []
    for row_tensor in order:
        row = int(row_tensor)
        neighbor = int(neighbors[row])
        if neighbor < 0:
            continue
        pair = tuple(sorted((row, neighbor)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        examples.append({
            "cosine": float(values[row]),
            "left": {
                "token_id": token_ids[row],
                "text": token_text[row],
                "bytes_hex": raw_bytes[row].hex(),
            },
            "right": {
                "token_id": token_ids[neighbor],
                "text": token_text[neighbor],
                "bytes_hex": raw_bytes[neighbor].hex(),
            },
        })
        if len(examples) >= worst_pairs:
            break

    return {
        "nearest_distinct_cosine_quantiles": quantiles,
        "near_collision_counts": threshold_counts,
        "minimum_self_retrieval_margin": (
            float(1.0 - valid_values.max()) if valid_values.numel() else None
        ),
        "worst_distinct_pairs": examples,
    }


@torch.inference_mode()
def encode_vocabulary(
    raw_bytes: list[bytes],
    dimension: int,
    batch_size: int,
    frequency_chunk_size: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, len(raw_bytes), batch_size):
        rows = raw_bytes[start : start + batch_size]
        active_len = max((len(row) for row in rows), default=0)
        buffer = torch.zeros((len(rows), active_len), dtype=torch.uint8)
        lengths = torch.tensor([len(row) for row in rows], dtype=torch.long)
        for index, row in enumerate(rows):
            if row:
                buffer[index, : len(row)] = torch.tensor(list(row), dtype=torch.uint8)
        chunks.append(fourier_codec(
            buffer,
            lengths,
            D=dimension,
            frequency_chunk_size=frequency_chunk_size,
        ).cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty((0, dimension))


def load_vocabulary(tokenizer, max_tokens: int | None) -> tuple[list[int], list[bytes], list[str]]:
    vocab_size = max(tokenizer.get_vocab().values()) + 1
    limit = min(vocab_size, max_tokens) if max_tokens else vocab_size
    token_ids, rows, texts = [], [], []
    for token_id in range(limit):
        try:
            raw = token_id_to_bytes(tokenizer, token_id)
        except Exception:
            continue
        token_ids.append(token_id)
        rows.append(bytes(raw))
        texts.append(tokenizer.decode([token_id]))
    return token_ids, rows, texts


def markdown_summary(report: dict) -> str:
    lines = [
        "# Fourier collision analysis",
        "",
        f"Tokenizer: `{report['tokenizer']}`  ",
        f"Analyzed tokens: {report['analyzed_tokens']:,}",
        "",
        "| D | Exact groups | Quantized groups (4 dp) | Near N | >=0.999 cosine | Worst cosine |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["dimensions"]:
        exact = result["exact"]["collision_groups"]
        quantized = result["quantized"]["4"]["collision_groups"]
        near = result["near"]["near_collision_counts"].get("0.999", 0)
        worst = result["near"]["nearest_distinct_cosine_quantiles"].get("1.0")
        worst_text = f"{worst:.8f}" if worst is not None else "n/a"
        lines.append(
            f"| {result['D']} | {exact} | {quantized} | "
            f"{result['near_analyzed_tokens']} | {near} | {worst_text} |"
        )
    lines.extend([
        "",
        "Exact and quantized groups exclude tokens with identical byte strings.",
        "Near-neighbour statistics likewise use only distinct byte sequences.",
    ])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[512])
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--codec-batch-size", type=int, default=512)
    parser.add_argument("--frequency-chunk-size", type=int, default=16)
    parser.add_argument("--similarity-block-size", type=int, default=256)
    parser.add_argument(
        "--near-max-tokens",
        type=int,
        default=10000,
        help="Exact all-pairs near-collision analysis limit; exact code collisions still use all tokens",
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.99, 0.999, 0.9999])
    parser.add_argument("--output", type=Path, default=Path("collision_results.json"))
    parser.add_argument("--summary", type=Path, default=Path("collision_summary.md"))
    return parser.parse_args()


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    if any(d <= 0 or d % 2 for d in args.dimensions):
        raise SystemExit("all dimensions must be positive even integers")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise SystemExit("max-tokens must be positive")
    if args.near_max_tokens <= 0:
        raise SystemExit("near-max-tokens must be positive")
    if args.codec_batch_size <= 0 or args.frequency_chunk_size <= 0:
        raise SystemExit("codec-batch-size and frequency-chunk-size must be positive")
    if args.similarity_block_size <= 0:
        raise SystemExit("similarity-block-size must be positive")
    if any(threshold < -1.0 or threshold > 1.0 for threshold in args.thresholds):
        raise SystemExit("cosine thresholds must fall within [-1, 1]")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    token_ids, rows, texts = load_vocabulary(tokenizer, args.max_tokens)
    report = {
        "tokenizer": args.tokenizer,
        "analyzed_tokens": len(rows),
        "max_tokens": args.max_tokens,
        "thresholds": args.thresholds,
        "dimensions": [],
    }

    for dimension in args.dimensions:
        started = time.perf_counter()
        codes = encode_vocabulary(
            rows,
            dimension=dimension,
            batch_size=args.codec_batch_size,
            frequency_chunk_size=args.frequency_chunk_size,
        )
        if not torch.isfinite(codes).all():
            raise RuntimeError(f"non-finite codec output detected at D={dimension}")
        exact = collision_groups(codes, rows)
        quantized = {
            str(decimals): collision_groups(codes, rows, decimals=decimals)
            for decimals in (3, 4, 5)
        }
        near_count = min(len(rows), args.near_max_tokens)
        near = near_collision_report(
            codes[:near_count],
            rows[:near_count],
            token_ids[:near_count],
            texts[:near_count],
            thresholds=args.thresholds,
            block_size=args.similarity_block_size,
        )
        report["dimensions"].append({
            "D": dimension,
            "elapsed_seconds": time.perf_counter() - started,
            "exact": exact,
            "quantized": quantized,
            "near_analyzed_tokens": near_count,
            "near": near,
        })
        print(
            f"D={dimension}: exact_groups={exact['collision_groups']} "
            f"worst_cosine={near['nearest_distinct_cosine_quantiles'].get('1.0')}"
        )

    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = markdown_summary(report)
    args.summary.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
