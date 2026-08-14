"""Matched tiny-transformer training for Dense, Kronecker, and Fourier arms."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fourier_embedding import FourierEmbedding
from kronecker_embeddings import KroneckerEmbedding

try:
    from determinism import configure_determinism, derived_seed, epoch_permutation, initialize_matched_model
except ImportError:
    from experiments.determinism import configure_determinism, derived_seed, epoch_permutation, initialize_matched_model


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 128
    dropout: float = 0.1


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model), nn.Dropout(cfg.dropout))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + self.dropout(attn_out)
        return x + self.ff(self.ln2(x))


class MiniLM(nn.Module):
    def __init__(self, cfg: ModelConfig, embedding_type: str, tokenizer, fourier_dim: int, max_byte_len: int):
        super().__init__()
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        if embedding_type == "dense":
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        elif embedding_type == "kronecker":
            self.tok_emb = KroneckerEmbedding(vocab_size=cfg.vocab_size, d_model=cfg.d_model, tokenizer=tokenizer, pos_dim=32, mode="dynamic")
        elif embedding_type == "fourier":
            self.tok_emb = FourierEmbedding(vocab_size=cfg.vocab_size, d_model=cfg.d_model, tokenizer=tokenizer, D=fourier_dim, max_byte_len=max_byte_len, mode="dynamic")
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        _, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        mask = torch.triu(torch.ones((length, length), device=input_ids.device, dtype=torch.bool), diagonal=1)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x, mask)
        return self.lm_head(self.ln_f(x))

    def count_params(self) -> dict[str, int]:
        return {"embedding": sum(p.numel() for p in self.tok_emb.parameters()), "total": sum(p.numel() for p in self.parameters())}


def synthetic_text() -> str:
    passages = (
        "The quick brown fox jumps over the lazy dog. ",
        "To be or not to be, that is the question. ",
        "All that glitters is not gold. ",
        "Mathematics describes patterns in number, space, and change. ",
        "A transformer predicts the next token from the tokens before it. ",
    )
    return "".join(passages * 20_000)


def load_token_ids(tokenizer, dataset: str, max_tokens: int) -> list[int]:
    if dataset == "synthetic":
        text = synthetic_text()
    elif dataset == "wikitext":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("--dataset wikitext requires the dev dependencies") from exc
        records = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(value for value in records["text"] if value.strip())
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    token_ids = tokenizer.encode(text)[:max_tokens]
    if len(token_ids) < 2:
        raise RuntimeError(f"Dataset produced only {len(token_ids)} tokens")
    return token_ids


def make_splits(token_ids: list[int], sequence_length: int) -> tuple[Tensor, Tensor]:
    row_length = sequence_length + 1
    row_count = len(token_ids) // row_length
    if row_count < 10:
        raise ValueError("Need at least 10 complete sequences; increase --max-tokens")
    data = torch.tensor(token_ids[: row_count * row_length]).reshape(row_count, row_length)
    train_count = max(1, int(0.9 * row_count))
    return data[:train_count], data[train_count:]


def train_steps(model, data, optimizer, device, *, seed: int, batch_size: int, grad_accumulation: int, max_steps: int, log_every: int):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    history, step, micro_step, epoch, loss_sum = [], 0, 0, 0, 0.0
    started = time.perf_counter()
    while step < max_steps:
        order = epoch_permutation(data.size(0), seed, epoch)
        for offset in range(0, data.size(0), batch_size):
            batch = data.index_select(0, order[offset:offset + batch_size]).to(device)
            logits = model(batch[:, :-1])
            raw_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
            (raw_loss / grad_accumulation).backward()
            loss_sum += raw_loss.item()
            micro_step += 1
            if micro_step % grad_accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % log_every == 0 or step == max_steps:
                elapsed = time.perf_counter() - started
                average = loss_sum / grad_accumulation
                history.append({"step": step, "train_loss": average, "elapsed_s": elapsed})
                print(f"step={step}/{max_steps} loss={average:.4f} elapsed={elapsed:.1f}s", flush=True)
            loss_sum = 0.0
            if step >= max_steps:
                break
        epoch += 1
    return history, time.perf_counter() - started


@torch.no_grad()
def evaluate(model, data, device, batch_size: int, max_batches: int):
    model.eval()
    losses = []
    for offset in range(0, min(data.size(0), batch_size * max_batches), batch_size):
        batch = data[offset:offset + batch_size].to(device)
        logits = model(batch[:, :-1])
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1)).item())
    average = sum(losses) / len(losses)
    return average, math.exp(average)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", choices=("dense", "kronecker", "fourier"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", choices=("synthetic", "wikitext"), default="synthetic")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--max-tokens", type=int, default=50_000)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--fourier-dim", type=int, default=512)
    parser.add_argument("--max-byte-len", type=int, default=256)
    parser.add_argument("--allow-nondeterministic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    positive = ("max_tokens", "sequence_length", "batch_size", "grad_accumulation", "max_steps", "eval_batches", "log_every", "d_model", "n_heads", "n_layers", "d_ff", "fourier_dim", "max_byte_len")
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.d_model % args.n_heads:
        parser.error("--d-model must be divisible by --n-heads")
    if args.fourier_dim % 2:
        parser.error("--fourier-dim must be even")
    return args


def main(argv: Optional[list[str]] = None) -> dict:
    args = parse_args(argv)
    configure_determinism(args.seed, strict=not args.allow_nondeterministic)
    from transformers import AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    token_ids = load_token_ids(tokenizer, args.dataset, args.max_tokens)
    train_data, validation_data = make_splits(token_ids, args.sequence_length)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, max_seq_len=args.sequence_length)
    model = MiniLM(cfg, args.embedding, tokenizer, args.fourier_dim, args.max_byte_len)
    initialize_matched_model(model, args.seed, args.embedding)
    model.to(device)
    torch.manual_seed(derived_seed(args.seed, "training"))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(derived_seed(args.seed, "training"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history, elapsed = train_steps(model, train_data, optimizer, device, seed=args.seed, batch_size=args.batch_size, grad_accumulation=args.grad_accumulation, max_steps=args.max_steps, log_every=args.log_every)
    validation_loss, validation_perplexity = evaluate(model, validation_data, device, args.batch_size, args.eval_batches)
    effective_batch = args.batch_size * args.grad_accumulation
    result = {
        "schema_version": 1, "embedding": args.embedding, "seed": args.seed,
        "dataset": args.dataset, "tokenizer": args.tokenizer, "device": str(device),
        "python": platform.python_version(), "torch": torch.__version__,
        "deterministic": not args.allow_nondeterministic, "model_config": asdict(cfg),
        "fourier_dim": args.fourier_dim if args.embedding == "fourier" else None,
        "training": {"max_tokens": len(token_ids), "train_sequences": train_data.size(0), "validation_sequences": validation_data.size(0), "batch_size": args.batch_size, "grad_accumulation": args.grad_accumulation, "effective_batch_size": effective_batch, "max_steps": args.max_steps, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay},
        "parameters": model.count_params(), "history": history,
        "final_validation_loss": validation_loss, "final_validation_perplexity": validation_perplexity,
        "elapsed_s": elapsed, "tokens_per_second": (effective_batch * args.sequence_length * args.max_steps) / elapsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"validation_loss={validation_loss:.4f} validation_perplexity={validation_perplexity:.2f} output={args.output}")
    return result


if __name__ == "__main__":
    main()
