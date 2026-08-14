"""Deterministic helpers for matched embedding experiments."""

from __future__ import annotations

import hashlib
import os
import random

import torch
from torch import nn


def configure_determinism(seed: int, *, strict: bool = True) -> None:
    """Seed process RNGs and request deterministic PyTorch algorithms."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(strict, warn_only=not strict)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def derived_seed(seed: int, namespace: str) -> int:
    """Derive a stable 63-bit seed without relying on Python's salted hash."""
    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def make_generator(seed: int, namespace: str) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derived_seed(seed, namespace))
    return generator


@torch.no_grad()
def initialize_matched_model(model: nn.Module, seed: int, arm: str) -> None:
    """Initialize shared tensors identically and embedding tensors per arm.

    Parameter names outside ``tok_emb`` use only ``seed`` and their name, so
    independently constructed experiment arms receive bit-identical transformer
    body, position embedding, and output-head initialization. Token embedding
    parameters additionally include the arm name because their shapes differ.
    """
    for name, parameter in model.named_parameters():
        if parameter.ndim <= 1:
            continue
        namespace = f"arm:{arm}:{name}" if name.startswith("tok_emb.") else f"shared:{name}"
        generator = make_generator(seed, namespace)
        nn.init.xavier_uniform_(parameter, generator=generator)


def epoch_permutation(size: int, seed: int, epoch: int) -> torch.Tensor:
    """Return the same CPU data order for every embedding arm."""
    return torch.randperm(size, generator=make_generator(seed, f"epoch:{epoch}"))
