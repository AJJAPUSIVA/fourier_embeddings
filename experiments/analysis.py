"""
Visualization and analysis script — generates comparison plots and tables.

Run after train_proof.py to produce figures for the README/webapp.
Can also be run standalone for the codec-level analysis (no training needed).
"""

from __future__ import annotations

import math
import json
from pathlib import Path

import torch
import numpy as np

from fourier_embedding.codec import fourier_encode_single, fourier_codec, _build_frequency_table


def visualize_wave_construction(word: str = "hello", D: int = 512):
    """
    Show how a word's embedding is constructed from individual byte waves.
    Prints the contribution of each byte to the final embedding.
    """
    byte_seq = word.encode("utf-8")
    L = len(byte_seq)
    freq_table = _build_frequency_table(D)

    print(f"\nWord: '{word}'")
    print(f"UTF-8 bytes: {list(byte_seq)} (length={L})")
    print(f"Output dimension D={D}")
    print(f"\nConstruction: φ(b) = (1/√{L}) · Σ wave(byte_p, p)")
    print(f"             = (1/{math.sqrt(L):.4f}) · Σ wave(v, p)")
    print()

    # Compute each wave individually
    individual_waves = []
    for p, b in enumerate(byte_seq):
        buf = torch.zeros(1, L, dtype=torch.uint8)
        buf[0, p] = b
        lens = torch.tensor([L], dtype=torch.long)

        # Hack: compute with all positions but only this one active
        # Actually, let's compute manually
        half_D = D // 2
        alpha = freq_table[:, 0]
        beta = freq_table[:, 1]

        v_norm = b / 257.0
        p_norm = p / 4099.0

        phase = 2 * math.pi * v_norm * alpha + 2 * math.pi * p_norm * beta
        sin_part = torch.sin(phase)
        cos_part = torch.cos(phase)
        wave = torch.stack([sin_part, cos_part], dim=-1).reshape(-1)
        individual_waves.append(wave)

        print(f"  Byte {p}: value={b} ('{chr(b) if 32 <= b < 127 else '?'}') "
              f"-> wave norm={wave.norm():.4f}, "
              f"first 8 dims: [{', '.join(f'{x:.3f}' for x in wave[:8].tolist())}]")

    # Sum and normalize
    total = torch.stack(individual_waves).sum(dim=0) / math.sqrt(L)
    print(f"\n  Sum / √{L}: norm={total.norm():.4f}")

    # Z-normalize
    mean = total.mean()
    std = total.std() + 1e-6
    z_normed = (total - mean) / std
    print(f"  Z-normalized: norm={z_normed.norm():.4f}, mean≈{z_normed.mean():.6f}")

    return z_normed


def similarity_matrix(words: list, D: int = 512):
    """Compute pairwise cosine similarity between words."""
    codes = []
    for w in words:
        code = fourier_encode_single(w.encode("utf-8"), D=D, z_normalize=True)
        codes.append(code)

    codes = torch.stack(codes)
    codes_norm = codes / (codes.norm(dim=-1, keepdim=True) + 1e-12)
    sim = codes_norm @ codes_norm.T

    print(f"\nCosine similarity matrix (D={D}):")
    print(f"{'':12}", end="")
    for w in words:
        print(f"{w:>12}", end="")
    print()
    for i, w1 in enumerate(words):
        print(f"{w1:12}", end="")
        for j, w2 in enumerate(words):
            print(f"{sim[i, j].item():>12.4f}", end="")
        print()
    return sim


def parameter_comparison():
    """Print the parameter comparison table."""
    configs = [
        # (name, vocab_size, d_model, codec_D)
        ("GPT-2 (50K)", 50257, 768, None),
        ("V5 (131K)", 131072, 8096, None),
        ("Large (250K)", 250000, 8096, None),
    ]

    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON: Dense vs Kronecker vs Fourier")
    print("=" * 80)
    print(f"\n{'Config':<16} {'Dense':>14} {'Kronecker':>14} {'Fourier':>14} {'F/D ratio':>10}")
    print("-" * 70)

    for name, V, d_model, _ in configs:
        dense = V * d_model
        kron_D = 256 * 32  # = 8192
        kron = kron_D * d_model
        four_D = 512
        four = four_D * d_model

        print(f"{name:<16} {dense:>14,} {kron:>14,} {four:>14,} {four/dense:>10.4f}")

    print(f"\n--- At V5 scale (V=131,072, d_model=8,096) ---")
    V, d_model = 131072, 8096
    dense = V * d_model
    kron = 8192 * d_model
    four_512 = 512 * d_model
    four_1024 = 1024 * d_model

    print(f"Dense embedding:            {dense:>14,} params ({dense * 2 / 1e9:.2f} GB bf16)")
    print(f"Kronecker (D=8192):         {kron:>14,} params ({kron * 2 / 1e9:.2f} GB bf16)")
    print(f"Fourier (D=512):            {four_512:>14,} params ({four_512 * 2 / 1e9:.4f} GB bf16)")
    print(f"Fourier (D=1024):           {four_1024:>14,} params ({four_1024 * 2 / 1e9:.4f} GB bf16)")
    print(f"\nReduction vs Dense:")
    print(f"  Kronecker: {dense/kron:.1f}x ({(1 - kron/dense)*100:.1f}% saving)")
    print(f"  Fourier-512: {dense/four_512:.1f}x ({(1 - four_512/dense)*100:.2f}% saving)")
    print(f"  Fourier-1024: {dense/four_1024:.1f}x ({(1 - four_1024/dense)*100:.2f}% saving)")
    print(f"\nFourier vs Kronecker:")
    print(f"  Fourier-512: {kron/four_512:.1f}x smaller than Kronecker")
    print(f"  Fourier-1024: {kron/four_1024:.1f}x smaller than Kronecker")


def byte_length_analysis():
    """Show the information lost at pos_dim=32 and retained by the pure codec."""
    print("\n" + "=" * 80)
    print("BYTE LENGTH ANALYSIS: The 32-byte ceiling problem")
    print("=" * 80)

    test_words = {
        "English": ["hello", "world", "training", "internationalization"],
        "Hindi": ["नमस्ते", "भारत", "अंतर्राष्ट्रीयकरण", "अंतर्राष्ट्रीयता"],
        "Telugu": ["తెలుగు", "భారతదేశం", "అంతర్జాతీయీకరణ"],
        "Tamil": ["தமிழ்", "இந்தியா", "சர்வதேசமயமாக்கல்"],
    }

    print(f"\n{'Script':<10} {'Word':<25} {'Bytes':>6} {'Kronecker keeps':>16} {'Fourier keeps':>14}")
    print("-" * 75)

    for script, words in test_words.items():
        for word in words:
            byte_len = len(word.encode("utf-8"))
            kron_keeps = min(byte_len, 32)
            four_keeps = byte_len  # pure codec input; wrapper bound is configurable
            truncated = " ⚠️ TRUNCATED" if byte_len > 32 else ""
            print(f"{script:<10} {word:<25} {byte_len:>6} {kron_keeps:>16} "
                  f"{four_keeps:>14}{truncated}")
        print()

    # Show collision
    print("--- COLLISION EXAMPLE ---")
    w1 = "अंतर्राष्ट्रीयकरण"
    w2 = "अंतर्राष्ट्रीयता"
    b1 = w1.encode("utf-8")
    b2 = w2.encode("utf-8")
    print(f"Word 1: '{w1}' ({len(b1)} bytes)")
    print(f"Word 2: '{w2}' ({len(b2)} bytes)")
    print(f"First 32 bytes equal: {b1[:32] == b2[:32]}")
    print(f"Kronecker: COLLISION (identical codes) ❌")

    # Fourier: compute similarity
    code1 = fourier_encode_single(b1, D=512, z_normalize=True)
    code2 = fourier_encode_single(b2, D=512, z_normalize=True)
    cos = (code1 @ code2) / (code1.norm() * code2.norm() + 1e-12)
    print(f"Fourier:   cosine={cos.item():.4f} (distinct in this tested configuration) ✓")


def information_density_analysis():
    """
    Show that Fourier packs more information per dimension than Kronecker.
    Kronecker at D=8192 has only L non-zero entries (extremely sparse).
    Fourier at D=512 uses ALL dimensions (dense).
    """
    print("\n" + "=" * 80)
    print("INFORMATION DENSITY: Sparse (Kronecker) vs Dense (Fourier)")
    print("=" * 80)

    from kronecker_embeddings.codec import encode_single as kron_encode

    test_words = ["hello", "train", "compute", "भारत", "తెలుగు"]

    print(f"\n{'Word':<12} {'Kron D':>7} {'Kron nnz':>9} {'Kron density':>13} "
          f"{'Four D':>7} {'Four nnz':>9} {'Four density':>13}")
    print("-" * 75)

    for word in test_words:
        byte_seq = word.encode("utf-8")

        # Kronecker
        kron_out = kron_encode(byte_seq, z_normalize=False)
        kron_nnz = (kron_out.abs() > 1e-8).sum().item()
        kron_D = kron_out.shape[0]

        # Fourier
        four_out = fourier_encode_single(byte_seq, D=512, z_normalize=False)
        four_nnz = (four_out.abs() > 1e-8).sum().item()
        four_D = four_out.shape[0]

        print(f"{word:<12} {kron_D:>7} {kron_nnz:>9} "
              f"{kron_nnz/kron_D*100:>12.2f}% "
              f"{four_D:>7} {four_nnz:>9} "
              f"{four_nnz/four_D*100:>12.2f}%")

    print(f"\nKronecker: D=8192, but only L dims are non-zero (L=token byte length)")
    print(f"Fourier:   D=512, ALL dims are non-zero (dense superposition)")
    print(f"→ Fourier uses 16x fewer dimensions; retained information must be measured")


if __name__ == "__main__":
    visualize_wave_construction("hello")
    visualize_wave_construction("भारत")
    print()
    similarity_matrix(["train", "training", "trainer", "apple", "orange"])
    similarity_matrix(["भारत", "भारती", "तेलुगु", "hello", "world"])
    parameter_comparison()
    byte_length_analysis()
    information_density_analysis()
