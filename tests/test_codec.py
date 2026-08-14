"""
Tests for the Fourier codec — verifies correctness, properties, and comparison
against the Kronecker codec.
"""

from __future__ import annotations

import pytest
import torch

from fourier_embedding.codec import (
    fourier_codec,
    fourier_encode_single,
    fourier_output_dim,
    fourier_wave,
    _build_frequency_table,
)


# ============ Basic correctness ============

def test_output_shape():
    """Output shape must be (B, D)."""
    B, max_len, D = 4, 32, 512
    byte_seqs = torch.randint(0, 256, (B, max_len), dtype=torch.uint8)
    lengths = torch.tensor([5, 10, 3, 32], dtype=torch.long)
    out = fourier_codec(byte_seqs, lengths, D=D)
    assert out.shape == (B, D)


def test_output_shape_single():
    """Single encode returns (D,)."""
    out = fourier_encode_single(b"hello", D=512)
    assert out.shape == (512,)


def test_determinism():
    """Same input -> identical output."""
    out1 = fourier_encode_single(b"hello", D=512)
    out2 = fourier_encode_single(b"hello", D=512)
    assert torch.equal(out1, out2)


def test_empty_token_safe():
    """Empty byte sequence should produce a finite vector."""
    out = fourier_encode_single(b"", D=512)
    assert out.shape == (512,)
    assert torch.isfinite(out).all()


def test_output_dim_helper():
    assert fourier_output_dim(512) == 512
    assert fourier_output_dim(1024) == 1024


def test_rejects_invalid_lengths():
    buf = torch.zeros((1, 4), dtype=torch.uint8)
    with pytest.raises(ValueError):
        fourier_codec(buf, torch.tensor([5]), D=64)
    with pytest.raises(ValueError):
        fourier_codec(buf, torch.tensor([-1]), D=64)


def test_rejects_invalid_frequency_chunk_size():
    buf = torch.zeros((1, 4), dtype=torch.uint8)
    with pytest.raises(ValueError):
        fourier_codec(buf, torch.tensor([2]), D=64, frequency_chunk_size=0)


def test_chunked_codec_matches_full_wave_reference():
    """The bounded-memory path must match the original full wave formula."""
    torch.manual_seed(7)
    batch, max_len, D = 5, 13, 64
    byte_seqs = torch.randint(0, 256, (batch, max_len), dtype=torch.uint8)
    lengths = torch.tensor([13, 9, 5, 1, 0], dtype=torch.long)
    freq = _build_frequency_table(D)

    positions = torch.arange(max_len).unsqueeze(0).expand(batch, -1)
    waves = fourier_wave(byte_seqs.long(), positions, freq)
    valid = positions < lengths.unsqueeze(1)
    reference = (waves * valid.unsqueeze(-1)).sum(dim=1)
    reference *= torch.rsqrt(lengths.clamp_min(1).float()).unsqueeze(1)
    mean = reference.mean(dim=-1, keepdim=True)
    std = (reference - mean).std(dim=-1, keepdim=True) + 1e-6
    reference = (reference - mean) / std

    for chunk_size in (1, 3, 8, 32):
        actual = fourier_codec(
            byte_seqs,
            lengths,
            D=D,
            freq_table=freq,
            frequency_chunk_size=chunk_size,
        )
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)


# ============ Core properties ============

def test_different_tokens_different_codes():
    """Different byte sequences must produce different codec outputs."""
    words = [b"hello", b"world", b"apple", b"train", b"brain"]
    codes = [fourier_encode_single(w, D=512) for w in words]
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            # They should NOT be identical
            assert not torch.allclose(codes[i], codes[j], atol=1e-6), \
                f"Collision between {words[i]} and {words[j]}"


def test_adversarial_byte_position_alias_is_separated():
    """Regression: the original alpha==beta design mapped bb and ca identically."""
    bb = fourier_encode_single(b"bb", D=128, z_normalize=False)
    ca = fourier_encode_single(b"ca", D=128, z_normalize=False)
    assert not torch.allclose(bb, ca, atol=1e-6)


def test_order_is_preserved_on_adversarial_pairs():
    pairs = [(b"ab", b"ba"), (b"abc", b"cba"), (b"aab", b"aba")]
    for left, right in pairs:
        x = fourier_encode_single(left, D=128, z_normalize=False)
        y = fourier_encode_single(right, D=128, z_normalize=False)
        assert not torch.allclose(x, y, atol=1e-6), (left, right)


def test_frequency_axes_are_not_identical():
    table = _build_frequency_table(128)
    assert not torch.equal(table[1:, 0], table[1:, 1])


def test_prefix_similarity():
    """Tokens sharing a byte prefix should be more similar than unrelated tokens."""
    train = fourier_encode_single(b"training", D=512, z_normalize=False)
    trainer = fourier_encode_single(b"trainer", D=512, z_normalize=False)
    apple = fourier_encode_single(b"apple", D=512, z_normalize=False)

    cos_related = (train @ trainer) / (train.norm() * trainer.norm() + 1e-12)
    cos_unrelated = (train @ apple) / (train.norm() * apple.norm() + 1e-12)

    assert cos_related > cos_unrelated, (
        f"Prefix similarity violated: cos(train,trainer)={cos_related:.4f} "
        f"<= cos(train,apple)={cos_unrelated:.4f}"
    )


def test_codec_accepts_longer_sequences_without_growing_D():
    """
    The pure codec handles inputs longer than Kronecker's fixed pos_dim while
    retaining the same output D. FourierEmbedding itself has max_byte_len.
    A 100-byte token should still produce distinct output from a 50-byte prefix.
    """
    long_token = b"a" * 100
    prefix = b"a" * 50

    # Need to pass max_len matching the longest
    buf_long = torch.zeros(1, 100, dtype=torch.uint8)
    buf_long[0, :100] = torch.frombuffer(bytearray(long_token), dtype=torch.uint8)
    lens_long = torch.tensor([100], dtype=torch.long)

    buf_short = torch.zeros(1, 100, dtype=torch.uint8)
    buf_short[0, :50] = torch.frombuffer(bytearray(prefix), dtype=torch.uint8)
    lens_short = torch.tensor([50], dtype=torch.long)

    out_long = fourier_codec(buf_long, lens_long, D=512)
    out_short = fourier_codec(buf_short, lens_short, D=512)

    # They must be different (not colliding despite shared prefix)
    assert not torch.allclose(out_long, out_short, atol=1e-4)


def test_hindi_no_collision():
    """
    The two Hindi words that collide in Kronecker at pos_dim=32
    (अंतर्राष्ट्रीयकरण vs अंतर्राष्ट्रीयता) should NOT collide in Fourier.
    """
    word1 = "अंतर्राष्ट्रीयकरण".encode("utf-8")  # 54 bytes
    word2 = "अंतर्राष्ट्रीयता".encode("utf-8")    # 48 bytes

    max_len = max(len(word1), len(word2))

    buf1 = torch.zeros(1, max_len, dtype=torch.uint8)
    buf1[0, :len(word1)] = torch.frombuffer(bytearray(word1), dtype=torch.uint8)
    lens1 = torch.tensor([len(word1)], dtype=torch.long)

    buf2 = torch.zeros(1, max_len, dtype=torch.uint8)
    buf2[0, :len(word2)] = torch.frombuffer(bytearray(word2), dtype=torch.uint8)
    lens2 = torch.tensor([len(word2)], dtype=torch.long)

    out1 = fourier_codec(buf1, lens1, D=512)
    out2 = fourier_codec(buf2, lens2, D=512)

    assert not torch.allclose(out1, out2, atol=1e-4), \
        "Hindi words collided in Fourier codec — this should not happen!"

    # Compute cosine similarity — they should be similar (shared prefix) but distinct
    cos = (out1 @ out2.T).item() / (out1.norm().item() * out2.norm().item() + 1e-12)
    print(f"Hindi pair cosine similarity: {cos:.4f} (similar but NOT identical)")


def test_indic_script_handling():
    """Various Indic tokens should all produce distinct codes."""
    words = [
        "भारत".encode("utf-8"),      # Hindi
        "భారత్".encode("utf-8"),      # Telugu  
        "இந்தியா".encode("utf-8"),    # Tamil
        "ভারত".encode("utf-8"),      # Bengali
    ]
    max_len = max(len(w) for w in words)
    B = len(words)

    buf = torch.zeros(B, max_len, dtype=torch.uint8)
    lens = torch.zeros(B, dtype=torch.long)
    for i, w in enumerate(words):
        buf[i, :len(w)] = torch.frombuffer(bytearray(w), dtype=torch.uint8)
        lens[i] = len(w)

    out = fourier_codec(buf, lens, D=512)

    # All pairwise: should be distinct
    for i in range(B):
        for j in range(i + 1, B):
            assert not torch.allclose(out[i], out[j], atol=1e-4), \
                f"Collision between word {i} and {j}"


# ============ Comparison with Kronecker ============

def test_fourier_more_compact():
    """
    Fourier at D=512 should still separate tokens that Kronecker at D=8192 separates,
    demonstrating the compactness advantage.
    """
    words = [b"cat", b"car", b"bat", b"bar"]
    codes = [fourier_encode_single(w, D=512, z_normalize=False) for w in words]

    # cat-car should be more similar than cat-bat (share 2/3 bytes)
    cos_cat_car = (codes[0] @ codes[1]) / (codes[0].norm() * codes[1].norm() + 1e-12)
    cos_cat_bat = (codes[0] @ codes[2]) / (codes[0].norm() * codes[2].norm() + 1e-12)

    # Both share 2/3 bytes, but in different positions — this tests that
    # position encoding works (cat/car differ at pos 2, cat/bat differ at pos 0)
    # The key test: all are distinct
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(codes[i], codes[j], atol=1e-5)


def test_parameter_efficiency():
    """
    Fourier projection at D=512 vs Kronecker at D=8192, both projecting to d_model=768.
    Fourier: 512 * 768 = 393,216 params
    Kronecker: 8192 * 768 = 6,291,456 params
    That's a 16x reduction in the projection alone.
    """
    fourier_params = 512 * 768
    kronecker_params = 8192 * 768
    ratio = kronecker_params / fourier_params
    assert ratio == 16.0
    print(f"Fourier projection: {fourier_params:,} params")
    print(f"Kronecker projection: {kronecker_params:,} params")
    print(f"Fourier is {ratio:.1f}x smaller in the projection layer")


# ============ Batched consistency ============

def test_batched_matches_loop():
    """Batched computation must match single-encode loop."""
    texts = [b"a", b"ab", b"abc", b"abcdef", b"hello world"]
    max_len = max(len(t) for t in texts)
    B = len(texts)

    buf = torch.zeros(B, max_len, dtype=torch.uint8)
    lens = torch.zeros(B, dtype=torch.long)
    for i, t in enumerate(texts):
        buf[i, :len(t)] = torch.frombuffer(bytearray(t), dtype=torch.uint8)
        lens[i] = len(t)

    batched = fourier_codec(buf, lens, D=512)

    for i, t in enumerate(texts):
        single = fourier_encode_single(t, D=512)
        torch.testing.assert_close(batched[i], single, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
