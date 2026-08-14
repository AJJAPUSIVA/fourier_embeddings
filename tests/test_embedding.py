"""Tests for bounded-memory FourierEmbedding lookup paths."""

import torch

from fourier_embedding import FourierEmbedding


def _buffers():
    vocab_size, max_len = 6, 256
    byte_buffer = torch.zeros((vocab_size, max_len), dtype=torch.uint8)
    raw = [b"", b"a", b"hello", b"world", b"hello!", b"x" * 80]
    lengths = torch.tensor([len(item) for item in raw], dtype=torch.int16)
    for token_id, item in enumerate(raw):
        if item:
            byte_buffer[token_id, : len(item)] = torch.tensor(list(item), dtype=torch.uint8)
    return byte_buffer, lengths


def test_dynamic_deduplicated_lookup_matches_cached_mode():
    byte_buffer, lengths = _buffers()
    common = dict(
        vocab_size=byte_buffer.size(0),
        d_model=24,
        D=64,
        max_byte_len=byte_buffer.size(1),
        byte_buffer=byte_buffer,
        length_buffer=lengths,
        frequency_chunk_size=3,
        codec_batch_size=2,
    )
    dynamic = FourierEmbedding(**common, mode="dynamic")
    cached = FourierEmbedding(**common, mode="cached")
    cached.projection.load_state_dict(dynamic.projection.state_dict())

    # Repeated IDs exercise unique-ID encoding and inverse gathering.
    ids = torch.tensor([[2, 2, 1, 5, 2], [3, 1, 3, 4, 1]])
    torch.testing.assert_close(dynamic(ids), cached(ids), atol=2e-5, rtol=2e-5)


def test_large_padding_does_not_change_codes():
    byte_buffer, lengths = _buffers()
    emb = FourierEmbedding(
        vocab_size=byte_buffer.size(0),
        d_model=16,
        D=32,
        max_byte_len=byte_buffer.size(1),
        byte_buffer=byte_buffer,
        length_buffer=lengths,
        frequency_chunk_size=2,
    )
    ids = torch.tensor([1, 2, 3, 4])
    result = emb._codec_lookup(ids)
    assert result.shape == (4, 32)
    assert torch.isfinite(result).all()
