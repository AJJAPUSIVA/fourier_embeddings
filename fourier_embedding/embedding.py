"""
FourierEmbedding: drop-in replacement for nn.Embedding using the Fourier codec.

Architecture::

    input_ids -> byte_buffer[input_ids] -> fourier_codec(...) -> Linear(D, d_model) -> output

Same contract as KroneckerEmbedding and nn.Embedding:
    forward(input_ids: Tensor[..., L]) -> Tensor[..., L, d_model]
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .codec import _build_frequency_table, fourier_codec

from kronecker_embeddings.tokenizer_utils import build_byte_buffer


class FourierEmbedding(nn.Module):
    """
    Drop-in replacement for nn.Embedding using a Fourier wave codec.

    Parameters
    ----------
    vocab_size : int
        Number of tokens.
    d_model : int
        Output dimension (transformer hidden size).
    tokenizer : optional
        HF tokenizer for building byte buffers.
    D : int, default 512
        Fourier codec output dimension. Can be much smaller than Kronecker's
        D=8192 because information is dense (superposed waves) not sparse (one-hots).
    max_byte_len : int, default 256
        Storage and compute bound for tokenizer byte buffers. Tokens beyond this
        value are UTF-8-safely truncated. The Fourier *dimension* does not grow
        with this value, but the implementation is intentionally honest that a
        finite buffer still has a configured bound (Problem 3 is separate).
    mode : {"dynamic", "cached"}
        Same semantics as KroneckerEmbedding.
    projection_init : {"normal", "xavier"}
        Init scheme for the projection.
    length_normalize : bool, default True
        Scale by 1/sqrt(L).
    z_normalize : bool, default True
        Per-token z-normalization.
    frequency_chunk_size : int, default 16
        Complex frequency pairs evaluated simultaneously by the codec.
    codec_batch_size : int, default 1024
        Number of vocabulary rows encoded at once when building cached mode.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        tokenizer=None,
        D: int = 512,
        max_byte_len: int = 256,
        mode: Literal["dynamic", "cached"] = "dynamic",
        byte_buffer: Optional[Tensor] = None,
        length_buffer: Optional[Tensor] = None,
        projection_init: Literal["normal", "xavier"] = "normal",
        length_normalize: bool = True,
        z_normalize: bool = True,
        frequency_chunk_size: int = 16,
        codec_batch_size: int = 1024,
    ):
        super().__init__()
        if D % 2 != 0:
            raise ValueError(f"D must be even; got {D}")
        if mode not in ("dynamic", "cached"):
            raise ValueError(f"mode must be 'dynamic' or 'cached'; got {mode!r}")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.D = D
        self.max_byte_len = max_byte_len
        self.mode = mode
        self.length_normalize = length_normalize
        self.z_normalize = z_normalize
        self.frequency_chunk_size = frequency_chunk_size
        self.codec_batch_size = codec_batch_size

        if max_byte_len <= 0:
            raise ValueError(f"max_byte_len must be positive; got {max_byte_len}")
        if frequency_chunk_size <= 0:
            raise ValueError("frequency_chunk_size must be positive")
        if codec_batch_size <= 0:
            raise ValueError("codec_batch_size must be positive")

        # Build byte buffer if not provided.
        if byte_buffer is None or length_buffer is None:
            if tokenizer is None:
                raise ValueError(
                    "Must pass either `tokenizer` or both `byte_buffer` and `length_buffer`."
                )
            bb, lb = build_byte_buffer(tokenizer, pos_dim=max_byte_len)
            byte_buffer = bb
            length_buffer = lb

        # Validate shapes.
        if byte_buffer.shape != (vocab_size, max_byte_len):
            raise ValueError(
                f"byte_buffer shape mismatch: expected ({vocab_size}, {max_byte_len}), "
                f"got {tuple(byte_buffer.shape)}"
            )
        if length_buffer.shape != (vocab_size,):
            raise ValueError(
                f"length_buffer shape mismatch: expected ({vocab_size},), "
                f"got {tuple(length_buffer.shape)}"
            )

        self.register_buffer("_byte_buffer", byte_buffer.to(torch.uint8), persistent=False)
        self.register_buffer("_length_buffer", length_buffer.to(torch.int16), persistent=False)

        # Pre-build frequency table (fixed, not learned)
        freq_table = _build_frequency_table(D)
        self.register_buffer("_freq_table", freq_table, persistent=False)

        # Trainable projection D -> d_model
        self.projection = nn.Linear(self.D, d_model, bias=False)
        if projection_init == "normal":
            nn.init.normal_(self.projection.weight, mean=0.0, std=1.0 / math.sqrt(self.D))
        elif projection_init == "xavier":
            nn.init.xavier_uniform_(self.projection.weight)
        else:
            raise ValueError(f"projection_init must be 'normal' or 'xavier'; got {projection_init!r}")

        # Cached mode: precompute the table in bounded row batches. Building
        # V x max_byte_len x D at once can otherwise exceed accelerator RAM.
        if mode == "cached":
            with torch.no_grad():
                chunks = []
                for start in range(0, self.vocab_size, self.codec_batch_size):
                    stop = min(start + self.codec_batch_size, self.vocab_size)
                    chunks.append(self._encode_rows(
                        self._byte_buffer[start:stop],
                        self._length_buffer[start:stop],
                    ))
                table = torch.cat(chunks, dim=0)
            self.register_buffer("_codec_table", table, persistent=False)

    @property
    def num_embeddings(self) -> int:
        return self.vocab_size

    @property
    def embedding_dim(self) -> int:
        return self.d_model

    def _encode_rows(self, byte_rows: Tensor, lengths: Tensor) -> Tensor:
        """Encode rows after trimming unused padded byte columns."""
        active_len = int(lengths.max().item()) if lengths.numel() else 0
        return fourier_codec(
            byte_rows[:, :active_len],
            lengths,
            D=self.D,
            length_normalize=self.length_normalize,
            z_normalize=self.z_normalize,
            freq_table=self._freq_table,
            frequency_chunk_size=self.frequency_chunk_size,
        )

    def _codec_lookup(self, input_ids: Tensor) -> Tensor:
        """Return (..., L, D) codec output for input_ids of shape (..., L)."""
        flat_ids = input_ids.reshape(-1)
        if self.mode == "cached":
            codec_out = self._codec_table.index_select(0, flat_ids)
        else:
            # Repeated token IDs are common in language batches. Encode each
            # distinct ID once, then gather occurrences back into sequence
            # order. This preserves exact output while reducing codec work.
            unique_ids, inverse = torch.unique(
                flat_ids, sorted=False, return_inverse=True
            )
            bytes_unique = self._byte_buffer.index_select(0, unique_ids)
            lens_unique = self._length_buffer.index_select(0, unique_ids)
            unique_codes = self._encode_rows(bytes_unique, lens_unique)
            codec_out = unique_codes.index_select(0, inverse)
        return codec_out.view(*input_ids.shape, self.D)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Map input_ids (..., L) -> embeddings (..., L, d_model).
        Same contract as nn.Embedding.forward.
        """
        codec_out = self._codec_lookup(input_ids)
        return self.projection(codec_out.to(self.projection.weight.dtype))

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, d_model={self.d_model}, "
            f"D={self.D}, max_byte_len={self.max_byte_len}, mode={self.mode!r}, "
            f"frequency_chunk_size={self.frequency_chunk_size}"
        )
