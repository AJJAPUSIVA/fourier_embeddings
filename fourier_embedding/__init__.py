"""
fourier_embedding — A Fourier-wave alternative to the Kronecker byte codec.

Each byte is represented as a sinusoidal waveform rather than a one-hot vector.
Bytes superpose into a fixed-D output. The codec dimension is independent of
the configured tokenizer byte-buffer bound.
"""

from .codec import (
    fourier_codec,
    fourier_encode_single,
    fourier_output_dim,
)
from .embedding import FourierEmbedding

__all__ = [
    "FourierEmbedding",
    "fourier_codec",
    "fourier_encode_single",
    "fourier_output_dim",
]
