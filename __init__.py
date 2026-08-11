"""
fourier_embedding — A Fourier-wave alternative to the Kronecker byte codec.

Each byte is represented as a sinusoidal waveform rather than a one-hot vector.
Bytes superpose (sum) into a fixed-D output regardless of token length, with no
pos_dim ceiling and no wasted dimensions for short tokens.
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
