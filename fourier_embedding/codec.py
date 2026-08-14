"""
Fourier codec — wave-based byte encoding as an alternative to Kronecker products.

=== The Idea ===

Kronecker codec:  c_{byte} ⊗ p_{position}  →  sparse one-hot in R^{256 × pos_dim}
Fourier codec:    wave(byte, position)       →  dense sinusoidal in R^{D}

For a token whose UTF-8 bytes are b_1, ..., b_L:

    φ(b) = (1/√L) · Σ_{p=1..L}  wave(b_p, p)

where wave(v, p) is a D-dimensional vector:

    wave(v, p)[2k]   = sin(2π · v / 256 · α_k  +  2π · p / P_max · β_k)
    wave(v, p)[2k+1] = cos(2π · v / 256 · α_k  +  2π · p / P_max · β_k)

α_k and β_k are fixed, independent samples on the byte and position
frequency axes, making each pair a sampled coefficient of a 2-D DFT.

=== Why This Works ===

1. DIMENSION DECOUPLED FROM WINDOW: Position is a phase, not a one-hot index.
   A 100-byte and a 3-byte token use the same D, although callers still choose
   a finite storage/compute bound for batching.

2. COMPACT D: Information distributes across all dimensions via superposition.
   D=512 is a hypothesis to test, not a guarantee of equivalent information.

3. DETERMINISTIC: Same bytes → same output. No learned parameters in the codec.

4. ORTHOGRAPHIC SIMILARITY PRESERVED: Tokens sharing byte prefixes share the
   first terms of their sum, so they remain geometrically close.

5. TESTABLE COLLISION BEHAVIOUR: A compact set of 2-D Fourier samples is not
   mathematically injective for arbitrary-length strings.  The implementation
   therefore makes no collision-free claim; it measures exact and near
   collisions on the target vocabulary and adversarial strings.

=== Frequency Design ===

The frequency pairs are deterministic samples of a 2-D discrete Fourier
transform over (byte value, byte position).  Byte and position frequencies
must be independent.  Reusing the same sequence for both axes makes the phase
depend only on ``byte + position`` and creates trivial collisions (``bb`` and
``ca`` in the original prototype).

This is analogous to how the original Transformer positional encoding uses
geometric frequency spacing, but here we encode BOTH byte value AND position
into a joint sinusoidal representation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


def _build_frequency_table(D: int, device: torch.device = None) -> Tensor:
    """
    Build the fixed frequency table for the Fourier codec.
    
    Returns shape (D//2, 2): columns are [alpha_k, beta_k] —
    the byte-value frequency and the position frequency for each pair of dims.
    """
    if D <= 0 or D % 2:
        raise ValueError(f"D must be a positive even integer; got {D}")
    half_D = D // 2
    k = torch.arange(half_D, device=device, dtype=torch.long)

    # Deterministic, independent walks through the two frequency axes.  Prime
    # moduli avoid short shared cycles.  Pair zero is the DC component, which
    # retains token-length information before optional normalization.
    alpha = ((73 * k + 19) % 257).to(torch.float32)
    beta = ((151 * k + 37) % 4099).to(torch.float32)
    alpha[0] = 0.0
    beta[0] = 0.0
    return torch.stack([alpha, beta], dim=-1)  # (half_D, 2)


def fourier_wave(
    byte_values: Tensor,
    positions: Tensor,
    freq_table: Tensor,
    P_max: float = 4099.0,
) -> Tensor:
    """
    Compute the wave vector for each (byte_value, position) pair.
    
    Parameters
    ----------
    byte_values : Tensor[long] of shape (B, L)
        Byte values (0-255).
    positions : Tensor[long] of shape (B, L)
        Byte positions (0-indexed).
    freq_table : Tensor of shape (D//2, 2)
        Fixed frequency table [alpha_k, beta_k].
    P_max : float
        Normalization constant for position (max expected token length).
    
    Returns
    -------
    Tensor of shape (B, L, D) — the wave vector for each byte at each position.
    """
    B, L = byte_values.shape
    half_D = freq_table.size(0)

    if P_max <= 0:
        raise ValueError(f"P_max must be positive; got {P_max}")
    # 2-D DFT coordinates.  257 is deliberately larger than the byte alphabet;
    # P_max is a positional period, not a claim of unlimited injectivity.
    v_norm = byte_values.float() / 257.0  # (B, L)
    p_norm = positions.float() / P_max    # (B, L)

    alpha = freq_table[:, 0]  # (half_D,)
    beta = freq_table[:, 1]   # (half_D,)

    # Phase: 2π * v * α_k + 2π * p * β_k
    # Shape: (B, L, half_D)
    phase = (
        2 * math.pi * v_norm.unsqueeze(-1) * alpha.unsqueeze(0).unsqueeze(0)
        + 2 * math.pi * p_norm.unsqueeze(-1) * beta.unsqueeze(0).unsqueeze(0)
    )

    # Interleave sin and cos -> (B, L, D)
    sin_part = torch.sin(phase)
    cos_part = torch.cos(phase)
    out = torch.stack([sin_part, cos_part], dim=-1)  # (B, L, half_D, 2)
    return out.reshape(B, L, half_D * 2)


def fourier_codec(
    byte_sequences: Tensor,
    lengths: Tensor,
    D: int = 512,
    length_normalize: bool = True,
    z_normalize: bool = True,
    eps: float = 1e-6,
    P_max: float = 4099.0,
    freq_table: Optional[Tensor] = None,
    frequency_chunk_size: int = 16,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """
    Compute the Fourier codec for a batch of byte sequences.
    
    φ(b) = (1/√L) · Σ_{p=1..L} wave(b_p, p)
    
    Parameters
    ----------
    byte_sequences : Tensor[uint8 or long] of shape (B, max_len)
        Padded byte sequences.
    lengths : Tensor[int16 or long] of shape (B,)
        Number of valid bytes per token.
    D : int, default 512
        Output dimension. Must be even.
    length_normalize : bool, default True
        Divide by sqrt(L).
    z_normalize : bool, default True
        Per-token z-normalization.
    eps : float, default 1e-6
        Numerical stabilizer.
    P_max : float, default 256.0
        Maximum expected byte length for position normalization.
    freq_table : Tensor, optional
        Pre-built frequency table. If None, built on the fly.
    frequency_chunk_size : int, default 16
        Number of complex frequency pairs evaluated at once. Bounds the
        temporary phase tensor to ``B * active_len * frequency_chunk_size``
        elements instead of materializing ``B * max_len * D`` wave values.
    out_dtype : torch.dtype, optional
        Cast final result to this dtype.
    
    Returns
    -------
    Tensor of shape (B, D).
    """
    if D <= 0 or D % 2 != 0:
        raise ValueError(f"D must be a positive even integer; got {D}")
    if frequency_chunk_size <= 0:
        raise ValueError(
            f"frequency_chunk_size must be positive; got {frequency_chunk_size}"
        )
    if byte_sequences.dim() != 2:
        raise ValueError(
            f"byte_sequences must be 2D (B, max_len); got {tuple(byte_sequences.shape)}"
        )

    device = byte_sequences.device
    B, max_len = byte_sequences.shape

    lens_long = lengths.to(torch.long).to(device)
    if lengths.dim() != 1 or lengths.numel() != B:
        raise ValueError(f"lengths must have shape ({B},); got {tuple(lengths.shape)}")
    if torch.any(lens_long < 0) or torch.any(lens_long > max_len):
        raise ValueError("each length must satisfy 0 <= length <= byte_sequences.size(1)")

    if freq_table is None:
        freq_table = _build_frequency_table(D, device=device)
    freq_table = freq_table.to(device)
    if freq_table.shape != (D // 2, 2):
        raise ValueError(
            f"freq_table must have shape ({D // 2}, 2); got {tuple(freq_table.shape)}"
        )

    # Ignore padded columns beyond the longest real sequence in this batch.
    # This is especially important for tokenizer buffers configured at 256
    # bytes when typical tokens contain only a handful of bytes.
    active_len = int(lens_long.max().item()) if B else 0
    bytes_float = byte_sequences[:, :active_len].to(torch.float32) / 257.0

    positions = torch.arange(active_len, device=device, dtype=torch.float32)
    positions = positions.unsqueeze(0).expand(B, -1)
    positions = positions / P_max
    valid = (
        torch.arange(active_len, device=device).unsqueeze(0)
        < lens_long.unsqueeze(1)
    )

    # Accumulate sampled Fourier coefficients in bounded frequency chunks.
    # ``out`` is the only B x D allocation. The largest temporary is
    # B x active_len x frequency_chunk_size rather than B x max_len x D.
    out = torch.empty((B, D), device=device, dtype=torch.float32)
    half_D = D // 2
    byte_term = bytes_float.unsqueeze(-1)
    position_term = positions.unsqueeze(-1)
    mask = valid.unsqueeze(-1)

    for start in range(0, half_D, frequency_chunk_size):
        stop = min(start + frequency_chunk_size, half_D)
        alpha = freq_table[start:stop, 0]
        beta = freq_table[start:stop, 1]
        # Keep the same fp32 operation order as ``fourier_wave``. Factoring
        # out 2*pi is algebraically equivalent but changes rounding at high
        # frequencies and breaks output-equivalence guarantees.
        phase = (
            2 * math.pi * byte_term * alpha.view(1, 1, -1)
            + 2 * math.pi * position_term * beta.view(1, 1, -1)
        )
        # Mask after sin/cos: padded byte-buffer values are not signal, and
        # cos(0)=1 would otherwise contribute at padded positions.
        sin_sum = torch.sin(phase).masked_fill(~mask, 0.0).sum(dim=1)
        cos_sum = torch.cos(phase).masked_fill(~mask, 0.0).sum(dim=1)
        out[:, 2 * start : 2 * stop : 2] = sin_sum
        out[:, 2 * start + 1 : 2 * stop : 2] = cos_sum

    # Length normalize: divide by sqrt(L)
    if length_normalize:
        scales = torch.rsqrt(lens_long.clamp_min(1).float())  # (B,)
        out = out * scales.unsqueeze(1)

    # Z-normalize
    if z_normalize:
        mean = out.mean(dim=-1, keepdim=True)
        std = (out - mean).std(dim=-1, keepdim=True) + eps
        out = (out - mean) / std

    if out_dtype is not None:
        out = out.to(out_dtype)
    return out


def fourier_encode_single(
    byte_seq: bytes,
    D: int = 512,
    length_normalize: bool = True,
    z_normalize: bool = True,
    eps: float = 1e-6,
    P_max: float = 4099.0,
) -> Tensor:
    """Convenience: encode a single bytes object."""
    L = len(byte_seq)
    if L == 0:
        # Return zeros that z-norm will handle gracefully
        buf = torch.zeros(1, 1, dtype=torch.uint8)
        lens = torch.tensor([0], dtype=torch.long)
        return fourier_codec(buf, lens, D=D, length_normalize=length_normalize,
                             z_normalize=z_normalize, eps=eps, P_max=P_max).squeeze(0)

    buf = torch.zeros(1, L, dtype=torch.uint8)
    buf[0, :L] = torch.frombuffer(bytearray(byte_seq), dtype=torch.uint8)
    lens = torch.tensor([L], dtype=torch.long)
    return fourier_codec(
        buf, lens, D=D, length_normalize=length_normalize,
        z_normalize=z_normalize, eps=eps, P_max=P_max
    ).squeeze(0)


def fourier_output_dim(D: int = 512) -> int:
    """Return the output dimension. Trivial but matches the Kronecker API."""
    return D
