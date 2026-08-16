# Theory and claim boundaries

This document states the Fourier codec implemented in
[`fourier_embedding/codec.py`](../fourier_embedding/codec.py), relates it to
prior work, and identifies which conclusions are mathematical facts versus
bounded empirical findings.

## 1. Byte-position event grid

Let a non-empty UTF-8 byte string be

\[
x=(b_0,b_1,\ldots,b_{L-1}), \qquad b_p\in\{0,\ldots,255\}.
\]

The implementation embeds byte values in a periodic byte axis of size 257 and
positions in a periodic axis of size 4099. Define the sparse event grid

\[
X_x(b,p)=\sum_{q=0}^{L-1}\mathbf 1[b=b_q]\mathbf 1[p=q],
\]

for \(b\in\mathbb Z_{257}\) and \(p\in\mathbb Z_{4099}\). Byte coordinate 256
is unused by valid UTF-8 bytes, and experiments impose a smaller finite storage
bound (`max_byte_len=256`) before the codec is evaluated.

Byte-level language modeling has precedent in ByT5 [6], but ByT5 passes bytes
as separate sequence elements. This codec instead maps all retained bytes of a
token to one vector.

## 2. Complete two-dimensional transform

Using the code's positive-phase convention, define the complete 2-D DFT

\[
\widehat X_x(u,v)=
\sum_{b=0}^{256}\sum_{p=0}^{4098}
X_x(b,p)\exp\left(2\pi i\left(\frac{ub}{257}+\frac{vp}{4099}\right)\right).
\]

Since the event grid is nonzero only at \((b_p,p)\),

\[
\widehat X_x(u,v)=
\sum_{p=0}^{L-1}
\exp\left(2\pi i\left(\frac{u b_p}{257}+\frac{v p}{4099}\right)\right).
\]

The complete transform is invertible by the ordinary inverse DFT. Consequently,
the full set of \(257\times4099\) complex coefficients determines the bounded
event grid exactly. Parseval's identity likewise preserves inner products and
distances up to the transform's normalization convention.

These are facts about the **complete transform before codec normalization**.
They are not injectivity results for the 512-dimensional implementation.

## 3. Implemented sampled codec

For output dimension \(D=2K\), the implementation constructs deterministic
frequency pairs

\[
\alpha_k=(73k+19)\bmod257,
\qquad
\beta_k=(151k+37)\bmod4099,
\]

and explicitly replaces the first pair with \((\alpha_0,\beta_0)=(0,0)\).
For each pair it computes

\[
Z_k(x)=\sum_{p=0}^{L-1}
\exp\left(2\pi i\left(
\frac{\alpha_k b_p}{257}+\frac{\beta_k p}{4099}
\right)\right).
\]

The real vector uses the code's interleaving order

\[
s(x)=[\operatorname{Im}Z_0,\operatorname{Re}Z_0,\ldots,
\operatorname{Im}Z_{K-1},\operatorname{Re}Z_{K-1}].
\]

With the defaults, the codec next divides non-empty token vectors by \(\sqrt L\)
and applies per-token z-normalization:

\[
t(x)=\frac{s(x)}{\sqrt L},\qquad
\phi(x)=\frac{t(x)-\mu(t(x))}{\sigma(t(x))+\varepsilon}.
\]

Thus Fourier-512 retains only 256 complex measurements, then applies a
non-invertible normalization. Complete-DFT invertibility cannot establish that
\(\phi\) is injective. The zero-collision result in this repository is an
empirical statement over the tested finite vocabulary and precisions.

The conceptual connection to Random Fourier Features [1,2] is the use of a
finite sinusoidal feature map. Classical RFF theory approximates specified
shift-invariant kernels using frequencies sampled from a distribution. This
codec instead uses fixed modular walks on a discrete 2-D grid, so an RFF error
bound must not be quoted as a proof for this construction without separately
establishing its assumptions.

## 4. Why independent axes are required

Suppose a prototype uses the same normalized frequency for byte and position.
Its phase then has the form

\[
\theta_k(b,p)=\omega_k(b+p).
\]

The aggregated representation depends only on the multiset of values \(b_p+p\).
For ASCII `bb`, that multiset is

\[
\{98+0,98+1\}=\{98,99\},
\]

while for ASCII `ca` it is

\[
\{99+0,97+1\}=\{99,98\}.
\]

Because summation is commutative, every feature is equal and the two strings
collide exactly. The revised phase

\[
\theta_k(b,p)=2\pi\left(
\frac{\alpha_k b}{257}+\frac{\beta_k p}{4099}
\right)
\]

uses different coordinate periods and independent modular walks. It removes
this particular forced alias. It does not rule out every possible collision in
a finite sampled representation. Multidimensional Fourier positional features
[5] provide related motivation for representing separate coordinate axes.

## 5. Parameter and memory calculation

With model width \(d_{model}\), the trainable Fourier projection contains

\[
P_F=Dd_{model}
\]

weights, excluding an optional bias. The tested Kronecker projection with 256
byte categories and `pos_dim=P` contains

\[
P_K=(256P)d_{model}.
\]

Therefore the projection reduction factor is

\[
R=\frac{P_K}{P_F}=\frac{256P}{D}.
\]

For \(P=32\), \(D=512\), and \(d_{model}=256\):

\[
P_K=2{,}097{,}152,\qquad P_F=131{,}072,\qquad R=16.
\]

This is 93.75% fewer projection weights. In float32, the raw projection-weight
storage is 8 MiB versus 0.5 MiB. This calculation does not include gradients,
optimizer state, activations, or the rest of the model, and it is not a claim of
16× end-to-end runtime improvement.

Hash Embeddings [7], compositional code learning [8], ALONE [9], and
differentiable product quantization [10] are relevant learned compression
baselines. Holographic Embeddings [11] provide adjacent evidence that
frequency-domain operations can form compact compositional representations.

## 6. Computational complexity

For token length \(L\) and \(K=D/2\) complex frequencies, direct codec
evaluation requires \(O(LD)\) trigonometric feature work. The learned projection
requires \(O(Dd_{model})\) multiply-add work. The implementation evaluates
frequencies in chunks so its largest phase temporary is proportional to
`batch × active_length × frequency_chunk_size`, rather than
`batch × max_length × D`.

Parameter count, peak memory, codec latency, projection latency, and training
throughput are distinct measurements and should be reported separately.

## 7. Supported conclusions

The current repository supports the following bounded statements:

1. A complete 2-D DFT is invertible before sampling and normalization.
2. A shared byte/position phase can cause the proved `bb`/`ca` alias.
3. Independent axes remove that specific structural alias.
4. Fourier-512 uses 16× fewer projection weights than the tested Kronecker
   configuration.
5. Fourier-512 met the committed empirical quality criterion in the completed
   deterministic three-seed WikiText-2 experiment.
6. No exact or tested quantized collision group was observed over the evaluated
   50,257 GPT-2 token IDs.

The repository does **not** establish universal collision freedom, recovery of
arbitrary strings, equal performance across datasets or scales, or 16×
end-to-end speedup.

## References

Numbered citations refer to [`references.md`](references.md).
