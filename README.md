# Fourier Embeddings: a tested alternative to the Kronecker codec

**ERA V5 Session 7 — Problem 4:** Can each character be represented as a
Fourier wave and the waves added to form a word?

## Result

Yes, with an important qualification: the wave must encode **byte and
position on independent frequency axes**. The first prototype used identical
byte and position frequencies. With the original normalization its phase
depended only on `byte + position`, making `bb` and `ca` exactly identical.
This revision fixes that alias and includes it as a regression test.

The revised codec is a compact, deterministic sample of the two-dimensional
discrete Fourier transform (2-D DFT) of a byte-position event grid. It is a
probabilistic/empirical compression, not a proof of collision-free encoding.

| Property | Kronecker | Revised Fourier |
|---|---:|---:|
| Default codec dimension | 8,192 | 512 (sweepable) |
| Projection parameters | `8192 × d_model` | `D × d_model` |
| Vocabulary-dependent parameters | No | No |
| Byte-position basis | Complete sparse grid | Sampled dense 2-D Fourier features |
| Order-sensitive | Yes | Yes, tested adversarially |
| Collision guarantee | Exact before truncation | No universal guarantee; measure it |
| Configured input bound | `pos_dim` | `max_byte_len` buffer bound |

At `D=512`, the Fourier projection is 16× smaller than a Kronecker projection
with `pos_dim=32`. Whether that compression preserves language-model quality
is an experimental question addressed by `train_proof.py`.

## Codec

For UTF-8 bytes `b_0, ..., b_(L-1)`, define one complex feature for each
frequency pair `(α_k, β_k)`:

```text
Z_k(token) = (1/√L) Σ_p exp(i 2π [α_k b_p / 257 + β_k p / 4099])
```

The real codec concatenates real and imaginary parts:

```text
φ(token) = [Im Z_0, Re Z_0, ..., Im Z_(K-1), Re Z_(K-1)]
D = 2K
```

The first pair is the DC component `(0, 0)`. Remaining pairs follow two
different deterministic modular walks:

```text
α_k = (73k + 19) mod 257
β_k = (151k + 37) mod 4099
```

Using different walks is essential. If `α_k == β_k` and both axes use the
same denominator, `(byte, position)` collapses to `byte + position`.

The codec is then projected into model width:

```text
embedding(token) = Linear(D, d_model)(φ(token))
```

Only the projection is trained. The frequencies are fixed and reproducible.

### Bounded-memory execution

The implementation does not materialize the full
`[input_tokens, max_byte_len, D]` wave tensor. It:

1. encodes each distinct token ID once per dynamic lookup;
2. trims byte buffers to the longest actual token in that lookup;
3. evaluates complex frequencies in configurable chunks; and
4. builds cached vocabulary tables in bounded row batches.

With `frequency_chunk_size=C`, the largest phase temporary is approximately
`unique_tokens × active_byte_length × C` fp32 values, rather than
`input_tokens × max_byte_len × D`. The output codes remain mathematically
equivalent to the unchunked formulation; regression tests compare both paths.

## What is and is not proved

### Established directly

- Determinism: identical bytes produce identical codes.
- Order sensitivity on adversarial pairs such as `ab/ba`, `abc/cba`, and
  `aab/aba`.
- Regression separation of `bb/ca`, which collided in the first prototype.
- Fixed codec dimension for different input lengths.
- Vocabulary-independent trainable parameter count.
- Batched and single-item implementations agree.

### Must be established empirically

- Collision and near-collision rates on the target vocabulary.
- Preservation of orthographic neighbourhoods.
- Language-model validation quality relative to dense and Kronecker inputs.
- Training throughput and memory, because `sin`/`cos` computation is denser
  than Kronecker's scatter operation.

### Not claimed

- Universal injectivity for arbitrary-length strings. A finite-dimensional
  additive code cannot provide that guarantee over an unbounded domain.
- Unlimited token length in `FourierEmbedding`. The pure codec accepts any
  supplied tensor length, but the tokenizer-backed module uses an explicit
  `max_byte_len` storage/compute bound (default 256). Dynamic storage is
  Problem 3, not this submission.
- Invertibility from the compressed Fourier vector.
- Comparable perplexity until `training_results.json` is produced by an
  actual matched run.

## Repository layout

```text
.
├── pyproject.toml
├── README.md
├── requirements.txt
├── fourier_embedding/
│   ├── __init__.py
│   ├── codec.py          # deterministic 2-D Fourier feature codec
│   └── embedding.py      # nn.Embedding-compatible wrapper
├── tests/
│   ├── test_codec.py     # PyTorch and adversarial regression tests
│   └── test_math_reference.py
├── experiments/
│   ├── train_proof.py    # matched tiny-transformer comparison
│   └── analysis.py       # codec analysis
└── index.html                 # interactive explanation
```

## Usage

```python
from transformers import AutoTokenizer
from fourier_embedding import FourierEmbedding

tok = AutoTokenizer.from_pretrained("gpt2")
emb = FourierEmbedding(
    vocab_size=tok.vocab_size,
    d_model=768,
    tokenizer=tok,
    D=512,
    max_byte_len=256,
)

ids = tok("hello world", return_tensors="pt").input_ids
out = emb(ids)  # (1, sequence_length, 768)
```

## Validation

Install dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run codec tests:

```bash
python -m pytest -q
```

Run lightweight analysis:

```bash
python experiments/analysis.py
```

Run the matched training experiment:

```bash
python experiments/train_proof.py
```

The training comparison keeps the tokenizer, corpus, transformer shape,
optimizer, steps, and output head fixed. The experimental arms are:

1. `nn.Embedding(V, d_model)`
2. `KroneckerEmbedding(pos_dim=32)`
3. `FourierEmbedding(D=512)`

Report validation loss/perplexity, embedding and total parameters, wall-clock
time, and codec geometry. Run multiple seeds before making a quality claim.

## Required experiment improvements

For a strong submission, sweep:

```text
D ∈ {64, 128, 256, 512, 1024}
seed ∈ {1, 2, 3}
```

Measure both exact and near collisions. Exact floating-point equality alone is
too weak; include nearest-neighbour cosine margin and quantized-code collision
counts. Add deliberately difficult sets:

- permutations (`ab/ba`);
- shifted-value aliases (`bb/ca`);
- repeated bytes;
- common prefixes and suffixes;
- Indic UTF-8 tokens;
- long synthetic tokens.

## Honest conclusion

The revised method is a real Fourier alternative: it represents the
byte-position occupancy signal through sampled 2-D Fourier coefficients and
adds those complex waves to form a token code. It offers a large parameter
reduction, but that reduction trades a complete Kronecker basis for a sampled
one. The submission succeeds only if collision geometry and matched training
results show that the sampled basis retains enough information.
