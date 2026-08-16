# Fourier Embeddings: a tested alternative to the Kronecker codec

**ERA V5 Session 7 — Problem 4:** Can each character be represented as a
Fourier wave and the waves added to form a word?

## Result-based verdict

The committed benchmark verdict is **PASS**, under two explicit committed criteria.
This is empirical evidence for the tested configuration, not a universal proof.

| Decision criterion | Required | Measured | Verdict |
|:---|---:|---:|:---:|
| Fourier validation-PPL regression vs Kronecker | ≤ 3.00% | 2.73% | **PASS** |
| Embedding-parameter reduction vs Kronecker | ≥ 16.00× | 16.00× | **PASS** |

The result comes from nine deterministic WikiText-2 runs: three embedding
families × three matched seeds. See
[`results/benchmark_results.md`](results/benchmark_results.md) for the complete
measurements, collision analysis, plots, and interpretation limits.

The construction requires byte and position on **independent frequency
axes**. The first prototype used identical
byte and position frequencies. With the original normalization its phase
depended only on `byte + position`, making `bb` and `ca` exactly identical.
This revision fixes that alias and includes it as a regression test.

The revised codec is a compact, deterministic sample of the two-dimensional
discrete Fourier transform (2-D DFT) of a byte-position event grid. It is a
compressed measurement whose collision behavior must be tested empirically,
not a proof of collision-free encoding.

| Design aspect | Kronecker (`pos_dim=32`) | Fourier (`D=512`) | Practical consequence |
|:---|:---|:---|:---|
| Codec dimension | 8,192 | 512 | Fourier projection is 16× smaller |
| Projection shape | `8192 × d_model` | `512 × d_model` | 2,097,152 vs 131,072 parameters at `d_model=256` |
| Vocabulary-dependent parameters | None | None | Both extend without a learned vocabulary table |
| Byte-position representation | Complete sparse grid | Sampled dense 2-D Fourier features | Fourier exchanges completeness for compression |
| Token order | Encoded explicitly | Encoded through the position-frequency axis | Fourier passed adversarial order tests |
| Collision statement | Exact for retained bytes; tokens are truncated at 32 bytes | No universal guarantee | Fourier had 0 measured exact/quantized groups over 50,257 GPT-2 IDs |
| Configured byte bound | 32 | 256 in this experiment | Fourier distinguishes more long-token bytes, but remains bounded |
| Mean validation PPL | 563.59 ± 7.36 | 578.97 ± 3.32 | Fourier regression was 2.73%, inside the 3% criterion |

The table separates architectural facts from measured outcomes. The PASS
verdict means that Fourier-512 met the stated quality and compression thresholds
in this experiment; it does not mean the codec is collision-free for arbitrary
strings or superior on every language-model task.

## Scientific foundation

The construction is motivated by Random Fourier Features, sinusoidal and
multidimensional positional encoding, byte-level language modeling, and prior
embedding-compression methods. Those papers motivate the design; the benchmark
and collision artifacts in this repository provide evidence for this particular
codec.

- [`docs/theory.md`](docs/theory.md) derives the implementation-matched codec,
  proves the original shared-axis alias, and states the claim boundaries.
- [`docs/references.md`](docs/references.md) gives the primary-paper bibliography
  and explains how each source relates to the project.
- [`results/benchmark_results.md`](results/benchmark_results.md) reports the
  committed empirical measurements and limitations.

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
- Performance outside the committed WikiText-2 configuration. The checked-in
  run passes its stated quality threshold; it does not establish universal
  language-model parity.

## Repository layout

```text
.
├── pyproject.toml
├── README.md
├── requirements.txt
├── docs/
│   ├── theory.md         # implementation-matched derivation and limits
│   └── references.md     # annotated primary-paper bibliography
├── fourier_embedding/
│   ├── __init__.py
│   ├── codec.py          # deterministic 2-D Fourier feature codec
│   └── embedding.py      # nn.Embedding-compatible wrapper
├── tests/
│   ├── test_codec.py     # PyTorch and adversarial regression tests
│   └── test_math_reference.py
├── experiments/
│   ├── train_proof.py    # matched tiny-transformer comparison
│   ├── analysis.py       # codec analysis
│   ├── collision_probe.py # vocabulary-scale collision report
│   └── plot_benchmarks.py # reproducible dependency-free SVG plots
├── results/
│   ├── training_results.json
│   ├── collision_results.json
│   ├── benchmark_results.md
│   └── plots/
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

Run exact and near-collision analysis:

```bash
python experiments/collision_probe.py \
  --tokenizer gpt2 \
  --dimensions 128 256 512 \
  --near-max-tokens 10000
```

Without local Python, open **Actions → Fourier collision analysis → Run
workflow**. The workflow uploads `collision_results.json` and
`collision_summary.md`. Collision groups exclude duplicate tokenizer entries
that have identical byte strings; near-collision statistics compare only
distinct byte sequences.
Exact and quantized collision counts cover the full requested vocabulary. The
more expensive cosine search is exact within the deterministic prefix selected
by `--near-max-tokens`; increase that value to cover the complete vocabulary.

The training proof is intentionally run as matched single-arm jobs. For a
local one-arm smoke run:

```bash
python experiments/train_proof.py \
  --embedding fourier \
  --seed 1337 \
  --dataset synthetic \
  --max-tokens 50000 \
  --max-steps 100 \
  --output training-fourier-1337.json
```

Without local Python, open **Actions → Deterministic matched training → Run
workflow**. Run the `smoke` profile first. It executes Dense, Kronecker, and
Fourier with seed 1337. After it passes, run `proof`, which uses seeds 1337,
2027, and 3407 on WikiText-2. Each arm receives identical shared-model
initialization, data order, effective batch size, optimizer settings, and step
count. A shared parameter-free token normalization gives every transformer arm
comparable input scale; raw and normalized RMS are recorded in the report. The
workflow uploads individual JSON records plus an aggregate Markdown
table containing mean ± standard deviation.

The proof profile uses the canonical `Salesforce/wikitext` repository,
configuration `wikitext-2-raw-v1`, at a pinned dataset revision. This avoids
the deprecated unnamespaced `wikitext` alias and records dataset provenance in
every result.

Do not enable `allow_nondeterministic` for final assignment evidence.

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
seed ∈ {1337, 2027, 3407}
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
one. Under the committed thresholds, the result is **PASS**: 2.73% mean PPL
regression versus the 3.00% limit, 16× embedding reduction, and zero measured
exact or quantized collisions in the GPT-2 vocabulary. These are bounded
experimental results, not unconditional claims about injectivity, speed, or
all language-model tasks.
