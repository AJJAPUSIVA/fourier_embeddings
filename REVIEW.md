# Review record

## Critical issue found

The original prototype generated identical `alpha` and `beta` frequency
sequences and used the same denominator for byte and position. Its phase was
therefore a function of `byte + position`. Because token codes add per-position
waves, distinct strings such as `bb` and `ca` had exactly equal codes.

## Corrections

- Replaced the shared log-spaced axis with independent deterministic 2-D DFT
  frequency pairs.
- Added a DC component and explicit input validation.
- Added adversarial order and shifted-value regression tests.
- Changed the tokenizer-backed default buffer bound from 64 to 256 bytes.
- Removed claims of universal injectivity, losslessness, unlimited length, and
  unmeasured language-model parity.
- Corrected the web demonstration to use the same frequencies as Python.
- Corrected the training script's causal attention mask and final partial-batch
  handling.
- Renamed duplicate-full-byte counts so they are not reported as measured
  Fourier collisions.
- Added a dependency-free mathematical reference test and requirements file.

## Validation performed here

- Python syntax compilation passed for every Python file.
- Four dependency-free reference tests passed, including the `bb`/`ca`
  regression and three byte-order adversarial pairs.

## Validation still required

PyTorch, Transformers, pytest, and datasets were not installed in the review
environment, so the PyTorch unit suite and transformer training experiment were
not executed here. Install `requirements.txt`, run both commands from the
README, and retain the resulting logs/JSON before claiming model-quality
parity. A credible final report should use at least three random seeds and a
dimension sweep.
