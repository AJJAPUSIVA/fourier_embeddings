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

## Validation completed remotely

- The GitHub Actions unit suite passed with the project dependencies installed.
- Collision analysis covered 50,257 GPT-2 token IDs at D=128/256/512.
- Matched WikiText-2 training completed for three embeddings and three seeds.
- Result-based criteria and limitations are committed in
  `results/benchmark_results.md`; raw aggregates are in
  `results/training_results.json`.

The local review environment still did not execute PyTorch training. The
committed results came from GitHub Actions and should be interpreted only under
their recorded configuration and thresholds.
