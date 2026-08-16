# Deterministic Fourier dimension sweep

**DESCRIPTIVE ONLY:** no post-hoc PASS/FAIL threshold is assigned.

Kronecker validation PPL: 563.59 ± 7.36

| Fourier D | Validation PPL | Mean paired regression | 95% paired CI | Embedding params | Reduction vs Kronecker | Tokens/s |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 574.89 ± 8.73 | +2.01% | [-1.73%, 5.75%] | 32,768 | 64.0× | 1188.0 ± 53.0 |
| 256 | 565.23 ± 10.39 | +0.29% | [-1.08%, 1.65%] | 65,536 | 32.0× | 1332.1 ± 280.2 |
| 512 | 578.97 ± 3.32 | +2.74% | [0.46%, 5.01%] | 131,072 | 16.0× | 1224.4 ± 133.1 |
| 1024 | 568.01 ± 8.56 | +0.78% | [-0.75%, 2.31%] | 262,144 | 8.0× | 1354.1 ± 364.0 |
