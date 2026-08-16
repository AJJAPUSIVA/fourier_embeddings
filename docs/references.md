# Scientific references

This bibliography separates prior scientific foundations from evidence produced
by this repository. The cited papers motivate Fourier features, positional
encoding, byte-level modeling, and embedding compression; none proves that this
repository's finite deterministic codec is collision-free.

## Fourier features and positional encoding

1. Ali Rahimi and Benjamin Recht. 2007. “Random Features for Large-Scale
   Kernel Machines.” *Advances in Neural Information Processing Systems 20*.
   [Paper](https://proceedings.neurips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)

   Introduces finite random sinusoidal feature maps for approximating
   shift-invariant kernels. It motivates compact Fourier maps, but its kernel
   guarantees do not directly establish injectivity of this token codec.

2. Dougal J. Sutherland and Jeff Schneider. 2015. “On the Error of Random
   Fourier Features.” *Proceedings of UAI 2015*.
   [Preprint](https://arxiv.org/abs/1506.02785)

   Analyzes finite-feature approximation error and motivates reporting results
   across Fourier dimensions rather than treating one dimension as universal.

3. Ashish Vaswani et al. 2017. “Attention Is All You Need.” *Advances in
   Neural Information Processing Systems 30*.
   [Preprint](https://arxiv.org/abs/1706.03762)

   Establishes sinusoidal position encodings as a practical way to supply
   sequence order to attention models.

4. Matthew Tancik et al. 2020. “Fourier Features Let Networks Learn High
   Frequency Functions in Low Dimensional Domains.” *NeurIPS 2020*.
   [Preprint](https://arxiv.org/abs/2006.10739)

   Shows that mapping coordinates through Fourier features can change the
   effective neural tangent kernel and improve learning of high-frequency
   functions. This supports Fourier features as neural inputs, not a
   collision-free claim.

5. Yang Li, Si Si, Gang Li, Cho-Jui Hsieh, and Samy Bengio. 2021. “Learnable
   Fourier Features for Multi-Dimensional Spatial Positional Encoding.”
   *NeurIPS 2021*.
   [Preprint](https://arxiv.org/abs/2106.02795)

   Directly motivates treating value and position as separate coordinates on
   independent frequency axes, although its application and learned mapping
   differ from this deterministic discrete codec.

## Byte-level language modeling

6. Linting Xue et al. 2022. “ByT5: Towards a Token-Free Future with
   Pre-trained Byte-to-Byte Models.” *Transactions of the Association for
   Computational Linguistics*, 10:291–306.
   [Paper](https://aclanthology.org/2022.tacl-1.17/)

   Demonstrates the practical viability of UTF-8 byte inputs. ByT5 processes
   bytes as sequence elements; this project instead aggregates a token's
   ordered bytes into one fixed-dimensional codec vector.

## Compact and compositional embeddings

7. Dan Svenstrup, Jonas Meinertz Hansen, and Ole Winther. 2017. “Hash
   Embeddings for Efficient Word Representations.” *NeurIPS 2017*.
   [Paper](https://proceedings.neurips.cc/paper_files/paper/2017/hash/f0f6ba4b5e0000340312d33c212c3ae8-Abstract.html)

   Provides a learned, parameter-efficient baseline based on hashing into a
   shared embedding pool.

8. Raphael Shu and Hideki Nakayama. 2018. “Compressing Word Embeddings via
   Deep Compositional Code Learning.” *ICLR 2018*.
   [Paper](https://openreview.net/forum?id=BJRZzFlRb)

   Represents vocabulary items with learned discrete codes composed from a
   small set of basis vectors.

9. Sho Takase and Sosuke Kobayashi. 2020. “All Word Embeddings from One
   Embedding.” *NeurIPS 2020*.
   [Paper](https://proceedings.neurips.cc/paper_files/paper/2020/hash/275d7fb2fd45098ad5c3ece2ed4a2824-Abstract.html)

   Generates vocabulary embeddings by transforming a shared embedding and
   supplies a strong comparison point for vocabulary-independent parameter
   counts.

10. Ting Chen et al. 2020. “Differentiable Product Quantization for End-to-End
    Embedding Compression.” *ICML 2020*.
    [Paper](https://proceedings.mlr.press/v119/chen20l.html)

    Learns compact discrete codes for embedding tables and provides an
    end-to-end learned compression baseline.

11. Maximilian Nickel, Lorenzo Rosasco, and Tomaso Poggio. 2016. “Holographic
    Embeddings of Knowledge Graphs.” *AAAI 2016*.
    [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/10314)

    Uses circular correlation for compressed compositional representations.
    It is relevant background for frequency-domain composition, but addresses
    relational knowledge graphs rather than ordered token bytes.

## Evidence produced by this repository

The following claims come from the committed experiment artifacts, not from
the papers above:

- the exact `bb`/`ca` shared-axis alias and its regression test;
- zero measured exact or tested quantized collision groups over 50,257 GPT-2
  token IDs for the evaluated codec dimensions;
- the deterministic three-seed WikiText-2 comparison;
- the 16× projection-parameter reduction and 2.73% aggregate-mean validation
  perplexity regression for Fourier-512 relative to the tested Kronecker arm.

See [`../results/benchmark_results.md`](../results/benchmark_results.md) and
the machine-readable JSON files in [`../results/`](../results/).
