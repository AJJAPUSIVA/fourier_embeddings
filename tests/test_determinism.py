import torch
from torch import nn

from experiments.determinism import (
    derived_seed,
    epoch_permutation,
    initialize_matched_model,
)


class TinyModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, 8)
        self.body = nn.Linear(8, 8)
        self.head = nn.Linear(8, 11, bias=False)


def test_seed_derivation_is_stable_and_namespaced():
    assert derived_seed(1337, "body") == derived_seed(1337, "body")
    assert derived_seed(1337, "body") != derived_seed(1337, "head")


def test_matched_initialization_shares_body_but_not_embedding():
    dense = TinyModel(13)
    fourier = TinyModel(17)
    initialize_matched_model(dense, 2027, "dense")
    initialize_matched_model(fourier, 2027, "fourier")

    torch.testing.assert_close(dense.body.weight, fourier.body.weight, rtol=0, atol=0)
    torch.testing.assert_close(dense.head.weight, fourier.head.weight, rtol=0, atol=0)
    assert not torch.equal(dense.tok_emb.weight[:13], fourier.tok_emb.weight[:13])


def test_epoch_order_repeats_for_arm_and_changes_by_epoch():
    first = epoch_permutation(100, 3407, 0)
    repeated = epoch_permutation(100, 3407, 0)
    second_epoch = epoch_permutation(100, 3407, 1)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, second_epoch)
