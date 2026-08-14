import torch
import sys
from types import SimpleNamespace

from experiments.determinism import initialize_matched_model
from experiments.train_proof import (
    MiniLM,
    ModelConfig,
    WIKITEXT_CONFIG,
    WIKITEXT_DATASET_ID,
    WIKITEXT_REVISION,
    embedding_scale_stats,
    load_token_ids,
)


class RecordingTokenizer:
    def __init__(self):
        self.kwargs = None

    def encode(self, text, **kwargs):
        self.kwargs = kwargs
        return list(range(kwargs["max_length"]))


def test_synthetic_tokenization_truncates_during_encode():
    tokenizer = RecordingTokenizer()
    token_ids = load_token_ids(tokenizer, "synthetic", 123)
    assert len(token_ids) == 123
    assert tokenizer.kwargs == {"truncation": True, "max_length": 123}


def test_wikitext_uses_namespaced_pinned_dataset(monkeypatch):
    captured = {}

    def fake_load_dataset(dataset_id, config, **kwargs):
        captured.update(dataset_id=dataset_id, config=config, **kwargs)
        return {"text": ["first", "", "second"]}

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    tokenizer = RecordingTokenizer()
    load_token_ids(tokenizer, "wikitext", 20)
    assert captured == {
        "dataset_id": WIKITEXT_DATASET_ID,
        "config": WIKITEXT_CONFIG,
        "split": "train",
        "revision": WIKITEXT_REVISION,
    }


def test_dense_embeddings_are_normalized_before_transformer():
    cfg = ModelConfig(vocab_size=31, d_model=16, n_heads=4, n_layers=1, d_ff=32, max_seq_len=8)
    model = MiniLM(cfg, "dense", tokenizer=None, fourier_dim=32, max_byte_len=16)
    initialize_matched_model(model, 1337, "dense")
    stats = embedding_scale_stats(model, cfg.vocab_size, torch.device("cpu"), sample_size=31)
    assert stats["raw_rms"] < 0.5
    assert abs(stats["normalized_rms"] - 1.0) < 0.01

    ids = torch.arange(8).reshape(1, 8)
    normalized = model.tok_norm(model.tok_emb(ids))
    torch.testing.assert_close(
        normalized.square().mean(dim=-1),
        torch.ones((1, 8)),
        atol=1e-3,
        rtol=1e-3,
    )
