import torch

from experiments.collision_probe import (
    collision_groups,
    nearest_distinct_neighbors,
)


def test_exact_collision_excludes_duplicate_byte_strings():
    codes = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    raw = [b"same", b"same", b"different", b"other"]
    report = collision_groups(codes, raw)
    assert report["collision_groups"] == 1
    assert report["tokens_in_collision_groups"] == 3


def test_quantized_collision_detects_close_codes():
    codes = torch.tensor([[0.12341, 1.0], [0.12344, 1.0], [0.5, 0.5]])
    raw = [b"a", b"b", b"c"]
    assert collision_groups(codes, raw)["collision_groups"] == 0
    assert collision_groups(codes, raw, decimals=3)["collision_groups"] == 1


def test_nearest_neighbor_excludes_same_byte_duplicates():
    codes = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ])
    raw = [b"a", b"a", b"b", b"c"]
    values, neighbors = nearest_distinct_neighbors(codes, raw, block_size=2)
    assert neighbors[0].item() == 2
    assert neighbors[1].item() == 2
    assert values[0].item() < 1.0
