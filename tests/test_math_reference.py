"""Dependency-free reference checks for the Fourier phase design.

These tests do not replace the PyTorch suite. They make the central aliasing
regression runnable with only the Python standard library.
"""

import cmath
import math
import unittest


def frequency_pairs(k_count: int):
    pairs = []
    for k in range(k_count):
        if k == 0:
            pairs.append((0, 0))
        else:
            pairs.append(((73 * k + 19) % 257, (151 * k + 37) % 4099))
    return pairs


def encode(raw: bytes, k_count: int = 64):
    if not raw:
        return tuple(0j for _ in range(k_count))
    scale = 1.0 / math.sqrt(len(raw))
    return tuple(
        scale
        * sum(
            cmath.exp(2j * math.pi * (alpha * byte / 257 + beta * pos / 4099))
            for pos, byte in enumerate(raw)
        )
        for alpha, beta in frequency_pairs(k_count)
    )


def max_distance(left, right):
    return max(abs(a - b) for a, b in zip(left, right))


class FourierReferenceTests(unittest.TestCase):
    def assertSeparated(self, left: bytes, right: bytes):
        self.assertGreater(max_distance(encode(left), encode(right)), 1e-8)

    def test_deterministic(self):
        self.assertEqual(encode(b"hello"), encode(b"hello"))

    def test_original_shift_alias_is_fixed(self):
        self.assertSeparated(b"bb", b"ca")

    def test_order_pairs(self):
        for left, right in [(b"ab", b"ba"), (b"abc", b"cba"), (b"aab", b"aba")]:
            with self.subTest(left=left, right=right):
                self.assertSeparated(left, right)

    def test_long_shared_prefix(self):
        self.assertSeparated(b"a" * 50, b"a" * 50 + b"b")


if __name__ == "__main__":
    unittest.main()
