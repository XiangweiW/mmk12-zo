"""Stable seeds. Python's builtin hash is not used."""

from __future__ import annotations

import hashlib

MASTER_SEED = 42


def mix_seed(*parts: int) -> int:
    x = 0
    for part in parts:
        x = (x + (int(part) & 0xFFFFFFFFFFFFFFFF) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 31
    return x & 0x7FFFFFFFFFFFFFFF


def batch_seed(step: int, master: int = MASTER_SEED) -> int:
    return mix_seed(master, 1, step)


def direction_seed(step: int, index: int, master: int = MASTER_SEED) -> int:
    return mix_seed(master, 2, step, index)


def tensor_seed(direction: int, name: str) -> int:
    digest = hashlib.sha256(f"{direction}\n{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFFFFFFFFFF
