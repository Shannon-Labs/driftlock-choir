"""Minimal MAC slot scheduling helpers for two-way handshakes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MacSlots:
    """Symmetric two-slot exchange description."""

    preamble_len: int
    narrowband_len: int
    guard_us: float
    asymmetric: bool = False

    def total_symbols(self) -> int:
        factor = 2 if not self.asymmetric else 1
        return factor * (self.preamble_len + self.narrowband_len)

    def guard_seconds(self) -> float:
        return self.guard_us * 1e-6
