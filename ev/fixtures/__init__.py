"""fixtures/ — Tagg-facing state builders for the advisor (Phase 5 rev 2, W6).

``FIXTURES`` maps a name (used as ``fixture:<name>`` in the advisor CLI) to a zero/kwarg
``build() -> MLBMatch`` callable.
"""
from __future__ import annotations

from .bloodstone_vs_invisible import build as bloodstone_vs_invisible

FIXTURES = {
    "bloodstone_vs_invisible": bloodstone_vs_invisible,
}

__all__ = ["FIXTURES", "bloodstone_vs_invisible"]
