"""ev/encode — the ENCODE-layer proof of concept (W-ENCODE-POC, 2026-08-26).

The analytic EV player's shop knowledge is hand-written and blind in three places
(``ev/EV_NOTES.md`` §8 item 4): scaling jokers show no immediate strength, tarots and
spectrals are unvalued, vouchers are a flat +0.02.  The proposed cure is a *fleet* of LLM
agents that read each item's Lua and emit a verified analytic value function.

This package is the **proof of concept for that loop, not the fleet**:

* ``registry.py`` — ten entries (eight real items + two deliberately-wrong negative
  controls), each with a ``predict(summary) -> float``, a tier, an assumption list, a
  ``file:line`` Lua citation and a ``generated_by`` note.  Written by the LLM that is
  standing in for one fleet worker.
* ``verify.py`` — the empirical harness.  Three measurement modes, all of them
  **marginal** (see ``verify.MARGINAL_RULE``), all of them instrumented for reachability.
* ``run_poc.py`` — the driver that runs every entry and prints the accept/reject table.
* ``POC_NOTES.md`` — results, engine-fidelity findings, and what L1 still needs.

Nothing here is imported by the player.  ``ev/player.py``, ``ev/hand.py``,
``ev/train_v.py`` and every ``runs/`` directory are untouched by design: the POC answers
"does generate -> verify work, and does verification bite?", nothing more.
"""

__all__ = ["registry", "verify"]
