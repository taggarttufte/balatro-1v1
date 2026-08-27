"""Engine BEHAVIOUR stamp — a hand-curated marker of gameplay-affecting engine changes.

Purpose (W-FIX 2026-08-26 finding): label/pair shards record the *policy* fingerprint but
nothing distinguishes corpora generated under different ENGINE dynamics — pre/post-fix data
was tellable apart only by date.  This stamp closes that gap: it is recorded into every
label/pair row's meta (``engine_stamp``) at generation time, and any trainer/analysis can
filter or stratify on it.

POLICY: bump the stamp (append a new dated tag) on ANY change that alters gameplay dynamics —
effect behaviour, rules, scoring, RNG consumption on effect paths, protocol semantics.  Do NOT
bump for pure refactors, docs, tests, or generation-layer changes already covered by parity.
Keep the history list append-only; the stamp is the LAST entry.

History:
- 2026-08-25.baseline      state at the repo split (Phases 0-5 + extraction layer)
- 2026-08-26.wpvp          PvP turn protocol added (flag-gated; canonical default identical)
- 2026-08-26.wfix          Blueprint copy-guard (15 jokers), Satellite global ledger,
                           Ice Cream melts+frees slot
- 2026-08-27.wshop-wcycle  shop economy (uncapped EV rerolls, pack-EV takes, Fool ordering)
                           + per-target tarot-dig lines (player-side: changes label POLICY
                           dynamics rather than engine rules, stamped for corpus clarity)
"""

ENGINE_BEHAVIOR_STAMP = "2026-08-27.wshop-wcycle"
