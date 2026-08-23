# DETERMINIZE_NOTES — Phase 5 W2 (2026-08-23)

Fixes Phase 4 MCTS's clairvoyance: `BalatroGame.clone()` copies the keyed RNG
(`run_state.rng`) and the draw-pile *order* verbatim, so every simulation saw the true
future (next draws, reroll results, pack contents, boss/tag/voucher rolls, Lucky/
Bloodstone-style probability rolls). `real1` (106 gens, mean ante ~5.5) was trained and
evaluated entirely under that cheat.

## 0. Files

| file | what |
|---|---|
| `mp/engine/balatro_sim/game.py` | `BalatroGame.clone_determinized(seed)` + private helper `_local_shuffle` |
| `mp/engine/balatro_sim/mlb_match.py` | `MLBMatch.clone_determinized(seed)` |
| `mp/engine/tests/engine_tests/test_determinize.py` | 33 invariant tests (engine-level) |
| `mp/agent/mcts/determinize.py` | `DeterminizedMCTSPlayer`, `make_determinized_player`, `make_determinizing_view`, `seed_stream` |
| `mp/agent/tests/test_determinize_player.py` | 24 tests (agent-level wrapper) |
| `mp/agent/scripts/measure_clairvoyance.py` | the gate-1 measurement (outcome + disagreement tables) |
| `mp/results/clairvoyance_smoke.{json,md}` | 2-seed, sims=8 smoke run (pipeline proof, NOT the gate-1 number — see §5) |
| `mp/results/clairvoyance_2026-08-23.{json,md}` | **not yet produced** — the real 30-seed/sims=40 run; command in §6 |

## 1. Semantics of `clone_determinized`

```python
BalatroGame.clone_determinized(self, seed: int | str | None = None) -> "BalatroGame"
MLBMatch.clone_determinized(self, seed=None) -> "MLBMatch"     # both games, SAME resolved seed
```

A `clone()` that keeps everything the player has **observed** bit-identical (hand,
discard pile, jokers + their state, consumables, dollars, current shop shelf incl.
prices/sold flags, booster choices on screen, this ante's boss, blind tags on offer,
vouchers, planet levels, lives/pvp fields, tag state, every counter, and the *whole*
generation-layer bookkeeping in `run_state` — `used_jokers`, `bosses_used`,
`banned_keys`, `shop_joker_max`, `key_scope`, pools, etc. — none of that is "the
future", it's either on screen or an already-materialised consequence of past draws),
then resamples the two things that ARE the future:

* **(b) the draw pile is reshuffled uniformly** — `deck` gets a fresh Fisher-Yates
  order via `_local_shuffle` (§2), a LOCAL, throwaway generator that never touches
  `run_state.rng`. Same `Card` objects (references into `full_deck`, exactly as
  `clone()` aliases them by id) — order only.
* **(c) `run_state.rng` is replaced** with a brand-new `PseudoRandom` on a fresh seed
  string, so every FUTURE keyed draw — shop rerolls, pack contents on open, next
  antes' bosses/tags/vouchers, future `'nr<ante>'` shuffles, Lucky/Bloodstone/etc.
  probability rolls — is decorrelated from the true run.

`seed=None` → `secrets`-random (fresh, non-reproducible, matching `BalatroGame(seed=
None)`); a given `seed` makes the reshuffle AND every future draw fully reproducible.
`game.seed_str` / `deck_key` / `stake` / `ruleset` are left **unchanged** on the clone —
they're observation features the player has always known, not part of the resample.
`run_state.seed` (the string the keyed RNG actually hashes with) tracks the *new* seed;
`game.seed_str` does not. `det.determinized: bool` and `det.det_seed` (the resolved
`seed` argument) are set on the result so a caller can tell — but only on the immediate
result: `clone()` is unmodified (out of scope — W2 owns only `clone_determinized`), and
it hand-copies a fixed attribute list rather than doing a generic `vars(self)` copy, so
a *further* plain `.clone()` of a determinized game silently drops both flags. This is a
known, accepted limitation of the "add attributes without touching `clone()`" ownership
boundary — a caller that clones a determinized game further should treat the result as
still-determinized by construction; `getattr(g, "determinized", False)` will read
`False` on it.

`MLBMatch.clone_determinized`: both games get the **same** resolved seed (resolved
once, before either `BalatroGame.clone_determinized` call) — real MLB shares one seed
between both players (`different_seeds = false`), and that correlation is real-game
behaviour that must survive determinization: the sampled world is "what if the run had
started on this other seed", not two players independently guessing different worlds.
Mirrors `MLBMatch.__init__`'s own `g1 = BalatroGame(seed=g0.seed_str, ...)` pattern.

## 2. The per-key-state decision: reset, not carried

`run_state.rng` becomes a **brand-new `PseudoRandom(new_seed_str)`** with an **empty**
per-key state table (`PseudoRandom._state = {}`), not the true game's per-key state
carried forward under a new seed label.

Reasoning: a per-key entry in `PseudoRandom._state` is `pseudohash(key + seed)` LCG-
stepped some number of times — a function of the (key, seed) pair. Carrying the TRUE
run's per-key values into a `PseudoRandom` object that now *claims* a different `seed`
would be internally inconsistent (those numbers were never `pseudohash(key +
new_seed)`-derived), not merely "still a little predictive of the future" — it would be
values that don't correspond to ANY real seed's chain at all. A fresh, empty table is
exactly what a real run that had actually started on `new_seed_str` would have at this
point in wall-clock time: every key gets computed fresh on first use, from the new
seed, in the same call order the real engine already uses everywhere else. This also
keeps the mod's queue semantics sane (Multiplayer's `key_scope`/resample counters read
generation-layer state that's independent of `PseudoRandom._state` — see `run_state`
fields in §1 — so resetting the per-key table doesn't touch any of that bookkeeping).

## 3. `_local_shuffle`: why not `PseudoRandom.pseudoshuffle`

First implementation reused `PseudoRandom.pseudoshuffle` (the bit-exact LuaJIT port
already used for every real game shuffle) on a second, throwaway `PseudoRandom`
instance — clean, reused validated code, zero new logic. Benchmarked **~1.75-2.0x**
`clone()` cost — over the brief's ≤1.5x budget. Root cause: `pseudoshuffle`'s Fisher-
Yates draws ~51 `LuaJITRandom.random()` calls (the TW223 combined-generator port), and
each is a genuinely expensive pure-Python computation — about 1.2 μs/call, ~60 μs for a
52-card shuffle, comparable to `clone()`'s own cost.

Replaced with `_local_shuffle` (private helper, module-level next to
`_fast_clone_run_state`): a splitmix64 stream seeded off a `hashlib.blake2b` digest of
the seed string. ~4x fewer cycles per draw (a few 64-bit multiplies/xors vs. LuaJIT's
combined generator), ~17 μs for the same 52-card shuffle. `hashlib` (not `random`) for
the string→int seed because `state_signature()`'s own comment explains why: Python's
built-in `hash()` of a string is salted per-process, so it isn't reproducible across
runs/processes — `blake2b` is. No stdlib `random` module import anywhere
(`test_no_random_module_in_engine` forbids it inside `balatro_sim/`, and the first draft
tripped this — `random.Random` is not a documented exception, and there is no allowlist
mechanism in that guard). `j = z % (i + 1)` has a negligible modulo bias for `i <= 51`
against a 64-bit `z` — not measurable at the pile sizes this method runs on, and the
chi-square sanity test (`test_deck_shuffle_is_approximately_uniform`) confirms it.

## 4. Benchmarks (this box, single process, no other load)

| state | `clone()` | `clone_determinized()` | ratio |
|---|---|---|---|
| fresh BLIND_SELECT | 77.3 μs | 105.6 μs | 1.37x |
| SHOP w/ 3 jokers + 2 consumables | 95.4 μs | 119.0 μs | 1.25x |
| MLB Nemesis-ready (ante 8) | 77.0 μs | 104.0 μs | 1.35x |

All under the ≤1.5x budget. The fixed overhead (~28 μs: a `clone()`, one `PseudoRandom`
construction for `run_state.rng`, one `_local_shuffle` call, two dict-attribute writes)
is roughly constant across state richness, so richer states (bigger jokers/shop lists,
which cost more in `clone()` itself) get a *better* ratio — the states MCTS actually
searches from mid-game are closer to the SHOP row than the bare fresh-game row.

## 5. Measurement method (`mp/agent/scripts/measure_clairvoyance.py`)

Both arms use `real1.sh`'s Stage B search hyperparameters exactly (`encoder=set`,
`strategy=gumbel`, `heuristic_prior=0.4`, `heuristic_tau=0.35`, `max_hand_candidates=32`,
`reuse=True`) loaded via `mcts.player.make_player` against `mp/agent/runs/real1/latest.pt`
— the ONLY variable between arms is what the search's clones can see.

* **Clairvoyant arm**: plain `MCTSPlayer` (unmodified).
* **Determinized arm**: `mcts.determinize.DeterminizedMCTSPlayer`, default
  `mode="per_sim"` (PIMC-style — every simulation gets an independently resampled
  world; see `mcts/determinize.py`'s module docstring for exactly how this is wired
  without touching `search.py`/`player.py`: the search's only per-simulation clone call
  is `root_game.clone()` in `MCTS._run_sims_iter`, so a `game.clone()` whose bound
  `.clone` method is shadowed, INSTANCE-ONLY, to call `clone_determinized` is
  sufficient). `mode="per_search"` (one determinized world per whole decision) is also
  implemented and exposed via `--determinize-mode`.
* **(a) Outcome table**: both players play an INDEPENDENT full vanilla game (max ante
  8, i.e. to natural `GAME_OVER`) on the same seed; paired-by-seed bootstrap CI
  (`mp/eval/common.py::paired_bootstrap_ci`, imported read-only) on final ante, ante-
  1/2/3 clear rate, blinds cleared, final $, mean s/decision.
* **(b) Disagreement table**: drive the CLAIRVOYANT trajectory (its own actual
  choices); at every REAL decision (>1 legal action — forced/single-legal states are
  skipped, nothing to disagree about) also ask a FRESH determinized player (own
  `make_determinized_player` call, same `sims`, no retained tree) what it would choose
  from that exact true state. Agreement rate broken down by `action["type"]` (`sell_joker`
  bucketed as `sell`; `advance`/`skip_booster`/`reroll_boss` bucketed as `other`).
* Parallelized over seeds with `multiprocessing.Pool` (one worker per seed: both
  outcome games + the whole disagreement walk, so the checkpoint load amortises).
  Each worker pins `torch.set_num_threads(1)` (`eval_checkpoint.py`'s documented
  finding: an unpinned single-leaf net actively hurts a box running many workers).

**Tree reuse and the determinized arm**: left ON (`reuse=True`, matching `real1.sh`) for
both arms rather than force-disabled. Reasoning: `ReuseConfig`'s default
`budget_mode="subtract"` counts retained visits toward `num_simulations`, so reuse only
buys wall-clock, never changes evidence-per-decision — there is no fairness confound to
guard against. Under the determinized arm, `TreeCache.store()` (which ALSO calls
`root_game.clone()`, hitting the same instance shadow in `mode="per_sim"`) computes a
next-state signature from a resampled world; that signature essentially never matches
the true next game's signature next decision (full `state_signature()` includes the RNG
digest and deck order), so `TreeCache.take()` reliably misses. Confirmed harmless (a
"miss" just means a fresh search, not a wrong one — false-positive reuse would need an
astronomically unlikely hash coincidence) and reported honestly via
`player.inner.reuse_stats` rather than hidden by disabling reuse.

## 6. Results

**`mp/results/clairvoyance_smoke.{json,md}`** — 2 seeds (`11111111`, `1558AXDL`),
`sims=8` (NOT the trained 40 — reduced purely for a fast pipeline check),
`--processes 2`, `determinize_mode=per_sim`, `n_boot=200`. Wall clock: 157s for the
pool (i.e. ~157s for the slower of the 2 parallel single-seed workers — each seed does
2 full games + a ~19-decision disagreement walk with a fresh determinized-player build
per probe). **This run exists ONLY to prove the pipeline (`clone_determinized` →
`DeterminizedMCTSPlayer` → outcome/disagreement tables → bootstrap CI → JSON/MD) works
end-to-end — the numbers are not meaningful evidence about real1's clairvoyance** (8
sims is far too few for real decisions; both players ended around ante 1-2 vs. real1's
trained mean ante ~5.5). For the curious: outcome diff (clairvoyant − determinized) was
+0.5 ante / +1.5 blinds cleared over these 2 seeds; the disagreement walk saw 38 real
decisions, overall agreement 0.34, with `play`/`discard` (hand decisions, the most
sims-hungry) at 0% agreement and `skip_blind`/`leave_shop`/`reroll` at 100% — the
expected shape at trivially low sims (hand EV needs many rollouts to stabilize;
skip/leave/reroll decisions are far coarser), not evidence of anything about the real
gate-1 question.

**The real 30-seed / sims=40 measurement has NOT been run.** Per the lead's explicit
instruction (2026-08-23, mid-session — Tagg was actively using the machine), only the
2-seed/sims=8/2-process smoke above was run; the full job is left for the lead to
launch when the box is idle. `mp/results/clairvoyance_2026-08-23.{json,md}` do not yet
exist.

**Wall-clock projection (rough — extrapolated from the sims=8 smoke, treat as a planning
number, not a promise)**: one seed at sims=8 took ~157s (2 full games + disagreement
walk, ~19 real decisions). Scaling sims 8→40 (~5x per-decision search cost) and
allowing for longer, deeper games at real search strength (real1's trained mean ante
~5.5 vs. this smoke's ~1.5, plausibly ~2-3x more real decisions per game) suggests
roughly 10-15x per seed → very roughly **25-40 minutes per seed** if run serially. With
30 seeds and `--processes 30` (32 cores, one seed per process — this workload is
CPU-bound single-leaf inference, not GPU, so 1:1 seed:process is reasonable), wall clock
for the WHOLE job is dominated by the slowest single seed, i.e. also **~25-40 min** —
right at the brief's ~40 min budget. If a run at `--processes 30 --sims 40` overshoots
that, the brief's own fallback applies: drop to `--sims 25` or `--n-seeds 20` and say so
in the results doc. Recommended first attempt (§7) uses the full `sims=40`/30 seeds/30
processes and a hard wall-clock watch; fall back only if it's clearly over budget.

## 7. Exact command for the lead to run later

```
python mp/agent/scripts/measure_clairvoyance.py \
    --checkpoint mp/agent/runs/real1/latest.pt \
    --n-seeds 30 --sims 40 --processes 30 --determinize-mode per_sim \
    --determinize-seed-base 0 --max-steps 20000 --n-boot 2000 \
    --out-json mp/results/clairvoyance_2026-08-23.json \
    --out-md   mp/results/clairvoyance_2026-08-23.md
```

`--determinize-seed-base 0` makes the whole run reproducible (rerun = identical numbers);
pass `--determinize-seed-base -1` for fresh (`secrets`) world-sampling randomness instead.
If wall clock needs trimming: `--sims 25` (still well above the heuristic-prior floor) or
`--processes` capped lower to leave the box usable for other work. The script prints
total wall clock and writes both the `.json` (full per-seed + per-probe records, for
follow-up analysis) and the `.md` (the two tables + a stub "Interpretation" section — fill
that in by hand after reading the numbers, the script does not editorialize).

## 8. Open issues / for the lead

1. `clone_determinized`'s `determinized`/`det_seed` flags don't survive a subsequent
   plain `.clone()` (§1) — fine for W2's own use (the wrapper only ever calls
   `clone_determinized` directly, never chains through a plain clone), but worth knowing
   if W3/W5's rollout code ever clones a determinized game further and checks the flag.
2. The clairvoyance measurement script is new infrastructure under
   `mp/agent/scripts/`, not explicitly named in the brief's ownership table — flagging in
   case another workstream also wants a script there; no filename collision as of this
   writing.
3. Real `clairvoyance_2026-08-23.{json,md}` still need the actual 30-seed run (§6/§7)
   before gate 1 can be called done — the "Interpretation" section in the `.md` is a stub
   until that data exists.
