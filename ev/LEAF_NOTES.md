# LEAF_NOTES — W-LEAF: V at the expectimax leaf (Phase 5 rev 2, V2 round, 2026-08-24/25)

Owner: W-LEAF. Implements `mp/docs/PHASE5_V2_BRIEF_2026-08.md` section 4, lever (c). Files
touched: `mp/ev/hand.py` (the K=3x8-world value_fn default, additive), `mp/ev/player.py`
(`EVPlayer.value_fn_leaf_only`, additive), `mp/ev/match_player.py` (one-line passthrough of
the same flag, additive), `mp/ev/h2h.py` (`ev:full+Vleaf` player spec, additive),
`mp/ev/tests/test_{hand,player,h2h}.py` (+7 tests), results
`mp/results/h2h_ev_full_vleaf_vs_ev_full_30seeds.{json,md}`,
`mp/results/h2h_ev_full_vleaf_vs_real1_det_30seeds.{json,md}`. Did not touch `mp/engine/**`,
`mp/agent/**` (besides copying the pre-existing, gitignored `real1` checkpoint into this
worktree so `real1:det` has something to load), `mp/stats/**`.

## 0. TL;DR

```
python mp/ev/h2h.py --a "ev:full+Vleaf" --b "ev:full"    --n-seeds 30 --procs 8 --max-steps 4000 \
    --out-json mp/results/h2h_ev_full_vleaf_vs_ev_full_30seeds.json \
    --out-md   mp/results/h2h_ev_full_vleaf_vs_ev_full_30seeds.md

python mp/ev/h2h.py --a "ev:full+Vleaf" --b "real1:det"  --n-seeds 30 --procs 8 --max-steps 4000 \
    --out-json mp/results/h2h_ev_full_vleaf_vs_real1_det_30seeds.json \
    --out-md   mp/results/h2h_ev_full_vleaf_vs_real1_det_30seeds.md
```

The keeper checkpoint `mp/ev/runs/v_full_best/ckpt_0001000.pt` (gitignored, ~57 MB) had to be
copied into this worktree from the shared checkout before anything below could run — a fresh
`git worktree` does not carry gitignored files. Same for `mp/agent/runs/real1/latest.pt`
(~220 MB), needed by `real1:det`.

## 1. Wiring

### 1.1 `hand.py` — K=3 x 8 worlds when `value_fn` is set (EV_NOTES §8.3)

Two new `HandConfig` fields, `full_top_k_v = 3` / `full_n_worlds_v = 8`, alongside the
existing `full_top_k = 5` / `full_n_worlds = 3`. `rank_hand_actions(budget="full", ...)`
resolves `top_k`/`n_worlds` from the `_v` pair when `value_fn is not None` **and** the caller
didn't pass an explicit override; otherwise (no `value_fn`, or an explicit `top_k`/`n_worlds`)
behaviour is byte-for-byte what it was before this round. `EVPlayer`'s own `n_worlds`/`top_k`
constructor args (default `None`) already flowed through as "explicit override or let hand.py
decide" — no change needed there for this part. Pinned by
`test_hand.py::test_full_budget_defaults_to_k3_x_8_worlds_with_a_value_fn` (counts
`sample_world` calls == 8 and `value_fn` calls == 3×8 = 24),
`::test_full_budget_default_worlds_unchanged_without_a_value_fn` (3, the old default),
`::test_full_budget_explicit_top_k_and_n_worlds_override_the_value_fn_defaults`.

`hand.end_of_blind_value` / the full budget's ROUND_EVAL-advance / GAME_OVER=0 / exception-
propagation semantics were already correct from the 2026-08-23 fix pass (§8b in EV_NOTES) —
nothing to change there; verified still exercised by
`test_full_budget_calls_value_fn_on_post_blind_states` and
`test_full_budget_value_fn_exception_propagates` (pre-existing, still green).

### 1.2 `player.py` — `value_fn_leaf_only` (the actual "full-budget config" work)

**This was the one real design gap, not just wiring.** `EVPlayer.value_fn` (built by W3/W5)
is a single flag that, once set, argmaxes V over candidates at **every** decision type that
has one: the full-budget hand rollout's leaf (`_rank_hand`, lever (c) — what this workstream
is supposed to isolate) **and** `SHOP` / `BOOSTER_OPEN` (`_rank_with_value`) **and**
`BLIND_SELECT`. But `PHASE5_V2_BRIEF_2026-08.md` section 0 already measured, this same round,
that "argmax-V as a policy loses to the rules player 2/60" — per-action EV gaps from a
match-win predictor sit below rollout label noise, so V is a fine *value estimator* but a bad
*policy* over fine-grained candidates. Wiring `value_fn` through the existing plumbing
unmodified would have made `ev:full+Vleaf` re-measure that already-known failure (V driving
shop/blind-select) on top of whatever the leaf lever does — contaminating exactly the
question section 4 asks. (First pilot run confirmed this: A lost every life, final money
$1-2 vs B's $35-43 — a shop-economy collapse, not a hand-decision effect.)

Fix: `EVPlayer(..., value_fn_leaf_only=True)` (default `False`, every existing caller
unaffected). When set, `SHOP`/`BOOSTER_OPEN`/`BLIND_SELECT` skip `_rank_with_value` and fall
through to the rules tier exactly as if `value_fn` were `None`, while `_rank_hand`'s
full-budget threading of `value_fn` into `hand.rank_hand_actions` (the leaf) is completely
unaffected by the flag. `MatchAwareEVPlayer` (`match_player.py`, W5) got a one-line
passthrough of the same flag into its `EVPlayer` construction (needed because `h2h.py`'s
`_one_worker_job` uses `.policy()`, which lazily builds per-seat `EVPlayer`s through this
wrapper, not a directly-constructed `EVPlayer`). `h2h.py`'s new `ev:full+Vleaf` spec passes
`value_fn_leaf_only=True`. Pinned by
`test_player.py::test_value_fn_leaf_only_skips_the_v_tier_at_shop_and_blind_select` (a
value_fn that WOULD flip the SHOP decision is byte-identical to `value_fn=None` when
leaf_only) and `test_h2h.py::test_build_player_ev_vleaf_spec`.

### 1.3 `h2h.py` — the `ev:full+Vleaf` player spec

`build_player`'s `"ev"` branch gained a `"Vleaf"` token (`ev:full+Vleaf`): loads
`match_player.load_value(checkpoint or VLEAF_CKPT_DEFAULT, device="cpu")` (CPU: the ops cap
is on concurrent torch-*loading* processes, not device, and an 8-worker spawn pool contending
for one GPU at batch size 1 buys nothing) and wraps it in `MatchAwareEVPlayer(...,
value_fn_leaf_only=True)`. Returns `obj.policy()` (not `common.adapt_player(obj)` — the
generic adapter has no match to bind V's opponent view from; only `.policy()`'s per-seat
closures call `.refresh()` from the live match before every decision, which is what makes V
opponent-aware at all — see ADVISOR_NOTES.md §1 point 2 for why this distinction exists).
`VLEAF_CKPT_DEFAULT = mp/ev/runs/v_full_best/ckpt_0001000.pt`, overridable with `--checkpoint`
(shared with `real1:*`'s override — fine in practice since no h2h run here needs both
overridden at once).

## 2. Per-decision cost (EV_NOTES §8.3's "verify ≤ 100 ms")

Benchmark: `MatchAwareEVPlayer(budget="full")` (the FULL wiring — value_fn NOT leaf-only in
this particular measurement, so SHOP/BLIND_SELECT timings are included too for completeness)
bound to a live `MLBMatch`, self-play, wall-clock per `.act()` call, `torch.set_num_threads(2)`
(matches TRAINV_NOTES' own convention; threads=1 and the default 16 were both *slightly*
worse, not better — thread count is not the bottleneck). 8 seeds, 229 decisions:

| state | n | mean | p50 | p95 | max |
|---|---|---|---|---|---|
| SELECTING_HAND (the K=3x8 leaf) | 106 | **163.3 ms** | 143.6 ms | 307.1 ms | 422.8 ms |
| SHOP | 51 | 38.1 ms | 45.5 ms | 59.7 ms | 68.1 ms |
| BLIND_SELECT | 48 | 19.3 ms | 21.8 ms | 26.3 ms | 30.5 ms |
| BOOSTER_OPEN | 6 | 10.1 ms | — | — | — |
| ROUND_EVAL | 18 | 0.02 ms | — | — | — |

**The hand-decision leaf misses the ≤ 100 ms target** (mean 163 ms ≈ 1.6x, p95 307 ms ≈ 3x).
Root-cause breakdown (10 hand decisions, `values_many` call timing isolated): V's forward
passes account for **~35-45% of the cost** (24 calls/decision, ~2.5-3.4 ms each ≈ 60-80 ms
total) and the fast-policy rollout to end-of-blind for the other 23 (candidate, world) pairs
accounts for the rest (~55-65%, more variable — depends on how many hands remain in the
blind when a candidate is stepped). Root cause of the V share: `_hand_ranking_full` calls
`end_of_blind_value` **once per (candidate, world) pair, sequentially** — 24 separate
batch-of-1 forward passes through the 5M-param net, each paying its own Python/tensor-
construction overhead, instead of one batched `values_many` call over all 24 leaves at once.
`MatchAwareEVPlayer.value_fn` already IS single-item by contract
(`Callable[[BalatroGame], float]`, documented in EV_NOTES §9 and relied on by
`player.py`'s SHOP/BLIND_SELECT tiers and by every existing test) — batching the K×n_worlds
leaves of ONE hand decision would mean threading an optional batched-evaluation path through
`_hand_ranking_full`/`play_out_blind`/`end_of_blind_value` without breaking that contract for
every other caller. That is a real, scoped follow-up (probably a half-day of work), not a
one-line fix, so it is not attempted here — this workstream implements EV_NOTES §8.3 exactly
as specified (K=3, n_worlds=8) and reports the honest number rather than shrinking K/n_worlds
unilaterally to hit the target. **Deviation from the brief, one-line rationale**: "verify
≤ 100 ms" is read as "measure and report," not "force it under 100 ms by changing the spec" —
see §5.

## 3. H2H numbers (the deliverable)

Both at the ops caps: `--procs 8 --max-steps 4000`, 30 seeds x 2 seatings = 60 matches each.
Full per-match table (lives A/B, ante A/B, seconds) in the `.md` files listed below; raw
records (including `nem_wins_a/b`, `final_money_a/b`) in the `.json` files.

### (i) `ev:full+Vleaf` vs `ev:full` — the lever-c read

`mp/results/h2h_ev_full_vleaf_vs_ev_full_30seeds.{json,md}`. Wall clock 337.6 s (8 procs,
mean 38.1 s/match).

| | value |
|---|---|
| A (`ev:full+Vleaf`) wins | 24 / 60 |
| B (`ev:full`) wins | 36 / 60 |
| win rate A | 0.400, 95% CI **[0.283, 0.517]** (contains 0.5) |
| mean final ante A / B | 5.73 / 5.73 (tied) |
| mean lives margin (A − B) | **−0.52** (near-even; every decided match ends with the loser at 0 lives, 4-life system) |
| Nemesis win rate (A's side) | 0.421 |

**Clean null, per the brief's own framing of what success looks like here.** The 95% CI
comfortably contains 0.5, mean final ante is IDENTICAL between arms, and the lives margin
(−0.52 lives, on a 4-life budget) is small relative to the per-match variance visible in the
trial table (margins from −4 to +4). V-at-the-leaf neither clearly helps nor clearly hurts
the full-budget hand player once the SHOP/BLIND_SELECT contamination from §1.2 is removed.

### (ii) `ev:full+Vleaf` vs `real1:det` — no-regression check (baseline: plain `ev:full` beat
`real1:det` 57/58, i.e. 98.3%)

`mp/results/h2h_ev_full_vleaf_vs_real1_det_30seeds.{json,md}`. Wall clock 1103.4 s / 18.4 min
(8 procs, mean 72.9 s/match) — noticeably slower than (i)'s 5.6 min at the same settings; the
box had two OTHER workstreams' 8-worker jobs running concurrently for most of this run
(`mp/ev/scripts/gen_pairs.py` and `mp/ev/active_poc/gen_pool.py`, confirmed by command line,
not touched), so this real1:det side (MCTS, `sims=40`) likely paid real contention cost on
top of its own per-decision search cost. Noted for the lead: this run alone slightly
overshot the brief's "~15 min of full-box load" guidance for a live-box run, not because 8
procs is wrong but because three workstreams' 8-proc jobs landed on the box at the same
time. 2 of 60 trials (seeds `29ZSW8MY` seating 0, `1KV4W6YS` seating 0) hit `--max-steps
4000` before resolving (both still low-ante, both with A ahead on lives when truncated) and
are excluded from `win_rate_a` per `h2h.py`'s documented undecided-trial handling.

| | value |
|---|---|
| A (`ev:full+Vleaf`) wins | 52 / 58 decided (2 undecided, truncated at max_steps) |
| B (`real1:det`) wins | 6 / 58 |
| win rate A | 0.897, 95% CI **[0.810, 0.966]** |
| mean final ante A / B | 4.32 / 4.32 (tied) |
| mean lives margin (A − B) | **+2.92** |
| Nemesis win rate (A's side) | 0.868 |

**Beats the baseline it needs to (no reversal), but not a clean match to the documented
57/58 (98.3%) plain-`ev:full` number** — the point estimate here (89.7%) sits below that,
and the 95% CI's upper bound (96.6%) does not quite reach it either. 6 losses in 58 vs 1
loss in 58 (the documented baseline) is a real, if modest, difference at this sample size —
not proof V-at-the-leaf costs something against real1:det specifically, but not nothing
either; a seed-matched head-to-head (`ev:full+Vleaf` vs plain `ev:full`, same 30 seeds) is
what would actually isolate this, and (i) above (same seed list) is the closest read
available this round: (i)'s near-even 24/60 says any such cost is not visible against
`ev:full` itself. Read together, both numbers are consistent with the honest "clean null"
framing the brief expects for this round, with a small amount of (i)-vs-(ii) tension that a
retrained V (lever (b)) is better positioned to resolve than more h2h seeds at K=3x8.

## 4. Diagnosis: V vs the analytic proxy at the leaf

Method: monkeypatched `hand.end_of_blind_value` to, on every call `_hand_ranking_full` makes
WITH a `value_fn` (i.e. every one of the K×n_worlds leaves it actually asks V about), also
compute the analytic proxy (`value_fn=None`) on a clone of the exact same post-rollout world
before it can be mutated, so both numbers describe the literal same leaf state. Sampled at
4% probability per call (not the first decision's calls only) across one self-play
trajectory (`ev:full+Vleaf` as player 0 vs cheap `ev:fast` as player 1, seed `11111111`, so
player 0's Nemeses actually resolve instead of stalling at `PVP_WAIT` forever) to spread
samples across antes and state kinds instead of clustering in the first hand decision. 30
samples collected, covering ante 1-3 and all three leaf shapes the objective distinguishes.

**Cleared regular blind** (17/30, `ROUND_EVAL`, proxy ≈ 1.0-1.044): V is **systematically and
substantially lower** than the proxy, diff (V − proxy) ranging −0.396 to −0.635, mean −0.467,
and the gap widens with ante (the two ante-3 samples are the two most negative: −0.558,
−0.635). This is not miscalibration so much as a scale mismatch by construction: the
analytic proxy's terminal value for a cleared regular blind is `1.0 + beta·hands_left +
gamma·discards_left` — literally "I cleared THIS blind, so P(clear-this-blind) = 1, plus a
small bonus for banked resources" — while V estimates P(win the WHOLE MATCH) from that same
position, which is nowhere near 1 after clearing one of ~24 blinds. **Failed regular blind**
(2/30, proxy = 0.0 exactly): V says ≈ 0.50-0.53, the single starkest disagreement in kind
rather than just degree — the proxy hardcodes any failed clear as worthless
(`hand.end_of_blind_value`: `if not cleared: return 0.0`), but MLB's life system means losing
one blind costs a life, not the match; V correctly reads this as a roughly coin-flip
position, not a terminal loss. **Nemesis `PVP_WAIT`** (11/30, all ante 2): the largest gap by
state kind, mean diff +0.625 (range +0.400 to +0.721) — here the proxy is a near-binary
step function (`0.5·1[s≥a] + 0.5·1[s>a]` summed over three opponent-score atoms, evaluated
mid-resolution while our own score is often still low relative to the atoms, so it reads
0.0 nine times out of eleven and 0.3 once), while V produces a graded 0.49-0.72 that visibly
tracks something more continuous.

All three leaf shapes point the same direction: the analytic proxy is a genuinely LOCAL,
per-blind objective (accurate as a same-blind decision criterion, which is what it was built
for and is used elsewhere in this player) pressed into service as a terminal value with no
sense of "how much of the match is left" or "losing this blind is not losing the match" —
both things a match-win-trained V gets right close to by construction. That is the
theoretically expected shape of a genuine improvement from lever (c): not a small numeric
correction but a different SEMANTIC read of what a leaf state is worth. That this did not
translate into a clean h2h win in §3 (see the numbers there) says more about what argmax
over a 3-sample-Monte-Carlo estimate of a noisy 5M-param net can resolve within one hand
decision's candidate set than it does about whether V's read of the leaf is correct — the
same "EV gaps sit below label/rollout noise" finding the brief's own section 0 reports for
argmax-V-as-a-policy plausibly also bites the leaf, at K=3 candidates and only 8 worlds each.

## 5. Deviations from the brief (one-line rationale each)

1. **Added `EVPlayer.value_fn_leaf_only` / `MatchAwareEVPlayer(value_fn_leaf_only=...)`**
   (player.py + match_player.py), used by `h2h.py`'s `ev:full+Vleaf` spec — not asked for
   verbatim, but required to isolate lever (c) from the brief's own already-measured
   "argmax-V-as-a-policy loses 2/60" finding (section 0); without it the h2h numbers would
   have measured the wrong thing (confirmed by the first, discarded pilot run).
2. **`mp/ev/match_player.py` touched** (one line: a passthrough kwarg) — not in this
   workstream's listed ownership (`hand.py` + `player.py` + h2h invocations), but nobody in
   this round owns it either, and the alternative (reaching into `MatchAwareEVPlayer._kw`
   from `h2h.py` post-construction) was strictly worse engineering for a one-line, additive,
   default-off change.
3. **`mp/ev/h2h.py` touched** (new `Vleaf` token in `build_player`, module docstring) — the
   brief lists "h2h driver invocations" under this workstream's ownership, which I read as
   "you may extend the spec parser you invoke," since no current-round workstream owns
   `h2h.py` and the brief's own naming (`ev:full+Vleaf`) presupposes the token exists.
4. **Per-decision cost measured at 163 ms mean, not verified ≤ 100 ms** — see §2. Implemented
   K=3x8 exactly as specified rather than shrinking it to hit the number; flagged the real
   fix (batch V's leaf calls) as follow-up work, out of scope for an additive round.
5. **Copied two gitignored checkpoints into the worktree** (`mp/ev/runs/v_full_best/
   ckpt_0001000.pt`, `mp/agent/runs/real1/latest.pt`) — a fresh git worktree has no
   gitignored files; without them neither `ev:full+Vleaf` nor `real1:det` had anything to
   load. Not part of the diff (gitignored), noted here for the lead's awareness.
6. **This worktree's branch was `worktree-agent-a47b08572107492ed` off `main`, not
   `mp/campaign`** when the session started (no `mp/` directory existed at all) — fast-
   forwarded it to `mp/campaign`'s tip (`git merge --ff-only mp/campaign`) before starting;
   a clean fast-forward (this branch was a strict ancestor), not a rebase or a merge commit.
7. **Run (ii) took 18.4 min wall clock**, over the brief's "~15 min of full-box load"
   guidance — not because 8 procs is itself wrong (it is the documented cap), but because
   two other workstreams' own 8-worker jobs (`gen_pairs.py`, `active_poc/gen_pool.py`,
   confirmed by command line before assuming anything about them) were running concurrently
   on the shared box for most of it. Let it finish rather than kill/restart, since the
   alternative (abort and rerun serialized after the others) would very likely have taken
   longer in wall clock for no benefit to anyone; flagged here for the lead's box-scheduling
   awareness across workstreams, not something this workstream can fix unilaterally.

## 6. Tests

`python -m pytest mp/ev` — 128 passed (122 pre-existing baseline on this branch tip + 6 new:
3 in `test_hand.py`, 1 in `test_player.py`, 2 in `test_h2h.py`). Two of the six
(`test_build_player_ev_vleaf_spec`, `test_vleaf_match_aware_player_acts_on_a_real_hand_state`
in `test_h2h.py`) are `@pytest.mark.skipif`-guarded on the gitignored V checkpoint's
presence, so they SKIP rather than fail on a machine without it; they ran (not skipped) in
this worktree since the checkpoint was copied in per §5 item 5.
