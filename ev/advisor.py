"""
advisor.py -- the snapshot advisor (Phase 5 rev 2, W6).

Given a state (a fixture, a step of a logged match, or a re-driven self-play seed), prints a
Tagg-facing report for one player:

  1. the situation (ante/blind, lives, $, jokers both sides, the opponent PUBLIC block)
  2. THREE P(win) +/- CI estimators: the rollout label (``labels.label_state``), the race
     calculator (``race.p_win`` on curves fit from ``match.pvp_log``), and V's estimate when
     a checkpoint is given (``value_net.make_value_fn``) -- flagged when any two disagree by
     more than 0.15
  3. the ranked action table for the live state (``EVPlayer.explain``), plus, in SHOP /
     BOOSTER_OPEN, the decision-statistics table (``stats.decide.explain``)
  4. a level-0 "what is the opponent probably doing" line from the public block

Every number here is computed by calling into W3/W4/W5's existing public APIs
(``player.EVPlayer``, ``hand.rank_hand_actions``, ``stats.decide``, ``ev.labels``, ``ev.race``,
``mcts.value_net`` / ``mcts.encoder_v2``) -- nothing is re-derived. Read-only on the ``match``
passed in: every helper here either calls a documented side-effect-free API or clones first
(``test_advisor.py`` pins ``match.signature()`` before/after ``advise()``).

State sources (``load_state_source(spec)``):

  ``fixture:<name>``       -- a builder in ``mp/ev/fixtures/`` (``FIXTURES`` registry)
  ``replay:<path>:<step>`` -- an ``mp/replay`` MatchLogger JSONL log, replayed to ``step`` ops
                             in (this module's own driver -- only ``mp/replay``'s PUBLIC
                             ``load_line`` is used, never its private helpers)
  ``seed:<seed>:<step>``   -- a fresh ``MLBMatch(seed=...)`` re-driven ``step`` decisions by
                             fresh ``EVPlayer`` self-play (deterministic given seed + policy
                             seeds)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent            # mp/ev
_MP = _HERE.parent
for _p in (str(_HERE), str(_MP), str(_MP / "eval"), str(_MP / "agent"), str(_MP / "stats")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap  # noqa: E402,F401
from _bootstrap import MLBMatch, State, MP_ROOT, game_keys  # noqa: E402

import player as P  # noqa: E402  (mp/ev/player.py, W3)
import labels  # noqa: E402       (mp/ev/labels.py, W5)
import race as _race  # noqa: E402  (mp/ev/race.py, W5)

import decide  # noqa: E402  (mp/stats/decide.py, W4 -- mp/stats is on sys.path above)

import mcts.encoder_v2 as encoder_v2  # noqa: E402  (mp/agent/mcts, W1)
import mcts.value_net as value_net  # noqa: E402

__all__ = [
    "load_state_source", "situation_lines", "opponent_public_block_lines",
    "opponent_read_lines", "prob_block", "action_table_lines", "stats_table_lines",
    "advise", "load_value_fn",
]


# =============================================================== state sources

def _fixtures_registry() -> dict:
    """The live ``fixtures.FIXTURES`` registry (``mp/ev/fixtures/__init__.py``), fetched
    fresh on every call so a fixture added after import time is still picked up."""
    import fixtures as _fx
    return dict(_fx.FIXTURES)


def _replay_to_step(path: str, step: int) -> MLBMatch:
    """Reconstruct an ``MLBMatch`` from an ``mp/replay`` MatchLogger JSONL log (the FIRST
    ``kind == "match"`` line in the file) and replay its first ``step`` ops.  Uses only
    ``mp/replay/replay.py``'s public ``load_line`` -- the "replay to a given step" driver is
    this module's own (``mp/replay`` has no such entry point and this workstream does not
    edit ``mp/replay``).  ``mp/replay`` is a real package (its modules use relative imports),
    so it must be imported as ``replay.replay`` with ``mp/`` (not ``mp/replay/``) on
    ``sys.path`` -- ``_MP`` already is (see the sys.path block above)."""
    import replay.replay as _replay  # mp/replay/replay.py, imported as a package submodule

    line = _replay.load_line(path, 0)
    if line.get("kind", "episode") != "match":
        raise ValueError(f"{path}: first line is kind={line.get('kind')!r}, want 'match'")
    m = MLBMatch(seed=line["seed"], deck_key=line.get("deck_key", "b_red"),
                stake=line.get("stake", 1), lives=line.get("lives_start", 4),
                pvp_start_round=line.get("pvp_start_round", 2))
    ops = line.get("ops") or []
    n = max(0, min(int(step), len(ops)))
    for op in ops[:n]:
        m.step(int(op["player"]), dict(op["action"]))
    return m


def _selfplay_to_step(seed: str, step: int, *, policy_seed: int = 0, budget: str = "fast",
                      deck_key: str = "b_red", stake=1, lives: int = 4,
                      max_steps: int = 40_000) -> MLBMatch:
    """A fresh ``MLBMatch`` driven ``step`` decisions by two fresh ``EVPlayer(budget=...,
    epsilon=0)`` -- deterministic given ``(seed, policy_seed, budget)``."""
    m = MLBMatch(seed=seed, deck_key=deck_key, stake=stake, lives=lives)
    pols = [P.EVPlayer(budget=budget, seed=policy_seed, epsilon=0.0),
           P.EVPlayer(budget=budget, seed=policy_seed + 1, epsilon=0.0)]
    n = max(0, int(step))
    steps = 0
    while not m.done and steps < n and steps < max_steps:
        p = m.current_player()
        if p is None:
            raise RuntimeError(f"self-play wedged at step {steps} ({m.state()})")
        acts = m.legal_actions(p)
        m.step(p, pols[p].act(m.games[p]))
        steps += 1
    return m


def load_state_source(spec: str, *, policy_seed: int = 0, budget: str = "fast"
                      ) -> tuple[MLBMatch, int]:
    """``spec`` -> ``(match, default_player)``.  See the module docstring for the three
    forms.  ``default_player`` is 0 for every source (the CLI's ``--player`` overrides it)."""
    kind, sep, rest = spec.partition(":")
    if not sep:
        raise ValueError(f"bad state source {spec!r} (want 'fixture:' / 'replay:' / 'seed:')")
    if kind == "fixture":
        reg = _fixtures_registry()
        if rest not in reg:
            raise ValueError(f"unknown fixture {rest!r} (have {sorted(reg)})")
        return reg[rest](), 0
    if kind == "replay":
        path, _, step_s = rest.rpartition(":")
        if not path:
            raise ValueError(f"bad replay source {spec!r} (want 'replay:<path>:<step>')")
        return _replay_to_step(path, int(step_s)), 0
    if kind == "seed":
        seed_s, _, step_s = rest.partition(":")
        if not step_s:
            raise ValueError(f"bad seed source {spec!r} (want 'seed:<seed>:<step>')")
        return _selfplay_to_step(seed_s, int(step_s), policy_seed=policy_seed, budget=budget), 0
    raise ValueError(f"unknown state source kind {kind!r} in {spec!r}")


# =============================================================== formatting helpers

def _joker_name(key: str) -> str:
    return game_keys.JOKER_NAME.get(key, key)


def _joker_list(jokers) -> str:
    if not jokers:
        return "(none)"
    return ", ".join(f"{_joker_name(j.key)}[{j.edition}]" if j.edition != "None" else _joker_name(j.key)
                     for j in jokers)


def _consumable_name(key: str) -> str:
    return (game_keys.TAROT_NAME.get(key) or game_keys.PLANET_NAME.get(key)
           or game_keys.SPECTRAL_NAME.get(key) or key)


def _fmt_action(game, action: dict) -> str:
    t = action.get("type")
    if t == "play":
        cards = [repr(game.hand[i]) for i in action.get("cards", ()) if i < len(game.hand)]
        return f"play {' '.join(cards)}"
    if t == "discard":
        cards = [repr(game.hand[i]) for i in action.get("cards", ()) if i < len(game.hand)]
        return f"discard {' '.join(cards)}"
    if t == "buy":
        idx = action.get("item_idx")
        if idx is not None and 0 <= idx < len(game.current_shop):
            it = game.current_shop[idx]
            return f"buy {it.name} (${it.price})"
        return f"buy item[{idx}]"
    if t == "sell_joker":
        idx = action.get("joker_idx")
        if idx is not None and 0 <= idx < len(game.jokers):
            return f"sell {_joker_name(game.jokers[idx].key)}"
        return "sell joker"
    if t == "reroll":
        cost = max(0, getattr(game, "reroll_cost", 0) - getattr(game, "reroll_discount", 0))
        return f"reroll (${cost})"
    if t == "leave_shop":
        return "leave shop"
    if t == "play_blind":
        return f"play {game.current_blind.name}"
    if t == "skip_blind":
        return "skip blind"
    if t == "reroll_boss":
        return "reroll boss"
    if t == "use_consumable":
        ci = action.get("consumable_idx")
        key = game.consumable_hand[ci] if ci is not None and 0 <= ci < len(game.consumable_hand) else "?"
        targets = action.get("target_cards") or []
        tstr = ""
        if targets:
            tstr = " -> " + " ".join(repr(game.hand[i]) for i in targets if i < len(game.hand))
        return f"use {_consumable_name(key)}{tstr}"
    if t == "pick_booster":
        names = []
        for i in action.get("indices", ()):
            if i < len(game.booster_choices):
                c = game.booster_choices[i]
                names.append(_joker_name(c.key) if c.key.startswith("j_") else _consumable_name(c.key))
        return f"pick {', '.join(names)}" if names else "pick booster"
    if t == "skip_booster":
        return "skip pack"
    if t == "advance":
        return "advance"
    return str(action)


# =============================================================== 1. the situation

def situation_lines(match: MLBMatch, player: int) -> list:
    me = match.games[player]
    opp = match.games[1 - player]
    blind = me.current_blind
    nem = " <NEMESIS>" if getattr(blind, "is_pvp", False) else ""
    lines = [
        f"Ante {me.ante}  blind {me.blind_idx} ({blind.kind}{nem})  state={me.state.name}",
        f"Lives:  me {me.lives}  vs  opp {opp.lives}      "
        f"$:  me {me.dollars}  vs  opp {opp.dollars}",
        f"My jokers  ({len(me.jokers)}/{me.joker_slots}): {_joker_list(me.jokers)}",
        f"Opponent jokers [HIDDEN IN A REAL MATCH -- shown here only because this state "
        f"source has full simulator access]: {_joker_list(opp.jokers)}",
    ]
    return lines


def opponent_public_block_lines(match: MLBMatch, player: int) -> list:
    opp = encoder_v2.opponent_view(match, player)
    lines = ["", "Opponent PUBLIC block (opponent_view -- what a live match actually reveals):"]
    if not opp.known:
        lines.append("  (unknown -- solo / non-MLB game)")
        return lines
    lines.append(f"  lives={opp.lives}  $={opp.dollars}  ante={opp.ante}  blind_idx={opp.blind_idx}  "
                f"state={opp.state}  phase={opp.phase}")
    lines.append(f"  chips_scored={opp.chips_scored}  hands_left={opp.hands_left}  "
                f"comeback_bonus={opp.comeback_bonus}  comeback_pending={opp.comeback_pending}")
    lines.append(f"  econ: sells_per_ante={opp.sells_per_ante}  spent_in_shop={opp.spent_in_shop}  "
                f"sells_total={opp.sells_total}  spent_total={opp.spent_total}")
    if opp.current_nemesis is not None:
        nl = opp.current_nemesis
        lines.append(f"  live Nemesis: them {nl.their_score} ({nl.their_hands_left} hands left)  "
                    f"vs  me {nl.my_score} ({nl.my_hands_left} hands left)")
    if opp.last_nemeses:
        lines.append("  last Nemeses (most recent first):")
        outcome_word = {1: "I took a life", 0: "tie", -1: "I lost a life"}
        for r in opp.last_nemeses:
            lines.append(f"    ante {r.ante}: they scored {r.their_score} in {r.their_hands_used} hands, "
                        f"I scored {r.my_score} -> {outcome_word.get(r.outcome, '?')}"
                        f"{' (early end)' if r.early_end else ''}")
    else:
        lines.append("  no Nemeses resolved yet")
    return lines


def opponent_read_lines(match: MLBMatch, player: int) -> list:
    opp = encoder_v2.opponent_view(match, player)
    lines = ["", "Opponent read (level-0 -- public features only, no belief model over the "
                 "shared menu / signalling yet):"]
    if not opp.known:
        lines.append("  (unknown -- solo / non-MLB game)")
        return lines
    if opp.last_nemeses:
        last = opp.last_nemeses[0]
        result = {1: "I won it", 0: "it tied", -1: "they won it"}.get(last.outcome, "?")
        lines.append(f"  most recent Nemesis (ante {last.ante}): they scored {last.their_score} "
                    f"in {last.their_hands_used} hands, {result}")
    else:
        lines.append("  no Nemeses resolved yet")
    lines.append(f"  shop spend this ante: ${opp.spent_in_shop}, sold {opp.sells_per_ante} joker(s) "
                f"this ante  (lifetime: ${opp.spent_total} spent, {opp.sells_total} sold)")
    lines.append("  no inference beyond these public numbers is attempted.")
    return lines


# =============================================================== 2. three P(win) numbers

def load_value_fn(checkpoint: str, device: str = "cpu"):
    """``(value_fn_raw, net, encoder, extra)``.  ``value_fn_raw(game, opp=None) -> float``."""
    net, encoder, extra = value_net.load_checkpoint(checkpoint, device=device)
    fn = value_net.make_value_fn(net, encoder, device=device)
    return fn, net, encoder, extra


def prob_block(match: MLBMatch, player: int, *, n_rollouts: int = 32, rollout_seed: int = 0,
               rollout_budget: str = "fast", checkpoint: Optional[str] = None,
               race_cfg: _race.RaceConfig = _race.DEFAULT, value_fn_raw=None) -> tuple:
    """``(text, numbers)``.  ``numbers`` = {"rollout": p, "rollout_ci": ci, "rollout_s": t,
    "race": p, "race_rows": [...], "v": p|None}."""
    game = match.games[player]
    opp_game = match.games[1 - player]
    ante = game.ante
    blinds_done = min(max(game.blind_idx, 0), race_cfg.regular_blinds)

    t0 = time.perf_counter()
    pol_factory = labels.make_policy_factory(budget=rollout_budget, epsilon=0.02)
    p_roll, ci_roll = labels.label_state(match, player, n_rollouts=n_rollouts, seed=rollout_seed,
                                         policy_factory=pol_factory)
    t_roll = time.perf_counter() - t0

    my_curve = _race.curve_from_history(match.pvp_log, player, ante, cfg=race_cfg)
    their_curve = _race.curve_from_history(match.pvp_log, 1 - player, ante, cfg=race_cfg)
    p_race = _race.p_win(my_curve, their_curve, game.lives, opp_game.lives, ante,
                         cfg=race_cfg, blinds_done=blinds_done)
    race_rows = _race.race_table(my_curve, their_curve, game.lives, opp_game.lives, ante,
                                 cfg=race_cfg, n_antes=4)

    p_v = None
    if checkpoint is not None or value_fn_raw is not None:
        fn = value_fn_raw
        if fn is None:
            fn, _net, _enc, _extra = load_value_fn(checkpoint)
        opp_view = encoder_v2.opponent_view(match, player)
        p_v = float(fn(game, opp_view))

    numbers = {"rollout": float(p_roll), "rollout_ci": float(ci_roll), "rollout_seconds": t_roll,
              "race": float(p_race), "race_rows": race_rows, "v": p_v,
              "my_curve_n_obs": my_curve.n_obs, "their_curve_n_obs": their_curve.n_obs}

    lines = ["", "P(win) -- three estimators:"]
    lines.append(f"  rollout  (n={n_rollouts}, budget={rollout_budget}, determinized, symmetric "
                f"analytic opponent): {p_roll:.3f} +/- {ci_roll:.3f}   [{t_roll:.1f}s]")
    lines.append(f"  race     (curve_from_history: my n_obs={my_curve.n_obs}, their "
                f"n_obs={their_curve.n_obs}, ante={ante}): {p_race:.3f}")
    if p_v is not None:
        lines.append(f"  V        (checkpoint={checkpoint}): {p_v:.3f}")
    else:
        lines.append("  V        (no --checkpoint given): n/a")

    pairs = [("rollout", p_roll), ("race", p_race)]
    if p_v is not None:
        pairs.append(("v", p_v))
    flags = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            (ka, va), (kb, vb) = pairs[i], pairs[j]
            if abs(va - vb) > 0.15:
                flags.append(f"{ka} vs {kb} differ by {abs(va - vb):.3f}")
    if flags:
        lines.append(f"  ** DISAGREEMENT: {'; '.join(flags)} **")

    lines.append("  race table (next 4 antes):")
    for row in race_rows:
        lines.append(f"    ante {row['ante']:>2d}  my mu/sigma={row['my_mu']:.2f}/{row['my_sigma']:.2f}  "
                    f"their mu/sigma={row['their_mu']:.2f}/{row['their_sigma']:.2f}  "
                    f"P(I lose Nemesis)={row['p_i_lose_nemesis']:.2f}  "
                    f"p_win_from_here={row['p_win_from_here']:.3f}")
    return "\n".join(lines), numbers


# =============================================================== 3. the ranked action table

def action_table_lines(game, *, explainer=None, budget: str = "full", top_n: int = 8) -> list:
    """``explainer``: anything with ``.explain(game) -> [(action, ev, reason)]`` -- either a
    plain ``player.EVPlayer`` (built here when ``explainer is None``) or a bound
    ``match_player.MatchAwareEVPlayer`` (V-guided AND opponent-aware -- see ``advise()``)."""
    pl = explainer if explainer is not None else P.EVPlayer(value_fn=None, stats=None,
                                                            budget=budget, epsilon=0.0)
    t0 = time.perf_counter()
    ranked = pl.explain(game)
    dt = time.perf_counter() - t0
    tag = " (V-guided, opponent-aware)" if explainer is not None else ""
    lines = ["", f"Ranked actions ({game.state.name}{tag}, budget={budget}, {len(ranked)} "
                f"candidates, {dt * 1000:.1f} ms):"]
    for i, (a, ev, reason) in enumerate(ranked[:top_n]):
        lines.append(f"  {i + 1}. {_fmt_action(game, a):<42s} ev={ev:+.4f}  {reason}")
    if len(ranked) > top_n:
        lines.append(f"  ... ({len(ranked) - top_n} more)")
    return lines


def stats_table_lines(game) -> list:
    if game.state not in (State.SHOP, State.BOOSTER_OPEN):
        return []
    rows = decide.decision_table(game)
    lines = ["", "Decision-stats table (P(hit), true cost incl. interest, urgency, net EV):"]
    lines.append(decide.explain(rows))
    return lines


# =============================================================== the whole report

def advise(match: MLBMatch, player: int = 0, *, n_rollouts: int = 32, rollout_seed: int = 0,
          rollout_budget: str = "fast", budget: str = "full", checkpoint: Optional[str] = None,
          top_n: int = 8) -> str:
    """The full advisor report for ``player`` on ``match`` (a snapshot; never mutated)."""
    game = match.games[player]
    lines = ["=" * 78, f"EV ADVISOR -- player {player}  (seed {match.seed_str})", "=" * 78]
    lines.extend(situation_lines(match, player))
    lines.extend(opponent_public_block_lines(match, player))

    # W5's match_player.MatchAwareEVPlayer (landed mid-Phase-5-rev-2) binds a mutable
    # opponent view into V's closure and refreshes it from `match` before every decision --
    # this is what makes V's use INSIDE the ranked action table opponent-aware too, not just
    # the standalone "V" number below (see ADVISOR_NOTES.md section 1, point 3 / section 6).
    value_fn_raw = None
    mp_player = None
    if checkpoint is not None:
        import match_player as MP  # mp/ev/match_player.py, W5
        net, encoder = MP.load_value(checkpoint)
        mp_player = MP.MatchAwareEVPlayer(net, encoder, budget=budget, seed=0)
        mp_player.bind(match, player)
        # `mp_player.values([g])` already uses its own bound opponent view (`_mp._opp`, kept
        # in sync with `match`/`player` by `.bind()`/`.refresh()`); the `opp` argument
        # `prob_block` passes is redundant with it (same match, same player) so it is ignored
        # here rather than double-computed.
        value_fn_raw = lambda g, _opp=None, _mp=mp_player: float(_mp.values([g])[0])

    prob_text, _numbers = prob_block(match, player, n_rollouts=n_rollouts, rollout_seed=rollout_seed,
                                     rollout_budget=rollout_budget, value_fn_raw=value_fn_raw,
                                     checkpoint=checkpoint)
    lines.append(prob_text)

    lines.extend(action_table_lines(game, explainer=mp_player, budget=budget, top_n=top_n))
    lines.extend(stats_table_lines(game))
    lines.extend(opponent_read_lines(match, player))
    lines.append("=" * 78)
    return "\n".join(lines)
