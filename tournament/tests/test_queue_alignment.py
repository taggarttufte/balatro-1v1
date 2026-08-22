"""Queue alignment (brief §1 item, GATE_NOTES.md's key classification): a sample of
independent agents on the SAME seed must show, at every shop-visit ordinal, that every RNG
key whose queue position differs between two agents falls into one of ``classify_key``'s
explained classes (imported from ``mlb_match_demo``, not copied); a key classified SHARED
must never actually differ (if it appears in ``diff_rng``'s output at all, that IS the
violation), and UNKNOWN always fails.

This generalises Phase 2's two-player ``TestAlignment`` (mp/tests/test_mlb_match_gate.py) to
N independent solo games instead of one ``MLBMatch`` pair -- the underlying invariant is the
same: same seed, no skips -> identical run-structure streams; only each agent's OWN shop
consumption (rerolls, purchases, packs opened) legitimately diverges the rest."""
from ..bootstrap import State, mlb_match_demo as D
from ..players import ScriptedPlayer, ScriptedPlayerAdapter
from ..runner import construct_games

MAX_ANTE = 4

SPECS = [
    ScriptedPlayer(name="opener", hand="greedy", open_pack_slot=0, pick_from_pack=True),
    ScriptedPlayer(name="reroller", hand="greedy", rerolls_per_visit=1, buy_slot0=True),
    ScriptedPlayer(name="reroller2", hand="greedy", rerolls_per_visit=2, buy_slot0=True, open_pack_slot=1),
    ScriptedPlayer(name="weak_opener", hand="greedy_until", weak_from_ante=3, open_pack_slot=0),
]


def _collect_shop_snapshots(game, player, max_ante=MAX_ANTE, max_steps=20_000) -> dict:
    """{(ante, blind_idx): rng_state_at_shop_entry} for every shop visited while
    game.ante <= max_ante.  Nemesis blinds are cashed out with a bare ``advance`` (no
    life-rule bookkeeping needed for this alignment check)."""
    snaps = {}
    prev = None
    n = 0
    while game.ante <= max_ante and game.state != State.GAME_OVER:
        s = game.state
        if s == State.SHOP and prev != State.SHOP:
            snaps[(game.ante, game.blind_idx)] = dict(game.run_state.rng.snapshot()["state"])
        if s == State.ROUND_EVAL and game.current_blind.is_pvp:
            game.step({"type": "advance"})
        else:
            game.step(player.act(game))
        prev = s
        n += 1
        if n > max_steps:
            raise RuntimeError("wedged collecting shop snapshots")
    return snaps


def test_first_shop_of_every_ante_stays_aligned_across_independent_agents():
    seed = "7I4M53DL"
    games, seed_str = construct_games(seed, len(SPECS), "b_red", 1, 4)
    players = [ScriptedPlayerAdapter(spec) for spec in SPECS]
    snaps = [_collect_shop_snapshots(g, p) for g, p in zip(games, players)]

    checked_any = False
    violations = []
    for a in range(len(SPECS)):
        for b in range(a + 1, len(SPECS)):
            common_visits = set(snaps[a]) & set(snaps[b])
            for vid in common_visits:
                d = D.diff_rng(seed_str, snaps[a][vid], snaps[b][vid])
                for key, cls, p0, p1 in d:
                    checked_any = True
                    if cls in ("SHARED", "UNKNOWN"):
                        violations.append((a, b, vid, key, cls, p0, p1))
    assert checked_any, "expected at least one differing key across these heterogeneous specs"
    assert violations == [], f"non-explained / SHARED-but-differing keys: {violations}"


def test_boss_stream_is_identical_at_every_common_shop_visit():
    """A direct spot-check of the SHARED shadow-boss stream (game.py's own 'boss' key,
    GATE_NOTES.md's SHARED class): its raw pseudorandom value must be equal between any two
    agents at every shop visit ordinal both reached, independent of ``diff_rng``'s own
    "only lists differences" logic."""
    seed = "7I4M53DL"
    games, seed_str = construct_games(seed, len(SPECS), "b_red", 1, 4)
    players = [ScriptedPlayerAdapter(spec) for spec in SPECS]
    snaps = [_collect_shop_snapshots(g, p) for g, p in zip(games, players)]
    compared = 0
    for a in range(len(SPECS)):
        for b in range(a + 1, len(SPECS)):
            for vid in set(snaps[a]) & set(snaps[b]):
                v0 = snaps[a][vid].get("boss")
                v1 = snaps[b][vid].get("boss")
                if v0 is None or v1 is None:
                    continue
                compared += 1
                assert v0 == v1, f"'boss' stream diverged at visit {vid} between specs {a},{b}"
    assert compared > 0
