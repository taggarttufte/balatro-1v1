"""
Phase 4 W2: tournament-driven training loop, the MCTS record hook, the population, the
external-target objective, and pause / resume.

The expensive parts (a real generation, a resume) run with a deliberately tiny net and a
4-sim search — this file is about the plumbing and the LABELS, not about learning. The
label maths is tested against synthetic `AnteMatrix`es so it is exact and fast.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from balatro_sim.game import BalatroGame, State
from mcts import MCTSConfig, MCTSPlayer, PolicyValueNet, get_encoder
from mcts.policy import make_policy
from train import (
    CheckpointHistory, MLBTrainConfig, MLBTrainer, PopulationConfig, RecordedDecision,
    SampleCollector, TrainConfig, build_population, load_checkpoint,
    make_sample, normalized_ranks, save_checkpoint, vanilla_boss_target,
)
from train.selfplay import (
    assign_value_targets, external_outcome_for, load_target_fn,
    play_solo_external_episode, tournament_module, value_targets_from_result,
)

SEED = "7I4M53DL"


def tiny_cfg(**kw) -> TrainConfig:
    base = dict(seed=0, sims=4, hidden=32, n_res_blocks=1, policy_hidden=16,
                encoder="mlb", ruleset="mlb", device="cpu", batch_size=4,
                min_buffer=4, buffer_capacity=2_000, max_decisions=400, max_antes=3,
                lives=4)
    base.update(kw)
    return TrainConfig(**base)


def tiny_mlb(**kw) -> MLBTrainConfig:
    base = dict(n_agents=4, m_current=2, p_history=2, seeds_per_generation=1,
                max_ante=3, sims_budgets=(1.0,), leaf_batch=4, episodes_per_generation=2)
    base.update(kw)
    return MLBTrainConfig(**base)


# ══════════════════════════════════════════════════════════════ ranks and value targets

def test_normalized_ranks_is_one_for_best_zero_for_worst_and_averages_ties():
    r = normalized_ranks([10, 0, 5, 5])
    assert r[0] == pytest.approx(1.0)
    assert r[1] == pytest.approx(0.0)
    assert r[2] == r[3] == pytest.approx(0.5)       # ascending positions 1 and 2 -> 1.5/3


def test_normalized_ranks_handles_absent_and_singleton_populations():
    r = normalized_ranks([7, np.nan, 1])
    assert math.isnan(r[1]) and r[0] == 1.0 and r[2] == 0.0
    assert normalized_ranks([3])[0] == 0.5          # nothing to compare against
    assert np.isnan(normalized_ranks([np.nan])).all()


class _FakeResult:
    """The three fields `value_targets_from_result` reads off a `TournamentResult`."""

    def __init__(self, ante_matrices, final_lives, last_score):
        self.ante_matrices = ante_matrices
        self.final_lives = final_lives
        self.last_score = last_score


def _matrices(scores_by_ante, n):
    AnteMatrix = tournament_module().matrix.AnteMatrix
    return [AnteMatrix.build(ante, n, sb) for ante, sb in sorted(scores_by_ante.items())]


def test_value_target_is_rank_at_the_next_nemesis_blended_with_the_final_standing():
    n = 4
    result = _FakeResult(
        ante_matrices=_matrices({2: {0: 900, 1: 100, 2: 500, 3: 300},
                                 3: {0: 100, 1: 900, 2: 500, 3: 300}}, n),
        final_lives=[4, 3, 2, 1],
        last_score={0: (3, 100), 1: (3, 900), 2: (3, 500), 3: (3, 300)},
    )
    t = value_targets_from_result(result, n, value_blend=1.0)     # pure short-horizon
    recs = [RecordedDecision(sample=make_sample(*_dummy_decision()), agent_idx=i, ante=a,
                             blind_idx=0, is_pvp=False, action_type="advance",
                             skip_offered=False, shortcut=True, n_legal=1)
            for i in range(n) for a in (1, 2, 3)]
    assign_value_targets(recs, t)
    z = {(r.agent_idx, r.ante): r.sample.z for r in recs}

    # Antes 1 and 2 both look ahead to the ante-2 Nemesis, where agent 0 won.
    assert z[(0, 1)] == z[(0, 2)] == pytest.approx(1.0)
    assert z[(1, 1)] == z[(1, 2)] == pytest.approx(0.0)
    # Ante 3 looks at the ante-3 Nemesis, where the order flipped.
    assert z[(0, 3)] == pytest.approx(0.0)
    assert z[(1, 3)] == pytest.approx(1.0)


def test_an_eliminated_agent_gets_the_worst_possible_short_horizon_target():
    """Absent from the next Nemesis == died before reaching it. If that were `nan` or the
    population mean, losing your last life would be free — the exact failure mode the
    overnight run exploited."""
    n = 3
    result = _FakeResult(
        ante_matrices=_matrices({2: {0: 500, 1: 100, 2: 300}, 3: {0: 500, 2: 300}}, n),
        final_lives=[4, 0, 2],
        last_score={0: (3, 500), 1: (2, 100), 2: (3, 300)},
    )
    t = value_targets_from_result(result, n, value_blend=1.0)
    recs = [RecordedDecision(sample=make_sample(*_dummy_decision()), agent_idx=1, ante=3,
                             blind_idx=0, is_pvp=False, action_type="advance",
                             skip_offered=False, shortcut=True, n_legal=1)]
    assign_value_targets(recs, t)
    assert recs[0].sample.z == pytest.approx(0.0)


def test_blend_mixes_the_two_terms_as_documented():
    n = 2
    result = _FakeResult(
        ante_matrices=_matrices({2: {0: 100, 1: 900}}, n),
        final_lives=[4, 1],
        last_score={0: (2, 100), 1: (2, 900)},
    )
    t = value_targets_from_result(result, n, value_blend=0.7)
    # Agent 0 lost the Nemesis (rank 0.0) but finished ahead on (rounds, lives, score)?
    # rounds survived tie 1-1, lives 4 > 1 -> agent 0's final standing is 1.0.
    assert t["outcome"][0] == pytest.approx(1.0)
    recs = [RecordedDecision(sample=make_sample(*_dummy_decision()), agent_idx=0, ante=2,
                             blind_idx=0, is_pvp=False, action_type="advance",
                             skip_offered=False, shortcut=True, n_legal=1)]
    assign_value_targets(recs, t)
    assert recs[0].sample.z == pytest.approx(0.7 * 0.0 + 0.3 * 1.0)


def test_value_targets_have_real_spread_for_a_spread_population():
    """The collapse detector's premise: dense ranks over N agents cannot be flat unless the
    population itself is."""
    n = 16
    scores = {2: {i: 100 * (i + 1) for i in range(n)}}
    result = _FakeResult(_matrices(scores, n), [4] * n,
                         {i: (2, 100 * (i + 1)) for i in range(n)})
    t = value_targets_from_result(result, n, value_blend=0.7)
    recs = [RecordedDecision(sample=make_sample(*_dummy_decision()), agent_idx=i, ante=2,
                             blind_idx=0, is_pvp=False, action_type="advance",
                             skip_offered=False, shortcut=True, n_legal=1)
            for i in range(n)]
    assign_value_targets(recs, t)
    assert np.std([r.sample.z for r in recs]) > 0.25


def _dummy_decision():
    """A minimal (game, legal, legal_keys, visits, encoder) tuple for `make_sample`."""
    game = BalatroGame(seed=SEED, ruleset="mlb")
    legal = game.legal_actions()[:1]
    return game, legal, [("play_blind",)], {}, get_encoder("mlb")


# ══════════════════════════════════════════════════════════════════════ the sample seam

def test_make_sample_policy_target_is_the_visit_distribution():
    game = BalatroGame(seed=SEED, ruleset="mlb")
    legal = game.legal_actions()
    keys = [(a["type"],) for a in legal]
    visits = {keys[0]: 30, keys[1]: 10}
    s = make_sample(game, legal, keys, visits, get_encoder("mlb"), z=0.25)
    assert s.target_policy.sum() == pytest.approx(1.0)
    assert s.target_policy[0] == pytest.approx(0.75)
    assert s.target_policy[1] == pytest.approx(0.25)
    assert s.z == 0.25
    assert s.action_features.shape[0] == len(legal)


def test_make_sample_is_uniform_on_a_forced_action_state():
    game, legal, keys, visits, enc = _dummy_decision()
    s = make_sample(game, legal, keys, visits, enc)
    assert s.target_policy.tolist() == [1.0]


def test_w1_sample_builder_satisfies_the_same_seam():
    """W1 published `train.sample.SampleBuilder` against this workstream's `sample_fn`
    signature (SETENC_NOTES §0.6). If that ever drifts, this fails here rather than 20
    minutes into a training run."""
    sample = pytest.importorskip("train.sample")
    game = BalatroGame(seed=SEED, ruleset="mlb")
    legal = game.legal_actions()
    keys = [(a["type"],) for a in legal]
    builder = sample.SampleBuilder(get_encoder("mlb"), rng=np.random.default_rng(0))
    s = builder(game, legal, keys, {keys[0]: 5}, None, 0.5)
    assert s.z == 0.5
    assert s.target_policy.sum() == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════ the record hook

def test_record_hook_collects_one_sample_per_decision():
    enc = get_encoder("mlb")
    net = PolicyValueNet(obs_dim=enc.dim, hidden=32, n_res_blocks=1)
    player = MCTSPlayer(policy=make_policy(net, encoder=enc), config=MCTSConfig(num_simulations=4),
                        rng=np.random.default_rng(0), no_action={"type": "advance"})
    collector = SampleCollector(0, enc)
    player.record_hook = collector

    game = BalatroGame(seed=SEED, ruleset="mlb")
    for _ in range(8):
        game.step(player.act(game))

    assert len(collector.records) == 8
    assert all(0 < r.sample.target_policy.sum() < 1.0 + 1e-5 for r in collector.records)
    assert any(r.skip_offered for r in collector.records), "BLIND_SELECT offers skip_blind"
    assert {r.action_type for r in collector.records} <= {
        "play", "discard", "buy", "sell_joker", "reroll", "leave_shop", "play_blind",
        "skip_blind", "reroll_boss", "use_consumable", "pick_booster", "skip_booster",
        "advance"}


def test_record_hook_is_free_when_unset():
    """The default must cost nothing: `MCTSPlayer` without a hook behaves exactly as it did
    in Phase 3."""
    enc = get_encoder("mlb")
    net = PolicyValueNet(obs_dim=enc.dim, hidden=32, n_res_blocks=1)
    kw = dict(policy=make_policy(net, encoder=enc), config=MCTSConfig(num_simulations=4),
              no_action={"type": "advance"})
    a = MCTSPlayer(rng=np.random.default_rng(7), **kw)
    b = MCTSPlayer(rng=np.random.default_rng(7), **kw)
    b.record_hook = SampleCollector(0, enc)

    g1, g2 = BalatroGame(seed=SEED, ruleset="mlb"), BalatroGame(seed=SEED, ruleset="mlb")
    for _ in range(6):
        g1.step(a.act(g1))
        g2.step(b.act(g2))
    assert g1.state_signature() == g2.state_signature()


def test_collector_cap_stops_recording_but_not_playing():
    enc = get_encoder("mlb")
    net = PolicyValueNet(obs_dim=enc.dim, hidden=32, n_res_blocks=1)
    player = MCTSPlayer(policy=make_policy(net, encoder=enc), config=MCTSConfig(num_simulations=4),
                        rng=np.random.default_rng(0), no_action={"type": "advance"})
    collector = SampleCollector(0, enc, max_records=3)
    player.record_hook = collector
    game = BalatroGame(seed=SEED, ruleset="mlb")
    for _ in range(8):
        game.step(player.act(game))
    assert len(collector.records) == 3 and collector.dropped == 5


# ══════════════════════════════════════════════════════════════════════ the population

def test_population_is_a_pure_function_of_its_arguments():
    cfg = PopulationConfig(n_agents=8, m_current=4, p_history=2, sims=40)
    a = build_population(cfg, generation=3, history=CheckpointHistory(2), base_seed=1)
    b = build_population(cfg, generation=3, history=CheckpointHistory(2), base_seed=1)
    assert [m.describe() for m in a] == [m.describe() for m in b]
    c = build_population(cfg, generation=4, history=CheckpointHistory(2), base_seed=1)
    assert [m.seed for m in c] != [m.seed for m in a], "generations must not repeat seeds"


def test_only_current_net_seats_are_marked_as_sample_producers():
    cfg = PopulationConfig(n_agents=8, m_current=3, p_history=2, sims=40)
    members = build_population(cfg, 0, CheckpointHistory(2))
    assert sum(m.is_current for m in members) == 3
    assert all(m.add_noise for m in members if m.is_current)


def test_population_is_heterogeneous_even_at_generation_zero():
    """No checkpoints exist yet, so the only axes are budget and root noise. If the seats
    were identical the N x N matrix would be all ties and the value target would be flat."""
    cfg = PopulationConfig(n_agents=6, m_current=3, sims=40, sims_budgets=(1.0, 0.5, 1.5))
    members = build_population(cfg, 0, CheckpointHistory(2))
    assert len({m.sims for m in members}) > 1
    assert len({m.seed for m in members}) == 6


def test_history_checkpoints_take_the_opponent_seats(tmp_path):
    hist = CheckpointHistory(capacity=2)
    for gen in (1, 2):
        p = tmp_path / f"ckpt_gen{gen:04d}.pt"
        p.write_bytes(b"x")
        hist.add(p, gen)
    cfg = PopulationConfig(n_agents=6, m_current=2, p_history=2, sims=40, anchor_frac=0)
    members = build_population(cfg, 3, hist)
    opponents = [m for m in members if not m.is_current]
    assert len(opponents) == 4
    assert all(m.checkpoint is not None for m in opponents)
    assert {m.generation for m in opponents} == {1, 2}, "newest first, round-robin"


def test_history_drops_pruned_files_instead_of_crashing(tmp_path):
    hist = CheckpointHistory(capacity=3)
    alive = tmp_path / "a.pt"
    alive.write_bytes(b"x")
    hist.add(tmp_path / "gone.pt", 1)
    hist.add(alive, 2)
    assert [e["generation"] for e in hist.existing()] == [2]
    members = build_population(
        PopulationConfig(n_agents=4, m_current=2, sims=10, anchor_frac=0), 3, hist)
    assert all(m.checkpoint == str(alive) for m in members if not m.is_current)


def test_history_state_dict_round_trips():
    h = CheckpointHistory(2)
    h.add("a.pt", 1)
    h.add("b.pt", 2)
    h.add("c.pt", 3)
    assert [e["generation"] for e in h.entries] == [2, 3]     # capacity honoured
    h2 = CheckpointHistory(9)
    h2.load_state_dict(h.state_dict())
    assert h2.entries == h.entries and h2.capacity == 2


# ══════════════════════════════════════════════════════════ external-target objective

def test_vanilla_boss_target_is_the_engine_formula_and_rises_with_the_ante():
    vals = [vanilla_boss_target(a) for a in range(1, 9)]
    assert vals == sorted(vals) and vals[0] == 600 and vals[7] == 100_000
    assert vanilla_boss_target(3, deck="b_plasma") == 2 * vanilla_boss_target(3)


def test_load_target_fn_reports_which_table_it_used_and_matches_the_engine():
    """W4's `eval/targets.py` is the campaign's table; the local fallback is the same
    formula. Whichever is in play, `target_fn(game)` is W4's shared signature and the
    number must be the vanilla Boss amount for that ante."""
    fn, source = load_target_fn("vanilla_boss", deck="b_red", stake=1, floor_frac=0)
    game = BalatroGame(seed=SEED, ruleset="mlb")
    game.ante = 3
    assert callable(fn) and int(fn(game, None)) == vanilla_boss_target(3)
    assert "targets.py" in source or source == "selfplay.vanilla_boss_target"


def test_target_multiplier_scales_the_table():
    fn, source = load_target_fn("vanilla_boss", deck="b_red", stake=1, multiplier=0.5,
                                floor_frac=0)
    game = BalatroGame(seed=SEED, ruleset="mlb")
    game.ante = 4
    assert int(fn(game, None)) == int(vanilla_boss_target(4) * 0.5)
    assert "x0.5" in source


def test_the_default_target_is_the_mirror_target_with_a_floor():
    """A raw `own_big_blind` target is 0 whenever the agent SKIPPED its Big blind, which
    makes the Nemesis free again — the overnight degeneracy, reached by a different road
    (measured: skip rate 77.5%, every Nemesis cleared, z mean 0.84). The floor is the fix."""
    from train.selfplay import big_blind_floor
    fn, source = load_target_fn(deck="b_red", stake=1)
    game = BalatroGame(seed=SEED, ruleset="mlb")
    game.ante = 4
    assert "own_big_blind" in source and "floor" in source
    assert int(fn(game, {})) == big_blind_floor(game)            # skipped: the floor bites
    assert int(fn(game, {4: 99_999})) == 99_999                  # played well: mirror wins
    assert big_blind_floor(game) == 7_500                        # ante 4 Big blind, White


def test_the_floor_can_be_switched_off_and_scaled():
    from train.selfplay import big_blind_floor
    game = BalatroGame(seed=SEED, ruleset="mlb")
    game.ante = 4
    raw, _ = load_target_fn(deck="b_red", stake=1, floor_frac=0)
    assert int(raw(game, {})) == 0
    half, src = load_target_fn(deck="b_red", stake=1, floor_frac=0.5)
    assert int(half(game, {})) == big_blind_floor(game) // 2
    assert "floor=0.5" in src


def test_the_floor_scales_with_the_deck():
    from train.selfplay import big_blind_floor
    plasma = BalatroGame(seed=SEED, deck_key="b_plasma", ruleset="mlb")
    red = BalatroGame(seed=SEED, deck_key="b_red", ruleset="mlb")
    plasma.ante = red.ante = 3
    assert big_blind_floor(plasma) == 2 * big_blind_floor(red)   # Plasma ante_scaling = 2


def test_external_outcome_charges_the_life_the_solo_engine_does_not():
    """This is the whole interim objective in one assertion: at a resolved Nemesis, being
    under the target must be valued strictly below being over it."""
    outcome = external_outcome_for(lambda g, b=None: 1_000, starting_lives=4, horizon_antes=4)
    game = BalatroGame(seed=SEED, ruleset="mlb")
    game.state = State.ROUND_EVAL
    game.current_blind.is_pvp = True
    game.ante = 2
    game.chips_scored = 5_000
    over = outcome.value(game)
    game.chips_scored = 10
    under = outcome.value(game)
    assert under < over


def test_solo_external_episode_actually_loses_a_life_when_short_of_target():
    enc = get_encoder("mlb")
    net = PolicyValueNet(obs_dim=enc.dim, hidden=32, n_res_blocks=1)
    player = MCTSPlayer(policy=make_policy(net, encoder=enc), config=MCTSConfig(num_simulations=4),
                        rng=np.random.default_rng(0), no_action={"type": "advance"},
                        outcome=external_outcome_for(lambda g, b=None: 10**9, starting_lives=20))
    game = BalatroGame(seed=SEED, ruleset="mlb")
    # A cold net fails most regular blinds, and each of those already costs a life; give it
    # enough lives to actually REACH a Nemesis, which is what this test is about.
    game.lives = 20
    ep = play_solo_external_episode(game, player, enc, lambda g, b=None: 10**9,
                                    max_decisions=400, max_antes=4, starting_lives=20)
    assert ep.nemesis_results, "the episode must reach at least one Nemesis"
    assert all(n["lost_life"] for n in ep.nemesis_results), "an impossible target costs a life"
    assert ep.final_lives < 20
    assert ep.records and all(0.0 <= r.sample.z <= 1.0 for r in ep.records)


def test_solo_external_episode_keeps_its_lives_against_a_trivial_target():
    enc = get_encoder("mlb")
    net = PolicyValueNet(obs_dim=enc.dim, hidden=32, n_res_blocks=1)
    player = MCTSPlayer(policy=make_policy(net, encoder=enc), config=MCTSConfig(num_simulations=4),
                        rng=np.random.default_rng(0), no_action={"type": "advance"})
    game = BalatroGame(seed=SEED, ruleset="mlb")
    game.lives = 20
    ep = play_solo_external_episode(game, player, enc, lambda g, b=None: 0,
                                    max_decisions=400, max_antes=3, starting_lives=20)
    assert ep.nemesis_results
    assert all(not n["lost_life"] for n in ep.nemesis_results)


def test_the_record_hook_is_always_disarmed_after_a_solo_episode():
    enc = get_encoder("mlb")
    net = PolicyValueNet(obs_dim=enc.dim, hidden=32, n_res_blocks=1)
    player = MCTSPlayer(policy=make_policy(net, encoder=enc), config=MCTSConfig(num_simulations=4),
                        rng=np.random.default_rng(0), no_action={"type": "advance"})
    play_solo_external_episode(BalatroGame(seed=SEED, ruleset="mlb"), player, enc,
                               lambda g, b=None: 0, max_decisions=40, max_antes=2)
    assert player.record_hook is None


# ══════════════════════════════════════════════════════════════════════ the generation loop

def test_one_tournament_generation_produces_labelled_samples_and_metrics():
    trainer = MLBTrainer(tiny_cfg(), tiny_mlb())
    m = trainer.run_generation()

    assert m.n_samples > 0
    assert m.episodes == 4                      # 1 seed x 4 agents
    assert trainer.generation == 1
    assert len(trainer.buffer) == m.n_samples
    assert 0.0 <= m.value_target_mean <= 1.0
    assert m.value_target_sd > 0.0, "a tournament objective must not be flat"
    assert m.skip_opportunities > 0 and 0.0 <= m.skip_rate <= 1.0
    assert m.distinct_joker_sets >= 1
    assert not math.isnan(m.tie_fraction)
    assert m.wall_clock_s > 0 and m.ep_per_min > 0
    assert m.train_steps >= 1


def test_generation_metrics_flag_a_collapse():
    from train import GenerationMetrics
    assert GenerationMetrics(value_target_sd=0.07).collapsed
    assert not GenerationMetrics(value_target_sd=0.30).collapsed
    assert "COLLAPSE" in GenerationMetrics(value_target_sd=0.07).console_line()


def test_external_objective_generation_runs():
    trainer = MLBTrainer(tiny_cfg(), tiny_mlb(objective="external",
                                              episodes_per_generation=2))
    m = trainer.run_generation()
    assert m.episodes == 2 and m.n_samples > 0
    assert m.value_target_sd >= 0.0
    assert trainer.target_source


def test_stop_check_between_tournaments_is_honoured():
    trainer = MLBTrainer(tiny_cfg(), tiny_mlb(seeds_per_generation=4))
    m = trainer.run_generation(stop_check=lambda: True)
    assert m.episodes == 0 and m.n_samples == 0
    assert trainer.generation == 1               # the generation still closed out cleanly


# ══════════════════════════════════════════════════════════════════════ checkpoint / resume

def test_checkpoint_carries_the_generation_counter_and_the_opponent_history(tmp_path):
    trainer = MLBTrainer(tiny_cfg(), tiny_mlb())
    trainer.run_generation()
    ckpt_path = tmp_path / "ckpt.pt"
    trainer.history.add(ckpt_path, trainer.generation)
    save_checkpoint(ckpt_path, trainer.state_dict())

    ck = load_checkpoint(ckpt_path)
    assert ck["mlb"]["generation"] == 1
    assert ck["mlb"]["history"]["entries"][0]["generation"] == 1
    assert ck["mlb"]["config"]["value_blend"] == 0.7

    resumed = MLBTrainer.from_checkpoint(ck)
    assert resumed.generation == 1
    assert resumed.history.entries == trainer.history.entries
    assert len(resumed.buffer) == len(trainer.buffer)
    for a, b in zip(resumed.net.parameters(), trainer.net.parameters()):
        assert (a == b).all()


def test_resume_faces_the_same_population_it_would_have(tmp_path):
    trainer = MLBTrainer(tiny_cfg(), tiny_mlb())
    trainer.run_generation()
    for gen in (1, 2):
        p = tmp_path / f"ckpt_gen{gen:04d}.pt"
        p.write_bytes(b"x")
        trainer.history.add(p, gen)
    expected = [m.describe() for m in
                build_population(trainer.pop_cfg, trainer.generation, trainer.history,
                                 base_seed=trainer.cfg.seed)]

    path = tmp_path / "latest.pt"
    save_checkpoint(path, trainer.state_dict())
    resumed = MLBTrainer.from_checkpoint(load_checkpoint(path))
    got = [m.describe() for m in
           build_population(resumed.pop_cfg, resumed.generation, resumed.history,
                            base_seed=resumed.cfg.seed)]
    assert got == expected


def test_resume_continues_the_rng_stream(tmp_path):
    """Two more generations from a resume must draw the same game seeds an uninterrupted
    run would have drawn — otherwise `--resume` is a new experiment wearing old weights."""
    a = MLBTrainer(tiny_cfg(), tiny_mlb())
    a.run_generation()
    path = tmp_path / "latest.pt"
    save_checkpoint(path, a.state_dict())
    straight = [a._episode_seed() for _ in range(4)]

    b = MLBTrainer.from_checkpoint(load_checkpoint(path))
    assert [b._episode_seed() for _ in range(4)] == straight


def test_resume_refuses_a_checkpoint_from_a_different_experiment(tmp_path):
    trainer = MLBTrainer(tiny_cfg(encoder="mlb"), tiny_mlb())
    path = tmp_path / "c.pt"
    save_checkpoint(path, trainer.state_dict())
    ck = load_checkpoint(path)
    with pytest.raises(ValueError, match="encoder"):
        MLBTrainer.from_checkpoint(ck, overrides={"encoder": "v7"})


def test_a_train_mlb_checkpoint_loads_as_a_policy(tmp_path):
    """The population's past selves are loaded with `mcts.load_policy`, so a generation
    checkpoint has to be readable by it — that is why `MLBTrainer` reuses `ColdTrainer`'s
    payload rather than inventing one."""
    from mcts import load_policy
    trainer = MLBTrainer(tiny_cfg(), tiny_mlb())
    path = tmp_path / "ckpt_gen0001.pt"
    save_checkpoint(path, trainer.state_dict(include_buffer=False))
    policy = load_policy(str(path), device="cpu")
    priors, value = policy(BalatroGame(seed=SEED, ruleset="mlb"))
    assert priors and isinstance(value, float)


# ══════════════════════════════════════════════════════════════════════ the CLI: pause / resume

@pytest.fixture
def cli():
    """`scripts/train_mlb.py` as a module, with the signal handlers it installs restored
    afterwards so a paused test cannot eat pytest's own Ctrl+C."""
    import signal
    import sys
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import train_mlb                                   # noqa: WPS433
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield train_mlb
    finally:
        for s, h in saved.items():
            signal.signal(s, h)


def _cli_args(run_dir: Path, name: str, **extra) -> list:
    args = ["--run-dir", str(run_dir), "--run-name", name, "--device", "cpu",
            "--encoder", "mlb", "--n-agents", "4", "--m-current", "2",
            "--seeds-per-gen", "1", "--max-ante", "3", "--sims", "4", "--leaf-batch", "4",
            "--hidden", "32", "--n-res-blocks", "1", "--batch-size", "4",
            "--min-buffer", "4", "--buffer-capacity", "500", "--minutes", "5"]
    for k, v in extra.items():
        args += [f"--{k.replace('_', '-')}"] + ([] if v is None else [str(v)])
    return args


def _jsonl(run_dir: Path, name: str) -> list:
    import json
    path = run_dir / name / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_cli_runs_a_generation_and_writes_a_checkpoint(cli, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_mlb.py"] + _cli_args(tmp_path, "r1", generations=1))
    assert cli.main() == 0
    run_dir = tmp_path / "r1"
    assert (run_dir / "latest.pt").is_file()
    assert list(run_dir.glob("ckpt_gen*.pt"))
    kinds = [r["kind"] for r in _jsonl(tmp_path, "r1")]
    assert kinds.count("generation") == 1 and "checkpoint" in kinds and kinds[-1] == "summary"


def test_pause_file_stops_the_loop_and_prints_the_resume_command(cli, tmp_path, capsys,
                                                                 monkeypatch):
    run_dir = tmp_path / "r2"
    run_dir.mkdir(parents=True)
    (run_dir / "PAUSE").write_text("")
    monkeypatch.setattr("sys.argv", ["train_mlb.py"] + _cli_args(tmp_path, "r2", generations=5))
    assert cli.main() == 0

    out = capsys.readouterr().out
    assert "Stopped (PAUSE)" in out
    assert "--resume" in out and str(run_dir / "latest.pt") in out
    records = _jsonl(tmp_path, "r2")
    assert records[-1]["kind"] == "summary" and records[-1]["stop_reason"] == "PAUSE"
    assert not any(r["kind"] == "generation" for r in records)
    assert (run_dir / "latest.pt").is_file(), "a pause must still checkpoint"


def test_resume_continues_the_generation_counter_and_clears_the_pause(cli, tmp_path,
                                                                     monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_mlb.py"] + _cli_args(tmp_path, "r3", generations=1))
    assert cli.main() == 0
    run_dir = tmp_path / "r3"
    (run_dir / "PAUSE").write_text("")

    monkeypatch.setattr("sys.argv", ["train_mlb.py", "--resume", str(run_dir / "latest.pt"),
                                     "--run-dir", str(tmp_path), "--run-name", "r3",
                                     "--device", "cpu", "--minutes", "5", "--generations", "1"])
    assert cli.main() == 0
    assert not (run_dir / "PAUSE").exists(), "--resume clears the pause that stopped the run"

    records = _jsonl(tmp_path, "r3")
    gens = [r["generation"] for r in records if r["kind"] == "generation"]
    assert gens == [0, 1], "the resumed run must continue, not restart"
    configs = [r for r in records if r["kind"] == "config"]
    assert configs[-1]["start_generation"] == 1
    assert configs[-1]["resumed_from"]


def test_prune_never_deletes_a_checkpoint_the_population_is_still_playing(cli, tmp_path):
    d = tmp_path / "ck"
    d.mkdir()
    paths = []
    for gen in range(6):
        p = d / f"ckpt_gen{gen:04d}.pt"
        p.write_bytes(b"x")
        paths.append(p)
    protect = {str(paths[0].resolve()), str(paths[1].resolve())}
    cli.prune_checkpoints(d, keep=2, protect=protect)
    survivors = {p.name for p in d.glob("ckpt_gen*.pt")}
    assert {paths[0].name, paths[1].name, paths[4].name, paths[5].name} <= survivors
    assert paths[2].name not in survivors


# ══════════════════════════════════════════════════════════════ W3 trajectory logging

def _replay_pkg():
    tournament_module()                      # puts the repo root on sys.path
    return pytest.importorskip("replay.log", reason="W3's replay has not landed")


def test_tournament_generation_logs_one_trajectory_per_agent(tmp_path):
    """W3's `TrajectoryLogger` is one game per `begin()/end()` pair and a tournament is N
    games, so the loop opens one logger per AGENT at `Tournament.on_fanout` — the only
    moment the games exist and nobody has acted yet — and `on_agent_done` closes each one.
    Every agent is logged, not just the sample-producing ones (brief §0.3: log every
    trajectory); `meta.is_current` says which is which."""
    import json
    log = _replay_pkg()
    path = tmp_path / "trajectories.jsonl"
    mlb = tiny_mlb()
    trainer = MLBTrainer(tiny_cfg(), mlb)
    trainer.run_generation(traj_logger_factory=lambda seed: log.TrajectoryLogger(str(path)))

    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == mlb.n_agents * mlb.seeds_per_generation
    assert {l["meta"]["agent"] for l in lines} == set(range(mlb.n_agents))
    assert sum(l["meta"]["is_current"] for l in lines) == mlb.m_current
    for line in lines:
        assert line["kind"] == "episode"
        assert line["ruleset"] == "mlb" and line["actions"]
        assert line["meta"]["source"] == "train_mlb"
        assert line["outcome"]["objective"] == "tournament"
        assert line["outcome"]["reason"] in ("died", "eliminated", "finished")


def test_external_generation_logs_one_trajectory_per_episode(tmp_path):
    import json
    log = _replay_pkg()
    path = tmp_path / "solo.jsonl"
    trainer = MLBTrainer(tiny_cfg(), tiny_mlb(objective="external",
                                              episodes_per_generation=2))
    trainer.run_generation(traj_logger_factory=lambda seed: log.TrajectoryLogger(str(path)))

    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert all(l["outcome"]["objective"] == "external" for l in lines)
    assert all("nemesis" in l["outcome"] for l in lines)


def test_logged_tournament_trajectories_replay_exactly(tmp_path):
    """The point of logging: W3's replayer must reproduce every recorded signature. This is
    also the strongest possible check that the runner's `on_step` hook sees EVERY mutation —
    the agent's actions, the no-progress guard's forced action, `_cash_out`'s advance, and
    the synthetic `__lose_life__` the cross-agent life rule performs outside `step()`."""
    log = _replay_pkg()
    tournament_module()
    replay = pytest.importorskip("replay.replay", reason="W3's replayer has not landed")
    path = tmp_path / "t.jsonl"
    trainer = MLBTrainer(tiny_cfg(), tiny_mlb(m_current=1, n_agents=2))
    trainer.run_generation(traj_logger_factory=lambda seed: log.TrajectoryLogger(str(path)))

    import json
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert lines
    for line in lines:
        # `replay_line` raises ReplayMismatch on the first divergent signature and returns
        # the rebuilt game otherwise.
        rebuilt = replay.replay_line(line)
        assert rebuilt.lives == line["final_state"]["lives"]
        assert rebuilt.ante == line["final_state"]["ante"]


# ══════════════════════════════════════════════════════════════════════ W1: the set encoder

def test_the_loop_runs_on_w1s_set_encoder_and_produces_v2_samples():
    """Tagg's decision §0.1: the set encoder lands BEFORE the first real training run, so
    the tournament loop has to work on it with no branch of its own. It does, because the
    only encoder-aware call site is `sample_fn` and `MLBTrainer` takes W1's `SampleBuilder`
    straight from `ColdTrainer`."""
    pytest.importorskip("mcts.encoder_set", reason="W1's set encoder has not landed")
    trainer = MLBTrainer(tiny_cfg(encoder="set"), tiny_mlb())
    m = trainer.run_generation()

    assert m.n_samples > 0 and m.value_target_sd > 0.0
    batch = trainer.buffer.sample(4, rng=np.random.default_rng(0))
    assert all(getattr(s, "version", 1) == 2 for s in batch)
    assert all(isinstance(s.obs, dict) for s in batch)
    assert m.train_steps >= 1, "Trainer.step must accept the v2 batch"


def test_rank_from_the_ante_matrix_agrees_with_the_reference_ranking():
    """`_rank_from_matrix` reads the tournament's OWN `AnteMatrix.rank` so the value target
    and the tournament's published metrics can never disagree about who came where. This
    pins it against the independent implementation `normalized_ranks`."""
    from train.selfplay import _rank_from_matrix
    n = 6
    scores = {0: 900, 1: 100, 2: 500, 3: 500, 5: 10}      # agent 4 absent
    am = _matrices({2: scores}, n)[0]
    got = _rank_from_matrix(am, n)
    want = normalized_ranks([scores.get(i, np.nan) for i in range(n)])
    assert np.allclose(got[~np.isnan(got)], want[~np.isnan(want)])
    assert np.isnan(got[4]) and np.isnan(want[4])


# ══════════════════════════════════════════════════════════════════ leaf-batch plumbing

@pytest.mark.parametrize("leaf_batch,batched", [(1, True), (16, False), (16, True)])
def test_leaf_batch_paths_all_produce_steppable_actions(leaf_batch, batched):
    """`MCTS._drive` answers leaf requests ONE AT A TIME by design (it is what keeps
    `run_gumbel` byte-identical to the pre-W3 implementation), so `leaf_batch > 1` alone
    changes the tree shape without ever batching the forward pass. `batch_leaf_eval` routes
    such a search through `BatchedSearch`, which does. All three combinations must play."""
    enc = get_encoder("mlb")
    net = PolicyValueNet(obs_dim=enc.dim, hidden=32, n_res_blocks=1)
    policy = make_policy(net, encoder=enc, batched=True)
    # NOTE the doubled `leaf_batch`: `MCTSPlayer.__post_init__` lets its OWN `leaf_batch`
    # field (default 1) overwrite the config's, so passing only `config=MCTSConfig(
    # leaf_batch=16)` silently gives L=1. See TRAIN_NOTES sec.9.
    player = MCTSPlayer(policy=policy, config=MCTSConfig(num_simulations=8,
                                                         leaf_batch=leaf_batch),
                        leaf_batch=leaf_batch,
                        rng=np.random.default_rng(0), no_action={"type": "advance"},
                        batch_leaf_eval=batched)
    game = BalatroGame(seed=SEED, ruleset="mlb")
    for _ in range(6):
        action = player.act(game)
        assert isinstance(action, dict)
        game.step(action)
    assert player.searches > 0
    if leaf_batch > 1 and batched:
        assert policy.leaves > 0, "the batched route must reach evaluate_many"


def test_leaf_batch_one_is_the_exact_serial_search():
    """The default. `batch_leaf_eval` must be a no-op at L=1, so the training loop's search
    is the reference search."""
    enc = get_encoder("mlb")
    net = PolicyValueNet(obs_dim=enc.dim, hidden=32, n_res_blocks=1)
    policy = make_policy(net, encoder=enc, batched=True)
    keys = []
    for batched in (False, True):
        p = MCTSPlayer(policy=policy, config=MCTSConfig(num_simulations=8, leaf_batch=1),
                       leaf_batch=1,
                       rng=np.random.default_rng(3), no_action={"type": "advance"},
                       batch_leaf_eval=batched)
        g = BalatroGame(seed=SEED, ruleset="mlb")
        seq = []
        for _ in range(6):
            a = p.act(g)
            seq.append(str(a))
            g.step(a)
        keys.append((seq, g.state_signature()))
    assert keys[0] == keys[1]


def test_the_leaf_batch_field_overrides_the_config_and_that_is_a_trap():
    """Pinned because it bit this workstream: `MCTSPlayer.__post_init__` compares its own
    `leaf_batch` FIELD (default 1) against `config.leaf_batch` and rewrites the config to
    match the field. So `MCTSPlayer(config=MCTSConfig(leaf_batch=16))` silently runs at
    L=1. `make_player` and `train.population.instantiate` both pass BOTH, which is why the
    tournament path is unaffected."""
    enc = get_encoder("mlb")
    net = PolicyValueNet(obs_dim=enc.dim, hidden=32, n_res_blocks=1)
    policy = make_policy(net, encoder=enc, batched=True)
    only_cfg = MCTSPlayer(policy=policy, config=MCTSConfig(leaf_batch=16))
    assert only_cfg.config.leaf_batch == 1 and only_cfg._batched is None
    both = MCTSPlayer(policy=policy, config=MCTSConfig(leaf_batch=16), leaf_batch=16)
    assert both.config.leaf_batch == 16 and both._batched is not None


# ══════════════════════════════════════════════════════════════════════ scripted anchors

def test_anchors_take_opponent_seats_never_current_net_seats():
    """The anchors' whole job is to be somebody the current net can LOSE the rank
    comparison to. They must never eat a sample-producing seat."""
    cfg = PopulationConfig(n_agents=16, m_current=8, sims=40, anchor_frac=0.25)
    members = build_population(cfg, 0, CheckpointHistory(4))
    anchors = [m for m in members if m.is_anchor]
    assert len(anchors) == 4
    assert all(not m.is_current for m in anchors)
    assert sum(m.is_current for m in members) == 8
    assert all(m.idx >= 8 for m in anchors)
    archetypes = {tuple(kv for kv in m.spec if kv[0] != "name") for m in anchors}
    assert len(archetypes) == 2, "both anchor archetypes must be present"


def test_anchor_fraction_is_configurable_and_can_be_switched_off():
    base = dict(n_agents=8, m_current=4, sims=40)
    assert not any(m.is_anchor for m in
                   build_population(PopulationConfig(**base, anchor_frac=0), 0, None))
    assert sum(m.is_anchor for m in
               build_population(PopulationConfig(**base, anchor_frac=0.5), 0, None)) == 4
    # It can never displace a current-net seat, however greedy the fraction.
    greedy = build_population(PopulationConfig(**base, anchor_frac=1.0), 0, None)
    assert sum(m.is_anchor for m in greedy) == 4 and sum(m.is_current for m in greedy) == 4


def test_anchors_instantiate_as_scripted_players_that_never_skip():
    """`mlb_match_demo`'s scripted policy answers BLIND_SELECT with `play_blind`
    unconditionally (mlb_match_demo.py:126), which is exactly the property that makes an
    anchor a reference: it always pays the cost of playing the blind."""
    from train.population import instantiate
    enc = get_encoder("mlb")
    net = PolicyValueNet(obs_dim=enc.dim, hidden=32, n_res_blocks=1)
    policy = make_policy(net, encoder=enc, batched=True)
    members = build_population(PopulationConfig(n_agents=4, m_current=2, sims=4,
                                                anchor_frac=0.5), 0, None)
    players = instantiate(members, policy, encoder="mlb", leaf_batch=1)
    anchors = [p for m, p in zip(members, players) if m.is_anchor]
    assert len(anchors) == 2
    game = BalatroGame(seed=SEED, ruleset="mlb")
    assert all(a.act(game) == {"type": "play_blind"} for a in anchors)
    for a in anchors:
        a.reset()                       # the runner calls this before every run


def test_a_generation_with_anchors_reports_the_rank_referendum():
    trainer = MLBTrainer(tiny_cfg(), tiny_mlb(n_agents=6, m_current=2, anchor_frac=0.34))
    m = trainer.run_generation()
    assert 0.0 <= m.rank_current <= 1.0
    assert 0.0 <= m.rank_anchor <= 1.0
    assert "rank cur" in m.console_line()
    assert trainer.extra_metrics["population"]["n_anchors"] == 2


def test_anchors_produce_no_samples():
    """Learning from a scripted player would be imitation, not self-play."""
    trainer = MLBTrainer(tiny_cfg(), tiny_mlb(n_agents=6, m_current=2, anchor_frac=0.34))
    trainer.run_generation()
    # `m_current` seats x 1 seed is the only source; anchors have no `record_hook` at all.
    members, players = trainer.build_players()
    assert all(getattr(p, "record_hook", None) is None for p in players)
    assert all(not hasattr(p, "record_hook")
               for m, p in zip(members, players) if m.is_anchor)


# ══════════════════════════════════════════════════════ skip cap, blind-clear rate, --init

def test_skip_cap_drops_skip_blind_once_the_ante_allowance_is_spent():
    from train.population import SkipCap
    cap = SkipCap(1)
    game = BalatroGame(seed=SEED, ruleset="mlb")
    legal = game.legal_actions()
    assert any(a["type"] == "skip_blind" for a in legal)
    assert cap(game, legal) is legal, "the first skip of the ante is allowed"
    game.step({"type": "skip_blind"})                     # engine's own `skips` counter
    legal = game.legal_actions()
    filtered = cap(game, legal)
    assert any(a["type"] == "skip_blind" for a in legal)
    assert not any(a["type"] == "skip_blind" for a in filtered)
    assert len(filtered) == len(legal) - 1, "nothing else is touched"


def test_skip_cap_allowance_refreshes_each_ante():
    from train.population import SkipCap
    cap = SkipCap(1)
    game = BalatroGame(seed=SEED, ruleset="mlb")
    cap(game, game.legal_actions())
    game.step({"type": "skip_blind"})
    assert not any(a["type"] == "skip_blind" for a in cap(game, game.legal_actions()))
    game.ante += 1                                        # a new ante: fresh allowance
    assert any(a["type"] == "skip_blind" for a in cap(game, game.legal_actions()))


def test_skip_cap_never_returns_an_empty_candidate_set():
    from train.population import SkipCap
    cap = SkipCap(0)
    game = BalatroGame(seed=SEED, ruleset="mlb")
    assert cap(game, [{"type": "skip_blind"}]) == [{"type": "skip_blind"}]


def test_a_capped_player_cannot_play_a_second_skip_in_an_ante():
    """End to end through `MCTSPlayer.legal_filter`: the search still explores skipping, but
    the CHOSEN action and the recorded policy target are restricted to the allowed set."""
    from train.population import SkipCap
    enc = get_encoder("mlb")
    net = PolicyValueNet(obs_dim=enc.dim, hidden=32, n_res_blocks=1)
    player = MCTSPlayer(policy=make_policy(net, encoder=enc, batched=True),
                        config=MCTSConfig(num_simulations=4), rng=np.random.default_rng(0),
                        no_action={"type": "advance"}, legal_filter=SkipCap(1))
    collector = SampleCollector(0, enc)
    player.record_hook = collector
    game = BalatroGame(seed=SEED, ruleset="mlb")
    skips_before = game.skips
    for _ in range(3):                        # Small, Big, then the Boss
        game.step(player.act(game))
    assert game.skips - skips_before <= 1
    for r in collector.records:
        assert r.action_type != "skip_blind" or r.ante  # recorded at all
    player.reset()
    assert player.legal_filter._ante is None


def test_the_skip_cap_only_binds_the_sample_producing_seats():
    """Opponents are there to be a reference, not to be trained; constraining THEM would
    change what the rank target measures."""
    from train.population import SkipCap, instantiate
    enc = get_encoder("mlb")
    net = PolicyValueNet(obs_dim=enc.dim, hidden=32, n_res_blocks=1)
    members = build_population(PopulationConfig(n_agents=6, m_current=2, sims=4,
                                                anchor_frac=0), 0, None)
    players = instantiate(members, make_policy(net, encoder=enc, batched=True),
                          encoder="mlb", leaf_batch=1, max_skips_per_ante=1)
    for m, p in zip(members, players):
        assert isinstance(p.legal_filter, SkipCap) is m.is_current


def test_blind_clear_rate_is_measured_at_every_non_pvp_round_eval():
    trainer = MLBTrainer(tiny_cfg(), tiny_mlb())
    m = trainer.run_generation()
    assert m.blinds_played > 0
    assert 0.0 <= m.blind_clear_rate <= 1.0
    assert "clear" in m.console_line()


def test_the_skip_cap_anneals_off_once_the_net_can_clear_blinds():
    trainer = MLBTrainer(tiny_cfg(), tiny_mlb(max_skips_per_ante=1,
                                              skip_cap_anneal_clear_rate=0.5))
    assert trainer.effective_skip_cap() == 1
    trainer.clear_rate_ema = 0.9
    assert trainer.effective_skip_cap() is None
    trainer.clear_rate_ema = 0.1
    assert trainer.effective_skip_cap() == 1


def test_the_skip_cap_also_anneals_on_a_generation_count():
    trainer = MLBTrainer(tiny_cfg(), tiny_mlb(max_skips_per_ante=1,
                                              skip_cap_anneal_generations=2))
    assert trainer.effective_skip_cap() == 1
    trainer.generation = 2
    assert trainer.effective_skip_cap() is None


def test_a_generation_with_a_skip_cap_actually_caps_the_skip_rate():
    capped = MLBTrainer(tiny_cfg(), tiny_mlb(max_skips_per_ante=0)).run_generation()
    assert capped.skip_rate == 0.0, "a 0/ante cap must remove skipping entirely"
    assert capped.n_samples > 0


def test_the_clear_rate_ema_survives_a_checkpoint_round_trip(tmp_path):
    trainer = MLBTrainer(tiny_cfg(), tiny_mlb(max_skips_per_ante=1))
    trainer.run_generation()
    assert trainer.clear_rate_ema is not None
    path = tmp_path / "c.pt"
    save_checkpoint(path, trainer.state_dict())
    resumed = MLBTrainer.from_checkpoint(load_checkpoint(path))
    assert resumed.clear_rate_ema == trainer.clear_rate_ema
    assert resumed.effective_skip_cap() == trainer.effective_skip_cap()


def test_init_takes_weights_only(cli, tmp_path, monkeypatch):
    """`--init` is the Stage-A -> Stage-B hand-off: weights cross the seam, nothing else."""
    # The CLI has no --policy-hidden, so the donor must use the same head width the
    # Stage-B run will build (that mismatch is exactly what `--init` should refuse loudly).
    donor = MLBTrainer(tiny_cfg(policy_hidden=128), tiny_mlb())
    donor.run_generation()
    donor_path = tmp_path / "stage_a.pt"
    save_checkpoint(donor_path, donor.state_dict())

    monkeypatch.setattr("sys.argv", ["train_mlb.py"] + _cli_args(tmp_path, "b1", generations=1)
                        + ["--init", str(donor_path)])
    assert cli.main() == 0

    ck = load_checkpoint(tmp_path / "b1" / "ckpt_gen0001.pt")
    assert ck["mlb"]["generation"] == 1, "counters start from scratch, not from the donor"
    records = _jsonl(tmp_path, "b1")
    assert records[0]["start_generation"] == 0
    out_dir = tmp_path / "b1"
    assert (out_dir / "latest.pt").is_file()


def test_init_refuses_a_different_encoder(cli, tmp_path, monkeypatch):
    donor = MLBTrainer(tiny_cfg(encoder="v7"), tiny_mlb())
    path = tmp_path / "v7.pt"
    save_checkpoint(path, donor.state_dict(include_buffer=False))
    monkeypatch.setattr("sys.argv", ["train_mlb.py"] + _cli_args(tmp_path, "b2", generations=1)
                        + ["--init", str(path)])
    with pytest.raises(SystemExit, match="encoder"):
        cli.main()


def test_init_actually_transfers_the_weights(tmp_path):
    from train.loop import TrainConfig as TC
    donor = MLBTrainer(tiny_cfg(), tiny_mlb())
    donor.run_generation()                       # move the weights off their init
    path = tmp_path / "donor.pt"
    save_checkpoint(path, donor.state_dict(include_buffer=False))

    import sys
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import train_mlb
    fresh = MLBTrainer(tiny_cfg(seed=7), tiny_mlb())
    note = train_mlb.init_weights(fresh, str(path), "cpu")
    assert "Initialised weights" in note
    for a, b in zip(fresh.net.parameters(), donor.net.parameters()):
        assert (a == b).all()
    assert fresh.generation == 0 and len(fresh.buffer) == 0
