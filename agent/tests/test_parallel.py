"""
Phase 5 W1: multi-process self-play with a shared batched evaluator.

Four things are being pinned here, in order of how much they matter:

1. **The seam is the same seam.**  A worker encodes a leaf and the evaluator forwards it;
   both halves have to be the code the single-process policy already runs, or a parallel
   run is quietly a different agent.  `test_leaf_encoder_matches_policy_encode_leaf` and
   `test_forward_matches_batched_policy` assert bit-equality against
   `NNPolicy`/`SetNNPolicy` and `BatchedNNPolicy`/`BatchedSetNNPolicy`, for BOTH encoders.
2. **The transport does not corrupt anything.**  `LeafLayout` round-trips every array of a
   real leaf bit-exact, and `RemotePolicy` returns what a local policy returns.
3. **The result does not depend on the worker count.**  One worker and four workers play
   the same tournament: same matrices, same lives, same value targets, same samples.
4. **Failure is survivable.**  A worker killed mid-tournament costs its agents, not the
   run; PAUSE drains; and every checkpoint crosses the serial/parallel seam in both
   directions, because that is the whole point of the exercise (the live `real1` run has
   to be resumable onto all 16 cores).

Multiprocessing tests are slow on Windows (`spawn` re-imports torch per worker, ~2 s
each), so the pool-level tests use 1-2 workers, a 3-agent population and a 32-unit net.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from balatro_sim.game import BalatroGame, State
from mcts import MCTSConfig, MCTSPlayer, MLBOutcome, UniformPolicy, get_encoder
from mcts.player import build_net
from mcts.policy import make_policy
from parallel.forward import forward_leaves
from parallel.layout import LeafLayout
from parallel.leaf import LeafEncoder
from parallel.pool import PoolConfig, partition_agents
from parallel.protocol import LIVE_POLICY_ID
from train import MLBTrainConfig, MLBTrainer, TrainConfig, load_checkpoint, save_checkpoint
from train.parallel import ParallelMLBTrainer, assign_policy_ids, merge_trajectory_parts
from train.population import PopulationMember

SEED = "7I4M53DL"
ENCODERS = ["mlb", "set"]


# ══════════════════════════════════════════════════════════════════════ fixtures

def _game(seed: str = SEED) -> BalatroGame:
    return BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset="mlb")


def sample_games(n: int = 4, seed: str = SEED) -> list:
    """A few games in genuinely different states (so leaf sizes differ and the padding /
    chunking paths are actually exercised)."""
    games = [_game(seed)]
    g = _game(seed)
    rng = np.random.default_rng(0)
    while len(games) < n:
        legal = g.legal_actions()
        if not legal or g.state == State.GAME_OVER:
            break
        g.step(legal[int(rng.integers(len(legal)))])
        if g.legal_actions():
            games.append(g.clone())
    return games[:n]


def tiny_cfg(**kw) -> TrainConfig:
    base = dict(seed=1, sims=4, hidden=32, n_res_blocks=1, policy_hidden=16,
                set_res_blocks=1, encoder="set", ruleset="mlb", device="cpu",
                batch_size=4, min_buffer=4, buffer_capacity=500, max_decisions=300,
                max_antes=3, lives=4)
    base.update(kw)
    return TrainConfig(**base)


def tiny_mlb(**kw) -> MLBTrainConfig:
    base = dict(objective="tournament", n_agents=3, m_current=2, p_history=0,
                seeds_per_generation=1, max_ante=3, life_rule="paired", anchor_frac=0.34,
                sims_budgets=(1.0,), leaf_batch=1)
    base.update(kw)
    return MLBTrainConfig(**base)


def make_trainer(tmp_path, workers=1, device="cpu", **mlb_kw) -> ParallelMLBTrainer:
    return ParallelMLBTrainer(
        tiny_cfg(), tiny_mlb(**mlb_kw),
        pool_cfg=PoolConfig(n_workers=workers, evaluator_device=device),
        work_dir=str(tmp_path))


# ══════════════════════════════════════════════════════════ 1. the leaf-encoding seam

@pytest.mark.parametrize("encoder", ENCODERS)
def test_leaf_encoder_matches_policy_encode_leaf(encoder):
    """The worker's net-free `encode_leaf` IS the policy's, array for array."""
    enc = get_encoder(encoder)
    policy = make_policy(build_net(enc), device="cpu", encoder=enc, batched=True)
    caps = enc.caps.as_dict() if getattr(enc, "is_set", False) else None
    worker = LeafEncoder(encoder, caps=caps)

    for game in sample_games(4):
        want = policy.encode_leaf(game)
        got = worker.encode_leaf(game)
        assert (want is None) == (got is None)
        if want is None:
            continue
        assert [a for a in got[0]] == [a for a in want[0]]
        _assert_leaf_equal(got[1], want[1])
        _assert_leaf_equal(got[2], want[2])


def _assert_leaf_equal(got, want):
    if isinstance(want, dict):
        assert sorted(got) == sorted(want)
        for k in want:
            assert np.array_equal(got[k], want[k]), k
            assert got[k].dtype == want[k].dtype, k
    else:
        assert np.array_equal(got, want)
        assert got.dtype == want.dtype


def test_leaf_encoder_returns_none_when_there_are_no_legal_actions():
    game = _game()
    game.state = State.GAME_OVER
    assert LeafEncoder("mlb").encode_leaf(game) is None


# ══════════════════════════════════════════════════════════════ 2. the shm layout

@pytest.mark.parametrize("encoder", ENCODERS)
def test_layout_pack_unpack_round_trips_bit_exact(encoder):
    caps = get_encoder(encoder).caps.as_dict() if encoder == "set" else None
    worker = LeafEncoder(encoder, caps=caps)
    games = sample_games(4)
    layout = LeafLayout.from_prototype(*worker.prototype(games[0]))

    buf = np.zeros(1 << 22, dtype=np.uint8)
    offset = 0
    packed = []
    for game in games:
        legal, obs, acts = worker.encode_leaf(game)
        n = len(legal)
        written = layout.pack(buf, offset, obs, acts, n)
        assert written == layout.record_bytes(n)
        packed.append((offset, n, obs, acts))
        offset += written

    for offset, n, obs, acts in packed:
        got_obs, got_acts = layout.unpack(buf, offset, n)
        _assert_leaf_equal(got_obs, obs)
        _assert_leaf_equal(got_acts, acts)


def test_layout_records_do_not_overlap_and_are_aligned():
    worker = LeafEncoder("set", caps=get_encoder("set").caps.as_dict())
    layout = LeafLayout.from_prototype(*worker.prototype(_game()))
    for n in (1, 7, 64, 436, 512):
        size = layout.record_bytes(n)
        assert size % 8 == 0
        assert size >= (layout.obs_f_count + n * layout.act_f_width) * 4


def test_layout_describe_names_every_key_of_a_real_leaf():
    worker = LeafEncoder("set", caps=get_encoder("set").caps.as_dict())
    _legal, obs, acts = worker.encode_leaf(_game())
    d = LeafLayout.from_prototype(obs, acts).describe()
    assert set(d["obs_float_keys"]) | set(d["obs_int_keys"]) == set(obs)
    assert set(d["act_float_keys"]) | set(d["act_int_keys"]) == set(acts)


# ══════════════════════════════════════════════════════════════ 3. the forward half

@pytest.mark.parametrize("encoder", ENCODERS)
def test_forward_matches_batched_policy(encoder):
    """`forward_leaves(model, encoded...)` == `BatchedPolicy.evaluate_many(games)`."""
    enc = get_encoder(encoder)
    net = build_net(enc)
    policy = make_policy(net, device="cpu", encoder=enc, batched=True)
    caps = enc.caps.as_dict() if getattr(enc, "is_set", False) else None
    worker = LeafEncoder(encoder, caps=caps)

    games = sample_games(4)
    want = policy.evaluate_many(games)
    encoded = [worker.encode_leaf(g) for g in games]
    live = [i for i, e in enumerate(encoded) if e is not None]
    probs, values = forward_leaves(net, [encoded[i][1] for i in live],
                                   [encoded[i][2] for i in live],
                                   is_set=bool(getattr(enc, "is_set", False)),
                                   device="cpu")

    for j, i in enumerate(live):
        want_priors, want_value = want[i]
        got = worker.priors_from_logits(encoded[i][0], probs[j])
        assert got == want_priors
        assert float(values[j]) == want_value


@pytest.mark.parametrize("encoder", ENCODERS)
def test_forward_batch_matches_single_leaf(encoder):
    """The evaluator's batching is not allowed to change what any single leaf computes
    (BATCH_NOTES §3's ~1e-7 is the contract; on this box it is exact)."""
    enc = get_encoder(encoder)
    net = build_net(enc)
    caps = enc.caps.as_dict() if getattr(enc, "is_set", False) else None
    worker = LeafEncoder(encoder, caps=caps)
    encoded = [worker.encode_leaf(g) for g in sample_games(4)]
    encoded = [e for e in encoded if e is not None]
    is_set = bool(getattr(enc, "is_set", False))

    batched_p, batched_v = forward_leaves(net, [e[1] for e in encoded],
                                          [e[2] for e in encoded], is_set=is_set,
                                          device="cpu")
    for i, e in enumerate(encoded):
        one_p, one_v = forward_leaves(net, [e[1]], [e[2]], is_set=is_set, device="cpu")
        assert np.allclose(batched_p[i], one_p[0], atol=1e-6)
        assert batched_v[i] == pytest.approx(one_v[0], abs=1e-6)


# ══════════════════════════════════════════════════════════════ 4. the lockstep decider

def _uniform_players(n: int) -> list:
    return [MCTSPlayer(policy=UniformPolicy(), config=MCTSConfig(num_simulations=8),
                       outcome=MLBOutcome(starting_lives=4, horizon_antes=4),
                       rng=np.random.default_rng(100 + i), reuse=True, leaf_batch=1,
                       no_action={"type": "advance"}, name=f"p{i}")
            for i in range(n)]


def test_lockstep_decider_is_byte_identical_to_the_serial_tournament():
    """The batching is exact: K trees in lockstep == K trees driven one at a time."""
    from parallel.lockstep import LockstepDecider
    from tournament.parallel import LocalDriver, ParallelTournament
    from tournament.runner import Tournament

    n = 5
    serial = Tournament(seed=SEED, n_agents=n, players=_uniform_players(n),
                        life_rule="paired", max_ante=4)
    r1 = serial.run()

    driver = LocalDriver(_uniform_players(n), decide_many=LockstepDecider())
    lock = ParallelTournament(seed=SEED, n_agents=n, players=[None] * n,
                              life_rule="paired", max_ante=4, driver=driver)
    r2 = lock.run()

    assert r1.steps_total == r2.steps_total
    assert r1.final_lives == r2.final_lives
    assert r1.forced_progress == r2.forced_progress
    assert ([g.state_signature() for g in serial._last_games]
            == [g.state_signature() for g in driver.games])
    for m1, m2 in zip(r1.ante_matrices, r2.ante_matrices):
        assert np.array_equal(m1.scores, m2.scores, equal_nan=True)
        assert m1.losers == m2.losers


def test_lockstep_decider_records_and_constrains_like_act_key():
    """`record_hook` and `legal_filter` are what make a decision a TRAINING decision;
    `BatchedMCTSPlayerGroup` has neither, which is why this decider exists."""
    from parallel.lockstep import LockstepDecider
    from train.population import SkipCap

    seen: list = []
    player = _uniform_players(1)[0]
    player.record_hook = seen.append
    player.legal_filter = SkipCap(0)
    game = _game()
    while game.state is not State.BLIND_SELECT:
        game.step(game.legal_actions()[0])

    action = LockstepDecider()([(0, game, player)])[0]
    assert action["type"] != "skip_blind"           # the cap removed it from the support
    assert seen and seen[-1].chosen[0] != "skip_blind"
    assert all(k[0] != "skip_blind" for k in seen[-1].visits)


# ══════════════════════════════════════════════════════════════ 5. partition / ids

def _members(n=8, m_current=4, n_anchor=2):
    out = []
    for i in range(n):
        anchor = i >= n - n_anchor
        out.append(PopulationMember(
            idx=i, name=f"a{i}", is_current=i < m_current,
            sims=0 if anchor else (40 if i < m_current else 20), seed=i,
            checkpoint=(None if i < m_current or anchor else f"ck{(i % 2)}.pt"),
            kind="anchor" if anchor else "net",
            spec=(("hand", "greedy"),) if anchor else None))
    return out


def test_partition_agents_covers_every_agent_exactly_once_and_is_deterministic():
    members = _members()
    for w in (1, 2, 3, 4, 8):
        buckets = partition_agents(members, w)
        assert len(buckets) == w
        flat = sorted(i for b in buckets for i in b)
        assert flat == [m.idx for m in members]
        assert buckets == partition_agents(members, w)


def test_partition_agents_spreads_the_expensive_seats():
    """An ante is a barrier: a worker holding only free scripted anchors idles while
    another grinds four 40-sim searches."""
    buckets = partition_agents(_members(), 4)
    by_idx = {m.idx: m for m in _members()}
    loads = [sum(by_idx[i].sims for i in b) for b in buckets]
    assert max(loads) - min(loads) <= 40


def test_assign_policy_ids_gives_the_live_net_zero_and_one_id_per_checkpoint():
    by_agent, by_id = assign_policy_ids(_members())
    assert all(by_agent[i] == LIVE_POLICY_ID for i in range(4))
    assert set(by_id) == {1, 2}
    assert sorted(by_id.values()) == ["ck0.pt", "ck1.pt"]
    assert 6 not in by_agent and 7 not in by_agent          # anchors have no net


# ══════════════════════════════════════════════════════════════ 6. the pool, end to end

def test_one_worker_and_two_workers_play_the_same_tournament(tmp_path):
    """The determinism contract: the result must not depend on the worker count."""
    a = _play_once(tmp_path / "w1", workers=1)
    b = _play_once(tmp_path / "w2", workers=2)
    assert a["matrices"] == b["matrices"]
    assert a["lives"] == b["lives"]
    assert a["z"] == b["z"]
    assert a["n_samples"] == b["n_samples"]


def _play_once(tmp_path, workers: int) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    trainer = make_trainer(tmp_path, workers=workers, n_agents=4, m_current=2,
                           anchor_frac=0.25)
    try:
        samples, _metrics = trainer._play_tournament(None, None, None)
        return {
            "matrices": [(m.ante, np.nan_to_num(m.scores, nan=-1.0).tolist(),
                          tuple(m.losers))
                         for r in trainer.last_results for m in r.ante_matrices],
            "lives": [r.final_lives for r in trainer.last_results],
            "z": [float(s.z) for s in samples],
            "n_samples": len(samples),
        }
    finally:
        trainer.close()


def test_a_generation_runs_and_reports_the_evaluator(tmp_path):
    trainer = make_trainer(tmp_path, workers=2)
    try:
        m = trainer.run_generation()
        extra = trainer.extra_metrics
        assert m.episodes == trainer.mlb.n_agents
        assert m.n_samples > 0
        assert extra["workers"] == 2 and extra["workers_live"] == 2
        assert extra["eval_leaves"] > 0
        assert extra["crashed_agents"] == [] and extra["dead_workers"] == []
        assert trainer.generation == 1
    finally:
        trainer.close()


def test_local_mode_runs_without_an_evaluator(tmp_path):
    """The control arm: every worker holds its own net.  Same result shape, no shared
    evaluator at all."""
    trainer = make_trainer(tmp_path, workers=2, device="local")
    try:
        m = trainer.run_generation()
        assert trainer.pool.evaluator is None
        assert m.n_samples > 0
        assert (tmp_path / "_worker_weights.pt").is_file()
    finally:
        trainer.close()


# ══════════════════════════════════════════════════════════════ 7. weights broadcast

def test_sync_weights_is_what_the_workers_actually_see(tmp_path):
    """The evaluator holds a MIRROR of the trainer's net, so "the workers play the net the
    last training step produced" is an explicit call, not an aliasing accident."""
    trainer = make_trainer(tmp_path, workers=1)
    try:
        pool = trainer.ensure_pool()
        trainer._register_models({})
        mirror = pool.evaluator.models[LIVE_POLICY_ID]
        assert mirror is not trainer.net                      # a copy, deliberately
        with torch.no_grad():
            for p in trainer.net.parameters():
                p.add_(1.0)
        before = [p.clone() for p in mirror.parameters()]
        pool.evaluator.sync_weights(trainer.net, LIVE_POLICY_ID)
        after = list(mirror.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after))
        for got, want in zip(after, trainer.net.parameters()):
            assert torch.equal(got.cpu(), want.cpu())
    finally:
        trainer.close()


# ══════════════════════════════════════════════════════════════ 8. crash handling

def test_a_dead_worker_costs_its_agents_not_the_run(tmp_path):
    """A worker killed mid-generation: its agents are marked crashed, the tournament is
    built from whoever is left, and the generation still produces samples."""
    from tournament.parallel import ParallelTournament
    from parallel.pool import MPDriver
    from parallel.protocol import OP_GENERATION

    trainer = make_trainer(tmp_path, workers=2, n_agents=4, m_current=2, anchor_frac=0.0)
    try:
        pool = trainer.ensure_pool()
        members = trainer._build_members()
        policy_ids, checkpoints = assign_policy_ids(members)
        trainer._register_models(checkpoints)
        buckets = partition_agents(members, pool.n)
        owners = {idx: w for w, idxs in enumerate(buckets) for idx in idxs}
        trainer._send_generation(pool, buckets, {m.idx: m for m in members},
                                 policy_ids, checkpoints, None)

        driver = MPDriver(pool, owners, trainer.mlb.n_agents,
                          {"life_rule": "paired", "max_ante": 3, "traj": None})
        tour = ParallelTournament(seed=SEED, n_agents=trainer.mlb.n_agents,
                                  players=[None] * trainer.mlb.n_agents,
                                  life_rule="paired", max_ante=3, driver=driver)
        pool.crash_worker(1)
        result = tour.run()

        assert 1 in pool.dead
        assert set(result.crashed) == set(buckets[1])
        assert len(pool.live) == 1
        # The matrix was still built, from the survivors and only from them.  (A survivor
        # may still DIE at a later ante, which is an ordinary game over, so only the first
        # Nemesis is asserted to have them present.)
        assert result.ante_matrices
        first = result.ante_matrices[0]
        assert any(not np.isnan(first.scores[i]) for i in buckets[0])
        assert all(np.isnan(first.scores[i]) for i in buckets[1])
    finally:
        trainer.close()


def test_respawn_brings_a_dead_worker_back_for_the_next_generation(tmp_path):
    trainer = make_trainer(tmp_path, workers=2)
    try:
        pool = trainer.ensure_pool()
        pool.crash_worker(0)
        # give the pool a call to notice
        pool.broadcast("stats")
        assert 0 in pool.dead
        assert pool.respawn_dead() == [0]
        assert pool.live == {0, 1} and not pool.dead
        assert set(pool.broadcast("stats")) == {0, 1}
    finally:
        trainer.close()


# ══════════════════════════════════════════════════════════════ 9. PAUSE and checkpoints

def test_pause_between_tournaments_drains_and_checkpoints(tmp_path):
    """The PAUSE contract, unchanged from the serial trainer: one tournament is the atomic
    unit, then the generation trains on what it collected and the checkpoint is loadable
    by the SINGLE-process trainer."""
    trainer = make_trainer(tmp_path, workers=1, seeds_per_generation=3)
    try:
        calls = {"n": 0}

        def stop_check():
            calls["n"] += 1
            return calls["n"] > 1                    # stop after the first tournament

        m = trainer.run_generation(stop_check=stop_check)
        assert m.episodes == trainer.mlb.n_agents    # exactly ONE tournament was played
        path = save_checkpoint(tmp_path / "paused.pt", trainer.state_dict())
    finally:
        trainer.close()

    ckpt = load_checkpoint(path)
    single = MLBTrainer.from_checkpoint(ckpt)
    assert type(single) is MLBTrainer
    assert single.generation == trainer.generation
    assert len(single.buffer) == len(trainer.buffer)
    assert single.mlb.__dict__ == trainer.mlb.__dict__


def test_a_single_process_checkpoint_resumes_into_the_parallel_trainer(tmp_path):
    """The swap the live run needs: PAUSE the single-process run, resume it on N cores."""
    serial = MLBTrainer(tiny_cfg(), tiny_mlb())
    serial.run_generation()
    path = save_checkpoint(tmp_path / "serial.pt", serial.state_dict())

    ckpt = load_checkpoint(path)
    par = ParallelMLBTrainer.from_checkpoint(ckpt)
    par.pool_cfg = PoolConfig(n_workers=1)
    par.work_dir = tmp_path
    try:
        assert par.generation == serial.generation
        assert len(par.buffer) == len(serial.buffer)
        assert par.mlb.__dict__ == serial.mlb.__dict__
        for got, want in zip(par.net.parameters(), serial.net.parameters()):
            assert torch.equal(got, want)
        m = par.run_generation()
        assert m.generation == serial.generation
        assert par.generation == serial.generation + 1
    finally:
        par.close()


def test_the_checkpoint_format_is_the_inherited_one(tmp_path):
    """No `--workers` anywhere in the payload: the worker count is a property of the box,
    not of the experiment."""
    par = ParallelMLBTrainer(tiny_cfg(), tiny_mlb(), pool_cfg=PoolConfig(n_workers=4))
    serial = MLBTrainer(tiny_cfg(), tiny_mlb())
    a, b = par.state_dict(include_buffer=False), serial.state_dict(include_buffer=False)
    assert sorted(a) == sorted(b)
    assert a["mlb"]["config"] == b["mlb"]["config"]
    assert a["config"] == b["config"]
    assert a["net_desc"] == b["net_desc"]
    assert "workers" not in str(a["mlb"]["config"])


# ══════════════════════════════════════════════════════════════ 10. trajectory merge

def test_merge_trajectory_parts_concatenates_and_removes_the_parts(tmp_path):
    parts = []
    for w in range(3):
        p = tmp_path / f"trajectories.w{w}.jsonl"
        p.write_text(f'{{"w": {w}}}\n{{"w": {w}}}\n', encoding="utf-8")
        parts.append(str(p))
    target = tmp_path / "trajectories.jsonl"
    assert merge_trajectory_parts(target, parts) == 6
    assert target.read_text(encoding="utf-8").count("\n") == 6
    assert not any(Path(p).exists() for p in parts)


def test_merge_trajectory_parts_tolerates_a_worker_that_wrote_nothing(tmp_path):
    target = tmp_path / "trajectories.jsonl"
    assert merge_trajectory_parts(target, [str(tmp_path / "missing.jsonl")]) == 0


def test_worker_trajectory_paths_agree_with_the_merge():
    """The worker names its own part file; the trainer builds the same names to merge
    them.  Two places, one convention — pin it."""
    from parallel.worker import _worker_traj_path

    got = _worker_traj_path("/runs/real1/trajectories.jsonl", 3)
    assert Path(got).name == "trajectories.w3.jsonl"
    assert Path(got).parent == Path("/runs/real1")
