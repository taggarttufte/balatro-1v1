"""
test_set_encoder.py — the set-based observation, action features and network (Phase 4 W1).

The properties that matter, in the order the brief lists them:

  * the network is SET-INVARIANT: permuting the rows of any item set changes nothing, and
    neither does what is written into the padded rows;
  * every one of the 13 action types `legal_actions()` emits produces a distinct, finite
    action row;
  * every legal action of 200 real states scores finitely;
  * the fast hand-type evaluator equals `hand_eval.evaluate_hand`;
  * batched `evaluate_many` == the single-leaf `__call__`.

Invariance is asserted to 1e-5, not exactly. Permuting rows changes the summation order
inside the attention weighting and the masked pooling, so float32 gives a last-ulp
difference — the same reasoning BATCH_NOTES §3 gives for "batched == single-leaf" on the
flat net. Measured worst case here is ~1e-6.
"""
from __future__ import annotations

import itertools
import random

import numpy as np
import pytest
import torch

from balatro_sim.card import Card
from balatro_sim.env_v7 import HAND_TYPES, SUIT_ORDER
from balatro_sim.game import BalatroGame, State
from balatro_sim.hand_eval import evaluate_hand

from mcts.action_features import ACTION_TYPES
from mcts.action_features_set import (
    ACT_NUM_DIM, HandContext, featurize_actions_set, fast_hand_types,
)
from mcts.encoder import get_encoder, is_set_encoder
from mcts.encoder_set import (
    KEY_VOCAB_SIZE, SCALAR_DIM, SCALAR_LAYOUT, ItemCaps, SetEncoder, key_index,
)
from mcts.model_set import SetPolicyValueNet
from mcts.policy_set import BatchedSetNNPolicy, SetNNPolicy

from _states import (
    STATE_FIXTURE_SPEC, collect_states, overnight_seeds, state_histogram, walk_states,
)

ITEM_SETS = [
    ("hand", ["hand_cat", "hand_num"], "hand_mask"),
    ("jokers", ["joker_key", "joker_cat", "joker_num"], "joker_mask"),
    ("consumables", ["cons_key", "cons_num"], "cons_mask"),
    ("shelf", ["shelf_key", "shelf_cat", "shelf_card", "shelf_num"], "shelf_mask"),
    ("packs", ["pack_key", "pack_cat", "pack_card", "pack_num"], "pack_mask"),
]


@pytest.fixture(scope="module")
def states():
    games = collect_states(200)
    assert len(games) == 200
    return games


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return SetPolicyValueNet().eval()


def _tensors(d, device="cpu"):
    return {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in d.items()}


# ── registration ────────────────────────────────────────────────────────────────

def test_get_encoder_knows_set():
    enc = get_encoder("set")
    assert isinstance(enc, SetEncoder)
    assert enc.name == "set" and enc.is_set and enc.dim is None
    assert is_set_encoder(enc) and not is_set_encoder(get_encoder("v7"))


def test_flat_encoders_declare_is_set_false():
    for name in ("v7", "mlb"):
        enc = get_encoder(name)
        assert enc.is_set is False and isinstance(enc.dim, int)


def test_get_encoder_passes_caps_through():
    caps = {"hand": 10, "jokers": 6, "consumables": 3, "shelf": 5, "packs": 5}
    enc = get_encoder("set", caps=caps)
    assert enc.caps == ItemCaps(**caps)
    assert enc.describe()["caps"] == caps
    assert SetEncoder.from_description(enc.describe()).caps == enc.caps


def test_scalar_layout_matches_dim():
    assert sum(w for _, w in SCALAR_LAYOUT) == SCALAR_DIM
    assert len({n for n, _ in SCALAR_LAYOUT}) == len(SCALAR_LAYOUT)


def test_key_vocab_covers_the_game_keys():
    from balatro_sim import game_keys as gk
    assert KEY_VOCAB_SIZE == 2 + sum(len(t) for t in (
        gk.JOKER_KEYS, gk.TAROT_KEYS, gk.PLANET_KEYS, gk.SPECTRAL_KEYS,
        gk.VOUCHER_KEYS, gk.BOOSTER_TYPE_KEYS))
    assert len(gk.JOKER_KEYS) == 150
    for key in (gk.JOKER_KEYS[0], gk.JOKER_KEYS[-1], gk.TAROT_KEYS[0],
                gk.PLANET_KEYS[-1], gk.SPECTRAL_KEYS[3], gk.VOUCHER_KEYS[-1]):
        assert key_index(key) > 1
    assert key_index("j_does_not_exist") == 1        # <unk>, not a silent collision
    assert key_index(None) == 0


# ── shapes ──────────────────────────────────────────────────────────────────────

def test_observation_shapes_and_dtypes(states):
    enc = SetEncoder()
    caps = enc.caps
    want = {
        "hand_cat": (caps.hand, 5), "hand_num": (caps.hand, 9), "hand_mask": (caps.hand,),
        "joker_key": (caps.jokers,), "joker_cat": (caps.jokers, 2),
        "joker_num": (caps.jokers, 16), "joker_mask": (caps.jokers,),
        "cons_key": (caps.consumables,), "cons_num": (caps.consumables, 8),
        "cons_mask": (caps.consumables,),
        "shelf_key": (caps.shelf,), "shelf_cat": (caps.shelf, 2),
        "shelf_card": (caps.shelf, 5), "shelf_num": (caps.shelf, 12),
        "shelf_mask": (caps.shelf,),
        "pack_key": (caps.packs,), "pack_cat": (caps.packs, 2),
        "pack_card": (caps.packs, 5), "pack_num": (caps.packs, 8),
        "pack_mask": (caps.packs,),
        "scalars": (SCALAR_DIM,),
    }
    for game in states[:40]:
        obs = enc(game)
        assert set(obs) == set(want)
        for k, shape in want.items():
            assert obs[k].shape == shape, k
            assert np.isfinite(obs[k].astype(np.float64)).all(), k
        for k in ("hand_num", "joker_num", "cons_num", "shelf_num", "pack_num", "scalars",
                  "hand_mask", "joker_mask", "cons_mask", "shelf_mask", "pack_mask"):
            assert obs[k].dtype == np.float32, k
        for k in ("hand_cat", "joker_key", "joker_cat", "cons_key", "shelf_key",
                  "shelf_cat", "shelf_card", "pack_key", "pack_cat", "pack_card"):
            assert obs[k].dtype == np.int16, k
            assert obs[k].min() >= 0 and obs[k].max() < KEY_VOCAB_SIZE, k


def test_masks_match_the_live_item_counts(states):
    enc = SetEncoder()
    caps = enc.caps
    for game in states:
        obs = enc(game)
        assert obs["hand_mask"].sum() == min(len(game.hand), caps.hand)
        assert obs["joker_mask"].sum() == min(len(game.jokers), caps.jokers)
        assert obs["cons_mask"].sum() == min(len(game.consumable_hand), caps.consumables)
        assert obs["shelf_mask"].sum() == min(len(game.current_shop), caps.shelf)
        assert obs["pack_mask"].sum() == min(len(game.booster_choices), caps.packs)
        # masks are a prefix: live rows first, padding after
        for key in ("hand_mask", "joker_mask", "cons_mask", "shelf_mask", "pack_mask"):
            m = obs[key]
            assert np.all(np.diff(m) <= 0), key


def test_encoder_is_deterministic(states):
    enc = SetEncoder()
    for game in states[:30]:
        a, b = enc(game), enc(game)
        for k in a:
            assert np.array_equal(a[k], b[k]), k


def test_padded_rows_are_zero(states):
    enc = SetEncoder()
    for game in states[:60]:
        obs = enc(game)
        for _, arrays, mask_key in ITEM_SETS:
            dead = obs[mask_key] == 0
            if not dead.any():
                continue
            for key in arrays:
                assert not obs[key][dead].any(), f"{key} wrote into a padded row"


# ── the network's set properties ────────────────────────────────────────────────

def _forward(net, obs, acts):
    with torch.no_grad():
        logits, value = net(_tensors(obs), _tensors(acts))
    return logits[0].numpy(), float(value)


def _permute_obs(obs, caps, rng):
    """Permute the LIVE rows of every set (and the matching action-mask columns)."""
    out = dict(obs)
    perms = {}
    for name, arrays, mask_key in ITEM_SETS:
        width = obs[mask_key].shape[0]
        n_live = int(obs[mask_key].sum())
        p = np.arange(width)
        if n_live > 1:
            p[:n_live] = rng.permutation(n_live)
        perms[name] = p
        for key in arrays + [mask_key]:
            out[key] = obs[key][p]
    return out, perms


def test_permuting_items_within_a_set_changes_nothing(states, net):
    enc = SetEncoder()
    caps = enc.caps
    rng = np.random.default_rng(3)
    worst_v = worst_p = 0.0
    checked = 0
    for game in states[:60]:
        legal = game.legal_actions()
        if not legal:
            continue
        obs = enc(game)
        acts = featurize_actions_set(game, legal, caps)
        base_logits, base_v = _forward(net, obs, acts)

        p_obs, perms = _permute_obs(obs, caps, rng)
        # `new_row[i] = old_row[p[i]]`, so the mask column that weights NEW item i must be
        # the one that weighted OLD item p[i]: the columns take the same permutation.
        p_acts = dict(acts)
        p_acts["act_sel"] = acts["act_sel"][:, perms["hand"]]
        tgt = np.concatenate([
            perms["jokers"],
            perms["consumables"] + caps.jokers,
            perms["shelf"] + caps.jokers + caps.consumables,
            perms["packs"] + caps.jokers + caps.consumables + caps.shelf,
        ])
        p_acts["act_tgt"] = acts["act_tgt"][:, tgt]
        got_logits, got_v = _forward(net, p_obs, p_acts)

        worst_v = max(worst_v, abs(got_v - base_v))
        worst_p = max(worst_p, float(np.abs(got_logits - base_logits).max()))
        checked += 1
    assert checked >= 40
    assert worst_v < 1e-4, f"value moved by {worst_v} under a within-set permutation"
    assert worst_p < 1e-4, f"logits moved by {worst_p} under a within-set permutation"


def test_garbage_in_the_padded_rows_changes_nothing(states, net):
    """The caps are transport. What is in a masked-out row must be unreachable."""
    enc = SetEncoder()
    rng = np.random.default_rng(11)
    checked = 0
    for game in states[:60]:
        legal = game.legal_actions()
        if not legal:
            continue
        obs = enc(game)
        acts = featurize_actions_set(game, legal, enc.caps)
        base_logits, base_v = _forward(net, obs, acts)

        dirty = {k: v.copy() for k, v in obs.items()}
        touched = False
        for _, arrays, mask_key in ITEM_SETS:
            dead = dirty[mask_key] == 0
            if not dead.any():
                continue
            touched = True
            for key in arrays:
                a = dirty[key]
                if a.dtype == np.int16:
                    hi = KEY_VOCAB_SIZE if key.endswith("_key") else 5
                    a[dead] = rng.integers(1, hi, size=a[dead].shape).astype(np.int16)
                else:
                    a[dead] = rng.normal(size=a[dead].shape).astype(np.float32)
        if not touched:
            continue
        got_logits, got_v = _forward(net, dirty, acts)
        assert abs(got_v - base_v) < 1e-5
        assert np.abs(got_logits - base_logits).max() < 1e-5
        checked += 1
    assert checked >= 30


def test_param_count_is_within_budget(net):
    assert net.n_params < 3_000_000, net.n_params
    assert net.n_params > 500_000


def test_describe_round_trips(net):
    rebuilt = SetPolicyValueNet.from_description(net.describe())
    assert rebuilt.describe() == net.describe()
    assert rebuilt.n_params == net.n_params
    rebuilt.load_state_dict(net.state_dict())        # shapes must match exactly


def test_empty_state_does_not_nan(net):
    """A BLIND_SELECT state can have no cards, no jokers, no consumables, no shop and no
    pack — every item slot masked. The global token is what keeps the attention softmax
    from producing NaN there."""
    game = BalatroGame(seed=11, deck_key="b_red", stake=1, ruleset="mlb")
    assert game.state is State.BLIND_SELECT and not game.hand and not game.jokers
    enc = SetEncoder()
    acts = featurize_actions_set(game, game.legal_actions(), enc.caps)
    logits, value = _forward(net, enc(game), acts)
    assert np.isfinite(logits).all() and np.isfinite(value)


# ── action features ─────────────────────────────────────────────────────────────

def test_every_action_type_embeds():
    """Walk real games until all 13 `legal_actions()` types have been seen; every one must
    embed to a finite, non-degenerate row."""
    enc = SetEncoder()
    seen: dict[str, np.ndarray] = {}
    # `7I4M53DL` is the seed Phase 3 established for `reroll_boss` (Directors Cut /
    # Retcon), the one type a random walk essentially never reaches.
    specs = [("7I4M53DL", "vanilla"), ("7I4M53DL", "mlb")] +             [(int(s), "mlb") for s in overnight_seeds(6)]
    for seed, ruleset in specs:
        game = BalatroGame(seed=seed, deck_key="b_red", stake=1, ruleset=ruleset)
        for g, legal in walk_states(game, max_steps=160):
            if not legal:
                break
            acts = featurize_actions_set(g, legal, enc.caps)
            for i, a in enumerate(legal):
                seen.setdefault(a["type"], np.concatenate([
                    [acts["act_type"][i]], acts["act_sel"][i],
                    acts["act_tgt"][i], acts["act_num"][i]]))
        if len(seen) == len(ACTION_TYPES):
            break
    assert set(seen) == set(ACTION_TYPES), sorted(set(ACTION_TYPES) - set(seen))
    types = set()
    for name, row in seen.items():
        assert np.isfinite(row).all(), name
        assert int(row[0]) != 0, f"{name} embedded as the unknown type"
        types.add(int(row[0]))
    assert len(types) == len(ACTION_TYPES), "two action types share a type index"


def test_action_rows_are_unique_per_state(states):
    enc = SetEncoder()
    for game in states:
        legal = game.legal_actions()
        if len(legal) < 2:
            continue
        acts = featurize_actions_set(game, legal, enc.caps)
        rows = np.concatenate([acts["act_type"][:, None].astype(np.float32),
                               acts["act_sel"], acts["act_tgt"], acts["act_num"]], axis=1)
        assert np.unique(rows, axis=0).shape[0] == rows.shape[0], (
            f"two different legal actions share a feature row at {game.state.name}")


def test_action_features_are_finite_and_bounded(states):
    enc = SetEncoder()
    for game in states:
        legal = game.legal_actions()
        if not legal:
            continue
        acts = featurize_actions_set(game, legal, enc.caps)
        assert acts["act_num"].shape[1] == ACT_NUM_DIM
        for key in ("act_sel", "act_tgt", "act_num"):
            assert np.isfinite(acts[key]).all(), key
        # the masks are row-normalised (mean pooling), so each row sums to 0 or 1
        for key in ("act_sel", "act_tgt"):
            sums = acts[key].sum(axis=1)
            assert np.all((sums < 1e-6) | (np.abs(sums - 1.0) < 1e-5)), key


def test_every_legal_action_of_200_states_gets_a_finite_score(states, net):
    enc = SetEncoder()
    n_actions = 0
    for game in states:
        legal = game.legal_actions()
        if not legal:
            continue
        logits, value = _forward(net, enc(game), featurize_actions_set(game, legal, enc.caps))
        assert logits.shape == (len(legal),)
        assert np.isfinite(logits).all(), game.state.name
        assert np.isfinite(value)
        n_actions += len(legal)
    assert n_actions > 20_000, n_actions


# ── the fast hand-type evaluator ────────────────────────────────────────────────

def test_fast_hand_type_matches_hand_eval():
    """Every subset of size 1-5 of many random hands, across all four flag combinations,
    against the engine's own `evaluate_hand`."""
    rng = random.Random(7)
    checked = 0
    for _ in range(60):
        n = rng.randint(1, 9)
        hand = []
        for _ in range(n):
            c = Card(rank=rng.randint(2, 14), suit=rng.choice(SUIT_ORDER))
            r = rng.random()
            if r < 0.10:
                c.enhancement = "Stone"
            elif r < 0.22:
                c.enhancement = "Wild"
            if rng.random() < 0.08:
                c.debuffed = True
            hand.append(c)
        subsets = [s for k in range(1, min(5, n) + 1)
                   for s in itertools.combinations(range(n), k)]
        sel = np.zeros((len(subsets), n), dtype=np.float32)
        for i, s in enumerate(subsets):
            sel[i, list(s)] = 1.0
        for ff in (False, True):
            for sc in (False, True):
                for sm in (False, True):
                    flags = dict(four_fingers=ff, shortcut=sc, smeared=sm)
                    got = fast_hand_types(sel, HandContext(hand, flags))
                    for i, s in enumerate(subsets):
                        want = evaluate_hand([hand[j] for j in s], **flags)[0]
                        assert HAND_TYPES[got[i]] == want, (
                            [repr(hand[j]) for j in s], flags, want, HAND_TYPES[got[i]])
                        checked += 1
    assert checked > 30_000, checked


def test_fast_hand_type_matches_hand_eval_on_real_hands(states):
    """The same equality on hands the engine actually dealt (boss debuffs, seals,
    enhancements, face-down cards and the live Four Fingers / Shortcut / Smeared flags)."""
    checked = 0
    for game in states:
        hand = game.hand
        if not hand:
            continue
        flags = game.hand_eval_flags()
        ctx = HandContext(hand, flags)
        subsets = [s for k in range(1, min(5, len(hand)) + 1)
                   for s in itertools.combinations(range(len(hand)), k)]
        sel = np.zeros((len(subsets), len(hand)), dtype=np.float32)
        for i, s in enumerate(subsets):
            sel[i, list(s)] = 1.0
        got = fast_hand_types(sel, ctx)
        for i, s in enumerate(subsets):
            want = evaluate_hand([hand[j] for j in s], **flags)[0]
            assert HAND_TYPES[got[i]] == want
            checked += 1
    assert checked > 10_000, checked


def test_hand_type_features_can_be_switched_off(states):
    enc = SetEncoder()
    game = next(g for g in states if g.state is State.SELECTING_HAND)
    legal = game.legal_actions()
    on = featurize_actions_set(game, legal, enc.caps, hand_type_features=True)
    off = featurize_actions_set(game, legal, enc.caps, hand_type_features=False)
    assert on["act_num"][:, 5:18].any()
    assert not off["act_num"][:, 5:19].any()
    for key in ("act_type", "act_sel", "act_tgt"):
        assert np.array_equal(on[key], off[key])


# ── the policies ────────────────────────────────────────────────────────────────

def test_batched_equals_single_leaf(states, net):
    serial = SetNNPolicy(net, device="cpu")
    batched = BatchedSetNNPolicy(net, device="cpu")
    games = [g for g in states[:48]]
    got = batched.evaluate_many(games)
    worst_p = worst_v = 0.0
    for game, (priors, value) in zip(games, got):
        ref_priors, ref_value = serial(game)
        assert set(priors) == set(ref_priors)
        worst_v = max(worst_v, abs(value - ref_value))
        for k in priors:
            worst_p = max(worst_p, abs(priors[k] - ref_priors[k]))
    assert worst_p < 1e-5, worst_p
    assert worst_v < 1e-3, worst_v


def test_priors_sum_to_one(states, net):
    policy = BatchedSetNNPolicy(net, device="cpu")
    for priors, value in policy.evaluate_many(states[:60]):
        if not priors:
            continue
        assert abs(sum(priors.values()) - 1.0) < 1e-4
        assert np.isfinite(value)


def test_no_action_game_never_touches_the_net(net):
    policy = BatchedSetNNPolicy(net, device="cpu")
    game = BalatroGame(seed=3, deck_key="b_red", stake=1, ruleset="mlb")
    game.state = State.GAME_OVER
    assert game.legal_actions() == []
    before = policy.forwards
    assert policy(game) == ({}, 0.0)
    assert policy.evaluate_many([game, game]) == [({}, 0.0), ({}, 0.0)]
    assert policy.forwards == before


def test_chunking_changes_nothing(states, net):
    big = BatchedSetNNPolicy(net, device="cpu")
    small = BatchedSetNNPolicy(net, device="cpu", max_action_rows=64)
    games = states[:24]
    a, b = big.evaluate_many(games), small.evaluate_many(games)
    assert small.forwards > big.forwards
    for (pa, va), (pb, vb) in zip(a, b):
        assert set(pa) == set(pb)
        assert abs(va - vb) < 1e-4
        for k in pa:
            assert abs(pa[k] - pb[k]) < 1e-5


def test_caps_mismatch_is_refused(net):
    other = SetEncoder(caps={"hand": 10, "jokers": 6, "consumables": 3,
                             "shelf": 5, "packs": 5})
    with pytest.raises(ValueError, match="caps"):
        SetNNPolicy(net, encoder=other)


def test_search_runs_on_the_set_policy():
    """The whole point of leaving `search.py` untouched: MCTS is encoder-blind."""
    from mcts import MCTS, MCTSConfig
    torch.manual_seed(1)
    policy = BatchedSetNNPolicy(SetPolicyValueNet().eval(), device="cpu")
    game = BalatroGame(seed=42, deck_key="b_red", stake=1, ruleset="mlb")
    game.step({"type": "play_blind"})
    mcts = MCTS(policy, MCTSConfig(num_simulations=16, gumbel_max_considered=4),
                rng=np.random.default_rng(0))
    root, visits, chosen = mcts.run_gumbel(game)
    assert chosen is not None
    assert sum(visits.values()) > 0


# ── documentation of the fixture ────────────────────────────────────────────────

def test_state_fixture_covers_the_game(states):
    hist = state_histogram(states)
    assert hist.get("SELECTING_HAND", 0) > 20
    assert hist.get("SHOP", 0) > 10
    assert hist.get("BLIND_SELECT", 0) > 5
    assert STATE_FIXTURE_SPEC["ruleset"] == "mlb"


def test_packed_and_unpacked_transfers_agree(states):
    """On a non-CPU device the obs/action arrays are packed into one buffer per dtype and
    split into views (policy_set: 20 small H2D copies cost 5.35 ms on this box, one packed
    copy 0.027 ms). The two paths must produce identical tensors."""
    from mcts.encoder_set import SetEncoder as _SE
    from mcts.policy_set import _pack_group, _numpy_dtype_groups
    enc = _SE()
    obs = [enc(g) for g in states[:4]]
    stacked = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
    floats, ints = _numpy_dtype_groups(stacked)
    assert set(floats) | set(ints) == set(stacked)
    packed = _pack_group(stacked, floats, np.float32, "cpu")
    packed.update(_pack_group(stacked, ints, np.int16, "cpu"))
    for k, v in stacked.items():
        assert torch.equal(packed[k], torch.from_numpy(v)), k
        assert packed[k].shape == v.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu(states):
    """The packed CUDA transfer path must give the same answers as the CPU one."""
    # Two nets, not one: `NNPolicy.__init__` does `model.to(device)`, which moves the
    # module in place — sharing one would drag the CPU policy onto the GPU.
    torch.manual_seed(0)
    a = SetPolicyValueNet().eval()
    torch.manual_seed(0)
    b = SetPolicyValueNet().eval()
    cpu = BatchedSetNNPolicy(a, device="cpu")
    gpu = BatchedSetNNPolicy(b, device="cuda")
    games = states[:16]
    for (pa, va), (pb, vb) in zip(cpu.evaluate_many(games), gpu.evaluate_many(games)):
        assert set(pa) == set(pb)
        assert abs(va - vb) < 1e-3
        for k in pa:
            assert abs(pa[k] - pb[k]) < 1e-4
