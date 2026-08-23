"""mcts — AlphaZero-style tree search for Balatro (mp/agent fork of balatro-mcts)."""
from .action import action_key, action_from_key, ActionKey
from .node import Node
from .outcome import (
    OutcomeFn, VanillaOutcome, MLBOutcome, ExternalOutcome,
    default_outcome_for, margin_to_value, is_stuck_state,
)
from .encoder import (
    OBS_DIM, MLB_OBS_DIM, encode_obs, encode_obs_mlb,
    ObsEncoder, V7Encoder, MLBEncoder, get_encoder, is_set_encoder,
)
# Phase 4 W1 — the set-based encoder / features / net. Imported here so `from mcts import
# SetEncoder` works; nothing in the flat path depends on them.
from .encoder_set import SetEncoder, ItemCaps, SCALAR_DIM, KEY_VOCAB_SIZE
from .action_features_set import featurize_actions_set, ACT_NUM_DIM
from .model_set import SetPolicyValueNet, StateEmbedding
from .action_features import ACTION_FEATURE_DIM, featurize_action, featurize_actions
from .policy import PolicyValueFn, PolicyValueBase, UniformPolicy, NNPolicy, make_policy
from .policy_set import SetNNPolicy, BatchedSetNNPolicy
from .model import PolicyValueNet
from .search import MCTS, MCTSConfig
# W0 (2026-08-22) — the heuristic hand prior / candidate mask. Imported here so
# `from mcts import shape_priors` works; nothing in the search depends on it unless
# `MCTSConfig.heuristic_prior_weight` or `.max_hand_candidates` is set.
from .heuristic import (
    HeuristicConfig, HandHeuristic, hand_action_scores, heuristic_distribution,
    shape_priors,
)
from .batched import (
    BatchedNNPolicy, BatchedSearch, SearchRequest, SearchResult, BatchStats,
)
from .reuse import ReuseConfig, ReuseStats, TreeCache, count_nodes
from .player import (
    MCTSPlayer, Decision, BatchedMCTSPlayerGroup, build_net, load_policy, make_player,
)

__all__ = [
    "action_key", "action_from_key", "ActionKey", "Node",
    "OutcomeFn", "VanillaOutcome", "MLBOutcome", "ExternalOutcome",
    "default_outcome_for", "margin_to_value", "is_stuck_state",
    "OBS_DIM", "MLB_OBS_DIM", "encode_obs", "encode_obs_mlb",
    "ObsEncoder", "V7Encoder", "MLBEncoder", "get_encoder", "is_set_encoder",
    "SetEncoder", "ItemCaps", "SCALAR_DIM", "KEY_VOCAB_SIZE",
    "featurize_actions_set", "ACT_NUM_DIM", "SetPolicyValueNet", "StateEmbedding",
    "SetNNPolicy", "BatchedSetNNPolicy", "make_policy", "build_net",
    "ACTION_FEATURE_DIM", "featurize_action", "featurize_actions",
    "PolicyValueFn", "PolicyValueBase", "UniformPolicy", "NNPolicy",
    "PolicyValueNet", "MCTS", "MCTSConfig", "MCTSPlayer", "Decision",
    "HeuristicConfig", "HandHeuristic", "hand_action_scores",
    "heuristic_distribution", "shape_priors",
    "BatchedNNPolicy", "BatchedSearch", "SearchRequest", "SearchResult", "BatchStats",
    "ReuseConfig", "ReuseStats", "TreeCache", "count_nodes",
    "BatchedMCTSPlayerGroup", "load_policy", "make_player",
]
