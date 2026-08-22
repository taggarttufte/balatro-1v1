"""train — self-play + replay buffer + trainer + checkpointing for the MP agent."""
from .trajectory import Sample, ReplayBuffer
from .agent import SelfPlayAgent, EpisodeResult
from .trainer import Trainer
from .loop import ColdTrainer, TrainConfig, Counters, rolling_summary
from .population import (
    PopulationMember, PopulationConfig, CheckpointHistory, build_population,
    instantiate as instantiate_population, population_summary,
)
from .selfplay import (
    make_sample, SampleCollector, RecordedDecision, TournamentSpec,
    run_tournament_generation, GenerationSamples, GenerationMetrics,
    MLBTrainConfig, MLBTrainer, normalized_ranks, value_targets_from_result,
    assign_value_targets, vanilla_boss_target, load_target_fn, external_outcome_for,
    play_solo_external_episode, SoloEpisodeResult, solo_metrics,
)
from .checkpoint import (
    save_checkpoint, load_checkpoint, latest_checkpoint,
    CHECKPOINT_VERSION, CHECKPOINT_KIND,
)

__all__ = [
    "Sample", "ReplayBuffer", "SelfPlayAgent", "EpisodeResult", "Trainer",
    "ColdTrainer", "TrainConfig", "Counters", "rolling_summary",
    "save_checkpoint", "load_checkpoint", "latest_checkpoint",
    "CHECKPOINT_VERSION", "CHECKPOINT_KIND",
    # Phase 4 W2: tournament-driven training
    "PopulationMember", "PopulationConfig", "CheckpointHistory", "build_population",
    "instantiate_population", "population_summary",
    "make_sample", "SampleCollector", "RecordedDecision", "TournamentSpec",
    "run_tournament_generation", "GenerationSamples", "GenerationMetrics",
    "MLBTrainConfig", "MLBTrainer", "normalized_ranks", "value_targets_from_result",
    "assign_value_targets", "vanilla_boss_target", "load_target_fn",
    "external_outcome_for", "play_solo_external_episode", "SoloEpisodeResult",
    "solo_metrics",
]
