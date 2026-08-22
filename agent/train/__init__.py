"""train — self-play + replay buffer + trainer + checkpointing for the MP agent."""
from .trajectory import Sample, ReplayBuffer
from .agent import SelfPlayAgent, EpisodeResult
from .trainer import Trainer
from .loop import ColdTrainer, TrainConfig, Counters, rolling_summary
from .checkpoint import (
    save_checkpoint, load_checkpoint, latest_checkpoint,
    CHECKPOINT_VERSION, CHECKPOINT_KIND,
)

__all__ = [
    "Sample", "ReplayBuffer", "SelfPlayAgent", "EpisodeResult", "Trainer",
    "ColdTrainer", "TrainConfig", "Counters", "rolling_summary",
    "save_checkpoint", "load_checkpoint", "latest_checkpoint",
    "CHECKPOINT_VERSION", "CHECKPOINT_KIND",
]
