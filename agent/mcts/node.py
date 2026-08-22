"""
node.py — MCTS tree node.

One node per (state, action) edge that has been visited. Children are
indexed by action key. The root node represents "before any action chosen
from current real state."

Stats stored:
  N — visit count
  W — total backed-up value (sum of leaf values from sub-tree)
  Q — mean value W/N (computed lazily)
  P — prior probability assigned by the policy network at expansion time

`is_terminal` means "descent stops here". Under MLB that is not only GAME_OVER:
a `PVP_WAIT` state (hands exhausted at a Nemesis, waiting for the opponent) and a
readied-but-not-started Nemesis both have NO legal actions and cannot be stepped
by this player. Those nodes are marked terminal with `stop_reason="stuck"` and are
valued by the OutcomeFn's pending estimate rather than by a win/loss.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .action import ActionKey


@dataclass
class Node:
    prior: float = 0.0          # P(a|s) from policy net at parent's expansion
    visit_count: int = 0        # N(s, a)
    value_sum: float = 0.0      # W(s, a)
    children: dict[ActionKey, "Node"] = field(default_factory=dict)
    is_expanded: bool = False
    is_terminal: bool = False   # descent stops here (game over OR no legal actions)
    terminal_value: float = 0.0  # only valid if is_terminal
    stop_reason: str = ""        # "" | "game_over" | "stuck" | "no_actions"

    @property
    def mean_value(self) -> float:
        """Q(s, a) = W / N, with a virtual prior to avoid div-by-zero."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def add_child(self, key: ActionKey, prior: float) -> "Node":
        child = Node(prior=prior)
        self.children[key] = child
        return child
