"""
matrix.py — N final scores at one Nemesis -> N x N outcome/margin matrices, population rank,
per-ante score distribution.  Design doc §6: "Build the full N x N comparison matrix at each
nemesis blind -- it's a sort, not a simulation."  Everything here is O(N^2) numpy over scores
the runner already collected; no game state is touched.

Serialization: one ``.npz`` per ante (``ante_<ante:04d>.npz``: ``scores``, ``outcome``,
``log_margin``, ``rank``, ``losers``) plus a JSONL summary (one line per ante: n_present,
mean/std/min/max/quantiles, tie_fraction, losers) under a run directory -- see
``write_run`` / TOURNAMENT_NOTES.md "file formats".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "outcome_matrix", "log_margin_matrix", "population_rank", "score_distribution",
    "tie_fraction", "AnteMatrix", "write_run",
]


def outcome_matrix(scores: np.ndarray) -> np.ndarray:
    """``+1`` row beats col (row's score is strictly higher), ``-1`` row loses to col,
    ``0`` on a tie, on the diagonal, or wherever either agent is absent this ante (NaN
    score -- e.g. already eliminated in a prior round).  Server rule (decision 0.2):
    strictly-lower score loses, an exact tie costs nobody -- so ``0`` doubles as "tie" and
    "no comparison possible"; ``tie_fraction`` below is the metric that needs to
    distinguish real ties from absences, and does (it only counts pairs where BOTH sides
    are present)."""
    diff = scores[:, None] - scores[None, :]
    with np.errstate(invalid="ignore"):
        out = np.sign(diff)
    out = np.where(np.isnan(diff), 0.0, out)
    np.fill_diagonal(out, 0.0)
    return out


def log_margin_matrix(scores: np.ndarray) -> np.ndarray:
    """``log1p(score_i) - log1p(score_j)``: antisymmetric, zero diagonal, NaN wherever
    either side is absent.  Chip scores span orders of magnitude by ante 6-8 (MLB scaling
    is unbounded past ante 8 too), so a log-space margin is the metric that stays
    comparable across antes; ``log1p`` avoids ``log(0)`` for a genuine 0-chip score
    (a fully-forfeited deck-out Nemesis, MLB_NOTES.md §2 1.3e-b)."""
    with np.errstate(invalid="ignore"):
        ls = np.log1p(scores)
    m = ls[:, None] - ls[None, :]
    np.fill_diagonal(m, 0.0)
    return m


def population_rank(scores: np.ndarray) -> np.ndarray:
    """1 = highest score; tied scores share the average rank (matches
    ``scipy.stats.rankdata(method="average")`` without the scipy dependency); ``NaN`` stays
    ``NaN`` for an absent agent."""
    n = scores.shape[0]
    rank = np.full(n, np.nan)
    present = np.where(~np.isnan(scores))[0]
    if present.size == 0:
        return rank
    vals = scores[present]
    order = np.argsort(-vals, kind="stable")
    sorted_vals = vals[order]
    ranks = np.empty(len(order), dtype=float)
    i = 0
    m = len(order)
    while i < m:
        j = i
        while j + 1 < m and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[i:j + 1] = avg_rank
        i = j + 1
    rank[present[order]] = ranks
    return rank


def score_distribution(scores: np.ndarray) -> dict:
    """Quantiles + mean/std over the PRESENT scores this ante -- "layer 1" from the
    self-play assessment (per-cell score distribution at ante k)."""
    present = scores[~np.isnan(scores)]
    if present.size == 0:
        return {"n_present": 0, "mean": None, "std": None, "min": None, "max": None,
                "quantiles": {}}
    qs = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    quantiles = {str(q): float(np.quantile(present, q)) for q in qs}
    return {
        "n_present": int(present.size),
        "mean": float(present.mean()),
        "std": float(present.std()),
        "min": float(present.min()),
        "max": float(present.max()),
        "quantiles": quantiles,
    }


def tie_fraction(outcome: np.ndarray, present_mask: np.ndarray) -> float:
    """Degeneracy metric (design doc §6): the fraction of OFF-DIAGONAL pairs, among agents
    PRESENT this ante, whose outcome is an exact tie.  A collapsed population (N identical
    policies) drives this toward 1; a heterogeneous one keeps it well below 1.  ``NaN`` when
    fewer than 2 agents are present (no pairs to measure)."""
    idx = np.where(present_mask)[0]
    n = idx.size
    if n < 2:
        return float("nan")
    sub = outcome[np.ix_(idx, idx)]
    total_pairs = n * (n - 1)
    ties = int(np.count_nonzero(sub == 0)) - n   # subtract the n diagonal zeros
    return ties / total_pairs


@dataclass
class AnteMatrix:
    ante: int
    n_agents: int
    scores: np.ndarray
    outcome: np.ndarray
    log_margin: np.ndarray
    rank: np.ndarray
    stats: dict
    tie_fraction: float
    losers: list = field(default_factory=list)

    @classmethod
    def build(cls, ante: int, n_agents: int, scores_by_agent: dict, losers=()) -> "AnteMatrix":
        scores = np.full(n_agents, np.nan)
        for i, s in scores_by_agent.items():
            scores[i] = s
        outcome = outcome_matrix(scores)
        margin = log_margin_matrix(scores)
        rank = population_rank(scores)
        present_mask = ~np.isnan(scores)
        stats = score_distribution(scores)
        tf = tie_fraction(outcome, present_mask)
        return cls(ante=ante, n_agents=n_agents, scores=scores, outcome=outcome,
                    log_margin=margin, rank=rank, stats=stats, tie_fraction=tf,
                    losers=sorted(losers))

    def save_npz(self, path: Path) -> None:
        np.savez(path, ante=np.array(self.ante), scores=self.scores, outcome=self.outcome,
                  log_margin=self.log_margin, rank=self.rank,
                  losers=np.array(self.losers, dtype=np.int64))

    def summary_dict(self) -> dict:
        d = {"ante": self.ante, "n_agents": self.n_agents, "tie_fraction": self.tie_fraction,
             "losers": list(self.losers)}
        d.update(self.stats)
        return d


def write_run(out_dir, result) -> None:
    """One ``ante_<ante:04d>.npz`` per Nemesis played + ``summary.jsonl`` (one line per
    ante) + ``meta.json`` (seed, n_agents, life_rule, max_ante, deck/stake, fan-out method,
    wall clock, final lives / alive-at-end) under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for m in result.ante_matrices:
        m.save_npz(out_dir / f"ante_{m.ante:04d}.npz")
        lines.append(json.dumps(m.summary_dict(), sort_keys=True))
    (out_dir / "summary.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""),
                                            encoding="utf-8")
    meta = {
        "seed": result.seed, "n_agents": result.n_agents, "life_rule": result.life_rule,
        "max_ante": result.max_ante, "deck_key": result.deck_key, "stake": result.stake,
        "fanout_method": result.fanout_method, "wall_clock_s": result.wall_clock_s,
        "steps_total": result.steps_total, "final_lives": result.final_lives,
        "alive_at_end": result.alive_at_end,
        "last_score": {str(k): list(v) for k, v in result.last_score.items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")
