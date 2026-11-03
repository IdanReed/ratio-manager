"""The feature vector, in one place.

Every model here — the ranker, the bandit, the decision transformer — reads
the same features in the same order, because the moment two models disagree
about what column 7 means, comparing them stops meaning anything.

Features are deliberately only what the manager knows *at decision time*.
Anything measured afterwards is a label, not a feature. The two easiest ways
to produce a model that looks excellent and is useless are to leak the
outcome into the features and to split the data at random; this module
addresses the first and :func:`ratio_ml.data.split_by_week` the second.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .data import Candidate, Decision

#: Boxes in a fixed order, so the one-hot block never shifts.
BOXES = ("home", "seedit4me", "pulsed")

#: Sources a decision can come from, same reason.
SOURCES = ("announce", "scout", "manual")

NAMES: tuple[str, ...] = (
    "log_size_gb",
    "demand",
    "log_demand",
    "seeders",
    "leechers",
    "freeleech",
    "hour_sin",
    "hour_cos",
    "weekend",
    "box_free_gb",
    "box_in_flight_gb",
    "box_active_downloads",
    "box_dl_speed_mib",
    "box_allowance_left",
    "baseline_score",
    *(f"box_is_{b}" for b in BOXES),
    *(f"source_is_{s}" for s in SOURCES),
)


def _f(d: dict[str, float], *keys: str, default: float = 0.0) -> float:
    """First present key, else the default. The event log's field names moved
    once; this keeps a loader working across both."""
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return default


def vector(decision: Decision, candidate: Candidate) -> list[float]:
    """The feature vector for one (decision, candidate) pair."""
    f = candidate.features
    size = _f(f, "size_gb", "size")
    demand = _f(f, "demand")
    hour = decision.ts.hour + decision.ts.minute / 60.0
    seeders = _f(f, "seeders", "num_complete")
    leechers = _f(f, "leechers", "num_incomplete")
    out = [
        math.log1p(max(size, 0.0)),
        demand,
        math.log1p(max(demand, 0.0)),
        seeders,
        leechers,
        1.0 if _f(f, "freeleech", default=1.0) else 0.0,
        math.sin(2 * math.pi * hour / 24.0),
        math.cos(2 * math.pi * hour / 24.0),
        1.0 if decision.ts.weekday() >= 5 else 0.0,
        _f(f, "box_free_gb", "free_gb", "placeable_gb"),
        _f(f, "box_in_flight_gb", "in_flight_gb"),
        _f(f, "box_active_downloads", "active"),
        _f(f, "box_dl_speed_mib", "dl_speed_mib"),
        _f(f, "box_allowance_left", "allowance_left_gb"),
        candidate.score,
    ]
    out += [1.0 if candidate.box == b else 0.0 for b in BOXES]
    src = decision.trigger or ""
    out += [1.0 if src == s else 0.0 for s in SOURCES]
    assert len(out) == len(NAMES), f"{len(out)} features but {len(NAMES)} names"
    return out


@dataclass
class Example:
    """One training row: features at decision time, and the realized return."""

    x: list[float]
    y: float
    box: str
    week: str
    #: The behaviour policy's probability, where the decision was randomized.
    propensity: float | None = None


def examples(
    pairs: list[tuple[Decision, "object"]],
    horizon: str = "ratio_24h",
    gb_quota: float = 0.35,
) -> list[Example]:
    """Build training rows from joined (decision, outcome) pairs.

    A pair whose horizon has not elapsed is skipped, not zero-filled.
    """
    rows: list[Example] = []
    for d, o in pairs:
        metered = o.box == "seedit4me"
        y = o.reward(horizon=horizon, gb_quota=gb_quota, metered=metered)
        if y is None:
            continue
        cand = next((c for c in d.chosen if c.box == o.box), None)
        if cand is None:
            continue
        rows.append(
            Example(
                x=vector(d, cand),
                y=y,
                box=o.box,
                week=d.week,
                propensity=d.probability,
            )
        )
    return rows
