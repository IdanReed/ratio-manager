"""Reading the event log, and turning it into examples.

The manager writes one JSON object per line into daily
``/mnt/slow/ratio-manager/events/YYYY-MM-DD.jsonl.zst`` files and exports each
day to Parquet overnight. Both are read here; Parquet is the fast path and the
JSONL is the fallback for a day that has not been exported yet.

Every event shares an envelope::

    {"v": 1, "ts": "...", "seq": 184467, "kind": "torrent_state",
     "box": "pulsed", "hash": "2cfeb7...", "actor": null, "data": {...}}

Three things come out of this module, and the order matters:

``decisions``
    What the policy chose, with the candidates and features it chose from.
    This is the behaviour policy, and offline RL cannot work without it.

``outcomes``
    What each add actually earned, joined to the decision that made it. The
    reward is computed after the fact from the ``races`` rollups, so no reward
    has to be guessed at decision time.

``splits``
    Train/test by **calendar week**, never at random. Races on the same
    release, on two boxes, minutes apart, are the same event; a random split
    puts one in train and one in test and every model then looks clairvoyant.
"""

from __future__ import annotations

import io
import json
import zstandard
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

GB = 1_073_741_824.0

#: Event kinds the training pipeline reads. Others are ignored rather than
#: rejected, so a new kind in the log never breaks a loader.
KINDS = {
    "announce",
    "market",
    "listing",
    "scrape",
    "decision",
    "action",
    "registration",
    "torrent_state",
    "box_state",
    "capacity",
    "rss_item",
    "peers",
    "tracker_status",
}


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield every event in one daily file, zstd-compressed or plain."""
    path = Path(path)
    if path.suffix == ".zst":
        with path.open("rb") as fh:
            reader = zstandard.ZstdDecompressor().stream_reader(fh)
            stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def read_days(dirpath: str | Path, days: Iterable[str] | None = None) -> Iterator[dict[str, Any]]:
    """Yield events from a directory of daily files, in day order.

    ``days`` selects specific ``YYYY-MM-DD`` days; without it every file in
    the directory is read.
    """
    d = Path(dirpath)
    files = sorted(d.glob("*.jsonl.zst")) + sorted(d.glob("*.jsonl"))
    for f in files:
        day = f.name.split(".")[0]
        if days is not None and day not in set(days):
            continue
        yield from read_jsonl(f)


def ts_of(event: dict[str, Any]) -> datetime:
    """The envelope's timestamp, as an aware UTC datetime."""
    raw = event["ts"]
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    return datetime.fromisoformat(raw).astimezone(timezone.utc)


def iso_week(event: dict[str, Any]) -> str:
    """``2026-W44``. The unit every split is made on."""
    y, w, _ = ts_of(event).isocalendar()
    return f"{y}-W{w:02d}"


@dataclass
class Candidate:
    """One (release, box) pair the policy considered, with its features."""

    box: str
    score: float
    features: dict[str, float] = field(default_factory=dict)
    chosen: bool = False


@dataclass
class Decision:
    """One decision, as the ``decision`` event recorded it."""

    seq: int
    ts: datetime
    week: str
    actor: str
    trigger: str
    hash: str | None
    name: str
    candidates: list[Candidate]
    #: The behaviour policy's probability for the action it took, when the
    #: choice was randomized. ``None`` means it was deterministic — which
    #: importance-weighted estimators have to be told about rather than
    #: silently treating as 1.0.
    probability: float | None = None

    @property
    def chosen(self) -> list[Candidate]:
        return [c for c in self.candidates if c.chosen]


def parse_decisions(events: Iterable[dict[str, Any]]) -> list[Decision]:
    """Pull every decision out of an event stream."""
    out: list[Decision] = []
    for e in events:
        if e.get("kind") != "decision":
            continue
        d = e.get("data") or {}
        cands = []
        for c in d.get("candidates") or []:
            cands.append(
                Candidate(
                    box=c.get("box", ""),
                    score=float(c.get("score", 0.0) or 0.0),
                    features={
                        k: float(v)
                        for k, v in (c.get("features") or {}).items()
                        if isinstance(v, (int, float))
                    },
                    chosen=bool(c.get("chosen", False)),
                )
            )
        out.append(
            Decision(
                seq=int(e.get("seq", 0)),
                ts=ts_of(e),
                week=iso_week(e),
                actor=e.get("actor") or d.get("actor") or "",
                trigger=d.get("trigger", ""),
                hash=e.get("hash"),
                name=d.get("name", ""),
                candidates=cands,
                probability=(
                    float(d["probability"]) if isinstance(d.get("probability"), (int, float)) else None
                ),
            )
        )
    return out


@dataclass
class Outcome:
    """What one add earned, keyed the way the ``races`` table keys it."""

    box: str
    hash: str
    name: str
    size_gb: float
    added_at: int
    demand_at_add: float
    ratio_1h: float | None
    ratio_3h: float | None
    ratio_24h: float | None
    ratio_7d: float | None
    last_ratio: float | None
    uploaded_gb: float | None
    removed_rule: str | None

    def reward(
        self,
        horizon: str = "ratio_24h",
        gb_quota: float = 0.35,
        metered: bool = False,
    ) -> float | None:
        """Credited upload in GB, less what the download cost on a metered box.

        ``None`` when the horizon has not elapsed yet — a missing outcome is
        not a zero one, and training on it as zero teaches the model that
        everything recent was worthless.
        """
        r = getattr(self, horizon)
        if r is None:
            return None
        up = r * self.size_gb
        cost = self.size_gb * gb_quota if metered else 0.0
        return up - cost


def join_outcomes(
    decisions: list[Decision], outcomes: Iterable[Outcome]
) -> list[tuple[Decision, Outcome]]:
    """Match each decision to the race it produced, by hash and box.

    A decision with no outcome is dropped rather than zero-filled. Reasons it
    legitimately has none: the add was refused by the guard, the release was
    never registered, or the horizon has not passed.
    """
    by_key: dict[tuple[str, str], Outcome] = {(o.box, o.hash): o for o in outcomes}
    pairs = []
    for d in decisions:
        if not d.hash:
            continue
        for c in d.chosen:
            o = by_key.get((c.box, d.hash))
            if o is not None:
                pairs.append((d, o))
    return pairs


def split_by_week(
    items: list[tuple[Decision, Outcome]], holdout_weeks: int = 2
) -> tuple[list[tuple[Decision, Outcome]], list[tuple[Decision, Outcome]]]:
    """Train on everything up to the last ``holdout_weeks`` weeks; test on those.

    Chronological, not random, and by week rather than by row: the same
    release raced on two boxes minutes apart must not straddle the split.
    """
    weeks = sorted({d.week for d, _ in items})
    if len(weeks) <= holdout_weeks:
        return [], list(items)
    test_weeks = set(weeks[-holdout_weeks:])
    train = [p for p in items if p[0].week not in test_weeks]
    test = [p for p in items if p[0].week in test_weeks]
    return train, test
