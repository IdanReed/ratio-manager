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
import re
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


def read_parquet(
    root: str | Path,
    kinds: Iterable[str] | None = None,
    days: Iterable[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield events from the nightly export, in day then sequence order.

    The exporter writes ``<root>/kind=<kind>/date=<YYYY-MM-DD>/part-0.parquet``
    with the envelope as columns and ``data`` as a JSON string, so this yields
    the same dicts :func:`read_jsonl` does and every downstream function is
    indifferent to which path the events came from.

    This is the fast path. A day the exporter has not reached yet is only in
    JSONL, so a full history generally means reading both.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "reading the Parquet export needs pyarrow: pip install '.[parquet]'. "
            "The JSONL path (read_jsonl / read_days) needs only zstandard."
        ) from e

    root = Path(root)
    kind_dirs = sorted(root.glob("kind=*"))
    if kinds is not None:
        want = set(kinds)
        kind_dirs = [d for d in kind_dirs if d.name.split("=", 1)[1] in want]

    # Day order across kinds, so a consumer sees the log roughly as it was
    # written rather than kind by kind.
    files: list[tuple[str, Path]] = []
    for kd in kind_dirs:
        for dd in sorted(kd.glob("date=*")):
            day = dd.name.split("=", 1)[1]
            if days is not None and day not in set(days):
                continue
            for f in sorted(dd.glob("*.parquet")):
                files.append((day, f))
    files.sort(key=lambda p: p[0])

    for _day, f in files:
        # `ParquetFile`, not `read_table`: the latter goes through the dataset
        # API, which infers Hive partitioning from the `kind=`/`date=`
        # directories and then collides with the real `kind` column the
        # exporter also writes ("incompatible types: string vs dictionary").
        # Reading the file directly sidesteps the discovery entirely.
        table = pq.ParquetFile(f).read()
        cols = {name: table.column(name).to_pylist() for name in table.schema.names}
        n = table.num_rows
        for i in range(n):
            raw = cols.get("data", [None] * n)[i]
            ts = cols["ts"][i]
            yield {
                "v": cols.get("v", [1] * n)[i],
                # The column is a timestamp; the rest of the pipeline expects
                # the envelope's string form.
                "ts": (
                    ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
                    if hasattr(ts, "astimezone")
                    else str(ts)
                ),
                "seq": cols.get("seq", [0] * n)[i],
                "kind": cols["kind"][i],
                "box": cols.get("box", [None] * n)[i],
                "hash": cols.get("hash", [None] * n)[i],
                "actor": cols.get("actor", [None] * n)[i],
                "data": json.loads(raw) if isinstance(raw, str) and raw else {},
            }


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
    """One (release, box) pair the policy considered — a `plan` in the log."""

    box: str
    score: float
    features: dict[str, float] = field(default_factory=dict)
    chosen: bool = False
    #: Whether the guard and the gates would have allowed this add at all.
    eligible: bool = True
    #: Why it was not eligible. These are the behaviour policy's own negative
    #: examples, and the only ones the log holds: nothing is ever refused by
    #: the guard, because the baseline filters first.
    why: list[str] = field(default_factory=list)
    #: The scoring basis, e.g. "prior 1.1 x demand 1.00" or
    #: "table seedit4me|race|m|mid (n=7) x demand 2.30".
    basis: str = ""


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
    #: The release the decision was about. Every feature about the release
    #: itself has to be joined from `release_index` by this.
    tl_id: str | None = None
    #: The behaviour policy's probability for the action it took, when the
    #: choice was randomized. ``None`` means it was deterministic — which
    #: importance-weighted estimators have to be told about rather than
    #: silently treating as 1.0. Today this is always ``None``.
    probability: float | None = None
    #: Which randomized experiment the decision belonged to, if any. Always
    #: ``None`` so far: the join-delay experiment is not firing.
    experiment: str | None = None

    @property
    def chosen(self) -> list[Candidate]:
        return [c for c in self.candidates if c.chosen]

    @property
    def rejected(self) -> list[Candidate]:
        return [c for c in self.candidates if not c.chosen]


#: `basis` strings look like "prior 1.1 x demand 1.00" or
#: "table seedit4me|race|m|mid (n=7) x demand 2.30". The demand multiplier is
#: the only place a decision event records the swarm it was judging, so it is
#: worth recovering rather than re-deriving from a scrape join that may have
#: happened minutes later.
_DEMAND_IN_BASIS = re.compile(r"demand\s+([0-9]*\.?[0-9]+)")
_TABLE_N = re.compile(r"\(n=(\d+)\)")


def parse_decisions(events: Iterable[dict[str, Any]]) -> list[Decision]:
    """Pull every decision out of an event stream.

    The emitted shape is **not** the one the design doc describes in prose.
    What the manager actually writes is::

        {"kind": "decision", "hash": "...", "data": {
            "name": "...",
            "chosen": ["seedit4me"],            # box names, not a per-plan flag
            "plans": [{"box", "est", "basis", "eligible", "why", "evict",
                       "placeable_gb"}],
            "extra": {"tl_id", "size", "experiment"}}}

    So the score is `est`, the per-candidate rejection reasons are `why`, and
    there is no per-candidate `features` dict — the features have to be joined
    from the release book by `tl_id` (see :func:`release_index`). Reading
    `candidates`/`score` instead silently yields zero decisions, which is the
    failure mode this docstring exists to prevent.
    """
    out: list[Decision] = []
    for e in events:
        if e.get("kind") != "decision":
            continue
        d = e.get("data") or {}
        chosen_boxes = set(d.get("chosen") or [])
        extra = d.get("extra") or {}
        cands = []
        for p in d.get("plans") or []:
            basis = str(p.get("basis", "") or "")
            m = _DEMAND_IN_BASIS.search(basis)
            n = _TABLE_N.search(basis)
            feats: dict[str, float] = {
                "placeable_gb": float(p.get("placeable_gb", 0.0) or 0.0),
                "eligible": 1.0 if p.get("eligible") else 0.0,
                "n_evict": float(len(p.get("evict") or [])),
            }
            if m:
                feats["demand"] = float(m.group(1))
            if n:
                # How many races the scoring cell rested on. A cell with three
                # races and one with seventy are not the same evidence, and the
                # baseline treats them identically once past its threshold.
                feats["table_n"] = float(n.group(1))
            size = extra.get("size")
            if isinstance(size, (int, float)):
                feats["size_gb"] = float(size) / GB
            cands.append(
                Candidate(
                    box=str(p.get("box", "")),
                    score=float(p.get("est", 0.0) or 0.0),
                    features=feats,
                    chosen=str(p.get("box", "")) in chosen_boxes,
                    eligible=bool(p.get("eligible")),
                    why=[str(w) for w in (p.get("why") or [])],
                    basis=basis,
                )
            )
        out.append(
            Decision(
                seq=int(e.get("seq", 0)),
                ts=ts_of(e),
                week=iso_week(e),
                actor=e.get("actor") or d.get("actor") or "",
                trigger=str(d.get("trigger") or e.get("kind") or ""),
                hash=e.get("hash"),
                name=str(d.get("name", "")),
                candidates=cands,
                tl_id=(str(extra["tl_id"]) if extra.get("tl_id") is not None else None),
                # Nothing emits a propensity today: the baseline is
                # deterministic and `extra.experiment` is always null. Offline
                # methods that need one have to wait for the bandit.
                probability=(
                    float(d["probability"])
                    if isinstance(d.get("probability"), (int, float))
                    else None
                ),
                experiment=(
                    str(extra["experiment"]) if extra.get("experiment") is not None else None
                ),
            )
        )
    return out


@dataclass
class ReleaseFacts:
    """What the log knows about one release, gathered across event kinds."""

    tl_id: str
    name: str = ""
    category: str = ""
    size_gb: float = 0.0
    freeleech: bool = False
    private: bool | None = None
    files: int = 0
    first_seen: datetime | None = None
    #: Scrape curve: (timestamp, seeders, leechers, downloaded), in time order.
    scrapes: list[tuple[datetime, int, int, int]] = field(default_factory=list)

    def demand_at(self, when: datetime) -> float | None:
        """Leechers per seeder at the scrape nearest before `when`."""
        best = None
        for ts, s, l, _d in self.scrapes:
            if ts <= when:
                best = (s, l)
            else:
                break
        if best is None:
            return None
        return best[1] / (best[0] + 1.0)

    def peak_leechers(self) -> int:
        return max((l for _t, _s, l, _d in self.scrapes), default=0)


def release_index(events: Iterable[dict[str, Any]]) -> dict[str, ReleaseFacts]:
    """Build the release book from `release`, `rss_item`, `announce` and `scrape`.

    A decision event names only a `tl_id`, so every feature about the release
    itself — its size, category, freeleech flag and swarm curve — comes from
    here. This is also the input the swarm generators fit on.
    """
    book: dict[str, ReleaseFacts] = {}

    def get(tl_id: str) -> ReleaseFacts:
        return book.setdefault(tl_id, ReleaseFacts(tl_id=tl_id))

    for e in events:
        kind = e.get("kind")
        d = e.get("data") or {}
        tl_id = d.get("tl_id")
        if tl_id is None:
            continue
        tl_id = str(tl_id)
        r = get(tl_id)
        ts = ts_of(e)
        if r.first_seen is None or ts < r.first_seen:
            r.first_seen = ts

        if kind in ("release", "announce"):
            r.name = str(d.get("name") or d.get("title") or r.name)
            r.category = str(d.get("category") or r.category)
            if isinstance(d.get("size"), (int, float)):
                r.size_gb = float(d["size"]) / GB
            if isinstance(d.get("freeleech"), bool):
                r.freeleech = d["freeleech"]
            if isinstance(d.get("private"), bool):
                r.private = d["private"]
            if isinstance(d.get("files"), (int, float)):
                r.files = int(d["files"])
        elif kind == "rss_item":
            r.name = str(d.get("title") or r.name)
            r.category = str(d.get("category") or r.category)
        elif kind == "scrape":
            r.scrapes.append(
                (
                    ts,
                    int(d.get("complete") or 0),
                    int(d.get("incomplete") or 0),
                    int(d.get("downloaded") or 0),
                )
            )

    for r in book.values():
        r.scrapes.sort(key=lambda t: t[0])
    return book


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
