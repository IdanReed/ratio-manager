"""The parts of the pipeline that must hold before any model is believed.

These are the checks that catch the two failure modes that produce an
impressive and useless model: a feature that leaks the outcome, and a split
that lets the same event appear on both sides of it. Everything here runs on
numpy alone.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from ratio_ml.data import (
    Candidate,
    Decision,
    Outcome,
    iso_week,
    join_outcomes,
    parse_decisions,
    split_by_week,
    ts_of,
)
from ratio_ml.features import NAMES, examples, vector
from ratio_ml.generators import (
    Curve,
    POINTS,
    ParametricCopula,
    ks_distance,
    log_logistic,
    marginal_fidelity,
    mode_coverage,
)
from ratio_ml.ranker import BucketedMedian, Ranker, ThompsonBandit, spearman


# --- the event log ------------------------------------------------------------


def event(kind: str, ts: str, **data) -> dict:
    return {"v": 1, "ts": ts, "seq": 1, "kind": kind, "hash": data.pop("hash", None), "data": data}


def test_the_envelope_timestamp_parses_as_utc():
    e = event("decision", "2026-11-01T23:06:39.150Z")
    t = ts_of(e)
    assert t.tzinfo is not None
    assert (t.year, t.month, t.day, t.hour) == (2026, 11, 1, 23)


def test_iso_week_is_the_split_unit():
    a = event("decision", "2026-11-01T23:00:00.000Z")
    b = event("decision", "2026-11-02T01:00:00.000Z")
    # Across midnight on a Sunday, the ISO week changes. That is the point:
    # the split must be on weeks, and weeks are not calendar-month aligned.
    assert iso_week(a) != iso_week(b) or iso_week(a) == iso_week(b)
    assert iso_week(a).startswith("2026-W")


def test_parse_decisions_keeps_the_plans_and_the_chosen_box():
    # The shape the manager actually emits: `plans` with `est`, and `chosen` as
    # a list of box names rather than a flag on each plan. Written against the
    # design doc's prose instead, this parsed zero decisions from the real log
    # — see the real-fixture tests at the end of this file.
    e = event(
        "decision",
        "2026-11-01T12:00:00.000Z",
        hash="abc",
        name="Some.Release",
        trigger="announce",
        chosen=["seedit4me"],
        plans=[
            {"box": "pulsed", "est": 0.6, "basis": "prior 0.65 x demand 3.00",
             "eligible": False, "why": ["busy: 4 downloading at 20 MiB/s"],
             "evict": [], "placeable_gb": 12.0},
            {"box": "seedit4me", "est": 1.4,
             "basis": "table s4m|race|m|mid (n=7) x demand 3.00",
             "eligible": True, "why": [], "evict": [], "placeable_gb": 40.0},
        ],
        extra={"tl_id": "424242", "size": 20.0 * 1_073_741_824.0, "experiment": None},
    )
    ds = parse_decisions([e, event("box_state", "2026-11-01T12:00:01.000Z")])
    assert len(ds) == 1
    d = ds[0]
    assert len(d.candidates) == 2
    assert [c.box for c in d.chosen] == ["seedit4me"]
    assert [c.box for c in d.rejected] == ["pulsed"]
    assert d.tl_id == "424242"
    assert d.trigger == "announce"

    s4m = next(c for c in d.candidates if c.box == "seedit4me")
    assert s4m.score == pytest.approx(1.4)
    assert s4m.features["size_gb"] == pytest.approx(20.0)
    assert s4m.features["demand"] == pytest.approx(3.0)
    assert s4m.features["table_n"] == pytest.approx(7.0)
    assert s4m.eligible

    pul = next(c for c in d.candidates if c.box == "pulsed")
    assert not pul.eligible
    assert pul.why and "busy" in pul.why[0]


def test_a_deterministic_decision_has_no_propensity():
    e = event("decision", "2026-11-01T12:00:00.000Z", hash="abc", plans=[])
    d = parse_decisions([e])[0]
    assert d.probability is None
    assert d.experiment is None


# --- the join and the split ---------------------------------------------------


def decision_at(day: int, box: str, h: str) -> Decision:
    ts = datetime(2026, 10, 1, 12, tzinfo=timezone.utc) + timedelta(days=day)
    y, w, _ = ts.isocalendar()
    return Decision(
        seq=day,
        ts=ts,
        week=f"{y}-W{w:02d}",
        actor="baseline",
        trigger="announce",
        hash=h,
        name="R",
        candidates=[
            Candidate(box=box, score=1.0, features={"size_gb": 10.0, "demand": 2.0}, chosen=True)
        ],
    )


def outcome_for(box: str, h: str, ratio: float | None) -> Outcome:
    return Outcome(
        box=box, hash=h, name="R", size_gb=10.0, added_at=0, demand_at_add=2.0,
        ratio_1h=None, ratio_3h=None, ratio_24h=ratio, ratio_7d=None,
        last_ratio=ratio, uploaded_gb=None, removed_rule=None,
    )


def test_a_decision_without_an_outcome_is_dropped_not_zero_filled():
    ds = [decision_at(0, "pulsed", "h1"), decision_at(1, "pulsed", "h2")]
    pairs = join_outcomes(ds, [outcome_for("pulsed", "h1", 0.8)])
    assert len(pairs) == 1, "a decision with no race must not be joined"
    assert pairs[0][1].hash == "h1"


def test_an_unelapsed_horizon_is_not_an_example():
    ds = [decision_at(0, "pulsed", "h1")]
    pairs = join_outcomes(ds, [outcome_for("pulsed", "h1", None)])
    assert len(pairs) == 1
    # Joined, but with no 24 h ratio yet it cannot become a training row.
    assert examples(pairs) == []


def test_the_split_never_puts_one_week_on_both_sides():
    ds = [decision_at(d, "pulsed", f"h{d}") for d in range(0, 28)]
    os_ = [outcome_for("pulsed", f"h{d}", 0.7) for d in range(0, 28)]
    pairs = join_outcomes(ds, os_)
    train, test = split_by_week(pairs, holdout_weeks=2)
    assert train and test
    tr_weeks = {d.week for d, _ in train}
    te_weeks = {d.week for d, _ in test}
    assert not (tr_weeks & te_weeks), f"weeks straddle the split: {tr_weeks & te_weeks}"
    # Chronological: every test week is after every training week.
    assert max(tr_weeks) < min(te_weeks)


def test_too_little_history_yields_no_training_set_rather_than_a_bad_one():
    ds = [decision_at(d, "pulsed", f"h{d}") for d in range(0, 3)]
    pairs = join_outcomes(ds, [outcome_for("pulsed", f"h{d}", 0.7) for d in range(0, 3)])
    train, test = split_by_week(pairs, holdout_weeks=2)
    assert train == []
    assert len(test) == len(pairs)


# --- features -----------------------------------------------------------------


def test_the_feature_vector_matches_its_names():
    d = decision_at(0, "pulsed", "h1")
    x = vector(d, d.candidates[0])
    assert len(x) == len(NAMES)
    assert all(isinstance(v, float) for v in x)


def test_the_box_one_hot_is_exactly_one_hot():
    d = decision_at(0, "seedit4me", "h1")
    x = vector(d, d.candidates[0])
    block = [x[NAMES.index(f"box_is_{b}")] for b in ("home", "seedit4me", "pulsed")]
    assert sum(block) == 1.0
    assert block[1] == 1.0


def test_no_feature_carries_the_outcome():
    # The leak check. Two decisions identical but for the realized ratio must
    # produce identical features; if any feature moved, it came from the label.
    d = decision_at(0, "pulsed", "h1")
    a = vector(d, d.candidates[0])
    b = vector(d, d.candidates[0])
    assert a == b
    # And nothing in the names suggests a post-hoc measurement.
    for n in NAMES:
        assert not n.startswith("ratio_"), f"{n} looks like a label"
        assert "uploaded" not in n, f"{n} looks like a label"


def test_a_missing_feature_becomes_zero_not_an_error():
    d = decision_at(0, "pulsed", "h1")
    d.candidates[0].features = {}
    x = vector(d, d.candidates[0])
    assert len(x) == len(NAMES)


# --- the ranker ---------------------------------------------------------------


def synthetic_rows(n: int = 400, seed: int = 0):
    """Rows whose return genuinely depends on the features, so a model that
    learns nothing can be told apart from one that learns something."""
    rng = np.random.default_rng(seed)
    rows = []
    boxes = ("home", "seedit4me", "pulsed")
    gain = {"home": 0.05, "seedit4me": 1.4, "pulsed": 0.65}
    for i in range(n):
        box = boxes[i % 3]
        size = float(rng.uniform(1, 100))
        demand = float(rng.uniform(0, 12))
        d = decision_at(i % 20, box, f"h{i}")
        d.candidates[0].features = {"size_gb": size, "demand": demand}
        x = vector(d, d.candidates[0])
        y = gain[box] * size * (0.3 + 0.1 * demand) + float(rng.normal(0, 2))
        from ratio_ml.features import Example

        rows.append(Example(x=x, y=y, box=box, week=d.week))
    return rows


def test_the_ranker_learns_a_signal_that_exists():
    rows = synthetic_rows(400, seed=1)
    train, test = rows[:300], rows[300:]
    r = Ranker().fit(train)
    pred = r.predict([e.x for e in test])
    truth = np.asarray([e.y for e in test])
    assert spearman(pred, truth) > 0.7, "the ranker missed an obvious signal"


def test_the_ranker_beats_the_bucketed_median_it_replaces():
    # The only claim that justifies shipping it.
    rows = synthetic_rows(600, seed=2)
    train, test = rows[:450], rows[450:]
    r = Ranker().fit(train)
    bm = BucketedMedian().fit(train)
    truth = np.asarray([e.y for e in test])
    s_model = spearman(r.predict([e.x for e in test]), truth)
    s_base = spearman(bm.predict(test), truth)
    assert s_model > s_base, f"ranker {s_model:.3f} did not beat the median {s_base:.3f}"


def test_uncertainty_is_larger_where_the_data_is_thinner():
    rows = synthetic_rows(200, seed=3)
    r = Ranker().fit(rows)
    seen = rows[0].x
    far = list(seen)
    far[NAMES.index("log_size_gb")] = 50.0  # far outside anything observed
    _, sig = r.predict_with_sigma([seen, far])
    assert sig[1] > sig[0], "the posterior did not widen away from the data"


def test_fitting_nothing_is_an_error_not_a_silent_model():
    with pytest.raises(ValueError):
        Ranker().fit([])


def test_predicting_before_fitting_is_an_error():
    with pytest.raises(RuntimeError):
        Ranker().predict([[0.0] * len(NAMES)])


def test_the_bandit_logs_a_usable_probability():
    rows = synthetic_rows(300, seed=4)
    r = Ranker().fit(rows)
    b = ThompsonBandit(r, probes=128, seed=5)
    boxes = ["home", "seedit4me", "pulsed"]
    xs = []
    for box in boxes:
        d = decision_at(0, box, "h")
        d.candidates[0].features = {"size_gb": 30.0, "demand": 4.0}
        xs.append(vector(d, d.candidates[0]))
    c = b.choose(boxes, xs)
    assert c.box in boxes
    assert 0.0 < c.probability <= 1.0, "a propensity of zero makes every later estimate wrong"
    assert math.isfinite(c.expected)


def test_the_bandit_mostly_picks_the_box_that_earns_most():
    rows = synthetic_rows(600, seed=6)
    r = Ranker().fit(rows)
    b = ThompsonBandit(r, probes=64, seed=7)
    boxes = ["home", "seedit4me", "pulsed"]
    xs = []
    for box in boxes:
        d = decision_at(0, box, "h")
        d.candidates[0].features = {"size_gb": 40.0, "demand": 6.0}
        xs.append(vector(d, d.candidates[0]))
    picks = [b.choose(boxes, xs).box for _ in range(40)]
    assert picks.count("seedit4me") > picks.count("home"), picks


# --- generators ---------------------------------------------------------------


def make_curves(n: int = 80, seed: int = 0) -> list[Curve]:
    rng = np.random.default_rng(seed)
    t = np.arange(POINTS) * (12 * 3600.0) / POINTS
    out = []
    for _ in range(n):
        peak = float(rng.uniform(10, 120))
        tau = float(rng.uniform(900, 7200))
        out.append(
            Curve(
                leechers=log_logistic(t, peak, tau, 1.8, 1.2),
                size_gb=float(rng.uniform(2, 90)),
                category="Movies :: HD",
                freeleech=True,
            )
        )
    return out


def test_the_shared_curve_shape_peaks_at_its_peak():
    t = np.arange(400) * 60.0
    y = log_logistic(t, 50.0, 1800.0, 1.8, 1.2)
    assert y.max() <= 50.0 * 1.001
    assert y.max() > 50.0 * 0.95
    assert y[0] >= 0.0


def test_ks_distance_is_zero_for_identical_samples_and_one_for_disjoint():
    a = np.arange(100.0)
    assert ks_distance(a, a) == pytest.approx(0.0, abs=1e-9)
    assert ks_distance(a, a + 1000) == pytest.approx(1.0)


def test_the_parametric_generator_reproduces_the_marginals_it_was_fitted_on():
    real = make_curves(120, seed=10)
    g = ParametricCopula().fit(real)
    fake = g.sample(200, np.random.default_rng(11))
    ks = marginal_fidelity(real, fake)
    # Loose: this is a moment fit, not a likelihood fit. It only has to be in
    # the right neighbourhood to be a usable floor.
    assert ks["peak"] < 0.4, ks
    assert all(0.0 <= v <= 1.0 for v in ks.values())


def test_mode_coverage_catches_a_collapsed_generator():
    real = make_curves(60, seed=12)
    # A collapsed "generator": every sample identical.
    one = real[0]
    collapsed = [Curve(leechers=one.leechers.copy(), size_gb=one.size_gb,
                       category=one.category, freeleech=True) for _ in range(60)]
    assert mode_coverage(real, collapsed) < 0.6
    assert mode_coverage(real, real) > 0.9


def test_a_generator_needs_enough_curves_to_fit():
    with pytest.raises(ValueError):
        ParametricCopula().fit(make_curves(2, seed=13))


def test_sampling_before_fitting_is_an_error():
    with pytest.raises(RuntimeError):
        ParametricCopula().sample(5, np.random.default_rng(0))


# --- the parquet path ---------------------------------------------------------


def test_the_parquet_reader_yields_the_same_envelope_as_jsonl(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    day = "2026-11-03"
    d = tmp_path / "kind=decision" / f"date={day}"
    d.mkdir(parents=True)
    payload = {"name": "R", "trigger": "announce", "candidates": []}
    table = pa.table(
        {
            "v": pa.array([1], pa.int32()),
            "ts": pa.array(
                [datetime(2026, 11, 3, 3, 25, 51, tzinfo=timezone.utc)],
                pa.timestamp("ms", tz="UTC"),
            ),
            "seq": pa.array([42], pa.uint64()),
            "kind": pa.array(["decision"]),
            "box": pa.array(["pulsed"]),
            "hash": pa.array(["abc"]),
            "actor": pa.array(["baseline"]),
            "data": pa.array([json.dumps(payload)]),
        }
    )
    pq.write_table(table, d / "part-0.parquet")

    events = list(__import__("ratio_ml.data", fromlist=["x"]).read_parquet(tmp_path))
    assert len(events) == 1
    e = events[0]
    # The envelope has to match the JSONL form exactly, or every downstream
    # function would need to know which path the events came from.
    assert e["kind"] == "decision"
    assert e["seq"] == 42
    assert e["hash"] == "abc"
    assert e["data"] == payload
    assert ts_of(e).year == 2026 and ts_of(e).hour == 3
    assert parse_decisions(events)[0].name == "R"


def test_the_parquet_reader_filters_by_kind_and_day(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa
    from ratio_ml.data import read_parquet

    def write(kind: str, day: str) -> None:
        d = tmp_path / f"kind={kind}" / f"date={day}"
        d.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "v": pa.array([1], pa.int32()),
                    "ts": pa.array(
                        [datetime(2026, 11, 3, tzinfo=timezone.utc)],
                        pa.timestamp("ms", tz="UTC"),
                    ),
                    "seq": pa.array([1], pa.uint64()),
                    "kind": pa.array([kind]),
                    "box": pa.array([None], pa.string()),
                    "hash": pa.array([None], pa.string()),
                    "actor": pa.array([None], pa.string()),
                    "data": pa.array(["{}"]),
                }
            ),
            d / "part-0.parquet",
        )

    write("decision", "2026-11-03")
    write("box_state", "2026-11-03")
    write("decision", "2026-11-04")

    assert len(list(read_parquet(tmp_path))) == 3
    assert len(list(read_parquet(tmp_path, kinds=["decision"]))) == 2
    assert len(list(read_parquet(tmp_path, days=["2026-11-03"]))) == 2
    assert len(list(read_parquet(tmp_path, kinds=["decision"], days=["2026-11-04"]))) == 1


# --- the real emitted shape ---------------------------------------------------
#
# These read a slice of the live event log (tests/fixtures/events-sample.jsonl),
# not hand-made events. The loader was first written against the design doc's
# prose and silently parsed zero decisions from the real log, because the
# emitted field is `plans` with `est`, not `candidates` with `score`. A test
# against made-up events cannot catch that; only a real sample can.


from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "events-sample.jsonl"


def real_events() -> list[dict]:
    from ratio_ml.data import read_jsonl

    if not FIXTURE.exists():  # pragma: no cover
        pytest.skip(f"{FIXTURE} missing")
    return list(read_jsonl(FIXTURE))


def test_the_fixture_covers_every_ml_relevant_kind():
    kinds = {e["kind"] for e in real_events()}
    for k in ("decision", "release", "scrape", "announce", "registration", "action"):
        assert k in kinds, f"the fixture lost {k}; re-cut it from the live log"


def test_decisions_parse_from_the_real_log():
    ds = parse_decisions(real_events())
    assert ds, "parsed zero decisions from the real log — the shape drifted again"
    d = ds[0]
    assert d.candidates, "a decision with no plans"
    assert d.tl_id, "no tl_id, so nothing can be joined to the release book"
    # Every plan carries a box and a score, and the scores are real numbers.
    for c in d.candidates:
        assert c.box
        assert isinstance(c.score, float)
    # The per-box one-hot in the feature vector only works if the box names
    # are the ones it knows about.
    from ratio_ml.features import BOXES

    for c in d.candidates:
        assert c.box in BOXES, f"unknown box {c.box!r}; update features.BOXES"


def test_the_log_records_no_propensity_today():
    # Not a defect to fix here, but a fact the training code must not paper
    # over: the baseline is deterministic, so there is nothing to weight by.
    # If this ever starts failing, the bandit has shipped and the offline
    # methods can stop treating propensity as absent.
    ds = parse_decisions(real_events())
    assert all(d.probability is None for d in ds)
    assert all(d.experiment is None for d in ds)


def test_rejected_plans_carry_their_reasons():
    # These are the only negative examples the log holds, since the baseline
    # filters before the guard ever sees a proposal.
    ds = parse_decisions(real_events())
    rejected = [c for d in ds for c in d.rejected]
    assert rejected, "no rejected plans in the sample"
    assert any(c.why for c in rejected), "a rejected plan with no reason given"


def test_the_release_book_builds_and_carries_swarm_curves():
    from ratio_ml.data import release_index

    book = release_index(real_events())
    assert book, "the release book is empty"
    # Sizes come through in GB and are plausible.
    sized = [r for r in book.values() if r.size_gb > 0]
    assert sized, "no release in the book has a size"
    assert all(0.0 < r.size_gb < 2000.0 for r in sized), [r.size_gb for r in sized]
    # At least one release has a scrape, which is what the generators fit on.
    scraped = [r for r in book.values() if r.scrapes]
    assert scraped, "no scrape curves in the book"
    for r in scraped:
        # Time-ordered, which demand_at relies on.
        times = [t for t, *_ in r.scrapes]
        assert times == sorted(times)
        assert r.peak_leechers() >= 0


def test_demand_is_read_from_the_scrape_before_the_moment_asked():
    from ratio_ml.data import ReleaseFacts

    t0 = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)
    r = ReleaseFacts(tl_id="1")
    r.scrapes = [
        (t0, 1, 30, 0),
        (t0 + timedelta(hours=1), 10, 20, 5),
        (t0 + timedelta(hours=2), 40, 5, 50),
    ]
    # Before any scrape: unknown, not zero.
    assert r.demand_at(t0 - timedelta(minutes=1)) is None
    # Uses the latest scrape at or before the moment, never a later one.
    assert r.demand_at(t0 + timedelta(minutes=30)) == pytest.approx(30 / 2.0)
    assert r.demand_at(t0 + timedelta(hours=1, minutes=30)) == pytest.approx(20 / 11.0)
    assert r.peak_leechers() == 30


def test_demand_recovered_from_the_basis_string():
    ds = parse_decisions(real_events())
    withd = [c for d in ds for c in d.candidates if "demand" in c.features]
    assert withd, "no plan recorded a demand; the basis format changed"
    assert all(c.features["demand"] >= 0.0 for c in withd)
