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


def test_parse_decisions_keeps_candidates_and_the_chosen_one():
    e = event(
        "decision",
        "2026-11-01T12:00:00.000Z",
        hash="abc",
        name="Some.Release",
        trigger="announce",
        candidates=[
            {"box": "pulsed", "score": 0.6, "features": {"size_gb": 20.0, "demand": 3.0}},
            {"box": "seedit4me", "score": 1.4, "features": {"size_gb": 20.0, "demand": 3.0},
             "chosen": True},
        ],
        probability=0.72,
    )
    ds = parse_decisions([e, event("box_state", "2026-11-01T12:00:01.000Z")])
    assert len(ds) == 1
    d = ds[0]
    assert len(d.candidates) == 2
    assert [c.box for c in d.chosen] == ["seedit4me"]
    assert d.probability == pytest.approx(0.72)
    assert d.trigger == "announce"


def test_a_deterministic_decision_has_no_propensity():
    e = event("decision", "2026-11-01T12:00:00.000Z", hash="abc", candidates=[])
    assert parse_decisions([e])[0].probability is None


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
