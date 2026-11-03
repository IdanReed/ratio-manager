# ratio-ml

Training and evaluation for the ratio manager's learned policy. Wave 4 of
`ServerNotes/designs/ratio-manager.md`; the plan, the decisions and the open
questions are in `ServerNotes/designs/ratio-ml.md`.

```bash
# the parts that need only numpy — these run on the services VM
pip install -e '.[dev]'
pytest -q

# the learned generators and the decision transformer
pip install -e '.[torch,dev]'
```

## What the model is replacing

The baseline's `estimate()` (`src/policy.rs`) is a lookup table:
`(box, source, size bucket, demand bucket)` → median ratio of past races in
that cell, seeded with priors, times the box's bias. Four bucketed features
and a median.

That sets the bar. `BucketedMedian` in `ranker.py` reimplements it so it can be
scored on the same split, and `test_the_ranker_beats_the_bucketed_median_it_replaces`
is the claim that justifies shipping anything at all. A model that cannot beat
the table on held-out races is not an improvement, however interesting.

## The order, and why it is not the obvious one

| Step | Module | Needs |
|---|---|---|
| 1 | `data`, `features` | only what the log already holds |
| 2 | `ranker` — expected-return model, in shadow | the same |
| 3 | the Rust simulator (`../sim`) and its calibration | the event log |
| 4 | `ranker.ThompsonBandit` — exploration, live | step 2 shipped |
| 5 | `generators` — the bake-off | a few hundred scrape curves |
| 6 | `dt` — decision transformer, and IQL, in simulation | step 4's data |

The decision transformer is the operator's preferred route and it comes last
on purpose. It is return-conditioned imitation: it can only reproduce returns
the behaviour policy sometimes achieved. Today's behaviour policy is a bucketed
median, so a DT trained on today's log would learn to imitate a bucketed
median. The bandit at step 4 is what makes step 6 worth doing, because it
produces decisions with **logged probabilities** and real diversity.

## The two ways to build a model that looks great and is useless

Both are guarded by tests, because both are easy and neither announces itself.

**Leaking the outcome into the features.** `features.py` reads only what the
manager knew at decision time. `test_no_feature_carries_the_outcome` checks
that nothing named like a measurement gets in, and that two calls on the same
decision agree.

**Splitting the data at random.** A release raced on two boxes minutes apart is
one event; a random split puts one copy in train and the other in test, and
every model then looks clairvoyant. `split_by_week` splits chronologically by
ISO week, and `test_the_split_never_puts_one_week_on_both_sides` holds it to
that.

A third, smaller trap: a decision whose reward horizon has not elapsed is
**dropped, not zero-filled**. Zero-filling teaches the model that everything
recent was worthless.

## Generators, ranked

The scarce thing is swarm-growth curves conditioned on release attributes, and
the tail where a release blooms. Races arrive a few dozen a day.

1. **`ParametricCopula`** — log-logistic curves with a Gaussian copula over
   their parameters. Works at a few hundred traces, which is the regime that
   exists. The floor to beat, and possibly enough. numpy only.
2. **`AutoregressiveCurves`** — a GRU over quantized deltas, conditioned on
   category, size and freeleech. Teacher-forced, so it has a likelihood and can
   be model-selected rather than eyeballed. The recommended learned family.
3. **`DiffusionCurves`** — good samples, clean conditioning, no likelihood,
   more machinery than the problem needs.
4. **`CurveGAN`** — included because it was asked about, kept as a bake-off arm
   rather than the plan. On a few thousand short traces, mode collapse is the
   default outcome and "the samples look right" is not a test. Watch
   `mode_coverage`: a collapsed generator can post respectable marginals while
   covering a fraction of the real variety.

`evaluate_generators` scores all four the same way and ranks by **TSTR** — train
the policy on synthetic curves, evaluate it in the simulator calibrated on real
held-out data. Fidelity metrics are diagnostics; only transfer decides. The
ranking never uses marginals alone, which is precisely how a collapsed
generator wins a bake-off.

## Where this runs

`data`, `features` and `ranker` are numpy-only by design: a model that cannot
be retrained on the services VM is a model that will not get retrained. The
learned generators and `dt` need torch and are meant for the desktop's GPU —
which is open question 1 in the design doc, since the dataset lives on the VM
and the desktop is not always up.
