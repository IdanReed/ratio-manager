# ratio-sim

An event-driven simulator of TorrentLeech, our boxes and the swarms, for
training and testing ratio-manager policies offline. Wave 4 of
`ServerNotes/designs/ratio-manager.md`; the plan is `ratio-ml.md`.

```bash
cargo run -p ratio-sim --bin rollout -- --help
cargo run -p ratio-sim --bin rollout                         # synthetic releases
cargo run -p ratio-sim --bin rollout -- --events /slow/ratio-manager/events/2026-11-03.jsonl.zst
cargo test -p ratio-sim
```

## The one rule

**The simulator is not evidence until it reproduces the fleet's measured
numbers on held-out races.** `validate.rs` is that gate: each box's median race
ratio has to land inside the interval the fleet actually produced (seedit4me
1.1–1.5, pulsed 0.4–0.9, home 0.02–0.15). A rollout that raced too little to
judge **fails** rather than passes, because a simulator nobody checked is a
source of confident nonsense, and a policy trained against one is worse than no
policy.

## What it does and does not model

| | |
|---|---|
| Models | registration delay and tracker deletions, per-box line and disk contention, tunnel starvation, traffic allowances, aggregate swarm growth, our share of a swarm's demand, several of our boxes splitting one swarm |
| Does not model | individual peers, pieces, piece selection, encryption overhead, tracker rate limits, IRC timing |

It **replays real releases** rather than inventing demand: sizes, categories,
freeleech flags and scrape curves come from the event log. The simulator only
decides what our boxes do with them.

## It links the real guard

`link.rs` builds a headless `ratio_manager::state::App` — in-memory store, no
network — and calls the service's own `guard::check_add` and
`guard::may_remove`. The simulator owns the physics; **the service owns every
rule about what is allowed**. A policy trained against a re-implemented guard
would learn to propose actions the real guard rejects, and each such proposal
is a wasted decision in production.

The same reasoning applies one level up: the simulator is meant to serve the
production `/api/v1` shapes, so a policy is one client program that cannot tell
which it is attached to. That is the part worth protecting as this grows.

## The models

- `tracker.rs` — a mixture, because one distribution does not fit. Most grabs
  register quickly and lognormally (measured p50 94 s); about a fifth go
  through moderation and some of those are never registered at all. A single
  lognormal fitted to the measured median and p90 puts 6% past six minutes
  where 22% were measured: the tail is a different process.
- `boxes.rs` — a fluid model. Writes mostly proceed and **reads** starve, which
  is why pulsed kept downloading at 110–200 MiB/s while its uploads fell under
  1 MiB/s. Degradation is driven by the write rate the box is *attempting*; if
  it were driven by the achieved rate it would undo itself.
- `swarm.rs` — aggregate flows. A unimodal leecher curve, a competitor count
  that grows over the first half of the race, and our share of demand as our
  serving capacity over everyone's. Joining early pays **cumulatively**, not in
  the first instant.
- `params.rs` — the calibration, every default carrying its provenance. Each
  value has an interval and is resampled per episode, so a policy cannot come
  to depend on a point estimate that was one week's artefact.

## Reproducibility

Everything is seeded. The same seed gives the same rollout; a rollout records
which calibration artefact produced it.
