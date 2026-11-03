//! `ratio-sim`: an event-driven simulator of TorrentLeech, our boxes and the
//! swarms, for training and testing ratio-manager policies offline.
//!
//! Wave 4 of `ServerNotes/designs/ratio-manager.md`; the plan is in
//! `ServerNotes/designs/ratio-ml.md`.
//!
//! Three models — a [tracker](tracker), a [box](boxes) and a [swarm](swarm) —
//! driven by an [environment](env) that replays real releases from
//! [traces](traces), with the calibration in [params]. [policies] holds the
//! references a learned policy has to beat.
//!
//! What this is for, and its one hard rule: **the simulator is not evidence
//! until it reproduces the fleet's measured numbers on held-out races.** Until
//! then it is a source of confident nonsense, and a policy trained against it
//! is worse than no policy. [`validate`] is that gate.

pub mod boxes;
pub mod env;
pub mod link;
pub mod params;
pub mod policies;
pub mod swarm;
pub mod tracker;
pub mod traces;
pub mod validate;

pub use env::{Act, Policy, Release, Reward, Sim};
pub use params::Params;
