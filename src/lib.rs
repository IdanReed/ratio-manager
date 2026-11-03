//! ratio-manager as a library.
//!
//! The binary (`main.rs`) wires these modules into the running service. They
//! are public so the simulator can link the real thing rather than a copy of
//! it — above all `guard`, whose rules a policy must not be able to learn
//! around, and `policy`, whose baseline is the bar a learned policy has to
//! beat. See `ServerNotes/designs/ratio-ml.md`.

pub mod actions;
pub mod api;
pub mod autobrr;
pub mod bencode;
pub mod config;
pub mod drivers;
pub mod events;
pub mod export;
pub mod guard;
pub mod policy;
pub mod qbit;
pub mod scout;
pub mod state;
pub mod store;
pub mod tl;
