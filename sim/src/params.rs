//! The simulator's calibration: every number the tracker, box and swarm
//! models need, in one serializable struct.
//!
//! The defaults here are the measurements taken off the live fleet while
//! `ratio-picker.py` ran, and each carries where it came from. They are a
//! starting point, not the calibration: `calibrate.rs` refits these from the
//! event log's Parquet export and writes `sim/params/<date>.json`. A rollout
//! records which parameter set produced it.
//!
//! Every field has a `_lo`/`_hi` interval or a spread where one was measured.
//! `Params::sample` draws a per-episode variant inside those intervals, so a
//! policy cannot come to depend on a point estimate that was only ever an
//! artefact of one week's measurements. See the "Don't let the policy exploit
//! it" rule in ServerNotes/designs/ratio-ml.md.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// A value measured with an interval. `sample` draws uniformly inside it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Range {
    pub lo: f64,
    pub hi: f64,
}

impl Range {
    pub const fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }
    /// A value known exactly (a configured cap, not a measurement).
    pub const fn exact(v: f64) -> Self {
        Self { lo: v, hi: v }
    }
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        if self.hi <= self.lo {
            self.lo
        } else {
            rng.gen_range(self.lo..=self.hi)
        }
    }
    pub fn mid(&self) -> f64 {
        (self.lo + self.hi) / 2.0
    }
}

/// How TorrentLeech behaves: when a torrent becomes announceable, whether it
/// survives moderation, and how stale the listing's counts are.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackerParams {
    /// Share of grabs that register on the fast path.
    ///
    /// Measured over 87 grabs from VictoriaLogs: 68 of 87 (78%) were seen
    /// registered within ~6 minutes; 19 (22%) were not.
    pub fast_share: Range,
    /// Fast path, lognormal in seconds. `mu = ln(median)`.
    ///
    /// The 68 fast grabs ran p50 94 s, p75 153 s, p90 279 s, max 316 s.
    pub fast_mu: Range,
    pub fast_sigma: Range,
    /// Slow path: the release is awaiting moderation. Exponential, in seconds.
    ///
    /// Observed registering 20–40 minutes after the announce.
    pub slow_mean_s: Range,
    /// Of the slow path, the share TorrentLeech never registers at all — the
    /// release is deleted minutes after the IRC announce and the grab sits at
    /// 0% forever ("unregistered torrent").
    pub never_share: Range,
    /// How far behind the live swarm the listing's seeder/leecher counts run.
    ///
    /// Picks listed at S=1, L=30–45 had S=17–60 by the time they landed.
    pub listing_lag_s: Range,
    /// The tracker's announce interval, seconds.
    pub announce_interval_s: Range,
}

impl Default for TrackerParams {
    fn default() -> Self {
        Self {
            fast_share: Range::new(0.72, 0.84),
            // ln(94) = 4.543. Widened a little either side of the median.
            fast_mu: Range::new(4.40, 4.70),
            // Fitted to p90/p50 = 2.97 with the heavy tail split off into the
            // slow path, which leaves the fast path tighter than the raw fit.
            fast_sigma: Range::new(0.50, 0.75),
            slow_mean_s: Range::new(1200.0, 2400.0),
            never_share: Range::new(0.20, 0.40),
            listing_lag_s: Range::new(120.0, 600.0),
            announce_interval_s: Range::exact(1800.0),
        }
    }
}

/// One box's physics. Rates are MiB/s throughout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoxParams {
    pub name: String,
    /// Raw line capacity.
    pub line_down: Range,
    pub line_up: Range,
    /// Total disk throughput, reads and writes together.
    pub disk_total: Range,
    /// Write rate past which the disk's service time degrades. Below the knee
    /// the queue stays short; above it, both directions suffer.
    pub disk_knee: Range,
    /// How sharply service degrades past the knee. 1.0 is proportional; higher
    /// collapses faster.
    pub disk_decay: Range,
    /// Download past which a single WireGuard tunnel starts dropping packets,
    /// which starves this box's own uploads. `None` for a box not behind one.
    pub tunnel_knee: Option<Range>,
    /// Upload lost per MiB/s of download past the tunnel knee, as a fraction.
    pub tunnel_starve: Range,
    /// Share of the disk that neighbours on a shared host are already using.
    /// Redrawn through an episode, not fixed.
    pub neighbour_load: Range,
    /// Traffic allowance per month in GB, and the rate the host throttles to
    /// once it is spent.
    pub quota_gb: Option<Range>,
    pub throttle_mbit: Range,
    /// Usable disk, GB.
    pub disk_gb: Range,
    /// What the box costs a month, for the reward's rent term.
    pub cost_eur_month: Range,
}

impl BoxParams {
    /// seedit4me: the strong racer. Holds ~100 MiB/s of download while still
    /// uploading; 5 TB/month, then throttled to 10 Mbit/s.
    pub fn seedit4me() -> Self {
        Self {
            name: "seedit4me".into(),
            line_down: Range::new(100.0, 130.0),
            line_up: Range::new(100.0, 120.0),
            disk_total: Range::new(200.0, 320.0),
            disk_knee: Range::new(150.0, 250.0),
            disk_decay: Range::new(0.8, 1.5),
            tunnel_knee: None,
            tunnel_starve: Range::exact(0.0),
            neighbour_load: Range::new(0.0, 0.15),
            quota_gb: Some(Range::exact(5000.0)),
            throttle_mbit: Range::exact(10.0),
            disk_gb: Range::exact(1650.0),
            cost_eur_month: Range::exact(30.0),
        }
    }

    /// pulsed: a shared PMSS host. Its disk only stays healthy near 20 MiB/s of
    /// writes — at 70 or 130 the I/O queue ran 100–1100 jobs and upload fell to
    /// 1–10 MiB/s. Unmetered traffic.
    pub fn pulsed() -> Self {
        Self {
            name: "pulsed".into(),
            line_down: Range::new(400.0, 500.0),
            line_up: Range::new(400.0, 500.0),
            // ~150 MB/s sustained writes on an idle host; neighbours take much
            // of it, which is what `neighbour_load` carries.
            disk_total: Range::new(120.0, 170.0),
            disk_knee: Range::new(18.0, 30.0),
            // Sharp: the measured collapse from 21 MiB/s of upload at a 20
            // MiB/s cap to 1–10 MiB/s at 70.
            disk_decay: Range::new(1.8, 3.0),
            tunnel_knee: None,
            tunnel_starve: Range::exact(0.0),
            // Host PSI io-full ran 47–73% from neighbours alone.
            neighbour_load: Range::new(0.35, 0.75),
            quota_gb: None,
            throttle_mbit: Range::exact(10.0),
            disk_gb: Range::exact(4500.0),
            cost_eur_month: Range::exact(12.0),
        }
    }

    /// home: the media stack's qBittorrent inside the AirVPN tunnel. ~545
    /// Mbit/s down and 130–190 Mbit/s up through Chicago; above ~45–60 MiB/s of
    /// download the tunnel loses ~7% and upload falls to 1–4 MiB/s.
    pub fn home() -> Self {
        Self {
            name: "home".into(),
            line_down: Range::new(60.0, 70.0),
            line_up: Range::new(16.0, 24.0),
            disk_total: Range::new(60.0, 120.0),
            disk_knee: Range::new(40.0, 80.0),
            disk_decay: Range::new(0.8, 1.5),
            tunnel_knee: Some(Range::new(35.0, 50.0)),
            tunnel_starve: Range::new(0.04, 0.09),
            // The VM itself sits near 41% iowait from bitmagnet and ClamAV.
            neighbour_load: Range::new(0.25, 0.55),
            quota_gb: None,
            throttle_mbit: Range::exact(10.0),
            disk_gb: Range::exact(62000.0),
            cost_eur_month: Range::exact(0.0),
        }
    }
}

/// How a swarm grows and how much of its demand we win.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmParams {
    /// Leechers at the peak, as a multiple of the release's listed leechers.
    pub peak_mult: Range,
    /// Seconds from announce to the leecher peak.
    pub time_to_peak_s: Range,
    /// Rise and decay exponents of the unimodal leecher curve.
    pub rise: Range,
    pub decay: Range,
    /// Competing seedboxes that race the same release from IRC. Our own boxes
    /// are counted separately; these are everyone else's.
    pub competitors: Range,
    /// A competitor's serving capacity, MiB/s.
    pub competitor_up: Range,
    /// What one leecher will pull, MiB/s. Caps total demand.
    pub leecher_down: Range,
    /// Progress below which a partial copy serves nothing. Above it a partial
    /// seed competes nearly as well as a complete one, because early in a race
    /// the pieces it holds are the rare ones — PS5 and game packages reached
    /// ratio 1.6–2.3 at 3% downloaded.
    pub min_serve_progress: Range,
    /// How much of its line a partial copy can actually place, as a fraction.
    pub partial_serve: Range,
}

impl Default for SwarmParams {
    fn default() -> Self {
        Self {
            peak_mult: Range::new(1.2, 4.0),
            time_to_peak_s: Range::new(900.0, 5400.0),
            rise: Range::new(1.2, 2.5),
            decay: Range::new(0.8, 2.0),
            competitors: Range::new(3.0, 25.0),
            competitor_up: Range::new(10.0, 90.0),
            leecher_down: Range::new(2.0, 12.0),
            min_serve_progress: Range::new(0.005, 0.03),
            partial_serve: Range::new(0.6, 1.0),
        }
    }
}

/// What an action is worth. Credited upload, less what it consumed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardParams {
    /// Price of a GB uploaded. The unit the reward is denominated in; 1.0
    /// means the reward is GB.
    pub gb_up: f64,
    /// Charge per GB of download that counts against a metered allowance.
    pub gb_quota: f64,
    /// Charge per GB-day a torrent occupies disk.
    pub gb_day_disk: f64,
    /// Whether a box's monthly rent is prorated into the episode's reward.
    pub charge_rent: bool,
}

impl Default for RewardParams {
    fn default() -> Self {
        Self {
            gb_up: 1.0,
            // Quota traffic on seedit4me is the scarce input while its 5 TB
            // lasts: a GB spent downloading is a GB not available to upload.
            gb_quota: 0.35,
            gb_day_disk: 0.02,
            charge_rent: true,
        }
    }
}

/// A whole calibration. `version` is the artefact's identity; a rollout
/// records it so a result can be traced to the numbers that produced it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Params {
    pub version: String,
    pub tracker: TrackerParams,
    pub boxes: Vec<BoxParams>,
    pub swarm: SwarmParams,
    pub reward: RewardParams,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            version: "fleet-measurements-2026-11".into(),
            tracker: TrackerParams::default(),
            boxes: vec![BoxParams::seedit4me(), BoxParams::pulsed(), BoxParams::home()],
            swarm: SwarmParams::default(),
            reward: RewardParams::default(),
        }
    }
}

impl Params {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&s)?)
    }

    pub fn box_params(&self, name: &str) -> Option<&BoxParams> {
        self.boxes.iter().find(|b| b.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn a_range_samples_inside_itself() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let r = Range::new(2.0, 5.0);
        for _ in 0..1000 {
            let v = r.sample(&mut rng);
            assert!((2.0..=5.0).contains(&v), "{v} outside the range");
        }
    }

    #[test]
    fn an_exact_range_is_the_value() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        assert_eq!(Range::exact(7.5).sample(&mut rng), 7.5);
    }

    #[test]
    fn the_defaults_round_trip_as_json() {
        let p = Params::default();
        let s = serde_json::to_string(&p).unwrap();
        let back: Params = serde_json::from_str(&s).unwrap();
        assert_eq!(back.version, p.version);
        assert_eq!(back.boxes.len(), 3);
        assert!(back.box_params("pulsed").is_some());
    }

    #[test]
    fn pulsed_knees_well_below_seedit4me() {
        // The whole reason the two boxes behave differently. If a refit ever
        // loses this, the simulator has stopped describing the fleet.
        let p = Params::default();
        let pulsed = p.box_params("pulsed").unwrap();
        let s4 = p.box_params("seedit4me").unwrap();
        assert!(pulsed.disk_knee.hi < s4.disk_knee.lo);
    }
}
