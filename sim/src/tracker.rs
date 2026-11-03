//! TorrentLeech as a set of measured distributions.
//!
//! Three behaviours matter to a policy, and all three were learned the hard
//! way by `ratio-picker.py`:
//!
//!  - A grab does not start downloading when it is added. It starts when the
//!    tracker registers it, which took a median of 94 s and sometimes 40
//!    minutes. A policy that assumes instant starts mis-prices every race.
//!  - Some releases are never registered at all. They sit at 0%, holding the
//!    in-flight reserve, until something deletes them.
//!  - The listing's seeder and leecher counts lag the live swarm by minutes,
//!    so a release that looks uncontested is often already crowded.

use crate::params::TrackerParams;
use rand::Rng;
use rand_distr::{Distribution, Exp, LogNormal};

/// What the tracker decided about one grab.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Registration {
    /// Registered this many seconds after the add.
    After(f64),
    /// Never registered. The release was deleted after the announce.
    Never,
}

impl Registration {
    pub fn seconds(&self) -> Option<f64> {
        match self {
            Registration::After(s) => Some(*s),
            Registration::Never => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tracker {
    fast_share: f64,
    fast_mu: f64,
    fast_sigma: f64,
    slow_mean_s: f64,
    never_share: f64,
    pub listing_lag_s: f64,
    pub announce_interval_s: f64,
}

impl Tracker {
    pub fn sample<R: Rng + ?Sized>(p: &TrackerParams, rng: &mut R) -> Self {
        Self {
            fast_share: p.fast_share.sample(rng),
            fast_mu: p.fast_mu.sample(rng),
            fast_sigma: p.fast_sigma.sample(rng),
            slow_mean_s: p.slow_mean_s.sample(rng),
            never_share: p.never_share.sample(rng),
            listing_lag_s: p.listing_lag_s.sample(rng),
            announce_interval_s: p.announce_interval_s.sample(rng),
        }
    }

    /// Draw a registration outcome for one grab.
    ///
    /// A mixture, because one distribution does not fit: most grabs register
    /// quickly and lognormally, and a fifth go through moderation, of which
    /// some never come back. A single lognormal fitted to the measured median
    /// and p90 puts only 6% past six minutes, where 22% were measured — the
    /// tail is a different process, not a longer draw from the same one.
    pub fn register<R: Rng + ?Sized>(&self, rng: &mut R) -> Registration {
        if rng.gen_bool(self.fast_share.clamp(0.0, 1.0)) {
            let d = LogNormal::new(self.fast_mu, self.fast_sigma).expect("valid lognormal");
            return Registration::After(d.sample(rng));
        }
        if rng.gen_bool(self.never_share.clamp(0.0, 1.0)) {
            return Registration::Never;
        }
        let d = Exp::new(1.0 / self.slow_mean_s.max(1.0)).expect("valid exponential");
        // Moderation does not resolve faster than the fast path would have.
        Registration::After(d.sample(rng) + 120.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::TrackerParams;
    use rand::SeedableRng;

    fn draws(n: usize, seed: u64) -> Vec<Registration> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let t = Tracker::sample(&TrackerParams::default(), &mut rng);
        (0..n).map(|_| t.register(&mut rng)).collect()
    }

    fn quantile(mut v: Vec<f64>, q: f64) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() as f64 - 1.0) * q).round() as usize]
    }

    #[test]
    fn the_registration_mixture_matches_the_measured_baseline() {
        // 87 grabs measured p50 94 s, and 22% still unregistered at ~6 min.
        // The mixture is only required to land in the right neighbourhood; the
        // real fit comes off the event log.
        let d = draws(20_000, 11);
        let secs: Vec<f64> = d.iter().filter_map(|r| r.seconds()).collect();
        let p50 = quantile(secs.clone(), 0.5);
        assert!((60.0..160.0).contains(&p50), "p50 {p50} s");

        let past_six_min = d
            .iter()
            .filter(|r| match r {
                Registration::Never => true,
                Registration::After(s) => *s > 360.0,
            })
            .count() as f64
            / d.len() as f64;
        assert!((0.10..0.35).contains(&past_six_min), "{past_six_min} past 6 min");
    }

    #[test]
    fn some_grabs_never_register() {
        let d = draws(5_000, 12);
        let never = d.iter().filter(|r| **r == Registration::Never).count();
        assert!(never > 0, "no grab was ever deleted — the 0% trap is unmodelled");
        assert!((never as f64 / d.len() as f64) < 0.3, "far too many deletions");
    }

    #[test]
    fn registration_is_never_instant() {
        for r in draws(2_000, 13) {
            if let Registration::After(s) = r {
                assert!(s > 0.0, "registered at or before the add");
            }
        }
    }
}
