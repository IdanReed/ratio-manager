//! A swarm as an aggregate flow, not a set of peers.
//!
//! The simulator never models a piece. What it has to get right is how much
//! demand exists for a release over time, how many other seedboxes are
//! competing to serve it, and therefore what share of that demand one of our
//! boxes wins. Those three are what decide a race's ratio, and all three are
//! measurable from scrape curves and peer snapshots.
//!
//! The policy's lever is *when* it joins and *how fast* it downloads. Joining
//! early into a leecher-heavy, seeder-poor swarm is what earned ratio 1.6–2.3
//! at 3% downloaded; joining late into a swarm a dozen seedboxes already serve
//! earns nothing however fast the box is.

use crate::params::SwarmParams;
use rand::Rng;

/// One release's swarm over the whole episode.
#[derive(Debug, Clone)]
pub struct Swarm {
    /// Leechers at the peak.
    pub peak: f64,
    /// Seconds from announce to peak.
    pub tau: f64,
    pub rise: f64,
    pub decay: f64,
    /// Other people's seedboxes racing this release.
    pub competitors: f64,
    /// A competitor's serving capacity, MiB/s.
    pub competitor_up: f64,
    /// What one leecher will pull, MiB/s.
    pub leecher_down: f64,
    pub min_serve_progress: f64,
    pub partial_serve: f64,
}

impl Swarm {
    /// Draw a swarm for a release whose listing showed `listed_leechers`.
    ///
    /// No floor on the peak: a release the listing shows with no leechers gets
    /// no modelled demand. Flooring it at one leecher would make every release
    /// worth something, and a policy could then never learn that most are not
    /// — the live picker rejected 42 of 70 candidates for low demand in a
    /// single pass. That the listing *understates* a fresh swarm is carried by
    /// `peak_mult`, not by a floor.
    pub fn sample<R: Rng + ?Sized>(p: &SwarmParams, listed_leechers: f64, rng: &mut R) -> Self {
        Self {
            peak: listed_leechers.max(0.0) * p.peak_mult.sample(rng),
            tau: p.time_to_peak_s.sample(rng),
            rise: p.rise.sample(rng),
            decay: p.decay.sample(rng),
            competitors: p.competitors.sample(rng),
            competitor_up: p.competitor_up.sample(rng),
            leecher_down: p.leecher_down.sample(rng),
            min_serve_progress: p.min_serve_progress.sample(rng),
            partial_serve: p.partial_serve.sample(rng),
        }
    }

    /// Leechers `t` seconds after the announce.
    ///
    /// A unimodal rise-then-decay curve: `peak * x^a / (1 + x^(a+b))`, scaled
    /// so the maximum is `peak`. Zero before the announce, a steep rise while
    /// the release is fresh, a long decay as the swarm is satisfied.
    pub fn leechers(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let x = t / self.tau;
        let a = self.rise;
        let b = self.decay;
        let raw = x.powf(a) / (1.0 + x.powf(a + b));
        // The shape's own maximum, so `peak` means what it says.
        let xm = (a / b.max(1e-6)).powf(1.0 / (a + b));
        let max = xm.powf(a) / (1.0 + xm.powf(a + b));
        if max <= 0.0 {
            0.0
        } else {
            self.peak * raw / max
        }
    }

    /// Competing seeders `t` seconds after the announce: the original uploader
    /// plus the other seedboxes as they finish. They arrive over roughly the
    /// first `tau`, which is why joining late is expensive.
    pub fn competing_seeders(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        let arrived = self.competitors * (1.0 - (-t / (self.tau * 0.5).max(1.0)).exp());
        1.0 + arrived
    }

    /// Total upload the swarm's leechers will absorb, MiB/s.
    pub fn demand(&self, t: f64) -> f64 {
        self.leechers(t) * self.leecher_down
    }

    /// What one of our boxes can offer, given how complete its copy is.
    ///
    /// A copy below `min_serve_progress` offers nothing — it has no pieces
    /// anyone wants yet. Above it, a partial copy competes at
    /// `partial_serve` of its line, because early in a race the pieces it
    /// holds are the rare ones.
    pub fn serving_capacity(&self, line_up: f64, progress: f64) -> f64 {
        if progress < self.min_serve_progress {
            return 0.0;
        }
        if progress >= 1.0 {
            line_up
        } else {
            line_up * self.partial_serve
        }
    }

    /// Demand offered to one of our boxes, MiB/s.
    ///
    /// Our share of the swarm's demand is our serving capacity over the total
    /// serving capacity in the swarm — ours plus every competitor's. This is
    /// the one equation the simulator's realism rests on, and it is
    /// deliberately an aggregate: per-peer behaviour is not recoverable from
    /// the data and does not change the aggregate much.
    ///
    /// `peer_capacity` is what our *other* boxes are offering on the same
    /// release, so racing a release on two boxes splits the swarm rather than
    /// doubling the return.
    pub fn offered_to(&self, t: f64, our_capacity: f64, peer_capacity: f64) -> f64 {
        if our_capacity <= 0.0 {
            return 0.0;
        }
        let theirs = self.competing_seeders(t) * self.competitor_up;
        let total = theirs + our_capacity + peer_capacity;
        if total <= 0.0 {
            return 0.0;
        }
        self.demand(t) * (our_capacity / total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SwarmParams;
    use rand::SeedableRng;

    fn swarm(seed: u64, listed: f64) -> Swarm {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        Swarm::sample(&SwarmParams::default(), listed, &mut rng)
    }

    #[test]
    fn the_leecher_curve_rises_then_decays_and_peaks_at_peak() {
        let s = swarm(1, 40.0);
        assert_eq!(s.leechers(-10.0), 0.0);
        assert_eq!(s.leechers(0.0), 0.0);
        let at_peak = s.leechers(s.tau);
        let early = s.leechers(s.tau * 0.1);
        let late = s.leechers(s.tau * 20.0);
        assert!(early < at_peak, "{early} !< {at_peak}");
        assert!(late < at_peak, "{late} !< {at_peak}");
        // The curve's maximum over a sweep must not exceed `peak`.
        let mut best: f64 = 0.0;
        for i in 1..4000 {
            best = best.max(s.leechers(s.tau * i as f64 / 100.0));
        }
        assert!(best <= s.peak * 1.001, "max {best} over peak {}", s.peak);
        assert!(best > s.peak * 0.95, "max {best} never reached peak {}", s.peak);
    }

    #[test]
    fn competing_seeders_grow_from_the_lone_uploader() {
        let s = swarm(2, 40.0);
        assert!((s.competing_seeders(0.0) - 1.0).abs() < 1e-9);
        let early = s.competing_seeders(60.0);
        let later = s.competing_seeders(s.tau * 2.0);
        assert!(early < later, "{early} !< {later}");
        assert!(later <= 1.0 + s.competitors + 1e-6);
    }

    /// Total demand offered to a box that joins at `join` and stays to the
    /// horizon. Joining early pays over the whole race, not in the first
    /// instant — at 30 seconds the leecher curve has barely begun to rise.
    fn earned_from(s: &Swarm, join: f64, cap: f64) -> f64 {
        let horizon = s.tau * 10.0;
        let dt = horizon / 2000.0;
        let mut total = 0.0;
        let mut t = join;
        while t < horizon {
            total += s.offered_to(t, cap, 0.0) * dt;
            t += dt;
        }
        total
    }

    #[test]
    fn joining_early_beats_joining_late() {
        // The whole reason autobrr exists: at the announce there is one seeder
        // and a crowd of leechers; an hour later a dozen seedboxes serve it.
        let s = swarm(3, 40.0);
        let early = earned_from(&s, 0.0, 100.0);
        let late = earned_from(&s, s.tau * 3.0, 100.0);
        assert!(early > late, "early {early} !> late {late}");
    }

    #[test]
    fn the_cost_of_joining_late_grows_with_the_delay() {
        let s = swarm(8, 40.0);
        let at_once = earned_from(&s, 0.0, 100.0);
        let bit_late = earned_from(&s, s.tau, 100.0);
        let very_late = earned_from(&s, s.tau * 4.0, 100.0);
        assert!(at_once > bit_late, "{at_once} !> {bit_late}");
        assert!(bit_late > very_late, "{bit_late} !> {very_late}");
    }

    #[test]
    fn a_copy_too_incomplete_to_serve_offers_nothing() {
        let s = swarm(4, 40.0);
        assert_eq!(s.serving_capacity(100.0, 0.0), 0.0);
        assert!(s.serving_capacity(100.0, s.min_serve_progress + 0.01) > 0.0);
    }

    #[test]
    fn a_partial_copy_still_earns() {
        // Measured: PS5 and game packages reached ratio 1.6–2.3 at 3%
        // downloaded. A model where partial copies earn nothing cannot
        // reproduce that, and would train a policy to wait.
        let s = swarm(5, 40.0);
        let partial = s.offered_to(60.0, s.serving_capacity(100.0, 0.03), 0.0);
        assert!(partial > 0.0, "a 3% copy offered nothing");
    }

    #[test]
    fn two_of_our_boxes_split_the_swarm() {
        let s = swarm(6, 40.0);
        let alone = s.offered_to(60.0, 100.0, 0.0);
        let shared = s.offered_to(60.0, 100.0, 100.0);
        assert!(shared < alone, "shared {shared} !< alone {alone}");
        // Splitting must not be free: two boxes together earn less than twice
        // one, which is the real trade-off in dual racing.
        assert!(shared * 2.0 < alone * 2.0);
    }

    #[test]
    fn no_capacity_means_no_demand() {
        let s = swarm(7, 40.0);
        assert_eq!(s.offered_to(60.0, 0.0, 50.0), 0.0);
    }
}
