//! A box as a fluid: rates in, rates out, one shared disk between them.
//!
//! No per-piece BitTorrent. A box has a line, a disk, optionally a tunnel and
//! optionally a traffic allowance, and the question the simulator asks it every
//! step is the same one the real boxes answer: given this much wanted download
//! and this much offered demand, what actually moves?
//!
//! The behaviour this has to reproduce is the fleet's hard-won lesson: a box's
//! download rate and its upload rate are not independent. They compete for the
//! same disk, and past a box-specific knee, pushing download down costs more
//! upload than the download was worth.

use crate::params::BoxParams;
use rand::Rng;

/// One box's sampled physics for an episode, plus the state that moves.
#[derive(Debug, Clone)]
pub struct Box_ {
    pub name: String,
    pub line_down: f64,
    pub line_up: f64,
    pub disk_total: f64,
    pub disk_knee: f64,
    pub disk_decay: f64,
    pub tunnel_knee: Option<f64>,
    pub tunnel_starve: f64,
    pub neighbour_load: f64,
    pub quota_gb: Option<f64>,
    pub throttle_mbit: f64,
    pub disk_gb: f64,
    pub cost_eur_month: f64,

    /// Download cap the manager has set on this box, MiB/s. This is the knob a
    /// policy actually turns.
    pub dl_cap: f64,
    /// GB of allowance already spent this period.
    pub traffic_used_gb: f64,
    /// GB of disk in use.
    pub disk_used_gb: f64,
}

/// What moved in one interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Flow {
    /// MiB/s actually downloaded.
    pub down: f64,
    /// MiB/s actually uploaded.
    pub up: f64,
    /// Fraction of the disk's capacity in use, past 1.0 when it is overloaded.
    pub disk_pressure: f64,
    /// True once the allowance is spent and the host has throttled the box.
    pub throttled: bool,
}

impl Box_ {
    pub fn sample<R: Rng + ?Sized>(p: &BoxParams, rng: &mut R) -> Self {
        let line_down = p.line_down.sample(rng);
        Self {
            name: p.name.clone(),
            line_down,
            line_up: p.line_up.sample(rng),
            disk_total: p.disk_total.sample(rng),
            disk_knee: p.disk_knee.sample(rng),
            disk_decay: p.disk_decay.sample(rng),
            tunnel_knee: p.tunnel_knee.as_ref().map(|r| r.sample(rng)),
            tunnel_starve: p.tunnel_starve.sample(rng),
            neighbour_load: p.neighbour_load.sample(rng),
            quota_gb: p.quota_gb.as_ref().map(|r| r.sample(rng)),
            throttle_mbit: p.throttle_mbit.sample(rng),
            disk_gb: p.disk_gb.sample(rng),
            cost_eur_month: p.cost_eur_month.sample(rng),
            dl_cap: line_down,
            traffic_used_gb: 0.0,
            disk_used_gb: 0.0,
        }
    }

    /// True once the allowance is spent. The live rule is two-part — the
    /// counter has reached the quota *and* the box has stopped moving faster
    /// than twice its throttle rate — because the hosts count differently from
    /// us. Here the counter is ours by construction, so the first half is all
    /// there is.
    pub fn spent(&self) -> bool {
        self.quota_gb.map(|q| self.traffic_used_gb >= q).unwrap_or(false)
    }

    /// Disk throughput available to us, after neighbours and after the
    /// degradation that sets in past the knee.
    ///
    /// Past the knee, service falls off as `1 / (1 + excess^decay)` where
    /// `excess` is how far past the knee the write rate sits, in knees. This
    /// is what makes pulsed collapse: writing at 70 MiB/s against a 20 MiB/s
    /// knee leaves a fraction of the disk for reads, and the uploads that
    /// depend on those reads fall to single MiB/s.
    pub fn disk_available(&self, write_rate: f64) -> f64 {
        let ours = self.disk_total * (1.0 - self.neighbour_load).max(0.0);
        if write_rate <= self.disk_knee || self.disk_knee <= 0.0 {
            return ours;
        }
        let excess = (write_rate - self.disk_knee) / self.disk_knee;
        ours / (1.0 + excess.powf(self.disk_decay))
    }

    /// Resolve one interval.
    ///
    /// `want_down` is what the swarms could supply us, `offered_up` what our
    /// leechers are asking for. Both MiB/s. Download is served first, because
    /// that is what qBittorrent does, and upload takes what the disk has left.
    pub fn resolve(&self, want_down: f64, offered_up: f64) -> Flow {
        let mut throttled = false;
        let mut line_down = self.line_down;
        let mut line_up = self.line_up;
        if self.spent() {
            // The host throttles the whole box, both directions.
            let mib = self.throttle_mbit / 8.0;
            line_down = line_down.min(mib);
            line_up = line_up.min(mib);
            throttled = true;
        }

        // What the policy, the line and the swarms would allow.
        let demanded = want_down.max(0.0).min(self.dl_cap).min(line_down);

        // Writes mostly proceed: they queue, and the disk absorbs them up to
        // its raw share. This is the measured behaviour — pulsed kept writing
        // at 110–200 MiB/s while its uploads fell under 1 MiB/s.
        let raw = (self.disk_total * (1.0 - self.neighbour_load)).max(0.0);
        let down = demanded.min(raw);

        // Service degrades with the write rate the box is *attempting*, not
        // the rate it achieves, and it is the reads that suffer. Driving this
        // off the achieved rate would make the degradation undo itself: a
        // download clamped down by a choked disk would make the disk look
        // healthy again and hand the upload its capacity back.
        let served = self.disk_available(demanded);
        let disk_left = (served - down).max(0.0);
        let mut up = offered_up.max(0.0).min(line_up).min(disk_left);

        // A single WireGuard tunnel carrying both directions: past the knee it
        // drops packets, and it is the uploads that starve. This is home's
        // behaviour — 45–60 MiB/s of download took upload from ~20 MiB/s to
        // 1–4.
        if let Some(knee) = self.tunnel_knee {
            if down > knee {
                let loss = ((down - knee) * self.tunnel_starve).clamp(0.0, 0.95);
                up *= 1.0 - loss;
            }
        }

        let pressure = if self.disk_total > 0.0 {
            (down + up) / (self.disk_total * (1.0 - self.neighbour_load).max(1e-9))
        } else {
            0.0
        };
        Flow { down, up, disk_pressure: pressure, throttled }
    }

    /// Advance the box's counters by `dt` seconds of the given flow.
    pub fn advance(&mut self, f: Flow, dt: f64) {
        let gb_down = f.down * dt / 1024.0;
        self.traffic_used_gb += gb_down;
        if let Some(_q) = self.quota_gb {
            // The hosts count upload against the allowance too.
            self.traffic_used_gb += f.up * dt / 1024.0;
        }
    }

    /// Free disk in GB, keeping a reserve.
    pub fn free_gb(&self, reserve_gb: f64) -> f64 {
        (self.disk_gb - self.disk_used_gb - reserve_gb).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::BoxParams;
    use rand::SeedableRng;

    fn boxed(p: &BoxParams, seed: u64) -> Box_ {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        Box_::sample(p, &mut rng)
    }

    #[test]
    fn download_is_capped_by_the_policys_cap() {
        let mut b = boxed(&BoxParams::seedit4me(), 1);
        b.dl_cap = 20.0;
        let f = b.resolve(500.0, 0.0);
        assert!(f.down <= 20.0 + 1e-9, "down {} ignored the cap", f.down);
    }

    #[test]
    fn a_box_cannot_download_more_than_the_swarm_supplies() {
        let b = boxed(&BoxParams::seedit4me(), 2);
        let f = b.resolve(3.0, 0.0);
        assert!((f.down - 3.0).abs() < 1e-9);
    }

    #[test]
    fn pulsed_uploads_well_at_twenty_and_collapses_at_seventy() {
        // The measured fact the model exists to carry: pulsed's uploads were
        // 21 MiB/s with downloads capped at 20, and 1–10 MiB/s at 70.
        let mut b = boxed(&BoxParams::pulsed(), 3);
        b.dl_cap = 20.0;
        let good = b.resolve(1000.0, 60.0);
        b.dl_cap = 70.0;
        let bad = b.resolve(1000.0, 60.0);
        assert!(good.up > bad.up * 2.0, "20 MiB/s cap gave {} up, 70 gave {}", good.up, bad.up);
    }

    #[test]
    fn seedit4me_holds_a_hundred_down_and_still_uploads() {
        let mut b = boxed(&BoxParams::seedit4me(), 4);
        b.dl_cap = 100.0;
        let f = b.resolve(1000.0, 100.0);
        assert!(f.down >= 95.0, "down {}", f.down);
        assert!(f.up > 20.0, "up {} — seedit4me raced and uploaded at once", f.up);
    }

    #[test]
    fn homes_tunnel_starves_upload_past_its_knee() {
        let mut b = boxed(&BoxParams::home(), 5);
        let knee = b.tunnel_knee.unwrap();
        b.dl_cap = knee * 0.5;
        let calm = b.resolve(1000.0, 50.0);
        b.dl_cap = knee + 20.0;
        let hot = b.resolve(1000.0, 50.0);
        assert!(calm.up > hot.up, "calm {} vs hot {}", calm.up, hot.up);
    }

    #[test]
    fn a_spent_allowance_throttles_both_directions() {
        let mut b = boxed(&BoxParams::seedit4me(), 6);
        b.traffic_used_gb = b.quota_gb.unwrap() + 1.0;
        let f = b.resolve(1000.0, 1000.0);
        assert!(f.throttled);
        let mib = b.throttle_mbit / 8.0;
        assert!(f.down <= mib + 1e-9 && f.up <= mib + 1e-9, "{f:?}");
    }

    #[test]
    fn an_unmetered_box_is_never_spent() {
        let mut b = boxed(&BoxParams::pulsed(), 7);
        b.traffic_used_gb = 500_000.0;
        assert!(!b.spent());
    }

    #[test]
    fn disk_degrades_monotonically_past_the_knee() {
        let b = boxed(&BoxParams::pulsed(), 8);
        let at_knee = b.disk_available(b.disk_knee);
        let past = b.disk_available(b.disk_knee * 3.0);
        let far = b.disk_available(b.disk_knee * 9.0);
        assert!(at_knee > past && past > far, "{at_knee} {past} {far}");
    }
}
