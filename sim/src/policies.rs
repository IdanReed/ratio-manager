//! Policies to measure against each other inside the simulator.
//!
//! These are references, not the shipping policy. The policy that ships is an
//! HTTP client of `/api/v1`; these exist so the simulator can be validated and
//! so a learned policy has something to beat. Three of them:
//!
//!  - `Nothing`: adds nothing. The floor, and the only way to see what the
//!    standing library earns by itself.
//!  - `Everywhere`: races everything on every box. The ceiling on adds, and a
//!    demonstration that more adds are not more upload once the disks choke.
//!  - `TableBaseline`: the bucketed-median rule the live manager uses, so a
//!    simulated result can be read against the fleet's real numbers.

use crate::env::{Act, Policy, Sim};
use std::collections::HashMap;

/// Adds nothing, ever.
pub struct Nothing;

impl Policy for Nothing {
    fn on_announce(&mut self, _sim: &Sim, _release: usize) -> Vec<Act> {
        Vec::new()
    }
    fn name(&self) -> &str {
        "nothing"
    }
}

/// Races every freeleech release on every box that has room.
pub struct Everywhere;

impl Policy for Everywhere {
    fn on_announce(&mut self, sim: &Sim, release: usize) -> Vec<Act> {
        let r = &sim.releases[release];
        if !r.freeleech {
            return Vec::new();
        }
        sim.boxes
            .iter()
            .filter(|b| sim.has_room(&b.name, r.size))
            .map(|b| Act::Add { box_name: b.name.clone(), release })
            .collect()
    }
    fn name(&self) -> &str {
        "everywhere"
    }
}

/// The live manager's rule: a lookup table of `(box, source, size bucket,
/// demand bucket)` to the median ratio past races in that cell returned,
/// seeded with priors and multiplied by the box's bias. Pick the box with the
/// best estimate, add if it clears a floor.
///
/// This is a port, not the original. It exists to give simulated results a
/// reading against the fleet's measured per-box ratios; the original stays in
/// `ratio_manager::policy`, where the service uses it. When the two disagree
/// about a cell, the service is right.
pub struct TableBaseline {
    /// `(box, size bucket, demand bucket)` to a median ratio.
    table: HashMap<(String, &'static str, &'static str), (f64, usize)>,
    /// Per-box prior, used until a cell has races in it.
    priors: HashMap<String, f64>,
    /// Estimates below this are not worth an add.
    pub floor: f64,
    /// How many boxes may race one release.
    pub max_boxes: usize,
}

pub fn size_bucket(gb: f64) -> &'static str {
    match gb {
        g if g < 5.0 => "xs",
        g if g < 20.0 => "s",
        g if g < 60.0 => "m",
        g if g < 150.0 => "l",
        _ => "xl",
    }
}

pub fn demand_bucket(d: f64) -> &'static str {
    match d {
        x if x < 0.5 => "none",
        x if x < 2.0 => "low",
        x if x < 8.0 => "mid",
        _ => "high",
    }
}

impl TableBaseline {
    /// Priors from the fleet's measured medians: seedit4me earns ~1.4 a race,
    /// pulsed ~0.65, home ~0.06.
    pub fn new() -> Self {
        let mut priors = HashMap::new();
        priors.insert("seedit4me".to_string(), 1.4);
        priors.insert("pulsed".to_string(), 0.65);
        priors.insert("home".to_string(), 0.06);
        Self { table: HashMap::new(), priors, floor: 0.2, max_boxes: 2 }
    }

    pub fn estimate(&self, box_name: &str, size: f64, demand: f64) -> f64 {
        let k = (box_name.to_string(), size_bucket(size), demand_bucket(demand));
        match self.table.get(&k) {
            Some((m, n)) if *n >= 3 => *m,
            _ => *self.priors.get(box_name).unwrap_or(&0.5),
        }
    }

    /// Fold a finished race into the table, as the live manager does when it
    /// joins an outcome to a decision.
    pub fn observe(&mut self, box_name: &str, size: f64, demand: f64, ratio: f64) {
        let k = (box_name.to_string(), size_bucket(size), demand_bucket(demand));
        let e = self.table.entry(k).or_insert((0.0, 0));
        // A running mean, which is not the median the service keeps, but is
        // what can be maintained in one pass. The difference matters only in
        // cells with few races, where the prior dominates anyway.
        e.1 += 1;
        e.0 += (ratio - e.0) / e.1 as f64;
    }
}

impl Default for TableBaseline {
    fn default() -> Self {
        Self::new()
    }
}

impl Policy for TableBaseline {
    fn on_announce(&mut self, sim: &Sim, release: usize) -> Vec<Act> {
        let r = &sim.releases[release];
        if !r.freeleech {
            return Vec::new();
        }
        let demand = r.listed_leechers / (r.listed_seeders + 1.0);
        let mut ranked: Vec<(f64, String)> = sim
            .boxes
            .iter()
            .filter(|b| !b.spent() && sim.has_room(&b.name, r.size))
            .map(|b| (self.estimate(&b.name, r.size, demand), b.name.clone()))
            .filter(|(e, _)| *e >= self.floor)
            .collect();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        ranked
            .into_iter()
            .take(self.max_boxes)
            .map(|(_, name)| Act::Add { box_name: name, release })
            .collect()
    }

    fn on_tick(&mut self, sim: &Sim) -> Vec<Act> {
        // Evict what the guard allows and what has stopped earning: a copy
        // past 1:1, or one the tracker never registered.
        let mut acts = Vec::new();
        for t in sim.torrents.iter().filter(|t| t.live()) {
            let size = sim.releases[t.release].size;
            let unregistered = t.registered_at.is_none() && sim.now - t.added_at > 3600.0;
            if t.ratio(size) >= 1.0 || unregistered {
                acts.push(Act::Remove { box_name: t.box_name.clone(), torrent: t.id });
            }
        }
        acts
    }

    fn name(&self) -> &str {
        "table-baseline"
    }
}
