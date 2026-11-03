//! The environment: a virtual clock, the releases being replayed, what our
//! boxes hold, and the integrator that moves bytes.
//!
//! A step is 30 virtual seconds, the same cadence the real box drivers poll
//! at, so a tick here corresponds to a `torrent_state` observation there. One
//! month of fleet time runs in seconds.
//!
//! The integrator's order matters and mirrors reality: the swarm decides how
//! much it will take, each box resolves what its line and disk can actually
//! move, and only then is that split back across the torrents it holds. A box
//! is the contended resource, so nothing may be credited to a torrent that
//! the box could not have moved.

use crate::boxes::{Box_, Flow};
use crate::params::Params;
use crate::swarm::Swarm;
use crate::link::{self, Link};
use crate::tracker::{Registration, Tracker};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

/// Seconds per step. The real drivers poll every 30 s.
pub const DT: f64 = 30.0;

/// A release being replayed: what TorrentLeech published and when.
#[derive(Debug, Clone)]
pub struct Release {
    pub tl_id: String,
    pub name: String,
    pub category: String,
    /// GB.
    pub size: f64,
    pub freeleech: bool,
    /// Seconds into the episode at which it is announced.
    pub announce_at: f64,
    /// What the listing showed, which is what a policy gets to see.
    pub listed_leechers: f64,
    pub listed_seeders: f64,
}

/// One copy of a release on one box.
#[derive(Debug, Clone)]
pub struct Torrent {
    pub id: usize,
    pub box_name: String,
    pub release: usize,
    pub added_at: f64,
    /// When the tracker registered it; until then it moves nothing.
    pub registration: Registration,
    pub registered_at: Option<f64>,
    /// 0.0 to 1.0.
    pub progress: f64,
    pub uploaded_gb: f64,
    pub downloaded_gb: f64,
    /// Virtual seconds spent seeding a complete copy.
    pub seeding_time: f64,
    pub removed_at: Option<f64>,
    pub removed_rule: Option<String>,
}

impl Torrent {
    pub fn done(&self) -> bool {
        self.progress >= 1.0
    }
    pub fn live(&self) -> bool {
        self.removed_at.is_none()
    }
    pub fn ratio(&self, size_gb: f64) -> f64 {
        if self.downloaded_gb > 0.0 {
            self.uploaded_gb / self.downloaded_gb
        } else if self.uploaded_gb > 0.0 {
            // Freeleech: nothing was charged, so report against the size.
            self.uploaded_gb / size_gb.max(1e-9)
        } else {
            0.0
        }
    }
    /// Registered and able to move bytes at time `t`.
    pub fn active(&self, t: f64) -> bool {
        self.live() && self.registered_at.map(|r| t >= r).unwrap_or(false)
    }
}

/// What a policy may do. Every one of these corresponds to a `POST /actions`
/// in the running manager.
#[derive(Debug, Clone, PartialEq)]
pub enum Act {
    Add { box_name: String, release: usize },
    Remove { box_name: String, torrent: usize },
    SetCap { box_name: String, mib: f64 },
}

/// Why an action was refused.
///
/// The reasons come from `ratio_manager::guard`, verbatim — the simulator does
/// not have its own opinion about what is allowed. Only the two cases the
/// guard cannot be asked about (an action naming something that does not
/// exist) are decided here.
#[derive(Debug, Clone, PartialEq)]
pub enum Refusal {
    NoSuchBox,
    NoSuchRelease,
    NoSuchTorrent,
    /// `guard::check_add` refused, with every reason it gave.
    Guard(Vec<String>),
    /// `guard::may_remove` allowed no rule, so the removal would be a
    /// hit-and-run or worse.
    NotRemovable,
}

impl Refusal {
    /// Whether any guard reason mentions `needle`. For tests and reports:
    /// the guard's wording is the service's, not ours to restate.
    pub fn mentions(&self, needle: &str) -> bool {
        match self {
            Refusal::Guard(rs) => rs.iter().any(|r| r.to_lowercase().contains(needle)),
            _ => false,
        }
    }
}

/// The outcome of one action, logged exactly as `/actions` logs it.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub act: Act,
    pub refused: Option<Refusal>,
}

/// The reward earned over an episode, and its parts, so a result can be read
/// rather than just compared.
#[derive(Debug, Clone, Default)]
pub struct Reward {
    pub uploaded_gb: f64,
    pub quota_cost: f64,
    pub disk_cost: f64,
    pub rent_cost: f64,
}

impl Reward {
    pub fn total(&self) -> f64 {
        self.uploaded_gb - self.quota_cost - self.disk_cost - self.rent_cost
    }
}

/// A policy: anything that can be handed the state and asked what to do. The
/// shipping policies are HTTP clients of `/api/v1`; in-process ones here let
/// the simulator be tested and a baseline be benchmarked without a server.
pub trait Policy {
    /// A release has just been announced on IRC.
    fn on_announce(&mut self, sim: &Sim, release: usize) -> Vec<Act>;
    /// Every tick, for evictions and limit changes.
    fn on_tick(&mut self, _sim: &Sim) -> Vec<Act> {
        Vec::new()
    }
    fn name(&self) -> &str;
}

pub struct Sim {
    pub params: Params,
    pub rng: StdRng,
    /// Virtual seconds since the episode began.
    pub now: f64,
    pub boxes: Vec<Box_>,
    pub tracker: Tracker,
    pub releases: Vec<Release>,
    pub swarms: Vec<Swarm>,
    pub torrents: Vec<Torrent>,
    pub outcomes: Vec<Outcome>,
    /// Reserve per box, GB, as the live gate keeps.
    pub reserve_gb: f64,
    /// The headless service, which owns every rule about what is allowed.
    pub link: Link,
    next_release: usize,
}

impl Sim {
    pub fn new(params: Params, releases: Vec<Release>, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let tracker = Tracker::sample(&params.tracker, &mut rng);
        let boxes: Vec<Box_> = params.boxes.iter().map(|p| Box_::sample(p, &mut rng)).collect();
        let swarms = releases
            .iter()
            .map(|r| Swarm::sample(&params.swarm, r.listed_leechers, &mut rng))
            .collect();
        Self {
            params,
            rng,
            now: 0.0,
            boxes,
            tracker,
            releases,
            swarms,
            torrents: Vec::new(),
            outcomes: Vec::new(),
            reserve_gb: 50.0,
            // Only fails if the simulator's own compiled-in registry or the
            // service's default settings stopped parsing, which is a build
            // error, not a runtime one. `link::tests` covers it.
            link: Link::new().expect("the simulator's registry must parse"),
            next_release: 0,
        }
    }

    pub fn box_index(&self, name: &str) -> Option<usize> {
        self.boxes.iter().position(|b| b.name == name)
    }

    pub fn box_by_name(&self, name: &str) -> Option<&Box_> {
        self.boxes.iter().find(|b| b.name == name)
    }

    /// The torrents a box currently holds.
    pub fn held<'a>(&'a self, box_name: &'a str) -> impl Iterator<Item = &'a Torrent> + 'a {
        self.torrents.iter().filter(move |t| t.box_name == box_name && t.live())
    }

    /// GB a box has still to write for the copies it is downloading.
    pub fn in_flight_gb(&self, box_name: &str) -> f64 {
        self.held(box_name)
            .filter(|t| !t.done())
            .map(|t| self.releases[t.release].size * (1.0 - t.progress))
            .sum()
    }

    /// Room for a new copy of `size` GB, the same arithmetic the live gate
    /// does: free disk, less what is still to be written, less the reserve.
    pub fn has_room(&self, box_name: &str, size: f64) -> bool {
        let Some(b) = self.box_by_name(box_name) else { return false };
        b.free_gb(self.reserve_gb) - self.in_flight_gb(box_name) >= size
    }

    /// Ask the real `guard::check_add`.
    fn guard_add(&self, box_name: &str, release: usize) -> Option<Refusal> {
        if self.box_index(box_name).is_none() {
            return Some(Refusal::NoSuchBox);
        }
        let Some(r) = self.releases.get(release) else { return Some(Refusal::NoSuchRelease) };
        self.link.sync(self);
        match self.link.check_add(box_name, r, release) {
            Ok(()) => None,
            Err(why) => Some(Refusal::Guard(why)),
        }
    }

    /// Ask the real `guard::may_remove`. It returns the rule that permits the
    /// removal, or nothing, in which case the removal does not happen.
    fn guard_remove(&self, box_name: &str, torrent: usize) -> Option<Refusal> {
        let Some(t) = self.torrents.get(torrent) else { return Some(Refusal::NoSuchTorrent) };
        if t.box_name != box_name || !t.live() {
            return Some(Refusal::NoSuchTorrent);
        }
        self.link.sync(self);
        let hash = link::sim_hash(&self.releases[t.release].tl_id, t.release);
        let unregistered = t.registration == Registration::Never;
        match self.link.may_remove(box_name, &hash, unregistered) {
            Some(_rule) => None,
            None => Some(Refusal::NotRemovable),
        }
    }

    /// The rule the guard cited for a removal, for the event the simulator
    /// records. Only called once the removal has been allowed.
    fn removal_rule(&self, box_name: &str, torrent: usize) -> String {
        let t = &self.torrents[torrent];
        let hash = link::sim_hash(&self.releases[t.release].tl_id, t.release);
        let unregistered = t.registration == Registration::Never;
        self.link
            .may_remove(box_name, &hash, unregistered)
            .unwrap_or_else(|| "allowed without a rule".into())
    }

    /// Apply one action, logging it the way `/actions` does.
    pub fn apply(&mut self, act: Act) -> Outcome {
        let refused = match &act {
            Act::Add { box_name, release } => self.guard_add(box_name, *release),
            Act::Remove { box_name, torrent } => self.guard_remove(box_name, *torrent),
            Act::SetCap { box_name, .. } => {
                if self.box_index(box_name).is_none() {
                    Some(Refusal::NoSuchBox)
                } else {
                    None
                }
            }
        };
        if refused.is_none() {
            match &act {
                Act::Add { box_name, release } => {
                    let reg = self.tracker.register(&mut self.rng);
                    let id = self.torrents.len();
                    self.torrents.push(Torrent {
                        id,
                        box_name: box_name.clone(),
                        release: *release,
                        added_at: self.now,
                        registered_at: reg.seconds().map(|s| self.now + s),
                        registration: reg,
                        progress: 0.0,
                        uploaded_gb: 0.0,
                        downloaded_gb: 0.0,
                        seeding_time: 0.0,
                        removed_at: None,
                        removed_rule: None,
                    });
                }
                Act::Remove { box_name, torrent } => {
                    // The guard's own rule, recorded as the removal's reason.
                    let rule = self.removal_rule(box_name, *torrent);
                    let size = self.releases[self.torrents[*torrent].release].size;
                    let t = &mut self.torrents[*torrent];
                    t.removed_at = Some(self.now);
                    t.removed_rule = Some(rule);
                    let used = size * t.progress;
                    let name = t.box_name.clone();
                    if let Some(i) = self.box_index(&name) {
                        self.boxes[i].disk_used_gb = (self.boxes[i].disk_used_gb - used).max(0.0);
                    }
                }
                Act::SetCap { box_name, mib } => {
                    let i = self.box_index(box_name).expect("checked above");
                    self.boxes[i].dl_cap = mib.max(0.0);
                }
            }
        }
        let out = Outcome { act, refused };
        self.outcomes.push(out.clone());
        out
    }

    /// Releases announced in the step just beginning.
    fn announcements(&mut self) -> Vec<usize> {
        let mut out = Vec::new();
        while self.next_release < self.releases.len()
            && self.releases[self.next_release].announce_at <= self.now
        {
            out.push(self.next_release);
            self.next_release += 1;
        }
        out
    }

    /// Move one step of `DT` virtual seconds. Returns the reward earned in it.
    pub fn step(&mut self, policy: &mut dyn Policy) -> Reward {
        for r in self.announcements() {
            for act in policy.on_announce(self, r) {
                self.apply(act);
            }
        }
        for act in policy.on_tick(self) {
            self.apply(act);
        }

        let mut reward = Reward::default();

        // Per box: ask the swarms what they would give and take, resolve what
        // the box can actually move, then split it back across its torrents.
        for bi in 0..self.boxes.len() {
            let box_name = self.boxes[bi].name.clone();
            let line_up = self.boxes[bi].line_up;

            // What each of this box's torrents wants to download, and what
            // demand it is being offered.
            let mut wants: Vec<(usize, f64)> = Vec::new();
            let mut offers: Vec<(usize, f64)> = Vec::new();
            for t in self.torrents.iter().filter(|t| t.box_name == box_name && t.active(self.now)) {
                let rel = &self.releases[t.release];
                let sw = &self.swarms[t.release];
                let since = self.now - rel.announce_at;
                if !t.done() {
                    // A swarm can only feed us what its seeders serve. Taking
                    // the seeders' aggregate capacity as the ceiling keeps a
                    // fresh release from downloading at line rate from one
                    // uploader — the NHL pack's 3–8 MiB/s from a lone seeder.
                    let supply = sw.competing_seeders(since) * sw.competitor_up;
                    wants.push((t.id, supply));
                }
                // Our other boxes on the same release split the swarm with us.
                let peers: f64 = self
                    .torrents
                    .iter()
                    .filter(|o| o.release == t.release && o.box_name != box_name && o.active(self.now))
                    .map(|o| sw.serving_capacity(line_up, o.progress))
                    .sum();
                let ours = sw.serving_capacity(line_up, t.progress);
                offers.push((t.id, sw.offered_to(since, ours, peers)));
            }

            let want_down: f64 = wants.iter().map(|(_, w)| w).sum();
            let offered_up: f64 = offers.iter().map(|(_, o)| o).sum();
            let flow: Flow = self.boxes[bi].resolve(want_down, offered_up);

            // Split the box's actual throughput proportionally to demand.
            if want_down > 0.0 {
                for (id, w) in &wants {
                    let share = flow.down * (w / want_down);
                    let gb = share * DT / 1024.0;
                    let size = self.releases[self.torrents[*id].release].size;
                    let t = &mut self.torrents[*id];
                    let room = (1.0 - t.progress) * size;
                    let moved = gb.min(room);
                    t.downloaded_gb += moved;
                    t.progress = (t.progress + moved / size.max(1e-9)).min(1.0);
                    self.boxes[bi].disk_used_gb += moved;
                }
            }
            if offered_up > 0.0 {
                for (id, o) in &offers {
                    let gb = flow.up * (o / offered_up) * DT / 1024.0;
                    self.torrents[*id].uploaded_gb += gb;
                    reward.uploaded_gb += gb;
                }
            }
            for t in self.torrents.iter_mut().filter(|t| t.box_name == box_name && t.live()) {
                if t.done() {
                    t.seeding_time += DT;
                }
            }

            // Costs. Metered download is the scarce input on a quota'd box.
            let before = self.boxes[bi].traffic_used_gb;
            self.boxes[bi].advance(flow, DT);
            if self.boxes[bi].quota_gb.is_some() {
                let spent = self.boxes[bi].traffic_used_gb - before;
                reward.quota_cost += spent * self.params.reward.gb_quota;
            }
            reward.disk_cost +=
                self.boxes[bi].disk_used_gb * self.params.reward.gb_day_disk * DT / 86400.0;
            if self.params.reward.charge_rent {
                reward.rent_cost +=
                    self.boxes[bi].cost_eur_month * DT / (30.0 * 86400.0);
            }
        }

        self.now += DT;
        reward
    }

    /// Run until `until` virtual seconds, summing the reward.
    pub fn run(&mut self, policy: &mut dyn Policy, until: f64) -> Reward {
        let mut total = Reward::default();
        while self.now < until {
            let r = self.step(policy);
            total.uploaded_gb += r.uploaded_gb;
            total.quota_cost += r.quota_cost;
            total.disk_cost += r.disk_cost;
            total.rent_cost += r.rent_cost;
        }
        total
    }

    /// Per-box race ratios, the numbers the validation gate checks against the
    /// fleet's measured distributions.
    pub fn race_ratios(&self) -> HashMap<String, Vec<f64>> {
        let mut out: HashMap<String, Vec<f64>> = HashMap::new();
        for t in &self.torrents {
            if t.downloaded_gb <= 0.0 {
                continue;
            }
            let size = self.releases[t.release].size;
            out.entry(t.box_name.clone()).or_default().push(t.ratio(size));
        }
        out
    }
}

/// Median of a sample, for the validation gate and the tests.
pub fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}
