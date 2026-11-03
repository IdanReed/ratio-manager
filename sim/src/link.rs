//! The bridge to the real service: a headless `App`, so the simulator asks
//! `ratio_manager::guard` the same questions the running manager asks it.
//!
//! This is the one piece that makes a simulated result mean something. A
//! policy trained against a re-implemented guard learns to propose actions the
//! real guard rejects, and every such proposal is a wasted decision in
//! production. So the rules are not modelled here at all: the simulator owns
//! the physics — what a line and a disk and a swarm will do — and the service
//! owns what is allowed.
//!
//! Nothing here touches the network or the disk. The store is in memory, the
//! HTTP client is never used, and the registry is parsed from a string so the
//! real parser still validates it.

use anyhow::{Context, Result};
use ratio_manager::bencode::TorrentFacts;
use ratio_manager::config::{Registry, Settings};
use ratio_manager::events::{Bus, BusReceivers};
use ratio_manager::state::{App, BoxView};
use ratio_manager::store::{Release as StoreRelease, Store};
use ratio_manager::{guard, tl};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

use crate::env::{Release, Sim, Torrent};

const GB: f64 = 1_073_741_824.0;

/// A registry describing the simulated fleet. The addresses match each box's
/// `tl_registered_ip` so guard rule 8 is satisfied, which is correct: a box
/// whose address has drifted is a condition to test deliberately, not the
/// default state of every rollout.
fn registry_yaml() -> &'static str {
    r#"
ntfy_url: null
prowlarr:
  url: http://prowlarr:9696
  api_key_env: SIM_UNUSED
  indexers: ["TorrentLeech (freeleech)"]
autobrr:
  url: http://autobrr:7474
  user: sim
  password_env: SIM_UNUSED
  intake_url: http://ratio-manager:8080/api/v1/intake/autobrr
  intake_from_host: autobrr
  retire_filters: []
  retire_clients: []
tl:
  rss_key_env: SIM_UNUSED
  rss_url: "https://rss.example.invalid/{key}"
boxes:
  - name: home
    url: http://gluetun:8080
    user_env: SIM_UNUSED
    password_env: SIM_UNUSED
    save_path: /data/downloads/ratio
    tunnel: true
  - name: seedit4me
    url: https://seedit4me.invalid/qbittorrent
    user_env: SIM_UNUSED
    password_env: SIM_UNUSED
    basic_auth: true
    manage_prefs: true
    private_only: true
    tl_registered_ip: 193.163.71.39
  - name: pulsed
    url: https://pulsed.invalid/qbittorrent
    user_env: SIM_UNUSED
    password_env: SIM_UNUSED
    basic_auth: true
    manage_prefs: true
    private_only: true
    tl_registered_ip: 185.148.0.79
"#
}

/// A deterministic 40-hex info-hash for a simulated copy. The guard keys
/// everything by hash, so these have to be stable and distinct per release.
pub fn sim_hash(release_tl_id: &str, release_index: usize) -> String {
    // A small FNV-1a over the identity, widened to 40 hex characters. Not a
    // real SHA-1 and not meant to be: nothing verifies it.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in release_tl_id.as_bytes().iter().chain(release_index.to_le_bytes().iter()) {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    let mut s = String::with_capacity(40);
    let mut x = h;
    while s.len() < 40 {
        s.push_str(&format!("{:016x}", x));
        x = x.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(1);
    }
    s.truncate(40);
    s
}

pub struct Link {
    pub app: Arc<App>,
    /// Held so the bus's channels stay open; a dropped receiver would make
    /// every `emit` fail silently and hide a real problem.
    _rx: BusReceivers,
}

impl Link {
    pub fn new() -> Result<Self> {
        let registry: Registry =
            serde_yaml::from_str(registry_yaml()).context("the simulator's registry")?;
        let (settings, _) = Settings::from_overrides(&json!({}))
            .context("default settings did not validate")?;
        // `Store::open_memory` is gated to the service's own tests, so go
        // through the public constructor; SQLite gives an in-memory database
        // for this path and nothing touches the disk.
        let store = Arc::new(Store::open(":memory:").context("in-memory store")?);
        let (bus, rx) = Bus::new(0);
        let app = Arc::new(App {
            registry: registry.clone(),
            settings: RwLock::new(Arc::new(settings)),
            boxes: RwLock::new(HashMap::new()),
            bus,
            store,
            tl: Arc::new(tl::Tl::new(
                String::new(),
                registry.tl.rss_url.clone(),
                registry.tl.base_url.clone(),
            )),
            prowlarr: Arc::new(tl::Prowlarr::new(&registry.prowlarr.url, String::new())),
            halt: RwLock::new(None),
            dry_run: true,
            announced: RwLock::new(HashMap::new()),
            candidates: RwLock::new(HashMap::new()),
            tables: RwLock::new(HashMap::new()),
            ntfy: None,
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(1))
                .build()
                .context("http client")?,
            started: 0,
        });
        Ok(Self { app, _rx: rx })
    }

    fn settings(&self) -> Arc<Settings> {
        self.app.settings.read().expect("settings lock").clone()
    }

    /// Render one simulated copy as the qBittorrent object the guard reads.
    /// Sizes are bytes, because that is what the service works in.
    fn as_qbit(t: &Torrent, rel: &Release, now: f64) -> Value {
        let size = rel.size * GB;
        let left = size * (1.0 - t.progress);
        let registered = t.registered_at.map(|r| now >= r).unwrap_or(false);
        json!({
            "hash": sim_hash(&rel.tl_id, t.release),
            "name": rel.name,
            "category": ratio_manager::state::CATEGORY,
            "tags": format!("{},race", ratio_manager::state::TAG),
            "size": size,
            "amount_left": left,
            "progress": t.progress,
            "downloaded": t.downloaded_gb * GB,
            "uploaded": t.uploaded_gb * GB,
            "ratio": t.ratio(rel.size),
            "seeding_time": t.seeding_time,
            "completion_on": if t.done() { (t.added_at + 1.0) as i64 } else { 0 },
            "state": if !registered {
                "metaDL"
            } else if t.done() {
                "uploading"
            } else {
                "downloading"
            },
            "num_complete": 1,
            "num_incomplete": rel.listed_leechers,
            "dlspeed": 0,
            "upspeed": 0,
        })
    }

    /// Refresh the app's box views from the simulator's state, so the guard
    /// sees what the simulator sees.
    pub fn sync(&self, sim: &Sim) {
        let mut views: HashMap<String, BoxView> = HashMap::new();
        for b in &sim.boxes {
            let mut torrents = HashMap::new();
            let mut all_sizes = HashMap::new();
            for t in sim.torrents.iter().filter(|t| t.box_name == b.name && t.live()) {
                let rel = &sim.releases[t.release];
                let v = Self::as_qbit(t, rel, sim.now);
                let h = sim_hash(&rel.tl_id, t.release);
                all_sizes.insert(h.clone(), (rel.size * GB, rel.size * GB * (1.0 - t.progress)));
                torrents.insert(h, v);
            }
            let cfg = sim.params.box_params(&b.name);
            let _ = cfg;
            let ips = self
                .app
                .registry
                .box_cfg(&b.name)
                .and_then(|c| c.tl_registered_ip.clone())
                .map(|ip| vec![ip])
                .unwrap_or_default();
            views.insert(
                b.name.clone(),
                BoxView {
                    name: b.name.clone(),
                    reachable: true,
                    last_ok: sim.now as i64,
                    last_error: None,
                    latency_ms: 5.0,
                    timeouts: 0,
                    torrents,
                    all_sizes,
                    server: json!({
                        "free_space_on_disk": (b.disk_gb - b.disk_used_gb).max(0.0) * GB,
                        "alltime_ul": 0.0,
                        "alltime_dl": b.traffic_used_gb * GB,
                        "dl_info_speed": 0.0,
                        "up_info_speed": 0.0,
                    }),
                    up_hist: HashMap::new(),
                    speeds: VecDeque::new(),
                    firewalled_since: None,
                    allowance_spent: b.spent(),
                    traffic_used: b.traffic_used_gb * GB,
                    ips,
                    psi: None,
                    pending: HashMap::new(),
                },
            );
        }
        *self.app.boxes.write().expect("boxes lock") = views;
    }

    /// The real `guard::check_add`. `Ok(())` or every reason it refused.
    pub fn check_add(&self, box_name: &str, rel: &Release, release_index: usize) -> Result<(), Vec<String>> {
        let facts = TorrentFacts {
            hash: sim_hash(&rel.tl_id, release_index),
            private: true,
            size: rel.size * GB,
            name: rel.name.clone(),
            files: 1,
            piece_length: 4 * 1024 * 1024,
            announce: String::new(),
        };
        let sr = StoreRelease {
            tl_id: rel.tl_id.clone(),
            name: rel.name.clone(),
            category: rel.category.clone(),
            size: rel.size * GB,
            freeleech: rel.freeleech,
            first_seen: rel.announce_at as i64,
            hash: Some(facts.hash.clone()),
            private: Some(true),
            announce_ts: Some(rel.announce_at as i64),
            seeders: Some(rel.listed_seeders as i64),
            leechers: Some(rel.listed_leechers as i64),
            completed: Some(0),
            last_scrape: None,
            delta_l: None,
            delta_s: None,
            delta_ts: None,
            fetch_failed: 0,
            sched_idx: 0,
            source: "sim".into(),
        };
        let s = self.settings();
        guard::check_add(&self.app, &s, box_name, &facts, Some(&sr), "baseline")
    }

    /// The real `guard::may_remove`. `Some(rule)` if the removal is allowed.
    pub fn may_remove(&self, box_name: &str, hash: &str, unregistered: bool) -> Option<String> {
        let s = self.settings();
        let boxes = self.app.boxes.read().expect("boxes lock");
        let view = boxes.get(box_name)?;
        guard::may_remove(&self.app, &s, view, hash, unregistered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_headless_app_builds_from_the_real_parsers() {
        let l = Link::new().expect("the simulator's registry and settings must be valid");
        assert_eq!(l.app.registry.boxes.len(), 3);
        assert!(l.app.registry.box_cfg("pulsed").is_some());
    }

    #[test]
    fn hashes_are_stable_distinct_and_forty_hex() {
        let a = sim_hash("123", 0);
        let b = sim_hash("123", 1);
        let c = sim_hash("124", 0);
        assert_eq!(a, sim_hash("123", 0));
        assert_ne!(a, b);
        assert_ne!(a, c);
        for h in [&a, &b, &c] {
            assert_eq!(h.len(), 40, "{h}");
            assert!(h.chars().all(|ch| ch.is_ascii_hexdigit()), "{h}");
        }
    }
}
