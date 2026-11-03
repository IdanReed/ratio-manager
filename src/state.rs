//! What every task shares: the latest view of each box, the settings, the
//! halt, and the handles to the bus, the store and TL.

use crate::config::{BoxCfg, Registry, Settings, GB, MIB, TB};
use crate::events::Bus;
use crate::qbit::{self, f, s};
use crate::store::Store;
use crate::tl::{Prowlarr, Tl};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};

pub const CATEGORY: &str = "ratio";
pub const TAG: &str = "ratio-manager";

/// One box as the poller last saw it. `torrents` holds only the `ratio`
/// category (the manager's), merged from maindata deltas.
#[derive(Debug, Clone, Default)]
pub struct BoxView {
    pub name: String,
    pub reachable: bool,
    pub last_ok: i64,
    pub last_error: Option<String>,
    pub latency_ms: f64,
    pub timeouts: u64,
    pub torrents: HashMap<String, Value>,
    /// Every torrent on the client, ratio or not: (size, amount_left).
    pub all_sizes: HashMap<String, (f64, f64)>,
    pub server: Value,
    /// Per hash: (unix time, uploaded bytes), oldest first.
    pub up_hist: HashMap<String, VecDeque<(f64, f64)>>,
    /// (unix time, up + down bytes/s) over the last 30 min.
    pub speeds: VecDeque<(f64, f64)>,
    pub firewalled_since: Option<i64>,
    pub allowance_spent: bool,
    pub traffic_used: f64,
    pub ips: Vec<String>,
    pub psi: Option<Value>,
    /// Adds the client has not listed yet (hash -> (placeholder, unix time)):
    /// kept in `torrents` until a poll shows them, so no decision in between
    /// sees the box without them.
    pub pending: HashMap<String, (Value, i64)>,
}

const UPLOAD_WINDOW: f64 = 1800.0;

impl BoxView {
    pub fn note_uploads(&mut self, now: f64) {
        let held: HashSet<String> = self.torrents.keys().cloned().collect();
        for (h, t) in &self.torrents {
            let hist = self.up_hist.entry(h.clone()).or_default();
            hist.push_back((now, f(t, "uploaded")));
            while hist.len() > 2 && now - hist[1].0 >= UPLOAD_WINDOW {
                hist.pop_front();
            }
        }
        self.up_hist.retain(|h, _| held.contains(h));
    }

    /// Bytes/s uploaded over the last 30 min (or as much of it as seen); the
    /// current rate until ten minutes have been seen.
    pub fn recent_upload(&self, hash: &str) -> f64 {
        let cur = self.torrents.get(hash).map(|t| f(t, "upspeed")).unwrap_or(0.0);
        match self.up_hist.get(hash) {
            Some(h) if h.len() >= 2 && h.back().unwrap().0 - h.front().unwrap().0 >= 600.0 => {
                let (t0, u0) = *h.front().unwrap();
                let (t1, u1) = *h.back().unwrap();
                ((u1 - u0) / (t1 - t0)).max(0.0)
            }
            _ => cur,
        }
    }

    pub fn active_downloads(&self) -> usize {
        self.torrents.values().filter(|t| qbit::downloading(t)).count()
    }

    pub fn dl_speed(&self) -> f64 {
        f(&self.server, "dl_info_speed")
    }
}

/// What a box can still take, given its settings.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Room {
    /// Bytes an add may still use: free disk under the quota, less what
    /// in-flight downloads will write and the reserve.
    pub placeable: f64,
    pub used: f64,
    pub quota: f64,
    pub active: usize,
    pub dl_speed: f64,
    pub headroom: bool,
}

pub fn room(view: &BoxView, settings: &Settings) -> Room {
    let b = settings.box_settings(&view.name);
    let fs_free = f(&view.server, "free_space_on_disk");
    let pending_all: f64 = view.all_sizes.values().map(|(_, left)| *left).sum();
    let ours_written: f64 = view.torrents.values().map(|t| f(t, "size") - f(t, "amount_left")).sum();
    let ours_pending: f64 = view.torrents.values().map(|t| f(t, "amount_left")).sum();
    let quota = b.disk.quota_gb * GB;
    let reserve = b.disk.reserve_gb * GB;
    // Under a quota, the manager's torrents' share of it; and never past what
    // the filesystem really has (a shared box reports the whole server's).
    let mut placeable = fs_free - pending_all - reserve;
    if quota > 0.0 {
        placeable = placeable.min(quota - ours_written - ours_pending - reserve);
    }
    let active = view.active_downloads();
    let dl = view.dl_speed();
    let headroom = active < b.downloads.max_active || dl < b.downloads.busy_above_mib * MIB;
    Room { placeable, used: ours_written, quota, active, dl_speed: dl, headroom }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Halt {
    pub reason: String,
    pub at: i64,
}

pub struct App {
    pub registry: Registry,
    pub settings: RwLock<Arc<Settings>>,
    pub boxes: RwLock<HashMap<String, BoxView>>,
    pub bus: Arc<Bus>,
    pub store: Arc<Store>,
    pub tl: Arc<Tl>,
    pub prowlarr: Arc<Prowlarr>,
    pub halt: RwLock<Option<Halt>>,
    pub dry_run: bool,
    /// hash -> announce unix time (for registration delays).
    pub announced: RwLock<HashMap<String, i64>>,
    /// Candidates per box from the policy's last evaluation, for the UI.
    pub candidates: RwLock<HashMap<String, Vec<Value>>>,
    /// The scoring tables: (box, source, size bucket, demand bucket) -> median ratio.
    pub tables: RwLock<HashMap<String, (f64, usize)>>,
    pub ntfy: Option<String>,
    pub http: reqwest::Client,
    pub started: i64,
}

impl App {
    pub fn settings(&self) -> Arc<Settings> {
        self.settings.read().map(|s| s.clone()).unwrap_or_else(|e| e.into_inner().clone())
    }

    pub fn halted(&self) -> Option<Halt> {
        self.halt.read().ok().and_then(|h| h.clone())
    }

    pub fn view(&self, name: &str) -> Option<BoxView> {
        self.boxes.read().ok().and_then(|b| b.get(name).cloned())
    }

    pub fn views(&self) -> Vec<BoxView> {
        let mut v: Vec<BoxView> = self.boxes.read().map(|b| b.values().cloned().collect()).unwrap_or_default();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }

    pub fn box_cfg(&self, name: &str) -> Option<&BoxCfg> {
        self.registry.box_cfg(name)
    }

    /// Every hash any box holds, reachable this pass or not (an unreachable
    /// box keeps its last-known holdings).
    pub fn held_anywhere(&self) -> HashMap<String, Vec<String>> {
        let mut out: HashMap<String, Vec<String>> = HashMap::new();
        if let Ok(b) = self.boxes.read() {
            for v in b.values() {
                for h in v.torrents.keys() {
                    out.entry(h.clone()).or_default().push(v.name.clone());
                }
            }
        }
        out
    }

    /// Every IP of every box but `except`.
    pub fn our_ips(&self, except: Option<&str>) -> HashMap<String, String> {
        let mut out = HashMap::new();
        if let Ok(b) = self.boxes.read() {
            for v in b.values() {
                if Some(v.name.as_str()) == except {
                    continue;
                }
                for ip in &v.ips {
                    out.insert(ip.clone(), v.name.clone());
                }
            }
        }
        out
    }

    pub async fn alert(&self, title: &str, msg: &str, priority: &str) {
        self.bus.emit("alert", None, None, Some("manager"), json!({"title": title, "message": msg, "priority": priority}));
        tracing::warn!("ALERT {title}: {msg}");
        if let Some(url) = &self.ntfy {
            let r = self
                .http
                .post(url)
                .header("Title", title)
                .header("Priority", priority)
                .header("Tags", "ratio")
                .body(msg.to_string())
                .send()
                .await;
            if let Err(e) = r {
                tracing::warn!("ntfy unreachable ({}) - alert was: {title}: {msg}", crate::qbit::strip_url(&e));
            }
        }
    }

    pub async fn set_halt(&self, reason: String) {
        let h = Halt { reason: reason.clone(), at: chrono::Utc::now().timestamp() };
        if let Ok(mut g) = self.halt.write() {
            if g.is_some() {
                return;
            }
            *g = Some(h.clone());
        }
        self.store.set("halt", &serde_json::to_string(&h).unwrap_or_default());
        self.alert("ratio-manager HALTED", &reason, "urgent").await;
    }

    pub fn clear_halt(&self, who: &str) -> Option<Halt> {
        let old = self.halt.write().ok().and_then(|mut g| g.take());
        self.store.del("halt");
        if let Some(h) = &old {
            self.bus.emit("alert", None, None, Some(who), json!({"title": "halt cleared", "message": h.reason, "priority": "default"}));
        }
        old
    }

    /// The snapshot `/state` serves.
    pub fn snapshot(&self) -> Value {
        let settings = self.settings();
        let boxes: Vec<Value> = self
            .views()
            .iter()
            .map(|v| {
                let r = room(v, &settings);
                let b = settings.box_settings(&v.name);
                let up: f64 = v.torrents.values().map(|t| f(t, "upspeed")).sum();
                let leechers: i64 = v.torrents.values().map(|t| qbit::i(t, "num_leechs")).sum();
                json!({
                    "name": v.name, "reachable": v.reachable, "last_ok": v.last_ok, "last_error": v.last_error,
                    "latency_ms": v.latency_ms, "timeouts": v.timeouts,
                    "torrents": v.torrents.len(),
                    "downloading": r.active, "seeding": v.torrents.values().filter(|t| qbit::finished(t)).count(),
                    "up_bps": up, "client_up_bps": f(&v.server, "up_info_speed"), "client_dl_bps": f(&v.server, "dl_info_speed"),
                    "leechers": leechers,
                    "room": r, "connection": s(&v.server, "connection_status"),
                    "queued_io_jobs": f(&v.server, "queued_io_jobs"), "average_time_queue": f(&v.server, "average_time_queue"),
                    "traffic_used_tb": v.traffic_used / TB, "traffic_quota_tb": b.traffic.quota_tb,
                    "allowance_spent": v.allowance_spent, "ips": v.ips, "psi": v.psi,
                    "policy": settings.policy.get(&v.name), "cost_eur_month": b.cost_eur_month,
                    "uploaded_held": v.torrents.values().map(|t| f(t, "uploaded")).sum::<f64>(),
                })
            })
            .collect();
        json!({
            "now": chrono::Utc::now().timestamp(), "started": self.started, "dry_run": self.dry_run,
            "halt": self.halted(), "vip": settings.vip_now(), "vip_until": settings.account.vip_until,
            "boxes": boxes, "seq": self.bus.last_seq(), "normal_lane": self.bus.normal.len(),
        })
    }
}
