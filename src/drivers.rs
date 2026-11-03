//! Per-box drivers. Each box runs its own loops, each on its own qBittorrent
//! session: the poller (state, observations, `capacity` deltas), the
//! re-announcer and unregistered cleanup, the reconciler (preferences, bans,
//! addresses), and the wave-2 sensors (peer snapshots, host pressure).

use crate::actions::{Action, Executor};
use crate::config::{BoxCfg, GB, MIB, TB};
use crate::guard;
use crate::qbit::{self, f, s, Qbit};
use crate::state::{room, App, BoxView, CATEGORY};
use crate::store::Race;
use hmac::{Hmac, Mac};
use serde_json::{json, Map, Value};
use sha2::Sha256;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::broadcast;

/// Fields that carry the passkey or local paths: never in an event.
const SENSITIVE: &[&str] = &["tracker", "magnet_uri", "content_path", "save_path", "download_path", "root_path", "infohash_v1", "infohash_v2"];

pub fn scrub(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(m.iter().filter(|(k, _)| !SENSITIVE.contains(&k.as_str())).map(|(k, v)| (k.clone(), v.clone())).collect()),
        other => other.clone(),
    }
}

pub fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

fn nowf() -> f64 {
    chrono::Utc::now().timestamp_millis() as f64 / 1000.0
}

// ---------------------------------------------------------------------------
// Poller
// ---------------------------------------------------------------------------

struct PollState {
    rid: i64,
    all: HashMap<String, Value>,
    server: Value,
    last_capacity: Option<Value>,
    race_written: HashMap<String, i64>,
}

pub async fn poller(app: Arc<App>, cfg: BoxCfg, exec: Executor) {
    let q = match Qbit::new(&cfg, cfg.timeout_s) {
        Ok(q) => q,
        Err(e) => {
            tracing::error!("[{}] cannot build a client: {e}", cfg.name);
            return;
        }
    };
    let mut st = PollState { rid: 0, all: HashMap::new(), server: json!({}), last_capacity: None, race_written: HashMap::new() };
    let mut logged_in = false;
    let mut tick = tokio::time::interval(Duration::from_secs(30));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tick.tick().await;
        let t0 = Instant::now();
        let res = async {
            if !logged_in {
                q.login().await?;
                logged_in = true;
            }
            q.maindata(st.rid).await
        }
        .await;
        match res {
            Ok(md) => {
                let latency = t0.elapsed().as_secs_f64() * 1000.0;
                poll_ok(&app, &cfg, &q, &exec, &mut st, md, latency).await;
            }
            Err(e) => {
                logged_in = false;
                st.rid = 0;
                let msg = e.to_string();
                let became = {
                    let mut b = app.boxes.write().unwrap_or_else(|x| x.into_inner());
                    let v = b.entry(cfg.name.clone()).or_insert_with(|| BoxView { name: cfg.name.clone(), ..Default::default() });
                    let was = v.reachable;
                    v.reachable = false;
                    v.last_error = Some(msg.clone());
                    if msg.contains("timed out") {
                        v.timeouts += 1;
                    }
                    was
                };
                tracing::warn!("[{}] poll failed: {msg}", cfg.name);
                if became {
                    app.bus.emit("capacity", Some(&cfg.name), None, Some("driver"), json!({"reachable": false, "error": msg}));
                }
            }
        }
    }
}

async fn poll_ok(app: &Arc<App>, cfg: &BoxCfg, q: &Qbit, exec: &Executor, st: &mut PollState, md: Value, latency: f64) {
    let name = cfg.name.as_str();
    let full = md.get("full_update").and_then(|v| v.as_bool()).unwrap_or(false) || st.rid == 0;
    st.rid = md.get("rid").and_then(|v| v.as_i64()).unwrap_or(0);
    if full {
        st.all.clear();
        st.server = json!({});
    }
    // Merge the delta; keep each ratio torrent's changed fields for the log.
    let mut changes: Vec<(String, Value, bool)> = Vec::new();
    if let Some(Value::Object(ts)) = md.get("torrents") {
        for (h, delta) in ts {
            let is_new = !st.all.contains_key(h);
            let entry = st.all.entry(h.clone()).or_insert_with(|| json!({"hash": h}));
            if let (Some(e), Some(d)) = (entry.as_object_mut(), delta.as_object()) {
                for (k, v) in d {
                    e.insert(k.clone(), v.clone());
                }
            }
            if s(entry, "category") == CATEGORY {
                changes.push((h.clone(), if is_new { scrub(entry) } else { scrub(delta) }, is_new));
            }
        }
    }
    if let Some(Value::Array(gone)) = md.get("torrents_removed") {
        for h in gone.iter().filter_map(|v| v.as_str()) {
            if st.all.get(h).map(|t| s(t, "category") == CATEGORY).unwrap_or(false) {
                app.bus.emit("torrent_state", Some(name), Some(h), None, json!({"removed": true}));
            }
            st.all.remove(h);
        }
    }
    if let (Some(srv), Some(d)) = (st.server.as_object_mut(), md.get("server_state").and_then(|v| v.as_object())) {
        for (k, v) in d {
            srv.insert(k.clone(), v.clone());
        }
    }
    for (h, data, is_new) in changes {
        let mut d = data;
        if is_new {
            d["new"] = json!(true);
        }
        app.bus.emit("torrent_state", Some(name), Some(&h), None, d);
    }

    let settings = app.settings();
    let bset = settings.box_settings(name);
    let tnow = nowf();
    let ours: HashMap<String, Value> = st.all.iter().filter(|(_, t)| s(t, "category") == CATEGORY).map(|(h, t)| (h.clone(), t.clone())).collect();
    let all_sizes: HashMap<String, (f64, f64)> = st.all.iter().map(|(h, t)| (h.clone(), (f(t, "size"), f(t, "amount_left")))).collect();

    // Allowance: our counter since the period's base, and whether the box
    // actually behaves throttled.
    let counter = f(&st.server, "alltime_ul") + f(&st.server, "alltime_dl");
    let traffic_used = if bset.traffic.quota_tb > 0.0 { counter - traffic_base(app, name, &bset, counter) } else { 0.0 };

    let (capacity, spent_changed, reach_changed, view_copy) = {
        let mut b = app.boxes.write().unwrap_or_else(|x| x.into_inner());
        let v = b.entry(name.to_string()).or_insert_with(|| BoxView { name: name.to_string(), ..Default::default() });
        let reach_changed = !v.reachable;
        v.reachable = true;
        v.last_ok = now();
        v.last_error = None;
        v.latency_ms = latency;
        let n0 = now();
        v.pending.retain(|h, (_, ts)| !ours.contains_key(h) && n0 - *ts < 180);
        let mut merged = ours;
        let mut sizes = all_sizes;
        for (h, (t, _)) in &v.pending {
            merged.entry(h.clone()).or_insert_with(|| t.clone());
            sizes.entry(h.clone()).or_insert((f(t, "size"), f(t, "amount_left")));
        }
        v.torrents = merged;
        v.all_sizes = sizes;
        v.server = st.server.clone();
        v.note_uploads(tnow);
        v.speeds.push_back((tnow, f(&st.server, "up_info_speed") + f(&st.server, "dl_info_speed")));
        while v.speeds.front().map(|(t, _)| tnow - t > 1800.0).unwrap_or(false) {
            v.speeds.pop_front();
        }
        v.traffic_used = traffic_used;
        let mut spent = false;
        if bset.traffic.quota_tb > 0.0 {
            let throttle = bset.traffic.throttle_mbit.unwrap_or(10.0) * 1e6 / 8.0;
            let fast = v.speeds.iter().map(|(_, x)| *x).fold(0.0, f64::max) > 2.0 * throttle;
            spent = traffic_used >= bset.traffic.quota_tb * TB && !fast;
        }
        let spent_changed = spent != v.allowance_spent;
        v.allowance_spent = spent;
        let r = room(v, &settings);
        let cap = json!({"placeable_gb": (r.placeable / GB).round(), "headroom": r.headroom, "active": r.active,
                         "spent": spent, "reachable": true});
        (cap, spent_changed, reach_changed, v.clone())
    };

    // box_state: the observation.
    let srv = &st.server;
    app.bus.emit(
        "box_state",
        Some(name),
        None,
        None,
        json!({
            "free_space_on_disk": f(srv, "free_space_on_disk"), "up_info_speed": f(srv, "up_info_speed"),
            "dl_info_speed": f(srv, "dl_info_speed"), "alltime_ul": f(srv, "alltime_ul"), "alltime_dl": f(srv, "alltime_dl"),
            "connection_status": s(srv, "connection_status"), "queued_io_jobs": f(srv, "queued_io_jobs"),
            "average_time_queue": f(srv, "average_time_queue"), "total_peer_connections": f(srv, "total_peer_connections"),
            "read_cache_overload": srv.get("read_cache_overload"), "write_cache_overload": srv.get("write_cache_overload"),
            "dht_nodes": f(srv, "dht_nodes"), "latency_ms": latency, "torrents": view_copy.torrents.len(),
            "traffic_used": traffic_used, "allowance_spent": view_copy.allowance_spent, "psi": view_copy.psi,
            "placeable": capacity["placeable_gb"], "active": capacity["active"],
        }),
    );
    if spent_changed {
        let msg = if view_copy.allowance_spent {
            format!("{name}: traffic {:.2} TB of {:.1} and no faster than its throttle for 30 min — allowance spent, no adds until it resets", traffic_used / TB, bset.traffic.quota_tb)
        } else {
            format!("{name}: allowance not spent (moving faster than its throttle, or reset) — adds resume")
        };
        app.alert("ratio-manager allowance", &msg, "default").await;
    }

    // capacity: a delta only when it changed enough to matter.
    let emit = match &st.last_capacity {
        None => true,
        Some(prev) => {
            reach_changed
                || spent_changed
                || prev["headroom"] != capacity["headroom"]
                || prev["active"] != capacity["active"]
                || (prev["placeable_gb"].as_f64().unwrap_or(0.0) - capacity["placeable_gb"].as_f64().unwrap_or(0.0)).abs() >= 20.0
        }
    };
    if emit {
        app.bus.emit("capacity", Some(name), None, Some("driver"), capacity.clone());
        st.last_capacity = Some(capacity);
    }

    // Leak check on unverified torrents (rule 3), a few per poll.
    let mut checked = 0;
    for t in view_copy.torrents.values() {
        if checked >= 10 || app.halted().is_some() {
            break;
        }
        let h = s(t, "hash");
        if h.is_empty() || app.store.is_verified(name, h) {
            continue;
        }
        checked += 1;
        guard::leak_check(app, q, name, t).await;
    }

    // Race outcomes, written at each milestone and every 10 minutes.
    let n = now();
    for (h, t) in &view_copy.torrents {
        let added = qbit::i(t, "added_on");
        if added <= 0 {
            continue;
        }
        let age = n - added;
        let size = f(t, "size").max(1.0);
        let ratio = f(t, "uploaded") / size;
        let last = st.race_written.get(h).copied().unwrap_or(0);
        let milestone = |secs: i64| age >= secs && last - added < secs;
        let due = last == 0 || n - last >= 600 || milestone(3600) || milestone(3 * 3600) || milestone(86400) || milestone(7 * 86400);
        if !due {
            continue;
        }
        st.race_written.insert(h.clone(), n);
        let existing = app.store.race(name, h);
        let source = existing.as_ref().map(|r| r.source.clone()).filter(|s| !s.is_empty()).unwrap_or_else(|| source_of(t));
        app.store.upsert_race(&Race {
            box_: name.to_string(),
            hash: h.clone(),
            name: s(t, "name").to_string(),
            source,
            size,
            added_at: added,
            completed_at: Some(qbit::i(t, "completion_on")).filter(|c| *c > 0),
            ratio_1h: (age >= 3600).then_some(ratio),
            ratio_3h: (age >= 3 * 3600).then_some(ratio),
            ratio_24h: (age >= 86400).then_some(ratio),
            ratio_7d: (age >= 7 * 86400).then_some(ratio),
            last_ratio: Some(ratio),
            last_uploaded: Some(f(t, "uploaded")),
            ..Default::default()
        });
    }
    st.race_written.retain(|h, _| view_copy.torrents.contains_key(h));

    // Listener repair: firewalled for ten minutes straight on a tunnel box.
    if cfg.tunnel {
        let status = s(&st.server, "connection_status").to_string();
        let since = {
            let mut b = app.boxes.write().unwrap_or_else(|x| x.into_inner());
            let v = b.get_mut(name).unwrap();
            if status == "firewalled" {
                *v.firewalled_since.get_or_insert(n)
            } else {
                v.firewalled_since = None;
                n
            }
        };
        if status == "firewalled" && n - since >= 600 && app.halted().is_none() && !app.dry_run {
            if let Ok(p) = q.preferences().await {
                let port = qbit::i(&p, "listen_port");
                if port > 0 {
                    // Apart: two settings landing together collapse to the
                    // unchanged port, a no-op.
                    let _ = q.set_preferences(&json!({"listen_port": port + 1})).await;
                    tokio::time::sleep(Duration::from_secs(3)).await;
                    let _ = q.set_preferences(&json!({"listen_port": port})).await;
                    app.bus.emit("alert", Some(name), None, Some("driver"), json!({"title": "listener repair", "message": format!("firewalled 10 min: bounced listen port {port}")}));
                    tracing::warn!("[{name}] firewalled 10 min - bounced listen port {port}");
                    if let Ok(mut b) = app.boxes.write() {
                        if let Some(v) = b.get_mut(name) {
                            v.firewalled_since = None;
                        }
                    }
                }
            }
        }
    }
    let _ = exec;
}

fn source_of(t: &Value) -> String {
    let tags = qbit::tags(t);
    for s in ["race", "scout", "handoff", "manual"] {
        if tags.iter().any(|t| t == s) {
            return s.to_string();
        }
    }
    if tags.iter().any(|t| t == "autobrr") {
        return "race".into();
    }
    if tags.iter().any(|t| t == "ratio-picker") {
        return "scout".into();
    }
    "unknown".into()
}

/// The client counter at the start of this traffic period; rebased when the
/// reset day passes.
fn traffic_base(app: &App, name: &str, b: &crate::config::BoxSettings, counter: f64) -> f64 {
    let today = chrono::Utc::now().date_naive();
    let day = b.traffic.reset_day.unwrap_or(1);
    use chrono::Datelike;
    let period = if today.day() >= day {
        format!("{:04}-{:02}", today.year(), today.month())
    } else {
        let prev = today - chrono::Months::new(1);
        format!("{:04}-{:02}", prev.year(), prev.month())
    };
    match app.store.traffic_base(name) {
        Some((p, base)) if p == period => base,
        Some(_) => {
            app.store.set_traffic_base(name, &period, counter);
            tracing::info!("[{name}] traffic period {period}: rebased at {:.2} TB", counter / TB);
            counter
        }
        None => {
            let base = b.traffic.base_tb * TB;
            app.store.set_traffic_base(name, &period, base);
            base
        }
    }
}

// ---------------------------------------------------------------------------
// Re-announcer and unregistered cleanup
// ---------------------------------------------------------------------------

struct Waiting {
    added: i64,
    name: String,
    sent: u32,
    last: f64,
    slow: bool,
}

pub async fn reannouncer(app: Arc<App>, cfg: BoxCfg, exec: Executor, nudges: broadcast::Sender<(String, String)>) {
    let Ok(q) = Qbit::new(&cfg, 15) else { return };
    let name = cfg.name.clone();
    let mut rx = nudges.subscribe();
    let mut waiting: HashMap<String, Waiting> = HashMap::new();
    let mut done: HashMap<String, i64> = HashMap::new();
    let mut nudged: HashSet<String> = HashSet::new();
    let mut first = true;
    let mut backoff = 0.0f64;
    let mut unreg_since: HashMap<String, i64> = HashMap::new();
    let mut last_cleanup = 0i64;
    let mut logged_in = false;
    loop {
        let settings = app.settings();
        let early = settings.reannounce.early.every_s;
        let wait = if backoff > 0.0 { backoff } else { early };
        // Sleep, or wake early for a nudge.
        let sleep = tokio::time::sleep(Duration::from_secs_f64(wait));
        tokio::pin!(sleep);
        loop {
            tokio::select! {
                _ = &mut sleep => break,
                n = rx.recv() => {
                    if let Ok((from, h)) = n {
                        if from != name {
                            nudged.insert(h);
                            break;
                        }
                    }
                }
            }
        }
        if app.halted().is_some() || !cfg.enabled() {
            continue;
        }
        let res: anyhow::Result<()> = async {
            if !logged_in {
                q.login().await?;
                logged_in = true;
            }
            let now_f = nowf();
            let now_i = now();
            let late_window = settings.reannounce.late.until_min.unwrap_or(60.0) * 60.0;
            let early_window = settings.reannounce.early.for_min.unwrap_or(5.0) * 60.0;
            let rows = q.torrents(&format!("?category={CATEGORY}&sort=added_on&reverse=true&limit=60")).await?;
            let fresh: HashMap<String, Value> = rows
                .into_iter()
                .filter(|x| now_f - f(x, "added_on") < late_window)
                .map(|x| (s(&x, "hash").to_string(), x))
                .collect();
            let mut due = Vec::new();
            for (h, x) in &fresh {
                if done.contains_key(h) {
                    continue;
                }
                let age = now_f - f(x, "added_on");
                if registered(&q, x).await {
                    done.insert(h.clone(), qbit::i(x, "added_on"));
                    let w = waiting.remove(h);
                    if !first {
                        let announced = app.announced.read().ok().and_then(|a| a.get(h).copied());
                        let since_announce = announced.map(|a| now_i - a);
                        app.bus.emit(
                            "registration",
                            Some(&name),
                            Some(h),
                            Some("driver"),
                            json!({"after_add_s": age.round(), "after_announce_s": since_announce,
                                   "reannounces": w.as_ref().map(|w| w.sent).unwrap_or(0), "name": s(x, "name")}),
                        );
                        app.store.upsert_race(&Race {
                            box_: name.clone(),
                            hash: h.clone(),
                            added_at: qbit::i(x, "added_on"),
                            registered_after: Some(age),
                            ..Default::default()
                        });
                        let _ = nudges.send((name.clone(), h.clone()));
                    }
                    continue;
                }
                let state = s(x, "state");
                if state.starts_with("stopped") || state.starts_with("paused") || state == "error" || state == "missingFiles" {
                    continue;
                }
                let w = waiting.entry(h.clone()).or_insert_with(|| Waiting {
                    added: qbit::i(x, "added_on"),
                    name: s(x, "name").to_string(),
                    sent: 0,
                    last: 0.0,
                    slow: false,
                });
                let every = if age < early_window && !w.slow { early } else { settings.reannounce.late.every_s };
                if nudged.contains(h) || now_f - w.last >= every - early / 2.0 {
                    due.push(h.clone());
                }
            }
            first = false;
            nudged.clear();
            if !due.is_empty() && !app.dry_run {
                q.reannounce(&due).await?;
                for h in &due {
                    if let Some(w) = waiting.get_mut(h) {
                        w.sent += 1;
                        w.last = now_f;
                        // The tracker's answer, every fifth re-announce.
                        if w.sent % 5 == 1 {
                            let rows = q.trackers(h).await.unwrap_or_default();
                            let msgs: Vec<String> = rows.iter().filter(|r| !s(r, "url").starts_with("** [")).map(|r| s(r, "msg").to_string()).collect();
                            app.bus.emit("tracker_status", Some(&name), Some(h), Some("driver"), json!({"messages": msgs, "reannounces": w.sent}));
                            if !w.slow && msgs.iter().any(|m| slow_down(m)) {
                                w.slow = true;
                                app.bus.emit("alert", Some(&name), Some(h), Some("driver"),
                                    json!({"title": "tracker asks to slow down", "message": msgs.join("; ").chars().take(160).collect::<String>()}));
                            }
                        }
                    }
                }
            }
            waiting.retain(|h, w| {
                let keep = fresh.contains_key(h);
                if !keep {
                    let mins = (now_i - w.added) as f64 / 60.0;
                    tracing::info!("[{name}] {}: not registered after {mins:.0} min ({} re-announces) - cleanup takes it", w.name.chars().take(60).collect::<String>(), w.sent);
                }
                keep
            });
            done.retain(|_, a| (now_i - *a) as f64 <= late_window);

            // Unregistered cleanup, once a minute: every tracker disowns it
            // for delete_after_min -> deleted (TL deleted it; it can neither
            // download nor seed).
            if now_i - last_cleanup >= 60 {
                last_cleanup = now_i;
                let all = q.torrents(&format!("?category={CATEGORY}")).await?;
                let mut seen = HashSet::new();
                for x in &all {
                    let h = s(x, "hash").to_string();
                    // A finished torrent registered once; skip it cheaply.
                    if qbit::finished(x) && !s(x, "tracker").is_empty() {
                        continue;
                    }
                    let rows = q.trackers(&h).await.unwrap_or_default();
                    let real: Vec<&Value> = rows.iter().filter(|r| !s(r, "url").starts_with("** [")).collect();
                    if real.is_empty() || !real.iter().all(|r| guard::unregistered_msg(s(r, "msg"))) {
                        continue;
                    }
                    seen.insert(h.clone());
                    let since = *unreg_since.entry(h.clone()).or_insert(now_i);
                    if (now_i - since) as f64 >= settings.reannounce.delete_after_min * 60.0 {
                        exec.run("driver", Action::Remove { bx: name.clone(), hash: h.clone(), reason: "unregistered".into() }, json!({})).await;
                        unreg_since.remove(&h);
                    } else if !fresh.contains_key(&h) && !app.dry_run {
                        let _ = q.reannounce(std::slice::from_ref(&h)).await;
                    }
                }
                unreg_since.retain(|h, _| seen.contains(h));
            }
            Ok(())
        }
        .await;
        match res {
            Ok(_) => backoff = 0.0,
            Err(e) => {
                logged_in = false;
                let stalled = e.to_string().contains("timed out");
                backoff = if stalled { 0.0 } else { (backoff * 2.0).clamp(30.0, 300.0) };
                tracing::debug!("[{name}] re-announcer: {e}");
            }
        }
    }
}

fn slow_down(m: &str) -> bool {
    static RE: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"(?i)too (fast|often|many|frequent)|slow down|rate.?limit|min.? interval").unwrap());
    RE.is_match(m)
}

/// Registered once a real tracker row reports Working (2). An empty
/// `tracker` rules it out cheaply; a non-empty one proves nothing (5.1+ names
/// a tracker there while it refuses the torrent).
async fn registered(q: &Qbit, x: &Value) -> bool {
    if s(x, "tracker").is_empty() {
        return false;
    }
    q.trackers(s(x, "hash"))
        .await
        .map(|rows| rows.iter().any(|r| !s(r, "url").starts_with("** [") && qbit::i(r, "status") == 2))
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Reconciler
// ---------------------------------------------------------------------------

pub async fn reconciler(app: Arc<App>, cfg: BoxCfg) {
    let Ok(q) = Qbit::new(&cfg, 60) else { return };
    let name = cfg.name.clone();
    let mut warned_ip: Option<Vec<String>> = None;
    loop {
        if cfg.enabled() {
            if let Err(e) = reconcile_once(&app, &cfg, &q, &mut warned_ip).await {
                tracing::warn!("[{name}] reconcile: {e}");
            }
        }
        tokio::time::sleep(Duration::from_secs(300)).await;
    }
}

pub async fn reconcile_once(app: &Arc<App>, cfg: &BoxCfg, q: &Qbit, warned_ip: &mut Option<Vec<String>>) -> anyhow::Result<()> {
    let name = cfg.name.as_str();
    // Addresses: a seedbox's, for bans and rule 8. Home's VPN exit changes.
    if cfg.is_seedbox() || !cfg.peer_ips.is_empty() {
        let ips = guard::resolve_ips(if cfg.is_seedbox() { cfg.host() } else { None }, &cfg.peer_ips).await;
        if let Ok(mut b) = app.boxes.write() {
            let v = b.entry(name.to_string()).or_insert_with(|| BoxView { name: name.to_string(), ..Default::default() });
            if !ips.is_empty() {
                v.ips = ips.clone();
            }
        }
        if let Some(reg) = &cfg.tl_registered_ip {
            if !ips.is_empty() && !ips.contains(reg) && warned_ip.as_ref() != Some(&ips) {
                *warned_ip = Some(ips.clone());
                app.alert("ratio-manager rule 8", &format!("{name} resolves to {ips:?}, not its TL-registered {reg}: no adds there until the registry or the TL profile is updated"), "high").await;
            }
        }
    }
    q.login().await?;
    let settings = app.settings();
    let b = settings.box_settings(name);
    let mut changes: Vec<String> = Vec::new();

    // The category, at its save path where one is set.
    let cats = q.categories().await?;
    let have = cats.get(CATEGORY);
    match (have, &cfg.save_path) {
        (None, p) => {
            if !app.dry_run {
                q.post("torrents/createCategory", &[("category", CATEGORY.to_string()), ("savePath", p.clone().unwrap_or_default())]).await?;
            }
            changes.push(format!("created category {CATEGORY}"));
        }
        (Some(c), Some(p)) if s(c, "savePath").trim_end_matches('/') != p.trim_end_matches('/') => {
            if !app.dry_run {
                q.post("torrents/editCategory", &[("category", CATEGORY.to_string()), ("savePath", p.clone())]).await?;
            }
            changes.push(format!("category {CATEGORY} -> {p}"));
        }
        _ => {}
    }

    let prefs = q.preferences().await?;
    let mut want = Map::new();
    if cfg.manage_prefs {
        if cfg.private_only {
            want.insert("dht".into(), json!(false));
            want.insert("pex".into(), json!(false));
            want.insert("lsd".into(), json!(false));
        }
        want.insert("dl_limit".into(), json!((b.download_cap_mib * MIB) as i64));
        if let Some(t) = b.async_io_threads {
            want.insert("async_io_threads".into(), json!(t));
        }
        if let Some(c) = b.disk_cache_mib {
            want.insert("disk_cache".into(), json!(c));
        }
        // A race is won by serving every leecher at once.
        for k in ["max_uploads", "max_uploads_per_torrent", "max_connec", "max_connec_per_torrent"] {
            want.insert(k.into(), json!(-1));
        }
        want.insert("queueing_enabled".into(), json!(false));
    }
    // Bans: every other box's addresses (rule 2).
    let ours = app.our_ips(Some(name));
    let banned: Vec<String> = s(&prefs, "banned_IPs").lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect();
    let mut add: Vec<String> = ours.keys().filter(|ip| !banned.contains(ip)).cloned().collect();
    add.sort();
    if !add.is_empty() {
        let mut all = banned.clone();
        all.extend(add.iter().cloned());
        want.insert("banned_IPs".into(), json!(all.join("\n")));
        changes.push(format!("banned {add:?} (our clients never trade)"));
    }
    let drift: Map<String, Value> = want.into_iter().filter(|(k, v)| prefs.get(k) != Some(v)).collect();
    if !drift.is_empty() {
        for (k, v) in &drift {
            if k != "banned_IPs" {
                changes.push(format!("{k} {} -> {v}", prefs.get(k).cloned().unwrap_or(Value::Null)));
            }
        }
        if !app.dry_run {
            q.set_preferences(&Value::Object(drift)).await?;
        }
    }
    if !changes.is_empty() {
        tracing::info!("[{name}] reconciled: {}", changes.join(", "));
        app.bus.emit("setting", Some(name), None, Some("reconciler"), json!({"applied": changes, "dry_run": app.dry_run}));
    }
    // A private-only box must hold DHT/PeX/LSD off client-wide (rule 3).
    if cfg.private_only && !app.dry_run {
        let p = q.preferences().await?;
        let on: Vec<&str> = ["dht", "pex", "lsd"].into_iter().filter(|k| p.get(*k).and_then(|v| v.as_bool()).unwrap_or(false)).collect();
        if !on.is_empty() {
            app.set_halt(format!("[{name}] {on:?} still on client-wide after setting them off — halted")).await;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Wave 2: peer snapshots and host pressure
// ---------------------------------------------------------------------------

pub fn peer_salt(app: &App) -> Vec<u8> {
    if let Some(s) = app.store.get("peer_salt") {
        if let Ok(b) = hex::decode(s) {
            return b;
        }
    }
    let salt: [u8; 32] = rand::random();
    app.store.set("peer_salt", &hex::encode(salt));
    salt.to_vec()
}

pub fn hash_ip(salt: &[u8], ip: &str) -> String {
    let mut m = <Hmac<Sha256> as Mac>::new_from_slice(salt).expect("hmac key");
    m.update(ip.as_bytes());
    hex::encode(&m.finalize().into_bytes()[..12])
}

pub async fn peer_snapshots(app: Arc<App>, cfg: BoxCfg) {
    let Ok(q) = Qbit::new(&cfg, 20) else { return };
    let salt = peer_salt(&app);
    let name = cfg.name.clone();
    let mut logged_in = false;
    loop {
        tokio::time::sleep(Duration::from_secs(30)).await;
        if !cfg.enabled() {
            continue;
        }
        let Some(view) = app.view(&name) else { continue };
        if !view.reachable {
            continue;
        }
        let n = now();
        let mut young: Vec<&Value> = view.torrents.values().filter(|t| n - qbit::i(t, "added_on") < 7200).collect();
        young.sort_by_key(|t| -qbit::i(t, "added_on"));
        for t in young.into_iter().take(8) {
            if !logged_in {
                if q.login().await.is_err() {
                    break;
                }
                logged_in = true;
            }
            let h = s(t, "hash");
            let Ok(p) = q.peers(h).await else {
                logged_in = false;
                break;
            };
            let peers: Vec<Value> = p
                .get("peers")
                .and_then(|v| v.as_object())
                .map(|m| {
                    m.values()
                        .map(|pp| {
                            json!({"ip": hash_ip(&salt, s(pp, "ip")), "client": s(pp, "client"),
                                   "progress": f(pp, "progress"), "dl_speed": f(pp, "dl_speed"), "up_speed": f(pp, "up_speed"),
                                   "downloaded": f(pp, "downloaded"), "uploaded": f(pp, "uploaded"),
                                   "flags": s(pp, "flags"), "connection": s(pp, "connection"), "country": s(pp, "country_code")})
                        })
                        .collect()
                })
                .unwrap_or_default();
            app.bus.emit("peers", Some(&name), Some(h), None, json!({"n": peers.len(), "peers": peers,
                "progress": f(t, "progress"), "ratio": f(t, "ratio")}));
        }
    }
}

/// /proc/pressure over SSH (sshpass with the box's login), once a minute.
pub async fn host_pressure(app: Arc<App>, cfg: BoxCfg) {
    let Some(ssh) = cfg.ssh.clone() else { return };
    let name = cfg.name.clone();
    loop {
        tokio::time::sleep(Duration::from_secs(60)).await;
        if !cfg.enabled() {
            continue;
        }
        let out = tokio::time::timeout(
            Duration::from_secs(30),
            tokio::process::Command::new("sshpass")
                .arg("-e")
                .arg("ssh")
                .args(["-o", "StrictHostKeyChecking=accept-new", "-o", "UserKnownHostsFile=/data/known_hosts",
                       "-o", "ConnectTimeout=15", "-o", "BatchMode=no", "-p", &ssh.port.to_string()])
                .arg(format!("{}@{}", cfg.user(), ssh.host))
                .arg("cat /proc/pressure/io /proc/pressure/cpu /proc/pressure/memory")
                .env("SSHPASS", cfg.password())
                .kill_on_drop(true)
                .output(),
        )
        .await;
        let Ok(Ok(out)) = out else { continue };
        if !out.status.success() {
            tracing::debug!("[{name}] psi: ssh exited {}", out.status);
            continue;
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let psi = parse_psi(&text);
        if let Ok(mut b) = app.boxes.write() {
            if let Some(v) = b.get_mut(&name) {
                v.psi = Some(psi);
            }
        }
    }
}

/// `some avg10=.. avg60=.. avg300=.. total=..` / `full ...`, three files in
/// order io, cpu, memory.
pub fn parse_psi(text: &str) -> Value {
    let mut out = Map::new();
    let names = ["io", "cpu", "memory"];
    let mut idx = 0usize;
    let mut cur = Map::new();
    let mut seen_some = false;
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let Some(kind) = parts.next() else { continue };
        if kind == "some" && seen_some {
            out.insert(names.get(idx).unwrap_or(&"x").to_string(), Value::Object(std::mem::take(&mut cur)));
            idx += 1;
        }
        if kind == "some" {
            seen_some = true;
        }
        for p in parts {
            if let Some((k, v)) = p.split_once('=') {
                if k == "avg60" || k == "avg10" {
                    cur.insert(format!("{kind}_{k}"), json!(v.parse::<f64>().unwrap_or(0.0)));
                }
            }
        }
    }
    if !cur.is_empty() {
        out.insert(names.get(idx).unwrap_or(&"x").to_string(), Value::Object(cur));
    }
    Value::Object(out)
}

// ---------------------------------------------------------------------------
// Rollups
// ---------------------------------------------------------------------------

pub async fn rollups(app: Arc<App>) {
    let mut tick = tokio::time::interval(Duration::from_secs(60));
    loop {
        tick.tick().await;
        let settings = app.settings();
        let minute = now() / 60;
        for v in app.views() {
            if !v.reachable {
                continue;
            }
            let r = room(&v, &settings);
            let b = settings.box_settings(&v.name);
            let up: f64 = v.torrents.values().map(|t| f(t, "upspeed")).sum();
            let dl: f64 = v.torrents.values().map(|t| f(t, "dlspeed")).sum();
            let n_seed = v.torrents.values().filter(|t| qbit::finished(t)).count() as i64;
            let uploaded: f64 = v.torrents.values().map(|t| f(t, "uploaded")).sum();
            let leechers: i64 = v.torrents.values().map(|t| qbit::i(t, "num_leechs")).sum();
            app.store.rollup(minute, &v.name, up, dl, r.active as i64, n_seed, v.torrents.len() as i64, r.used,
                             r.quota, v.traffic_used, b.traffic.quota_tb * TB, uploaded, leechers);
        }
        if minute % 60 == 0 {
            app.store.prune_rollups(minute - 40 * 24 * 60);
            app.store.prune_decisions(now() - 40 * 86400);
            app.store.prune_releases(now() - 14 * 86400);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn psi() {
        let t = "some avg10=1.00 avg60=2.00 avg300=3.00 total=1\nfull avg10=0.50 avg60=0.70 avg300=1 total=2\nsome avg10=4.00 avg60=5.00 avg300=6 total=3\nfull avg10=0 avg60=0 avg300=0 total=0\nsome avg10=7 avg60=8 avg300=9 total=1\nfull avg10=0 avg60=0 avg300=0 total=0\n";
        let v = super::parse_psi(t);
        assert_eq!(v["io"]["some_avg60"], 2.0);
        assert_eq!(v["io"]["full_avg60"], 0.7);
        assert_eq!(v["cpu"]["some_avg60"], 5.0);
        assert_eq!(v["memory"]["some_avg60"], 8.0);
    }

    #[test]
    fn scrub_drops_passkey_fields() {
        let v = serde_json::json!({"tracker": "https://x/a/KEY/announce", "magnet_uri": "magnet:?tr=KEY", "name": "n"});
        let s = super::scrub(&v);
        assert!(s.get("tracker").is_none() && s.get("magnet_uri").is_none());
        assert_eq!(s["name"], "n");
    }
}
