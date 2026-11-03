//! The baseline policy. It wakes only on deltas — `announce` on the fast
//! lane; `market`, `capacity`, `tick` and `setting` on the coalescing normal
//! lane — and every choice is an `/actions` call the guard checks.
//!
//! One scoring rule for every source: a candidate's estimate is the ratio it
//! would earn on a box over the horizon (per-box tables rebuilt nightly from
//! the race outcomes, seeded with priors), times the box's bias for its
//! source. A holding's estimate comes from what it uploaded in the last 30
//! minutes. A box with room adds anything worth `min_ratio`; a full box
//! evicts its weakest holdings only for a candidate `swap_margin` better.

use crate::actions::{Action, Executor};
use crate::bencode::TorrentFacts;
use crate::config::{Settings, GB, MIB};
use crate::events::Event;
use crate::guard;
use crate::qbit::{self, f, s};
use crate::scout::{intake_ok, Scout};
use crate::state::{room, App, BoxView};
use crate::store::Release;
use rand::Rng;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

pub struct Baseline {
    app: Arc<App>,
    exec: Executor,
    scout: Scout,
    no_handoff: std::sync::Mutex<HashSet<String>>,
    tables_day: std::sync::Mutex<String>,
}

fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

pub fn size_bucket(size: f64) -> &'static str {
    match size / GB {
        x if x < 5.0 => "lt5",
        x if x < 20.0 => "5-20",
        x if x < 60.0 => "20-60",
        _ => "60+",
    }
}

pub fn demand_bucket(d: Option<f64>) -> &'static str {
    match d {
        None => "none",
        Some(x) if x < 1.0 => "lt1",
        Some(x) if x < 2.0 => "1-2",
        Some(x) if x < 5.0 => "2-5",
        Some(_) => "5+",
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Plan {
    #[serde(rename = "box")]
    pub bx: String,
    pub est: f64,
    pub basis: String,
    pub eligible: bool,
    pub why: Vec<String>,
    pub evict: Vec<String>,
    pub placeable_gb: f64,
}

impl Baseline {
    pub fn new(app: Arc<App>, exec: Executor, scout: Scout) -> Baseline {
        Baseline { app, exec, scout, no_handoff: Default::default(), tables_day: Default::default() }
    }

    pub async fn run(self, mut fast: mpsc::UnboundedReceiver<Event>) {
        // Act only once every enabled box has been seen (or two minutes passed):
        // a decision on an empty view would double-place.
        let t0 = now();
        loop {
            let enabled: Vec<String> = self.app.registry.boxes.iter().filter(|b| b.enabled()).map(|b| b.name.clone()).collect();
            let seen = enabled.iter().all(|n| self.app.view(n).map(|v| v.last_ok > 0).unwrap_or(false));
            if seen || now() - t0 > 120 {
                break;
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
        self.rebuild_tables();
        tracing::info!("baseline: acting");
        let me = Arc::new(self);
        // The fast lane never waits behind the normal one: every announce is
        // decided in its own task at once.
        let fast_me = me.clone();
        tokio::spawn(async move {
            while let Some(e) = fast.recv().await {
                let m = fast_me.clone();
                tokio::spawn(async move { m.on_announce(e).await });
            }
        });
        let normal = me.app.bus.normal.clone();
        loop {
            let e = normal.pop().await;
            match e.kind.as_str() {
                "market" => me.on_market(e).await,
                "capacity" => me.on_capacity(e).await,
                "tick" => me.on_tick().await,
                "setting" => me.on_capacity_all().await,
                _ => {}
            }
        }
    }

    // -- estimates ------------------------------------------------------------

    pub fn estimate(&self, settings: &Settings, bx: &str, source: &str, size: f64, demand: Option<f64>) -> (f64, String) {
        let bias = settings.bias(bx, source);
        if bias <= 0.0 {
            return (0.0, "bias 0".into());
        }
        let tables = self.app.tables.read().map(|t| t.clone()).unwrap_or_default();
        let min_n = settings.scoring.min_samples;
        let k1 = format!("{bx}|{source}|{}|{}", size_bucket(size), demand_bucket(demand));
        let k2 = format!("{bx}|{source}|*|*");
        let (base, basis) = match (tables.get(&k1), tables.get(&k2)) {
            (Some((m, n)), _) if *n >= min_n => (*m, format!("table {k1} (n={n})")),
            (_, Some((m, n))) if *n >= min_n => {
                let adj = if source == "scout" { demand.map(|d| d.clamp(0.25, 3.0)).unwrap_or(1.0) } else { 1.0 };
                (m * adj, format!("table {k2} (n={n}) x demand {adj:.2}"))
            }
            _ => {
                let p = settings.prior(bx, source);
                let adj = if source == "scout" { demand.map(|d| d.clamp(0.25, 3.0)).unwrap_or(1.0) } else { 1.0 };
                (p * adj, format!("prior {p} x demand {adj:.2}"))
            }
        };
        (base * bias, basis)
    }

    fn holding_est(&self, settings: &Settings, view: &BoxView, t: &Value) -> f64 {
        let size = f(t, "size").max(1.0);
        let rate = view.recent_upload(s(t, "hash"));
        rate * settings.horizon_h(&view.name) * 3600.0 * settings.scoring.holding_decay / size
    }

    /// The holdings to evict for `need` bytes, weakest first, each beaten by
    /// `cand_est` by the swap margin; None when that cannot free enough.
    fn evictions(&self, settings: &Settings, view: &BoxView, need: f64, cand_est: f64) -> Option<Vec<String>> {
        if need <= 0.0 {
            return Some(vec![]);
        }
        if !settings.box_settings(&view.name).evict {
            return None;
        }
        let margin = settings.swap_margin(&view.name);
        let keep = settings.keep_above(&view.name);
        let mut c: Vec<(f64, f64, String)> = view
            .torrents
            .values()
            .filter(|t| qbit::finished(t))
            .filter(|t| view.recent_upload(s(t, "hash")) < keep)
            .map(|t| (self.holding_est(settings, view, t), f(t, "size"), s(t, "hash").to_string()))
            .filter(|(est, _, _)| est * margin < cand_est)
            .filter(|(_, _, h)| guard::may_remove(&self.app, settings, view, h, false).is_some())
            .collect();
        c.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)));
        let mut freed = 0.0;
        let mut out = vec![];
        for (_, size, h) in c {
            if freed >= need {
                break;
            }
            freed += size;
            out.push(h);
        }
        (freed >= need).then_some(out)
    }

    /// One box's plan for one candidate.
    fn plan(&self, settings: &Settings, bx: &str, source: &str, size: f64, demand: Option<f64>, hash: &str) -> Plan {
        let (est, basis) = self.estimate(settings, bx, source, size, demand);
        let mut why = Vec::new();
        let mut evict = vec![];
        let mut placeable = 0.0;
        let cfg = self.app.box_cfg(bx);
        let view = self.app.view(bx);
        if !settings.baseline_acts(bx) {
            why.push(format!("policy is {}", settings.policy.get(bx).cloned().unwrap_or_default()));
        }
        match (cfg, &view) {
            (Some(c), Some(v)) if c.enabled() && v.reachable => {
                let r = room(v, settings);
                placeable = r.placeable;
                if v.torrents.contains_key(hash) {
                    why.push("already holds it".into());
                }
                if v.allowance_spent {
                    why.push("allowance spent".into());
                }
                if !r.headroom {
                    why.push(format!("busy: {} downloading at {:.0} MiB/s", r.active, r.dl_speed / MIB));
                }
                if est < settings.min_ratio(bx) {
                    why.push(format!("estimate {est:.2} under min_ratio"));
                }
                let cap = settings.box_settings(bx).download_cap_mib * MIB;
                if cap > 0.0 && size / cap / 3600.0 > settings.max_download_h(bx) {
                    why.push(format!("{:.1} h to download at its cap", size / cap / 3600.0));
                }
                if why.is_empty() && size > r.placeable {
                    match self.evictions(settings, v, size - r.placeable, est) {
                        Some(e) => evict = e,
                        None => why.push(format!("no room: {:.1} GB placeable, nothing weaker to evict", r.placeable.max(0.0) / GB)),
                    }
                }
            }
            (Some(c), _) if !c.enabled() => why.push("login is a placeholder".into()),
            _ => why.push("unreachable".into()),
        }
        Plan { bx: bx.to_string(), est, basis, eligible: why.is_empty(), why, evict, placeable_gb: (placeable / GB * 10.0).round() / 10.0 }
    }

    fn box_names(&self) -> Vec<String> {
        self.app.registry.boxes.iter().map(|b| b.name.clone()).collect()
    }

    async fn carry_out(&self, plan: &Plan, add: Action, why: &str) -> bool {
        for h in &plan.evict {
            let out = self.exec.run("baseline", Action::Remove { bx: plan.bx.clone(), hash: h.clone(), reason: format!("evicted for {why}") }, json!({})).await;
            if !out.executed() {
                return false;
            }
        }
        self.exec.run("baseline", add, json!({"est": plan.est, "basis": plan.basis})).await.executed()
    }

    fn record(&self, trigger: &str, hash: Option<&str>, name: &str, plans: &[Plan], chosen: &[String], extra: Value) {
        let e = self.app.bus.emit(
            "decision",
            None,
            hash,
            Some("baseline"),
            json!({"trigger": trigger, "name": name, "plans": plans, "chosen": chosen, "extra": extra}),
        );
        self.app.store.decision(e.seq, now(), "baseline", trigger, hash, name, &chosen.join(","), &json!({"plans": plans, "extra": e.data["extra"]}));
    }

    // -- announce (fast lane) --------------------------------------------------

    async fn on_announce(&self, e: Event) {
        let settings = self.app.settings();
        let d = &e.data;
        let tl_id = d.get("tl_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let name = d.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let category = d.get("category").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let freeleech = d.get("freeleech").and_then(|v| v.as_bool()).unwrap_or(false);
        if tl_id.is_empty() {
            return;
        }
        // Announces older than a minute (a backlog after a restart) are not races.
        if now() - e.ts.timestamp() > 60 {
            return;
        }
        if let Err(why) = intake_ok(&settings, &name, &category, freeleech, None) {
            if freeleech {
                self.record("announce", None, &name, &[], &[], json!({"tl_id": tl_id, "skip": why}));
            }
            return;
        }
        if self.app.halted().is_some() {
            self.record("announce", None, &name, &[], &[], json!({"tl_id": tl_id, "skip": "halted"}));
            return;
        }
        let facts: TorrentFacts = match self.exec.torrent(Some(&tl_id), None, None, true).await {
            Ok((_, f)) => f,
            Err(err) => {
                self.record("announce", None, &name, &[], &[], json!({"tl_id": tl_id, "skip": format!("no .torrent: {err}")}));
                return;
            }
        };
        self.app.store.set_release_facts(&tl_id, &facts.hash, facts.private, facts.size);
        if let Ok(mut a) = self.app.announced.write() {
            a.insert(facts.hash.clone(), e.ts.timestamp());
            if a.len() > 5000 {
                let cutoff = now() - 86400;
                a.retain(|_, t| *t > cutoff);
            }
        }
        if let Err(why) = intake_ok(&settings, &name, &category, freeleech, Some(facts.size)) {
            self.record("announce", Some(&facts.hash), &name, &[], &[], json!({"tl_id": tl_id, "skip": why}));
            return;
        }
        if !facts.private {
            self.record("announce", Some(&facts.hash), &name, &[], &[], json!({"tl_id": tl_id, "skip": "no private flag"}));
            self.app.alert("ratio-manager refused a torrent", &format!("{name}: no private flag"), "high").await;
            return;
        }
        let plans: Vec<Plan> = self.box_names().iter().map(|bx| self.plan(&settings, bx, "race", facts.size, None, &facts.hash)).collect();
        let winners: Vec<Plan> = plans.iter().filter(|p| p.eligible).cloned().collect();
        let chosen: Vec<String> = winners.iter().map(|p| p.bx.clone()).collect();
        // The join-delay experiment: one of several winners joins late.
        let mut delays: HashMap<String, f64> = HashMap::new();
        let mut experiment = Value::Null;
        let jd = &settings.experiments.join_delay;
        if winners.len() >= 2 && jd.share > 0.0 && !jd.delays_s.is_empty() {
            let mut rng = rand::thread_rng();
            if rng.gen::<f64>() < jd.share {
                let which = rng.gen_range(0..winners.len());
                let delay = jd.delays_s[rng.gen_range(0..jd.delays_s.len())];
                delays.insert(winners[which].bx.clone(), delay);
                experiment = json!({"join_delay": {"box": winners[which].bx, "delay_s": delay,
                    "p_slice": jd.share, "p_box": 1.0 / winners.len() as f64, "p_delay": 1.0 / jd.delays_s.len() as f64}});
            }
        }
        self.record("announce", Some(&facts.hash), &name, &plans, &chosen, json!({"tl_id": tl_id, "size": facts.size, "experiment": experiment}));
        let me: &Baseline = self;
        let name_ref: &str = &name;
        let futs = winners.iter().map(|p| {
            let add = Action::Add {
                bx: p.bx.clone(),
                tl_id: Some(tl_id.clone()),
                from_box: None,
                hash: Some(facts.hash.clone()),
                source: "race".into(),
                delay_s: delays.get(&p.bx).copied().unwrap_or(0.0),
                reason: format!("race, estimate {:.2}", p.est),
            };
            async move { me.carry_out(p, add, name_ref).await }
        });
        futures::future::join_all(futs).await;
    }

    // -- market and capacity (normal lane) -------------------------------------

    async fn on_market(&self, e: Event) {
        let settings = self.app.settings();
        let Some(hash) = e.hash.clone() else { return };
        let tl_id = e.data.get("tl_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let Some(r) = self.app.store.release(&tl_id) else { return };
        if let Some(why) = self.scout_skip(&settings, &r) {
            let _ = why;
            return;
        }
        let demand = r.leechers.map(|l| l as f64 / (r.seeders.unwrap_or(0) as f64 + 1.0));
        let mut plans: Vec<Plan> = self.box_names().iter().map(|bx| self.plan(&settings, bx, "scout", r.size, demand, &hash)).collect();
        plans.sort_by(|a, b| b.est.partial_cmp(&a.est).unwrap_or(std::cmp::Ordering::Equal));
        // A scout pick goes to one box: two of ours would split a small swarm.
        let best = plans.iter().find(|p| p.eligible).cloned();
        let chosen: Vec<String> = best.iter().map(|p| p.bx.clone()).collect();
        let any_eligible_or_close = plans.iter().any(|p| p.est >= settings.scoring.min_ratio);
        if any_eligible_or_close {
            self.record("market", Some(&hash), &r.name, &plans, &chosen, json!({"tl_id": tl_id, "demand": demand, "reason": e.data.get("reason")}));
        }
        if let Some(p) = best {
            let add = Action::Add { bx: p.bx.clone(), tl_id: Some(tl_id.clone()), from_box: None, hash: Some(hash.clone()), source: "scout".into(), delay_s: 0.0, reason: format!("scout pick, estimate {:.2}", p.est) };
            self.carry_out(&p, add, &r.name).await;
        }
    }

    /// Why a release is not a scout candidate right now, if it isn't.
    fn scout_skip(&self, settings: &Settings, r: &Release) -> Option<String> {
        let Some(h) = r.hash.as_deref() else { return Some("no .torrent facts yet".into()) };
        if r.private != Some(true) {
            return Some("not private".into());
        }
        if let Err(w) = intake_ok(settings, &r.name, &r.category, r.freeleech, Some(r.size)) {
            return Some(w);
        }
        if self.app.held_anywhere().contains_key(h) {
            return Some("held".into());
        }
        if r.seeders.unwrap_or(0) < settings.scoring.min_seeders {
            return Some("no seeder".into());
        }
        if self.app.store.recently_removed(h, now() - (settings.scoring.readd_after_h * 3600.0) as i64) {
            return Some("evicted recently".into());
        }
        None
    }

    async fn on_capacity(&self, e: Event) {
        let Some(bx) = e.bx.clone() else { return };
        self.fill(&bx, 3).await;
    }

    async fn on_capacity_all(&self) {
        for bx in self.box_names() {
            self.fill(&bx, 1).await;
        }
    }

    /// Fill a box's room from the top of its scout candidates.
    async fn fill(&self, bx: &str, max_adds: usize) {
        let settings = self.app.settings();
        if settings.bias(bx, "scout") <= 0.0 || !settings.baseline_acts(bx) || self.app.halted().is_some() {
            return;
        }
        let Some(view) = self.app.view(bx) else { return };
        if !view.reachable || room(&view, &settings).placeable < 5.0 * GB {
            return;
        }
        let book = self.app.store.releases_since(now() - (settings.scout.refresh.track_hours * 3600.0) as i64);
        let mut cands: Vec<(f64, Release)> = book
            .into_iter()
            .filter(|r| self.scout_skip(&settings, r).is_none())
            .map(|r| {
                let demand = r.leechers.map(|l| l as f64 / (r.seeders.unwrap_or(0) as f64 + 1.0));
                (self.estimate(&settings, bx, "scout", r.size, demand).0, r)
            })
            .collect();
        cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut added = 0;
        for (_, r) in cands.into_iter().take(20) {
            if added >= max_adds {
                break;
            }
            let h = r.hash.clone().unwrap_or_default();
            let demand = r.leechers.map(|l| l as f64 / (r.seeders.unwrap_or(0) as f64 + 1.0));
            let p = self.plan(&settings, bx, "scout", r.size, demand, &h);
            if !p.eligible || !p.evict.is_empty() {
                continue;
            }
            self.record("capacity", Some(&h), &r.name, std::slice::from_ref(&p), &[bx.to_string()], json!({"tl_id": r.tl_id, "demand": demand}));
            let add = Action::Add { bx: bx.to_string(), tl_id: Some(r.tl_id.clone()), from_box: None, hash: Some(h), source: "scout".into(), delay_s: 0.0, reason: format!("fills room, estimate {:.2}", p.est) };
            if self.carry_out(&p, add, &r.name).await {
                added += 1;
            }
        }
    }

    // -- tick ------------------------------------------------------------------

    async fn on_tick(&self) {
        let settings = self.app.settings();
        let day = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let rebuild = {
            let mut d = self.tables_day.lock().unwrap_or_else(|x| x.into_inner());
            let due = chrono::Utc::now().format("%H").to_string() == "03" && *d != day;
            if due {
                *d = day;
            }
            due
        };
        if rebuild {
            self.rebuild_tables();
        }
        if self.app.halted().is_some() {
            return;
        }
        for bx in self.box_names() {
            if settings.baseline_acts(&bx) && settings.box_settings(&bx).focus_downloads.is_some() {
                self.focus(&settings, &bx).await;
            }
        }
        if !settings.vip_now() {
            self.hand_off(&settings).await;
        }
        self.refresh_candidates(&settings);
    }

    /// Home's download: full speed for the hottest few and every hand-off;
    /// the rest trickle.
    async fn focus(&self, settings: &Settings, bx: &str) {
        let Some(view) = self.app.view(bx) else { return };
        if !view.reachable {
            return;
        }
        let b = settings.box_settings(bx);
        let n = b.focus_downloads.unwrap_or(3);
        let trickle = (b.trickle_kib.unwrap_or(256.0) * 1024.0) as i64;
        let dl: Vec<&Value> = view.torrents.values().filter(|t| !qbit::finished(t)).collect();
        let hand: Vec<&Value> = dl.iter().copied().filter(|t| qbit::tags(t).iter().any(|x| x == "handoff")).collect();
        let mut races: Vec<&Value> = dl.iter().copied().filter(|t| !qbit::tags(t).iter().any(|x| x == "handoff")).collect();
        races.sort_by(|a, b| qbit::swarm_demand(b).partial_cmp(&qbit::swarm_demand(a)).unwrap_or(std::cmp::Ordering::Equal));
        let mut want: HashMap<String, i64> = HashMap::new();
        for t in hand.iter().chain(races.iter().take(n)) {
            want.insert(s(t, "hash").to_string(), 0);
        }
        for t in races.iter().skip(n) {
            want.insert(s(t, "hash").to_string(), trickle);
        }
        let mut by_limit: HashMap<i64, Vec<String>> = HashMap::new();
        for t in &dl {
            let h = s(t, "hash").to_string();
            let cur = qbit::i(t, "dl_limit").max(0);
            if let Some(w) = want.get(&h) {
                if *w != cur {
                    by_limit.entry(*w).or_default().push(h);
                }
            }
        }
        for (limit, hashes) in by_limit {
            self.exec.run("baseline", Action::SetLimits { bx: bx.to_string(), hashes, dl_limit: limit, reason: "download focus".into() }, json!({})).await;
        }
    }

    /// Outside VIP: a racer box short of room hands its quietest torrents
    /// still under the hit-and-run minimum to a keeper box, which re-downloads
    /// them from the swarm while they are still freeleech.
    async fn hand_off(&self, settings: &Settings) {
        let keeper = self.box_names().into_iter().find(|b| settings.bias(b, "race") <= 0.0 && settings.box_settings(b).focus_downloads.is_some());
        let Some(keeper) = keeper else { return };
        let Some(kview) = self.app.view(&keeper) else { return };
        if !kview.reachable {
            return;
        }
        let active = kview.torrents.values().filter(|t| !qbit::finished(t) && qbit::tags(t).iter().any(|x| x == "handoff")).count();
        let mut budget = settings.handoff.max_active.saturating_sub(active).min(settings.handoff.per_tick);
        if budget == 0 {
            return;
        }
        let mut kspace = room(&kview, settings).placeable;
        let indexer = self.app.prowlarr.indexer(&self.app.registry.prowlarr.indexers).await.ok().flatten();
        for src in self.box_names() {
            if src == keeper || settings.bias(&src, "race") <= 0.0 {
                continue;
            }
            let Some(v) = self.app.view(&src) else { continue };
            if room(&v, settings).placeable >= settings.handoff.need_gb * GB {
                continue;
            }
            let mut cands: Vec<&Value> = v
                .torrents
                .values()
                .filter(|t| qbit::finished(t))
                .filter(|t| !kview.torrents.contains_key(s(t, "hash")) && !self.no_handoff.lock().map(|n| n.contains(s(t, "hash"))).unwrap_or(false))
                .filter(|t| guard::may_remove(&self.app, settings, &v, s(t, "hash"), false).is_none())
                .collect();
            cands.sort_by(|a, b| qbit::swarm_demand(a).partial_cmp(&qbit::swarm_demand(b)).unwrap_or(std::cmp::Ordering::Equal));
            for t in cands {
                if budget == 0 {
                    return;
                }
                let h = s(t, "hash").to_string();
                let size = f(t, "total_size").max(f(t, "size"));
                if size > kspace {
                    continue;
                }
                // Still freeleech: a listed release of exactly its size.
                let Some((iid, _)) = &indexer else { return };
                let q = s(t, "name").replace(['.', '_'], " ");
                let free = match self.app.prowlarr.search(*iid, &q).await {
                    Ok(rows) => rows.iter().any(|r| r.get("size").and_then(|x| x.as_f64()) == Some(size)
                        && r.get("indexerFlags").and_then(|x| x.as_array()).map(|a| a.iter().any(|y| y.as_str() == Some("freeleech"))).unwrap_or(false)),
                    Err(_) => false,
                };
                if !free {
                    if let Ok(mut n) = self.no_handoff.lock() {
                        n.insert(h);
                    }
                    continue;
                }
                // Record the confirmation where the guard reads it.
                let rel_id = format!("hash:{h}");
                self.app.store.note_release(&rel_id, s(t, "name"), "", size, Some(true), None, "handoff");
                self.app.store.set_release_facts(&rel_id, &h, true, size);
                let add = Action::Add { bx: keeper.clone(), tl_id: None, from_box: Some(src.clone()), hash: Some(h.clone()), source: "handoff".into(), delay_s: 0.0, reason: format!("hand-off from {src}") };
                let out = self.exec.run("baseline", add, json!({})).await;
                if out.executed() {
                    budget -= 1;
                    kspace -= size;
                } else if let Ok(mut n) = self.no_handoff.lock() {
                    n.insert(h);
                }
            }
        }
    }

    fn refresh_candidates(&self, settings: &Settings) {
        let book = self.app.store.releases_since(now() - (settings.scout.refresh.track_hours * 3600.0) as i64);
        let mut out: HashMap<String, Vec<Value>> = HashMap::new();
        for bx in self.box_names() {
            let mut rows: Vec<Value> = book
                .iter()
                .filter(|r| self.scout_skip(settings, r).is_none())
                .map(|r| {
                    let demand = r.leechers.map(|l| l as f64 / (r.seeders.unwrap_or(0) as f64 + 1.0));
                    let (est, basis) = self.estimate(settings, &bx, "scout", r.size, demand);
                    json!({"tl_id": r.tl_id, "name": r.name, "size": r.size, "seeders": r.seeders, "leechers": r.leechers,
                           "demand": demand, "est": est, "basis": basis, "age_h": (now() - r.first_seen) as f64 / 3600.0})
                })
                .collect();
            rows.sort_by(|a, b| b["est"].as_f64().partial_cmp(&a["est"].as_f64()).unwrap_or(std::cmp::Ordering::Equal));
            rows.truncate(30);
            out.insert(bx, rows);
        }
        if let Ok(mut c) = self.app.candidates.write() {
            *c = out;
        }
        let _ = &self.scout;
    }

    /// The scoring tables from race outcomes: median ratio per (box, source,
    /// size bucket, demand bucket), and per (box, source).
    pub fn rebuild_tables(&self) {
        let samples = self.app.store.ratio_samples(now() - 60 * 86400, 86400);
        let mut cells: HashMap<String, Vec<f64>> = HashMap::new();
        for (bx, source, size, demand, ratio) in samples {
            cells.entry(format!("{bx}|{source}|{}|{}", size_bucket(size), demand_bucket(demand))).or_default().push(ratio);
            cells.entry(format!("{bx}|{source}|*|*")).or_default().push(ratio);
        }
        let tables: HashMap<String, (f64, usize)> = cells
            .into_iter()
            .map(|(k, mut v)| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = v.len();
                (k, (v[n / 2], n))
            })
            .collect();
        tracing::info!("baseline: scoring tables rebuilt ({} cells)", tables.len());
        self.app.bus.emit("tables", None, None, Some("baseline"), json!(tables.iter().map(|(k, (m, n))| json!({"cell": k, "median": m, "n": n})).collect::<Vec<_>>()));
        if let Ok(mut t) = self.app.tables.write() {
            *t = tables;
        }
    }
}

/// The clock sensor: a `tick` delta every minute.
pub async fn clock(app: Arc<App>) {
    let mut t = tokio::time::interval(Duration::from_secs(60));
    loop {
        t.tick().await;
        app.bus.emit("tick", None, None, Some("clock"), json!({}));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets() {
        assert_eq!(size_bucket(3e9), "lt5");
        assert_eq!(size_bucket(30e9), "20-60");
        assert_eq!(demand_bucket(Some(0.5)), "lt1");
        assert_eq!(demand_bucket(None), "none");
    }
}
