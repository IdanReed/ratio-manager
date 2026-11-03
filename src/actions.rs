//! The one path to a box: an action is checked by the guard, executed by the
//! box's driver session, and recorded as an `action` event either way.

use crate::bencode::{torrent_facts, TorrentFacts};
use crate::config::Settings;
use crate::guard;
use crate::qbit::{s, Qbit};
use crate::state::{App, CATEGORY, TAG};
use crate::store::Race;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    Add {
        #[serde(rename = "box")]
        bx: String,
        #[serde(default)]
        tl_id: Option<String>,
        /// A hand-off: the `.torrent` is exported from this box.
        #[serde(default)]
        from_box: Option<String>,
        #[serde(default)]
        hash: Option<String>,
        #[serde(default = "manual")]
        source: String,
        #[serde(default)]
        delay_s: f64,
        #[serde(default)]
        reason: String,
    },
    Remove {
        #[serde(rename = "box")]
        bx: String,
        hash: String,
        #[serde(default)]
        reason: String,
    },
    SetLimits {
        #[serde(rename = "box")]
        bx: String,
        hashes: Vec<String>,
        /// Bytes/s; 0 = unlimited.
        dl_limit: i64,
        #[serde(default)]
        reason: String,
    },
    Pause {
        #[serde(rename = "box")]
        bx: String,
        hash: String,
    },
    Resume {
        #[serde(rename = "box")]
        bx: String,
        hash: String,
    },
}

fn manual() -> String {
    "manual".into()
}

impl Action {
    pub fn bx(&self) -> &str {
        match self {
            Action::Add { bx, .. }
            | Action::Remove { bx, .. }
            | Action::SetLimits { bx, .. }
            | Action::Pause { bx, .. }
            | Action::Resume { bx, .. } => bx,
        }
    }
    pub fn kind(&self) -> &'static str {
        match self {
            Action::Add { .. } => "add",
            Action::Remove { .. } => "remove",
            Action::SetLimits { .. } => "set_limits",
            Action::Pause { .. } => "pause",
            Action::Resume { .. } => "resume",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Outcome {
    pub status: String,
    pub reasons: Vec<String>,
    pub hash: Option<String>,
}

impl Outcome {
    fn ok(hash: Option<String>, note: &str) -> Outcome {
        Outcome { status: "executed".into(), reasons: if note.is_empty() { vec![] } else { vec![note.into()] }, hash }
    }
    fn rejected(reasons: Vec<String>, hash: Option<String>) -> Outcome {
        Outcome { status: "rejected".into(), reasons, hash }
    }
    pub fn executed(&self) -> bool {
        self.status == "executed"
    }
}

#[derive(Clone)]
pub struct Executor {
    pub app: Arc<App>,
    pub qbits: Arc<HashMap<String, Qbit>>,
    /// One add at a time per box: the guard's room check and the add it
    /// allows happen together.
    pub locks: Arc<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl Executor {
    pub fn new(app: Arc<App>, qbits: HashMap<String, Qbit>) -> Executor {
        let locks = qbits.keys().map(|k| (k.clone(), Arc::new(tokio::sync::Mutex::new(())))).collect();
        Executor { app, qbits: Arc::new(qbits), locks: Arc::new(locks) }
    }
}

pub fn is_manual(actor: &str) -> bool {
    actor == "manual" || actor.starts_with("manual:")
}

impl Executor {
    /// Run one action as `actor`. Policy gating: the baseline only proposes for
    /// boxes it acts on (policy.rs); a model's actions run only where the box's
    /// policy is `model`, are logged unexecuted under `shadow`, and refused
    /// otherwise; a human's always reach the guard.
    pub async fn run(&self, actor: &str, action: Action, extra: Value) -> Outcome {
        let settings = self.app.settings();
        let bx = action.bx().to_string();
        let gate = if is_manual(actor) || actor == "baseline" || actor == "driver" || actor == "guard" {
            "run"
        } else {
            match settings.policy.get(&bx).map(|s| s.as_str()) {
                Some("model") => "run",
                Some("shadow") => "shadow",
                _ => "refuse",
            }
        };
        let out = match gate {
            "shadow" => Outcome { status: "shadow".into(), reasons: vec!["policy is shadow: logged, not executed".into()], hash: None },
            "refuse" => Outcome::rejected(vec![format!("{bx}'s policy does not take actions from {actor}")], None),
            _ => self.execute(actor, &action, &settings).await,
        };
        let mut data = json!({"action": action, "status": out.status, "reasons": out.reasons, "extra": extra});
        if self.app.dry_run && out.executed() {
            data["dry_run"] = json!(true);
        }
        self.app.bus.emit("action", Some(&bx), out.hash.as_deref(), Some(actor), data);
        if out.executed() || out.status == "rejected" {
            tracing::info!("{} {} [{}] {}: {}", actor, action.kind(), bx, out.status, out.reasons.join("; "));
        }
        out
    }

    async fn execute(&self, actor: &str, action: &Action, settings: &Settings) -> Outcome {
        let manual = is_manual(actor);
        if let Some(h) = self.app.halted() {
            let allowed = manual && matches!(action, Action::Pause { .. } | Action::Remove { .. });
            if !allowed {
                return Outcome::rejected(vec![format!("halted: {}", h.reason)], None);
            }
        }
        let bx = action.bx();
        let Some(q) = self.qbits.get(bx) else {
            return Outcome::rejected(vec![format!("no driver for {bx}")], None);
        };
        match action {
            Action::Add { bx, tl_id, from_box, hash, source, delay_s, reason } => {
                let (raw, facts) = match self.torrent(tl_id.as_deref(), from_box.as_deref(), hash.as_deref(), source == "race").await {
                    Ok(x) => x,
                    Err(e) => return Outcome::rejected(vec![format!("no .torrent: {e}")], hash.clone()),
                };
                let release = tl_id
                    .as_deref()
                    .and_then(|id| self.app.store.release(id))
                    .or_else(|| self.app.store.release_by_hash(&facts.hash));
                if *delay_s > 0.0 {
                    if let Err(why) = guard::check_add(&self.app, settings, bx, &facts, release.as_ref(), source) {
                        return Outcome::rejected(why, Some(facts.hash));
                    }
                    tokio::time::sleep(Duration::from_secs_f64(*delay_s)).await;
                }
                // The guard's room check and the add it allows, together.
                let lock = self.locks.get(bx.as_str()).cloned();
                let _held = match &lock {
                    Some(l) => Some(l.lock().await),
                    None => None,
                };
                if let Err(why) = guard::check_add(&self.app, &self.app.settings(), bx, &facts, release.as_ref(), source) {
                    return Outcome::rejected(why, Some(facts.hash));
                }
                if self.app.dry_run {
                    return Outcome::ok(Some(facts.hash), "dry run: not added");
                }
                let tags = format!("{TAG},{source}");
                let fields = [("category", CATEGORY.to_string()), ("tags", tags), ("autoTMM", "true".to_string())];
                if let Err(e) = q.add(&raw, &facts.hash, &fields).await {
                    return Outcome::rejected(vec![e.to_string()], Some(facts.hash));
                }
                self.note_add(bx, &facts, release.as_ref(), source);
                let me = self.clone();
                let (b2, h2) = (bx.clone(), facts.hash.clone());
                tokio::spawn(async move { me.verify_soon(&b2, &h2).await });
                Outcome::ok(Some(facts.hash), reason)
            }
            Action::Remove { bx, hash, reason } => {
                let Some(view) = self.app.view(bx) else {
                    return Outcome::rejected(vec![format!("{bx} not seen yet")], Some(hash.clone()));
                };
                if !view.torrents.contains_key(hash) {
                    return Outcome::rejected(vec![format!("{bx} does not hold it")], Some(hash.clone()));
                }
                let unreg = if reason.contains("unregistered") { self.all_unregistered(q, hash).await } else { false };
                let Some(rule) = guard::may_remove(&self.app, settings, &view, hash, unreg) else {
                    return Outcome::rejected(vec!["hit-and-run: it met no per-torrent minimum (rule 4)".into()], Some(hash.clone()));
                };
                if self.app.dry_run {
                    return Outcome::ok(Some(hash.clone()), &format!("dry run: would remove ({rule})"));
                }
                if let Err(e) = q.delete(std::slice::from_ref(hash), true).await {
                    return Outcome::rejected(vec![e.to_string()], Some(hash.clone()));
                }
                self.app.store.mark_removed(bx, hash, &rule);
                if let Ok(mut b) = self.app.boxes.write() {
                    if let Some(v) = b.get_mut(bx) {
                        v.torrents.remove(hash);
                        v.all_sizes.remove(hash);
                        v.pending.remove(hash);
                    }
                }
                Outcome::ok(Some(hash.clone()), &format!("{rule}; {reason}"))
            }
            Action::SetLimits { bx: _, hashes, dl_limit, reason } => {
                if self.app.dry_run {
                    return Outcome::ok(None, "dry run");
                }
                match q.set_download_limit(hashes, *dl_limit).await {
                    Ok(_) => Outcome::ok(None, reason),
                    Err(e) => Outcome::rejected(vec![e.to_string()], None),
                }
            }
            Action::Pause { hash, .. } => {
                if self.app.dry_run {
                    return Outcome::ok(Some(hash.clone()), "dry run");
                }
                match q.stop(std::slice::from_ref(hash)).await {
                    Ok(_) => Outcome::ok(Some(hash.clone()), ""),
                    Err(e) => Outcome::rejected(vec![e.to_string()], Some(hash.clone())),
                }
            }
            Action::Resume { hash, .. } => {
                if self.app.dry_run {
                    return Outcome::ok(Some(hash.clone()), "dry run");
                }
                match q.start(std::slice::from_ref(hash)).await {
                    Ok(_) => Outcome::ok(Some(hash.clone()), ""),
                    Err(e) => Outcome::rejected(vec![e.to_string()], Some(hash.clone())),
                }
            }
        }
    }

    /// The `.torrent`: from TL by id, or exported from another of our boxes.
    pub async fn torrent(&self, tl_id: Option<&str>, from_box: Option<&str>, hash: Option<&str>, urgent: bool) -> Result<(Arc<Vec<u8>>, TorrentFacts)> {
        let every = Duration::from_secs_f64(self.app.settings().politeness.torrent_fetch_every_s);
        if let Some(id) = tl_id {
            return self.app.tl.fetch(id, every, urgent).await;
        }
        if let (Some(src), Some(h)) = (from_box, hash) {
            let q = self.qbits.get(src).ok_or_else(|| anyhow!("no driver for {src}"))?;
            let raw = q.export(h).await?;
            let facts = torrent_facts(&raw)?;
            if facts.hash != h {
                return Err(anyhow!("exported .torrent is {}, not {h}", facts.hash));
            }
            return Ok((Arc::new(raw), facts));
        }
        Err(anyhow!("an add needs tl_id, or from_box and hash"))
    }

    async fn all_unregistered(&self, q: &Qbit, hash: &str) -> bool {
        let rows = q.trackers(hash).await.unwrap_or_default();
        let real: Vec<&Value> = rows.iter().filter(|r| !s(r, "url").starts_with("** [")).collect();
        !real.is_empty() && real.iter().all(|r| guard::unregistered_msg(s(r, "msg")))
    }

    fn note_add(&self, bx: &str, facts: &TorrentFacts, release: Option<&crate::store::Release>, source: &str) {
        let now = chrono::Utc::now().timestamp();
        let demand = release.and_then(|r| match (r.leechers, r.seeders) {
            (Some(l), Some(s)) => Some(l as f64 / (s as f64 + 1.0)),
            _ => None,
        });
        let announce = self.app.announced.read().ok().and_then(|a| a.get(&facts.hash).copied()).or(release.and_then(|r| r.announce_ts));
        self.app.store.upsert_race(&Race {
            box_: bx.to_string(),
            hash: facts.hash.clone(),
            name: release.map(|r| r.name.clone()).unwrap_or_else(|| facts.name.clone()),
            source: source.to_string(),
            category: release.map(|r| r.category.clone()).unwrap_or_default(),
            size: facts.size,
            added_at: now,
            demand_at_add: demand,
            announce_ts: announce,
            ..Default::default()
        });
        // Count it at once, so the next decision sees the room it takes.
        if let Ok(mut b) = self.app.boxes.write() {
            if let Some(v) = b.get_mut(bx) {
                v.torrents.insert(
                    facts.hash.clone(),
                    json!({"hash": facts.hash, "name": facts.name, "size": facts.size, "total_size": facts.size,
                           "amount_left": facts.size, "progress": 0.0, "state": "metaDL", "category": CATEGORY,
                           "tags": format!("{TAG}, {source}"), "added_on": now, "uploaded": 0, "downloaded": 0,
                           "ratio": 0.0, "upspeed": 0, "dlspeed": 0}),
                );
                v.all_sizes.insert(facts.hash.clone(), (facts.size, facts.size));
                if let Some(t) = v.torrents.get(&facts.hash).cloned() {
                    v.pending.insert(facts.hash.clone(), (t, now));
                }
            }
        }
    }

    /// Catch a leak within seconds of an add, not a poll later.
    async fn verify_soon(&self, bx: &str, hash: &str) {
        for wait in [20u64, 40, 120] {
            tokio::time::sleep(Duration::from_secs(wait)).await;
            let Some(q) = self.qbits.get(bx) else { return };
            let Ok(list) = q.torrents(&format!("?hashes={hash}")).await else { continue };
            let Some(t) = list.first() else { return };
            match guard::leak_check(&self.app, q, bx, t).await {
                Some(_) => return,
                None => continue,
            }
        }
    }
}
