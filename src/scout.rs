//! The scout: a sensor. It keeps the release book (every release seen on the
//! listing, the IRC intake or the RSS feed), refreshes swarms with paced,
//! batched scrapes, and emits a `market` delta only when a candidate's supply
//! or demand moved past `scout.delta`. It never ranks or picks.

use crate::config::{Settings, GB};
use crate::state::App;
use crate::store::Release;
use crate::tl::tl_id_of;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

pub fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

/// Whether a release is something the baseline may consider (intake filter).
pub fn intake_ok(settings: &Settings, name: &str, category: &str, freeleech: bool, size: Option<f64>) -> Result<(), String> {
    let i = &settings.intake;
    if i.freeleech_only && !freeleech {
        return Err("not freeleech".into());
    }
    let hay = format!("{} {}", name.to_lowercase(), category.to_lowercase());
    let words: Vec<String> = hay.split(|c: char| !c.is_alphanumeric() && c != '/').map(|w| w.to_string()).collect();
    for x in &i.exclude {
        let x = x.to_lowercase();
        if words.iter().any(|w| *w == x) || category.to_lowercase().contains(&x) {
            return Err(format!("excluded ({x})"));
        }
    }
    if let Some(sz) = size {
        if sz < i.size_gb[0] * GB || sz > i.size_gb[1] * GB {
            return Err(format!("size {:.1} GB outside {}..{} GB", sz / GB, i.size_gb[0], i.size_gb[1]));
        }
    }
    Ok(())
}

pub fn candidate(settings: &Settings, r: &Release) -> bool {
    r.hash.is_some() && r.private == Some(true) && intake_ok(settings, &r.name, &r.category, r.freeleech, Some(r.size)).is_ok()
}

#[derive(Clone)]
pub struct Scout {
    pub app: Arc<App>,
    pub fetch_tx: mpsc::UnboundedSender<String>,
}

impl Scout {
    pub fn start(app: Arc<App>) -> Scout {
        let (fetch_tx, fetch_rx) = mpsc::unbounded_channel();
        let s = Scout { app: app.clone(), fetch_tx };
        tokio::spawn(s.clone().fetcher(fetch_rx));
        tokio::spawn(s.clone().listing());
        tokio::spawn(s.clone().refresher());
        tokio::spawn(s.clone().rss());
        s
    }

    /// Queue a `.torrent` fetch for a release that needs its facts.
    pub fn want_facts(&self, tl_id: &str) {
        let _ = self.fetch_tx.send(tl_id.to_string());
    }

    async fn fetcher(self, mut rx: mpsc::UnboundedReceiver<String>) {
        let mut queued: HashSet<String> = HashSet::new();
        let mut pending: std::collections::VecDeque<String> = Default::default();
        loop {
            // Drain what arrived, newest releases first is not needed: FIFO.
            while let Ok(id) = rx.try_recv() {
                if queued.insert(id.clone()) {
                    pending.push_back(id);
                }
            }
            let Some(id) = pending.pop_front() else {
                match rx.recv().await {
                    Some(id) => {
                        if queued.insert(id.clone()) {
                            pending.push_back(id);
                        }
                        continue;
                    }
                    None => return,
                }
            };
            queued.remove(&id);
            let Some(r) = self.app.store.release(&id) else { continue };
            if r.hash.is_some() || r.fetch_failed >= 3 {
                continue;
            }
            let every = Duration::from_secs_f64(self.app.settings().politeness.torrent_fetch_every_s);
            match self.app.tl.fetch(&id, every, false).await {
                Ok((_, facts)) => {
                    self.app.store.set_release_facts(&id, &facts.hash, facts.private, facts.size);
                    self.app.bus.emit("release", None, Some(&facts.hash), Some("scout"),
                        json!({"tl_id": id, "name": r.name, "category": r.category, "size": facts.size, "files": facts.files,
                               "piece_length": facts.piece_length, "private": facts.private, "freeleech": r.freeleech, "source": r.source}));
                }
                Err(e) => {
                    self.app.store.release_fetch_failed(&id);
                    tracing::debug!("scout: .torrent {id}: {e}");
                }
            }
        }
    }

    async fn listing(self) {
        let mut indexer: Option<(i64, String)> = None;
        let mut indexer_at = 0i64;
        loop {
            let settings = self.app.settings();
            if now() - indexer_at > 1800 || indexer.is_none() {
                match self.app.prowlarr.indexer(&self.app.registry.prowlarr.indexers).await {
                    Ok(i) => {
                        if i.is_none() {
                            tracing::warn!("scout: Prowlarr has none of {:?} enabled", self.app.registry.prowlarr.indexers);
                        }
                        indexer = i;
                        indexer_at = now();
                    }
                    Err(e) => tracing::warn!("scout: Prowlarr indexers: {e}"),
                }
            }
            if let Some((id, _)) = &indexer {
                for page in 0..settings.scout.scan_pages.max(1) {
                    match self.app.prowlarr.newest(*id, page * 100, 100).await {
                        Ok(rows) => self.take_listing(&settings, rows),
                        Err(e) => {
                            tracing::warn!("scout: listing scan: {e}");
                            break;
                        }
                    }
                }
            }
            tokio::time::sleep(Duration::from_secs(settings.scout.scan_every_s.max(30))).await;
        }
    }

    fn take_listing(&self, settings: &Settings, rows: Vec<Value>) {
        let mut seen = Vec::new();
        for r in rows {
            let fields: Vec<String> = ["infoUrl", "guid", "commentUrl"].iter().map(|k| r.get(*k).and_then(|v| v.as_str()).unwrap_or("").to_string()).collect();
            let refs: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();
            let Some(tl_id) = tl_id_of(&refs) else { continue };
            let title = r.get("title").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let size = r.get("size").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let freeleech = r.get("indexerFlags").and_then(|v| v.as_array()).map(|a| a.iter().any(|x| x.as_str() == Some("freeleech"))).unwrap_or(false);
            let category = r
                .get("categories")
                .and_then(|v| v.as_array())
                .and_then(|a| a.iter().filter_map(|c| c.get("name").and_then(|n| n.as_str())).last())
                .unwrap_or("")
                .to_string();
            let before = self.app.store.release(&tl_id);
            let new = self.app.store.note_release(&tl_id, &title, &category, size, Some(freeleech), None, "listing");
            if let Some(b) = &before {
                if b.freeleech != freeleech {
                    self.app.store.set_release_freeleech(&tl_id, freeleech);
                    if let Some(r) = self.app.store.release(&tl_id) {
                        if r.hash.is_some() {
                            self.emit_market(&r, "freeleech changed");
                        }
                    }
                }
            }
            let ok = intake_ok(settings, &title, &category, freeleech, Some(size)).is_ok();
            if (new || before.as_ref().map(|b| b.hash.is_none()).unwrap_or(false)) && (ok || settings.scout.every_release) {
                self.want_facts(&tl_id);
            }
            seen.push(json!({"tl_id": tl_id, "s": r.get("seeders"), "l": r.get("leechers"), "size": size, "fl": freeleech}));
        }
        self.app.bus.emit("listing", None, None, Some("scout"), json!({"n": seen.len(), "rows": seen}));
    }

    async fn rss(self) {
        loop {
            let settings = self.app.settings();
            if !settings.scout.every_release {
                tokio::time::sleep(Duration::from_secs(300)).await;
                continue;
            }
            match self.app.tl.rss().await {
                Ok(items) => {
                    for it in items {
                        let new = self.app.store.note_release(&it.tl_id, &it.title, &it.category, 0.0, None, None, "rss");
                        if new {
                            self.app.bus.emit("rss_item", None, None, Some("scout"), json!(it));
                            self.want_facts(&it.tl_id);
                        }
                    }
                }
                Err(e) => tracing::debug!("scout: rss: {e}"),
            }
            tokio::time::sleep(Duration::from_secs(settings.scout.rss_every_s.max(60))).await;
        }
    }

    /// Scrapes what is due: the fixed schedule for every release (wave 2), and
    /// the adaptive refresh for candidates.
    async fn refresher(self) {
        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            if !self.app.tl.can_scrape() {
                continue;
            }
            let settings = self.app.settings();
            let n = now();
            let book = self.app.store.releases_since(n - (settings.scout.refresh.track_hours * 3600.0) as i64);
            let mut due: Vec<(Release, Option<i64>)> = Vec::new();
            for r in book {
                if r.hash.is_none() {
                    continue;
                }
                let age = n - r.first_seen;
                let last = r.last_scrape.unwrap_or(0);
                // The fixed schedule: +0, 2, 5, ... minutes after first sight.
                let sched = &settings.scout.schedule_min;
                let idx = r.sched_idx as usize;
                if settings.scout.every_release && idx < sched.len() && age as f64 >= sched[idx] * 60.0 {
                    let mut next = idx;
                    while next < sched.len() && age as f64 >= sched[next] * 60.0 {
                        next += 1;
                    }
                    due.push((r, Some(next as i64)));
                    continue;
                }
                if candidate(&settings, &r) {
                    let moving = r.delta_ts.map(|t| n - t < 3600).unwrap_or(true);
                    let young = (age as f64) < settings.scout.refresh.active_hours * 3600.0;
                    let every = if young || moving { settings.scout.refresh.active_every_s as i64 } else { settings.scout.refresh.settled_every_min as i64 * 60 };
                    if n - last >= every {
                        due.push((r, None));
                    }
                }
            }
            if due.is_empty() {
                continue;
            }
            // One refresh round at a time, capped so a burst spreads out.
            due.truncate(200);
            let hashes: Vec<String> = due.iter().filter_map(|(r, _)| r.hash.clone()).collect();
            match self.app.tl.scrape(&hashes, settings.politeness.scrape_per_s, settings.politeness.scrape_batch).await {
                Ok(got) => {
                    let ts = now();
                    for (r, sched) in due {
                        let h = r.hash.clone().unwrap_or_default();
                        let Some((c, inc, d)) = got.get(&h).copied() else {
                            // Not in the reply: still bump the schedule so a dead one is not retried every round.
                            self.app.store.set_release_swarm(&r.tl_id, r.seeders.unwrap_or(0), r.leechers.unwrap_or(0), r.completed.unwrap_or(0), ts, sched);
                            continue;
                        };
                        self.app.store.set_release_swarm(&r.tl_id, c, inc, d, ts, sched);
                        self.app.bus.emit("scrape", None, Some(&h), Some("scout"),
                            json!({"tl_id": r.tl_id, "complete": c, "incomplete": inc, "downloaded": d, "age_s": ts - r.first_seen, "scheduled": sched.is_some()}));
                        if candidate(&settings, &r) {
                            let mut now_r = r.clone();
                            now_r.seeders = Some(c);
                            now_r.leechers = Some(inc);
                            now_r.completed = Some(d);
                            if let Some(reason) = moved(&settings, &r, c as f64, inc as f64) {
                                self.emit_market(&now_r, &reason);
                            }
                        }
                    }
                }
                Err(e) => tracing::debug!("scout: scrape: {e}"),
            }
        }
    }

    pub fn emit_market(&self, r: &Release, reason: &str) {
        let (s, l) = (r.seeders.unwrap_or(0) as f64, r.leechers.unwrap_or(0) as f64);
        self.app.store.set_release_delta(&r.tl_id, l, s, now());
        self.app.bus.emit(
            "market",
            None,
            r.hash.as_deref(),
            Some("scout"),
            json!({"tl_id": r.tl_id, "name": r.name, "category": r.category, "size": r.size, "freeleech": r.freeleech,
                   "seeders": r.seeders, "leechers": r.leechers, "completed": r.completed, "demand": l / (s + 1.0),
                   "prev": {"leechers": r.delta_l, "seeders": r.delta_s}, "reason": reason}),
        );
    }
}

/// A meaningful move since the last `market` for this release: the relative
/// and absolute thresholds both crossed, on leechers or on seeders.
pub fn moved(settings: &Settings, r: &Release, s: f64, l: f64) -> Option<String> {
    let d = &settings.scout.delta;
    match (r.delta_l, r.delta_s) {
        (None, _) | (_, None) => Some("entered the book".into()),
        (Some(l0), Some(s0)) => {
            let dl = (l - l0).abs();
            let ds = (s - s0).abs();
            if dl >= d.leechers && dl / l0.max(1.0) >= d.relative {
                Some(format!("leechers {l0:.0} -> {l:.0}"))
            } else if ds >= d.leechers && ds / s0.max(1.0) >= d.relative {
                Some(format!("seeders {s0:.0} -> {s:.0}"))
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn settings() -> Settings {
        Settings::from_overrides(&json!({})).unwrap().0
    }

    #[test]
    fn intake_filter() {
        let s = settings();
        assert!(intake_ok(&s, "Movie.2026.1080p.WEB", "Movies :: HD", true, Some(5e9)).is_ok());
        assert!(intake_ok(&s, "Movie.2026.CAM.x264", "Movies :: Cam", true, Some(5e9)).is_err());
        assert!(intake_ok(&s, "Movie.2026.1080p", "Movies :: HD", false, Some(5e9)).is_err());
        assert!(intake_ok(&s, "Movie.2026.1080p", "Movies :: HD", true, Some(500e9)).is_err());
        assert!(intake_ok(&s, "Camera.Obscura.2026", "TV :: Episodes HD", true, Some(5e9)).is_ok());
    }

    #[test]
    fn deltas() {
        let s = settings();
        let mut r = Release {
            tl_id: "1".into(), name: "n".into(), category: "c".into(), size: 1.0, freeleech: true, first_seen: 0,
            hash: Some("h".into()), private: Some(true), announce_ts: None, seeders: None, leechers: None,
            completed: None, last_scrape: None, delta_l: None, delta_s: None, delta_ts: None, fetch_failed: 0,
            sched_idx: 0, source: "x".into(),
        };
        assert!(moved(&s, &r, 1.0, 1.0).is_some());
        r.delta_l = Some(2.0);
        r.delta_s = Some(1.0);
        assert!(moved(&s, &r, 1.0, 3.0).is_none(), "2 -> 3 leechers is noise");
        r.delta_l = Some(20.0);
        assert!(moved(&s, &r, 1.0, 30.0).is_some(), "20 -> 30 matters");
        assert!(moved(&s, &r, 1.0, 22.0).is_none());
    }
}
