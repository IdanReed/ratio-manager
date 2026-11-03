//! The autobrr driver: autobrr only hears IRC. One filter forwards every
//! announce to the manager's intake by webhook (announce fields only, so
//! autobrr fetches no `.torrent`), and the layout that let autobrr add to
//! boxes itself — its race filter and the seedboxes as download clients — is
//! retired. autobrr-init still declares the account, indexer and IRC network.

use crate::config::{env_secret, placeholder, AutobrrCfg};
use crate::qbit::strip_url;
use crate::state::App;
use anyhow::{anyhow, bail, Result};
use reqwest::Client;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

pub const FILTER: &str = "TorrentLeech intake";
pub const ACTION: &str = "ratio-manager intake";

/// autobrr's Go template (sprig's toJson escapes names) -> the intake body.
pub const PAYLOAD: &str = r#"{"tl_id":{{ toJson .TorrentID }},"name":{{ toJson .TorrentName }},"category":{{ toJson .Category }},"freeleech":{{ .Freeleech }},"freeleech_percent":{{ .FreeleechPercent }},"uploader":{{ toJson .Uploader }},"size":{{ .Size }},"indexer":{{ toJson .IndexerIdentifier }},"title":{{ toJson .Title }},"resolution":{{ toJson .Resolution }},"source":{{ toJson .Source }},"group":{{ toJson .Group }},"year":{{ .Year }},"season":{{ .Season }},"episode":{{ .Episode }},"autobrr_ms":{{ .CurrentTimeUnixMS }}}"#;

pub struct Autobrr {
    cfg: AutobrrCfg,
    client: Client,
    password: String,
    logged_in: bool,
}

impl Autobrr {
    pub fn new(cfg: AutobrrCfg) -> Autobrr {
        let password = env_secret(&cfg.password_env);
        Autobrr {
            client: Client::builder().cookie_store(true).timeout(Duration::from_secs(30)).build().expect("http client"),
            cfg,
            password,
            logged_in: false,
        }
    }

    async fn call(&mut self, method: &str, path: &str, body: Option<Value>) -> Result<Value> {
        if !self.logged_in {
            self.login().await?;
        }
        for attempt in 0..2 {
            let url = format!("{}/api/{path}", self.cfg.url.trim_end_matches('/'));
            let mut req = match method {
                "GET" => self.client.get(&url),
                "POST" => self.client.post(&url),
                "PUT" => self.client.put(&url),
                "PATCH" => self.client.patch(&url),
                "DELETE" => self.client.delete(&url),
                _ => bail!("method {method}"),
            };
            if let Some(b) = &body {
                req = req.json(b);
            }
            let r = req.send().await.map_err(|e| anyhow!("autobrr {path}: {}", strip_url(&e)))?;
            let status = r.status();
            if (status.as_u16() == 401 || status.as_u16() == 403) && attempt == 0 {
                self.login().await?;
                continue;
            }
            let text = r.text().await.unwrap_or_default();
            if !status.is_success() {
                bail!("autobrr {method} {path}: {status} {}", text.chars().take(200).collect::<String>());
            }
            return Ok(serde_json::from_str(&text).unwrap_or(Value::Null));
        }
        bail!("autobrr {path}: refused after a fresh login")
    }

    async fn login(&mut self) -> Result<()> {
        let url = format!("{}/api/auth/login", self.cfg.url.trim_end_matches('/'));
        let r = self
            .client
            .post(&url)
            .json(&json!({"username": self.cfg.user, "password": self.password, "remember_me": false}))
            .send()
            .await
            .map_err(|e| anyhow!("autobrr login: {}", strip_url(&e)))?;
        if !r.status().is_success() {
            bail!("autobrr refused the login ({})", r.status());
        }
        self.logged_in = true;
        Ok(())
    }

    fn webhook_action(&self, existing: Option<&Value>, filter_id: i64) -> Value {
        let mut a = json!({
            "name": ACTION, "type": "WEBHOOK", "enabled": true,
            "webhook_host": self.cfg.intake_url, "webhook_type": "", "webhook_method": "POST",
            "webhook_data": PAYLOAD, "webhook_headers": [],
            "client_id": 0, "filter_id": filter_id,
            "category": "", "tags": "", "paused": false, "ignore_rules": false,
            "skip_hash_check": false, "first_last_piece_prio": false, "content_layout": "", "priority": "",
            "save_path": "", "download_path": "", "limit_upload_speed": 0, "limit_download_speed": 0,
            "limit_ratio": 0, "limit_seed_time": 0, "reannounce_skip": true, "reannounce_delete": false,
            "reannounce_interval": 7, "reannounce_max_attempts": 25,
        });
        if let Some(id) = existing.and_then(|e| e.get("id")).and_then(|v| v.as_i64()) {
            a["id"] = json!(id);
        }
        a
    }

    /// Declare the intake filter and retire the old layout. Returns what changed.
    pub async fn reconcile(&mut self, dry_run: bool) -> Result<Vec<String>> {
        if placeholder(&self.password) {
            bail!("the autobrr password is unset");
        }
        let mut changes = Vec::new();
        let indexers = self.call("GET", "indexer", None).await?;
        let tl = indexers
            .as_array()
            .and_then(|a| a.iter().find(|i| i.get("identifier").and_then(|v| v.as_str()) == Some("torrentleech")))
            .cloned()
            .ok_or_else(|| anyhow!("autobrr has no TorrentLeech indexer yet (autobrr-init declares it)"))?;
        let filters = self.call("GET", "filters", None).await?;
        let list = filters.as_array().cloned().unwrap_or_default();
        let mut cur = list.iter().find(|f| f.get("name").and_then(|v| v.as_str()) == Some(FILTER)).cloned();
        if cur.is_none() {
            if dry_run {
                changes.push(format!("(dry run) would add the filter {FILTER:?}"));
            } else {
                let created = self
                    .call("POST", "filters", Some(json!({"name": FILTER, "enabled": false, "resolutions": [], "codecs": [],
                        "sources": [], "containers": [], "origins": []})))
                    .await?;
                changes.push(format!("added the filter {FILTER:?}"));
                cur = Some(created);
            }
        }
        if let Some(c) = cur {
            let id = c.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
            let mut full = self.call("GET", &format!("filters/{id}"), None).await?;
            for k in ["announce_types", "origins", "except_origins", "bonus", "resolutions", "codecs", "sources", "containers",
                      "match_hdr", "except_hdr", "match_other", "except_other", "match_release_types", "formats", "quality",
                      "media", "match_language", "except_language"] {
                if full.get(k).map(|v| v.is_null()).unwrap_or(true) {
                    full[k] = json!([]);
                }
            }
            let acts = full.get("actions").and_then(|v| v.as_array()).cloned().unwrap_or_default();
            let mine = acts.iter().find(|a| a.get("name").and_then(|v| v.as_str()) == Some(ACTION));
            let want_action = self.webhook_action(mine, id);
            let mut drift = Vec::new();
            match mine {
                None => drift.push("webhook action"),
                Some(m) => {
                    for k in ["webhook_host", "webhook_data", "type"] {
                        if m.get(k) != want_action.get(k) {
                            drift.push("webhook action");
                            break;
                        }
                    }
                    if !m.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false) {
                        drift.push("action enabled");
                    }
                }
            }
            if acts.iter().any(|a| a.get("name").and_then(|v| v.as_str()) != Some(ACTION)) {
                drift.push("foreign actions");
            }
            // Every announce, unconditionally: the manager's intake filters.
            let declared = json!({"enabled": true, "priority": 1, "freeleech": false, "match_categories": "",
                "except_categories": "", "min_size": "", "max_size": "", "max_downloads": 0, "delay": 0});
            for (k, v) in declared.as_object().unwrap() {
                let have = full.get(k).cloned().unwrap_or(Value::Null);
                let zero = match v {
                    Value::Bool(_) => json!(false),
                    Value::Number(_) => json!(0),
                    Value::String(_) => json!(""),
                    _ => Value::Null,
                };
                let have = if have.is_null() { zero } else { have };
                if &have != v {
                    drift.push("filter fields");
                    break;
                }
            }
            let has_tl = full
                .get("indexers")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().any(|i| i.get("identifier").and_then(|v| v.as_str()) == Some("torrentleech")))
                .unwrap_or(false);
            if !has_tl {
                drift.push("indexer");
            }
            if !drift.is_empty() {
                if dry_run {
                    changes.push(format!("(dry run) would set {FILTER:?}: {}", drift.join(", ")));
                } else {
                    let mut want = full.clone();
                    for (k, v) in declared.as_object().unwrap() {
                        want[k] = v.clone();
                    }
                    want["indexers"] = json!([{"id": tl["id"], "name": tl["name"], "identifier": tl["identifier"]}]);
                    want["actions"] = json!([want_action]);
                    self.call("PUT", &format!("filters/{id}"), Some(want)).await?;
                    changes.push(format!("filter {FILTER:?}: {} -> every announce to {}", drift.join(", "), self.cfg.intake_url));
                }
            }
        }
        // Retire the old layout: filters that sent to boxes directly, then the
        // boxes as download clients (no action may reach a box but ours).
        for f in &list {
            let name = f.get("name").and_then(|v| v.as_str()).unwrap_or("");
            if self.cfg.retire_filters.iter().any(|r| name == r || name.starts_with(&format!("{r} ("))) {
                let id = f.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
                if dry_run {
                    changes.push(format!("(dry run) would remove the filter {name:?}"));
                } else {
                    self.call("DELETE", &format!("filters/{id}"), None).await?;
                    changes.push(format!("removed the filter {name:?}"));
                }
            }
        }
        let clients = self.call("GET", "downloaders", None).await?;
        for c in clients.as_array().cloned().unwrap_or_default() {
            let name = c.get("name").and_then(|v| v.as_str()).unwrap_or("");
            if self.cfg.retire_clients.iter().any(|r| r == name) {
                let id = c.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
                if dry_run {
                    changes.push(format!("(dry run) would remove the download client {name:?}"));
                } else {
                    self.call("DELETE", &format!("downloaders/{id}"), None).await?;
                    changes.push(format!("removed the download client {name:?}"));
                }
            }
        }
        Ok(changes)
    }
}

pub async fn reconciler(app: Arc<App>) {
    let mut a = Autobrr::new(app.registry.autobrr.clone());
    let mut delay = 20u64;
    loop {
        tokio::time::sleep(Duration::from_secs(delay)).await;
        match a.reconcile(app.dry_run).await {
            Ok(changes) => {
                delay = 600;
                if !changes.is_empty() {
                    tracing::info!("autobrr: {}", changes.join("; "));
                    app.bus.emit("setting", None, None, Some("autobrr"), json!({"applied": changes}));
                }
            }
            Err(e) => {
                delay = 120;
                a.logged_in = false;
                tracing::warn!("autobrr: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn payload_shape() {
        // Every macro is closed and the body is one JSON object.
        assert_eq!(super::PAYLOAD.matches("{{").count(), super::PAYLOAD.matches("}}").count());
        assert!(super::PAYLOAD.starts_with('{') && super::PAYLOAD.ends_with('}'));
    }
}
