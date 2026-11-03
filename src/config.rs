//! The registry (git: which boxes exist and how to reach them) and the
//! settings (the UI: bounds and weights the policy works within).

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const DEFAULTS_YAML: &str = include_str!("defaults.yaml");

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Registry {
    pub ntfy_url: Option<String>,
    pub prowlarr: ProwlarrCfg,
    pub autobrr: AutobrrCfg,
    pub tl: TlCfg,
    #[serde(default)]
    pub victorialogs_url: Option<String>,
    pub boxes: Vec<BoxCfg>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProwlarrCfg {
    pub url: String,
    pub api_key_env: String,
    /// Tried in order; the first enabled one is the scout's listing.
    pub indexers: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AutobrrCfg {
    pub url: String,
    pub user: String,
    pub password_env: String,
    /// Where autobrr's webhook reaches the manager (the compose network).
    pub intake_url: String,
    /// The hostname autobrr's requests come from; only its addresses may post
    /// to the intake.
    pub intake_from_host: String,
    /// Filters an earlier layout declared, removed with their download clients.
    #[serde(default)]
    pub retire_filters: Vec<String>,
    #[serde(default)]
    pub retire_clients: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TlCfg {
    pub rss_key_env: String,
    /// Where `.torrent` files are fetched (`/rss/download/<id>/<key>/<id>.torrent`).
    #[serde(default = "tl_base")]
    pub base_url: String,
    /// `{key}` is replaced by the RSS key.
    pub rss_url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BoxCfg {
    pub name: String,
    /// qBittorrent WebUI; `{user}` is replaced by the login name.
    pub url: String,
    pub user_env: String,
    pub password_env: String,
    #[serde(default)]
    pub basic_auth: bool,
    /// The `ratio` category's save path, where the box needs one.
    #[serde(default)]
    pub save_path: Option<String>,
    /// The manager owns the client-wide preferences (a box serving only
    /// TorrentLeech). Home's belong to media-init.
    #[serde(default)]
    pub manage_prefs: bool,
    /// DHT, PeX and LSD held off client-wide (a box serving only private
    /// torrents). Elsewhere each torrent is checked.
    #[serde(default)]
    pub private_only: bool,
    /// Behind a VPN tunnel whose reconnects drop qBittorrent's listener.
    #[serde(default)]
    pub tunnel: bool,
    /// The IP registered on the TL profile (Profile -> Seedbox). A seedbox
    /// whose address differs takes no adds (guard rule 8).
    #[serde(default)]
    pub tl_registered_ip: Option<String>,
    /// Addresses its peers see besides its URL's host.
    #[serde(default)]
    pub peer_ips: Vec<String>,
    /// Shell access for host metrics (PSI); the login is the box's.
    #[serde(default)]
    pub ssh: Option<SshCfg>,
    #[serde(default = "default_timeout")]
    pub timeout_s: u64,
}

fn tl_base() -> String {
    "https://www.torrentleech.org".into()
}

fn default_timeout() -> u64 {
    45
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SshCfg {
    pub host: String,
    #[serde(default = "default_ssh_port")]
    pub port: u16,
}

fn default_ssh_port() -> u16 {
    22
}

impl BoxCfg {
    pub fn user(&self) -> String {
        std::env::var(&self.user_env).unwrap_or_default().trim().to_string()
    }
    pub fn password(&self) -> String {
        std::env::var(&self.password_env).unwrap_or_default()
    }
    pub fn resolved_url(&self) -> String {
        self.url.replace("{user}", &self.user()).trim_end_matches('/').to_string()
    }
    pub fn host(&self) -> Option<String> {
        reqwest::Url::parse(&self.resolved_url())
            .ok()
            .and_then(|u| u.host_str().map(|s| s.to_string()))
    }
    /// A login still a `changeme` placeholder leaves the box off.
    pub fn enabled(&self) -> bool {
        !placeholder(&self.user()) && !placeholder(&self.password())
    }
    pub fn is_seedbox(&self) -> bool {
        self.tl_registered_ip.is_some()
    }
}

pub fn placeholder(v: &str) -> bool {
    v.trim().is_empty() || v.contains("changeme")
}

pub fn env_secret(name: &str) -> String {
    std::env::var(name).unwrap_or_default()
}

impl Registry {
    pub fn load(path: &str) -> Result<Registry> {
        let raw = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
        let reg: Registry = serde_yaml::from_str(&raw).with_context(|| format!("parsing {path}"))?;
        let mut seen = std::collections::HashSet::new();
        for b in &reg.boxes {
            if !seen.insert(b.name.clone()) {
                bail!("registry: box {} is declared twice", b.name);
            }
        }
        Ok(reg)
    }
    pub fn box_cfg(&self, name: &str) -> Option<&BoxCfg> {
        self.boxes.iter().find(|b| b.name == name)
    }
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Settings {
    pub account: Account,
    pub intake: Intake,
    pub scout: Scout,
    pub scoring: Scoring,
    pub reannounce: Reannounce,
    pub experiments: Experiments,
    pub politeness: Politeness,
    pub handoff: Handoff,
    pub boxes: BTreeMap<String, BoxSettings>,
    pub policy: BTreeMap<String, String>,
    #[serde(default)]
    pub bandwidth: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Account {
    pub tracker: String,
    pub vip_until: Option<String>,
    pub vip_cost_eur_month: f64,
    pub hnr_min_seed_days: f64,
    pub min_overall_ratio: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Intake {
    pub freeleech_only: bool,
    pub size_gb: [f64; 2],
    pub exclude: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Scout {
    pub scan_every_s: u64,
    pub scan_pages: u32,
    pub refresh: Refresh,
    pub delta: Delta,
    pub every_release: bool,
    pub rss_every_s: u64,
    pub schedule_min: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Refresh {
    pub active_every_s: u64,
    pub settled_every_min: u64,
    pub active_hours: f64,
    pub track_hours: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Delta {
    pub relative: f64,
    pub leechers: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Bias {
    #[serde(default)]
    pub race: Option<f64>,
    #[serde(default)]
    pub scout: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Scoring {
    pub horizon_h: f64,
    pub min_ratio: f64,
    pub swap_margin: f64,
    pub keep_above_mib: f64,
    pub max_download_h: f64,
    pub holding_decay: f64,
    pub bias: Bias,
    pub priors: BTreeMap<String, Prior>,
    pub min_samples: usize,
    pub min_seeders: i64,
    pub readd_after_h: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Prior {
    pub race: f64,
    pub scout: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ScoringOverride {
    #[serde(default)]
    pub bias: Bias,
    pub min_ratio: Option<f64>,
    pub swap_margin: Option<f64>,
    pub keep_above_mib: Option<f64>,
    pub max_download_h: Option<f64>,
    pub horizon_h: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Reannounce {
    pub early: Cadence,
    pub late: Cadence,
    pub delete_after_min: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Cadence {
    pub every_s: f64,
    #[serde(default)]
    pub for_min: Option<f64>,
    #[serde(default)]
    pub until_min: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Experiments {
    pub join_delay: JoinDelay,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JoinDelay {
    pub share: f64,
    pub delays_s: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Politeness {
    pub torrent_fetch_every_s: f64,
    pub scrape_per_s: f64,
    pub scrape_batch: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Handoff {
    pub per_tick: usize,
    pub max_active: usize,
    pub need_gb: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct BoxSettings {
    #[serde(default)]
    pub expires: Option<String>,
    #[serde(default)]
    pub cost_eur_month: f64,
    #[serde(default)]
    pub traffic: Traffic,
    #[serde(default)]
    pub disk: Disk,
    #[serde(default = "yes")]
    pub evict: bool,
    /// MiB/s; 0 = no cap.
    #[serde(default)]
    pub download_cap_mib: f64,
    #[serde(default)]
    pub disk_cache_mib: Option<i64>,
    #[serde(default)]
    pub async_io_threads: Option<i64>,
    #[serde(default)]
    pub focus_downloads: Option<usize>,
    #[serde(default)]
    pub trickle_kib: Option<f64>,
    #[serde(default)]
    pub downloads: Downloads,
    #[serde(default)]
    pub scoring: ScoringOverride,
}

fn yes() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Traffic {
    /// TB a month; 0 = unlimited.
    #[serde(default)]
    pub quota_tb: f64,
    #[serde(default)]
    pub reset_day: Option<u32>,
    #[serde(default)]
    pub throttle_mbit: Option<f64>,
    #[serde(default)]
    pub mode: Option<String>,
    /// The client's all-time up + down at the start of the current period
    /// (rebased automatically on reset_day).
    #[serde(default)]
    pub base_tb: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Disk {
    #[serde(default)]
    pub quota_gb: f64,
    #[serde(default)]
    pub reserve_gb: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Downloads {
    pub max_active: usize,
    pub busy_above_mib: f64,
}

impl Default for Downloads {
    fn default() -> Self {
        Downloads { max_active: 3, busy_above_mib: 50.0 }
    }
}

pub fn defaults_value() -> Value {
    serde_yaml::from_str(DEFAULTS_YAML).expect("defaults.yaml parses")
}

/// Deep merge: objects merge key by key, anything else is replaced.
pub fn merge(base: &mut Value, over: &Value) {
    match (base, over) {
        (Value::Object(b), Value::Object(o)) => {
            for (k, v) in o {
                match b.get_mut(k) {
                    Some(bv) if bv.is_object() && v.is_object() => merge(bv, v),
                    _ => {
                        b.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (b, o) => *b = o.clone(),
    }
}

/// The overrides that turn `base` into `full`: only what differs.
pub fn diff(base: &Value, full: &Value) -> Option<Value> {
    match (base, full) {
        (Value::Object(b), Value::Object(f)) => {
            let mut out = serde_json::Map::new();
            for (k, v) in f {
                match b.get(k) {
                    Some(bv) => {
                        if let Some(d) = diff(bv, v) {
                            out.insert(k.clone(), d);
                        }
                    }
                    None => {
                        out.insert(k.clone(), v.clone());
                    }
                }
            }
            if out.is_empty() {
                None
            } else {
                Some(Value::Object(out))
            }
        }
        (b, f) if b == f => None,
        (_, f) => Some(f.clone()),
    }
}

impl Settings {
    pub fn from_overrides(overrides: &Value) -> Result<(Settings, Value)> {
        let mut full = defaults_value();
        merge(&mut full, overrides);
        let s: Settings = serde_json::from_value(full.clone()).context("settings do not validate")?;
        s.validate()?;
        Ok((s, full))
    }

    pub fn validate(&self) -> Result<()> {
        let sc = &self.scoring;
        if sc.horizon_h <= 0.0 || sc.swap_margin < 1.0 || sc.min_ratio < 0.0 || sc.max_download_h <= 0.0 {
            bail!("scoring: horizon_h > 0, swap_margin >= 1, min_ratio >= 0, max_download_h > 0");
        }
        if self.intake.size_gb[0] < 0.0 || self.intake.size_gb[1] < self.intake.size_gb[0] {
            bail!("intake.size_gb must be [min, max] with 0 <= min <= max");
        }
        if self.reannounce.early.every_s < 1.0 || self.reannounce.late.every_s < self.reannounce.early.every_s {
            bail!("reannounce: early.every_s >= 1 and late.every_s >= early.every_s");
        }
        if self.politeness.torrent_fetch_every_s < 1.0 || self.politeness.scrape_per_s <= 0.0 || self.politeness.scrape_per_s > 5.0 {
            bail!("politeness: torrent_fetch_every_s >= 1, 0 < scrape_per_s <= 5");
        }
        if !(0.0..=1.0).contains(&self.experiments.join_delay.share) {
            bail!("experiments.join_delay.share is a fraction");
        }
        for (name, p) in &self.policy {
            if !["baseline", "model", "shadow", "manual"].contains(&p.as_str()) {
                bail!("policy.{name}: {p} is not baseline | model | shadow | manual");
            }
        }
        for (name, b) in &self.boxes {
            if b.disk.reserve_gb < 0.0 || b.disk.quota_gb < 0.0 || b.download_cap_mib < 0.0 {
                bail!("boxes.{name}: quotas, reserves and caps are not negative");
            }
            if let Some(d) = b.traffic.reset_day {
                if !(1..=28).contains(&d) {
                    bail!("boxes.{name}.traffic.reset_day is 1..28");
                }
            }
        }
        if let Some(v) = &self.account.vip_until {
            chrono::NaiveDate::parse_from_str(v, "%Y-%m-%d").context("account.vip_until is YYYY-MM-DD")?;
        }
        Ok(())
    }

    pub fn box_settings(&self, name: &str) -> BoxSettings {
        self.boxes.get(name).cloned().unwrap_or_default()
    }

    /// The baseline acts on a box whose policy is baseline (or shadow: a model
    /// beside it only logs).
    pub fn baseline_acts(&self, name: &str) -> bool {
        matches!(self.policy.get(name).map(|s| s.as_str()).unwrap_or("baseline"), "baseline" | "shadow")
    }

    pub fn bias(&self, name: &str, source: &str) -> f64 {
        let b = self.box_settings(name);
        let over = match source {
            "race" => b.scoring.bias.race,
            _ => b.scoring.bias.scout,
        };
        over.unwrap_or(match source {
            "race" => self.scoring.bias.race.unwrap_or(1.0),
            _ => self.scoring.bias.scout.unwrap_or(1.0),
        })
    }

    pub fn prior(&self, name: &str, source: &str) -> f64 {
        let p = self.scoring.priors.get(name).or_else(|| self.scoring.priors.get("default"));
        match (p, source) {
            (Some(p), "race") => p.race,
            (Some(p), _) => p.scout,
            (None, _) => 0.4,
        }
    }

    pub fn min_ratio(&self, name: &str) -> f64 {
        self.box_settings(name).scoring.min_ratio.unwrap_or(self.scoring.min_ratio)
    }
    pub fn swap_margin(&self, name: &str) -> f64 {
        self.box_settings(name).scoring.swap_margin.unwrap_or(self.scoring.swap_margin)
    }
    pub fn keep_above(&self, name: &str) -> f64 {
        self.box_settings(name).scoring.keep_above_mib.unwrap_or(self.scoring.keep_above_mib) * MIB
    }
    pub fn max_download_h(&self, name: &str) -> f64 {
        self.box_settings(name).scoring.max_download_h.unwrap_or(self.scoring.max_download_h)
    }
    pub fn horizon_h(&self, name: &str) -> f64 {
        self.box_settings(name).scoring.horizon_h.unwrap_or(self.scoring.horizon_h)
    }

    /// True while the account is VIP.
    pub fn vip_now(&self) -> bool {
        self.vip_until_ts().map(|t| chrono::Utc::now().timestamp() < t).unwrap_or(false)
    }
    /// The end of `vip_until` (UTC midnight after it), as a Unix time.
    pub fn vip_until_ts(&self) -> Option<i64> {
        let d = chrono::NaiveDate::parse_from_str(self.account.vip_until.as_deref()?, "%Y-%m-%d").ok()?;
        Some(d.succ_opt()?.and_hms_opt(0, 0, 0)?.and_utc().timestamp())
    }
}

pub const MIB: f64 = 1_048_576.0;
pub const GB: f64 = 1e9;
pub const TB: f64 = 1e12;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_validate() {
        let (s, _) = Settings::from_overrides(&json!({})).unwrap();
        assert_eq!(s.bias("home", "race"), 0.0);
        assert_eq!(s.bias("pulsed", "race"), 1.0);
        assert_eq!(s.bias("seedit4me", "scout"), 0.0);
        assert!(s.baseline_acts("pulsed"));
    }

    #[test]
    fn overrides_round_trip() {
        let (_, full) = Settings::from_overrides(&json!({"scoring": {"swap_margin": 2.0}})).unwrap();
        let d = diff(&defaults_value(), &full).unwrap();
        assert_eq!(d, json!({"scoring": {"swap_margin": 2.0}}));
    }

    #[test]
    fn rejects_bad_policy() {
        assert!(Settings::from_overrides(&json!({"policy": {"home": "yolo"}})).is_err());
    }
}
