//! qBittorrent WebAPI v2. One instance is one session: the poller, the
//! re-announcer and the executor each hold their own per box, so one slow
//! call never stalls another loop.

use crate::config::BoxCfg;
use anyhow::{anyhow, bail, Result};
use reqwest::{header, Client, StatusCode};
use serde_json::Value;
use std::time::Duration;

#[derive(Clone)]
pub struct Qbit {
    pub name: String,
    base: String,
    user: String,
    pass: String,
    client: Client,
}

impl Qbit {
    pub fn new(cfg: &BoxCfg, timeout_s: u64) -> Result<Qbit> {
        let base = cfg.resolved_url();
        let mut headers = header::HeaderMap::new();
        headers.insert(header::REFERER, header::HeaderValue::from_str(&base)?);
        if cfg.basic_auth {
            let token = base64_encode(format!("{}:{}", cfg.user(), cfg.password()).as_bytes());
            let mut v = header::HeaderValue::from_str(&format!("Basic {token}"))?;
            v.set_sensitive(true);
            headers.insert(header::AUTHORIZATION, v);
        }
        let client = Client::builder()
            .cookie_store(true)
            .default_headers(headers)
            .timeout(Duration::from_secs(timeout_s))
            .connect_timeout(Duration::from_secs(15))
            .user_agent("ratio-manager")
            .build()?;
        Ok(Qbit { name: cfg.name.clone(), base, user: cfg.user(), pass: cfg.password(), client })
    }

    pub async fn login(&self) -> Result<()> {
        let res = self
            .send(|| self.client.post(format!("{}/api/v2/auth/login", self.base)).form(&[("username", &self.user), ("password", &self.pass)]))
            .await?;
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        // "Ok." or an empty 204 with a session cookie; a wrong password is a
        // 200 "Fails.".
        if body.trim() == "Fails." || !(status.is_success()) {
            bail!("{}: qBittorrent refused the WebUI login ({status} {})", self.name, body.chars().take(40).collect::<String>());
        }
        Ok(())
    }

    /// One retry when the connection failed before the request went out (a
    /// busy box's web front now and then hangs a fresh connection).
    async fn send<F>(&self, build: F) -> Result<reqwest::Response>
    where
        F: Fn() -> reqwest::RequestBuilder,
    {
        match build().send().await {
            Ok(r) => Ok(r),
            Err(e) if e.is_connect() => build().send().await.map_err(|e| anyhow!("{}: {}", self.name, strip_url(&e))),
            Err(e) => Err(anyhow!("{}: {}", self.name, strip_url(&e))),
        }
    }

    async fn call<F>(&self, build: F) -> Result<reqwest::Response>
    where
        F: Fn() -> reqwest::RequestBuilder,
    {
        let r = self.send(&build).await?;
        if r.status() == StatusCode::FORBIDDEN {
            self.login().await?;
            let r = self.send(&build).await?;
            return check(r, &self.name).await;
        }
        check(r, &self.name).await
    }

    pub async fn get_json(&self, path: &str) -> Result<Value> {
        let url = format!("{}/api/v2/{}", self.base, path);
        let r = self.call(|| self.client.get(&url)).await?;
        let text = r.text().await?;
        Ok(serde_json::from_str(&text).unwrap_or(Value::String(text)))
    }

    pub async fn get_bytes(&self, path: &str) -> Result<Vec<u8>> {
        let url = format!("{}/api/v2/{}", self.base, path);
        let r = self.call(|| self.client.get(&url)).await?;
        Ok(r.bytes().await?.to_vec())
    }

    pub async fn post(&self, path: &str, form: &[(&str, String)]) -> Result<String> {
        let url = format!("{}/api/v2/{}", self.base, path);
        let form: Vec<(String, String)> = form.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
        let r = self.call(|| self.client.post(&url).form(&form)).await?;
        Ok(r.text().await.unwrap_or_default())
    }

    pub async fn maindata(&self, rid: i64) -> Result<Value> {
        self.get_json(&format!("sync/maindata?rid={rid}")).await
    }

    pub async fn torrents(&self, query: &str) -> Result<Vec<Value>> {
        match self.get_json(&format!("torrents/info{query}")).await? {
            Value::Array(a) => Ok(a),
            other => bail!("{}: torrents/info returned {other}", self.name),
        }
    }

    pub async fn trackers(&self, hash: &str) -> Result<Vec<Value>> {
        match self.get_json(&format!("torrents/trackers?hash={hash}")).await? {
            Value::Array(a) => Ok(a),
            _ => Ok(vec![]),
        }
    }

    pub async fn properties(&self, hash: &str) -> Result<Value> {
        self.get_json(&format!("torrents/properties?hash={hash}")).await
    }

    pub async fn preferences(&self) -> Result<Value> {
        self.get_json("app/preferences").await
    }

    pub async fn set_preferences(&self, prefs: &Value) -> Result<()> {
        self.post("app/setPreferences", &[("json", prefs.to_string())]).await.map(|_| ())
    }

    pub async fn reannounce(&self, hashes: &[String]) -> Result<()> {
        if hashes.is_empty() {
            return Ok(());
        }
        self.post("torrents/reannounce", &[("hashes", hashes.join("|"))]).await.map(|_| ())
    }

    pub async fn delete(&self, hashes: &[String], files: bool) -> Result<()> {
        self.post("torrents/delete", &[("hashes", hashes.join("|")), ("deleteFiles", files.to_string())]).await.map(|_| ())
    }

    /// `stop`/`start` since WebAPI 2.11 (qBittorrent 5); `pause`/`resume` before.
    pub async fn stop(&self, hashes: &[String]) -> Result<()> {
        let h = hashes.join("|");
        match self.post("torrents/stop", &[("hashes", h.clone())]).await {
            Ok(_) => Ok(()),
            Err(_) => self.post("torrents/pause", &[("hashes", h)]).await.map(|_| ()),
        }
    }

    pub async fn start(&self, hashes: &[String]) -> Result<()> {
        let h = hashes.join("|");
        match self.post("torrents/start", &[("hashes", h.clone())]).await {
            Ok(_) => Ok(()),
            Err(_) => self.post("torrents/resume", &[("hashes", h)]).await.map(|_| ()),
        }
    }

    pub async fn set_download_limit(&self, hashes: &[String], limit: i64) -> Result<()> {
        self.post("torrents/setDownloadLimit", &[("hashes", hashes.join("|")), ("limit", limit.to_string())]).await.map(|_| ())
    }

    pub async fn peers(&self, hash: &str) -> Result<Value> {
        self.get_json(&format!("sync/torrentPeers?hash={hash}&rid=0")).await
    }

    pub async fn export(&self, hash: &str) -> Result<Vec<u8>> {
        self.get_bytes(&format!("torrents/export?hash={hash}")).await
    }

    pub async fn categories(&self) -> Result<Value> {
        self.get_json("torrents/categories").await
    }

    pub async fn add(&self, raw: &[u8], hash: &str, fields: &[(&str, String)]) -> Result<()> {
        let url = format!("{}/api/v2/torrents/add", self.base);
        let raw = raw.to_vec();
        let fields: Vec<(String, String)> = fields.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
        let build = || {
            let mut form = reqwest::multipart::Form::new();
            for (k, v) in &fields {
                form = form.text(k.clone(), v.clone());
            }
            let part = reqwest::multipart::Part::bytes(raw.clone())
                .file_name(format!("{hash}.torrent"))
                .mime_str("application/x-bittorrent")
                .expect("static mime");
            self.client.post(&url).multipart(form.part("torrents", part))
        };
        let r = self.call(build).await?;
        let body = r.text().await.unwrap_or_default();
        // "Ok." before WebAPI 2.15, a JSON count since.
        let ok = body.trim() == "Ok."
            || serde_json::from_str::<Value>(&body)
                .ok()
                .and_then(|v| v.get("success_count").and_then(|c| c.as_i64()))
                .map(|c| c > 0)
                .unwrap_or(false);
        if !ok {
            bail!("{}: qBittorrent did not add it ({})", self.name, body.chars().take(120).collect::<String>());
        }
        Ok(())
    }
}

async fn check(r: reqwest::Response, name: &str) -> Result<reqwest::Response> {
    if r.status().is_success() {
        return Ok(r);
    }
    let status = r.status();
    let body = r.text().await.unwrap_or_default();
    bail!("{name}: HTTP {status} {}", body.chars().take(160).collect::<String>())
}

/// A reqwest error without its URL (a URL can carry a passkey).
pub fn strip_url(e: &reqwest::Error) -> String {
    let kind = if e.is_timeout() {
        "timed out"
    } else if e.is_connect() {
        "connect failed"
    } else if e.is_decode() {
        "bad body"
    } else if e.is_status() {
        "bad status"
    } else {
        "request failed"
    };
    match e.status() {
        Some(s) => format!("{kind} ({s})"),
        None => kind.to_string(),
    }
}

pub fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

// Field accessors over a torrent's JSON (maindata merges partial updates).
pub fn f(t: &Value, k: &str) -> f64 {
    t.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0)
}
pub fn i(t: &Value, k: &str) -> i64 {
    t.get(k).and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|x| x as i64))).unwrap_or(0)
}
pub fn s<'a>(t: &'a Value, k: &str) -> &'a str {
    t.get(k).and_then(|v| v.as_str()).unwrap_or("")
}
pub fn tags(t: &Value) -> Vec<String> {
    s(t, "tags").split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()
}

/// States that hold a download slot. stalledDL does not: a swarm whose
/// seeders left can stall a torrent for hours.
pub fn downloading(t: &Value) -> bool {
    matches!(s(t, "state"), "downloading" | "metaDL" | "forcedMetaDL" | "queuedDL" | "forcedDL" | "checkingDL" | "allocating")
}

pub fn finished(t: &Value) -> bool {
    f(t, "progress") >= 1.0
}

/// Leechers per seeder across the swarm the tracker reports.
pub fn swarm_demand(t: &Value) -> f64 {
    f(t, "num_incomplete") / (f(t, "num_complete") + 1.0)
}

#[cfg(test)]
mod tests {
    #[test]
    fn b64() {
        assert_eq!(super::base64_encode(b"user:pass"), "dXNlcjpwYXNz");
        assert_eq!(super::base64_encode(b"a"), "YQ==");
    }
}
