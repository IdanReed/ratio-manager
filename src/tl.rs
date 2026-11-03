//! TorrentLeech (`.torrent` fetches by RSS key, tracker scrapes, the RSS feed)
//! and Prowlarr (the listing). Every request to TL is paced (guard rule 7), and
//! no URL is ever logged: the RSS key and the passkey ride in them.

use crate::bencode::{scrape_reply, torrent_facts, TorrentFacts};
use crate::qbit::strip_url;
use anyhow::{anyhow, bail, Result};
use regex::Regex;
use reqwest::Client;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

static TL_ID: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"torrentleech\.[a-z]+/torrent/(\d+)").unwrap());

pub fn tl_id_of(fields: &[&str]) -> Option<String> {
    fields.iter().find_map(|f| TL_ID.captures(f).map(|c| c[1].to_string()))
}

/// At most one request per `every` (for the tasks that wait their turn). A
/// slot is reserved under the lock and slept for outside it, so an urgent
/// request never queues behind a waiting one.
pub struct Pacer {
    next: Mutex<Instant>,
}

impl Pacer {
    pub fn new() -> Self {
        Pacer { next: Mutex::new(Instant::now()) }
    }
    pub async fn wait(&self, every: Duration) {
        let slot = {
            let mut next = self.next.lock().unwrap_or_else(|e| e.into_inner());
            let at = (*next).max(Instant::now());
            *next = at + every;
            at
        };
        let now = Instant::now();
        if slot > now {
            tokio::time::sleep(slot - now).await;
        }
    }
    /// Counts a request that could not wait (a race), pushing the next
    /// background request back.
    pub async fn note(&self, every: Duration) {
        let mut next = self.next.lock().unwrap_or_else(|e| e.into_inner());
        *next = (*next).max(Instant::now()) + every;
    }
}

impl Default for Pacer {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Tl {
    client: Client,
    rss_key: String,
    rss_url: String,
    base_url: String,
    pub fetch_pacer: Pacer,
    pub scrape_pacer: Pacer,
    /// The tracker's announce URL (it carries the passkey): learned from the
    /// first `.torrent`, held in memory only.
    announce: Mutex<Option<String>>,
    /// 0 unknown, 1 batched scrapes work, 2 one hash per request.
    batch: AtomicU8,
    cache: Mutex<(VecDeque<String>, HashMap<String, Arc<Vec<u8>>>)>,
}

const CACHE: usize = 400;

impl Tl {
    pub fn new(rss_key: String, rss_url: String, base_url: String) -> Tl {
        Tl {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .connect_timeout(Duration::from_secs(10))
                .user_agent("ratio-manager")
                .build()
                .expect("http client"),
            rss_key,
            rss_url,
            base_url: base_url.trim_end_matches('/').to_string(),
            fetch_pacer: Pacer::new(),
            scrape_pacer: Pacer::new(),
            announce: Mutex::new(None),
            batch: AtomicU8::new(0),
            cache: Mutex::new((VecDeque::new(), HashMap::new())),
        }
    }

    pub fn configured(&self) -> bool {
        !crate::config::placeholder(&self.rss_key)
    }

    pub fn cached(&self, tl_id: &str) -> Option<Arc<Vec<u8>>> {
        self.cache.lock().ok().and_then(|c| c.1.get(tl_id).cloned())
    }

    fn remember(&self, tl_id: &str, raw: Arc<Vec<u8>>) {
        if let Ok(mut c) = self.cache.lock() {
            if c.1.insert(tl_id.to_string(), raw).is_none() {
                c.0.push_back(tl_id.to_string());
            }
            while c.0.len() > CACHE {
                if let Some(old) = c.0.pop_front() {
                    c.1.remove(&old);
                }
            }
        }
    }

    /// The `.torrent` for a release, from TL by RSS key. `urgent` (a race)
    /// doesn't wait for the pacer but still counts against it.
    pub async fn fetch(&self, tl_id: &str, every: Duration, urgent: bool) -> Result<(Arc<Vec<u8>>, TorrentFacts)> {
        if let Some(raw) = self.cached(tl_id) {
            let facts = torrent_facts(&raw)?;
            return Ok((raw, facts));
        }
        if !self.configured() {
            bail!("the TL RSS key is unset");
        }
        if urgent {
            self.fetch_pacer.note(every).await;
        } else {
            self.fetch_pacer.wait(every).await;
        }
        let url = format!("{}/rss/download/{tl_id}/{}/{tl_id}.torrent", self.base_url, self.rss_key);
        let mut last = String::new();
        for attempt in 0..2 {
            match self.client.get(&url).send().await {
                Ok(r) if r.status().is_success() => {
                    let body = r.bytes().await.map_err(|e| anyhow!("{}", strip_url(&e)))?.to_vec();
                    if body.first() != Some(&b'd') {
                        bail!("TL returned no .torrent for {tl_id}");
                    }
                    let facts = torrent_facts(&body)?;
                    if facts.announce.contains("/announce") {
                        if let Ok(mut a) = self.announce.lock() {
                            *a = Some(facts.announce.clone());
                        }
                    }
                    let raw = Arc::new(body);
                    self.remember(tl_id, raw.clone());
                    return Ok((raw, facts));
                }
                Ok(r) => last = format!("HTTP {}", r.status()),
                Err(e) => last = strip_url(&e),
            }
            if attempt == 0 {
                tokio::time::sleep(Duration::from_secs(3)).await;
            }
        }
        bail!("TL download of {tl_id} failed ({last})")
    }

    pub fn can_scrape(&self) -> bool {
        self.announce.lock().map(|a| a.is_some()).unwrap_or(false)
    }

    /// Live (complete, incomplete, downloaded) for each hash. Batched when the
    /// tracker answers several hashes at once (learned on the first call).
    pub async fn scrape(&self, hashes: &[String], per_s: f64, batch: usize) -> Result<BTreeMap<String, (i64, i64, i64)>> {
        let ann = self.announce.lock().ok().and_then(|a| a.clone()).ok_or_else(|| anyhow!("no announce URL learned yet"))?;
        let base = ann.replacen("/announce", "/scrape", 1);
        let every = Duration::from_secs_f64(1.0 / per_s.max(0.01));
        let mut out = BTreeMap::new();
        let mode = self.batch.load(Ordering::Relaxed);
        let size = if mode == 2 { 1 } else { batch.max(1) };
        for chunk in hashes.chunks(size) {
            self.scrape_pacer.wait(every).await;
            let mut url = base.clone();
            let mut sep = if url.contains('?') { '&' } else { '?' };
            for h in chunk {
                let bytes = hex::decode(h)?;
                url.push(sep);
                sep = '&';
                url.push_str("info_hash=");
                url.push_str(&percent_encoding::percent_encode(&bytes, percent_encoding::NON_ALPHANUMERIC).to_string());
            }
            let r = self.client.get(&url).send().await.map_err(|e| anyhow!("scrape: {}", strip_url(&e)))?;
            if !r.status().is_success() {
                bail!("scrape: HTTP {}", r.status());
            }
            let body = r.bytes().await.map_err(|e| anyhow!("scrape: {}", strip_url(&e)))?;
            let got = scrape_reply(&body)?;
            if chunk.len() >= 2 && mode == 0 {
                let m = if got.len() >= 2 { 1 } else { 2 };
                self.batch.store(m, Ordering::Relaxed);
                tracing::info!("TL scrape: {}", if m == 1 { "batched requests work" } else { "one hash per request" });
                if m == 2 {
                    // Redo the rest one at a time.
                    let missing: Vec<String> = chunk.iter().filter(|h| !got.contains_key(*h)).cloned().collect();
                    out.extend(got);
                    let rest = Box::pin(self.scrape(&missing, per_s, 1)).await?;
                    out.extend(rest);
                    continue;
                }
            }
            out.extend(got);
        }
        Ok(out)
    }

    /// The RSS feed's items: (tl_id, title, category, published unix time).
    pub async fn rss(&self) -> Result<Vec<RssItem>> {
        if !self.configured() {
            bail!("the TL RSS key is unset");
        }
        let url = self.rss_url.replace("{key}", &self.rss_key);
        let r = self.client.get(&url).send().await.map_err(|e| anyhow!("rss: {}", strip_url(&e)))?;
        if !r.status().is_success() {
            bail!("rss: HTTP {}", r.status());
        }
        let body = r.text().await.map_err(|e| anyhow!("rss: {}", strip_url(&e)))?;
        Ok(parse_rss(&body))
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RssItem {
    pub tl_id: String,
    pub title: String,
    pub category: String,
    pub published: Option<i64>,
}

pub fn parse_rss(xml: &str) -> Vec<RssItem> {
    use quick_xml::events::Event as X;
    let mut reader = quick_xml::Reader::from_str(xml);
    let mut out = Vec::new();
    let mut in_item = false;
    let mut field = String::new();
    let (mut title, mut cat, mut link, mut guid, mut comments, mut pubd) =
        (String::new(), String::new(), String::new(), String::new(), String::new(), String::new());
    loop {
        match reader.read_event() {
            Ok(X::Start(e)) => {
                let n = String::from_utf8_lossy(e.name().as_ref()).to_string();
                if n == "item" {
                    in_item = true;
                    title.clear();
                    cat.clear();
                    link.clear();
                    guid.clear();
                    comments.clear();
                    pubd.clear();
                }
                field = n;
            }
            Ok(X::Text(t)) if in_item => {
                let v = t.unescape().map(|c| c.to_string()).unwrap_or_default();
                push_field(&field, &v, &mut title, &mut cat, &mut link, &mut guid, &mut comments, &mut pubd);
            }
            Ok(X::CData(t)) if in_item => {
                let v = String::from_utf8_lossy(t.as_ref()).to_string();
                push_field(&field, &v, &mut title, &mut cat, &mut link, &mut guid, &mut comments, &mut pubd);
            }
            Ok(X::End(e)) => {
                let n = String::from_utf8_lossy(e.name().as_ref()).to_string();
                if n == "item" && in_item {
                    in_item = false;
                    let id = tl_id_of(&[&comments, &guid, &link]).or_else(|| {
                        // A download link: /rss/download/<id>/<key>/...
                        Regex::new(r"/download/(\d+)/").ok()?.captures(&link).map(|c| c[1].to_string())
                    });
                    if let Some(tl_id) = id {
                        out.push(RssItem {
                            tl_id,
                            title: title.trim().to_string(),
                            category: cat.trim().to_string(),
                            published: chrono::DateTime::parse_from_rfc2822(pubd.trim()).ok().map(|d| d.timestamp()),
                        });
                    }
                }
                field.clear();
            }
            Ok(X::Eof) | Err(_) => break,
            _ => {}
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn push_field(field: &str, v: &str, title: &mut String, cat: &mut String, link: &mut String, guid: &mut String, comments: &mut String, pubd: &mut String) {
    match field {
        "title" => title.push_str(v),
        "category" => cat.push_str(v),
        "link" => link.push_str(v),
        "guid" => guid.push_str(v),
        "comments" => comments.push_str(v),
        "pubDate" => pubd.push_str(v),
        _ => {}
    }
}

pub struct Prowlarr {
    client: Client,
    url: String,
    key: String,
}

impl Prowlarr {
    pub fn new(url: &str, key: String) -> Prowlarr {
        Prowlarr {
            client: Client::builder().timeout(Duration::from_secs(120)).build().expect("http client"),
            url: url.trim_end_matches('/').to_string(),
            key,
        }
    }

    async fn get(&self, path: &str) -> Result<Value> {
        let mut last = String::new();
        for attempt in 1..=3 {
            match self.client.get(format!("{}{}", self.url, path)).header("X-Api-Key", &self.key).send().await {
                Ok(r) if r.status().is_success() => return Ok(r.json().await?),
                Ok(r) => {
                    let status = r.status();
                    let body = r.text().await.unwrap_or_default();
                    last = format!("HTTP {status} {}", body.chars().take(160).collect::<String>());
                    // A benched indexer (Prowlarr's own SQLite lock) is benched
                    // for about a minute.
                    if attempt < 3 {
                        let wait = if body.contains("unavailable") || body.contains("disabled till") { 70 } else { 15 };
                        tokio::time::sleep(Duration::from_secs(wait)).await;
                    }
                }
                Err(e) => {
                    last = strip_url(&e);
                    if attempt < 3 {
                        tokio::time::sleep(Duration::from_secs(15)).await;
                    }
                }
            }
        }
        bail!("Prowlarr {}: {last}", path.split('?').next().unwrap_or(""))
    }

    /// The id of the first enabled indexer among `names`.
    pub async fn indexer(&self, names: &[String]) -> Result<Option<(i64, String)>> {
        let list = self.get("/api/v1/indexer").await?;
        let arr = list.as_array().cloned().unwrap_or_default();
        for n in names {
            if let Some(i) = arr.iter().find(|i| i.get("name").and_then(|x| x.as_str()) == Some(n.as_str())
                && i.get("enable").and_then(|x| x.as_bool()).unwrap_or(false))
            {
                return Ok(Some((i.get("id").and_then(|x| x.as_i64()).unwrap_or(0), n.clone())));
            }
        }
        Ok(None)
    }

    /// The listing's newest releases, `offset` into it.
    pub async fn newest(&self, indexer: i64, offset: u32, limit: u32) -> Result<Vec<Value>> {
        let v = self
            .get(&format!("/api/v1/search?query=&indexerIds={indexer}&type=search&limit={limit}&offset={offset}"))
            .await?;
        Ok(v.as_array().cloned().unwrap_or_default())
    }

    pub async fn search(&self, indexer: i64, query: &str) -> Result<Vec<Value>> {
        let q: String = percent_encoding::utf8_percent_encode(query, percent_encoding::NON_ALPHANUMERIC).to_string();
        let v = self.get(&format!("/api/v1/search?query={q}&indexerIds={indexer}&type=search&limit=50")).await?;
        Ok(v.as_array().cloned().unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids() {
        assert_eq!(tl_id_of(&["https://www.torrentleech.org/torrent/241234#torrentinfo"]), Some("241234".into()));
        assert_eq!(tl_id_of(&["nope"]), None);
    }

    #[test]
    fn rss_items() {
        let xml = r#"<rss><channel><item><title><![CDATA[Some.Movie.2026.1080p]]></title>
<link>https://www.torrentleech.org/rss/download/123456/KEY/Some.Movie.torrent</link>
<category>Movies :: HD</category><pubDate>Mon, 02 Nov 2026 20:00:00 +0000</pubDate></item></channel></rss>"#;
        let items = parse_rss(xml);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].tl_id, "123456");
        assert_eq!(items[0].title, "Some.Movie.2026.1080p");
        assert_eq!(items[0].category, "Movies :: HD");
        assert!(items[0].published.is_some());
    }
}
