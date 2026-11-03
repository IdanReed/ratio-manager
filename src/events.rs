//! Every event goes through the bus: the event log (daily `.jsonl.zst`, one
//! zstd frame per flush), the SSE stream, and — for deltas only — the policy's
//! lanes. Observations never wake the policy.

use crate::store::Store;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{broadcast, mpsc, Notify};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub v: u8,
    pub ts: DateTime<Utc>,
    pub seq: u64,
    pub kind: String,
    #[serde(rename = "box")]
    pub bx: Option<String>,
    pub hash: Option<String>,
    pub actor: Option<String>,
    pub data: Value,
}

/// Kinds that wake the policy, and on which lane.
pub fn lane_of(kind: &str) -> Option<Lane> {
    match kind {
        "announce" => Some(Lane::Fast),
        "market" | "capacity" | "tick" | "setting" => Some(Lane::Normal),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    Fast,
    Normal,
}

/// The normal lane coalesces: of several waiting deltas for one release or
/// box, only the latest is processed.
pub struct Coalescing {
    inner: Mutex<(VecDeque<String>, HashMap<String, Event>)>,
    notify: Notify,
}

impl Coalescing {
    fn new() -> Self {
        Coalescing { inner: Mutex::new((VecDeque::new(), HashMap::new())), notify: Notify::new() }
    }
    fn key(e: &Event) -> String {
        format!("{}|{}|{}", e.kind, e.bx.as_deref().unwrap_or(""), e.hash.as_deref().unwrap_or(""))
    }
    pub fn push(&self, e: Event) {
        let k = Self::key(&e);
        let mut g = self.inner.lock().unwrap_or_else(|x| x.into_inner());
        if g.1.insert(k.clone(), e).is_none() {
            g.0.push_back(k);
        }
        drop(g);
        self.notify.notify_one();
    }
    pub fn try_pop(&self) -> Option<Event> {
        let mut g = self.inner.lock().unwrap_or_else(|x| x.into_inner());
        while let Some(k) = g.0.pop_front() {
            if let Some(e) = g.1.remove(&k) {
                return Some(e);
            }
        }
        None
    }
    pub async fn pop(&self) -> Event {
        loop {
            if let Some(e) = self.try_pop() {
                return e;
            }
            self.notify.notified().await;
        }
    }
    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.1.len()).unwrap_or(0)
    }
}

pub struct Bus {
    seq: AtomicU64,
    log_tx: mpsc::UnboundedSender<Event>,
    pub sse: broadcast::Sender<Event>,
    pub fast: mpsc::UnboundedSender<Event>,
    pub normal: Arc<Coalescing>,
    recent: Mutex<VecDeque<Event>>,
}

pub struct BusReceivers {
    pub log_rx: mpsc::UnboundedReceiver<Event>,
    pub fast_rx: mpsc::UnboundedReceiver<Event>,
}

const RECENT: usize = 5000;

impl Bus {
    pub fn new(start_seq: u64) -> (Arc<Bus>, BusReceivers) {
        let (log_tx, log_rx) = mpsc::unbounded_channel();
        let (fast, fast_rx) = mpsc::unbounded_channel();
        let (sse, _) = broadcast::channel(8192);
        let bus = Arc::new(Bus {
            seq: AtomicU64::new(start_seq),
            log_tx,
            sse,
            fast,
            normal: Arc::new(Coalescing::new()),
            recent: Mutex::new(VecDeque::with_capacity(RECENT)),
        });
        (bus, BusReceivers { log_rx, fast_rx })
    }

    pub fn last_seq(&self) -> u64 {
        self.seq.load(Ordering::SeqCst)
    }

    pub fn emit(&self, kind: &str, bx: Option<&str>, hash: Option<&str>, actor: Option<&str>, data: Value) -> Event {
        let e = Event {
            v: 1,
            ts: Utc::now(),
            seq: self.seq.fetch_add(1, Ordering::SeqCst) + 1,
            kind: kind.to_string(),
            bx: bx.map(|s| s.to_string()),
            hash: hash.map(|s| s.to_string()),
            actor: actor.map(|s| s.to_string()),
            data,
        };
        let _ = self.log_tx.send(e.clone());
        let _ = self.sse.send(e.clone());
        {
            let mut r = self.recent.lock().unwrap_or_else(|x| x.into_inner());
            if r.len() >= RECENT {
                r.pop_front();
            }
            r.push_back(e.clone());
        }
        match lane_of(kind) {
            Some(Lane::Fast) => {
                let _ = self.fast.send(e.clone());
            }
            Some(Lane::Normal) => self.normal.push(e.clone()),
            None => {}
        }
        e
    }

    /// Events after `since` still held in memory, or None when `since` is
    /// older than the in-memory window (read the files instead).
    pub fn recent_since(&self, since: u64, kinds: &Option<Vec<String>>, limit: usize) -> Option<Vec<Event>> {
        let r = self.recent.lock().unwrap_or_else(|x| x.into_inner());
        let first = r.front().map(|e| e.seq).unwrap_or(u64::MAX);
        if since + 1 < first && !r.is_empty() {
            return None;
        }
        Some(
            r.iter()
                .filter(|e| e.seq > since)
                .filter(|e| kinds.as_ref().map(|k| k.contains(&e.kind)).unwrap_or(true))
                .take(limit)
                .cloned()
                .collect(),
        )
    }
}

/// Appends events to `<dir>/YYYY-MM-DD.jsonl.zst`, one zstd frame per flush,
/// so a crash loses at most the unflushed batch and every complete frame stays
/// readable (`zstd -dc` reads a multi-frame file).
pub struct EventLog {
    pub dir: PathBuf,
}

impl EventLog {
    pub fn path_for(&self, day: &str) -> PathBuf {
        self.dir.join(format!("{day}.jsonl.zst"))
    }

    pub async fn run(self, mut rx: mpsc::UnboundedReceiver<Event>, store: Arc<Store>) {
        let _ = std::fs::create_dir_all(&self.dir);
        let mut batch: Vec<Event> = Vec::new();
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(2));
        loop {
            tokio::select! {
                e = rx.recv() => match e {
                    Some(e) => {
                        batch.push(e);
                        if batch.len() >= 2000 {
                            self.flush(&mut batch, &store);
                        }
                    }
                    None => {
                        self.flush(&mut batch, &store);
                        return;
                    }
                },
                _ = tick.tick() => self.flush(&mut batch, &store),
            }
        }
    }

    fn flush(&self, batch: &mut Vec<Event>, store: &Store) {
        if batch.is_empty() {
            return;
        }
        let mut by_day: std::collections::BTreeMap<String, Vec<&Event>> = Default::default();
        for e in batch.iter() {
            by_day.entry(e.ts.format("%Y-%m-%d").to_string()).or_default().push(e);
        }
        for (day, evs) in by_day {
            let mut raw = Vec::with_capacity(evs.len() * 256);
            let mut counts: HashMap<&str, (i64, i64)> = HashMap::new();
            for e in &evs {
                let line = serde_json::to_vec(e).unwrap_or_default();
                let c = counts.entry(e.kind.as_str()).or_default();
                c.0 += 1;
                c.1 += line.len() as i64 + 1;
                raw.extend_from_slice(&line);
                raw.push(b'\n');
            }
            let frame = match zstd::encode_all(&raw[..], 9) {
                Ok(f) => f,
                Err(err) => {
                    tracing::warn!("event log: zstd failed ({err}); batch dropped");
                    continue;
                }
            };
            let path = self.path_for(&day);
            let res = std::fs::OpenOptions::new().create(true).append(true).open(&path).and_then(|mut f| {
                f.write_all(&frame)?;
                f.flush()
            });
            if let Err(err) = res {
                tracing::warn!("event log: cannot append to {} ({err})", path.display());
                continue;
            }
            for (k, (n, b)) in counts {
                store.count_day(&day, k, n, b);
            }
            let first = evs.first().map(|e| e.seq).unwrap_or(0);
            let last = evs.iter().map(|e| e.seq).max().unwrap_or(0);
            store.note_day(&day, first, last);
            store.set("seq", &last.max(store.get("seq").and_then(|s| s.parse().ok()).unwrap_or(0)).to_string());
        }
        batch.clear();
    }

    /// Every event in a day file, in order. A torn last frame (a crash
    /// mid-write) ends the read rather than failing it.
    pub fn read_day(&self, day: &str) -> Vec<Event> {
        let Ok(data) = std::fs::read(self.path_for(day)) else { return vec![] };
        let mut out = Vec::new();
        let mut dec = match zstd::stream::read::Decoder::new(&data[..]) {
            Ok(d) => d,
            Err(_) => return out,
        };
        let mut buf = Vec::new();
        let _ = std::io::Read::read_to_end(&mut dec, &mut buf);
        for line in buf.split(|b| *b == b'\n') {
            if line.is_empty() {
                continue;
            }
            if let Ok(e) = serde_json::from_slice::<Event>(line) {
                out.push(e);
            }
        }
        out
    }

    /// The highest seq on disk, for a store that lost its cursor.
    pub fn last_seq_on_disk(&self) -> u64 {
        let mut days: Vec<String> = std::fs::read_dir(&self.dir)
            .map(|it| {
                it.filter_map(|e| e.ok())
                    .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
                    .filter(|n| n.ends_with(".jsonl.zst"))
                    .map(|n| n.trim_end_matches(".jsonl.zst").to_string())
                    .collect()
            })
            .unwrap_or_default();
        days.sort();
        days.last().map(|d| self.read_day(d).iter().map(|e| e.seq).max().unwrap_or(0)).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalescing_keeps_latest_per_key() {
        let c = Coalescing::new();
        let mk = |seq: u64, hash: &str| Event {
            v: 1,
            ts: Utc::now(),
            seq,
            kind: "market".into(),
            bx: None,
            hash: Some(hash.into()),
            actor: None,
            data: Value::Null,
        };
        c.push(mk(1, "a"));
        c.push(mk(2, "b"));
        c.push(mk(3, "a"));
        assert_eq!(c.try_pop().unwrap().seq, 3);
        assert_eq!(c.try_pop().unwrap().seq, 2);
        assert!(c.try_pop().is_none());
    }

    #[test]
    fn log_round_trips_multi_frame() {
        let dir = std::env::temp_dir().join(format!("rm-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let log = EventLog { dir: dir.clone() };
        let store = Store::open_memory().unwrap();
        let day = Utc::now().format("%Y-%m-%d").to_string();
        for round in 0..3u64 {
            let mut batch: Vec<Event> = (0..10)
                .map(|i| Event {
                    v: 1,
                    ts: Utc::now(),
                    seq: round * 10 + i + 1,
                    kind: "tick".into(),
                    bx: None,
                    hash: None,
                    actor: None,
                    data: serde_json::json!({"i": i}),
                })
                .collect();
            log.flush(&mut batch, &store);
        }
        let evs = log.read_day(&day);
        assert_eq!(evs.len(), 30);
        assert_eq!(evs.last().unwrap().seq, 30);
        assert_eq!(log.last_seq_on_disk(), 30);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
