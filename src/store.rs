//! SQLite on /mnt/fast: settings and their audit, rollups, the release book,
//! races and their outcomes, decisions, cursors. The event log itself lives
//! in daily files (events.rs); this holds what the API and the policy query.

use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::Value;
use std::sync::Mutex;

pub struct Store {
    conn: Mutex<Connection>,
}

const SCHEMA: &str = r#"
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY, ts INTEGER NOT NULL, who TEXT NOT NULL,
  old TEXT NOT NULL, new TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rollups (
  minute INTEGER NOT NULL, box TEXT NOT NULL,
  up REAL, dl REAL, n_dl INTEGER, n_seed INTEGER, n_total INTEGER,
  disk_used REAL, disk_quota REAL, traffic_used REAL, traffic_quota REAL,
  uploaded REAL, leechers INTEGER,
  PRIMARY KEY (minute, box));
CREATE TABLE IF NOT EXISTS races (
  box TEXT NOT NULL, hash TEXT NOT NULL, name TEXT, source TEXT, category TEXT,
  size REAL, added_at INTEGER, completed_at INTEGER, demand_at_add REAL,
  announce_ts INTEGER, registered_after REAL,
  ratio_1h REAL, ratio_3h REAL, ratio_24h REAL, ratio_7d REAL,
  last_ratio REAL, last_uploaded REAL, removed_at INTEGER, removed_rule TEXT,
  PRIMARY KEY (box, hash));
CREATE INDEX IF NOT EXISTS races_added ON races (added_at);
CREATE TABLE IF NOT EXISTS decisions (
  seq INTEGER PRIMARY KEY, ts INTEGER NOT NULL, actor TEXT, trigger TEXT,
  hash TEXT, name TEXT, chosen TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS releases (
  tl_id TEXT PRIMARY KEY, name TEXT, category TEXT, size REAL,
  freeleech INTEGER, first_seen INTEGER, hash TEXT, private INTEGER,
  announce_ts INTEGER, seeders INTEGER, leechers INTEGER, completed INTEGER,
  last_scrape INTEGER, delta_l REAL, delta_s REAL, delta_ts INTEGER,
  fetch_failed INTEGER DEFAULT 0, sched_idx INTEGER DEFAULT 0, source TEXT);
CREATE INDEX IF NOT EXISTS releases_hash ON releases (hash);
CREATE INDEX IF NOT EXISTS releases_seen ON releases (first_seen);
CREATE TABLE IF NOT EXISTS verified (box TEXT NOT NULL, hash TEXT NOT NULL, PRIMARY KEY (box, hash));
CREATE TABLE IF NOT EXISTS removed (box TEXT NOT NULL, hash TEXT NOT NULL, ts INTEGER NOT NULL, PRIMARY KEY (box, hash));
CREATE TABLE IF NOT EXISTS day_counts (day TEXT NOT NULL, kind TEXT NOT NULL, n INTEGER NOT NULL, bytes INTEGER NOT NULL, PRIMARY KEY (day, kind));
CREATE TABLE IF NOT EXISTS days (day TEXT PRIMARY KEY, first_seq INTEGER, last_seq INTEGER);
CREATE TABLE IF NOT EXISTS traffic (box TEXT PRIMARY KEY, period TEXT NOT NULL, base REAL NOT NULL);
"#;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Release {
    pub tl_id: String,
    pub name: String,
    pub category: String,
    pub size: f64,
    pub freeleech: bool,
    pub first_seen: i64,
    pub hash: Option<String>,
    pub private: Option<bool>,
    pub announce_ts: Option<i64>,
    pub seeders: Option<i64>,
    pub leechers: Option<i64>,
    pub completed: Option<i64>,
    pub last_scrape: Option<i64>,
    pub delta_l: Option<f64>,
    pub delta_s: Option<f64>,
    pub delta_ts: Option<i64>,
    pub fetch_failed: i64,
    pub sched_idx: i64,
    pub source: String,
}

fn row_release(r: &rusqlite::Row) -> rusqlite::Result<Release> {
    Ok(Release {
        tl_id: r.get(0)?,
        name: r.get::<_, Option<String>>(1)?.unwrap_or_default(),
        category: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
        size: r.get::<_, Option<f64>>(3)?.unwrap_or(0.0),
        freeleech: r.get::<_, Option<i64>>(4)?.unwrap_or(0) != 0,
        first_seen: r.get::<_, Option<i64>>(5)?.unwrap_or(0),
        hash: r.get(6)?,
        private: r.get::<_, Option<i64>>(7)?.map(|v| v != 0),
        announce_ts: r.get(8)?,
        seeders: r.get(9)?,
        leechers: r.get(10)?,
        completed: r.get(11)?,
        last_scrape: r.get(12)?,
        delta_l: r.get(13)?,
        delta_s: r.get(14)?,
        delta_ts: r.get(15)?,
        fetch_failed: r.get::<_, Option<i64>>(16)?.unwrap_or(0),
        sched_idx: r.get::<_, Option<i64>>(17)?.unwrap_or(0),
        source: r.get::<_, Option<String>>(18)?.unwrap_or_default(),
    })
}

const RELEASE_COLS: &str = "tl_id, name, category, size, freeleech, first_seen, hash, private, announce_ts, seeders, leechers, completed, last_scrape, delta_l, delta_s, delta_ts, fetch_failed, sched_idx, source";

#[derive(Debug, Clone, serde::Serialize, Default)]
pub struct Race {
    pub box_: String,
    pub hash: String,
    pub name: String,
    pub source: String,
    pub category: String,
    pub size: f64,
    pub added_at: i64,
    pub completed_at: Option<i64>,
    pub demand_at_add: Option<f64>,
    pub announce_ts: Option<i64>,
    pub registered_after: Option<f64>,
    pub ratio_1h: Option<f64>,
    pub ratio_3h: Option<f64>,
    pub ratio_24h: Option<f64>,
    pub ratio_7d: Option<f64>,
    pub last_ratio: Option<f64>,
    pub last_uploaded: Option<f64>,
    pub removed_at: Option<i64>,
    pub removed_rule: Option<String>,
}

fn row_race(r: &rusqlite::Row) -> rusqlite::Result<Race> {
    Ok(Race {
        box_: r.get(0)?,
        hash: r.get(1)?,
        name: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
        source: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
        category: r.get::<_, Option<String>>(4)?.unwrap_or_default(),
        size: r.get::<_, Option<f64>>(5)?.unwrap_or(0.0),
        added_at: r.get::<_, Option<i64>>(6)?.unwrap_or(0),
        completed_at: r.get(7)?,
        demand_at_add: r.get(8)?,
        announce_ts: r.get(9)?,
        registered_after: r.get(10)?,
        ratio_1h: r.get(11)?,
        ratio_3h: r.get(12)?,
        ratio_24h: r.get(13)?,
        ratio_7d: r.get(14)?,
        last_ratio: r.get(15)?,
        last_uploaded: r.get(16)?,
        removed_at: r.get(17)?,
        removed_rule: r.get(18)?,
    })
}

const RACE_COLS: &str = "box, hash, name, source, category, size, added_at, completed_at, demand_at_add, announce_ts, registered_after, ratio_1h, ratio_3h, ratio_24h, ratio_7d, last_ratio, last_uploaded, removed_at, removed_rule";

impl Store {
    pub fn open(path: &str) -> Result<Store> {
        let conn = Connection::open(path)?;
        conn.busy_timeout(std::time::Duration::from_secs(10))?;
        conn.execute_batch(SCHEMA)?;
        Ok(Store { conn: Mutex::new(conn) })
    }

    #[cfg(test)]
    pub fn open_memory() -> Result<Store> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA)?;
        Ok(Store { conn: Mutex::new(conn) })
    }

    fn c(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    // kv -------------------------------------------------------------------
    pub fn get(&self, k: &str) -> Option<String> {
        self.c().query_row("SELECT v FROM kv WHERE k = ?1", [k], |r| r.get(0)).optional().ok().flatten()
    }
    pub fn set(&self, k: &str, v: &str) {
        let _ = self.c().execute("INSERT INTO kv (k, v) VALUES (?1, ?2) ON CONFLICT(k) DO UPDATE SET v = excluded.v", params![k, v]);
    }
    pub fn del(&self, k: &str) {
        let _ = self.c().execute("DELETE FROM kv WHERE k = ?1", [k]);
    }

    // settings ---------------------------------------------------------------
    pub fn overrides(&self) -> Value {
        self.get("settings").and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Object(Default::default()))
    }
    pub fn save_overrides(&self, who: &str, old: &Value, new: &Value) {
        let c = self.c();
        let _ = c.execute(
            "INSERT INTO kv (k, v) VALUES ('settings', ?1) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
            [new.to_string()],
        );
        let _ = c.execute(
            "INSERT INTO audit (ts, who, old, new) VALUES (?1, ?2, ?3, ?4)",
            params![chrono::Utc::now().timestamp(), who, old.to_string(), new.to_string()],
        );
    }
    pub fn audit(&self, limit: i64) -> Vec<Value> {
        let c = self.c();
        let mut st = c.prepare("SELECT ts, who, old, new FROM audit ORDER BY id DESC LIMIT ?1").unwrap();
        st.query_map([limit], |r| {
            Ok(serde_json::json!({
                "ts": r.get::<_, i64>(0)?, "who": r.get::<_, String>(1)?,
                "old": serde_json::from_str::<Value>(&r.get::<_, String>(2)?).unwrap_or(Value::Null),
                "new": serde_json::from_str::<Value>(&r.get::<_, String>(3)?).unwrap_or(Value::Null),
            }))
        })
        .map(|it| it.filter_map(|x| x.ok()).collect())
        .unwrap_or_default()
    }

    // rollups ----------------------------------------------------------------
    #[allow(clippy::too_many_arguments)]
    pub fn rollup(&self, minute: i64, bx: &str, up: f64, dl: f64, n_dl: i64, n_seed: i64, n_total: i64,
                  disk_used: f64, disk_quota: f64, traffic_used: f64, traffic_quota: f64, uploaded: f64, leechers: i64) {
        let _ = self.c().execute(
            "INSERT INTO rollups VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)
             ON CONFLICT(minute, box) DO UPDATE SET up=excluded.up, dl=excluded.dl, n_dl=excluded.n_dl,
               n_seed=excluded.n_seed, n_total=excluded.n_total, disk_used=excluded.disk_used,
               disk_quota=excluded.disk_quota, traffic_used=excluded.traffic_used,
               traffic_quota=excluded.traffic_quota, uploaded=excluded.uploaded, leechers=excluded.leechers",
            params![minute, bx, up, dl, n_dl, n_seed, n_total, disk_used, disk_quota, traffic_used, traffic_quota, uploaded, leechers],
        );
    }
    pub fn rollups(&self, since_minute: i64, step: i64) -> Vec<Value> {
        let c = self.c();
        let mut st = c
            .prepare(
                "SELECT (minute / ?2) * ?2 AS m, box, AVG(up), AVG(dl), MAX(n_dl), MAX(n_seed), MAX(n_total),
                        MAX(disk_used), MAX(disk_quota), MAX(traffic_used), MAX(traffic_quota), MAX(uploaded), MAX(leechers)
                 FROM rollups WHERE minute >= ?1 GROUP BY m, box ORDER BY m",
            )
            .unwrap();
        st.query_map(params![since_minute, step.max(1)], |r| {
            Ok(serde_json::json!({
                "minute": r.get::<_, i64>(0)?, "box": r.get::<_, String>(1)?,
                "up": r.get::<_, Option<f64>>(2)?, "dl": r.get::<_, Option<f64>>(3)?,
                "n_dl": r.get::<_, Option<i64>>(4)?, "n_seed": r.get::<_, Option<i64>>(5)?,
                "n_total": r.get::<_, Option<i64>>(6)?, "disk_used": r.get::<_, Option<f64>>(7)?,
                "disk_quota": r.get::<_, Option<f64>>(8)?, "traffic_used": r.get::<_, Option<f64>>(9)?,
                "traffic_quota": r.get::<_, Option<f64>>(10)?, "uploaded": r.get::<_, Option<f64>>(11)?,
                "leechers": r.get::<_, Option<i64>>(12)?,
            }))
        })
        .map(|it| it.filter_map(|x| x.ok()).collect())
        .unwrap_or_default()
    }
    pub fn prune_rollups(&self, before_minute: i64) {
        let _ = self.c().execute("DELETE FROM rollups WHERE minute < ?1", [before_minute]);
    }

    // races ------------------------------------------------------------------
    pub fn upsert_race(&self, r: &Race) {
        let _ = self.c().execute(
            "INSERT INTO races (box, hash, name, source, category, size, added_at, completed_at, demand_at_add,
               announce_ts, registered_after, ratio_1h, ratio_3h, ratio_24h, ratio_7d, last_ratio, last_uploaded,
               removed_at, removed_rule)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19)
             ON CONFLICT(box, hash) DO UPDATE SET
               name=COALESCE(excluded.name, name), source=CASE WHEN source IS NULL OR source = '' THEN excluded.source ELSE source END,
               category=COALESCE(excluded.category, category), size=COALESCE(excluded.size, size),
               completed_at=COALESCE(completed_at, excluded.completed_at),
               demand_at_add=COALESCE(demand_at_add, excluded.demand_at_add),
               announce_ts=COALESCE(announce_ts, excluded.announce_ts),
               registered_after=COALESCE(registered_after, excluded.registered_after),
               ratio_1h=COALESCE(ratio_1h, excluded.ratio_1h), ratio_3h=COALESCE(ratio_3h, excluded.ratio_3h),
               ratio_24h=COALESCE(ratio_24h, excluded.ratio_24h), ratio_7d=COALESCE(ratio_7d, excluded.ratio_7d),
               last_ratio=COALESCE(excluded.last_ratio, last_ratio), last_uploaded=COALESCE(excluded.last_uploaded, last_uploaded),
               removed_at=COALESCE(removed_at, excluded.removed_at), removed_rule=COALESCE(removed_rule, excluded.removed_rule)",
            params![r.box_, r.hash, r.name, r.source, r.category, r.size, r.added_at, r.completed_at, r.demand_at_add,
                    r.announce_ts, r.registered_after, r.ratio_1h, r.ratio_3h, r.ratio_24h, r.ratio_7d, r.last_ratio,
                    r.last_uploaded, r.removed_at, r.removed_rule],
        );
    }
    pub fn race(&self, bx: &str, hash: &str) -> Option<Race> {
        let c = self.c();
        c.query_row(&format!("SELECT {RACE_COLS} FROM races WHERE box = ?1 AND hash = ?2"), params![bx, hash], row_race)
            .optional()
            .ok()
            .flatten()
    }
    pub fn races_since(&self, since: i64, limit: i64) -> Vec<Race> {
        let c = self.c();
        let mut st = c
            .prepare(&format!("SELECT {RACE_COLS} FROM races WHERE added_at >= ?1 ORDER BY added_at DESC LIMIT ?2"))
            .unwrap();
        st.query_map(params![since, limit], row_race).map(|it| it.filter_map(|x| x.ok()).collect()).unwrap_or_default()
    }
    pub fn mark_removed(&self, bx: &str, hash: &str, rule: &str) {
        let now = chrono::Utc::now().timestamp();
        let c = self.c();
        let _ = c.execute(
            "UPDATE races SET removed_at = COALESCE(removed_at, ?3), removed_rule = COALESCE(removed_rule, ?4) WHERE box = ?1 AND hash = ?2",
            params![bx, hash, now, rule],
        );
        let _ = c.execute(
            "INSERT INTO removed (box, hash, ts) VALUES (?1, ?2, ?3) ON CONFLICT(box, hash) DO UPDATE SET ts = excluded.ts",
            params![bx, hash, now],
        );
    }
    pub fn recently_removed(&self, hash: &str, since: i64) -> bool {
        self.c()
            .query_row("SELECT 1 FROM removed WHERE hash = ?1 AND ts >= ?2 LIMIT 1", params![hash, since], |_| Ok(()))
            .optional()
            .ok()
            .flatten()
            .is_some()
    }

    // decisions --------------------------------------------------------------
    pub fn decision(&self, seq: u64, ts: i64, actor: &str, trigger: &str, hash: Option<&str>, name: &str, chosen: &str, summary: &Value) {
        let _ = self.c().execute(
            "INSERT OR REPLACE INTO decisions VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![seq as i64, ts, actor, trigger, hash, name, chosen, summary.to_string()],
        );
    }
    pub fn decisions(&self, since_seq: i64, limit: i64) -> Vec<Value> {
        let c = self.c();
        let mut st = c
            .prepare(
                "SELECT d.seq, d.ts, d.actor, d.trigger, d.hash, d.name, d.chosen, d.summary
                 FROM decisions d WHERE d.seq > ?1 ORDER BY d.seq DESC LIMIT ?2",
            )
            .unwrap();
        let rows: Vec<(i64, i64, String, String, Option<String>, String, String, String)> = st
            .query_map(params![since_seq, limit], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?))
            })
            .map(|it| it.filter_map(|x| x.ok()).collect())
            .unwrap_or_default();
        drop(st);
        rows.into_iter()
            .map(|(seq, ts, actor, trigger, hash, name, chosen, summary)| {
                let outcomes: Vec<Value> = match &hash {
                    Some(h) => {
                        let mut st2 = c
                            .prepare("SELECT box, ratio_1h, ratio_3h, ratio_24h, ratio_7d, last_ratio, last_uploaded FROM races WHERE hash = ?1")
                            .unwrap();
                        st2.query_map([h], |r| {
                            Ok(serde_json::json!({"box": r.get::<_, String>(0)?, "ratio_1h": r.get::<_, Option<f64>>(1)?,
                                "ratio_3h": r.get::<_, Option<f64>>(2)?, "ratio_24h": r.get::<_, Option<f64>>(3)?,
                                "ratio_7d": r.get::<_, Option<f64>>(4)?, "last_ratio": r.get::<_, Option<f64>>(5)?,
                                "uploaded": r.get::<_, Option<f64>>(6)?}))
                        })
                        .map(|it| it.filter_map(|x| x.ok()).collect())
                        .unwrap_or_default()
                    }
                    None => vec![],
                };
                serde_json::json!({"seq": seq, "ts": ts, "actor": actor, "trigger": trigger, "hash": hash,
                    "name": name, "chosen": chosen,
                    "summary": serde_json::from_str::<Value>(&summary).unwrap_or(Value::Null), "outcomes": outcomes})
            })
            .collect()
    }
    pub fn prune_decisions(&self, before_ts: i64) {
        let _ = self.c().execute("DELETE FROM decisions WHERE ts < ?1", [before_ts]);
    }

    // releases ---------------------------------------------------------------
    pub fn release(&self, tl_id: &str) -> Option<Release> {
        self.c()
            .query_row(&format!("SELECT {RELEASE_COLS} FROM releases WHERE tl_id = ?1"), [tl_id], row_release)
            .optional()
            .ok()
            .flatten()
    }
    pub fn release_by_hash(&self, hash: &str) -> Option<Release> {
        self.c()
            .query_row(&format!("SELECT {RELEASE_COLS} FROM releases WHERE hash = ?1"), [hash], row_release)
            .optional()
            .ok()
            .flatten()
    }
    /// Insert a release first seen now, or fill in what an earlier source left
    /// out. Returns true when it was new.
    pub fn note_release(&self, tl_id: &str, name: &str, category: &str, size: f64, freeleech: Option<bool>,
                        announce_ts: Option<i64>, source: &str) -> bool {
        let now = chrono::Utc::now().timestamp();
        let c = self.c();
        let existed: bool = c
            .query_row("SELECT 1 FROM releases WHERE tl_id = ?1", [tl_id], |_| Ok(()))
            .optional()
            .ok()
            .flatten()
            .is_some();
        if existed {
            let _ = c.execute(
                "UPDATE releases SET
                   name = CASE WHEN name IS NULL OR name = '' THEN ?2 ELSE name END,
                   category = CASE WHEN category IS NULL OR category = '' THEN ?3 ELSE category END,
                   size = CASE WHEN size IS NULL OR size = 0 THEN ?4 ELSE size END,
                   freeleech = COALESCE(?5, freeleech),
                   announce_ts = COALESCE(announce_ts, ?6)
                 WHERE tl_id = ?1",
                params![tl_id, name, category, size, freeleech.map(|b| b as i64), announce_ts],
            );
        } else {
            let _ = c.execute(
                "INSERT INTO releases (tl_id, name, category, size, freeleech, first_seen, announce_ts, source)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
                params![tl_id, name, category, size, freeleech.unwrap_or(false) as i64, now, announce_ts, source],
            );
        }
        !existed
    }
    pub fn set_release_facts(&self, tl_id: &str, hash: &str, private: bool, size: f64) {
        let _ = self.c().execute(
            "UPDATE releases SET hash = ?2, private = ?3, size = ?4 WHERE tl_id = ?1",
            params![tl_id, hash, private as i64, size],
        );
    }
    pub fn release_fetch_failed(&self, tl_id: &str) {
        let _ = self.c().execute("UPDATE releases SET fetch_failed = fetch_failed + 1 WHERE tl_id = ?1", [tl_id]);
    }
    pub fn set_release_swarm(&self, tl_id: &str, seeders: i64, leechers: i64, completed: i64, ts: i64, sched_idx: Option<i64>) {
        let _ = self.c().execute(
            "UPDATE releases SET seeders = ?2, leechers = ?3, completed = ?4, last_scrape = ?5,
               sched_idx = COALESCE(?6, sched_idx) WHERE tl_id = ?1",
            params![tl_id, seeders, leechers, completed, ts, sched_idx],
        );
    }
    pub fn set_release_delta(&self, tl_id: &str, l: f64, s: f64, ts: i64) {
        let _ = self.c().execute(
            "UPDATE releases SET delta_l = ?2, delta_s = ?3, delta_ts = ?4 WHERE tl_id = ?1",
            params![tl_id, l, s, ts],
        );
    }
    pub fn set_release_freeleech(&self, tl_id: &str, fl: bool) {
        let _ = self.c().execute("UPDATE releases SET freeleech = ?2 WHERE tl_id = ?1", params![tl_id, fl as i64]);
    }
    pub fn releases_since(&self, since: i64) -> Vec<Release> {
        let c = self.c();
        let mut st = c
            .prepare(&format!("SELECT {RELEASE_COLS} FROM releases WHERE first_seen >= ?1 ORDER BY first_seen DESC"))
            .unwrap();
        st.query_map([since], row_release).map(|it| it.filter_map(|x| x.ok()).collect()).unwrap_or_default()
    }
    pub fn prune_releases(&self, before: i64) {
        let _ = self.c().execute("DELETE FROM releases WHERE first_seen < ?1", [before]);
    }

    // verified ---------------------------------------------------------------
    pub fn is_verified(&self, bx: &str, hash: &str) -> bool {
        self.c()
            .query_row("SELECT 1 FROM verified WHERE box = ?1 AND hash = ?2", params![bx, hash], |_| Ok(()))
            .optional()
            .ok()
            .flatten()
            .is_some()
    }
    pub fn set_verified(&self, bx: &str, hash: &str) {
        let _ = self.c().execute("INSERT OR IGNORE INTO verified VALUES (?1, ?2)", params![bx, hash]);
    }

    // event-log bookkeeping --------------------------------------------------
    pub fn count_day(&self, day: &str, kind: &str, n: i64, bytes: i64) {
        let _ = self.c().execute(
            "INSERT INTO day_counts VALUES (?1,?2,?3,?4) ON CONFLICT(day, kind) DO UPDATE SET n = n + excluded.n, bytes = bytes + excluded.bytes",
            params![day, kind, n, bytes],
        );
    }
    pub fn note_day(&self, day: &str, first: u64, last: u64) {
        let _ = self.c().execute(
            "INSERT INTO days VALUES (?1,?2,?3) ON CONFLICT(day) DO UPDATE SET first_seq = MIN(first_seq, excluded.first_seq), last_seq = MAX(last_seq, excluded.last_seq)",
            params![day, first as i64, last as i64],
        );
    }
    pub fn day_of_seq(&self, seq: u64) -> Option<String> {
        self.c()
            .query_row("SELECT day FROM days WHERE last_seq >= ?1 ORDER BY day LIMIT 1", [seq as i64], |r| r.get(0))
            .optional()
            .ok()
            .flatten()
    }
    pub fn days_from(&self, day: &str) -> Vec<String> {
        let c = self.c();
        let mut st = c.prepare("SELECT day FROM days WHERE day >= ?1 ORDER BY day").unwrap();
        st.query_map([day], |r| r.get(0)).map(|it| it.filter_map(|x| x.ok()).collect()).unwrap_or_default()
    }
    pub fn day_counts(&self, since_day: &str) -> Vec<Value> {
        let c = self.c();
        let mut st = c.prepare("SELECT day, kind, n, bytes FROM day_counts WHERE day >= ?1 ORDER BY day, kind").unwrap();
        st.query_map([since_day], |r| {
            Ok(serde_json::json!({"day": r.get::<_, String>(0)?, "kind": r.get::<_, String>(1)?,
                "n": r.get::<_, i64>(2)?, "bytes": r.get::<_, i64>(3)?}))
        })
        .map(|it| it.filter_map(|x| x.ok()).collect())
        .unwrap_or_default()
    }

    // traffic ----------------------------------------------------------------
    pub fn traffic_base(&self, bx: &str) -> Option<(String, f64)> {
        self.c()
            .query_row("SELECT period, base FROM traffic WHERE box = ?1", [bx], |r| Ok((r.get(0)?, r.get(1)?)))
            .optional()
            .ok()
            .flatten()
    }
    pub fn set_traffic_base(&self, bx: &str, period: &str, base: f64) {
        let _ = self.c().execute(
            "INSERT INTO traffic VALUES (?1,?2,?3) ON CONFLICT(box) DO UPDATE SET period = excluded.period, base = excluded.base",
            params![bx, period, base],
        );
    }

    /// Per-box, per-source ratio samples for the scoring tables: races at
    /// least `min_age` seconds old, with the best outcome known for them.
    pub fn ratio_samples(&self, since: i64, min_age: i64) -> Vec<(String, String, f64, Option<f64>, f64)> {
        let now = chrono::Utc::now().timestamp();
        let c = self.c();
        let mut st = c
            .prepare(
                "SELECT box, source, size, demand_at_add, COALESCE(ratio_24h, last_ratio)
                 FROM races WHERE added_at >= ?1 AND added_at <= ?2 AND COALESCE(ratio_24h, last_ratio) IS NOT NULL",
            )
            .unwrap();
        st.query_map(params![since, now - min_age], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                r.get::<_, Option<f64>>(2)?.unwrap_or(0.0), r.get::<_, Option<f64>>(3)?, r.get::<_, f64>(4)?))
        })
        .map(|it| it.filter_map(|x| x.ok()).collect())
        .unwrap_or_default()
    }
}
