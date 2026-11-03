//! Where the releases a rollout replays come from.
//!
//! Two sources, and the distinction decides what a result is worth:
//!
//!  - **Replay** (`from_jsonl`): the real RSS releases out of the event log's
//!    `rss_item` and `listing` records. This is the only source a reported
//!    number should rest on.
//!  - **Synthetic** (`synthetic`): drawn from a fitted distribution. Useful
//!    for filling in volume while the log is young, and for the generator
//!    bake-off, but a policy trained on these has to be re-tested against
//!    replayed releases before it means anything.
//!
//! A rollout records which source it used, because the two are not
//! interchangeable and a mixed result is the easiest way to fool yourself.

use crate::env::Release;
use anyhow::{Context, Result};
use rand::Rng;
use serde_json::Value;

/// TorrentLeech's categories, with roughly the share of releases each carries.
/// Only used by `synthetic`; replay takes categories from the log.
const CATEGORIES: &[(&str, f64)] = &[
    ("Movies :: HD", 0.22),
    ("Movies :: BlurayRip", 0.10),
    ("Movies :: 4K", 0.06),
    ("TV :: Episodes HD", 0.28),
    ("TV :: BoxSets", 0.05),
    ("Animation :: Anime", 0.08),
    ("Games :: PC", 0.06),
    ("Games :: Console", 0.04),
    ("Apps :: PC", 0.05),
    ("Music :: Audio", 0.06),
];

/// Read releases from the event log's JSONL, taking `rss_item` and `announce`
/// events. The event envelope is `{v, ts, seq, kind, box, hash, actor, data}`.
pub fn from_jsonl(text: &str, episode_start_ts: Option<i64>) -> Result<Vec<Release>> {
    // Collected with absolute timestamps first, then rebased on the earliest.
    // Anchoring on the first line read would be wrong: the log is ordered by
    // sequence, and a day's file can open with an event later than one that
    // follows it, which would put a release before the episode began.
    let mut out: Vec<Release> = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).context("event log line is not JSON")?;
        let kind = v.get("kind").and_then(|k| k.as_str()).unwrap_or("");
        if kind != "rss_item" && kind != "announce" {
            continue;
        }
        let d = v.get("data").unwrap_or(&Value::Null);
        let ts = v
            .get("ts")
            .and_then(|t| t.as_str())
            .and_then(|s| chrono_ts(s))
            .or_else(|| d.get("announce_ts").and_then(|t| t.as_i64()));
        let Some(ts) = ts else { continue };
        let size_bytes = d.get("size").and_then(|s| s.as_f64()).unwrap_or(0.0);
        out.push(Release {
            tl_id: d
                .get("tl_id")
                .map(|i| i.to_string().trim_matches('"').to_string())
                .unwrap_or_default(),
            name: d.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string(),
            category: d.get("category").and_then(|c| c.as_str()).unwrap_or("").to_string(),
            size: size_bytes / 1_073_741_824.0,
            freeleech: d.get("freeleech").and_then(|f| f.as_bool()).unwrap_or(false),
            // Absolute for now; rebased below.
            announce_at: ts as f64,
            listed_leechers: d.get("leechers").and_then(|l| l.as_f64()).unwrap_or(0.0),
            listed_seeders: d.get("seeders").and_then(|s| s.as_f64()).unwrap_or(0.0),
        });
    }
    let t0 = match episode_start_ts {
        Some(t) => t as f64,
        None => out.iter().map(|r| r.announce_at).fold(f64::INFINITY, f64::min),
    };
    if t0.is_finite() {
        for r in out.iter_mut() {
            r.announce_at -= t0;
        }
    }
    out.sort_by(|a, b| a.announce_at.partial_cmp(&b.announce_at).unwrap());
    Ok(out)
}

/// Parse the event envelope's RFC 3339 timestamp to unix seconds without
/// pulling in a date library: the format is fixed by `events.rs`.
fn chrono_ts(s: &str) -> Option<i64> {
    // "2026-11-01T23:06:39.150Z"
    let b = s.as_bytes();
    if b.len() < 20 || b[4] != b'-' || b[10] != b'T' {
        return None;
    }
    let num = |a: usize, z: usize| -> Option<i64> { s.get(a..z)?.parse().ok() };
    let (y, mo, d) = (num(0, 4)?, num(5, 7)?, num(8, 10)?);
    let (h, mi, sec) = (num(11, 13)?, num(14, 16)?, num(17, 19)?);
    // Days since the unix epoch, by the civil-from-days algorithm.
    let (y2, mo2) = if mo <= 2 { (y - 1, mo + 12) } else { (y, mo) };
    let era = if y2 >= 0 { y2 } else { y2 - 399 } / 400;
    let yoe = y2 - era * 400;
    let doy = (153 * (mo2 - 3) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + h * 3600 + mi * 60 + sec)
}

/// Draw a day of releases from a fitted distribution.
///
/// `per_day` is the arrival rate; TorrentLeech's own RSS runs about 900 items
/// a day across every category, of which the freeleech share is what a racer
/// ever sees.
pub fn synthetic<R: Rng + ?Sized>(
    days: f64,
    per_day: f64,
    freeleech_share: f64,
    rng: &mut R,
) -> Vec<Release> {
    let n = (days * per_day) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let cat = pick_category(rng);
        out.push(Release {
            tl_id: format!("syn-{i}"),
            name: format!("Synthetic.Release.{i}"),
            category: cat.to_string(),
            size: size_for(cat, rng),
            freeleech: rng.gen_bool(freeleech_share.clamp(0.0, 1.0)),
            announce_at: rng.gen_range(0.0..(days * 86_400.0)),
            // A fresh release's listing is seeder-poor and leecher-rich, and
            // lags besides.
            listed_leechers: rng.gen_range(0.0..60.0),
            listed_seeders: rng.gen_range(0.0..3.0),
        });
    }
    out.sort_by(|a, b| a.announce_at.partial_cmp(&b.announce_at).unwrap());
    out
}

fn pick_category<R: Rng + ?Sized>(rng: &mut R) -> &'static str {
    let total: f64 = CATEGORIES.iter().map(|(_, w)| w).sum();
    let mut x = rng.gen_range(0.0..total);
    for (c, w) in CATEGORIES {
        if x < *w {
            return c;
        }
        x -= w;
    }
    CATEGORIES[0].0
}

/// Sizes differ by category far more than by anything else: an episode is a
/// few GB, a box set or a console game is tens to hundreds.
fn size_for<R: Rng + ?Sized>(cat: &str, rng: &mut R) -> f64 {
    let (lo, hi) = match cat {
        c if c.contains("4K") => (40.0, 100.0),
        c if c.contains("BoxSets") => (30.0, 100.0),
        c if c.contains("Episodes") => (1.0, 8.0),
        c if c.contains("Console") || c.contains("Games") => (10.0, 100.0),
        c if c.contains("Audio") => (0.1, 2.0),
        c if c.contains("Apps") => (0.5, 20.0),
        c if c.contains("Anime") => (1.0, 30.0),
        _ => (4.0, 40.0),
    };
    rng.gen_range(lo..hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn the_timestamp_parser_agrees_with_a_known_instant() {
        // 2026-11-01T23:06:39Z. Checked against the civil-from-days algorithm.
        let got = chrono_ts("2026-11-01T23:06:39.150Z").unwrap();
        assert_eq!(got, 1793574399);
        assert_eq!(chrono_ts("1970-01-01T00:00:00.000Z").unwrap(), 0);
        assert_eq!(chrono_ts("2000-03-01T00:00:00.000Z").unwrap(), 951868800);
    }

    #[test]
    fn a_malformed_timestamp_is_rejected_rather_than_guessed() {
        assert!(chrono_ts("not a date").is_none());
        assert!(chrono_ts("2026/11/01 23:06").is_none());
    }

    #[test]
    fn jsonl_keeps_only_release_events_and_sorts_them() {
        let text = r#"
{"v":1,"ts":"2026-11-01T23:06:39.150Z","seq":2,"kind":"rss_item","data":{"tl_id":"2","name":"B","category":"TV :: Episodes HD","size":5368709120,"freeleech":true,"leechers":30,"seeders":1}}
{"v":1,"ts":"2026-11-01T23:00:00.000Z","seq":1,"kind":"rss_item","data":{"tl_id":"1","name":"A","category":"Movies :: HD","size":10737418240,"freeleech":false,"leechers":5,"seeders":9}}
{"v":1,"ts":"2026-11-01T23:10:00.000Z","seq":3,"kind":"box_state","data":{"free":1}}
"#;
        let rs = from_jsonl(text, None).unwrap();
        assert_eq!(rs.len(), 2, "box_state is not a release");
        // Sorted by announce time, and the first one anchors the episode.
        assert_eq!(rs[0].name, "A");
        assert_eq!(rs[0].announce_at, 0.0);
        assert_eq!(rs[1].name, "B");
        assert_eq!(rs[1].announce_at, 399.0);
        assert!((rs[1].size - 5.0).abs() < 1e-9, "size in GB: {}", rs[1].size);
        assert!(rs[1].freeleech && !rs[0].freeleech);
    }

    #[test]
    fn empty_input_is_not_an_error() {
        assert!(from_jsonl("", None).unwrap().is_empty());
        assert!(from_jsonl("\n\n", None).unwrap().is_empty());
    }

    #[test]
    fn synthetic_releases_land_in_the_window_and_are_sorted() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let rs = synthetic(2.0, 100.0, 0.3, &mut rng);
        assert_eq!(rs.len(), 200);
        for w in rs.windows(2) {
            assert!(w[0].announce_at <= w[1].announce_at);
        }
        for r in &rs {
            assert!((0.0..2.0 * 86_400.0).contains(&r.announce_at));
            assert!(r.size > 0.0);
        }
        let fl = rs.iter().filter(|r| r.freeleech).count() as f64 / rs.len() as f64;
        assert!((0.2..0.4).contains(&fl), "freeleech share {fl}");
    }

    #[test]
    fn episode_sizes_follow_the_category() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(10);
        for _ in 0..200 {
            assert!(size_for("TV :: Episodes HD", &mut rng) <= 8.0);
            assert!(size_for("Movies :: 4K", &mut rng) >= 40.0);
        }
    }
}
