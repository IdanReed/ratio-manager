//! The hard rules. Every action passes `check_add` / `check_remove`, whatever
//! proposed it; the periodic checks (leaks, self-trade, addresses) can halt.

use crate::bencode::TorrentFacts;
use crate::config::{Settings, GB};
use crate::qbit::{self, f, s, Qbit};
use crate::state::{room, App, BoxView};
use crate::store::Release;
use serde_json::json;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

static UNREGISTERED: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"(?i)unregistered|not registered").unwrap());

pub fn unregistered_msg(m: &str) -> bool {
    UNREGISTERED.is_match(m)
}

/// Rules 1, 2, 3, 5, 6, 8 for an add. Err(reasons) refuses it.
pub fn check_add(app: &App, settings: &Settings, bx: &str, facts: &TorrentFacts, release: Option<&Release>, source: &str) -> Result<(), Vec<String>> {
    let mut why = Vec::new();
    if let Some(h) = app.halted() {
        why.push(format!("halted: {}", h.reason));
    }
    let Some(cfg) = app.box_cfg(bx) else {
        return Err(vec![format!("no box {bx} in the registry")]);
    };
    let Some(view) = app.view(bx) else {
        return Err(vec![format!("{bx} has not been seen yet")]);
    };
    if !view.reachable {
        why.push(format!("{bx} is unreachable"));
    }
    // Rule 1: freeleech, and the private flag.
    match release {
        Some(r) if r.freeleech => {}
        Some(_) => why.push("not freeleech".into()),
        None => why.push("no release record: freeleech unconfirmed".into()),
    }
    if !facts.private {
        why.push("the .torrent has no private flag".into());
    }
    // Rule 2: not twice on one box (several boxes may hold it).
    if view.torrents.contains_key(&facts.hash) {
        why.push(format!("{bx} already holds it"));
    }
    // Rule 3 is verified per torrent after the add, and client-wide by the
    // reconciler on a private-only box.
    // Rule 5: disk.
    let r = room(&view, settings);
    if facts.size > r.placeable {
        why.push(format!("disk: needs {:.1} GB, {:.1} GB placeable", facts.size / GB, r.placeable.max(0.0) / GB));
    }
    // Rule 6: allowance.
    if view.allowance_spent && source != "manual" {
        why.push(format!("{bx}'s traffic allowance is spent"));
    }
    // Rule 8: a seedbox's address must be the one registered with TL.
    if let Some(reg) = &cfg.tl_registered_ip {
        if !view.ips.is_empty() && !view.ips.iter().any(|ip| ip == reg) {
            why.push(format!("{bx}'s address {:?} is not the TL-registered {reg}", view.ips));
        }
        if view.ips.is_empty() {
            why.push(format!("{bx}'s address is not resolved yet"));
        }
    }
    if why.is_empty() {
        Ok(())
    } else {
        Err(why)
    }
}

/// Rule 4: which way a torrent may go, or None. `reason` is what the proposer
/// claims; the guard checks it itself.
pub fn may_remove(app: &App, settings: &Settings, view: &BoxView, hash: &str, trackers_unregistered: bool) -> Option<String> {
    let t = view.torrents.get(hash)?;
    let now = chrono::Utc::now().timestamp();
    let downloaded = f(t, "downloaded");
    let done = qbit::finished(t);
    if downloaded <= 0.0 && !done {
        return Some("downloaded nothing".into());
    }
    if trackers_unregistered {
        return Some("unregistered (deleted by TL)".into());
    }
    let completion = qbit::i(t, "completion_on");
    if let Some(vip_end) = settings.vip_until_ts() {
        // Under VIP, removals ignore hit-and-run; after it, a torrent
        // completed during VIP stays exempt.
        if now < vip_end {
            return Some("VIP".into());
        }
        if done && completion > 0 && completion < vip_end {
            return Some("completed during VIP".into());
        }
    }
    if f(t, "ratio") >= 1.0 {
        return Some("past 1:1".into());
    }
    if !done {
        // Partial: no seed time accrues; only 1:1 clears it.
        return None;
    }
    if f(t, "seeding_time") >= settings.account.hnr_min_seed_days * 86400.0 {
        return Some(format!("seeded {:.0} days", f(t, "seeding_time") / 86400.0));
    }
    // Another of our boxes seeds a complete, verified copy on.
    if let Ok(boxes) = app.boxes.read() {
        for other in boxes.values() {
            if other.name == view.name {
                continue;
            }
            if let Some(o) = other.torrents.get(hash) {
                if qbit::finished(o) && app.store.is_verified(&other.name, hash) {
                    return Some(format!("{} seeds it on", other.name));
                }
            }
        }
    }
    None
}

/// Rule 3, per torrent: private, with DHT, PeX and LSD disabled (status 0).
/// Some(true) verified, None not listable yet (seconds old), Some(false) a
/// leak: the torrent is stopped and everything halts.
pub async fn leak_check(app: &Arc<App>, q: &Qbit, bx: &str, t: &serde_json::Value) -> Option<bool> {
    let hash = s(t, "hash").to_string();
    if app.store.is_verified(bx, &hash) {
        return Some(true);
    }
    let props = q.properties(&hash).await.ok()?;
    let rows = q.trackers(&hash).await.ok()?;
    let peers: HashMap<String, i64> = rows
        .iter()
        .filter(|r| s(r, "url").starts_with("** ["))
        .map(|r| (s(r, "url").to_string(), qbit::i(r, "status")))
        .collect();
    let leaks: Vec<&String> = peers.iter().filter(|(_, st)| **st != 0).map(|(u, _)| u).collect();
    let private = if let Some(p) = props.get("is_private").and_then(|v| v.as_bool()) {
        p
    } else if let Some(p) = t.get("private").and_then(|v| v.as_bool()) {
        p
    } else {
        peers.len() == 3 && leaks.is_empty()
    };
    if private && peers.len() == 3 && leaks.is_empty() {
        app.store.set_verified(bx, &hash);
        return Some(true);
    }
    if private && leaks.is_empty() {
        return None;
    }
    if !app.dry_run {
        let _ = q.stop(std::slice::from_ref(&hash)).await;
    }
    let reason = format!(
        "[{bx}] {} is_private={private} peer sources={:?} — stopped; every policy halted until cleared",
        s(t, "name").chars().take(70).collect::<String>(),
        peers
    );
    app.set_halt(reason).await;
    Some(false)
}

/// Rule 2's backstop: a torrent two of our clients hold must never list one
/// of our addresses among its peers. A hit stops it on all but one client and
/// halts.
pub async fn self_trade_check(app: &Arc<App>, qbits: &HashMap<String, Qbit>) {
    if app.halted().is_some() {
        return;
    }
    let held = app.held_anywhere();
    let now = chrono::Utc::now().timestamp();
    for (hash, holders) in held.into_iter().filter(|(_, v)| v.len() >= 2) {
        for bx in &holders {
            let Some(view) = app.view(bx) else { continue };
            let Some(t) = view.torrents.get(&hash) else { continue };
            // Only a live swarm can trade: downloading, uploading, or young.
            if !(qbit::downloading(t) || f(t, "upspeed") > 0.0 || now - qbit::i(t, "added_on") < 86400) {
                continue;
            }
            let Some(q) = qbits.get(bx) else { continue };
            let Ok(p) = q.peers(&hash).await else { continue };
            let ours = app.our_ips(Some(bx));
            let peers = p.get("peers").and_then(|v| v.as_object()).cloned().unwrap_or_default();
            let hits: Vec<String> = peers
                .values()
                .filter_map(|pp| pp.get("ip").and_then(|v| v.as_str()).map(|s| s.to_string()))
                .filter(|ip| ours.contains_key(ip))
                .collect();
            if hits.is_empty() {
                continue;
            }
            let keep = holders[0].clone();
            for other in holders.iter().filter(|o| **o != keep) {
                if let Some(oq) = qbits.get(other) {
                    if !app.dry_run {
                        let _ = oq.stop(std::slice::from_ref(&hash)).await;
                    }
                }
            }
            let names: Vec<String> = hits.iter().map(|ip| format!("{} ({ip})", ours[ip])).collect();
            app.bus.emit("alert", Some(bx), Some(&hash), Some("guard"), json!({"rule": 2, "peers": names}));
            app.set_halt(format!(
                "[{bx}] {} peers with our own {} — stopped on all but {keep}; every policy halted until cleared",
                s(t, "name").chars().take(60).collect::<String>(),
                names.join(", ")
            ))
            .await;
            return;
        }
    }
}

/// The addresses a box's peers see: its URL's host, resolved, plus any
/// listed in the registry.
pub async fn resolve_ips(host: Option<String>, extra: &[String]) -> Vec<String> {
    let mut out: Vec<String> = extra.to_vec();
    if let Some(h) = host {
        if let Ok(addrs) = tokio::net::lookup_host((h.as_str(), 443)).await {
            for a in addrs {
                let ip = a.ip().to_string();
                if !out.contains(&ip) {
                    out.push(ip);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn unregistered_messages() {
        assert!(super::unregistered_msg("Unregistered torrent"));
        assert!(super::unregistered_msg("torrent not registered with this tracker"));
        assert!(!super::unregistered_msg("Working"));
    }
}
