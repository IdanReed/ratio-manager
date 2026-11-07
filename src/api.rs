//! `/api/v1`: the one surface the UI, the baseline's peers (scripts, a model)
//! and autobrr's webhook use. Behind Caddy forward auth (Admin); the intake
//! is reached on the compose network and accepts only autobrr's address.

use crate::actions::{is_manual, Action, Executor};
use crate::config::{defaults_value, diff, Settings};
use crate::events::{Event, EventLog};
use crate::state::App;
use axum::extract::{ConnectInfo, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use tokio_stream::wrappers::BroadcastStream;

#[derive(Clone)]
pub struct Api {
    pub app: Arc<App>,
    pub exec: Executor,
    pub log: Arc<EventLog>,
    pub scout: crate::scout::Scout,
}

pub fn router(api: Api) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/favicon.ico", get(|| async { StatusCode::NO_CONTENT }))
        .route("/", get(|| async { asset("index.html") }))
        .route("/app.js", get(|| async { asset("app.js") }))
        .route("/app.css", get(|| async { asset("app.css") }))
        .route("/uPlot.iife.min.js", get(|| async { asset("uPlot.iife.min.js") }))
        .route("/uPlot.min.css", get(|| async { asset("uPlot.min.css") }))
        .route("/metrics", get(prometheus))
        .route("/api/v1/state", get(state))
        .route("/api/v1/candidates", get(candidates))
        .route("/api/v1/events", get(events))
        .route("/api/v1/events/stream", get(stream))
        .route("/api/v1/actions", post(actions))
        .route("/api/v1/settings", get(get_settings).put(put_settings))
        .route("/api/v1/settings/audit", get(audit))
        .route("/api/v1/metrics", get(metrics))
        .route("/api/v1/decisions", get(decisions))
        .route("/api/v1/races", get(races))
        .route("/api/v1/data", get(data))
        .route("/api/v1/account", get(account))
        .route("/api/v1/halt/clear", post(clear_halt))
        .route("/api/v1/intake/autobrr", post(intake))
        .with_state(api)
}

fn asset(name: &str) -> Response {
    let (body, ctype): (&'static [u8], &str) = match name {
        "index.html" => (include_bytes!("../ui/index.html"), "text/html; charset=utf-8"),
        "app.js" => (include_bytes!("../ui/app.js"), "text/javascript; charset=utf-8"),
        "app.css" => (include_bytes!("../ui/app.css"), "text/css; charset=utf-8"),
        "uPlot.iife.min.js" => (include_bytes!("../ui/uPlot.iife.min.js"), "text/javascript; charset=utf-8"),
        "uPlot.min.css" => (include_bytes!("../ui/uPlot.min.css"), "text/css; charset=utf-8"),
        _ => return StatusCode::NOT_FOUND.into_response(),
    };
    ([(header::CONTENT_TYPE, ctype), (header::CACHE_CONTROL, "no-cache")], body).into_response()
}

async fn healthz(State(api): State<Api>) -> Json<Value> {
    Json(json!({"ok": true, "seq": api.app.bus.last_seq(), "halt": api.app.halted().is_some()}))
}

/// The user forward auth signed in. Caddy sets X-Authentik-Username from the
/// outpost and strips any a client sent, so a request without it reached the
/// loopback port directly.
fn forward_auth_user(headers: &HeaderMap) -> Option<&str> {
    headers.get("x-authentik-username").and_then(|v| v.to_str().ok()).filter(|s| !s.is_empty())
}

fn who(headers: &HeaderMap) -> String {
    format!("manual:{}", forward_auth_user(headers).unwrap_or("operator"))
}

async fn state(State(api): State<Api>) -> Json<Value> {
    Json(api.app.snapshot())
}

async fn candidates(State(api): State<Api>) -> Json<Value> {
    Json(json!(api.app.candidates.read().map(|c| c.clone()).unwrap_or_default()))
}

#[derive(Deserialize)]
struct EventsQ {
    since: Option<u64>,
    kind: Option<String>,
    limit: Option<usize>,
}

fn kinds(k: &Option<String>) -> Option<Vec<String>> {
    k.as_ref().map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect())
}

async fn events(State(api): State<Api>, Query(q): Query<EventsQ>) -> Json<Value> {
    let since = q.since.unwrap_or(0);
    let limit = q.limit.unwrap_or(1000).min(10000);
    let ks = kinds(&q.kind);
    if let Some(evs) = api.app.bus.recent_since(since, &ks, limit) {
        return Json(json!({"events": evs, "source": "memory"}));
    }
    let app = api.app.clone();
    let log = api.log.clone();
    let evs = tokio::task::spawn_blocking(move || {
        let mut out: Vec<Event> = Vec::new();
        let Some(day) = app.store.day_of_seq(since + 1) else { return out };
        for d in app.store.days_from(&day) {
            for e in log.read_day(&d) {
                if e.seq > since && ks.as_ref().map(|k| k.contains(&e.kind)).unwrap_or(true) {
                    out.push(e);
                    if out.len() >= limit {
                        return out;
                    }
                }
            }
        }
        out
    })
    .await
    .unwrap_or_default();
    Json(json!({"events": evs, "source": "files"}))
}

async fn stream(State(api): State<Api>, Query(q): Query<EventsQ>) -> impl IntoResponse {
    let ks = kinds(&q.kind);
    let rx = api.app.bus.sse.subscribe();
    let s = BroadcastStream::new(rx).filter_map(move |e| {
        let ks = ks.clone();
        async move {
            let e = e.ok()?;
            if let Some(k) = &ks {
                if !k.contains(&e.kind) {
                    return None;
                }
            }
            Some(Ok::<_, std::convert::Infallible>(
                axum::response::sse::Event::default().id(e.seq.to_string()).event(e.kind.clone()).data(serde_json::to_string(&e).unwrap_or_default()),
            ))
        }
    });
    Sse::new(s).keep_alive(KeepAlive::default())
}

async fn actions(State(api): State<Api>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    let actor = headers
        .get("x-actor")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .filter(|a| !is_manual(a) && a != "baseline" && a != "driver" && a != "guard")
        .unwrap_or_else(|| who(&headers));
    let list: Vec<Value> = match body {
        Value::Array(a) => a,
        other => vec![other],
    };
    let mut out = Vec::new();
    for v in list {
        match serde_json::from_value::<Action>(v.clone()) {
            Ok(a) => {
                let r = api.exec.run(&actor, a, json!({"via": "api"})).await;
                out.push(json!({"action": v, "outcome": r}));
            }
            Err(e) => out.push(json!({"action": v, "outcome": {"status": "rejected", "reasons": [format!("not an action: {e}")]}})),
        }
    }
    Json(json!({"actor": actor, "results": out})).into_response()
}

async fn get_settings(State(api): State<Api>) -> Json<Value> {
    let overrides = api.app.store.overrides();
    let (_, full) = Settings::from_overrides(&overrides).unwrap_or_else(|_| Settings::from_overrides(&json!({})).unwrap());
    Json(json!({"settings": full, "overrides": overrides, "defaults": defaults_value()}))
}

/// Body: the full settings document (as the UI edits it). Stored as the
/// overrides that differ from the defaults; validated first.
///
/// Only through forward auth: settings are a person's decision, made in the
/// UI. A script or an agent on the host proposes a change; it does not make
/// one.
async fn put_settings(State(api): State<Api>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    if forward_auth_user(&headers).is_none() {
        let why = "settings change only through the UI (forward auth); propose the change instead";
        return (StatusCode::FORBIDDEN, Json(json!({"ok": false, "error": why}))).into_response();
    }
    let doc = body.get("settings").cloned().unwrap_or(body);
    let overrides = diff(&defaults_value(), &doc).unwrap_or(json!({}));
    match Settings::from_overrides(&overrides) {
        Ok((s, _)) => {
            let old = api.app.store.overrides();
            let who = who(&headers);
            api.app.store.save_overrides(&who, &old, &overrides);
            if let Ok(mut g) = api.app.settings.write() {
                *g = Arc::new(s);
            }
            api.app.bus.emit("setting", None, None, Some(&who), json!({"old": old, "new": overrides}));
            Json(json!({"ok": true, "overrides": overrides})).into_response()
        }
        Err(e) => (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({"ok": false, "error": format!("{e:#}")}))).into_response(),
    }
}

async fn audit(State(api): State<Api>) -> Json<Value> {
    Json(json!(api.app.store.audit(100)))
}

#[derive(Deserialize)]
struct RangeQ {
    range: Option<String>,
}

async fn metrics(State(api): State<Api>, Query(q): Query<RangeQ>) -> Json<Value> {
    let (mins, step) = match q.range.as_deref() {
        Some("7d") => (7 * 24 * 60, 15),
        Some("30d") => (30 * 24 * 60, 60),
        _ => (6 * 60, 1),
    };
    let since = chrono::Utc::now().timestamp() / 60 - mins;
    Json(json!({"step_min": step, "rows": api.app.store.rollups(since, step)}))
}

async fn prometheus(State(api): State<Api>) -> Response {
    let mut out = String::new();
    let settings = api.app.settings();
    for v in api.app.views() {
        let r = crate::state::room(&v, &settings);
        let up: f64 = v.torrents.values().map(|t| crate::qbit::f(t, "upspeed")).sum();
        out.push_str(&format!("ratio_box_reachable{{box=\"{}\"}} {}\n", v.name, v.reachable as i32));
        out.push_str(&format!("ratio_box_upload_bytes_per_second{{box=\"{}\"}} {up}\n", v.name));
        out.push_str(&format!("ratio_box_torrents{{box=\"{}\"}} {}\n", v.name, v.torrents.len()));
        out.push_str(&format!("ratio_box_placeable_bytes{{box=\"{}\"}} {}\n", v.name, r.placeable));
    }
    out.push_str(&format!("ratio_halted {}\n", api.app.halted().is_some() as i32));
    out.push_str(&format!("ratio_event_seq {}\n", api.app.bus.last_seq()));
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], out).into_response()
}

#[derive(Deserialize)]
struct SinceQ {
    since: Option<i64>,
    limit: Option<i64>,
    days: Option<i64>,
}

async fn decisions(State(api): State<Api>, Query(q): Query<SinceQ>) -> Json<Value> {
    Json(json!(api.app.store.decisions(q.since.unwrap_or(0), q.limit.unwrap_or(200).min(2000))))
}

async fn races(State(api): State<Api>, Query(q): Query<SinceQ>) -> Json<Value> {
    let since = chrono::Utc::now().timestamp() - q.days.unwrap_or(3) * 86400;
    Json(json!(api.app.store.races_since(since, q.limit.unwrap_or(500).min(5000))))
}

async fn data(State(api): State<Api>, Query(q): Query<SinceQ>) -> Json<Value> {
    let day = (chrono::Utc::now() - chrono::Duration::days(q.days.unwrap_or(14))).format("%Y-%m-%d").to_string();
    Json(json!(api.app.store.day_counts(&day)))
}

async fn account(State(api): State<Api>) -> Json<Value> {
    let settings = api.app.settings();
    let races = api.app.store.races_since(chrono::Utc::now().timestamp() - 30 * 86400, 100000);
    let vip_only: Vec<_> = races
        .iter()
        .filter(|r| matches!(r.removed_rule.as_deref(), Some("VIP") | Some("completed during VIP")))
        .filter(|r| r.last_ratio.unwrap_or(0.0) < 1.0)
        .collect();
    let freed: f64 = vip_only.iter().map(|r| r.size).sum();
    let total_up: f64 = races.iter().map(|r| r.last_uploaded.unwrap_or(0.0)).sum();
    let by_source: HashMap<String, (usize, f64)> = races.iter().fold(HashMap::new(), |mut m, r| {
        let e = m.entry(r.source.clone()).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += r.last_uploaded.unwrap_or(0.0);
        m
    });
    Json(json!({
        "vip": settings.vip_now(), "vip_until": settings.account.vip_until, "vip_cost_eur_month": settings.account.vip_cost_eur_month,
        "hnr_min_seed_days": settings.account.hnr_min_seed_days, "min_overall_ratio": settings.account.min_overall_ratio,
        "vip_only_removals_30d": vip_only.len(), "vip_only_freed_bytes_30d": freed,
        "uploaded_30d_by_races": total_up, "by_source_30d": by_source,
        "halt": api.app.halted(),
    }))
}

async fn clear_halt(State(api): State<Api>, headers: HeaderMap) -> Json<Value> {
    let old = api.app.clear_halt(&who(&headers));
    Json(json!({"cleared": old}))
}

/// autobrr's webhook. Only autobrr's address may post here (it carries no
/// secret: autobrr's webhook action sends no custom headers).
async fn intake(State(api): State<Api>, ConnectInfo(peer): ConnectInfo<SocketAddr>, Json(body): Json<Value>) -> Response {
    let host = api.app.registry.autobrr.intake_from_host.clone();
    let allowed: Vec<IpAddr> = tokio::net::lookup_host((host.as_str(), 0)).await.map(|it| it.map(|a| a.ip()).collect()).unwrap_or_default();
    let peer_ip = match peer.ip() {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().map(IpAddr::V4).unwrap_or(IpAddr::V6(v6)),
        v4 => v4,
    };
    if !allowed.contains(&peer_ip) && !peer_ip.is_loopback() {
        tracing::warn!("intake: refused a post from {peer_ip} (only {host} {:?})", allowed);
        return StatusCode::FORBIDDEN.into_response();
    }
    let tl_id = body.get("tl_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let name = body.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let category = body.get("category").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let freeleech = body.get("freeleech").and_then(|v| v.as_bool()).unwrap_or(false);
    if tl_id.is_empty() || name.is_empty() {
        return (StatusCode::BAD_REQUEST, "tl_id and name required").into_response();
    }
    let ts = chrono::Utc::now().timestamp();
    let size = body.get("size").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let new = api.app.store.note_release(&tl_id, &name, &category, size, Some(freeleech), Some(ts), "announce");
    let mut data = body.clone();
    data["new"] = json!(new);
    api.app.bus.emit("announce", None, None, Some("autobrr"), data);
    // Wave 2: every release's facts, for the scout's schedule (the policy
    // fetches a freeleech one itself, at once).
    if !freeleech && api.app.settings().scout.every_release {
        api.scout.want_facts(&tl_id);
    }
    (StatusCode::OK, "ok").into_response()
}
