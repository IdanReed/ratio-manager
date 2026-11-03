//! ratio-manager: TorrentLeech ratio building across home and rented
//! seedboxes. Sensors (autobrr intake, the scout, box drivers, the clock) put
//! events on a bus; only the baseline policy decides; every action passes the
//! guard; everything lands in an append-only event log that doubles as the
//! training dataset; `/api/v1` and the dashboard sit on top.

mod actions;
mod api;
mod autobrr;
mod bencode;
mod config;
mod drivers;
mod events;
mod export;
mod guard;
mod policy;
mod qbit;
mod scout;
mod state;
mod store;
mod tl;

use anyhow::{Context, Result};
use serde_json::json;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

fn env(k: &str, d: &str) -> String {
    std::env::var(k).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| d.to_string())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("serve");
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    match cmd {
        "healthcheck" => rt.block_on(healthcheck()),
        "export" => {
            let day = args.get(2).context("usage: ratio-manager export YYYY-MM-DD")?;
            let slow = PathBuf::from(env("RATIO_SLOW", "/slow"));
            let log = events::EventLog { dir: slow.join("events") };
            let n = export::export_day(&log, &slow.join("parquet"), day)?;
            println!("{day}: {n} events exported");
            Ok(())
        }
        "check-settings" => {
            let (s, _) = config::Settings::from_overrides(&json!({}))?;
            println!("{}", serde_json::to_string_pretty(&s)?);
            Ok(())
        }
        _ => rt.block_on(serve()),
    }
}

async fn healthcheck() -> Result<()> {
    let port = env("RATIO_LISTEN", "0.0.0.0:8080").rsplit(':').next().unwrap_or("8080").to_string();
    let r = reqwest::Client::new()
        .get(format!("http://127.0.0.1:{port}/healthz"))
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await?;
    if !r.status().is_success() {
        anyhow::bail!("healthz {}", r.status());
    }
    Ok(())
}

async fn serve() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with_target(false)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stdout()))
        .init();

    let registry = config::Registry::load(&env("RATIO_REGISTRY", "/etc/ratio-manager/registry.yaml"))?;
    let data = PathBuf::from(env("RATIO_DATA", "/data"));
    let slow = PathBuf::from(env("RATIO_SLOW", "/slow"));
    std::fs::create_dir_all(&data)?;
    std::fs::create_dir_all(slow.join("events"))?;
    let dry_run = matches!(env("RATIO_DRY_RUN", "false").to_lowercase().as_str(), "1" | "true" | "yes");

    let store = Arc::new(store::Store::open(data.join("manager.db").to_str().unwrap())?);
    let log = Arc::new(events::EventLog { dir: slow.join("events") });
    let start_seq = store.get("seq").and_then(|s| s.parse().ok()).unwrap_or(0u64).max(log.last_seq_on_disk());
    let (bus, rx) = events::Bus::new(start_seq);

    let overrides = store.overrides();
    let settings = match config::Settings::from_overrides(&overrides) {
        Ok((s, _)) => s,
        Err(e) => {
            tracing::error!("stored settings do not validate ({e:#}); running on the defaults until fixed in the UI");
            config::Settings::from_overrides(&json!({}))?.0
        }
    };
    let halt: Option<state::Halt> = store.get("halt").and_then(|h| serde_json::from_str::<serde_json::Value>(&h).ok()).map(|v| state::Halt {
        reason: v.get("reason").and_then(|r| r.as_str()).unwrap_or("").to_string(),
        at: v.get("at").and_then(|a| a.as_i64()).unwrap_or(0),
    });

    let tl = Arc::new(tl::Tl::new(config::env_secret(&registry.tl.rss_key_env), registry.tl.rss_url.clone(), registry.tl.base_url.clone()));
    let prowlarr = Arc::new(tl::Prowlarr::new(&registry.prowlarr.url, config::env_secret(&registry.prowlarr.api_key_env)));
    let app = Arc::new(state::App {
        registry: registry.clone(),
        settings: RwLock::new(Arc::new(settings)),
        boxes: RwLock::new(HashMap::new()),
        bus: bus.clone(),
        store: store.clone(),
        tl,
        prowlarr,
        halt: RwLock::new(halt.clone()),
        dry_run,
        announced: RwLock::new(HashMap::new()),
        candidates: RwLock::new(HashMap::new()),
        tables: RwLock::new(HashMap::new()),
        ntfy: registry.ntfy_url.clone(),
        http: reqwest::Client::builder().timeout(std::time::Duration::from_secs(10)).build()?,
        started: chrono::Utc::now().timestamp(),
    });

    tokio::spawn(events::EventLog { dir: slow.join("events") }.run(rx.log_rx, store.clone()));
    bus.emit("start", None, None, Some("manager"), json!({"version": env!("CARGO_PKG_VERSION"), "dry_run": dry_run,
        "boxes": registry.boxes.iter().map(|b| json!({"name": b.name, "enabled": b.enabled()})).collect::<Vec<_>>(),
        "halt": halt.as_ref().map(|h| h.reason.clone())}));
    tracing::info!("ratio-manager {} starting (dry_run={dry_run}, seq={start_seq}, boxes={:?})", env!("CARGO_PKG_VERSION"),
        registry.boxes.iter().map(|b| (b.name.clone(), b.enabled())).collect::<Vec<_>>());

    let mut qbits = HashMap::new();
    for b in registry.boxes.iter().filter(|b| b.enabled()) {
        qbits.insert(b.name.clone(), qbit::Qbit::new(b, b.timeout_s)?);
    }
    let exec = actions::Executor::new(app.clone(), qbits.clone());

    // Addresses and bans first: the self-trade rule needs them before any add.
    for b in registry.boxes.iter().filter(|b| b.enabled()) {
        let Ok(q) = qbit::Qbit::new(b, 60) else { continue };
        let mut warned = None;
        if let Err(e) = drivers::reconcile_once(&app, b, &q, &mut warned).await {
            tracing::warn!("[{}] first reconcile: {e}", b.name);
        }
    }
    let (nudges, _) = tokio::sync::broadcast::channel(256);
    for b in registry.boxes.iter().filter(|b| b.enabled()) {
        tokio::spawn(drivers::poller(app.clone(), b.clone(), exec.clone()));
        tokio::spawn(drivers::reannouncer(app.clone(), b.clone(), exec.clone(), nudges.clone()));
        tokio::spawn(drivers::reconciler(app.clone(), b.clone()));
        tokio::spawn(drivers::peer_snapshots(app.clone(), b.clone()));
        tokio::spawn(drivers::host_pressure(app.clone(), b.clone()));
    }
    {
        let (app2, qb) = (app.clone(), qbits.clone());
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(120)).await;
                guard::self_trade_check(&app2, &qb).await;
            }
        });
    }
    let scout = scout::Scout::start(app.clone());
    tokio::spawn(autobrr::reconciler(app.clone()));
    tokio::spawn(policy::clock(app.clone()));
    tokio::spawn(drivers::rollups(app.clone()));
    tokio::spawn(export::nightly(app.clone(), log.clone(), slow.join("parquet")));
    {
        let (app2, dir) = (app.clone(), slow.join("backfill"));
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(300)).await;
            if let Err(e) = export::backfill(app2, dir).await {
                tracing::warn!("backfill: {e:#}");
            }
        });
    }
    tokio::spawn(policy::Baseline::new(app.clone(), exec.clone(), scout.clone()).run(rx.fast_rx));

    let api = api::Api { app: app.clone(), exec, log, scout };
    let addr: SocketAddr = env("RATIO_LISTEN", "0.0.0.0:8080").parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("listening on {addr}");
    axum::serve(listener, api::router(api).into_make_service_with_connect_info::<SocketAddr>()).await?;
    Ok(())
}
