//! Run policies against the simulator and report what each earned.
//!
//! ```text
//! rollout                               # synthetic releases, every reference policy
//! rollout --days 7 --episodes 20        # longer, more seeds
//! rollout --events /slow/ratio-manager/events/2026-11-03.jsonl.zst
//! rollout --params sim/params/2026-11-10.json
//! rollout --validate                    # just the gate, exit 1 if it fails
//! ```
//!
//! The report names its trace source, because a number from synthetic releases
//! and a number from replayed ones are not the same kind of evidence.

use anyhow::{Context, Result};
use ratio_sim::env::{Policy, Release, Sim};
use ratio_sim::params::Params;
use ratio_sim::policies::{Everywhere, Nothing, Oracle, TableBaseline};
use ratio_sim::{traces, validate};

struct Args {
    days: f64,
    episodes: u64,
    per_day: f64,
    freeleech_share: f64,
    events: Option<String>,
    params: Option<String>,
    validate_only: bool,
    seed: u64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            days: 3.0,
            episodes: 5,
            // TorrentLeech's RSS carries roughly 900 items a day.
            per_day: 900.0,
            // Of which the freeleech share is what a racer ever acts on.
            freeleech_share: 0.08,
            events: None,
            params: None,
            validate_only: false,
            seed: 1,
        }
    }
}

fn parse() -> Result<Args> {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let next = |i: usize| -> Result<String> {
            argv.get(i + 1).cloned().with_context(|| format!("{} needs a value", argv[i]))
        };
        match argv[i].as_str() {
            "--days" => {
                a.days = next(i)?.parse()?;
                i += 1;
            }
            "--episodes" => {
                a.episodes = next(i)?.parse()?;
                i += 1;
            }
            "--per-day" => {
                a.per_day = next(i)?.parse()?;
                i += 1;
            }
            "--freeleech-share" => {
                a.freeleech_share = next(i)?.parse()?;
                i += 1;
            }
            "--events" => {
                a.events = Some(next(i)?);
                i += 1;
            }
            "--params" => {
                a.params = Some(next(i)?);
                i += 1;
            }
            "--seed" => {
                a.seed = next(i)?.parse()?;
                i += 1;
            }
            "--validate" => a.validate_only = true,
            "-h" | "--help" => {
                println!("{}", include_str!("rollout-help.txt"));
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
        i += 1;
    }
    Ok(a)
}

/// Read the releases to replay. A `.zst` path is decompressed; the event log
/// writes `YYYY-MM-DD.jsonl.zst`.
fn load_events(path: &str) -> Result<Vec<Release>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    let text = if path.ends_with(".zst") {
        let out = zstd_decode(&bytes)?;
        String::from_utf8(out).context("event log is not UTF-8")?
    } else {
        String::from_utf8(bytes).context("event log is not UTF-8")?
    };
    traces::from_jsonl(&text, None)
}

fn zstd_decode(bytes: &[u8]) -> Result<Vec<u8>> {
    // The event log's frames carry no content size, so decode streaming.
    let mut out = Vec::new();
    let mut dec = zstd::stream::read::Decoder::new(std::io::Cursor::new(bytes))?;
    std::io::copy(&mut dec, &mut out).context("decompressing the event log")?;
    Ok(out)
}

fn policies() -> Vec<Box<dyn Policy>> {
    vec![
        Box::new(Nothing),
        Box::new(TableBaseline::new()),
        Box::new(Oracle::new()),
        Box::new(Everywhere),
    ]
}

fn main() -> Result<()> {
    let args = parse()?;
    let params = match &args.params {
        Some(p) => Params::load(p).with_context(|| format!("loading {p}"))?,
        None => Params::default(),
    };

    let (source, replay) = match &args.events {
        Some(p) => (format!("replay {p}"), Some(load_events(p)?)),
        None => ("SYNTHETIC releases — not evidence about the fleet".to_string(), None),
    };

    println!("calibration: {}", params.version);
    println!("trace source: {source}");
    println!("episodes: {}  days each: {}\n", args.episodes, args.days);

    if let Some(r) = &replay {
        println!("{} releases replayed, {} of them freeleech", r.len(), r.iter().filter(|x| x.freeleech).count());
        if r.is_empty() {
            anyhow::bail!("the event log held no rss_item or announce events — nothing to replay");
        }
    }

    let until = args.days * 86_400.0;
    let mut gate_failures = 0;

    for mut p in policies() {
        let mut totals = Vec::new();
        let mut last_report = None;
        for e in 0..args.episodes {
            let seed = args.seed.wrapping_mul(1_000_003).wrapping_add(e);
            let releases = match &replay {
                Some(r) => r.clone(),
                None => {
                    use rand::SeedableRng;
                    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0xa5a5);
                    traces::synthetic(args.days, args.per_day, args.freeleech_share, &mut rng)
                }
            };
            let mut sim = Sim::new(params.clone(), releases, seed);
            let r = sim.run(p.as_mut(), until);
            totals.push(r);
            last_report = Some(validate::check(&sim));
        }

        let n = totals.len() as f64;
        let up: f64 = totals.iter().map(|r| r.uploaded_gb).sum::<f64>() / n;
        let net: f64 = totals.iter().map(|r| r.total()).sum::<f64>() / n;
        let quota: f64 = totals.iter().map(|r| r.quota_cost).sum::<f64>() / n;
        let disk: f64 = totals.iter().map(|r| r.disk_cost).sum::<f64>() / n;
        println!(
            "{:<15} up {:>9.1} GB   net {:>9.1}   (quota {:>7.1}, disk {:>6.1})   {:>6.1} GB/day up",
            p.name(),
            up,
            net,
            quota,
            disk,
            up / args.days
        );
        if let Some(rep) = last_report {
            if p.name() == "table-baseline" {
                print!("\n  the gate, against the baseline's own races:\n");
                for line in rep.render().lines() {
                    println!("  {line}");
                }
                println!();
                if !rep.passed() {
                    gate_failures += 1;
                }
            }
        }
    }

    if args.validate_only && gate_failures > 0 {
        anyhow::bail!("the validation gate failed — this calibration is not fit to train against");
    }
    if replay.is_none() {
        println!("NOTE: synthetic releases. Re-run with --events before reading anything into these numbers.");
    }
    Ok(())
}
