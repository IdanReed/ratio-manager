# ratio-manager

TorrentLeech ratio building across every qBittorrent we control — a home box
behind a VPN and any number of rented seedboxes — with a dashboard, settings
edited in the UI, an append-only event log that doubles as a training
dataset, and an action API a learned policy can drive.

```
sensors (observe only) ──► event bus ──► baseline policy ──► /api/v1/actions ──► guard ──► drivers ──► boxes
```

| Part | What it does |
|---|---|
| **Sensors** | autobrr intake (a webhook carrying every IRC announce), the scout (listing scans, paced batched scrapes, the RSS feed), box drivers (state every 30 s), the clock. They emit *observations* (logged) and *deltas* (`announce`, `market`, `capacity`, `tick`, `setting`), and only deltas wake the policy |
| **Event bus** | Every event to the log, the rollups, SSE and the policy: a fast lane for announces, a coalescing normal lane for the rest |
| **Baseline policy** | One scoring rule for races, scout picks and evictions: a candidate's expected ratio over the horizon (per-box tables rebuilt nightly from race outcomes, seeded with priors) times the box's bias for its source, against what each holding uploaded in the last 30 minutes. Plus home's download focus and hand-offs outside VIP |
| **Guard** | The hard rules, on every action: freeleech and private only; our clients never trade (mutual bans and a peer check); DHT/PeX/LSD off; hit-and-run; disk; traffic allowance; politeness toward TorrentLeech; seedbox IPs registered with TL. A leak or self-trade halts everything until a human clears it |
| **Drivers** | qBittorrent WebAPI per box (adds, limits, bans, the parallel re-announcer, unregistered cleanup), plus host extras (listener repair behind a VPN, quota accounting, host pressure over SSH), and autobrr (the intake filter and its webhook) |
| **Event log** | Daily `.jsonl.zst` files (one zstd frame per flush), a nightly Parquet export, and a one-time backfill from VictoriaLogs |

The design, decisions and open questions live in the homelab's design notes;
the deployment (`stacks/ratio/`) lives in the BasicBastardSelfhosted repo.

## Run

```bash
ratio-manager serve           # RATIO_REGISTRY, RATIO_DATA (/data), RATIO_SLOW (/slow), RATIO_LISTEN (0.0.0.0:8080), RATIO_DRY_RUN
ratio-manager healthcheck     # the container healthcheck
ratio-manager export YYYY-MM-DD   # one day's events to Parquet now
ratio-manager check-settings  # print the default settings
```

The registry (`registry.yaml`) says which boxes exist, how to reach them and
which environment variables hold their logins; settings are edited in the UI
and stored as overrides of `src/defaults.yaml`. A settings change is accepted
only with forward auth's `X-Authentik-Username` (403 otherwise): a person
makes it in the UI, while a script or agent on the host proposes it. Against
the local rig, send the header yourself.

## Develop

```bash
nix develop
cargo test
dev/run.sh up     # three real qbittorrent-nox, plus fake TL, tracker, Prowlarr, autobrr and ntfy
```

`dev/fakes.py` documents the fakes. Images are built by `.github/workflows/image.yml`
on a version tag (`nix build .#image` builds the same one locally).
