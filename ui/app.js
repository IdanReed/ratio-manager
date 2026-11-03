"use strict";
const $ = (s) => document.querySelector(s);
// Every dynamic value in the templates below passes through esc(); render() is
// the one place markup is parsed.
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function render(el, html) { el.replaceChildren(document.createRange().createContextualFragment(html)); }
const GB = 1e9, MIB = 1048576;
const gb = (b) => b == null ? "–" : (b / GB).toFixed(1) + " GB";
const mibs = (b) => b == null ? "–" : (b / MIB).toFixed(1) + " MiB/s";
const r2 = (x) => x == null ? "–" : Number(x).toFixed(2);
const ago = (ts) => { if (!ts) return "–"; const s = Date.now() / 1000 - ts; return s < 90 ? `${s | 0}s ago` : s < 5400 ? `${(s / 60) | 0}m ago` : s < 172800 ? `${(s / 3600).toFixed(1)}h ago` : `${(s / 86400).toFixed(1)}d ago`; };
const when = (ts) => ts ? new Date(ts * 1000).toLocaleString() : "–";
async function get(p) { const r = await fetch(p); if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.json(); }
const COLORS = ["#2f6fdf", "#d9822b", "#2e8b57", "#9b59b6", "#c0392b", "#16a085"];

function lsGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function lsSet(k, v) { try { localStorage.setItem(k, v); } catch { /* private window */ } }
let tab = lsGet("tab") || "overview";

function show(t) {
  tab = t; lsSet("tab", t);
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("on", b.dataset.tab === t));
  document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("on", s.id === t));
  refresh();
}
document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));

function table(cols, rows) {
  const head = cols.map((c) => `<th>${esc(c[0])}</th>`).join("");
  const body = rows.map((r) => "<tr>" + cols.map((c) => `<td class="${c[2] || ""}">${c[1](r)}</td>`).join("") + "</tr>").join("");
  return `<div class="wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

let state = null;
async function loadState() {
  state = await get("api/v1/state");
  const h = state.halt;
  $("#halt").hidden = !h;
  if (h) $("#halt-reason").textContent = `${h.reason} (${ago(h.at)})`;
  const up = state.boxes.reduce((a, b) => a + (b.up_bps || 0), 0);
  $("#status").textContent = `${state.dry_run ? "DRY RUN · " : ""}${state.vip ? "VIP · " : ""}up ${mibs(up)} · seq ${state.seq}`;
}
$("#halt-clear").addEventListener("click", async () => {
  if (!confirm("Clear the halt? Every policy resumes.")) return;
  await fetch("api/v1/halt/clear", { method: "POST" });
  refresh();
});

// Overview ------------------------------------------------------------------
function chart(id, series, data, fmt) {
  const el = $(id);
  el.replaceChildren();
  const width = Math.max(300, el.clientWidth - 12);
  const opts = {
    width, height: 220,
    scales: { x: { time: true } },
    axes: [{}, { values: (u, v) => v.map(fmt), size: 70 }],
    series: [{}, ...series.map((s, i) => ({ label: s, stroke: COLORS[i % COLORS.length], width: 1.5, value: (u, v) => v == null ? "–" : fmt(v) }))],
  };
  new uPlot(opts, data, el);
}
function series(rows, boxes, field) {
  const ts = [...new Set(rows.map((r) => r.minute))].sort((a, b) => a - b);
  const idx = new Map(ts.map((t, i) => [t, i]));
  const out = [ts.map((t) => t * 60)];
  for (const b of boxes) {
    const col = new Array(ts.length).fill(null);
    rows.filter((r) => r.box === b).forEach((r) => { col[idx.get(r.minute)] = r[field]; });
    out.push(col);
  }
  return out;
}
async function overview() {
  const m = await get(`api/v1/metrics?range=${$("#range").value}`);
  const boxes = [...new Set(m.rows.map((r) => r.box))].sort();
  const k = state.boxes.map((b) => `<div class="kpi"><span>${esc(b.name)} ${b.reachable ? "" : '<span class="bad">unreachable</span>'}</span><b>${mibs(b.up_bps)}</b>
    <span>${b.torrents} torrents · ${b.downloading} dl · ${b.leechers} leechers</span><br><span>${gb(b.room.placeable)} placeable${b.allowance_spent ? ' · <span class="warn">allowance spent</span>' : ""}</span></div>`);
  const total = state.boxes.reduce((a, b) => a + (b.up_bps || 0), 0);
  render($("#kpis"), `<div class="kpi"><span>All boxes</span><b>${mibs(total)}</b><span>${state.boxes.reduce((a, b) => a + b.torrents, 0)} torrents</span></div>` + k.join(""));
  chart("#chart-up", boxes, series(m.rows, boxes, "up"), (v) => (v / MIB).toFixed(1) + " MiB/s");
  chart("#chart-n", boxes, series(m.rows, boxes, "n_total"), (v) => String(v | 0));
  chart("#chart-disk", boxes, series(m.rows, boxes, "disk_used"), (v) => (v / GB).toFixed(0) + " GB");
  chart("#chart-traffic", boxes, series(m.rows, boxes, "traffic_used"), (v) => (v / 1e12).toFixed(2) + " TB");
}
$("#range").addEventListener("change", () => overview());

// Account -------------------------------------------------------------------
async function account() {
  const a = await get("api/v1/account");
  const src = Object.entries(a.by_source_30d || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="n">${v[0]}</td><td class="n">${gb(v[1])}</td></tr>`).join("");
  render($("#account-body"), `<div class="cards"><div class="card"><h3>VIP</h3><dl>
    <dt>Status</dt><dd>${a.vip ? '<span class="good">VIP</span>' : "not VIP"}</dd>
    <dt>Until</dt><dd>${esc(a.vip_until)}</dd><dt>Cost</dt><dd>€${esc(a.vip_cost_eur_month)}/month</dd>
    <dt>VIP-only removals (30 d)</dt><dd>${a.vip_only_removals_30d} torrents, ${gb(a.vip_only_freed_bytes_30d)} freed</dd>
    <dt>H&amp;R minimum</dt><dd>1:1 or ${esc(a.hnr_min_seed_days)} days</dd></dl></div>
    <div class="card"><h3>Upload by source (30 d)</h3><table><tr><th>Source</th><th>Races</th><th>Uploaded</th></tr>${src}</table></div></div>`);
}

// Boxes ---------------------------------------------------------------------
async function boxes() {
  render($("#box-cards"), state.boxes.map((b) => {
    const psi = b.psi && b.psi.io ? `io ${r2(b.psi.io.some_avg60)}% · full ${r2(b.psi.io.full_avg60)}%` : "–";
    return `<div class="card"><h3>${esc(b.name)} <span class="${b.reachable ? "good" : "bad"}">${b.reachable ? "up" : "down"}</span></h3><dl>
    <dt>Policy</dt><dd>${esc(b.policy)}</dd>
    <dt>Upload</dt><dd>${mibs(b.up_bps)} (client ${mibs(b.client_up_bps)})</dd>
    <dt>Download</dt><dd>${mibs(b.client_dl_bps)}, ${b.downloading} active, headroom ${b.room.headroom ? "yes" : '<span class="warn">no</span>'}</dd>
    <dt>Disk</dt><dd>${gb(b.room.used)} of ${b.room.quota ? gb(b.room.quota) : "the filesystem"} · ${gb(b.room.placeable)} placeable</dd>
    <dt>Traffic</dt><dd>${b.traffic_quota_tb ? `${r2(b.traffic_used_tb)} of ${esc(b.traffic_quota_tb)} TB${b.allowance_spent ? ' <span class="warn">spent</span>' : ""}` : "unlimited"}</dd>
    <dt>Connection</dt><dd>${esc(b.connection)} · ${b.latency_ms ? b.latency_ms.toFixed(0) + " ms" : "–"} · ${b.timeouts} timeouts</dd>
    <dt>Disk queue</dt><dd>${esc(b.queued_io_jobs)} jobs · ${esc(b.average_time_queue)} ms</dd>
    <dt>Host pressure</dt><dd>${psi}</dd>
    <dt>Addresses</dt><dd>${esc((b.ips || []).join(", ") || "–")}</dd>
    <dt>Cost</dt><dd>${b.cost_eur_month ? "€" + esc(b.cost_eur_month) + "/month" : "–"}</dd>
    <dt>Last poll</dt><dd>${ago(b.last_ok)}${b.last_error ? ` <span class="bad">${esc(b.last_error)}</span>` : ""}</dd></dl></div>`;
  }).join(""));
}

// Races ---------------------------------------------------------------------
async function races() {
  const rows = await get(`api/v1/races?days=${$("#race-days").value}`);
  render($("#race-table"), table([
    ["Added", (r) => ago(r.added_at)], ["Box", (r) => esc(r.box_)], ["Source", (r) => esc(r.source)],
    ["Name", (r) => esc(String(r.name).slice(0, 90))], ["Size", (r) => gb(r.size), "n"],
    ["Reg. after", (r) => r.registered_after == null ? "–" : r.registered_after.toFixed(0) + "s", "n"],
    ["1 h", (r) => r2(r.ratio_1h), "n"], ["3 h", (r) => r2(r.ratio_3h), "n"], ["24 h", (r) => r2(r.ratio_24h), "n"],
    ["Now", (r) => r2(r.last_ratio), "n"], ["Removed", (r) => r.removed_at ? `${ago(r.removed_at)} (${esc(r.removed_rule)})` : ""],
  ], rows));
}
$("#race-days").addEventListener("change", () => races());

// Candidates ----------------------------------------------------------------
async function candidates() {
  const c = await get("api/v1/candidates");
  const html = Object.entries(c).map(([b, rows]) => `<h2>${esc(b)}</h2>` + table([
    ["Est.", (r) => r2(r.est), "n"], ["Name", (r) => esc(String(r.name).slice(0, 90))], ["Size", (r) => gb(r.size), "n"],
    ["S", (r) => esc(r.seeders ?? "–"), "n"], ["L", (r) => esc(r.leechers ?? "–"), "n"], ["Demand", (r) => r2(r.demand), "n"],
    ["Age", (r) => r.age_h.toFixed(1) + " h", "n"], ["Basis", (r) => `<span class="muted">${esc(r.basis)}</span>`],
  ], rows)).join("");
  render($("#cand-body"), html || '<p class="muted">No scout candidates right now.</p>');
}

// Decisions -----------------------------------------------------------------
async function decisions() {
  const rows = await get("api/v1/decisions?limit=200");
  render($("#dec-table"), table([
    ["When", (r) => ago(r.ts)], ["Trigger", (r) => esc(r.trigger)], ["Name", (r) => esc(String(r.name).slice(0, 80))],
    ["Chosen", (r) => esc(r.chosen) || '<span class="muted">none</span>'],
    ["Plans", (r) => `<details><summary>${(r.summary?.plans || []).length} boxes</summary>${(r.summary?.plans || []).map((p) =>
      `<div>${esc(p.box)}: est ${r2(p.est)} ${p.eligible ? '<span class="good">eligible</span>' : `<span class="muted">${esc((p.why || []).join("; "))}</span>`}${p.evict?.length ? ` · evicts ${p.evict.length}` : ""}</div>`).join("")}
      ${r.summary?.extra?.skip ? `<div class="muted">${esc(r.summary.extra.skip)}</div>` : ""}</details>`],
    ["Outcome", (r) => (r.outcomes || []).map((o) => `${esc(o.box)} ${r2(o.last_ratio)}`).join("<br>")],
  ], rows));
}

// Settings ------------------------------------------------------------------
async function settings() {
  const s = await get("api/v1/settings");
  const ta = $("#settings-text");
  if (document.activeElement !== ta) ta.value = JSON.stringify(s.settings, null, 2);
  const a = await get("api/v1/settings/audit");
  render($("#audit"), table([["When", (r) => when(r.ts)], ["Who", (r) => esc(r.who)], ["Overrides now", (r) => `<code>${esc(JSON.stringify(r.new))}</code>`]], a));
}
$("#settings-save").addEventListener("click", async () => {
  let doc;
  try { doc = JSON.parse($("#settings-text").value); } catch (e) { $("#settings-msg").textContent = "Not JSON: " + e.message; return; }
  const r = await fetch("api/v1/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ settings: doc }) });
  const j = await r.json();
  $("#settings-msg").textContent = j.ok ? "Saved." : j.error;
  $("#settings-msg").className = j.ok ? "good" : "bad";
  settings();
});

// Data ----------------------------------------------------------------------
async function data() {
  const rows = await get("api/v1/data?days=14");
  const days = [...new Set(rows.map((r) => r.day))].sort().reverse();
  const kinds = [...new Set(rows.map((r) => r.kind))].sort();
  const cell = new Map(rows.map((r) => [r.day + "|" + r.kind, r]));
  render($("#data-table"), '<p class="muted">Events per kind per day; an empty cell on a day that has others is a gap.</p>' + table(
    [["Day", (d) => esc(d)], ...kinds.map((k) => [k, (d) => { const c = cell.get(d + "|" + k); return c ? c.n.toLocaleString() : '<span class="bad">0</span>'; }, "n"]),
     ["Bytes", (d) => gb(kinds.reduce((a, k) => a + (cell.get(d + "|" + k)?.bytes || 0), 0)), "n"]], days));
}

// Events --------------------------------------------------------------------
let es = null;
const KINDS = ["announce", "market", "capacity", "decision", "action", "alert", "registration", "setting", "tick", "torrent_state", "box_state",
  "scrape", "listing", "peers", "rss_item", "release", "tracker_status", "start", "tables", "backfill"];
$("#ev-go").addEventListener("click", () => {
  if (es) es.close();
  const k = $("#ev-kinds").value.trim();
  es = new EventSource("api/v1/events/stream" + (k ? `?kind=${encodeURIComponent(k)}` : ""));
  const log = $("#ev-log");
  log.textContent = "";
  const on = (ev) => {
    const e = JSON.parse(ev.data);
    log.textContent = `${e.ts} #${e.seq} ${e.kind} ${e.box || ""} ${e.hash ? e.hash.slice(0, 8) : ""} ${JSON.stringify(e.data).slice(0, 400)}\n` + log.textContent.slice(0, 200000);
  };
  KINDS.forEach((kind) => es.addEventListener(kind, on));
});

async function refresh() {
  try {
    await loadState();
    const pages = { overview, account, boxes, races, candidates, decisions, settings, data, events: async () => { } };
    await pages[tab]?.();
  } catch (e) { $("#status").textContent = "error: " + e.message; }
}
show(tab);
setInterval(() => { if (tab !== "settings" && tab !== "events") refresh(); else loadState().catch(() => { }); }, 30000);
