/* PhacoSight SPA — no dependencies, hash routing.
   Routes: #/surgeries (default) · #/video/<case>?t=<s> · #/phases/<phase>
           #/physicians/<name> · #/upload   (old #/videos #/library #/progress alias) */
"use strict";

const PHASE_COLORS = {
  idle: "#d9d7d0", "Incision": "#2a78d6", "Viscoelastic": "#86b6ef",
  "Capsulorhexis": "#1baf7a", "Hydrodissection": "#7fd6b3",
  "Phacoemulsification": "#eb6834", "Irrigation/Aspiration": "#f3a97e",
  "Capsule Pulishing": "#eda100", "Lens Implantation": "#4a3aa7",
  "Lens positioning": "#9085e9", "Viscoelastic_Suction": "#e87ba4",
  "Anterior_Chamber Flushing": "#c2569b", "Tonifying/Antibiotics": "#008300",
};
const nice = p => p === "Capsule Pulishing" ? "Capsule Polishing" : p.replaceAll("_", " ");
const fmtT = s => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
const esc = s => String(s ?? "").replace(/[&<>"']/g,
  c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
};
const api = async (path) => {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
};
const view = document.getElementById("view");
const tooltip = document.getElementById("tooltip");
const isFlagged = s => s.phase !== "idle" && (s.confidence < 0.7 || s.disagreement > 0.5);

let videosCache = null, videosCacheAt = 0;
async function getVideos() {
  if (!videosCache || Date.now() - videosCacheAt > 60_000) {
    videosCache = await api("/api/videos");
    videosCacheAt = Date.now();
  }
  return videosCache;
}
const invalidateVideos = () => { videosCache = null; };
let lastListOrder = null;  // case ids in the surgeries table's current filter order

let overlayInfo = null;    // {available, classes: [{name, color}]}
async function getOverlayInfo() {
  if (!overlayInfo) {
    try { overlayInfo = await api("/api/overlay/legend"); }
    catch { overlayInfo = {available: false, classes: []}; }
  }
  return overlayInfo;
}

/* ---------------- router ---------------- */
window.addEventListener("hashchange", route);
window.addEventListener("load", route);
const NAV_OF = {surgeries: "surgeries", videos: "surgeries", video: "surgeries",
                phases: "phases", library: "phases",
                physicians: "physicians", progress: "physicians", upload: "upload"};

async function route() {
  const [_, page, rawArg] = location.hash.split("/");
  const [arg, query] = (rawArg || "").split("?");
  const params = new URLSearchParams(query || "");
  document.querySelectorAll("nav a").forEach(a =>
    a.classList.toggle("active", a.dataset.nav === (NAV_OF[page] || "surgeries")));
  tooltip.hidden = true;
  try {
    if (page === "video" && arg) return await renderDetail(decodeURIComponent(arg), params);
    if (page === "phases" || page === "library")
      return await renderPhases(arg ? decodeURIComponent(arg) : null);
    if (page === "physicians" || page === "progress")
      return await renderPhysicians(arg ? decodeURIComponent(arg) : null);
    if (page === "upload") return renderUpload();
    return await renderList();
  } catch (err) {
    showError(err);
  }
}

function showError(err) {
  view.innerHTML = "";
  const c = el("div", "card error-card");
  c.append(el("h2", "", "Something went wrong"));
  c.append(el("p", "sub", esc(err && err.message || err)));
  const b = el("button", "", "Retry");
  b.onclick = () => { invalidateVideos(); route(); };
  c.append(b);
  view.append(c);
}

/* ---------------- surgeries list ---------------- */
async function renderList() {
  view.innerHTML = "<p class='empty'>Loading surgeries…</p>";
  const vids = await getVideos();
  const n = vids.length;
  const flagged = vids.filter(v => v.flag_fraction > 0.25).length;
  const uploaded = vids.filter(v => v.source === "uploaded").length;
  const hours = vids.reduce((a, v) => a + v.duration_s, 0) / 3600;

  view.innerHTML = "";
  view.append(el("h1", "", "Surgeries"));
  view.append(el("p", "sub",
    "Every analyzed surgery, with automated quality flags. Click a row to review."));
  try {
    const active = (await api("/api/jobs")).filter(j => !["done", "error"].includes(j.status));
    if (active.length) {
      const b = el("a", "banner",
        `⏳ ${active.length} analys${active.length === 1 ? "is" : "es"} running — view queue`);
      b.href = "#/upload";
      view.append(b);
    }
  } catch { /* banner is best-effort */ }
  const stats = el("div", "stats");
  for (const [b, s] of [[n, "surgeries analyzed"], [hours.toFixed(0) + " h", "video indexed"],
                        [flagged, "need manual review"], [uploaded, "uploaded by you"]]) {
    const c = el("div", "stat card"); c.append(el("b", "num", b), el("span", "", s));
    stats.append(c);
  }
  view.append(stats);

  const bar = el("div", "toolbar");
  const search = el("input"); search.type = "search"; search.placeholder = "Search case…";
  const srcSel = el("select", "", `<option value="">all sources</option>
    <option value="library">base archive</option><option value="labeled">annotated set</option>
    <option value="uploaded">uploaded</option>`);
  const docSel = el("select", "", `<option value="">all physicians</option>` +
    [...new Set(vids.map(v => v.physician).filter(Boolean))].sort()
      .map(d => `<option>${esc(d)}</option>`).join(""));
  const flagSel = el("select", "", `<option value="">any status</option>
    <option value="review">needs review</option><option value="clean">clean</option>`);
  bar.append(search, srcSel, docSel, flagSel);
  view.append(bar);

  const card = el("div", "card");
  const table = el("table", "videos",
    `<thead><tr><th>Case</th><th>Timeline</th><th>Duration</th><th>Physician</th>
     <th>Date</th><th>Flags</th><th>Status</th></tr></thead>`);
  const tbody = el("tbody");
  table.append(tbody); card.append(table); view.append(card);

  let showAll = false;
  const draw = () => {
    const q = search.value.toLowerCase();
    tbody.innerHTML = "";
    const rows = vids.filter(v =>
      v.case.toLowerCase().includes(q) &&
      (!srcSel.value || v.source === srcSel.value) &&
      (!docSel.value || v.physician === docSel.value) &&
      (!flagSel.value || (flagSel.value === "review") === (v.flag_fraction > 0.25)));
    lastListOrder = rows.map(v => v.case);
    const shown = showAll ? rows : rows.slice(0, 400);
    for (const v of shown) {
      const tr = el("tr");
      const status = v.flag_fraction > 0.25
        ? "<span class='chip warn'>review</span>" : "<span class='chip ok'>clean</span>";
      const src = {library: "archive", labeled: "annotated", uploaded: "uploaded"}[v.source];
      tr.innerHTML = `<td><b>${esc(v.case)}</b><br><span class='chip neutral'>${src}</span></td>
        <td><div class='minitl' data-case='${esc(v.case)}'></div></td>
        <td class='num'>${fmtT(v.duration_s)}</td>
        <td>${v.physician ? esc(v.physician) : "<span class='sub'>—</span>"}</td>
        <td class='num'>${v.surgery_date ? esc(v.surgery_date) : "<span class='sub'>—</span>"}</td>
        <td class='num'>${(v.flag_fraction * 100).toFixed(0)}%</td><td>${status}</td>`;
      tr.onclick = () => location.hash = `#/video/${encodeURIComponent(v.case)}`;
      tbody.append(tr);
    }
    if (!showAll && rows.length > shown.length) {
      const tr = el("tr", "show-all");
      tr.innerHTML = `<td colspan='7'>Showing ${shown.length} of ${rows.length} — show all</td>`;
      tr.onclick = () => { showAll = true; draw(); };
      tbody.append(tr);
    }
    lazyMiniTimelines();
    if (!rows.length) tbody.innerHTML = "<tr><td colspan='7' class='empty'>No matches.</td></tr>";
  };
  [search, srcSel, docSel, flagSel].forEach(x => x.addEventListener("input", () => {
    showAll = false; draw();
  }));
  draw();
}

const miniCache = {};
function lazyMiniTimelines() {
  const io = new IntersectionObserver(async entries => {
    for (const en of entries) {
      if (!en.isIntersecting) continue;
      io.unobserve(en.target);
      const c = en.target.dataset.case;
      try {
        miniCache[c] = miniCache[c] || (async () => (await api(`/api/videos/${encodeURIComponent(c)}`)))();
        const d = await miniCache[c];
        const segs = d.segments.length ? d.segments : (d.ground_truth || []);
        const total = d.duration_s;
        en.target.innerHTML = segs.map(s =>
          `<i style="width:${(100 * (s.end_s - s.start_s) / total).toFixed(2)}%;
            background:${PHASE_COLORS[s.phase]}"></i>`).join("");
      } catch { en.target.textContent = "–"; }
    }
  });
  document.querySelectorAll(".minitl").forEach(t => io.observe(t));
}

/* ---------------- detail ---------------- */
async function renderDetail(caseId, params) {
  view.innerHTML = "<p class='empty'>Loading analysis…</p>";
  const d = await api(`/api/videos/${encodeURIComponent(caseId)}`);
  view.innerHTML = "";

  /* header with prev/next case navigation */
  const head = el("div", "detail-head");
  const titleWrap = el("div");
  titleWrap.append(el("h1", "", esc(caseId)));
  const sub = d.metrics
    ? `Automated timeline · ${d.metrics.n_segments} segments · median confidence
       ${d.metrics.median_confidence.toFixed(2)} ${d.ground_truth ? "· expert annotations available" : ""}
       ${d.operator ? `· operator: <b>${esc(d.operator)}</b>` : ""}`
    : "Expert annotations only (not model-analyzed)";
  titleWrap.append(el("p", "sub", sub));
  head.append(titleWrap);
  const nav = el("div", "case-nav");
  head.append(nav);
  try {
    const order = (lastListOrder && lastListOrder.includes(caseId))
      ? lastListOrder : (await getVideos()).map(v => v.case).sort();
    const i = order.indexOf(caseId);
    if (i > 0) nav.append(Object.assign(el("a", "ghost-link",
      `← ${esc(order[i - 1])}`), {href: `#/video/${encodeURIComponent(order[i - 1])}`}));
    if (i >= 0 && i < order.length - 1) nav.append(Object.assign(el("a", "ghost-link",
      `${esc(order[i + 1])} →`), {href: `#/video/${encodeURIComponent(order[i + 1])}`}));
  } catch { /* prev/next is optional */ }
  if (d.source === "uploaded") {
    const re = el("button", "ghost", "Re-analyze");
    re.onclick = async () => {
      re.disabled = true;
      const r = await fetch(`/api/reanalyze/${encodeURIComponent(caseId)}`, {method: "POST"});
      if (r.ok) location.hash = "#/upload";
      else { re.disabled = false; alert(`Re-analyze failed: ${await r.text()}`); }
    };
    const del = el("button", "ghost danger", "Delete");
    del.onclick = async () => {
      if (!confirm(`Delete ${caseId} — its video and analysis? This cannot be undone.`)) return;
      const r = await fetch(`/api/videos/${encodeURIComponent(caseId)}`, {method: "DELETE"});
      if (r.ok) { invalidateVideos(); location.hash = "#/surgeries"; }
      else alert(`Delete failed: ${await r.text()}`);
    };
    nav.append(re, del);
  }
  view.append(head);

  const grid = el("div", "detail-grid");
  view.append(grid);

  /* left: player + timelines */
  const left = el("div", "card player-card");
  let vid = null;
  const ov = await getOverlayInfo();
  const canOverlay = ov.available && d.has_video !== false;
  if (d.has_video === false) {
    left.append(el("div", "no-video",
      "No video file available for this case — timelines and metrics only."));
  } else {
    const wrap = el("div", "player-wrap");
    vid = document.createElement("video");
    vid.controls = true; vid.preload = "metadata";
    vid.src = `/api/stream/${encodeURIComponent(caseId)}`;
    wrap.append(vid);
    left.append(wrap);
    if (canOverlay) {
      const ovImg = el("img", "ov-img"); ovImg.hidden = true; ovImg.alt = "";
      const ovSpin = el("div", "ov-spin", "rendering overlay…"); ovSpin.hidden = true;
      wrap.append(ovImg, ovSpin);
      const row = el("div", "ov-row");
      const toggle = el("label", "ov-toggle",
        `<input type="checkbox"> AI anatomy overlay <span class='sub' style='margin:0'>(paused frames)</span>`);
      const box = toggle.querySelector("input");
      row.append(toggle);
      const lg = el("span", "legend ov-legend");
      for (const c of ov.classes)
        lg.append(el("span", "", `<i style="background:${c.color}"></i>${esc(c.name)}`));
      row.append(lg);
      left.append(row);
      const updateOv = async () => {
        if (!box.checked || !vid.paused) { ovImg.hidden = true; ovSpin.hidden = true; return; }
        ovSpin.hidden = false;
        const t = vid.currentTime.toFixed(1);
        const src = `/api/overlay?case=${encodeURIComponent(caseId)}&t=${t}&h=480`;
        try {
          await new Promise((res, rej) => {
            const im = new Image(); im.onload = res; im.onerror = rej; im.src = src;
          });
          if (box.checked && vid.paused && vid.currentTime.toFixed(1) === t) {
            ovImg.src = src; ovImg.hidden = false;
          }
        } catch { /* overlay is best-effort */ }
        ovSpin.hidden = true;
      };
      box.addEventListener("change", updateOv);
      vid.addEventListener("pause", updateOv);
      vid.addEventListener("seeked", () => { if (vid.paused) updateOv(); });
      vid.addEventListener("play", () => { ovImg.hidden = true; ovSpin.hidden = true; });
    }
  }
  const now = el("div", "now-playing", "&nbsp;");
  left.append(now);
  const tlWrap = el("div", "tl-wrap");
  left.append(tlWrap);
  const segs = d.segments.length ? d.segments : null;
  if (segs) {
    tlWrap.append(el("div", "tl-label", vid ? "model timeline — click to jump" : "model timeline"));
    tlWrap.append(timelineSVG(segs, d.duration_s, vid, true));
  }
  if (d.ground_truth) {
    const gtSegs = withIdleGaps(d.ground_truth, d.duration_s);
    tlWrap.append(el("div", "tl-label", "expert annotation"));
    tlWrap.append(timelineSVG(gtSegs, d.duration_s, vid, false));
  }
  const legend = el("div", "legend");
  const present = new Set((segs || []).concat(d.ground_truth || []).map(s => s.phase));
  for (const p of d.phase_names.filter(p => present.has(p)))
    legend.append(el("span", "", `<i style="background:${PHASE_COLORS[p]}"></i>${nice(p)}`));
  left.append(legend);

  /* AI-annotated keyframe per phase (midpoint of its longest confident segment) */
  if (canOverlay && segs && d.metrics) {
    const kfs = [];
    for (const p of d.metrics.phases) {
      const cand = segs.filter(s => s.phase === p.phase && !isFlagged(s));
      if (!cand.length) continue;
      const best = cand.reduce((a, b) => b.end_s - b.start_s > a.end_s - a.start_s ? b : a);
      kfs.push({phase: p.phase, t: (best.start_s + best.end_s) / 2});
    }
    if (kfs.length) {
      left.append(el("div", "tl-label", "AI-annotated keyframes — click to jump"));
      const strip = el("div", "keyframes");
      for (const k of kfs) {
        const c = el("div", "kf");
        c.innerHTML = `<img loading="lazy" style="border-color:${PHASE_COLORS[k.phase]}"
            src="/api/overlay?case=${encodeURIComponent(caseId)}&t=${k.t.toFixed(1)}&h=180"
            alt="${esc(nice(k.phase))}">
          <span>${nice(k.phase)}</span>`;
        if (vid) c.onclick = () => { vid.currentTime = k.t; vid.pause(); };
        strip.append(c);
      }
      left.append(strip);
    }
  }
  grid.append(left);

  /* deep-link seek (#/video/<case>?t=<sec>) */
  const t0 = parseFloat(params && params.get("t"));
  if (vid && Number.isFinite(t0)) {
    vid.addEventListener("loadedmetadata", () => { vid.currentTime = t0; }, {once: true});
    const hit = tlWrap.querySelector(
      `rect.seg[data-start]`) && [...tlWrap.querySelectorAll("rect.seg[data-start]")]
      .find(r => +r.dataset.start <= t0 && t0 < +r.dataset.end);
    if (hit) { hit.classList.add("flash"); setTimeout(() => hit.classList.remove("flash"), 2500); }
  }

  /* playhead sync */
  if (vid) vid.addEventListener("timeupdate", () => {
    tlWrap.querySelectorAll(".playhead").forEach(ph =>
      ph.setAttribute("x", `${(100 * vid.currentTime / d.duration_s).toFixed(3)}%`));
    const cur = (segs || []).find(s => vid.currentTime >= s.start_s && vid.currentTime < s.end_s);
    now.innerHTML = cur
      ? `<i style="width:10px;height:10px;border-radius:3px;background:${PHASE_COLORS[cur.phase]}"></i>
         <b>${nice(cur.phase)}</b> · confidence ${cur.confidence.toFixed(2)}
         ${isFlagged(cur) ? "<span class='chip warn'>flagged</span>" : ""}`
      : "&nbsp;";
  });

  /* right: metrics */
  const side = el("div", "side");
  grid.append(side);
  if (d.metrics) {
    const m = d.metrics;
    const panel = el("div", "card panel");
    panel.append(el("h2", "", "Phase durations vs cohort"));
    panel.append(el("p", "sub", `Marker = this surgery; shaded band = cohort interquartile
      range (n≈${m.phases[0]?.n_cohort ?? "–"}). Phase names link to the cohort profile.`));
    for (const p of m.phases) panel.append(phaseRow(p));
    const idle = el("div", "ph-row");
    idle.innerHTML = `<span class='ph-name'><i style="background:${PHASE_COLORS.idle}"></i>idle
      (between steps)</span><span></span>
      <span class='ph-val num'>${p0(m.idle_s)}</span>`;
    panel.append(idle);
    side.append(panel);

    const fl = el("div", "card panel");
    fl.append(el("h2", "", m.flagged_idx.length
      ? `Flagged segments (${m.flagged_idx.length})` : "Flagged segments"));
    if (m.flagged_idx.length) {
      fl.append(el("p", "sub", "Low confidence or model disagreement — verify manually."));
      const ul = el("ul", "flags");
      for (const i of m.flagged_idx) {
        const s = d.segments[i];
        const li = el("li", "", `<b>${nice(s.phase)}</b> ${fmtT(s.start_s)}–${fmtT(s.end_s)}
          · conf ${s.confidence.toFixed(2)}`);
        if (vid) li.onclick = () => { vid.currentTime = s.start_s; vid.play(); };
        ul.append(li);
      }
      fl.append(ul);
    } else {
      fl.append(el("p", "sub", "None — every segment passed the confidence gate."));
    }
    side.append(fl);

    const prov = el("div", "card panel");
    prov.append(el("h2", "", "Provenance"));
    prov.append(el("p", "sub", `${esc((d.provenance || {}).stack || "")}<br>
      model ${esc(((d.provenance || {}).git_sha || "").slice(0, 8))} · analyzed at
      ${esc(d.inference_fps || "?")} fps`));
    side.append(prov);
  }
}

function p0(v) { return `${Math.round(v)} s`; }

function phaseRow(p) {
  const row = el("div", "ph-row");
  const maxS = Math.max(p.cohort_p75 * 1.6 || 0, p.total_s * 1.15, 1);
  const x = v => `${(100 * v / maxS).toFixed(1)}%`;
  const pctText = p.percentile === null ? "" :
    `<small>p${Math.round(p.percentile)} of cohort</small>`;
  row.innerHTML = `
    <a class='ph-name' href="#/phases/${encodeURIComponent(p.phase)}"
       title="cohort profile for ${esc(nice(p.phase))}">
      <i style="background:${PHASE_COLORS[p.phase]}"></i>${nice(p.phase)}</a>
    <span class='pct-track'>
      <span class='pct-band' style="left:0;right:0"></span>
      <span class='pct-iqr' style="left:${x(p.cohort_p25)};width:${x(p.cohort_p75 - p.cohort_p25)}"></span>
      <span class='pct-marker' style="left:${x(p.total_s)}"></span>
    </span>
    <span class='ph-val num'>${p0(p.total_s)}${pctText}</span>`;
  return row;
}

function withIdleGaps(gt, total) {
  const out = [];
  let t = 0;
  for (const s of [...gt].sort((a, b) => a.start_s - b.start_s)) {
    if (s.start_s > t + 0.01) out.push({phase: "idle", start_s: t, end_s: s.start_s});
    out.push(s);
    t = Math.max(t, s.end_s);
  }
  if (t < total) out.push({phase: "idle", start_s: t, end_s: total});
  return out;
}

function timelineSVG(segs, total, vid, isModel) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("timeline");
  svg.setAttribute("viewBox", "0 0 100 10");
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("height", isModel ? "34" : "22");
  svg.innerHTML = `<defs><pattern id="hatch" width="3" height="3"
      patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
      <rect width="3" height="3" fill="rgba(0,0,0,0)"/>
      <line x1="0" y1="0" x2="0" y2="3" stroke="rgba(30,30,30,.55)" stroke-width="1.1"/>
    </pattern></defs>`;
  for (const s of segs) {
    const r = document.createElementNS(svg.namespaceURI, "rect");
    r.classList.add("seg");
    r.setAttribute("x", 100 * s.start_s / total);
    r.setAttribute("width", Math.max(0.15, 100 * (s.end_s - s.start_s) / total));
    r.setAttribute("y", 0); r.setAttribute("height", 10);
    r.setAttribute("fill", PHASE_COLORS[s.phase]);
    r.dataset.start = s.start_s; r.dataset.end = s.end_s;
    if (isModel && s.confidence !== undefined)
      r.setAttribute("fill-opacity", (0.45 + 0.55 * Math.min(1, s.confidence)).toFixed(2));
    svg.append(r);
    if (isModel && isFlagged(s)) {
      const h = r.cloneNode(); h.setAttribute("fill", "url(#hatch)");
      h.classList.add("seg"); svg.append(h);
    }
    const target = svg.lastChild;
    for (const t of (isModel && isFlagged(s)) ? [r, target] : [r]) {
      t.addEventListener("click", () => { if (vid) { vid.currentTime = s.start_s; vid.play(); } });
      t.addEventListener("mousemove", ev => {
        tooltip.hidden = false;
        tooltip.style.left = `${ev.clientX + 14}px`;
        tooltip.style.top = `${ev.clientY + 14}px`;
        tooltip.innerHTML = `<b>${nice(s.phase)}</b> ${fmtT(s.start_s)}–${fmtT(s.end_s)}
          ${s.confidence !== undefined ? `<br><small>confidence ${s.confidence.toFixed(2)}
          · disagreement ${s.disagreement.toFixed(2)}</small>` : ""}`;
      });
      t.addEventListener("mouseleave", () => tooltip.hidden = true);
    }
  }
  const ph = document.createElementNS(svg.namespaceURI, "rect");
  ph.classList.add("playhead");
  ph.setAttribute("x", "0"); ph.setAttribute("y", "-0.5");
  ph.setAttribute("width", "0.28"); ph.setAttribute("height", "11");
  ph.setAttribute("fill", "#1c211f");
  svg.append(ph);
  return svg;
}

/* ---------------- phase explorer ---------------- */
async function renderPhases(phase) {
  const phases = Object.keys(PHASE_COLORS).filter(p => p !== "idle");
  if (!phase || !phases.includes(phase)) phase = "Incision";
  view.innerHTML = "";
  view.append(el("h1", "", "Phase explorer"));
  view.append(el("p", "sub",
    "The cohort profile for each surgical step, with high-confidence example segments " +
    "from every indexed surgery (flagged segments are excluded)."));
  const bar = el("div", "toolbar");
  const sel = el("select");
  for (const p of phases) sel.append(new Option(nice(p), p));
  sel.value = phase;
  sel.addEventListener("input", () =>
    location.hash = `#/phases/${encodeURIComponent(sel.value)}`);
  const conf = el("input"); conf.type = "range"; conf.min = 0.7; conf.max = 0.99;
  conf.step = 0.01; conf.value = 0.9;
  const confLabel = el("span", "sub", "");
  const count = el("span", "sub", "");
  bar.append(sel, confLabel, conf, count);
  view.append(bar);
  const hero = el("div", "card phase-hero");
  view.append(hero);
  const grid = el("div", "lib-grid");
  view.append(grid);

  const drawHero = async () => {
    try {
      const st = await api(`/api/phase_stats?phase=${encodeURIComponent(phase)}`);
      const color = PHASE_COLORS[phase];
      hero.innerHTML = `
        <div>
          <h2 style="display:flex;align-items:center">
            <i style="width:11px;height:11px;border-radius:3px;background:${color};
              display:inline-block;margin-right:8px"></i>${nice(phase)} — cohort profile</h2>
          <div class="tiles">
            <div class="tile"><b class="num">${st.total.p50}s</b>
              <span>median total per surgery</span></div>
            <div class="tile"><b class="num">${st.total.p25}–${st.total.p75}s</b>
              <span>interquartile range</span></div>
            <div class="tile"><b class="num">${(st.presence * 100).toFixed(0)}%</b>
              <span>of ${st.n_videos.toLocaleString()} surgeries include it</span></div>
            <div class="tile"><b class="num">${st.segments_per_video}×</b>
              <span>typical repeats (median ${st.segment_median_s}s each)</span></div>
          </div>
          <div style="margin-top:14px">
            <span class="sub" style="margin:0">typical position in surgery</span>
            <div class="posbar"><i style="left:${(st.typical_position * 100).toFixed(1)}%"></i></div>
            <div style="display:flex;justify-content:space-between" class="axis-note">
              <span>start</span><span>end</span></div>
          </div>
        </div>
        <div>
          <span class="sub" style="margin:0">distribution of per-surgery totals (s)</span>
          ${histSVG(st.histogram, color)}
        </div>`;
    } catch { hero.innerHTML = "<p class='empty'>No cohort statistics for this phase.</p>"; }
  };

  const draw = async () => {
    drawHero();
    confLabel.textContent = `min confidence ${Number(conf.value).toFixed(2)}`;
    grid.innerHTML = "<p class='empty'>Searching…</p>";
    const res = await api(`/api/search?phase=${encodeURIComponent(phase)}&min_conf=${conf.value}`);
    count.textContent = `${res.total} segments found`;
    grid.innerHTML = "";
    for (const h of res.hits) {
      const mid = (h.start_s + h.end_s) / 2;
      const seek = `#/video/${encodeURIComponent(h.case)}?t=${h.start_s.toFixed(1)}`;
      const c = el("div", "hit card");
      c.innerHTML = `<a href="${seek}"><img loading="lazy"
          src="/api/frame?case=${encodeURIComponent(h.case)}&t=${mid.toFixed(1)}"
          alt="${esc(h.case)}"></a>
        <div class="meta"><a href="${seek}">${esc(h.case)}</a>
          <span class="num">${fmtT(h.start_s)}–${fmtT(h.end_s)}</span>
          <a href="/api/clip?case=${encodeURIComponent(h.case)}&start=${h.start_s}&end=${h.end_s}"
             title="download clip">⬇ clip</a></div>`;
      grid.append(c);
    }
    if (!res.hits.length) grid.innerHTML = "<p class='empty'>No segments at this confidence.</p>";
  };
  conf.addEventListener("change", draw);
  await draw();
}

/* ---------------- upload ---------------- */
function renderUpload() {
  view.innerHTML = "";
  view.append(el("h1", "", "Analyze a new surgery"));
  view.append(el("p", "sub",
    "Upload an mp4 recording. Analysis runs on the department GPU — a 7-minute surgery " +
    "takes about a minute once the models are warm."));
  const box = el("div", "card upload-box");
  const input = el("input"); input.type = "file"; input.accept = "video/mp4";
  const doc = el("input"); doc.type = "text"; doc.placeholder = "e.g. Dr. Rivera";
  const date = el("input"); date.type = "date";
  const op = el("select", "", `<option value="">not specified</option>
    <option value="resident">resident</option><option value="attending">attending</option>
    <option value="mixed">mixed (hand-offs)</option>`);
  const fileF = el("label", "field", "Recording (mp4)");
  fileF.append(input);
  const docF = el("label", "field",
    "Physician <span class='sub' style='margin:0'>(enables the progress view)</span>");
  docF.append(doc);
  const dateF = el("label", "field", "Surgery date");
  dateF.append(date);
  const opF = el("label", "field",
    "Operator <span class='sub' style='margin:0'>(who performed the surgery)</span>");
  opF.append(op);
  const go = el("button", "", "Upload & analyze");
  const msg = el("p", "form-msg"); msg.hidden = true;
  const prog = el("div", "progress"); prog.hidden = true;
  prog.innerHTML = `<div class="bar"><i style="width:0%"></i></div><p></p>`;
  box.append(fileF, docF, dateF, opF, go, msg, prog);
  view.append(box);

  const showMsg = t => { msg.hidden = false; msg.textContent = t; };
  const setBar = (pct, text) => {
    prog.hidden = false;
    prog.querySelector("i").style.width = `${pct.toFixed(0)}%`;
    prog.querySelector("p").textContent = text;
  };

  /* stages after the transfer itself (0-70%) */
  const stages = {queued: 72, loading: 78, features: 86, inference: 94, done: 100};
  const pollJob = (jobId, caseId) => {
    let delay = 1500;
    const goDetail = () => {
      invalidateVideos();
      location.hash = `#/video/${encodeURIComponent(caseId)}`;
    };
    const tick = async () => {
      if (!document.body.contains(prog)) return;   // navigated away
      try {
        const j = await api(`/api/jobs/${jobId}`);
        delay = 1500;
        setBar(stages[j.status] ?? 72, `${j.status} — ${j.detail}`);
        if (j.status === "done") return goDetail();
        if (j.status === "error") {
          showMsg(`Analysis failed: ${j.detail}`); go.disabled = false; return;
        }
      } catch (e) {
        if (String(e.message).startsWith("404")) {
          /* server restarted and lost the job — did the timeline land anyway? */
          try { await api(`/api/videos/${encodeURIComponent(caseId)}`); return goDetail(); }
          catch {
            showMsg("The server restarted and lost this job — re-upload if the analysis "
              + "didn't finish."); go.disabled = false; return;
          }
        }
        delay = Math.min(delay * 2, 10_000);       // transient failure: back off
      }
      setTimeout(tick, delay);
    };
    setTimeout(tick, delay);
  };

  go.onclick = () => {
    if (!input.files.length) return showMsg("Choose an mp4 first.");
    msg.hidden = true; go.disabled = true;
    const fd = new FormData(); fd.append("file", input.files[0]);
    fd.append("physician", doc.value); fd.append("surgery_date", date.value);
    fd.append("operator", op.value);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");
    xhr.upload.onprogress = e => {
      if (e.lengthComputable) setBar(70 * e.loaded / e.total,
        `uploading — ${(e.loaded / 1e6).toFixed(0)} / ${(e.total / 1e6).toFixed(0)} MB`);
    };
    xhr.onerror = () => { showMsg("Upload failed — network error."); go.disabled = false; };
    xhr.onload = () => {
      if (xhr.status !== 200) {
        showMsg(`Upload failed: ${xhr.responseText}`); go.disabled = false; return;
      }
      setBar(70, "upload complete — queued for analysis");
      const {job_id, case: caseId} = JSON.parse(xhr.responseText);
      pollJob(job_id, caseId);
    };
    xhr.send(fd);
  };

  /* recent analyses (in-memory queue on the server) */
  const jobsCard = el("div", "card panel jobs-card");
  view.append(jobsCard);
  const drawJobs = async () => {
    if (!document.body.contains(jobsCard)) return;  // navigated away
    try {
      const js = await api("/api/jobs");
      jobsCard.innerHTML = "<h2>Recent analyses</h2>";
      if (!js.length) {
        jobsCard.append(el("p", "sub",
          "None since the server started. Finished analyses live under Surgeries."));
      } else {
        const ul = el("ul", "jobs");
        for (const j of js) {
          const chip = {done: "ok", error: "warn"}[j.status] || "neutral";
          const name = j.status === "done"
            ? `<a href="#/video/${encodeURIComponent(j.case)}">${esc(j.case)}</a>`
            : `<b>${esc(j.case)}</b>`;
          ul.append(el("li", "", `<span class='chip ${chip}'>${esc(j.status)}</span>
            ${name} <span class='sub' style='margin:0'>${esc(j.detail)}</span>`));
        }
        jobsCard.append(ul);
      }
    } catch { /* jobs list is best-effort */ }
    setTimeout(drawJobs, 5000);
  };
  drawJobs();
}

/* ---------------- shared mini-charts ---------------- */
function histSVG(h, color) {
  const max = Math.max(...h.counts, 1);
  const n = h.counts.length;
  const bars = h.counts.map((c, i) => {
    const bh = 86 * c / max;
    return `<rect x="${(i * 100 / n).toFixed(2)}" y="${(90 - bh).toFixed(2)}"
      width="${(100 / n - 0.5).toFixed(2)}" height="${bh.toFixed(2)}"
      fill="${color}" fill-opacity="0.85" rx="0.6"></rect>`;
  }).join("");
  const lo = h.edges[0], hi = h.edges[h.edges.length - 1];
  return `<svg class="hist" viewBox="0 0 100 100" preserveAspectRatio="none">
      <line x1="0" y1="90" x2="100" y2="90" stroke="#d0d5d1" stroke-width="0.6"/>
      ${bars}
      <text x="0" y="98" font-size="6" fill="#8a948f">${Math.round(lo)}s</text>
      <text x="100" y="98" font-size="6" fill="#8a948f" text-anchor="end">${Math.round(hi)}s</text>
    </svg>`;
}

function sparkSVG(points, cohort, color) {
  /* points: [{x: 0-1, y, label, href?}], cohort: {p25,p50,p75} in y units */
  const ys = points.map(p => p.y).concat(cohort ? [cohort.p25, cohort.p75] : []);
  const yMax = Math.max(...ys) * 1.15 || 1;
  const X = x => 6 + 90 * x, Y = y => 92 - 84 * y / yMax;
  let band = "";
  if (cohort) {
    band = `<rect x="6" y="${Y(cohort.p75).toFixed(1)}" width="90"
        height="${(Y(cohort.p25) - Y(cohort.p75)).toFixed(1)}" fill="#e3f1ef"></rect>
      <line x1="6" x2="96" y1="${Y(cohort.p50).toFixed(1)}" y2="${Y(cohort.p50).toFixed(1)}"
        stroke="#0e7569" stroke-width="0.5" stroke-dasharray="2 2"/>`;
  }
  const path = points.map((p, i) =>
    `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(" ");
  const dots = points.map(p => {
    const c = `<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="1.8" fill="${color}">
       <title>${esc(p.label)}</title></circle>`;
    return p.href ? `<a href="${p.href}">${c}</a>` : c;
  }).join("");
  return `<svg class="spark" viewBox="0 0 100 100" preserveAspectRatio="none">
      ${band}
      <line x1="6" y1="92" x2="96" y2="92" stroke="#d0d5d1" stroke-width="0.5"/>
      ${points.length > 1 ? `<path d="${path}" fill="none" stroke="${color}" stroke-width="1.1"/>` : ""}
      ${dots}
      <text x="6" y="8" font-size="6" fill="#8a948f">${Math.round(yMax)}s</text>
    </svg>`;
}

/* ---------------- physicians ---------------- */
async function renderPhysicians(name) {
  if (name) return renderPhysicianDetail(name);
  view.innerHTML = "";
  view.append(el("h1", "", "Physicians"));
  view.append(el("p", "sub",
    "Every physician with uploaded surgeries. Open one to see their timing trends " +
    "against the cohort."));
  const docs = await api("/api/physicians");
  if (!docs.length) {
    view.append(el("p", "empty",
      "No surgeries with a physician name yet. Upload a video with the physician field " +
      "filled in and it will appear here."));
    return;
  }
  const grid = el("div", "doc-grid");
  for (const d of docs) {
    const c = el("a", "card doc-card");
    c.href = `#/physicians/${encodeURIComponent(d.name)}`;
    c.innerHTML = `<b>${esc(d.name)}</b>
      <span class="sub" style="margin:0">${d.n_videos} surger${d.n_videos === 1 ? "y" : "ies"}
      ${d.last_date ? `· latest ${esc(d.last_date)}` : ""}</span>`;
    grid.append(c);
  }
  view.append(grid);
}

async function renderPhysicianDetail(name) {
  view.innerHTML = "";
  const head = el("div", "detail-head");
  const tw = el("div");
  tw.append(el("h1", "", esc(name)));
  tw.append(el("p", "sub",
    "Surgeries over time. Shaded band = cohort interquartile range; " +
    "dashed line = cohort median. Dots link to the surgery."));
  head.append(tw);
  head.append(Object.assign(el("a", "ghost-link", "← all physicians"), {href: "#/physicians"}));
  view.append(head);
  const wrap = el("div");
  view.append(wrap);

  wrap.innerHTML = "<p class='empty'>Loading…</p>";
  const pr = await api(`/api/progress?physician=${encodeURIComponent(name)}`);
  wrap.innerHTML = "";
  const vids = pr.videos;
  if (!vids.length) { wrap.append(el("p", "empty", "No surgeries yet.")); return; }

  /* x positions: by date when >=2 dated surgeries span time, else by index */
  const dates = vids.map(v => Date.parse(v.surgery_date || ""));
  const valid = dates.filter(Number.isFinite);
  const span = valid.length >= 2 ? Math.max(...valid) - Math.min(...valid) : 0;
  const xs = vids.map((v, i) => {
    if (span > 0 && Number.isFinite(dates[i]))
      return (dates[i] - Math.min(...valid)) / span;
    return vids.length > 1 ? i / (vids.length - 1) : 0.5;
  });
  const grid = el("div", "prog-grid");

  const total = el("div", "card prog-card");
  total.innerHTML = `<h3>Total operative time</h3>` +
    sparkSVG(vids.map((v, i) => ({x: xs[i], y: v.duration_s,
      href: `#/video/${encodeURIComponent(v.case)}`,
      label: `${v.case} · ${v.surgery_date || "no date"} · ${fmtT(v.duration_s)}`})), null, "#1c211f") +
    `<div class="axis-note">${esc(vids[0].surgery_date || "")} → ${esc(vids[vids.length - 1].surgery_date || "")}
     · ${vids.length} surgeries</div>`;
  grid.append(total);

  const phaseOrder = Object.keys(PHASE_COLORS).filter(p => p !== "idle");
  for (const p of phaseOrder) {
    const pts = vids.map((v, i) => v.phases[p] ? {x: xs[i], y: v.phases[p].total_s,
      href: `#/video/${encodeURIComponent(v.case)}`,
      label: `${v.case} · ${v.surgery_date || "no date"} · ${v.phases[p].total_s}s` +
        (v.phases[p].percentile !== null ? ` (p${Math.round(v.phases[p].percentile)})` : "")}
      : null).filter(Boolean);
    if (!pts.length) continue;
    const c = el("div", "card prog-card");
    c.innerHTML = `<h3><a href="#/phases/${encodeURIComponent(p)}" class="ph-link">
        <i style="background:${PHASE_COLORS[p]}"></i>${nice(p)}</a></h3>` +
      sparkSVG(pts, pr.cohort[p], PHASE_COLORS[p]) +
      `<div class="axis-note">latest: ${pts[pts.length - 1].y.toFixed(0)}s
       ${vids[vids.length - 1].phases[p] && vids[vids.length - 1].phases[p].percentile !== null
         ? `· p${Math.round(vids[vids.length - 1].phases[p].percentile)} of cohort` : ""}</div>`;
    grid.append(c);
  }
  wrap.append(grid);
}
