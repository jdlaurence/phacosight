/* PhacoSight SPA — no dependencies, hash routing. */
"use strict";

const PHASE_COLORS = {
  idle: "#d9d7d0", "Incision": "#2a78d6", "Viscoelastic": "#86b6ef",
  "Capsulorhexis": "#1baf7a", "Hydrodissection": "#7fd6b3",
  "Phacoemulsification": "#eb6834", "Irrigation/Aspiration": "#f3a97e",
  "Capsule Pulishing": "#eda100", "Lens Implantation": "#4a3aa7",
  "Lens positioning": "#9085e9", "Viscoelastic_Suction": "#e87ba4",
  "Anterior_Chamber Flushing": "#c2569b", "Tonifying/Antibiotics": "#008300",
};
const nice = p => p === "Capsule Pulishing" ? "Capsule Polishing" : p.replace("_", " ");
const fmtT = s => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
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

/* ---------------- router ---------------- */
window.addEventListener("hashchange", route);
window.addEventListener("load", route);
function route() {
  const [_, page, arg] = location.hash.split("/");
  document.querySelectorAll("nav a").forEach(a =>
    a.classList.toggle("active", a.dataset.nav === (page || "videos")));
  tooltip.hidden = true;
  if (page === "video" && arg) return renderDetail(decodeURIComponent(arg));
  if (page === "library") return renderLibrary();
  if (page === "progress") return renderProgress();
  if (page === "upload") return renderUpload();
  return renderList();
}

/* ---------------- surgeries list ---------------- */
let videosCache = null;
async function renderList() {
  view.innerHTML = "<p class='empty'>Loading surgeries…</p>";
  videosCache = videosCache || await api("/api/videos");
  const vids = videosCache;
  const n = vids.length;
  const flagged = vids.filter(v => v.flag_fraction > 0.25).length;
  const uploaded = vids.filter(v => v.source === "uploaded").length;
  const hours = vids.reduce((a, v) => a + v.duration_s, 0) / 3600;

  view.innerHTML = "";
  view.append(el("h1", "", "Surgeries"));
  view.append(el("p", "sub",
    "Every analyzed surgery, with automated quality flags. Click a row to review."));
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
      .map(d => `<option>${d}</option>`).join(""));
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

  const draw = () => {
    const q = search.value.toLowerCase();
    tbody.innerHTML = "";
    const rows = vids.filter(v =>
      v.case.toLowerCase().includes(q) &&
      (!srcSel.value || v.source === srcSel.value) &&
      (!docSel.value || v.physician === docSel.value) &&
      (!flagSel.value || (flagSel.value === "review") === (v.flag_fraction > 0.25)));
    for (const v of rows.slice(0, 400)) {
      const tr = el("tr");
      const status = v.flag_fraction > 0.25
        ? "<span class='chip warn'>review</span>" : "<span class='chip ok'>clean</span>";
      const src = {library: "archive", labeled: "annotated", uploaded: "uploaded"}[v.source];
      tr.innerHTML = `<td><b>${v.case}</b><br><span class='chip neutral'>${src}</span></td>
        <td><div class='minitl' data-case='${v.case}'></div></td>
        <td class='num'>${fmtT(v.duration_s)}</td>
        <td>${v.physician || "<span class='sub'>—</span>"}</td>
        <td class='num'>${v.surgery_date || "<span class='sub'>—</span>"}</td>
        <td class='num'>${(v.flag_fraction * 100).toFixed(0)}%</td><td>${status}</td>`;
      tr.onclick = () => location.hash = `#/video/${v.case}`;
      tbody.append(tr);
    }
    lazyMiniTimelines();
    if (!rows.length) tbody.innerHTML = "<tr><td colspan='7' class='empty'>No matches.</td></tr>";
  };
  [search, srcSel, docSel, flagSel].forEach(x => x.addEventListener("input", draw));
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
        miniCache[c] = miniCache[c] || (async () => (await api(`/api/videos/${c}`)))();
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
async function renderDetail(caseId) {
  view.innerHTML = "<p class='empty'>Loading analysis…</p>";
  const d = await api(`/api/videos/${caseId}`);
  view.innerHTML = "";
  const head = el("div");
  head.append(el("h1", "", caseId));
  const sub = d.metrics
    ? `Automated timeline · ${d.metrics.n_segments} segments · median confidence
       ${d.metrics.median_confidence.toFixed(2)} ${d.ground_truth ? "· expert annotations available" : ""}`
    : "Expert annotations only (not model-analyzed)";
  head.append(el("p", "sub", sub));
  view.append(head);

  const grid = el("div", "detail-grid");
  view.append(grid);

  /* left: player + timelines */
  const left = el("div", "card player-card");
  const vid = document.createElement("video");
  vid.controls = true; vid.preload = "metadata";
  vid.src = `/api/stream/${caseId}`;
  left.append(vid);
  const now = el("div", "now-playing", "&nbsp;");
  left.append(now);
  const tlWrap = el("div", "tl-wrap");
  left.append(tlWrap);
  const segs = d.segments.length ? d.segments : null;
  if (segs) {
    tlWrap.append(el("div", "tl-label", "model timeline — click to jump"));
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
  grid.append(left);

  /* playhead sync */
  vid.addEventListener("timeupdate", () => {
    document.querySelectorAll(".playhead").forEach(ph =>
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
      range (n≈${m.phases[0]?.n_cohort ?? "–"}).`));
    for (const p of m.phases) panel.append(phaseRow(p));
    const idle = el("div", "ph-row");
    idle.innerHTML = `<span class='ph-name'><i style="background:${PHASE_COLORS.idle}"></i>idle
      (between steps)</span><span></span>
      <span class='ph-val num'>${p0(m.idle_s)}<small>±6 s uncertainty</small></span>`;
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
        li.onclick = () => { vid.currentTime = s.start_s; vid.play(); };
        ul.append(li);
      }
      fl.append(ul);
    } else {
      fl.append(el("p", "sub", "None — every segment passed the confidence gate."));
    }
    side.append(fl);

    const prov = el("div", "card panel");
    prov.append(el("h2", "", "Provenance"));
    prov.append(el("p", "sub", `${(d.provenance || {}).stack || ""}<br>
      model ${((d.provenance || {}).git_sha || "").slice(0, 8)} · analyzed at
      ${d.inference_fps || "?"} fps`));
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
    <span class='ph-name'><i style="background:${PHASE_COLORS[p.phase]}"></i>${nice(p.phase)}</span>
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

/* ---------------- library ---------------- */
async function renderLibrary() {
  view.innerHTML = "";
  view.append(el("h1", "", "Phase library"));
  view.append(el("p", "sub",
    "Search every indexed surgery by surgical step. Results are high-confidence segments; " +
    "flagged ones are excluded."));
  const bar = el("div", "toolbar");
  const sel = el("select");
  for (const p of Object.keys(PHASE_COLORS).filter(p => p !== "idle"))
    sel.append(new Option(nice(p), p));
  sel.value = "Incision";
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
      const st = await api(`/api/phase_stats?phase=${encodeURIComponent(sel.value)}`);
      const color = PHASE_COLORS[sel.value];
      hero.innerHTML = `
        <div>
          <h2 style="display:flex;align-items:center">
            <i style="width:11px;height:11px;border-radius:3px;background:${color};
              display:inline-block;margin-right:8px"></i>${nice(sel.value)} — cohort profile</h2>
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
    const res = await api(`/api/search?phase=${encodeURIComponent(sel.value)}&min_conf=${conf.value}`);
    count.textContent = `${res.total} segments found`;
    grid.innerHTML = "";
    for (const h of res.hits) {
      const mid = (h.start_s + h.end_s) / 2;
      const c = el("div", "hit card");
      c.innerHTML = `<img loading="lazy" src="/api/frame?case=${h.case}&t=${mid.toFixed(1)}"
          alt="${h.case}">
        <div class="meta"><a href="#/video/${h.case}">${h.case}</a>
          <span class="num">${fmtT(h.start_s)}–${fmtT(h.end_s)}</span>
          <a href="/api/clip?case=${h.case}&start=${h.start_s}&end=${h.end_s}"
             title="download clip">⬇ clip</a></div>`;
      grid.append(c);
    }
    if (!res.hits.length) grid.innerHTML = "<p class='empty'>No segments at this confidence.</p>";
  };
  sel.addEventListener("input", draw);
  conf.addEventListener("change", draw);
  draw();
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
  const fileF = el("label", "field", "Recording (mp4)");
  fileF.append(input);
  const docF = el("label", "field",
    "Physician <span class='sub' style='margin:0'>(enables the progress view)</span>");
  docF.append(doc);
  const dateF = el("label", "field", "Surgery date");
  dateF.append(date);
  const go = el("button", "", "Upload & analyze");
  const prog = el("div", "progress"); prog.hidden = true;
  prog.innerHTML = `<div class="bar"><i style="width:0%"></i></div><p></p>`;
  box.append(fileF, docF, dateF, go, prog);
  view.append(box);

  go.onclick = async () => {
    if (!input.files.length) return alert("Choose an mp4 first.");
    go.disabled = true; prog.hidden = false;
    const fd = new FormData(); fd.append("file", input.files[0]);
    fd.append("physician", doc.value); fd.append("surgery_date", date.value);
    prog.querySelector("p").textContent = "Uploading…";
    prog.querySelector("i").style.width = "10%";
    const r = await fetch("/api/upload", {method: "POST", body: fd});
    if (!r.ok) { prog.querySelector("p").textContent = "Upload failed: " + await r.text();
      go.disabled = false; return; }
    const {job_id, case: caseId} = await r.json();
    const stages = {queued: 20, loading: 35, features: 60, inference: 85, done: 100};
    const poll = setInterval(async () => {
      const j = await api(`/api/jobs/${job_id}`);
      prog.querySelector("i").style.width = `${stages[j.status] ?? 15}%`;
      prog.querySelector("p").textContent = `${j.status} — ${j.detail}`;
      if (j.status === "done") {
        clearInterval(poll); videosCache = null;
        location.hash = `#/video/${caseId}`;
      }
      if (j.status === "error") { clearInterval(poll); go.disabled = false; }
    }, 1500);
  };
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
  /* points: [{x: 0-1, y, label}], cohort: {p25,p50,p75} in y units */
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
  const dots = points.map(p =>
    `<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="1.8" fill="${color}">
       <title>${p.label}</title></circle>`).join("");
  return `<svg class="spark" viewBox="0 0 100 100" preserveAspectRatio="none">
      ${band}
      <line x1="6" y1="92" x2="96" y2="92" stroke="#d0d5d1" stroke-width="0.5"/>
      ${points.length > 1 ? `<path d="${path}" fill="none" stroke="${color}" stroke-width="1.1"/>` : ""}
      ${dots}
      <text x="6" y="8" font-size="6" fill="#8a948f">${Math.round(yMax)}s</text>
    </svg>`;
}

/* ---------------- progress ---------------- */
async function renderProgress() {
  view.innerHTML = "";
  view.append(el("h1", "", "My progress"));
  view.append(el("p", "sub",
    "Your uploaded surgeries over time. Shaded band = cohort interquartile range; " +
    "dashed line = cohort median."));
  const docs = await api("/api/physicians");
  if (!docs.length) {
    view.append(el("p", "empty",
      "No surgeries with a physician name yet. Upload a video with your name filled in " +
      "and it will appear here."));
    return;
  }
  const bar = el("div", "toolbar");
  const sel = el("select");
  for (const d of docs) sel.append(new Option(`${d.name} (${d.n_videos})`, d.name));
  bar.append(sel);
  view.append(bar);
  const wrap = el("div");
  view.append(wrap);

  const draw = async () => {
    wrap.innerHTML = "<p class='empty'>Loading…</p>";
    const pr = await api(`/api/progress?physician=${encodeURIComponent(sel.value)}`);
    wrap.innerHTML = "";
    const vids = pr.videos;
    if (!vids.length) { wrap.append(el("p", "empty", "No surgeries yet.")); return; }
    const xs = vids.map((v, i) => vids.length > 1 ? i / (vids.length - 1) : 0.5);
    const grid = el("div", "prog-grid");

    const total = el("div", "card prog-card");
    total.innerHTML = `<h3>Total operative time</h3>` +
      sparkSVG(vids.map((v, i) => ({x: xs[i], y: v.duration_s,
        label: `${v.case} · ${v.surgery_date || "no date"} · ${fmtT(v.duration_s)}`})), null, "#1c211f") +
      `<div class="axis-note">${vids[0].surgery_date || ""} → ${vids[vids.length - 1].surgery_date || ""}
       · ${vids.length} surgeries</div>`;
    grid.append(total);

    const phaseOrder = Object.keys(PHASE_COLORS).filter(p => p !== "idle");
    for (const p of phaseOrder) {
      const pts = vids.map((v, i) => v.phases[p] ? {x: xs[i], y: v.phases[p].total_s,
        label: `${v.case} · ${v.surgery_date || "no date"} · ${v.phases[p].total_s}s` +
          (v.phases[p].percentile !== null ? ` (p${Math.round(v.phases[p].percentile)})` : "")}
        : null).filter(Boolean);
      if (!pts.length) continue;
      const c = el("div", "card prog-card");
      c.innerHTML = `<h3><i style="background:${PHASE_COLORS[p]}"></i>${nice(p)}</h3>` +
        sparkSVG(pts, pr.cohort[p], PHASE_COLORS[p]) +
        `<div class="axis-note">latest: ${pts[pts.length - 1].y.toFixed(0)}s
         ${vids[vids.length - 1].phases[p] && vids[vids.length - 1].phases[p].percentile !== null
           ? `· p${Math.round(vids[vids.length - 1].phases[p].percentile)} of cohort` : ""}</div>`;
      grid.append(c);
    }
    wrap.append(grid);
  };
  sel.addEventListener("input", draw);
  draw();
}
