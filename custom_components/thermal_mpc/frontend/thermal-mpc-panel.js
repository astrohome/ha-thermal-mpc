// Thermal model panel: heat-flow diagram, per-room heat budgets and
// model-vs-measured charts. Plain web component, no build step.

const PALETTE = {
  // Validated pairs (see README): measured/model = categorical slots 1-2,
  // gain/loss = diverging red/blue with gray midpoint.
  light: {
    measured: "#2a78d6", model: "#eb6834", gain: "#e34948", loss: "#2a78d6",
    mid: "#f0efec", flow: "#eb6834", grid: "#e1e0d9", axis: "#c3c2b7",
  },
  dark: {
    measured: "#3987e5", model: "#d95926", gain: "#e66767", loss: "#3987e5",
    mid: "#383835", flow: "#d95926", grid: "#2c2c2a", axis: "#383835",
  },
};

const REFRESH_MS = 60_000;
const BOX_W = 210;
const BOX_H = 128;
const GAP_X = 130;
const GAP_Y = 150;
const BAND_H = 44;
const PAD = 24;

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "–" : v.toFixed(d));
const signed = (v, d = 2) => (v == null ? "–" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`);
const clip = (s, n) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);

function hexToRgb(h) {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function mix(a, b, t) {
  const x = hexToRgb(a);
  const y = hexToRgb(b);
  const c = x.map((v, i) => Math.round(v + (y[i] - v) * t));
  return `rgb(${c.join(",")})`;
}
/** Diverging blue - gray - red for a deviation, saturating at +/- span. */
function diverging(pal, dev, span) {
  if (dev == null) return pal.mid;
  const t = Math.max(-1, Math.min(1, dev / span));
  return t >= 0 ? mix(pal.mid, pal.gain, t) : mix(pal.mid, pal.loss, -t);
}
function relTime(iso) {
  if (!iso) return "never";
  const mins = Math.round((Date.now() - Date.parse(iso)) / 60000);
  if (mins < 60) return `${mins} min ago`;
  const h = Math.round(mins / 60);
  return h < 48 ? `${h} h ago` : `${Math.round(h / 24)} d ago`;
}
/** Horizontal bar path with only the data end rounded. */
function barPath(x0, x1, y, h, r = 4) {
  const w = Math.abs(x1 - x0);
  if (w < 0.5) return "";
  const rr = Math.min(r, w, h / 2);
  if (x1 >= x0) {
    return `M${x0},${y}H${x1 - rr}Q${x1},${y} ${x1},${y + rr}V${y + h - rr}` +
      `Q${x1},${y + h} ${x1 - rr},${y + h}H${x0}Z`;
  }
  return `M${x0},${y}H${x1 + rr}Q${x1},${y} ${x1},${y + rr}V${y + h - rr}` +
    `Q${x1},${y + h} ${x1 + rr},${y + h}H${x0}Z`;
}

class ThermalMpcPanel extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._mode = "now";
    this._entryIdx = 0;
    this._charts = new Map();
  }

  set hass(hass) {
    const first = !this._hass;
    this._hass = hass;
    if (first) this._load();
    const menu = this.shadowRoot.querySelector("ha-menu-button");
    if (menu) menu.hass = hass;
  }

  set narrow(narrow) {
    const changed = this._narrow !== narrow;
    this._narrow = narrow;
    if (changed && this._data) this._render();
  }

  set panel(_p) {}

  connectedCallback() {
    this._timer = setInterval(() => this._load(), REFRESH_MS);
    if (this._hass && !this._data) this._load();
  }

  disconnectedCallback() {
    clearInterval(this._timer);
  }

  get _pal() {
    return this._hass?.themes?.darkMode ? PALETTE.dark : PALETTE.light;
  }

  async _load() {
    if (!this._hass) return;
    try {
      const res = await this._hass.callWS({ type: "thermal_mpc/overview" });
      this._data = res.entries;
      this._error = null;
    } catch (err) {
      this._error = err.message || String(err);
    }
    this._render();
  }

  async _retrain(entryId) {
    this._busy = true;
    this._render();
    try {
      await this._hass.callWS({ type: "thermal_mpc/retrain", entry_id: entryId });
    } catch (err) {
      this._error = err.message || String(err);
    }
    this._busy = false;
    await this._load();
  }

  // ------------------------------------------------------------------ render

  _render() {
    const entries = this._data || [];
    const entry = entries[Math.min(this._entryIdx, entries.length - 1)];
    this._charts.clear();
    this.shadowRoot.innerHTML = `
      <style>${STYLE}</style>
      <div class="toolbar">
        <ha-menu-button></ha-menu-button>
        <div class="title">Thermal model</div>
        ${entries.length > 1 ? `<select class="entry">${entries.map((e, i) =>
          `<option value="${i}" ${i === this._entryIdx ? "selected" : ""}>${esc(e.title)}</option>`).join("")}</select>` : ""}
      </div>
      <div class="content">
        ${this._error ? `<div class="banner error">${esc(this._error)}</div>` : ""}
        ${!this._data ? `<div class="empty">Loading…</div>` : ""}
        ${this._data && !entry ? `<div class="empty">No Thermal MPC entry is loaded. Add the integration under Settings → Devices &amp; services.</div>` : ""}
        ${entry ? this._entryHtml(entry) : ""}
      </div>
      <div class="tooltip" hidden></div>`;
    const menu = this.shadowRoot.querySelector("ha-menu-button");
    if (menu) {
      menu.hass = this._hass;
      menu.narrow = this._narrow;
    }
    this._bind(entry);
  }

  _entryHtml(e) {
    const trained = e.status === "trained";
    const worst = Math.max(...e.rooms.map((r) => r.validation?.["6h"] ?? -1));
    const statusLabel = { trained: "Trained", collecting: "Collecting data", error: "Error" }[e.status];
    const statusIcon = { trained: "✓", collecting: "…", error: "!" }[e.status];
    return `
      <section class="head">
        <div class="status ${e.status}"><span class="icon" aria-hidden="true">${statusIcon}</span>${statusLabel}</div>
        <div class="tiles">
          <div class="tile"><div class="k">Training data</div>
            <div class="v">${fmt(e.training_days, 1)} d</div>
            ${trained ? "" : `<div class="s">first fit at ${fmt(e.min_fit_days, 0)} d</div>`}</div>
          <div class="tile"><div class="k">Prediction error, 6 h ahead</div>
            <div class="v">${worst >= 0 ? `${fmt(worst)} K` : "–"}</div>
            <div class="s">worst room · aim &lt; 0.3 K</div></div>
          <div class="tile"><div class="k">Last fit</div>
            <div class="v">${relTime(e.last_fit)}</div>
            <div class="s"><button class="retrain" ${this._busy ? "disabled" : ""}>${this._busy ? "Retraining…" : "Retrain now"}</button></div></div>
        </div>
        ${e.message ? `<div class="banner">${esc(e.message)}</div>` : ""}
      </section>

      <section class="card">
        <div class="card-head">
          <h2>Where the heat goes</h2>
          <div class="seg" role="tablist">
            <button data-mode="now" class="${this._mode === "now" ? "on" : ""}">Now</button>
            <button data-mode="24h" class="${this._mode === "24h" ? "on" : ""}">Last 24 h average</button>
          </div>
        </div>
        ${this._diagram(e)}
        <p class="note">Arrows point the way heat moves. Thickness and labels give how fast that
        path alone changes the room's temperature, in K/h (kelvin per hour).
        The stripe on each room shows how far it is from the house average
        (<span class="sw" style="background:${this._pal.loss}"></span> cooler,
        <span class="sw" style="background:${this._pal.gain}"></span> warmer).</p>
      </section>

      ${trained ? `<section class="grid">${e.rooms.map((r) => this._roomCard(e, r)).join("")}</section>` : `
        <section class="card"><p class="note">Heat budgets and model checks appear after the first fit,
        once ${fmt(e.min_fit_days, 0)} days of complete data are collected.</p></section>`}`;
  }

  _budget(e, roomId) {
    const src = this._mode === "now" ? e.budget_now : e.budget_24h;
    return src?.[roomId];
  }

  // ------------------------------------------------------------ flow diagram

  _diagram(e) {
    const pal = this._pal;
    const rooms = e.rooms;
    const n = rooms.length;
    const maxCols = this._narrow ? 1 : 4;
    const cols = Math.max(1, Math.ceil(n / Math.ceil(n / maxCols)));
    const rows = Math.ceil(n / cols);
    // A single column needs room on the right for arcs between distant rooms.
    const arcRoom = cols === 1 && n > 2 ? 90 : 0;
    const width = PAD * 2 + cols * BOX_W + (cols - 1) * GAP_X + arcRoom;
    const top = PAD + BAND_H + 70;
    const height = top + rows * BOX_H + (rows - 1) * GAP_Y + (cols > 2 ? 80 : 30);
    const pos = {};
    rooms.forEach((r, i) => {
      const c = i % cols;
      const row = Math.floor(i / cols);
      pos[r.id] = { x: PAD + c * (BOX_W + GAP_X), y: top + row * (BOX_H + GAP_Y), row, col: c };
    });
    const temps = rooms.map((r) => r.temperature).filter((t) => t != null);
    const mean = temps.length ? temps.reduce((a, b) => a + b, 0) / temps.length : null;

    // Magnitudes for scaling: every outdoor and room-room term in view.
    const mags = [];
    for (const r of rooms) {
      const b = this._budget(e, r.id);
      if (!b) continue;
      if (b.outdoor != null) mags.push(Math.abs(b.outdoor));
      for (const v of Object.values(b.rooms || {})) if (v != null) mags.push(Math.abs(v));
    }
    const maxMag = Math.max(0.05, ...mags);
    const widthFor = (v) => 1.5 + 9 * Math.min(1, Math.abs(v) / maxMag);
    const speedFor = (v) => `${Math.max(0.6, 4 - 3.4 * Math.min(1, Math.abs(v) / maxMag))}s`;

    let links = "";
    const labels = []; // [x, y, text, anchor], placed after collision nudging
    // Room <-> outdoor.
    for (const r of rooms) {
      const b = this._budget(e, r.id);
      const v = b?.outdoor;
      const p = pos[r.id];
      const cx = p.row === 0 ? p.x + BOX_W / 2 : p.x + 40;
      const y1 = p.y;
      const y0 = p.row === 0 ? PAD + BAND_H : p.y - 58;
      if (p.row > 0) {
        labels.push([cx, y0 - 6, "Outdoor", "middle", "mini"]);
      }
      if (v == null || Math.abs(v) < 1e-4) {
        links += `<line x1="${cx}" y1="${y0}" x2="${cx}" y2="${y1}" class="nolink"/>`;
        continue;
      }
      // Negative = the room is losing heat to outdoors: arrow points up.
      const [ya, yb] = v < 0 ? [y1, y0] : [y0, y1];
      links += this._arrow(`M${cx},${ya}L${cx},${yb}`, v, widthFor(v), speedFor(v), pal,
        `${r.name} ↔ outdoor: ${signed(v, 3)} K/h for the room`);
      labels.push([cx + 10, (y0 + y1) / 2 + 4, `${fmt(Math.abs(v))} K/h`, "start", "val"]);
    }
    // Room <-> room, drawn once per pair from the warmer (sending) side.
    const done = new Set();
    const byId = Object.fromEntries(rooms.map((x) => [x.id, x]));
    for (const r of rooms) {
      const b = this._budget(e, r.id);
      for (const other of Object.keys(b?.rooms || {})) {
        const key = [r.id, other].sort().join("|");
        if (done.has(key) || !pos[other]) continue;
        done.add(key);
        // Each side has its own fitted coupling; either may be zero.
        const coupled = other in (r.coupling_h || {}) ||
          r.id in (byId[other]?.coupling_h || {});
        const vIn = b.rooms[other]; // > 0: r is warmed by other
        const vOut = this._budget(e, other)?.rooms?.[r.id]; // > 0: other warmed by r
        if (!coupled || (vIn == null && vOut == null)) continue;
        const sign = Math.abs(vIn ?? 0) >= Math.abs(vOut ?? 0) ? (vIn ?? 0) : -(vOut ?? 0);
        const recv = Math.max(Math.abs(vIn ?? 0), Math.abs(vOut ?? 0));
        if (recv < 1e-4) continue;
        const [from, to] = sign > 0 ? [other, r.id] : [r.id, other];
        const a = pos[from];
        const c = pos[to];
        let d;
        let lx;
        let ly;
        if (a.row === c.row && Math.abs(a.col - c.col) === 1) {
          const y = a.y + BOX_H / 2;
          const xa = a.col < c.col ? a.x + BOX_W : a.x;
          const xc = a.col < c.col ? c.x : c.x + BOX_W;
          d = `M${xa},${y}L${xc},${y}`;
          lx = (xa + xc) / 2;
          ly = y - 12;
        } else if (a.row === c.row) {
          // Same row, not neighbours: arc through the gap below the row.
          const xa = a.x + BOX_W / 2;
          const xc = c.x + BOX_W / 2;
          const yb = a.y + BOX_H;
          const dip = yb + 36 + Math.abs(a.col - c.col) * 14;
          d = `M${xa},${yb}Q${(xa + xc) / 2},${dip * 2 - yb - 36} ${xc},${yb}`;
          lx = (xa + xc) / 2;
          ly = dip + 4;
        } else if (a.col === c.col && Math.abs(a.row - c.row) > 1) {
          // Same column, rooms in between: arc out to the right.
          const xe = a.x + BOX_W;
          const ya = a.y + BOX_H / 2 + 14;
          const yc = c.y + BOX_H / 2 + 14;
          const bulge = xe + 50 + 12 * Math.abs(a.row - c.row);
          d = `M${xe},${ya}C${bulge},${ya} ${bulge},${yc} ${xe},${yc}`;
          lx = xe + 8;
          ly = (ya + yc) / 2 + 4;
        } else {
          // Different rows: straight from the upper box's bottom to the lower's top.
          const [up, lo] = a.row < c.row ? [a, c] : [c, a];
          const sameCol = up.col === lo.col;
          const xu = sameCol ? up.x + BOX_W - 40 : up.x + BOX_W / 2;
          const xl = sameCol ? lo.x + BOX_W - 40 : lo.x + BOX_W / 2;
          const yu = up.y + BOX_H;
          const yl = lo.y;
          d = up === a ? `M${xu},${yu}L${xl},${yl}` : `M${xl},${yl}L${xu},${yu}`;
          // Label near the upper end, clear of the lower rows' outdoor stubs.
          lx = xu + (xl - xu) * 0.3 + 8;
          ly = yu + (yl - yu) * 0.3 + 4;
        }
        const nameFrom = rooms.find((x) => x.id === from)?.name;
        const nameTo = rooms.find((x) => x.id === to)?.name;
        links += this._arrow(d, recv, widthFor(recv), speedFor(recv), pal,
          `Heat flows ${nameFrom} → ${nameTo}: ${fmt(recv, 3)} K/h`);
        labels.push([lx, ly, `${fmt(recv)} K/h`, a.row === c.row ? "middle" : "start", "val"]);
      }
    }

    // Nudge labels down until they no longer overlap one already placed.
    const placed = [];
    for (const [x, y0, text, anchor, cls] of labels) {
      const w = text.length * 6.6;
      const left = anchor === "middle" ? x - w / 2 : x;
      let y = y0;
      for (let k = 0; k < 8; k++) {
        const hit = placed.some((b) => left < b.r && left + w > b.l && y - 12 < b.b && y > b.t);
        if (!hit) break;
        y += 15;
      }
      placed.push({ l: left, r: left + w, t: y - 12, b: y });
      links += `<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}">${esc(text)}</text>`;
    }

    const boxes = rooms.map((r) => {
      const p = pos[r.id];
      const b = this._budget(e, r.id);
      const dev = r.temperature != null && mean != null ? r.temperature - mean : null;
      const inputs = Object.entries(b?.inputs || {})
        .filter(([, v]) => v != null && Math.abs(v) >= 0.005)
        .sort((x, y) => Math.abs(y[1]) - Math.abs(x[1]))
        .slice(0, 2);
      const tau = r.tau_out_h != null ? `τ ${fmt(r.tau_out_h, 0)} h` : "τ –";
      const net = b?.net;
      return `
        <g class="room" data-tip="${esc(`${r.name}: ${fmt(r.temperature, 1)} °C (${dev == null ? "–" : `${signed(dev, 1)} K vs average`}) · net ${signed(net, 3)} K/h`)}">
          <rect x="${p.x}" y="${p.y}" width="${BOX_W}" height="${BOX_H}" rx="12" class="box"/>
          <path d="M${p.x},${p.y + 12}Q${p.x},${p.y} ${p.x + 12},${p.y}H${p.x + BOX_W - 12}Q${p.x + BOX_W},${p.y} ${p.x + BOX_W},${p.y + 12}V${p.y + 8}H${p.x}Z"
            fill="${diverging(pal, dev, 2)}"/>
          <text class="name" x="${p.x + 14}" y="${p.y + 32}">${esc(clip(r.name, 26))}</text>
          <text class="temp" x="${p.x + 14}" y="${p.y + 64}">${fmt(r.temperature, 1)} °C</text>
          <text class="sub" x="${p.x + 14}" y="${p.y + 86}">${tau} · net ${signed(net)} K/h</text>
          ${inputs.map(([k, v], i) => `
            <circle cx="${p.x + 18}" cy="${p.y + 104 + i * 16 - 4}" r="4" fill="${v >= 0 ? pal.gain : pal.loss}"/>
            <text class="sub" x="${p.x + 28}" y="${p.y + 104 + i * 16}">${esc(e.labels[k] || k)} ${signed(v)} K/h</text>`).join("")}
        </g>`;
    }).join("");

    return `
      <div class="diagram">
        <svg viewBox="0 0 ${width} ${height}" style="max-width:${width}px" role="img" aria-label="Heat flow between rooms and outdoors">
          <defs>
            <marker id="arrowhead" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
              <path d="M0,0L10,5L0,10Z" fill="${pal.flow}"/>
            </marker>
          </defs>
          <rect x="${PAD}" y="${PAD}" width="${width - 2 * PAD}" height="${BAND_H}" rx="10" class="band"/>
          <text class="name" x="${PAD + 14}" y="${PAD + 28}">Outdoor · ${fmt(e.outdoor?.temperature, 1)} °C</text>
          ${links}
          ${boxes}
        </svg>
      </div>`;
  }

  _arrow(d, v, w, speed, pal, tip) {
    return `
      <path d="${d}" class="hit" data-tip="${esc(tip)}"/>
      <path d="${d}" fill="none" stroke="${pal.flow}" stroke-opacity="0.35" stroke-width="${w}" stroke-linecap="round"/>
      <path d="${d}" fill="none" stroke="${pal.flow}" stroke-width="${Math.max(1.5, w * 0.45)}"
        stroke-linecap="round" class="flow" style="animation-duration:${speed}" marker-end="url(#arrowhead)"/>`;
  }

  // -------------------------------------------------------------- room cards

  _roomCard(e, r) {
    const pal = this._pal;
    const b = this._budget(e, r.id);
    const rows = [];
    let hidden = 0;
    const add = (label, v) => {
      if (v == null || Math.abs(v) >= 0.005) rows.push([label, v]);
      else hidden += 1;
    };
    if (b) {
      rows.push(["Outdoor", b.outdoor]);
      for (const [k, v] of Object.entries(b.rooms || {})) add(e.labels[k] || k, v);
      for (const [k, v] of Object.entries(b.inputs || {})) add(e.labels[k] || k, v);
      add("Baseline (unexplained)", b.offset);
    }
    const src = this._mode === "now" ? e.budget_now : e.budget_24h;
    const allMags = Object.values(src || {}).flatMap((x) => [
      x.outdoor, x.offset, x.net, ...Object.values(x.rooms || {}), ...Object.values(x.inputs || {}),
    ]).filter((v) => v != null).map(Math.abs);
    const scale = Math.max(0.05, ...allMags);

    const W = 460;
    const labelW = 150;
    const valueW = 64;
    const barW = W - labelW - valueW;
    const zero = labelW + barW / 2;
    const rowH = 24;
    const all = [...rows, ["Net change", b?.net]];
    const H = all.length * rowH + 8;
    const bars = all.map(([label, v], i) => {
      const y = 4 + i * rowH;
      const isNet = i === all.length - 1;
      const x1 = v == null ? zero : zero + (v / scale) * (barW / 2 - 4);
      return `
        <g class="brow" data-tip="${esc(`${label}: ${signed(v, 3)} K/h`)}">
          <rect x="0" y="${y}" width="${W}" height="${rowH}" class="hitrow"/>
          ${isNet ? `<line x1="0" x2="${W}" y1="${y}" y2="${y}" class="sep"/>` : ""}
          <text class="lbl ${isNet ? "strong" : ""}" x="0" y="${y + 16}">${esc(clip(label, 22))}</text>
          <path d="${v == null ? "" : barPath(zero, x1, y + 5, rowH - 10)}" fill="${v >= 0 ? pal.gain : pal.loss}"/>
          <text class="num ${isNet ? "strong" : ""}" x="${W}" y="${y + 16}" text-anchor="end">${signed(v)}</text>
        </g>`;
    }).join("");

    const v = r.validation || {};
    return `
      <div class="card room-card">
        <div class="card-head">
          <h2>${esc(r.name)}</h2>
          <div class="muted">${fmt(r.temperature, 1)} °C</div>
        </div>
        <div class="legend">
          <span><span class="sw" style="background:${pal.loss}"></span>cools the room</span>
          <span><span class="sw" style="background:${pal.gain}"></span>warms the room</span>
          <span class="muted">K/h · ${this._mode === "now" ? "right now" : "24 h average"}</span>
        </div>
        ${b ? `<svg class="budget" viewBox="0 0 ${W} ${H}" role="img" aria-label="Heat budget for ${esc(r.name)}">
          <line x1="${zero}" x2="${zero}" y1="0" y2="${H}" class="zero"/>${bars}</svg>`
          : `<p class="note">No budget: some inputs are unavailable right now.</p>`}
        ${hidden ? `<div class="muted small">${hidden} term${hidden > 1 ? "s" : ""} below 0.005 K/h not shown</div>` : ""}
        <dl class="facts">
          <div><dt>Time constant to outdoor</dt><dd>${r.tau_out_h != null ? `${fmt(r.tau_out_h, 1)} h` : "not identified yet"}</dd></div>
          ${Object.entries(r.coupling_h || {}).map(([k, h]) => `<div><dt>Coupling to ${esc(e.labels[k] || k)}</dt><dd>${fmt(h, 1)} h</dd></div>`).join("")}
          <div><dt>Prediction error 1 / 3 / 6 h</dt><dd>${fmt(v["1h"])} / ${fmt(v["3h"])} / ${fmt(v["6h"])} K</dd></div>
        </dl>
        ${this._replayChart(e, r)}
      </div>`;
  }

  // ------------------------------------------------------------ replay chart

  _replayChart(e, r) {
    const rep = e.replay;
    const meas = rep?.measured?.[r.id];
    const pred = rep?.predicted?.[r.id];
    if (!rep?.times?.length || !meas) return "";
    const pal = this._pal;
    const id = `chart-${this._charts.size}`;
    const W = 460;
    const H = 170;
    const L = 36;
    const R = 8;
    const T = 10;
    const B = 24;
    const t = rep.times;
    const vals = [...meas, ...pred].filter((x) => x != null);
    if (!vals.length) return "";
    let lo = Math.min(...vals);
    let hi = Math.max(...vals);
    const padY = Math.max(0.2, (hi - lo) * 0.08);
    const stepY = [0.1, 0.2, 0.25, 0.5, 1, 2, 5].find((st) => st >= (hi - lo + 2 * padY) / 4) ?? 10;
    lo = Math.floor((lo - padY) / stepY) * stepY;
    hi = Math.ceil((hi + padY) / stepY) * stepY;
    const x = (i) => L + ((t[i] - t[0]) / (t[t.length - 1] - t[0] || 1)) * (W - L - R);
    const y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);

    const line = (arr, breakEvery) => {
      let d = "";
      let pen = false;
      arr.forEach((v, i) => {
        const restart = breakEvery && i > 0 && i % breakEvery === 0;
        if (v == null) { pen = false; return; }
        d += `${pen && !restart ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
        pen = true;
      });
      return d;
    };
    const seg = Math.round((rep.segment_hours * 3600) / ((t[1] - t[0]) || 300));
    const yTicks = [];
    for (let v = lo; v <= hi + 1e-9; v += stepY) yTicks.push(v);
    const xTicks = [];
    const first = Math.ceil(t[0] / 43200) * 43200;
    for (let s = first; s <= t[t.length - 1]; s += 43200) xTicks.push(s);
    const fmtT = (s) => new Date(s * 1000).toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit" });

    this._charts.set(id, { t, meas, pred, x, L, W, R, fmtT, name: r.name });
    return `
      <div class="chart-head">
        <h3>Model check · last 48 h</h3>
        <div class="legend">
          <span><span class="sw line" style="background:${pal.measured}"></span>Measured</span>
          <span><span class="sw line" style="background:${pal.model}"></span>Model (restarted every ${fmt(rep.segment_hours, 0)} h)</span>
        </div>
      </div>
      <svg class="replay" id="${id}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Measured versus modelled temperature for ${esc(r.name)}">
        ${yTicks.map((v) => `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" class="gridline"/>
          <text class="tick" x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(stepY < 1 ? 1 : 0)}</text>`).join("")}
        ${xTicks.map((s) => {
          const xi = L + ((s - t[0]) / (t[t.length - 1] - t[0])) * (W - L - R);
          return `<text class="tick" x="${xi}" y="${H - 6}" text-anchor="middle">${esc(fmtT(s))}</text>`;
        }).join("")}
        <path d="${line(meas)}" fill="none" stroke="${pal.measured}" stroke-width="2" stroke-linejoin="round"/>
        <path d="${line(pred, seg)}" fill="none" stroke="${pal.model}" stroke-width="2" stroke-linejoin="round"/>
        <line class="cross" x1="0" x2="0" y1="${T}" y2="${H - B}" visibility="hidden"/>
        <rect class="hover" x="${L}" y="0" width="${W - L - R}" height="${H}" fill="transparent"/>
      </svg>`;
  }

  // ------------------------------------------------------------------ events

  _bind(entry) {
    const root = this.shadowRoot;
    root.querySelector("select.entry")?.addEventListener("change", (ev) => {
      this._entryIdx = Number(ev.target.value);
      this._render();
    });
    root.querySelectorAll(".seg button").forEach((btn) =>
      btn.addEventListener("click", () => {
        this._mode = btn.dataset.mode;
        this._render();
      }));
    root.querySelector("button.retrain")?.addEventListener("click", () => {
      if (entry) this._retrain(entry.entry_id);
    });

    const tip = root.querySelector(".tooltip");
    const show = (ev, html) => {
      tip.innerHTML = html;
      tip.hidden = false;
      const box = this.getBoundingClientRect();
      let left = ev.clientX - box.left + 14;
      const top = ev.clientY - box.top + 14;
      if (left + 240 > box.width) left = ev.clientX - box.left - 250;
      tip.style.left = `${left}px`;
      tip.style.top = `${top}px`;
    };
    const hide = () => { tip.hidden = true; };
    root.querySelectorAll("[data-tip]").forEach((el) => {
      el.addEventListener("mousemove", (ev) => show(ev, esc(el.dataset.tip)));
      el.addEventListener("mouseleave", hide);
    });
    const pal = this._pal;
    for (const [id, c] of this._charts) {
      const svg = root.getElementById(id);
      if (!svg) continue;
      const cross = svg.querySelector(".cross");
      const hover = svg.querySelector(".hover");
      hover.addEventListener("mousemove", (ev) => {
        const pt = svg.createSVGPoint();
        pt.x = ev.clientX;
        pt.y = ev.clientY;
        const p = pt.matrixTransform(svg.getScreenCTM().inverse());
        const frac = (p.x - c.L) / (c.W - c.L - c.R);
        const i = Math.max(0, Math.min(c.t.length - 1, Math.round(frac * (c.t.length - 1))));
        cross.setAttribute("x1", c.x(i));
        cross.setAttribute("x2", c.x(i));
        cross.setAttribute("visibility", "visible");
        show(ev, `<div class="tt-h">${esc(c.fmtT(c.t[i]))}</div>
          <div><span class="sw line" style="background:${pal.measured}"></span>Measured <b>${fmt(c.meas[i], 2)} °C</b></div>
          <div><span class="sw line" style="background:${pal.model}"></span>Model <b>${fmt(c.pred[i], 2)} °C</b></div>`);
      });
      hover.addEventListener("mouseleave", () => {
        cross.setAttribute("visibility", "hidden");
        hide();
      });
    }
  }
}

const STYLE = `
  :host {
    display: block; position: relative; min-height: 100vh;
    background: var(--primary-background-color);
    color: var(--primary-text-color);
    font-family: var(--paper-font-body1_-_font-family, system-ui, -apple-system, "Segoe UI", sans-serif);
  }
  .toolbar {
    display: flex; align-items: center; gap: 8px; height: 56px; padding: 0 12px;
    background: var(--app-header-background-color, var(--primary-color));
    color: var(--app-header-text-color, var(--text-primary-color));
    position: sticky; top: 0; z-index: 2;
  }
  .toolbar .title { font-size: 20px; flex: 1; }
  .toolbar select { font: inherit; padding: 4px 8px; border-radius: 6px; }
  .content { padding: 16px; max-width: 1400px; margin: 0 auto; box-sizing: border-box; }
  .head { display: flex; flex-direction: column; gap: 12px; margin-bottom: 16px; }
  .status { display: inline-flex; align-items: center; gap: 8px; font-weight: 500; }
  .status .icon {
    display: inline-grid; place-items: center; width: 20px; height: 20px; border-radius: 50%;
    font-size: 12px; color: #fff; background: var(--secondary-text-color);
  }
  .status.trained .icon { background: #0ca30c; }
  .status.error .icon { background: #d03b3b; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
  .tile, .card {
    background: var(--card-background-color); border-radius: var(--ha-card-border-radius, 12px);
    border: 1px solid var(--divider-color); padding: 14px 16px; box-sizing: border-box;
  }
  .tile .k { color: var(--secondary-text-color); font-size: 13px; }
  .tile .v { font-size: 26px; font-weight: 500; margin-top: 4px; }
  .tile .s { color: var(--secondary-text-color); font-size: 12px; margin-top: 4px; }
  .card { margin-bottom: 16px; min-width: 0; }
  .card-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
  h2 { font-size: 18px; font-weight: 500; margin: 0; }
  h3 { font-size: 14px; font-weight: 500; margin: 0; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
  .grid .card { margin-bottom: 0; }
  button {
    font: inherit; font-size: 13px; cursor: pointer; border-radius: 8px; padding: 6px 12px;
    border: 1px solid var(--divider-color); background: transparent; color: var(--primary-text-color);
  }
  button:disabled { opacity: .6; cursor: default; }
  .seg { display: inline-flex; border: 1px solid var(--divider-color); border-radius: 8px; overflow: hidden; }
  .seg button { border: 0; border-radius: 0; }
  .seg button.on { background: var(--primary-color); color: var(--text-primary-color, #fff); }
  .diagram { overflow-x: auto; margin-top: 8px; }
  .diagram svg { width: 100%; min-width: 280px; height: auto; display: block; margin: 0 auto; }
  .band { fill: var(--secondary-background-color, rgba(127,127,127,.12)); }
  .box { fill: var(--card-background-color); stroke: var(--divider-color); }
  .room:hover .box { stroke: var(--primary-text-color); }
  text { fill: var(--primary-text-color); }
  .name { font-size: 15px; font-weight: 500; }
  .temp { font-size: 26px; font-weight: 500; }
  .sub, .mini, .val { font-size: 12px; fill: var(--secondary-text-color); }
  .val { font-variant-numeric: tabular-nums; }
  .nolink { stroke: var(--divider-color); stroke-dasharray: 2 4; }
  .hit { fill: none; stroke: transparent; stroke-width: 18; cursor: default; }
  .flow { stroke-dasharray: 6 8; animation: dash linear infinite; }
  @keyframes dash { to { stroke-dashoffset: -28; } }
  @media (prefers-reduced-motion: reduce) { .flow { animation: none; } }
  .note { color: var(--secondary-text-color); font-size: 13px; line-height: 1.5; margin: 10px 0 0; }
  .muted { color: var(--secondary-text-color); font-size: 13px; }
  .small { font-size: 12px; }
  .legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: var(--secondary-text-color); margin: 10px 0 6px; align-items: center; }
  .sw { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
  .sw.line { height: 3px; width: 14px; border-radius: 2px; vertical-align: 3px; }
  .budget, .replay { width: 100%; height: auto; display: block; }
  .zero { stroke: var(--secondary-text-color); stroke-opacity: .5; }
  .sep { stroke: var(--divider-color); }
  .lbl, .num { font-size: 13px; }
  .num { font-variant-numeric: tabular-nums; }
  .strong { font-weight: 600; }
  .hitrow { fill: transparent; }
  .brow:hover .hitrow { fill: var(--secondary-background-color, rgba(127,127,127,.08)); }
  .facts { display: grid; grid-template-columns: 1fr; gap: 4px; margin: 12px 0; font-size: 13px; }
  .facts div { display: flex; justify-content: space-between; gap: 12px; }
  .facts dt { color: var(--secondary-text-color); }
  .facts dd { margin: 0; font-variant-numeric: tabular-nums; }
  .chart-head { margin-top: 8px; }
  .gridline { stroke: var(--divider-color); stroke-width: 1; }
  .tick { font-size: 11px; fill: var(--secondary-text-color); font-variant-numeric: tabular-nums; }
  .cross { stroke: var(--secondary-text-color); stroke-width: 1; }
  .banner { background: var(--secondary-background-color); border-radius: 8px; padding: 10px 12px; font-size: 13px; }
  .banner.error { border-left: 4px solid #d03b3b; }
  .empty { padding: 48px 0; text-align: center; color: var(--secondary-text-color); }
  .tooltip {
    position: absolute; z-index: 5; pointer-events: none; max-width: 240px;
    background: var(--card-background-color); color: var(--primary-text-color);
    border: 1px solid var(--divider-color); border-radius: 8px; padding: 8px 10px;
    font-size: 12px; line-height: 1.5; box-shadow: 0 4px 14px rgba(0,0,0,.18);
  }
  .tt-h { font-weight: 600; margin-bottom: 2px; }
`;

customElements.define("thermal-mpc-panel", ThermalMpcPanel);
