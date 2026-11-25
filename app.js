// Page logic: loading, interface, camera, interaction tools, frame loop. GPU work is in renderer.js, names and defaults
// in config.js.
import { FurRenderer } from './renderer.js?v=23';
import { Orbit, sub, norm, add, scl } from './camera.js?v=23';
import * as C from './config.js?v=23';
import { SKIN_DEFAULT } from './skin.js?v=23';

const $ = (id) => document.getElementById(id);
const qs = new URLSearchParams(location.search);
const QUALITY = [{ res: 1, dens: 1 }, { res: 0.85, dens: 1 }, { res: 0.7, dens: 1 }, { res: 0.7, dens: 0.6 }, { res: 0.55, dens: 0.4 }];

const st = {
  lang: qs.get('lang') || C.DEFAULT_LANG,
  motion: C.DEFAULT_MOTION, view: C.DEFAULT_VIEW, compare: C.DEFAULT_COMPARE, split: C.DEFAULT_SPLIT,
  playing: qs.get('play') !== '0', slow: false, dyn: true,
  gsize: C.GAUSS.size, gdens: C.GAUSS.density, gcolour: C.GAUSS.colour, gstyle: C.GAUSS.style,
  selected: 0xffffffff, solo: false, pickInfo: null, selInfo: null,
  body: qs.get('body') || C.DEFAULT_BODY, bodyHex: C.BODY_SWATCHES[0].hex, shOn: true, follow: true, quality: qs.get('quality') || 'auto', level: 0, stats: true,
  preset: C.DEFAULT_PRESET, params: null, flags: { collide: true, floor: true },
  skin: { ...SKIN_DEFAULT, on: qs.get('skin') !== '0' },
  // interaction tools (left rail): the active tool, whether its settings are open, the slider values of each tool
  tool: null, toolPanel: false,
  // how strands move (config.js STRAND_MODES); URL ?strands=0|1|2
  strandMode: qs.get('strands') !== null ? +qs.get('strands') : C.DEFAULT_STRAND_MODE, strandErr: null,
  sparams: Object.fromEntries(C.STRAND_SLIDERS.map((s) => [s.key, s.value])),
  toolCfg: Object.fromEntries(C.TOOLS.map((t) => [t.id, Object.fromEntries(t.sliders.map((s) => [s.key, s.value]))])),
};
// pointer state of the tools: hover (client px), hit = body point under the pointer (read back from the body depth),
// the hand's direction and speed on the body, time of the last touch, time of the gusts
const tch = { hover: null, hit: null, pressing: false, dir: [1, 0, 0], speed: 0, lastW: null, lastT: 0, lastAct: -1e9, time: 0, holdAcc: 0, lastZ: 0 };
if (st.compare && st.view === 'fur') st.view = 'gauss';
const T = () => C.TEXT[st.lang];

// ------------------------------------------------------------------------------------------------ loading
async function fetchBin(url, onBytes) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const reader = res.body.getReader(); const chunks = []; let n = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); n += value.length; onBytes(value.length); }
  const out = new Uint8Array(n); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; }
  return out.buffer;
}
async function loadAsset() {
  const manifest = await (await fetch('data/manifest.json', { cache: 'no-cache' })).json();
  const total = manifest.total_bytes; let got = 0;
  const onBytes = (n) => { got += n; $('lfill').style.width = `${(100 * got / total).toFixed(1)}%`;
    $('lmsg').textContent = `${T().loading} ${(got / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`; };
  const names = Object.keys(manifest.files); const files = {}; let i = 0;
  const worker = async () => { while (i < names.length) { const nm = names[i++]; files[nm] = await fetchBin(`data/${nm}.bin`, onBytes); } };
  await Promise.all([worker(), worker(), worker(), worker()]);
  return { manifest, files };
}

// ------------------------------------------------------------------------------------------------ interface text
function applyStaticText() {
  const t = T();
  document.documentElement.lang = st.lang;
  for (const el of document.querySelectorAll('[data-t]')) { const v = t[el.dataset.t]; if (typeof v === 'string') el.textContent = v; }
}
function applyLang() {
  const t = T();
  applyStaticText();
  $('lang').textContent = t.lang;
  $('followlbl').textContent = t.follow;
  $('fs').title = t.fullscreen; $('advbtn').title = t.advanced;
  buildMotions(); buildViews(); buildPresets(); buildSliders(); buildSkinSliders(); buildGaussSelects(); buildQuality(); updatePanels(); updateCard(); fillStats();
  buildTools(); buildStrandUI(); buildBodyUI();
  $('stclose').title = t.hideStats;
  buildAbout();
}
function buildAbout() {                  // citation, licences, controls (config.js ABOUT)
  const A = C.ABOUT, L = st.lang, t = T();
  const keys = (s) => s.split(' / ').map((alt) => alt.replace(/\[([^\]]+)\]/g, '<kbd>$1</kbd>')).join('<span class="or">/</span>');
  $('about').innerHTML =
    `<h4>${t.abCite}</h4><p class="cite">${A.cite.authors}. <span class="t">${A.cite.title}</span>. ${A.cite.venue[L] ?? A.cite.venue}.</p>` +
    `<h4>${t.abLicence}</h4><ul>${A.licences.map((l) => `<li><span>${l.what[L]}</span><span class="lic">${l.lic}${l.note ? `<small>${l.note[L]}</small>` : ''}</span></li>`).join('')}</ul>` +
    `<h4>${t.abControls}</h4><dl>${A.controls.map(([k, v]) => `<dt>${keys(k[L])}</dt><dd>${v[L]}</dd>`).join('')}</dl>`;
}
// body colour under the fur: from the fur (baked per face), one plain colour (swatches or picker), or none
const hexToRgba8 = (h) => { const v = parseInt(h.slice(1), 16); return (((v >> 16) & 255) | (((v >> 8) & 255) << 8) | ((v & 255) << 16) | (255 << 24)) >>> 0; };
function buildBodyUI() {
  const seg = $('bodyseg'); seg.innerHTML = '';
  for (const m of C.BODY_MODES) {
    const b = document.createElement('button'); b.textContent = m[st.lang];
    if (m.id === st.body) b.className = 'on';
    b.onclick = () => { st.body = m.id; buildBodyUI(); dirty = true; };
    seg.appendChild(b);
  }
  const sw = $('swatches'); sw.innerHTML = '';
  sw.classList.toggle('show', st.body === 'plain');
  let preset = false;
  for (const s of C.BODY_SWATCHES) {
    const b = document.createElement('button');
    const on = s.hex.toLowerCase() === st.bodyHex.toLowerCase(); preset = preset || on;
    b.className = 'swatch' + (on ? ' on' : '');
    b.innerHTML = `<i style="background:${s.hex}"></i>${s[st.lang]}`;
    b.onclick = () => { st.bodyHex = s.hex; buildBodyUI(); dirty = true; };
    sw.appendChild(b);
  }
  const pk = document.createElement('label');
  pk.className = 'picker' + (preset ? '' : ' on');
  pk.innerHTML = `<input type="color" value="${st.bodyHex}">${T().pickColour}`;
  const inp = pk.querySelector('input');
  inp.oninput = () => { st.bodyHex = inp.value; dirty = true; };
  inp.onchange = () => { st.bodyHex = inp.value; buildBodyUI(); dirty = true; };
  sw.appendChild(pk);
}
// strand-motion switch: the same choice in the Advanced drawer and at the foot of the stroke panel
function setStrandMode(m) {
  st.strandMode = m;
  if (rnd && rnd.setStrandMode) rnd.setStrandMode(st.strandErr ? 0 : m, st.sparams);
  buildStrandUI(); dirty = true;
}
function buildStrandUI() {
  const fill = (host, short) => {
    if (!host) return;
    host.innerHTML = '';
    for (const m of C.STRAND_MODES) {
      const b = document.createElement('button');
      b.textContent = short ? m.short[st.lang] : m[st.lang]; b.title = m[st.lang];
      if (m.id === st.strandMode) b.className = 'on';
      b.disabled = !!st.strandErr && m.id !== 0;
      b.onclick = () => setStrandMode(m.id);
      host.appendChild(b);
    }
  };
  fill($('strandseg'), false); fill($('tpstrandseg'), true);         // full names in the drawer, short in the panel
  const cur = C.STRAND_MODES.find((m) => m.id === st.strandMode) || C.STRAND_MODES[0];
  $('strandnote').textContent = cur.note[st.lang]; $('tpstrandnote').textContent = cur.note[st.lang];
  const host = $('strandsliders'); host.innerHTML = '';
  for (const s of C.STRAND_SLIDERS) {
    if (s.mode !== undefined && s.mode !== st.strandMode) continue;
    if (st.strandMode === 0) continue;
    const row = document.createElement('div'); row.className = 'row';
    const val = st.sparams[s.key], dig = s.step < 0.1 ? 2 : 0;
    row.innerHTML = `<label>${s[st.lang]}</label><input type="range" min="${s.min}" max="${s.max}" step="${s.step}" value="${val}"><span class="v">${(+val).toFixed(dig)}${s.unit}</span>`;
    const inp = row.querySelector('input'), sp = row.querySelector('.v');
    inp.oninput = () => { st.sparams[s.key] = +inp.value; sp.textContent = (+inp.value).toFixed(dig) + s.unit; if (rnd.setStrandMode) rnd.setStrandMode(undefined, st.sparams); dirty = true; };
    host.appendChild(row);
  }
  const msg = $('strandmsg');
  msg.textContent = st.strandErr ? T().strandErr : ''; msg.classList.toggle('show', !!st.strandErr);
}
function buildSkinSliders() {
  const host = $('skinsliders'); host.innerHTML = '';
  for (const s of C.SKIN_SLIDERS) {
    const row = document.createElement('div'); row.className = 'row';
    const val = st.skin[s.key];
    const dig = s.step < 0.1 ? 2 : 1;
    row.innerHTML = `<label>${s[st.lang]}</label><input type="range" min="${s.min}" max="${s.max}" step="${s.step}" value="${val}"><span class="v">${(+val).toFixed(dig)}${s.unit}</span>`;
    const inp = row.querySelector('input'), sp = row.querySelector('.v');
    inp.oninput = () => { sp.textContent = (+inp.value).toFixed(dig) + s.unit; };
    inp.onchange = () => { st.skin[s.key] = +inp.value; applySkin(); };     // the clip is recomputed on release
    inp.disabled = !st.skin.on;
    host.appendChild(row);
  }
}
function fillStats() {
  if (!man) return;
  const t = T(), loc = st.lang === 'zh' ? 'zh-CN' : 'en-US';
  const n = (v, u) => `${v.toLocaleString(loc)}${u ? ' ' + u : ''}`;
  $('stS').textContent = n(man.counts.roots, t.uStrands);
  $('stG').textContent = n(man.counts.gaussians, t.uGauss);
  $('stK').textContent = n(man.guides.count, t.uKernels);
}
function buildMotions() {
  const host = $('motions'); host.innerHTML = '';
  for (const m of C.MOTIONS) {
    const b = document.createElement('button'); b.textContent = m[st.lang]; b.title = `${m[st.lang]} (${m.key})`;
    if (m.id === st.motion) b.className = 'on';
    b.onclick = () => selectMotion(m.id); host.appendChild(b);
  }
}
function buildViews() {
  const host = $('viewseg'); host.innerHTML = '';
  for (const [id, v] of Object.entries(C.VIEWS)) {
    const b = document.createElement('button'); b.textContent = v[st.lang];
    if (id === st.view) b.className = 'on';
    b.onclick = () => selectView(id); host.appendChild(b);
  }
  $('compare').classList.toggle('on', st.compare);
  $('tagL').textContent = C.VIEWS.fur[st.lang]; $('tagR').textContent = C.VIEWS[st.view][st.lang];
}
function buildPresets() {
  const host = $('presetseg'); host.innerHTML = '';
  for (const p of C.PRESETS) {
    const b = document.createElement('button'); b.textContent = p[st.lang];
    if (p.id === st.preset) b.className = 'on';
    b.onclick = () => { st.preset = p.id; st.params = { ...man.guides.params, ...man.presets.presets[p.id] }; applyParams(true); buildPresets(); buildSliders(); };
    host.appendChild(b);
  }
}
function buildSliders() {
  const host = $('sliders'); host.innerHTML = '';
  for (const s of C.SLIDERS) {
    const row = document.createElement('div'); row.className = 'row';
    const val = st.params[s.key] ?? 0;            // floor_give_deg exists only in the newer presets
    const dig = s.step < 0.1 ? 2 : (s.step < 1 ? 1 : 0);
    row.innerHTML = `<label>${s[st.lang]}</label><input type="range" min="${s.min}" max="${s.max}" step="${s.step}" value="${val}"><span class="v">${(+val).toFixed(dig)}${s.unit}</span>`;
    const inp = row.querySelector('input'), sp = row.querySelector('.v');
    inp.oninput = () => { st.params[s.key] = +inp.value; sp.textContent = (+inp.value).toFixed(dig) + s.unit; st.preset = null; buildPresets(); applyParams(false); };
    host.appendChild(row);
  }
}
function buildGaussSelects() {
  const fill = (el, items, cur) => { el.innerHTML = ''; for (const it of items) { const o = document.createElement('option'); o.value = it.v; o.textContent = it[st.lang]; el.appendChild(o); } el.value = cur; };
  fill($('gcolour'), C.GAUSS_COLOURS, st.gcolour); fill($('gstyle'), C.GAUSS_STYLES, st.gstyle);
  $('gsize').value = st.gsize; $('gdens').value = st.gdens; $('gdens_v').textContent = `${Math.round(st.gdens * 100)}%`;
}
function buildQuality() {
  const t = T(), el = $('quality');
  const items = [['auto', t.auto], ['0', t.qHigh], ['2', t.qMed], ['4', t.qLow]];
  el.innerHTML = ''; for (const [v, n] of items) { const o = document.createElement('option'); o.value = v; o.textContent = n; el.appendChild(o); }
  el.value = st.quality;
}
// ------------------------------------------------------------------------------------------------ interaction tools
const GEAR = '<svg viewBox="0 0 24 24"><path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2.2"/><circle cx="9" cy="17" r="2.2"/></svg>';
// the tool's glyph in the text colour: a line-drawn open hand with two motion marks (a hand over strands, as in the
// video's tile, does not read at 20 px); a tool without an entry here keeps its image from config.js
const TOOL_ICONS = {
  stroke: '<svg class="ic" viewBox="0 0 24 24" aria-hidden="true"><path d="M2 6h3.2M2.6 9h2.2"/>'
    + '<path d="M8.4 12.2V5.6a1.7 1.7 0 0 1 3.4 0V11M11.8 11V4.2a1.7 1.7 0 0 1 3.4 0V11M15.2 11V5.4a1.7 1.7 0 0 1 3.4 0V12'
    + 'M18.6 12V8.4a1.6 1.6 0 0 1 3.2 0v5.4a7.6 7.6 0 0 1-7.6 7.6h-1.4a6.6 6.6 0 0 1-5-2.3l-3.6-4.3a1.6 1.6 0 0 1 2.4-2.1l2 2.1V12.2"/></svg>',
};
function buildTools() {
  const host = $('toollist'); host.innerHTML = '';
  for (const t of C.TOOLS) {
    const b = document.createElement('div');
    b.className = 'tool' + (st.tool === t.id ? ' on' : ''); b.setAttribute('role', 'button'); b.tabIndex = 0;
    b.title = `${t[st.lang]} (${t.key.toUpperCase()})${st.tool === t.id ? ' · ' + T().toolOff : ''}`;
    b.innerHTML = `${TOOL_ICONS[t.id] || `<img src="${t.icon}" alt="">`}<span class="tname">${t[st.lang]}</span><kbd>${t.key.toUpperCase()}</kbd>`;
    const g = document.createElement('button');
    g.className = 'gear' + (st.tool === t.id && st.toolPanel ? ' open' : ''); g.title = T().settings; g.innerHTML = GEAR;
    g.onclick = (e) => { e.stopPropagation(); st.toolPanel = !st.toolPanel; buildTools(); };
    b.appendChild(g);
    b.onclick = () => setTool(t.id);
    b.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setTool(t.id); } };
    host.appendChild(b);
  }
  // the settings panel of the active tool, beside its button
  const t = C.TOOLS.find((x) => x.id === st.tool);
  const pan = $('toolpanel');
  if (!t || !st.toolPanel) { pan.style.display = 'none'; }
  else {
    $('tpicon').src = t.icon; $('tptitle').textContent = t[st.lang]; $('tphint').textContent = t.hint[st.lang];
    const sh = $('tpsliders'); sh.innerHTML = '';
    for (const s of t.sliders) {
      const row = document.createElement('div'); row.className = 'row';
      const val = st.toolCfg[t.id][s.key], dig = s.step < 0.1 ? 2 : 1;
      row.innerHTML = `<label>${s[st.lang]}</label><input type="range" min="${s.min}" max="${s.max}" step="${s.step}" value="${val}"><span class="v">${(+val).toFixed(dig)}${s.unit}</span>`;
      const inp = row.querySelector('input'), sp = row.querySelector('.v');
      inp.oninput = () => { st.toolCfg[t.id][s.key] = +inp.value; sp.textContent = (+inp.value).toFixed(dig) + s.unit; updateCursor(); };
      sh.appendChild(row);
    }
    pan.style.display = 'block';
    const btn = host.children[C.TOOLS.indexOf(t)].getBoundingClientRect();
    pan.style.top = `${Math.max(12, Math.min(window.innerHeight - pan.offsetHeight - 12, btn.top + btn.height / 2 - pan.offsetHeight / 2))}px`;
  }
  const hint = $('hint');
  hint.textContent = t ? t.hint[st.lang] : ''; hint.classList.toggle('show', !!t);
  $('cv').classList.toggle('grab', !st.tool);
}
function setTool(id) {
  const was = st.tool;
  st.tool = st.tool === id ? null : id;
  tch.pressing = false; tch.hit = null;
  if (st.tool && !was) {
    if (!st.dyn) { st.dyn = true; rnd.reset(); }
    // a still dog to touch: pause; a running clip goes back to its first frame (the standing pose for walk, trot, shake)
    if (st.playing) { st.playing = false; rnd.seek(0); }
    st.selected = 0xffffffff; st.pickInfo = null; st.solo = false; updateCard();
  }
  if (!st.tool) st.toolPanel = false;
  buildTools(); updatePanels(); updateCursor(); dirty = true;
}
// world radius (m) of the hand on the body
function toolRadius() {
  return st.tool === 'stroke' ? st.toolCfg.stroke.size / 100 : 0;
}
// the cursor of the tool: a ring the size of the hand on the body
function updateCursor() {
  const br = $('brush');
  const h = tch.hover, hit = hitNow(400);
  br.style.display = st.tool === 'stroke' && h ? 'block' : 'none';
  $('cv').classList.toggle('tooling', !!(st.tool && h && (hit || tch.pressing)));
  if (!st.tool || !h) return;
  const cv = $('cv');
  const z = hit ? hit.z : (tch.lastZ || orbit.dist);
  const rpx = Math.max(6, toolRadius() * 1.5 * cv.clientHeight / z);      // fy = 1.5 H (render), here in CSS px
  br.style.transform = `translate(${h.x}px, ${h.y}px)`;
  const ring = br.firstElementChild; ring.style.width = ring.style.height = `${2 * rpx}px`;
  br.classList.toggle('down', tch.pressing && !!hit); br.classList.toggle('miss', !hit);
}
// the touch uniform of this frame (renderer.setTouch); subH = length of one dynamics substep (s)
function touchUniform(subH) {
  const decay = Math.exp(-subH / C.TOOL_GAIN.decay);
  const mode = st.tool === 'stroke' ? 1 : 0;
  const hit = hitNow(250);
  if (!mode || !tch.pressing || !hit) return { mode, active: false, decay };
  const c = st.toolCfg.stroke;
  const gshare = rnd.smode === 1 && rnd.sdReady ? 0.3 : 1;      // mode 1: the strands take most of the hand
  return { mode, active: true, gshare, p: hit.world, radius: c.size / 100, dir: tch.dir, dirW: Math.min(1, tch.speed / C.TOOL_GAIN.strokeSpeed),
    strength: c.force * C.TOOL_GAIN.stroke, press: c.press, time: tch.time, decay };
}
// the scene as seen: camera and frame (a body point read under a still pointer holds while this is unchanged)
const viewKey = () => orbit ? `${orbit.yaw.toFixed(5)},${orbit.pitch.toFixed(5)},${orbit.dist.toFixed(5)},${orbit.target.join()},${rnd.frame}` : '';
// the body point under the pointer if it is still valid: read within maxAge ms, or read for this very pointer
// position and view while nothing moves
function hitNow(maxAge) {
  const h = tch.hit; if (!h) return null;
  if (performance.now() - h.t < maxAge) return h;
  if (!st.playing && tch.hover && h.at && h.at.x === tch.hover.x && h.at.y === tch.hover.y && h.view === viewKey()) return h;
  return null;
}
// canvas pixel of a client point
function canvasPx(h) {
  const cv = $('cv'), r = cv.getBoundingClientRect();
  return [(h.x - r.left) / r.width * rnd.W, (h.y - r.top) / r.height * rnd.H];
}
// ask for the body point under the pointer in the frame on screen (no redraw); retried by the loop while busy
function requestHit() {
  if (!tch.hover || !rnd.readDepth) return;
  tch.wantDepth = !rnd.readDepth(canvasPx(tch.hover));
}
// a press that had no fresh body point under it: decided when the read-back arrives (tool, or orbit as usual)
function resolvePendingDown(hit) {
  const pd = tch.pendingDown; tch.pendingDown = null;
  if (hit) { tch.pressing = true; tch.lastW = null; tch.speed = 0; tch.lastAct = performance.now(); dirty = true; }
  else { drag = { x: pd.x, y: pd.y, b: 0, moved: 0, shift: false, t: performance.now() }; $('cv').classList.add('grabbing'); }
  updateCursor();
}
// a body point under the pointer arrived (read back from the frame drawn)
function onDepth(hit, req) {
  const now = performance.now();
  if (!hit || hit.gap > 7) { tch.hit = null; if (tch.pendingDown) resolvePendingDown(null); updateCursor(); return; }
  tch.hit = { world: hit.world, z: hit.z, camPos: req.cam.pos, t: now, at: tch.hover ? { ...tch.hover } : null, view: viewKey() };
  tch.lastZ = hit.z;
  if (tch.pendingDown) resolvePendingDown(tch.hit);
  if (tch.pressing && st.tool === 'stroke') {
    if (tch.lastW) {
      const d = sub(hit.world, tch.lastW), len = Math.hypot(d[0], d[1], d[2]);
      const dt = Math.max(1 / 120, (now - tch.lastT) / 1000);
      if (len > 5e-4) { tch.dir = norm(add(scl(tch.dir, 0.45), scl(d, 0.55 / len))); tch.speed = 0.55 * tch.speed + 0.45 * len / dt; }
      else tch.speed *= 0.7;
    }
    tch.lastW = hit.world; tch.lastT = now;
  }
  updateCursor();
}
function legendText() {
  const t = T();
  return { 5: t.legendStrand, 0: t.legendNative, 2: t.legendDir, 3: t.legendBend }[st.gcolour] || '';
}
function updatePanels() {
  const right = st.view;
  $('gtools').style.display = right === 'gauss' ? 'block' : 'none';
  $('klegend').style.display = right === 'kernels' ? 'block' : 'none';
  $('glegend').textContent = legendText();
  $('divider').style.display = st.compare ? 'block' : 'none';
  $('divider').style.left = `${st.split * 100}%`;
  $('slow').classList.toggle('on', st.slow); $('dyn').classList.toggle('on', st.dyn);
  $('stats').style.display = st.stats ? 'block' : 'none'; $('statsbtn').classList.toggle('on', st.stats);
  $('playicon').innerHTML = st.playing ? '<path d="M9 6v12M15 6v12"/>' : '<path d="M8.5 5.8v12.4l10-6.2z" fill="currentColor"/>';
  $('play').title = st.playing ? T().pause : T().play;
}
function updateCard() {
  const c = st.pickInfo;
  if (!c || st.selected === 0xffffffff) { $('card').style.display = 'none'; return; }
  const t = T();
  $('cardtext').textContent = t.card({ n: c.n, len: c.len.toFixed(1), i: c.i, gl: c.gl.toFixed(2), w: c.w.toFixed(3), op: c.op.toFixed(2), k: c.k,
    bend: st.selInfo ? st.selInfo.bendDeg.toFixed(1) : '–' });
  $('solo').textContent = t.solo; $('solo').classList.add('toggle'); $('solo').classList.toggle('on', st.solo);
  $('card').style.display = 'block';
}

// ------------------------------------------------------------------------------------------------ actions
let man, rnd, orbit, dirty = true, acc = 0, pickReq = null, drag = null;
// a motion: a clip, or Free (a new random chain of the clips; fresh = false keeps the current one)
function loadMotion(m, fresh = true) { if (m.free) rnd.loadFree(C.FREE, fresh); else rnd.loadClip(m.clip); }
function selectMotion(id) {
  st.motion = id; const m = C.MOTIONS.find((x) => x.id === id);
  loadMotion(m); acc = 0; st.playing = true; buildMotions(); updatePanels(); dirty = true;
}
function selectView(id) {
  if (id === 'fur') { st.compare = false; st.view = 'fur'; } else st.view = id;
  buildViews(); updatePanels(); dirty = true;
}
function toggleCompare() {
  st.compare = !st.compare;
  if (st.compare && st.view === 'fur') st.view = 'gauss';
  buildViews(); updatePanels(); dirty = true;
}
function applySkin() {                // reload the clip with the new loose skin and continue at the same frame
  const fr = rnd.frame;
  rnd.skin = { ...st.skin };
  loadMotion(C.MOTIONS.find((x) => x.id === st.motion), false);
  if (fr > 0) rnd.seek(fr);
  buildSkinSliders(); dirty = true;
}
function applyParams(reset) {
  rnd.setDynParams(st.params, st.flags);
  if (reset) rnd.reset();
  dirty = true;
}
function resetCamera() {
  orbit.fromView(man.camera.viewmat, rnd.clip.restCen);
  orbit.fit(0.62, rnd.W, rnd.H, 1.5 * rnd.H);
  dirty = true;
}
function sizeCanvas() {
  const lvl = QUALITY[st.quality === 'auto' ? st.level : +st.quality];
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5) * lvl.res;
  const cv = $('cv');
  rnd.resize(Math.max(64, Math.round(cv.clientWidth * dpr)), Math.max(64, Math.round(cv.clientHeight * dpr)));
  return lvl;
}

// ------------------------------------------------------------------------------------------------ main
async function main() {
  applyStaticText();
  if (qs.get('ui') === 'preview') { await uiPreview(); return; }
  if (!navigator.gpu) { $('lmsg').textContent = T().noWebGPU; return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { $('lmsg').textContent = T().noWebGPU; return; }
  const { manifest, files } = await loadAsset();
  man = manifest;
  $('lmsg').textContent = T().preparing;
  try { rnd = await FurRenderer.create(adapter, man, files, $('cv')); }
  catch (e) { $('lmsg').textContent = e.code === 'tooBig' ? T().tooBig(e.need, e.have) : String(e.message || e); return; }
  rnd.onLost = () => { $('loading').style.display = 'flex'; $('lmsg').textContent = T().deviceLost; };
  st.params = { ...man.guides.params, ...man.presets.presets[st.preset] };
  rnd.setDynParams(st.params, st.flags);
  rnd.skin = { ...st.skin };
  orbit = new Orbit();
  rnd.onPick = (info) => { st.pickInfo = info; st.selected = info ? info.strand : 0xffffffff; st.selInfo = null; if (!info) st.solo = false; updateCard(); dirty = true; };
  rnd.onSel = (s) => { st.selInfo = s; updateCard(); };
  rnd.onDepth = onDepth;
  rnd.onStrandDyn = (err) => { if (err) { st.strandErr = err; rnd.setStrandMode(0); } buildStrandUI(); dirty = true; };
  rnd.setStrandMode(st.strandMode, st.sparams);
  if (rnd.sdError) rnd.onStrandDyn(rnd.sdError);            // compiled (and failed) before the callback was set
  wire();
  applyLang();
  const m = C.MOTIONS.find((x) => x.id === (qs.get('motion') || st.motion)) || C.MOTIONS[0];
  st.motion = m.id; loadMotion(m); buildMotions();
  sizeCanvas(); resetCamera();
  $('loading').style.display = 'none';
  requestAnimationFrame(loop);
  window.fur = { rnd, st, orbit, compareTips, setTool, tch, setDirty: () => { dirty = true; } };     // for testing from the console
}

// interface preview without WebGPU (?ui=preview): the page, its panels and the tool cursors over a still of the dog
// (debug_ref/ui_preview_bg.webp, an RGBA frame of the video's opening); no asset is loaded and nothing is simulated.
// For checking the layout on machines (or headless browsers) where the GPU must not be used.
async function uiPreview() {
  man = { counts: { roots: 600000, gaussians: 7307051 }, guides: { count: 20000, params: {}, fps: 30, substeps: 4 },
    presets: { presets: Object.fromEntries(C.PRESETS.map((p) => [p.id, {}])) } };
  st.params = Object.fromEntries(C.SLIDERS.map((s) => [s.key, s.min]));
  const noop = () => {};
  rnd = { reset: noop, seek: noop, loadClip: noop, setDynParams: noop, setTouch: noop, applyRepair: noop, render: noop,
    frame: 0, total: 300, W: 1, H: 1, gpuMs: 0, drawn: 0, goal: null, pendingReset: false, clip: { restCen: [0, 0, 0] } };
  orbit = new Orbit(); orbit.dist = 2;
  const img = new Image(); img.src = qs.get('bg') || 'debug_ref/ui_preview_bg.webp';
  await img.decode();
  const cv = $('cv'), ctx = cv.getContext('2d');
  const draw = () => {
    cv.width = cv.clientWidth; cv.height = cv.clientHeight;
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, cv.width, cv.height);
    const s = Math.min(cv.width / img.width, cv.height / img.height) * +(qs.get('bgscale') || 0.8);   // about the player's framing
    ctx.drawImage(img, (cv.width - img.width * s) / 2, (cv.height - img.height * s) / 2, img.width * s, img.height * s);
  };
  draw(); window.addEventListener('resize', draw);
  // a body point wherever the still is opaque, at the camera distance (the cursors get their size from it)
  cv.addEventListener('pointermove', (e) => {
    if (!st.tool) return;
    const a = ctx.getImageData(Math.round(e.clientX), Math.round(e.clientY), 1, 1).data[3];
    onDepth(a > 40 ? { world: [0, 0, 0], z: 2, gap: 0 } : null, { cam: { pos: [0, 0, 2] } });
  });
  rnd.readDepth = (at) => {                     // the still's alpha under the pointer stands in for the body depth
    const h = tch.hover; if (!h) return true;
    const a = ctx.getImageData(Math.round(h.x), Math.round(h.y), 1, 1).data[3];
    setTimeout(() => onDepth(a > 40 ? { world: [0, 0, 0], z: 2, gap: 0 } : null, { cam: { pos: [0, 0, 2] } }), 16);
    return true;
  };
  rnd.W = cv.width; rnd.H = cv.height;
  wire(); applyLang();
  st.playing = false; updatePanels();
  $('stF').textContent = '–'; $('stD').textContent = '–';
  $('loading').style.display = 'none';
  window.fur = { st, setTool, tch, buildTools, updateCursor, selectView, toggleCompare, applyLang, updatePanels, updateCard };
}

// ------------------------------------------------------------------------------------------------ input
function wire() {
  const cv = $('cv');
  cv.addEventListener('contextmenu', (e) => e.preventDefault());
  cv.addEventListener('pointerdown', (e) => {
    cv.setPointerCapture(e.pointerId);
    tch.hover = { x: e.clientX, y: e.clientY };
    // with a tool, a press on the body touches the fur; elsewhere it orbits as usual
    const hit = hitNow(150);
    if (st.tool && e.button === 0 && !e.shiftKey) {
      drag = null;
      if (hit) { tch.pressing = true; tch.lastW = null; tch.speed = 0; tch.lastAct = performance.now(); updateCursor(); dirty = true; return; }
      if (rnd.readDepth) { tch.pendingDown = { x: e.clientX, y: e.clientY }; requestHit(); return; }   // decided on the read-back
    }
    drag = { x: e.clientX, y: e.clientY, b: e.button, moved: 0, shift: e.shiftKey, t: performance.now() }; orbit.vyaw = 0; orbit.vpitch = 0;
    cv.classList.add('grabbing');
  });
  cv.addEventListener('pointermove', (e) => {
    tch.hover = { x: e.clientX, y: e.clientY };
    if (st.tool) { updateCursor(); if (!tch.pressing) requestHit(); }
    if (tch.pressing) { tch.lastAct = performance.now(); dirty = true; }
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY; drag.moved += Math.abs(dx) + Math.abs(dy); drag.t = performance.now();
    if (drag.b === 2 || drag.b === 1 || drag.shift) {
      const P = orbit.pose([0, 0, 0]); const k = orbit.dist / (1.5 * cv.clientHeight);
      orbit.target = orbit.target.map((v, i) => v - P.x[i] * dx * k - P.y[i] * dy * k);
    } else {
      orbit.yaw -= dx * 0.006; orbit.pitch = Math.max(-0.2, Math.min(1.35, orbit.pitch + dy * 0.006));
      orbit.vyaw = -dx * 0.006; orbit.vpitch = dy * 0.006;
    }
    dirty = true;
  });
  cv.addEventListener('pointerleave', () => { if (!tch.pressing && !drag) { tch.hover = null; updateCursor(); } });
  cv.addEventListener('pointerup', (e) => {
    cv.classList.remove('grabbing');
    if (tch.pendingDown) { tch.pendingDown = null; return; }
    if (tch.pressing) { tch.pressing = false; tch.lastAct = performance.now(); updateCursor(); dirty = true; return; }
    if (drag && drag.moved < 4 && drag.b === 0 && !st.tool) {
      const r = cv.getBoundingClientRect();
      pickReq = [(e.clientX - r.left) / r.width * rnd.W, (e.clientY - r.top) / r.height * rnd.H];
      orbit.vyaw = 0; orbit.vpitch = 0;
    } else if (drag && performance.now() - drag.t > 80) { orbit.vyaw = 0; orbit.vpitch = 0; }   // released without moving: no spin
    drag = null; dirty = true;
  });
  cv.addEventListener('dblclick', () => resetCamera());
  cv.addEventListener('wheel', (e) => { e.preventDefault(); orbit.dist = Math.max(0.25, Math.min(8, orbit.dist * Math.exp(e.deltaY * 0.0012))); dirty = true; }, { passive: false });
  // divider
  const knob = document.querySelector('#divider .knob');
  let dd = false;
  knob.addEventListener('pointerdown', (e) => { dd = true; knob.setPointerCapture(e.pointerId); });
  knob.addEventListener('pointermove', (e) => { if (!dd) return; st.split = Math.max(0.08, Math.min(0.92, e.clientX / window.innerWidth)); updatePanels(); dirty = true; });
  knob.addEventListener('pointerup', () => { dd = false; });
  // bottom bar
  $('play').onclick = () => { st.playing = !st.playing; updatePanels(); dirty = true; };
  $('slow').onclick = () => { st.slow = !st.slow; updatePanels(); };
  $('dyn').onclick = () => { st.dyn = !st.dyn; rnd.reset(); updatePanels(); dirty = true; };
  $('compare').onclick = toggleCompare;
  $('statsbtn').onclick = () => { st.stats = !st.stats; updatePanels(); };
  $('stclose').onclick = () => { st.stats = false; updatePanels(); };
  $('lang').onclick = () => { st.lang = st.lang === 'zh' ? 'en' : 'zh'; applyLang(); };
  // full screen: hidden where the page may not use it (an embedded browser pane ignores the request); if a request is
  // refused or nothing happens within a second, the button goes away instead of staying dead
  if (!document.fullscreenEnabled) $('fs').style.display = 'none';
  $('fs').onclick = () => {
    if (document.fullscreenElement) { document.exitFullscreen().catch(() => {}); return; }
    const hide = () => { if (!document.fullscreenElement) $('fs').style.display = 'none'; };
    Promise.resolve(document.documentElement.requestFullscreen()).catch(hide);
    setTimeout(hide, 1000);
  };
  $('advbtn').onclick = () => $('adv').classList.toggle('open');
  $('advclose').onclick = () => $('adv').classList.remove('open');
  const prog = $('progress'); let pd = false;
  const seekAt = (e) => { const r = prog.getBoundingClientRect(); const u = Math.max(0, Math.min(1, (e.clientX - r.left - 8) / (r.width - 16))); rnd.seek(Math.round(u * (rnd.total - 1))); dirty = true; };
  prog.addEventListener('pointerdown', (e) => { pd = true; prog.setPointerCapture(e.pointerId); seekAt(e); });
  prog.addEventListener('pointermove', (e) => { if (pd) seekAt(e); });
  prog.addEventListener('pointerup', () => { pd = false; });
  // Gaussian tools
  $('gsize').oninput = () => { st.gsize = +$('gsize').value; dirty = true; };
  $('gdens').oninput = () => { st.gdens = +$('gdens').value; $('gdens_v').textContent = `${Math.round(st.gdens * 100)}%`; dirty = true; };
  $('gcolour').onchange = () => { st.gcolour = +$('gcolour').value; updatePanels(); dirty = true; };
  $('gstyle').onchange = () => { st.gstyle = +$('gstyle').value; dirty = true; };
  // selection card
  $('solo').onclick = () => { st.solo = !st.solo; updateCard(); dirty = true; };
  $('unselect').onclick = () => { st.selected = 0xffffffff; st.pickInfo = null; st.solo = false; updateCard(); dirty = true; };
  // advanced
  $('collide').onchange = () => { st.flags.collide = $('collide').checked; applyParams(false); };
  $('floor').onchange = () => { st.flags.floor = $('floor').checked; applyParams(false); };
  $('restart').onclick = () => { rnd.reset(); dirty = true; };
  $('shon').onchange = () => { st.shOn = $('shon').checked; dirty = true; };
  $('skin').checked = st.skin.on;
  $('skin').onchange = () => { st.skin.on = $('skin').checked; applySkin(); };
  $('repair').onchange = () => { rnd.applyRepair($('repair').checked); dirty = true; };
  $('follow').onchange = () => { st.follow = $('follow').checked; dirty = true; };
  $('quality').onchange = () => { st.quality = $('quality').value; st.level = 0; dirty = true; };
  window.addEventListener('resize', () => { dirty = true; if (st.toolPanel) buildTools(); });
  $('tpclose').onclick = () => { st.toolPanel = false; buildTools(); };
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    const m = C.MOTIONS.find((x) => x.key === e.key);
    if (m) { selectMotion(m.id); return; }
    const k = e.key.toLowerCase();
    if (e.code === 'Space') { e.preventDefault(); $('play').click(); }
    else if (k === 's') $('slow').click();
    else if (k === 'd') $('dyn').click();
    else if (k === 'c') toggleCompare();
    else if (k === 'g') selectView('gauss');
    else if (k === 'f') $('fs').click();
    else if (k === 'r') resetCamera();
    else if (k === 'h') setTool('stroke');
    else if (e.key === 'Escape') { $('adv').classList.remove('open'); if (st.tool) setTool(st.tool); }
  });
  document.addEventListener('visibilitychange', () => { dirty = true; });
}

// ------------------------------------------------------------------------------------------------ frame loop
let last = performance.now(), lastDraw = 0, fpsN = 0, fpsT = 0, slowSince = 0, fastSince = 0;
function loop(now) {
  requestAnimationFrame(loop);
  if (document.hidden) return;
  const spinning = orbit.step();
  if (tch.wantDepth) requestHit();
  // the fur keeps moving while it is touched and for a few seconds after (the motion may be paused)
  const live = !!st.tool && st.dyn && (tch.pressing || now - tch.lastAct < C.TOOL_GAIN.settle * 1000);
  const busy = st.playing || dirty || spinning || pickReq || rnd.goal !== null || rnd.pendingReset || live;
  if (!busy || now - lastDraw < 1000 / 61) return;
  const dt = Math.min(0.1, (now - last) / 1000); last = now; lastDraw = now; dirty = false;
  fpsN++; fpsT += dt;
  if (fpsT > 0.5) {
    const f = (fpsN / fpsT).toFixed(0);
    $('fpsv').textContent = `${f} fps · GPU ${rnd.gpuMs.toFixed(1)} ms · ${rnd.W}×${rnd.H}`;
    $('stF').textContent = `${f} fps`;
    $('stD').textContent = `${(rnd.drawn / 1e6).toFixed(2)} M ${T().uDrawnSuffix}`;
    fpsN = 0; fpsT = 0;
  }
  const lvl = sizeCanvas();
  const W = rnd.W, H = rnd.H;
  let target = rnd.frame;
  if (st.playing) {
    acc += dt * (st.slow ? 0.25 : 1) * 30;
    const n = Math.floor(acc); acc -= n;
    target = rnd.frame + Math.min(n, 4);
    if (target >= rnd.total) {                    // end of the clip: again from the start; Free draws a new chain
      if (C.MOTIONS.find((x) => x.id === st.motion)?.free) loadMotion(C.MOTIONS[0]); else rnd.reset();
      target = 0;
    }
  }
  const camFn = (f) => orbit.pose(st.follow ? rnd.followOffset(f) : [0, 0, 0]);
  const fur = { ...C.VIEWS.fur, body: st.body === 'fur' ? 1 : st.body === 'plain' ? 3 : 0, size: 1, lod: lvl.dens };
  const other = (id) => (id === 'gauss' ? { ...C.VIEWS.gauss, colour: st.gcolour, style: st.gstyle, size: st.gsize, lod: st.gdens }
    : id === 'kernels' ? { ...C.VIEWS.kernels, size: 1, lod: 1 } : fur);
  const sides = st.compare ? [{ ...fur, x0: 0, x1: st.split * W }, { ...other(st.view), x0: st.split * W, x1: W }]
    : [{ ...other(st.view), x0: 0, x1: W }];
  const pick = pickReq; pickReq = null;
  // tools: dynamics frames with the body held still (paused), the touch of this frame, the body depth under the pointer
  let hold = 0;
  if (live) {
    tch.time += dt;
    if (!st.playing) { tch.holdAcc += dt * (st.slow ? 0.25 : 1) * man.guides.fps; hold = Math.min(4, Math.floor(tch.holdAcc)); tch.holdAcc -= hold; }
  } else tch.holdAcc = 0;
  rnd.setTouch(touchUniform(1 / (man.guides.fps * man.guides.substeps)));
  const depthAt = st.tool && tch.hover ? canvasPx(tch.hover) : null;
  rnd.render({ target, dynOn: st.dyn, shOn: st.shOn, camFn, fy: 1.5 * H, dist: orbit.dist, sides, pick, hold, depthAt,
    selected: st.selected, solo: st.solo, highlight: true, bgTop: C.BG.top, bgBot: C.BG.bottom, vignette: C.BG.vignette,
    bodyRgb: hexToRgba8(st.bodyHex) });
  const u = rnd.frame / Math.max(1, rnd.total - 1);
  $('pfill').style.width = `calc((100% - 16px) * ${u})`; $('phead').style.left = `calc(8px + (100% - 16px) * ${u})`;     // track inset 8 px (style.css)
  // automatic quality: step down when frames take long, back up when there is room
  if (st.quality === 'auto' && st.playing) {
    if (rnd.gpuMs > 30) { if (!slowSince) slowSince = now; if (now - slowSince > 3000 && st.level < QUALITY.length - 1) { st.level++; slowSince = 0; } }
    else slowSince = 0;
  }
}

// compare the browser dynamics with an H100 run (debug_ref/), e.g. fur.compareTips('debug_ref/tips_mocap_gallop_turn.npy', [10, 30])
async function compareTips(url, frames) {
  const buf = await (await fetch(url)).arrayBuffer();
  const hl = new DataView(buf).getUint16(8, true); const ref = new Float32Array(buf, 10 + hl);
  const K = rnd.K, out = [];
  for (let n = 0; n < frames.length; n++) {
    const tip = await rnd.tipsAt(frames[n]);
    let mx = 0; const d = [];
    for (let k = 0; k < K; k++) { const o = (n * K + k) * 3; const v = Math.hypot(tip[3 * k] - ref[o], tip[3 * k + 1] - ref[o + 1], tip[3 * k + 2] - ref[o + 2]); d.push(v); mx = Math.max(mx, v); }
    d.sort((a, b) => a - b);
    out.push({ frame: frames[n], median_mm: d[K >> 1] * 1e3, max_mm: mx * 1e3 });
  }
  st.playing = false; updatePanels(); dirty = true;
  return out;
}

main().catch((e) => { console.error(e); $('lmsg').textContent = String(e.message || e); });
