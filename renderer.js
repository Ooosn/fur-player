// Everything on the GPU: buffers, pipelines, the guide dynamics, and one frame of drawing (one or two screen
// regions, each with its own view: fur render, Gaussians or kernels). No DOM here.
import * as S from './shaders.js?v=23';
import { rebuild } from './dyn.js?v=23';
import { looseSkin, SKIN_DEFAULT } from './skin.js?v=23';
import { buildFree } from './free.js?v=23';

export const HOLD = 45;                  // rest frames after a clip, the fur settles (motion_run.py --extra-rest-frames 45)
const DSTRIDE = 256;                     // dynamic uniform stride
const FRAME_BYTES = 320;
const DEPTH_WIN = 15;                    // body-depth window read back under the pointer (pixels, odd)

function buildStrandData(man, f) {
  const R = man.counts.roots;
  const goff = new Uint32Array(f.strand_goff);
  const q = new Uint32Array(f.strand_q);
  const jn = new Uint32Array(f.strand_joints);
  const gd = new Uint32Array(f.strand_guides);
  const sh = new Uint32Array(f.strand_sh);
  const fr = new Uint32Array(f.strand_frame);
  const d = new Uint32Array(R * S.STRAND_STRIDE);
  for (let r = 0; r < R; r++) {
    const b = r * S.STRAND_STRIDE;
    d[b] = goff[r]; d[b + 1] = goff[r + 1] - goff[r];
    d[b + 2] = q[3 * r]; d[b + 3] = q[3 * r + 1]; d[b + 4] = q[3 * r + 2];
    for (let k = 0; k < 6; k++) d[b + 5 + k] = jn[6 * r + k];
    for (let k = 0; k < 12; k++) d[b + 11 + k] = gd[12 * r + k];
    for (let k = 0; k < 4; k++) d[b + 23 + k] = sh[4 * r + k];
    d[b + 27] = fr[2 * r]; d[b + 28] = fr[2 * r + 1];
  }
  return { d, goff };
}

const f16 = (h) => { const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; return e === 0 ? s * m * 2 ** -24 : s * (1 + m / 1024) * 2 ** (e - 15); };

export class FurRenderer {
  static async create(adapter, man, files, canvas) {
    const lim = adapter.limits;
    const R = man.counts.roots, N = man.counts.gaussians;
    const need = Math.max(N * 32, R * S.STRAND_STRIDE * 4, files.gauss.byteLength);
    if (lim.maxStorageBufferBindingSize < need || lim.maxBufferSize < need) {
      const e = new Error('buffer limit'); e.code = 'tooBig'; e.need = Math.round(need / 1e6); e.have = Math.round(lim.maxStorageBufferBindingSize / 1e6); throw e;
    }
    const feats = adapter.features.has('float32-filterable') ? ['float32-filterable'] : [];
    const device = await adapter.requestDevice({ requiredFeatures: feats, requiredLimits: {
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize,
      maxStorageBuffersPerShaderStage: Math.min(lim.maxStorageBuffersPerShaderStage, 10),
      maxComputeWorkgroupsPerDimension: lim.maxComputeWorkgroupsPerDimension } });
    const r = new FurRenderer();
    await r.init(device, man, files, canvas);
    return r;
  }

  async init(device, man, files, canvas) {
    this.device = device; this.man = man; this.files = files;
    const R = this.R = man.counts.roots, N = this.N = man.counts.gaussians, K = this.K = man.guides.count;
    this.ctx = canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device, format: this.format, alphaMode: 'opaque' });
    this.canvas = canvas;
    const BU = GPUBufferUsage;
    this.BU = BU;
    const mk = (data, usage = BU.STORAGE) => {
      const b = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage: usage | BU.COPY_DST, mappedAtCreation: true });
      new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)); b.unmap(); return b;
    };
    const empty = (size, usage = BU.STORAGE) => device.createBuffer({ size: Math.max(16, size), usage: usage | BU.COPY_DST | BU.COPY_SRC });
    this.mk = mk; this.empty = empty;
    // static data
    const { d: strandData, goff } = buildStrandData(man, files);
    this.strandData = strandData; this.goff = goff;
    this.gauss = new Uint32Array(files.gauss);
    // colour repair of fur no training view saw (scripts/repair_unseen_colours.py): applied by default, switchable
    this.repair = null;
    if (files.repair_gauss && files.repair_sh) {
      const pg = new Uint32Array(files.repair_gauss), ps = new Uint32Array(files.repair_sh);
      const n = pg.length / 2, m = ps.length / 5;
      const gi = new Uint32Array(n), gNew = new Uint32Array(n), gOld = new Uint32Array(n);
      for (let k = 0; k < n; k++) { gi[k] = pg[2 * k]; gNew[k] = pg[2 * k + 1]; gOld[k] = this.gauss[3 * gi[k] + 2]; }
      const si = new Uint32Array(m), sNew = new Uint32Array(4 * m), sOld = new Uint32Array(4 * m);
      for (let k = 0; k < m; k++) {
        si[k] = ps[5 * k];
        for (let c = 0; c < 4; c++) { sNew[4 * k + c] = ps[5 * k + 1 + c]; sOld[4 * k + c] = strandData[si[k] * S.STRAND_STRIDE + 23 + c]; }
      }
      this.repair = { gi, gNew, gOld, si, sNew, sOld, on: false };
      this.applyRepair(true, false);
    }
    this.bStrand = mk(strandData);
    this.bGauss = mk(this.gauss);
    this.gstatic = new Float32Array(files.guide_static);
    this.bGstatic = mk(this.gstatic);
    this.bGjoints = mk(new Uint32Array(files.guide_joints));
    this.nb = { row: new Uint32Array(files.guide_nb_row), ids: new Uint32Array(files.guide_nb_ids), w: new Float32Array(files.guide_nb_w) };
    this.bNbRow = mk(this.nb.row); this.bNbIds = mk(this.nb.ids); this.bNbW = mk(this.nb.w);
    this.bSdfT = mk(new Uint32Array(files.sdf_table));
    this.bSdfB = mk(new Uint32Array(files.sdf_bricks));
    const sdfU = new ArrayBuffer(64); {
      const i = new Int32Array(sdfU), f = new Float32Array(sdfU), s = man.sdf;
      i[0] = s.dims[0]; i[1] = s.dims[1]; i[2] = s.dims[2]; i[3] = s.bricks[0]; i[4] = s.bricks[1]; i[5] = s.bricks[2];
      f[6] = s.step; f[7] = s.band; f[8] = s.lo[0]; f[9] = s.lo[1]; f[10] = s.lo[2]; f[11] = s.ext[0]; f[12] = s.ext[1]; f[13] = s.ext[2];
    }
    this.bSdfU = mk(new Uint8Array(sdfU), BU.UNIFORM);
    this.bodyV = new Float32Array(files.body_v);
    this.bodyF = new Uint32Array(files.body_f);
    this.bBodyV = mk(this.bodyV); this.bBodyJ = mk(new Uint32Array(files.body_joints));
    this.bBodyF = mk(this.bodyF); this.bBodyRgb = mk(new Uint32Array(files.body_rgb));
    let zmin = Infinity; for (let i = 2; i < this.bodyV.length; i += 3) zmin = Math.min(zmin, this.bodyV[i]);
    this.groundZ = zmin;
    const g = { K, hlen: new Float32Array(K), ca0: new Float32Array(K), sa0: new Float32Array(K) };
    for (let k = 0; k < K; k++) { g.hlen[k] = this.gstatic[k * 28 + 25]; g.ca0[k] = this.gstatic[k * 28 + 26]; g.sa0[k] = this.gstatic[k * 28 + 27]; }
    this.g = g;
    this.rig = man.rig_joints;
    // dynamic buffers
    this.bGst = empty(S.GSTATE.total * K * 4);
    this.bCoef = empty(K * 10 * 4);
    this.bRecs = empty(N * 32);
    this.bKeys = empty(N * 4);
    this.bSorted = empty(N * 4);
    this.bScratch = empty((S.NBINS + 64) * 4);
    this.bBlocks = empty(256 * 4);
    this.bIndirect = empty(8 * 4, BU.STORAGE | BU.INDIRECT);
    this.bFrame = [empty(FRAME_BYTES, BU.UNIFORM), empty(FRAME_BYTES, BU.UNIFORM)];
    this.bRV = empty(16, BU.UNIFORM);
    this.bCount = mk(new Uint32Array([N, 0, 0, 0]), BU.UNIFORM);
    this.bPickU = [mk(new Float32Array(4), BU.UNIFORM), mk(new Float32Array(4), BU.UNIFORM)];
    this.bPickBest = empty(16);
    this.rbPick = device.createBuffer({ size: 16, usage: BU.MAP_READ | BU.COPY_DST });
    this.rbSel = device.createBuffer({ size: 32, usage: BU.MAP_READ | BU.COPY_DST });
    this.rbStats = device.createBuffer({ size: 32, usage: BU.MAP_READ | BU.COPY_DST });
    this.statsBusy = false; this.drawn = 0;
    // interaction (stroke / dryer): one uniform for the guide drive, and a read-back of the body depth under the pointer
    this.bTouch = empty(64, BU.UNIFORM);
    this.touchBuf = new ArrayBuffer(64);
    this.rbDepth = device.createBuffer({ size: 256 * DEPTH_WIN, usage: BU.MAP_READ | BU.COPY_DST });
    this.depthBusy = false; this.onDepth = null;
    this.initStrandDyn();
    this.makePipelines();
    this.frameBuf = new ArrayBuffer(FRAME_BYTES);
    this.clip = null; this.W = 0; this.H = 0;
    this.frame = 0; this.simFrame = 0; this.pendingReset = true; this.goal = null;
    this.gpuMs = 0; this.gpuBusy = false; this.pickBusy = false; this.selBusy = false;
    this.onPick = null; this.onSel = null;
    device.lost.then((info) => { if (this.onLost) this.onLost(info); });
  }

  // switch the colour repair on or off (CPU arrays, then one upload of the two buffers)
  applyRepair(on, upload = true) {
    const rp = this.repair;
    if (!rp || rp.on === on) return;
    const g = on ? rp.gNew : rp.gOld, s = on ? rp.sNew : rp.sOld;
    for (let k = 0; k < rp.gi.length; k++) this.gauss[3 * rp.gi[k] + 2] = g[k];
    for (let k = 0; k < rp.si.length; k++) for (let c = 0; c < 4; c++) this.strandData[rp.si[k] * S.STRAND_STRIDE + 23 + c] = s[4 * k + c];
    rp.on = on;
    if (upload) {
      this.device.queue.writeBuffer(this.bGauss, 0, this.gauss);
      this.device.queue.writeBuffer(this.bStrand, 0, this.strandData);
    }
  }

  makePipelines() {
    const device = this.device, format = this.format;
    const mod = (code, label) => {
      const m = device.createShaderModule({ code, label });
      m.getCompilationInfo().then((info) => { for (const msg of info.messages) if (msg.type === 'error') console.error(label, msg.lineNum, msg.message); });
      return m;
    };
    const cp = (code, label, layout = 'auto', entry = 'main') => device.createComputePipeline({ label, layout, compute: { module: mod(code, label), entryPoint: entry } });
    const BL = (entries) => device.createBindGroupLayout({ entries: entries.map(([binding, type, dyn]) => ({
      binding, visibility: GPUShaderStage.COMPUTE, buffer: { type, hasDynamicOffset: !!dyn } })) });
    this.layDrive = BL([[0, 'uniform', 1], [1, 'read-only-storage'], [2, 'read-only-storage'], [3, 'read-only-storage'], [4, 'read-only-storage'], [5, 'storage'],
      [6, 'uniform']]);
    this.layProj = BL([[0, 'uniform', 1], [1, 'read-only-storage'], [2, 'read-only-storage'], [3, 'read-only-storage'], [4, 'read-only-storage'],
      [5, 'read-only-storage'], [6, 'storage'], [7, 'read-only-storage'], [8, 'read-only-storage'], [9, 'uniform']]);
    const PL = (l) => device.createPipelineLayout({ bindGroupLayouts: [l] });
    this.pDrive = cp(S.DRIVE, 'drive', PL(this.layDrive));
    this.pProj = cp(S.PROJECT, 'project', PL(this.layProj));
    this.pRest = cp(S.REST_DISTANCE, 'rest', PL(this.layProj));
    this.pRot = cp(S.ROTVEC, 'rotvec');
    this.pStr = cp(S.STRANDS, 'strands');
    const scanMod = mod(S.SCAN, 'scan');
    this.layScan = device.createBindGroupLayout({ entries: [0, 1, 2].map((b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } })) });
    this.pScan = ['reduce', 'top', 'down'].map((e) => device.createComputePipeline({ layout: PL(this.layScan), compute: { module: scanMod, entryPoint: e } }));
    this.pScat = cp(S.SCATTER, 'scatter');
    this.pPick = cp(S.PICK, 'pick');
    const depth = (write, compare) => ({ format: 'depth24plus', depthWriteEnabled: write, depthCompare: compare });
    const premul = { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' } };
    const bodyMod = mod(S.BODY, 'body'), backMod = mod(S.BACKDROP, 'backdrop'), splatMod = mod(S.SPLAT, 'splat'), linesMod = mod(S.LINES, 'lines');
    const rp = (label, module, vs, fs, targets, topology, ds) => device.createRenderPipeline({ label, layout: 'auto',
      vertex: { module, entryPoint: vs }, fragment: { module, entryPoint: fs, targets }, primitive: { topology, cullMode: 'none' }, depthStencil: ds });
    this.pBg = rp('bg', backMod, 'vs_bg', 'fs_bg', [{ format }, { format: 'r32float' }], 'triangle-list', depth(false, 'always'));
    this.pShadow = rp('shadow', backMod, 'vs_shadow', 'fs_shadow', [{ format, blend: premul }, { format: 'r32float', writeMask: 0 }], 'triangle-list', depth(false, 'always'));
    this.pBody = rp('body', bodyMod, 'vs', 'fs', [{ format }, { format: 'r32float' }], 'triangle-list', depth(true, 'less-equal'));
    this.pSplat = rp('splat', splatMod, 'vs', 'fs', [{ format, blend: premul }], 'triangle-strip', depth(false, 'always'));
    this.pLines = rp('lines', linesMod, 'vs', 'fs', [{ format }], 'line-list', depth(false, 'less-equal'));
    // bind groups that never change
    const bg = (pipe, entries, layout) => device.createBindGroup({ layout: layout || pipe.getBindGroupLayout(0),
      entries: entries.map(([binding, res]) => ({ binding, resource: res.createView ? res.createView() : { buffer: res } })) });
    this.bg = bg;
    this.bgScan = bg(null, [[0, this.bScratch], [1, this.bBlocks], [2, this.bIndirect]], this.layScan);
    this.bgScat = bg(this.pScat, [[0, this.bCount], [1, this.bKeys], [2, this.bScratch], [3, this.bSorted]]);
    this.bgPick = [0, 1].map((p) => bg(this.pPick, [[0, this.bPickU[p]], [1, this.bRecs], [2, this.bKeys], [3, this.bPickBest]]));
    this.bgSplat = [0, 1].map((i) => bg(this.pSplat, [[0, this.bFrame[i]], [1, this.bRecs], [2, this.bSorted]]));
    this.bgBg = [0, 1].map((i) => bg(this.pBg, [[0, this.bFrame[i]]]));
    this.bgShadow = [0, 1].map((i) => bg(this.pShadow, [[0, this.bFrame[i]]]));
  }

  // ------------------------------------------------------------------------------------------- dynamics parameters
  setDynParams(p, flags) {
    const rb = rebuild(p, this.g, this.nb, this.man.guides.fps, this.man.guides.substeps);
    this.device.queue.writeBuffer(this.bCoef, 0, rb.coef);
    this.dyn = { ...rb, p, flags: { ...flags } };
    this.satDeg = p.sat_deg ?? this.man.sim.sat_deg;
    if (this.clip) this.writeDynUniforms();
    return rb.info;
  }
  writeDynUniforms() {
    const c = this.clip, d = this.dyn, p = d.p, K = this.K, man = this.man;
    const n = c.F * c.S + 2;                // + init at frame 0 (reset) + init at a given frame (kernels restarted)
    const ab = new ArrayBuffer(n * DSTRIDE);
    const u = new Uint32Array(ab), f = new Float32Array(ab);
    for (let e = 0; e < n; e++) {
      const b = e * DSTRIDE / 4, init = e >= c.F * c.S;
      u[b] = K; u[b + 1] = 39; u[b + 2] = init ? 1 : 0; u[b + 3] = init ? c.S - 1 : e;
      f[b + 4] = 1 / d.h; f[b + 5] = 1 / 9.81; f[b + 6] = p.inertia_gain; f[b + 7] = p.inertia_limit_g;
      f[b + 8] = p.gravity_gain; f[b + 9] = man.guides.gravity[0]; f[b + 10] = man.guides.gravity[1]; f[b + 11] = man.guides.gravity[2];
      f[b + 12] = d.coh; f[b + 13] = p.velocity_correction; f[b + 14] = d.cone_c; f[b + 15] = d.cone_s;
      u[b + 16] = p.cohesion > 0 ? 1 : 0; u[b + 17] = d.flags.collide ? 1 : 0; u[b + 18] = d.flags.floor ? 1 : 0; f[b + 19] = p.margin;
    }
    this.device.queue.writeBuffer(c.bDyn, 0, ab);
  }

  // ------------------------------------------------------------------------------------------- clips
  loadClip(name) {
    const info = this.man.clips.find((c) => c.name === name);
    return this.installClip(name, info.frames, info.substeps, new Float32Array(this.files[`motion_${name}`]), HOLD);
  }
  // Free motion (free.js): the clips chained in random order into one long clip, no rest frames after it.
  // spec: { seconds, fade, clips: [{ clip, w }] }; fresh = false reuses the last sequence (e.g. after a skin change)
  loadFree(spec, fresh = true) {
    if (fresh || !this.freeSeq) {
      const clips = spec.clips.map((c) => {
        const info = this.man.clips.find((x) => x.name === c.clip);
        return { name: c.clip, rec: new Float32Array(this.files[`motion_${c.clip}`]), F: info.frames, S: info.substeps, w: c.w };
      });
      this.freeSeq = buildFree(clips, this.rig, this.man.guides.fps, spec.seconds, spec.fade);
    }
    const s = this.freeSeq;
    return this.installClip('free', s.F, s.S, s.rec, 0);
  }
  installClip(name, F, Sx, rec, hold) {
    const info = { name, frames: F, substeps: Sx }, J = 39;
    const jm = new Float32Array(F * Sx * J * 12);
    for (let i = 0; i < F * Sx * J; i++) {
      const b = i * 8, x = rec[b], y = rec[b + 1], z = rec[b + 2], w = rec[b + 3], o = i * 12;
      jm[o] = 1 - 2 * (y * y + z * z); jm[o + 1] = 2 * (x * y - z * w); jm[o + 2] = 2 * (x * z + y * w); jm[o + 3] = rec[b + 4];
      jm[o + 4] = 2 * (x * y + z * w); jm[o + 5] = 1 - 2 * (x * x + z * z); jm[o + 6] = 2 * (y * z - x * w); jm[o + 7] = rec[b + 5];
      jm[o + 8] = 2 * (x * z - y * w); jm[o + 9] = 2 * (y * z + x * w); jm[o + 10] = 1 - 2 * (x * x + y * y); jm[o + 11] = rec[b + 6];
    }
    // per frame: centre of all joints and heading (pelvis -> chest) for the shadow; middle of pelvis and chest for the
    // camera (the centre of all joints swings with the legs, head and tail at the stride rate: the camera shook)
    const rig = this.rig;
    const cen = new Float32Array(F * 3), head = new Float32Array(F * 2), core = new Float32Array(F * 3);
    for (let fr = 0; fr < F; fr++) {
      const base = ((fr * Sx + Sx - 1) * J) * 12;
      let cx = 0, cy = 0, cz = 0; const P = [];
      for (let j = 0; j < J; j++) {
        const o = base + j * 12, q = rig[j];
        const px = jm[o] * q[0] + jm[o + 1] * q[1] + jm[o + 2] * q[2] + jm[o + 3];
        const py = jm[o + 4] * q[0] + jm[o + 5] * q[1] + jm[o + 6] * q[2] + jm[o + 7];
        const pz = jm[o + 8] * q[0] + jm[o + 9] * q[1] + jm[o + 10] * q[2] + jm[o + 11];
        cx += px; cy += py; cz += pz; P.push([px, py, pz]);
      }
      cen[3 * fr] = cx / J; cen[3 * fr + 1] = cy / J; cen[3 * fr + 2] = cz / J;
      for (let d = 0; d < 3; d++) core[3 * fr + d] = 0.5 * (P[5][d] + P[8][d]);
      const hx = P[8][0] - P[5][0], hy = P[8][1] - P[5][1], hn = Math.hypot(hx, hy) || 1;
      head[2 * fr] = hx / hn; head[2 * fr + 1] = hy / hn;
    }
    const smooth = (a, dim, sig) => {               // Gaussian smoothing along the frames
      const out = new Float32Array(a.length), rad = Math.ceil(3 * sig);
      for (let fr = 0; fr < F; fr++) for (let d = 0; d < dim; d++) {
        let s = 0, ws = 0;
        for (let k = -rad; k <= rad; k++) { const t = Math.min(F - 1, Math.max(0, fr + k)); const w = Math.exp(-0.5 * k * k / (sig * sig)); s += w * a[t * dim + d]; ws += w; }
        out[fr * dim + d] = s / ws;
      }
      return out;
    };
    // loose skin of the trunk (skin.js) after the camera and shadow tracks, which keep following the bones
    const sk = this.skin || SKIN_DEFAULT;
    const skinStats = sk.on && sk.amount > 0 ? looseSkin(jm, F, Sx, rig, this.man.guides.fps, sk) : null;
    if (this.clip) { this.clip.bJoints.destroy(); this.clip.bDyn.destroy(); }
    const bJoints = this.mk(jm);
    const bDyn = this.empty((F * Sx + 2) * DSTRIDE, this.BU.UNIFORM);
    const c = this.clip = { name, info, F, S: Sx, hold, bJoints, bDyn, cen: smooth(cen, 3, 5), head: smooth(head, 2, 5), skinStats };
    const trackXY = smooth(core, 3, 12), trackZ = smooth(core, 3, 24);       // 0.4 s horizontally, 0.8 s vertically
    c.track = new Float32Array(F * 3);
    for (let fr = 0; fr < F; fr++) { c.track[3 * fr] = trackXY[3 * fr]; c.track[3 * fr + 1] = trackXY[3 * fr + 1]; c.track[3 * fr + 2] = trackZ[3 * fr + 2]; }
    const restCen = [0, 0, 0];
    for (const q of rig) { restCen[0] += q[0] / rig.length; restCen[1] += q[1] / rig.length; restCen[2] += q[2] / rig.length; }
    c.restCen = restCen;
    c.restCore = [0, 1, 2].map((d) => 0.5 * (rig[5][d] + rig[8][d]));
    const device = this.device, bg = this.bg;
    c.bgDrive = device.createBindGroup({ layout: this.layDrive, entries: [
      { binding: 0, resource: { buffer: bDyn, size: 80 } }, { binding: 1, resource: { buffer: bJoints } },
      { binding: 2, resource: { buffer: this.bGjoints } }, { binding: 3, resource: { buffer: this.bGstatic } },
      { binding: 4, resource: { buffer: this.bCoef } }, { binding: 5, resource: { buffer: this.bGst } },
      { binding: 6, resource: { buffer: this.bTouch } }] });
    c.bgProj = device.createBindGroup({ layout: this.layProj, entries: [
      { binding: 0, resource: { buffer: bDyn, size: 80 } }, { binding: 1, resource: { buffer: this.bNbRow } },
      { binding: 2, resource: { buffer: this.bNbIds } }, { binding: 3, resource: { buffer: this.bNbW } },
      { binding: 4, resource: { buffer: this.bGstatic } }, { binding: 5, resource: { buffer: this.bCoef } },
      { binding: 6, resource: { buffer: this.bGst } }, { binding: 7, resource: { buffer: this.bSdfT } },
      { binding: 8, resource: { buffer: this.bSdfB } }, { binding: 9, resource: { buffer: this.bSdfU } }] });
    c.bgRot = bg(this.pRot, [[0, this.bRV], [1, bJoints], [2, this.bGjoints], [3, this.bGstatic], [4, this.bGst]]);
    c.bgBody = [0, 1].map((i) => bg(this.pBody, [[0, this.bFrame[i]], [1, bJoints], [2, this.bBodyV], [3, this.bBodyJ], [4, this.bBodyF], [5, this.bBodyRgb]]));
    c.bgLines = [0, 1].map((i) => bg(this.pLines, [[0, this.bFrame[i]], [1, bJoints], [2, this.bGjoints], [3, this.bGstatic], [4, this.bGst]]));
    this.makeStrandBG();
    this.makeStrandDynBG();
    if (this.dyn) this.writeDynUniforms();
    this.reset();
    this.kernelsLive = true;
    return { frames: F + hold };
  }
  get total() { return this.clip ? this.clip.F + this.clip.hold : 1; }
  reset() { this.frame = 0; this.simFrame = 0; this.pendingReset = true; this.goal = null; }
  seek(f) { this.goal = Math.max(0, Math.min(this.total - 1, f)); if (this.goal < this.simFrame) this.pendingReset = true; }
  // camera follow offset and shadow of a frame
  followOffset(fr) {
    const c = this.clip, i = Math.min(fr, c.F - 1);
    return [c.track[3 * i] - c.restCore[0], c.track[3 * i + 1] - c.restCore[1], (c.track[3 * i + 2] - c.restCore[2]) * 0.5];
  }

  // ------------------------------------------------------------------------------------------- targets
  makeStrandBG() {
    if (!this.texZ || !this.clip) return;
    this.bgStr = [0, 1].map((i) => this.bg(this.pStr, [[0, this.bFrame[i]], [1, this.clip.bJoints], [2, this.bStrand], [3, this.bGauss],
      [4, this.bGst], [5, this.bRecs], [6, this.bKeys], [7, this.bScratch], [8, this.texZ], [9, this.bSOut]]));
  }
  resize(W, H) {
    if (W === this.W && H === this.H) return;
    this.W = W; this.H = H; this.canvas.width = W; this.canvas.height = H;
    if (this.texZ) { this.texZ.destroy(); this.texDepth.destroy(); }
    this.texZ = this.device.createTexture({ size: [W, H], format: 'r32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.texDepth = this.device.createTexture({ size: [W, H], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.makeStrandBG();
  }

  // ------------------------------------------------------------------------------------------- one frame
  encodeReset(enc) {
    const K = this.K, c = this.clip;
    enc.clearBuffer(this.bGst, 0, S.GSTATE.tip * K * 4);
    enc.clearBuffer(this.bSState); enc.clearBuffer(this.bSOut); this.soutValid = false;     // strands at rest too
    const ps = enc.beginComputePass();
    ps.setPipeline(this.pRest); ps.setBindGroup(0, c.bgProj, [0]); ps.dispatchWorkgroups(Math.ceil(K / 64));
    ps.setPipeline(this.pDrive); ps.setBindGroup(0, c.bgDrive, [(c.F * c.S) * DSTRIDE]); ps.dispatchWorkgroups(Math.ceil(K / 64));
    ps.end();
  }
  encodeKernelRestart(enc, f) {
    const c = this.clip, K = this.K, fs = Math.min(f, c.F - 1), e = c.F * c.S + 1;
    this.device.queue.writeBuffer(c.bDyn, e * DSTRIDE + 12, new Uint32Array([fs * c.S + c.S - 1]));
    enc.clearBuffer(this.bGst, 0, S.GSTATE.tip * K * 4);
    const ps = enc.beginComputePass();
    ps.setPipeline(this.pRest); ps.setBindGroup(0, c.bgProj, [0]); ps.dispatchWorkgroups(Math.ceil(K / 64));
    ps.setPipeline(this.pDrive); ps.setBindGroup(0, c.bgDrive, [e * DSTRIDE]); ps.dispatchWorkgroups(Math.ceil(K / 64));
    ps.end();
  }
  encodeStep(enc, f) {
    const c = this.clip, K = this.K, fs = Math.min(f, c.F - 1);
    const ps = enc.beginComputePass();
    for (let s = 0; s < c.S; s++) {
      const off = (fs * c.S + s) * DSTRIDE;
      ps.setPipeline(this.pDrive); ps.setBindGroup(0, c.bgDrive, [off]); ps.dispatchWorkgroups(Math.ceil(K / 64));
      ps.setPipeline(this.pProj); ps.setBindGroup(0, c.bgProj, [off]); ps.dispatchWorkgroups(Math.ceil(K / 64));
    }
    ps.end();
  }
  // one frame of dynamics with the body held still in the pose of frame f (every substep uses the frame's last slot):
  // the fur keeps moving under the hand or the dryer while the motion is paused
  encodeHold(enc, f) {
    const c = this.clip, K = this.K, fs = Math.min(f, c.F - 1);
    const off = (fs * c.S + c.S - 1) * DSTRIDE;
    const ps = enc.beginComputePass();
    for (let s = 0; s < c.S; s++) {
      ps.setPipeline(this.pDrive); ps.setBindGroup(0, c.bgDrive, [off]); ps.dispatchWorkgroups(Math.ceil(K / 64));
      ps.setPipeline(this.pProj); ps.setBindGroup(0, c.bgProj, [off]); ps.dispatchWorkgroups(Math.ceil(K / 64));
    }
    ps.end();
  }
  // t: { mode: 0 none | 1 stroke | 2 dryer, active, p: [x, y, z], radius, dir: [x, y, z], dirW, strength, press, time, decay }
  setTouch(t) {
    const u = new Uint32Array(this.touchBuf), f = new Float32Array(this.touchBuf);
    u[0] = t.mode || 0; u[1] = t.active ? 1 : 0; f[2] = t.gshare ?? 1; u[3] = 0;
    const p = t.p || [0, 0, 0], d = t.dir || [0, 0, 1];
    f[4] = p[0]; f[5] = p[1]; f[6] = p[2]; f[7] = t.radius || 0.05;
    f[8] = d[0]; f[9] = d[1]; f[10] = d[2]; f[11] = t.dirW || 0;
    f[12] = t.strength || 0; f[13] = t.press || 0; f[14] = t.time || 0; f[15] = t.decay ?? 0.97;
    this.device.queue.writeBuffer(this.bTouch, 0, this.touchBuf);
  }
  // advance the simulation toward `target` (at most `budget` frames now); returns the frame shown
  advance(enc, target, dynOn, budget = 40, kernels = true) {
    if (this.goal !== null) target = this.goal;
    if (this.pendingReset) { this.encodeReset(enc); this.simFrame = 0; this.pendingReset = false; this.kernelsLive = true; }
    if (dynOn && kernels) {
      if (!this.kernelsLive) { this.encodeKernelRestart(enc, this.simFrame); this.kernelsLive = true; }
      while (this.simFrame < target && budget-- > 0) { this.simFrame++; this.encodeStep(enc, this.simFrame); }
    } else {
      // no kernel dynamics (fur dynamics off, or Strands only): the frame moves on; kernels restart when needed again
      if (dynOn) this.simFrame = Math.min(target, this.simFrame + budget); else this.simFrame = target;
      if (dynOn) this.kernelsLive = false;
    }
    this.frame = this.simFrame;
    if (this.goal !== null && this.simFrame >= this.goal) this.goal = null;
    return this.frame;
  }

  // o: { cam: {m, pos}, fy, dynOn, shOn, under, selected, solo, highlight, density, sides: [{x0, x1, view, colour, style, size, lod}], pick }
  render(o) {
    const device = this.device, c = this.clip, W = this.W, H = this.H, K = this.K, R = this.R, N = this.N;
    const enc = device.createCommandEncoder();
    const before = this.pendingReset ? 0 : this.simFrame;
    // Strands only does not use the kernels: they are not simulated unless the Kernels view shows them
    const kernels = !(this.smode === 2 && this.sdReady && this.bgSDyn) || o.sides.some((s) => s.kernels);
    this.advance(enc, o.target, o.dynOn, 40, kernels);
    let held = 0;
    if (o.dynOn && o.hold > 0 && !this.pendingReset) for (let i = 0; i < o.hold; i++) { if (kernels) this.encodeHold(enc, this.frame); held++; }
    const nsteps = Math.max(0, this.simFrame - before) + held;             // dynamics frames run in this draw
    if (o.camFn) o.cam = o.camFn(this.frame);       // the camera of the frame actually drawn (it was one frame behind)
    this.lastCam = o.cam; this.lastDrawnFrame = this.frame;
    const slot = Math.min(this.frame, c.F - 1) * c.S + c.S - 1;
    device.queue.writeBuffer(this.bRV, 0, new Uint32Array([K, slot, 0, 0]));
    device.queue.writeBuffer(this.bRV, 8, new Float32Array([this.satDeg * Math.PI / 180]));
    if (o.dynOn && kernels) {
      const ps = enc.beginComputePass();
      ps.setPipeline(this.pRot); ps.setBindGroup(0, c.bgRot); ps.dispatchWorkgroups(Math.ceil(K / 64));
      ps.end();
    }
    const smOn = o.dynOn && this.encodeStrandDyn(enc, nsteps, slot);   // per-strand dynamics (after the kernel rotations)
    // shadow of this frame
    const fi = Math.min(this.frame, c.F - 1);
    const lift = c.cen[3 * fi + 2] - c.restCen[2];
    const strength = 0.42 * Math.max(0.25, Math.min(1, 1 - lift / 0.5));
    const view = this.ctx.getCurrentTexture().createView();
    const fy = o.fy, zc = o.dist;
    let pickSide = -1, statsCopied = false;
    if (o.pick && !this.pickBusy) pickSide = o.sides.findIndex((s) => o.pick[0] >= s.x0 && o.pick[0] < s.x1);
    o.sides.forEach((sd, i) => {
      const fu = new Uint32Array(this.frameBuf), ff = new Float32Array(this.frameBuf);
      ff.set(o.cam.m, 0);
      ff[16] = o.cam.pos[0]; ff[17] = o.cam.pos[1]; ff[18] = o.cam.pos[2]; ff[19] = 0;
      ff[20] = fy; ff[21] = fy; ff[22] = W / 2; ff[23] = H / 2;
      ff[24] = W; ff[25] = H; ff[26] = 1 / W; ff[27] = 1 / H;
      ff[28] = Math.max(0.02, zc - 1.6); ff[29] = zc + 1.6; ff[30] = 0.01; ff[31] = 100;
      fu[32] = R; fu[33] = N; fu[34] = K; fu[35] = slot;
      fu[36] = o.dynOn ? 1 : 0; fu[37] = o.shOn ? 1 : 0; fu[38] = sd.body; fu[39] = sd.colour;
      fu[40] = sd.style; fu[41] = o.selected; fu[42] = o.solo && o.selected !== 0xffffffff ? 1 : 0; ff[43] = sd.lod;
      ff[44] = sd.size; ff[45] = this.man.config.overlap; ff[46] = this.man.config.length_floor; ff[47] = this.man.sh_scale;
      ff[48] = this.man.config.depth_abs_tolerance; ff[49] = this.man.config.depth_rel_tolerance; ff[50] = 0; ff[51] = 0;
      ff[52] = o.bgTop[0]; ff[53] = o.bgTop[1]; ff[54] = o.bgTop[2]; ff[55] = 1;
      fu[56] = o.highlight ? 1 : 0; fu[57] = sd.fur ? 1 : 0; fu[58] = o.bodyRgb ?? 0x8c8a8e; fu[59] = smOn ? 1 : 0;
      ff[60] = sd.x0; ff[61] = sd.x1; ff[62] = 0; ff[63] = 0;
      ff[64] = c.cen[3 * fi]; ff[65] = c.cen[3 * fi + 1]; ff[66] = this.groundZ; ff[67] = 0.42;
      ff[68] = strength; ff[69] = c.head[2 * fi]; ff[70] = c.head[2 * fi + 1]; ff[71] = 0;
      ff.set([...o.bgTop, 1], 72); ff.set([...o.bgBot, o.vignette ?? 1], 76);      // bg_bot.w = vignette
      device.queue.writeBuffer(this.bFrame[i], 0, this.frameBuf);
      const sx = Math.max(0, Math.floor(sd.x0)), sw = Math.max(1, Math.min(W, Math.ceil(sd.x1)) - sx);
      const first = i === 0;
      // backdrop, shadow, body
      {
        const rp = enc.beginRenderPass({
          colorAttachments: [{ view, clearValue: { r: 1, g: 1, b: 1, a: 1 }, loadOp: first ? 'clear' : 'load', storeOp: 'store' },
            { view: this.texZ.createView(), clearValue: { r: 1e30, g: 0, b: 0, a: 0 }, loadOp: first ? 'clear' : 'load', storeOp: 'store' }],
          depthStencilAttachment: { view: this.texDepth.createView(), depthClearValue: 1, depthLoadOp: first ? 'clear' : 'load', depthStoreOp: 'store' } });
        rp.setScissorRect(sx, 0, sw, H);
        rp.setPipeline(this.pBg); rp.setBindGroup(0, this.bgBg[i]); rp.draw(3);
        rp.setPipeline(this.pShadow); rp.setBindGroup(0, this.bgShadow[i]); rp.draw(6);
        rp.setPipeline(this.pBody); rp.setBindGroup(0, c.bgBody[i]); rp.draw(this.bodyF.length);
        rp.end();
      }
      if (sd.fur) {
        enc.clearBuffer(this.bScratch, 0, S.NBINS * 4);
        const ps = enc.beginComputePass();
        ps.setPipeline(this.pStr); ps.setBindGroup(0, this.bgStr[i]); ps.dispatchWorkgroups(Math.ceil(R / 64));
        ps.setBindGroup(0, this.bgScan);
        ps.setPipeline(this.pScan[0]); ps.dispatchWorkgroups(256);
        ps.setPipeline(this.pScan[1]); ps.dispatchWorkgroups(1);
        ps.setPipeline(this.pScan[2]); ps.dispatchWorkgroups(256);
        ps.setPipeline(this.pScat); ps.setBindGroup(0, this.bgScat); ps.dispatchWorkgroups(Math.ceil(N / 256));
        if (pickSide === i) {
          device.queue.writeBuffer(this.bPickBest, 0, new Uint32Array([0, 0xffffffff, 0, 0]));
          for (const p of [0, 1]) {
            const ab = new ArrayBuffer(16); new Uint32Array(ab)[0] = N; new Float32Array(ab).set([o.pick[0], o.pick[1]], 1); new Uint32Array(ab)[3] = p;
            device.queue.writeBuffer(this.bPickU[p], 0, ab);
          }
          for (const p of [0, 1]) { ps.setPipeline(this.pPick); ps.setBindGroup(0, this.bgPick[p]); ps.dispatchWorkgroups(Math.ceil(N / 256)); }
        }
        ps.end();
        if (!statsCopied && !this.statsBusy) { enc.copyBufferToBuffer(this.bIndirect, 0, this.rbStats, 0, 32); statsCopied = true; }
      }
      {
        const rp = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: 'load', storeOp: 'store' }],
          depthStencilAttachment: { view: this.texDepth.createView(), depthLoadOp: 'load', depthStoreOp: 'store' } });
        rp.setScissorRect(sx, 0, sw, H);
        if (sd.fur) { rp.setPipeline(this.pSplat); rp.setBindGroup(0, this.bgSplat[i]); rp.drawIndirect(this.bIndirect, 0); }
        if (sd.kernels) { rp.setPipeline(this.pLines); rp.setBindGroup(0, c.bgLines[i]); rp.draw(2 * K); }
        rp.end();
      }
    });
    const doPick = pickSide >= 0 && o.sides[pickSide].fur;
    if (doPick) enc.copyBufferToBuffer(this.bPickBest, 0, this.rbPick, 0, 16);
    const doSel = o.selected !== 0xffffffff && !this.selBusy;
    if (doSel) enc.copyBufferToBuffer(this.bScratch, (S.NBINS + 8) * 4, this.rbSel, 0, 32);
    // body depth around the pointer (canvas pixels), for the hand and the dryer
    this.lastFy = o.fy;
    const depthReq = o.depthAt ? this.depthCopy(enc, o.depthAt, o.cam, o.fy) : null;
    device.queue.submit([enc.finish()]);
    if (depthReq) this.depthRead(depthReq);
    if (!this.gpuBusy) {
      this.gpuBusy = true; const t0 = performance.now();
      device.queue.onSubmittedWorkDone().then(() => { const dt = performance.now() - t0; this.gpuMs = this.gpuMs ? 0.8 * this.gpuMs + 0.2 * dt : dt; this.gpuBusy = false; });
    }
    if (statsCopied) {
      this.statsBusy = true;
      this.rbStats.mapAsync(GPUMapMode.READ).then(() => { this.drawn = new Uint32Array(this.rbStats.getMappedRange())[1]; this.rbStats.unmap(); this.statsBusy = false; });
    }
    if (doPick) {
      this.pickBusy = true;
      this.rbPick.mapAsync(GPUMapMode.READ).then(() => {
        const gid = new Uint32Array(this.rbPick.getMappedRange())[1]; this.rbPick.unmap(); this.pickBusy = false;
        if (this.onPick) this.onPick(gid !== 0xffffffff && gid < N ? this.describe(gid) : null);
      });
    } else if (o.pick && pickSide >= 0 && this.onPick) this.onPick(null);
    if (doSel) {
      this.selBusy = true;
      this.rbSel.mapAsync(GPUMapMode.READ).then(() => {
        const a = new Float32Array(this.rbSel.getMappedRange());
        const s = { bendDeg: Math.hypot(a[3], a[4], a[5]) * 180 / Math.PI, gain: a[6] };
        this.rbSel.unmap(); this.selBusy = false;
        if (this.onSel) this.onSel(s);
      });
    }
    return this.frame;
  }

  // ------------------------------------------------------------------------------------------- per-strand dynamics
  // v20: mode 0 = kernels only (as before), 1 = kernels + per-strand touch and decorrelation, 2 = strands only
  // (shaders.js SDYN). Pipelines compile asynchronously; if they fail, the page stays in mode 0 (onStrandDyn(error)).
  initStrandDyn() {
    const R = this.R, BU = this.BU;
    this.bSChord = this.empty(R * 16);
    this.bSState = this.empty(R * 80);
    this.bSOut = this.empty(R * 16);
    this.bSD = this.empty(144, BU.UNIFORM);
    this.sdBuf = new ArrayBuffer(144);
    // velocity grid of mode 2 over the rest roots (the strands' skinning points), cubic cells
    const fq = new Float32Array(this.strandData.buffer, this.strandData.byteOffset, this.strandData.length);
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let r = 0; r < R; r++) for (let d = 0; d < 3; d++) {
      const v = fq[r * S.STRAND_STRIDE + 2 + d];
      if (v < lo[d]) lo[d] = v;
      if (v > hi[d]) hi[d] = v;
    }
    const G = 64, cell = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / (G - 2);
    const glo = lo.map((v) => v - cell), dims = [0, 1, 2].map((d) => Math.min(G + 2, Math.ceil((hi[d] - glo[d]) / cell) + 2));
    this.sgrid = { lo: glo, cell, dims };
    this.bGrid = this.empty(dims[0] * dims[1] * dims[2] * 16);
    this.smode = 0; this.sparams = { variation: 1, coh: 25 };
    this.sdReady = false; this.soutValid = false; this.sdError = null; this.clearStrand = false; this.onStrandDyn = null;
    this.makeStrandDyn();
  }
  async makeStrandDyn() {
    const dev = this.device;
    const mk = (code, label) => dev.createComputePipelineAsync({ label, layout: 'auto',
      compute: { module: dev.createShaderModule({ code, label }), entryPoint: 'main' } });
    try {
      const [pc, pg, pd] = await Promise.all([mk(S.SCHORD, 'schord'), mk(S.SGRID, 'sgrid'), mk(S.SDYN, 'sdyn')]);
      this.pSChord = pc; this.pSGrid = pg; this.pSDyn = pd;
      this.writeSD(1 / 30, 0);
      const enc = dev.createCommandEncoder();
      const ps = enc.beginComputePass();
      ps.setPipeline(pc); ps.setBindGroup(0, this.bg(pc, [[0, this.bSD], [1, this.bStrand], [2, this.bGauss], [3, this.bSChord]]));
      ps.dispatchWorkgroups(Math.ceil(this.R / 64));
      ps.end();
      dev.queue.submit([enc.finish()]);
      this.bgSGrid = this.bg(pg, [[0, this.bSD], [1, this.bStrand], [2, this.bSState], [3, this.bGrid]]);
      this.sdReady = true;
      this.makeStrandDynBG();
      if (this.onStrandDyn) this.onStrandDyn(null);
    } catch (e) {
      this.sdError = String((e && e.message) || e);
      console.error('per-strand dynamics unavailable:', e);
      if (this.onStrandDyn) this.onStrandDyn(this.sdError);
    }
  }
  makeStrandDynBG() {
    if (!this.sdReady || !this.clip) return;
    this.bgSDyn = this.bg(this.pSDyn, [[0, this.bSD], [1, this.clip.bJoints], [2, this.bStrand], [3, this.bGst], [4, this.bGstatic],
      [5, this.bSChord], [6, this.bSState], [7, this.bSOut], [8, this.bGrid], [9, this.bTouch]]);
  }
  // mode: 0 | 1 | 2; sp: { variation (0-2), coh (1/s, mode 2) }
  setStrandMode(mode, sp) {
    if (sp) this.sparams = { ...this.sparams, ...sp };
    if (mode !== undefined && mode !== this.smode) { this.smode = mode; this.clearStrand = true; }
  }
  writeSD(dt, slot) {
    const u = new Uint32Array(this.sdBuf), f = new Float32Array(this.sdBuf);
    const p = this.dyn ? this.dyn.p : {}, v = this.sparams.variation, g = this.man.guides.gravity, G = this.sgrid;
    const deg = Math.PI / 180;
    u[0] = this.R; u[1] = this.K; u[2] = slot; u[3] = this.smode;
    f[4] = dt; f[5] = 1 / dt; f[6] = 0.15 * v; f[7] = 0.04 * v;             // gain jitter, longest lag (s)
    f[8] = p.f_ref ?? 3; f[9] = p.h_ref ?? 0.05; f[10] = p.len_exp ?? 1.5; f[11] = p.f_min ?? 1.5;
    f[12] = p.f_max ?? 30; f[13] = p.zeta ?? 0.35; f[14] = p.drag ?? 0; f[15] = 0.2 * v;   // frequency/damping jitter
    f[16] = p.inertia_gain ?? 1; f[17] = p.inertia_limit_g ?? 2; f[18] = p.gravity_gain ?? 1; f[19] = this.sparams.coh;
    f[20] = g[0]; f[21] = g[1]; f[22] = g[2]; f[23] = (this.satDeg ?? 30) * deg;
    f[24] = (p.floor_give_deg ?? 0) * deg; f[25] = p.velocity_correction ?? 1; f[26] = Math.min(0.9, 0.45 * v); f[27] = 0.6 * v;
    f[28] = G.lo[0]; f[29] = G.lo[1]; f[30] = G.lo[2]; f[31] = G.cell;
    u[32] = G.dims[0]; u[33] = G.dims[1]; u[34] = G.dims[2]; u[35] = 0;
    this.device.queue.writeBuffer(this.bSD, 0, this.sdBuf);
  }
  // one step of the strands for the n dynamics frames of this draw; returns whether the strands read their rotation
  encodeStrandDyn(enc, n, slot) {
    if (!this.smode || !this.sdReady || !this.bgSDyn) return false;
    if (this.clearStrand) { enc.clearBuffer(this.bSState); enc.clearBuffer(this.bSOut); this.soutValid = false; this.clearStrand = false; }
    if (n <= 0 && this.soutValid) return true;
    const dt = n > 0 ? Math.min(n / this.man.guides.fps, 0.5) : 1 / 240;
    this.writeSD(dt, slot);
    if (this.smode === 2) {
      enc.clearBuffer(this.bGrid);
      const pg = enc.beginComputePass();
      pg.setPipeline(this.pSGrid); pg.setBindGroup(0, this.bgSGrid); pg.dispatchWorkgroups(Math.ceil(this.R / 64));
      pg.end();
    }
    const ps = enc.beginComputePass();
    ps.setPipeline(this.pSDyn); ps.setBindGroup(0, this.bgSDyn); ps.dispatchWorkgroups(Math.ceil(this.R / 64));
    ps.end();
    this.soutValid = true;
    return true;
  }

  // ------------------------------------------------------------------------------------------- body depth read-back
  // copy the body depth around canvas pixel `at` into the read-back buffer (null while a read is in flight)
  depthCopy(enc, at, cam, fy) {
    if (this.depthBusy || !this.texZ || !cam) return null;
    const W = this.W, H = this.H, h = (DEPTH_WIN - 1) / 2;
    const x0 = Math.max(0, Math.min(W - DEPTH_WIN, Math.round(at[0]) - h));
    const y0 = Math.max(0, Math.min(H - DEPTH_WIN, Math.round(at[1]) - h));
    enc.copyTextureToBuffer({ texture: this.texZ, origin: { x: x0, y: y0 } }, { buffer: this.rbDepth, bytesPerRow: 256, rowsPerImage: DEPTH_WIN },
      { width: DEPTH_WIN, height: DEPTH_WIN });
    this.depthBusy = true;
    return { x0, y0, px: at[0], py: at[1], cam, fy, W, H, frame: this.frame };
  }
  depthRead(req) {
    this.rbDepth.mapAsync(GPUMapMode.READ).then(() => {
      const a = new Float32Array(this.rbDepth.getMappedRange().slice(0));
      this.rbDepth.unmap(); this.depthBusy = false;
      // the body pixel nearest to the pointer inside the window (its depth is camera z)
      let best = null, bd = Infinity;
      for (let j = 0; j < DEPTH_WIN; j++) for (let i = 0; i < DEPTH_WIN; i++) {
        const z = a[j * 64 + i];
        if (!(z < 1e20) || z <= 0) continue;
        const dx = req.x0 + i - req.px, dy = req.y0 + j - req.py, d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = { z, u: req.x0 + i + 0.5, v: req.y0 + j + 0.5 }; }
      }
      if (this.onDepth) this.onDepth(best ? { ...best, ...this.unproject(req, best.u, best.v, best.z), gap: Math.sqrt(bd) } : null, req);
    }).catch(() => { this.depthBusy = false; });
  }
  // the body point under the pointer in the frame already on screen, without drawing a new frame (hover, press)
  readDepth(at) {
    if (this.depthBusy || !this.lastCam) return false;
    const enc = this.device.createCommandEncoder();
    const req = this.depthCopy(enc, at, this.lastCam, this.lastFy);
    this.device.queue.submit([enc.finish()]);
    if (req) this.depthRead(req);
    return !!req;
  }
  // world point of canvas pixel (u, v) at camera depth z, for the camera of a drawn frame (OpenCV, cx = W/2, cy = H/2)
  unproject(req, u, v, z) {
    const cam = req.cam, X = (u - req.W / 2) / req.fy * z, Y = (v - req.H / 2) / req.fy * z;
    const world = [0, 1, 2].map((i) => cam.pos[i] + X * cam.x[i] + Y * cam.y[i] + z * cam.f[i]);
    return { world, ray: cam.f };
  }
  // strand and Gaussian behind a Gaussian index (CPU tables)
  describe(gid) {
    const goff = this.goff; let lo = 0, hi = this.R;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (goff[m] <= gid) lo = m; else hi = m; }
    const r = lo, n = goff[r + 1] - goff[r];
    let L = 0; for (let i = goff[r]; i < goff[r + 1]; i++) L += f16(this.gauss[3 * i + 1] & 0xffff);
    const w1 = this.gauss[3 * gid + 1], w2 = this.gauss[3 * gid + 2];
    const gw = this.strandData.subarray(r * S.STRAND_STRIDE + 19, r * S.STRAND_STRIDE + 23);
    let k = 0; for (let i = 0; i < 16; i++) if ((gw[i >> 2] >>> (8 * (i & 3))) & 255) k++;
    return { gid, strand: r, i: gid - goff[r] + 1, n, len: L * 1e3, gl: f16(w1 & 0xffff) * 1e3, w: f16(w1 >>> 16) * 1e3, op: (w2 >>> 24) / 255, k };
  }

  // debugging: guide tips after simulating to frame f (compare with the H100 runs)
  async tipsAt(f, dynOn = true) {
    const enc = this.device.createCommandEncoder();
    this.encodeReset(enc);
    for (let i = 1; i <= f; i++) this.encodeStep(enc, i);
    const rb = this.device.createBuffer({ size: this.K * 12, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(this.bGst, S.GSTATE.tip * this.K * 4, rb, 0, this.K * 12);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange()).slice(); rb.unmap(); rb.destroy();
    this.simFrame = f; this.frame = f; this.pendingReset = false; this.goal = null;
    return out;
  }
}
