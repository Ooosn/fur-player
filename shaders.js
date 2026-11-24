// WGSL for the fur player. Ported from the CUDA kernels of scripts/dyn_core.py (guide_drive, guide_project,
// rest_distance), scripts/baked_fused.py (guide_rotvecs, frame_gaussians) and gsplat's projection/rasterization
// (antialiased mode). Layouts follow scripts/export_web_asset.py.

export const NB_BITS = 18;                 // depth bins of the counting sort
export const NBINS = 1 << NB_BITS;
export const SENTINEL = 0xffffffff;
export const STRAND_STRIDE = 32;           // u32 per strand in strandData
export const GSTATE = {                    // f32 offsets (in units of K) of the guide state sections
  e: 0, ed: 3, xstar: 6, vstar: 9, Rk: 12, ei: 21, bn: 27, aux: 33, tip: 37, allowed: 40, phiw: 41, total: 44,
};

const COMMON = /* wgsl */`
fn polar3(Min: mat3x3<f32>) -> mat3x3<f32> {
  // rows of the row-major 3x3 are the columns of this mat3x3 (we store R as rows: r0 = a, r1 = b, r2 = c)
  var a = Min[0]; var b = Min[1]; var c = Min[2];
  for (var it = 0; it < 5; it++) {
    let k0 = cross(b, c); let k1 = cross(c, a); let k2 = cross(a, b);
    let det = dot(a, k0);
    a = 0.5 * (a + k0 / det); b = 0.5 * (b + k1 / det); c = 0.5 * (c + k2 / det);
  }
  return mat3x3<f32>(a, b, c);
}
// row-major helpers: R rows r0,r1,r2 stored as columns of mat3x3
fn rmul(R: mat3x3<f32>, v: vec3<f32>) -> vec3<f32> { return vec3<f32>(dot(R[0], v), dot(R[1], v), dot(R[2], v)); }
fn rtmul(R: mat3x3<f32>, v: vec3<f32>) -> vec3<f32> { return R[0] * v.x + R[1] * v.y + R[2] * v.z; }
fn rodrigues(k: vec3<f32>, ang: f32, v: vec3<f32>) -> vec3<f32> {
  let s = sin(ang); let c = cos(ang);
  return v * c + cross(k, v) * s + k * (dot(k, v) * (1.0 - c));
}
fn unorm16(w: u32, half: u32) -> f32 { return f32((w >> (16u * half)) & 0xffffu) / 65535.0; }
fn byte(w: u32, i: u32) -> u32 { return (w >> (8u * i)) & 0xffu; }
fn sbyte(w: u32, i: u32) -> f32 { return f32(bitcast<i32>(w << (24u - 8u * i)) >> 24u); }
fn shalf(w: u32, h: u32) -> f32 { return f32(bitcast<i32>(w << (16u - 16u * h)) >> 16u); }
fn hash32(x0: u32) -> u32 {
  var x = x0; x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u; return x;
}
fn hashcol(x: u32) -> vec3<f32> {
  let h = hash32(x);
  return vec3<f32>(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u)) / 255.0 * 0.8 + 0.2;
}
fn turbo(t0: f32) -> vec3<f32> {
  let t = clamp(t0, 0.0, 1.0);
  let r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  let g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  let b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}
`;

// joints: per slot 39 x 12 f32 (row-major 3x4)
const JOINTS = /* wgsl */`
fn jrot(slot: u32, j: u32) -> mat3x3<f32> {
  let b = (slot * 39u + j) * 12u;
  return mat3x3<f32>(vec3<f32>(joints[b], joints[b + 1u], joints[b + 2u]),
                     vec3<f32>(joints[b + 4u], joints[b + 5u], joints[b + 6u]),
                     vec3<f32>(joints[b + 8u], joints[b + 9u], joints[b + 10u]));
}
fn jtr(slot: u32, j: u32) -> vec3<f32> {
  let b = (slot * 39u + j) * 12u;
  return vec3<f32>(joints[b + 3u], joints[b + 7u], joints[b + 11u]);
}
// top-8 joints: ids in words w0,w1 (u8 x 8), weights in w2..w5 (u16 x 8)
struct Skin { M: mat3x3<f32>, base: vec3<f32> };
fn skin8(slot: u32, ids0: u32, ids1: u32, w0: u32, w1: u32, w2: u32, w3: u32, q: vec3<f32>) -> Skin {
  var M = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  var base = vec3<f32>(0.0);
  let ws = array<u32, 4>(w0, w1, w2, w3);
  for (var i = 0u; i < 8u; i++) {
    let id = select(byte(ids1, i - 4u), byte(ids0, i), i < 4u);
    let w = unorm16(ws[i / 2u], i % 2u);
    if (w > 0.0) {
      let R = jrot(slot, id);
      M = mat3x3<f32>(M[0] + w * R[0], M[1] + w * R[1], M[2] + w * R[2]);
      base += w * (rmul(R, q) + jtr(slot, id));
    }
  }
  return Skin(M, base);
}
`;

// ----------------------------------------------------------------------------------------------------------------
// guide dynamics
export const DYN_STRUCT = /* wgsl */`
struct Dyn {
  K: u32, J: u32, init: u32, slot: u32,
  inv_h: f32, inv_g981: f32, inertia_gain: f32, inertia_limit_g: f32,
  gravity_gain: f32, gx: f32, gy: f32, gz: f32,
  coh: f32, vc: f32, cone_c: f32, cone_s: f32,
  cohesion_on: u32, collide: u32, floor_on: u32, margin: f32,
};
`;
const SDF = /* wgsl */`
struct SdfInfo { nx: i32, ny: i32, nz: i32, bx: i32, by: i32, bz: i32, step: f32, band: f32,
                 lox: f32, loy: f32, loz: f32, ex: f32, ey: f32, ez: f32, pad0: f32, pad1: f32 };
fn sdf_voxel(x: i32, y: i32, z: i32) -> f32 {
  let b = (x >> 3) * sdf.by * sdf.bz + (y >> 3) * sdf.bz + (z >> 3);
  let t = sdf_table[b];
  if (t == 0xffffffffu) { return sdf.band; }
  if (t == 0xfffffffeu) { return -sdf.band; }
  let li = u32(((x & 7) * 8 + (y & 7)) * 8 + (z & 7));
  let idx = t * 512u + li;
  let wd = sdf_bricks[idx >> 1u];
  return shalf(wd, idx & 1u) * sdf.step;
}
// sim_collie.Field / sdf_fetch: align_corners, border clamp, corners outside the grid dropped
fn sdf_dist(q: vec3<f32>) -> f32 {
  var ix = ((((q.x - sdf.lox) / sdf.ex) * 2.0 - 1.0 + 1.0) / 2.0) * f32(sdf.nx - 1);
  var iy = ((((q.y - sdf.loy) / sdf.ey) * 2.0 - 1.0 + 1.0) / 2.0) * f32(sdf.ny - 1);
  var iz = ((((q.z - sdf.loz) / sdf.ez) * 2.0 - 1.0 + 1.0) / 2.0) * f32(sdf.nz - 1);
  ix = min(f32(sdf.nx - 1), max(ix, 0.0)); iy = min(f32(sdf.ny - 1), max(iy, 0.0)); iz = min(f32(sdf.nz - 1), max(iz, 0.0));
  let x0 = i32(floor(ix)); let y0 = i32(floor(iy)); let z0 = i32(floor(iz));
  let fx = ix - f32(x0); let fy = iy - f32(y0); let fz = iz - f32(z0);
  var out = 0.0;
  for (var c = 0; c < 8; c++) {
    let dx = c & 1; let dy = (c >> 1) & 1; let dz = (c >> 2) & 1;
    let x = x0 + dx; let y = y0 + dy; let z = z0 + dz;
    if (x < sdf.nx && y < sdf.ny && z < sdf.nz) {
      let w = select(1.0 - fx, fx, dx == 1) * select(1.0 - fy, fy, dy == 1) * select(1.0 - fz, fz, dz == 1);
      out += w * sdf_voxel(x, y, z);
    }
  }
  return out;
}
fn sdf_grad(q: vec3<f32>) -> vec3<f32> {
  let hx = sdf.ex / f32(sdf.nx - 1); let hy = sdf.ey / f32(sdf.ny - 1); let hz = sdf.ez / f32(sdf.nz - 1);
  return vec3<f32>((sdf_dist(q + vec3<f32>(hx, 0.0, 0.0)) - sdf_dist(q - vec3<f32>(hx, 0.0, 0.0))) / (2.0 * hx),
                   (sdf_dist(q + vec3<f32>(0.0, hy, 0.0)) - sdf_dist(q - vec3<f32>(0.0, hy, 0.0))) / (2.0 * hy),
                   (sdf_dist(q + vec3<f32>(0.0, 0.0, hz)) - sdf_dist(q - vec3<f32>(0.0, 0.0, hz))) / (2.0 * hz));
}
`;
// guide_static: q(0..2) proot(3..5) ptip(6..8) Grel1(9..11) n1(12..14) r_dir(15..17) t_rest(18..20) rest_chord(21..23)
// Lg 24 hlen 25 ca0 26 sa0 27; guide_coef: om 0, A 1..4, sleep_e 5, sleep_v 6, ca 7, sa 8 (stride 10)
const GDEF = /* wgsl */`
fn gs3(k: u32, o: u32) -> vec3<f32> { let b = k * 28u + o; return vec3<f32>(gstatic[b], gstatic[b + 1u], gstatic[b + 2u]); }
fn gs1(k: u32, o: u32) -> f32 { return gstatic[k * 28u + o]; }
fn st3(sec: u32, k: u32) -> vec3<f32> { let b = sec * D.K + 3u * k; return vec3<f32>(gst[b], gst[b + 1u], gst[b + 2u]); }
fn put3(sec: u32, k: u32, v: vec3<f32>) { let b = sec * D.K + 3u * k; gst[b] = v.x; gst[b + 1u] = v.y; gst[b + 2u] = v.z; }
`;

// interaction (v18/v19): a hand stroking the fur, as an external acceleration on the guides (the v18 hair dryer was
// dropped by the author).
// Added to the equilibrium of each guide's oscillator, so the closed-form step stays exact and the fur springs back
// with its own dynamics when the force stops. aux[3] of a guide = how strongly it is being touched (decays after).
export const TOUCH_STRUCT = /* wgsl */`
struct Touch {
  mode: u32, touching: u32, gshare: f32, pad1: u32,  // mode 0 none, 1 stroke; touching = pressed on the body;
                                                    // gshare = share of the hand acting on the kernels
  p: vec4<f32>,      // hit point on the body (world), radius of the hand (m)
  dir: vec4<f32>,    // direction of the hand (unit), share of combing vs pressing
  s: vec4<f32>,      // strength (m/s^2), pressing, time (s), decay of the touch weight per substep
};
`;

export const DRIVE = COMMON + DYN_STRUCT + TOUCH_STRUCT + /* wgsl */`
@group(0) @binding(0) var<uniform> D: Dyn;
@group(0) @binding(1) var<storage, read> joints: array<f32>;
@group(0) @binding(2) var<storage, read> gjoints: array<u32>;
@group(0) @binding(3) var<storage, read> gstatic: array<f32>;
@group(0) @binding(4) var<storage, read> gcoef: array<f32>;
@group(0) @binding(5) var<storage, read_write> gst: array<f32>;
@group(0) @binding(6) var<uniform> T: Touch;
` + JOINTS + GDEF + /* wgsl */`
// external acceleration on guide k (world), and its touch weight; root = world root, nw = world normal at the root
fn touch_force(k: u32, root: vec3<f32>, nw: vec3<f32>) -> vec4<f32> {
  if (T.mode == 1u) {                                   // hand: comb along its motion, press toward the body
    let w = 1.0 - smoothstep(0.45 * T.p.w, T.p.w, distance(root, T.p.xyz));
    if (w <= 0.0) { return vec4<f32>(0.0); }
    var t = T.dir.xyz - dot(T.dir.xyz, nw) * nw;
    t = t / max(length(t), 1e-6);
    let a = T.s.x * w * (T.dir.w * t - T.s.y * nw);
    return vec4<f32>(a, w);
  }
  return vec4<f32>(0.0);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= D.K) { return; }
  let g6 = k * 6u;
  let sk = skin8(D.slot, gjoints[g6], gjoints[g6 + 1u], gjoints[g6 + 2u], gjoints[g6 + 3u], gjoints[g6 + 4u], gjoints[g6 + 5u], gs3(k, 0u));
  let R = polar3(sk.M);
  let rb = (${GSTATE.Rk}u * D.K) + 9u * k;
  gst[rb] = R[0].x; gst[rb + 1u] = R[0].y; gst[rb + 2u] = R[0].z;
  gst[rb + 3u] = R[1].x; gst[rb + 4u] = R[1].y; gst[rb + 5u] = R[1].z;
  gst[rb + 6u] = R[2].x; gst[rb + 7u] = R[2].y; gst[rb + 8u] = R[2].z;
  let xs = sk.base + rmul(R, gs3(k, 9u));
  if (D.init == 1u) {
    put3(${GSTATE.xstar}u, k, xs); put3(${GSTATE.vstar}u, k, vec3<f32>(0.0));
    return;
  }
  let vs = (xs - st3(${GSTATE.xstar}u, k)) * D.inv_h;
  var d = vs - st3(${GSTATE.vstar}u, k);
  let acc = length(d) * D.inv_h * D.inv_g981;
  if (D.inertia_limit_g > 0.0) { d *= min((1.0 / max(acc, 1e-9)) * D.inertia_limit_g, 1.0); }
  let v = st3(${GSTATE.ed}u, k) - D.inertia_gain * d;
  put3(${GSTATE.xstar}u, k, xs); put3(${GSTATE.vstar}u, k, vs);
  let o = gcoef[k * 10u];
  let o2 = o * o;
  let g = vec3<f32>(D.gx, D.gy, D.gz);
  var ep = (D.gravity_gain * (g - rmul(R, g))) / o2;
  // interaction: shift the equilibrium by a / omega^2, saturated below the guide's length
  let ab0 = ${GSTATE.aux}u * D.K + 4u * k;
  var tw = gst[ab0 + 3u] * T.s.w;
  if (T.touching == 1u) {
    let root = sk.base + rmul(R, gs3(k, 3u) - gs3(k, 0u));
    let nw = rmul(R, gs3(k, 12u));
    let fa = touch_force(k, root, nw);
    if (fa.w > 0.0) {
      var off = T.gshare * fa.xyz / o2;
      let lo = length(off);
      let cap = 0.75 * gs1(k, 24u);
      if (lo > 1e-9) { off *= cap * tanh(lo / cap) / lo; }
      ep += off;
      tw = max(tw, fa.w);
    }
  }
  gst[ab0 + 3u] = tw;
  let de = st3(${GSTATE.e}u, k) - ep;
  let a11 = gcoef[k * 10u + 1u]; let a12 = gcoef[k * 10u + 2u]; let a21 = gcoef[k * 10u + 3u]; let a22 = gcoef[k * 10u + 4u];
  let r0 = a11 * de + a12 * v + ep;
  let r1 = a21 * de + a22 * v;
  let hl = gs1(k, 25u);
  let eb = ${GSTATE.ei}u * D.K + 6u * k;
  let bb = ${GSTATE.bn}u * D.K + 6u * k;
  gst[eb] = r0.x; gst[eb + 1u] = r0.y; gst[eb + 2u] = r0.z; gst[eb + 3u] = r1.x; gst[eb + 4u] = r1.y; gst[eb + 5u] = r1.z;
  gst[bb] = r0.x / hl; gst[bb + 1u] = r0.y / hl; gst[bb + 2u] = r0.z / hl;
  gst[bb + 3u] = r1.x / hl; gst[bb + 4u] = r1.y / hl; gst[bb + 5u] = r1.z / hl;
  let ab = ${GSTATE.aux}u * D.K + 4u * k;
  gst[ab] = length(ep); gst[ab + 1u] = length(d); gst[ab + 2u] = acc;
}
`;

export const PROJECT = COMMON + DYN_STRUCT + SDF + /* wgsl */`
@group(0) @binding(0) var<uniform> D: Dyn;
@group(0) @binding(1) var<storage, read> nbrow: array<u32>;
@group(0) @binding(2) var<storage, read> nbids: array<u32>;
@group(0) @binding(3) var<storage, read> nbw: array<f32>;
@group(0) @binding(4) var<storage, read> gstatic: array<f32>;
@group(0) @binding(5) var<storage, read> gcoef: array<f32>;
@group(0) @binding(6) var<storage, read_write> gst: array<f32>;
@group(0) @binding(7) var<storage, read> sdf_table: array<u32>;
@group(0) @binding(8) var<storage, read> sdf_bricks: array<u32>;
@group(0) @binding(9) var<uniform> sdf: SdfInfo;
` + GDEF + /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= D.K) { return; }
  var v: array<f32, 6>;
  let bb = ${GSTATE.bn}u * D.K;
  if (D.cohesion_on == 1u) {
    var s = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (var i = nbrow[k]; i < nbrow[k + 1u]; i++) {
      let q = bb + 6u * nbids[i];
      let w = nbw[i];
      for (var c = 0u; c < 6u; c++) { s[c] += w * gst[q + c]; }
    }
    let hl = gs1(k, 25u);
    for (var c = 0u; c < 6u; c++) {
      let b = gst[bb + 6u * k + c];
      v[c] = (b + D.coh * (s[c] - b)) * hl;
    }
  } else {
    for (var c = 0u; c < 6u; c++) { v[c] = gst[${GSTATE.ei}u * D.K + 6u * k + c]; }
  }
  let e = vec3<f32>(v[0], v[1], v[2]);
  var d = vec3<f32>(v[3], v[4], v[5]);
  let isact = max(max(abs(e.x), abs(e.y)), abs(e.z)) > 0.0;
  let rb = ${GSTATE.Rk}u * D.K + 9u * k;
  let R = mat3x3<f32>(vec3<f32>(gst[rb], gst[rb + 1u], gst[rb + 2u]), vec3<f32>(gst[rb + 3u], gst[rb + 4u], gst[rb + 5u]),
                      vec3<f32>(gst[rb + 6u], gst[rb + 7u], gst[rb + 8u]));
  let p0 = gs3(k, 3u); let p1 = gs3(k, 6u);
  var y = p1 + rtmul(R, e);
  if (D.collide == 1u) {
    let f0 = sdf_dist(y);
    let al = gst[${GSTATE.allowed}u * D.K + k];
    if (f0 < al) {
      let n = sdf_grad(y);
      let nn = max(length(n), 1e-12);
      y += n / nn * (al - f0);
    }
  }
  var c = y - p0;
  c = c / max(length(c), 1e-12);
  if (D.floor_on == 1u) {
    let nr = gs3(k, 12u);
    let el = dot(c, nr);
    var t = c - el * nr;
    if (!(length(t) > 1e-6)) { t = gs3(k, 18u); }
    // under a hand or in the air jet the fur may lie flatter than at rest (touch weight aux[3], decays after)
    let tw = gst[${GSTATE.aux}u * D.K + 4u * k + 3u];
    let ca = mix(gcoef[k * 10u + 7u], min(gcoef[k * 10u + 7u], 0.06), tw);
    let sa = sqrt(max(1.0 - ca * ca, 0.0));
    if (el < ca) { c = ca * nr + sa * (t / max(length(t), 1e-12)); }
  }
  {
    let r = gs3(k, 15u);
    let cc = dot(r, c);
    if (cc < D.cone_c) {
      var q = c - cc * r;
      q = q / max(length(q), 1e-12);
      let l = D.cone_c * r + D.cone_s * q;
      c = l / max(length(l), 1e-12);
    }
  }
  y = p0 + c * gs1(k, 24u);
  var n = e;
  if (isact) { n = rmul(R, y - p1); }
  d = d + D.vc * (n - e) * D.inv_h;
  let se = gcoef[k * 10u + 5u];
  let ab = ${GSTATE.aux}u * D.K + 4u * k;
  let awake = length(n) > se || length(d) > gcoef[k * 10u + 6u] || gst[ab] > se || gst[ab + 1u] > 0.0;
  put3(${GSTATE.e}u, k, select(vec3<f32>(0.0), n, awake));
  put3(${GSTATE.ed}u, k, select(vec3<f32>(0.0), d, awake));
  put3(${GSTATE.tip}u, k, select(p1, y, isact && awake));
}
`;

export const REST_DISTANCE = COMMON + DYN_STRUCT + SDF + /* wgsl */`
@group(0) @binding(0) var<uniform> D: Dyn;
@group(0) @binding(4) var<storage, read> gstatic: array<f32>;
@group(0) @binding(6) var<storage, read_write> gst: array<f32>;
@group(0) @binding(7) var<storage, read> sdf_table: array<u32>;
@group(0) @binding(8) var<storage, read> sdf_bricks: array<u32>;
@group(0) @binding(9) var<uniform> sdf: SdfInfo;
` + GDEF + /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= D.K) { return; }
  gst[${GSTATE.allowed}u * D.K + k] = min(sdf_dist(gs3(k, 6u)), D.margin);
  put3(${GSTATE.tip}u, k, gs3(k, 6u));
}
`;

// guide rotation vectors for the strands (baked_fused.guide_rotvecs), from the render joints
export const ROTVEC = COMMON + DYN_STRUCT + /* wgsl */`
struct RV { K: u32, slot: u32, sat: f32, pad: f32 };
@group(0) @binding(0) var<uniform> P: RV;
@group(0) @binding(1) var<storage, read> joints: array<f32>;
@group(0) @binding(2) var<storage, read> gjoints: array<u32>;
@group(0) @binding(3) var<storage, read> gstatic: array<f32>;
@group(0) @binding(4) var<storage, read_write> gst: array<f32>;
` + JOINTS + /* wgsl */`
fn gs3(k: u32, o: u32) -> vec3<f32> { let b = k * 28u + o; return vec3<f32>(gstatic[b], gstatic[b + 1u], gstatic[b + 2u]); }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= P.K) { return; }
  let g6 = k * 6u;
  let sk = skin8(P.slot, gjoints[g6], gjoints[g6 + 1u], gjoints[g6 + 2u], gjoints[g6 + 3u], gjoints[g6 + 4u], gjoints[g6 + 5u], gs3(k, 0u));
  let R = polar3(sk.M);
  let a0 = gs3(k, 21u);
  let tb = ${GSTATE.tip}u * P.K + 3u * k;
  let b0 = vec3<f32>(gst[tb], gst[tb + 1u], gst[tb + 2u]) - gs3(k, 3u);
  let ob = ${GSTATE.phiw}u * P.K + 3u * k;
  if (all(b0 == a0)) { gst[ob] = 0.0; gst[ob + 1u] = 0.0; gst[ob + 2u] = 0.0; return; }
  let a = a0 / max(length(a0), 1e-12);
  let b = b0 / max(length(b0), 1e-12);
  let cx = cross(a, b);
  let s = length(cx);
  let ang0 = atan2(s, dot(a, b));
  var p = cx / max(s, 1e-12) * ang0;
  if (P.sat > 0.0) {
    let ang = length(p);
    let sat = P.sat * (1.0 + 0.8 * gst[${GSTATE.aux}u * P.K + 4u * k + 3u]);    // touched fur may bend further
    p *= sat * tanh(ang / sat) / max(ang, 1e-9);
  }
  let w = rmul(R, p);
  gst[ob] = w.x; gst[ob + 1u] = w.y; gst[ob + 2u] = w.z;
}
`;

// ----------------------------------------------------------------------------------------------------------------
// per-strand dynamics (v20). Each strand gets its own damped single-mode oscillator: a tip offset e (world) that bends
// the whole strand about its root with the same profile as the kernels (one mode per strand, so it cannot kink).
//   mode 1 (kernels + strands): the kernels still move the coat; on top, each strand responds to the hand by itself and
//          the kernel motion is decorrelated per strand (jittered blend weights, gain, a small time lag).
//   mode 2 (strands only): no kernels; each strand is driven by the inertia of its own rigid tip, the turn of gravity in
//          the body frame and the hand; neighbours are coupled by velocity friction on a coarse grid over the rest roots.
// Per strand: state 5 x vec4 (e, init | ed, touch | previous rigid tip | its velocity | lagged kernel rotation),
// out 1 x vec4 (rotation in the root rest frame, touch), chord 1 x vec4 (rest chord, arc length).
export const SD_STRUCT = /* wgsl */`
struct SD {
  R: u32, K: u32, slot: u32, mode: u32,
  dt: f32, inv_dt: f32, gain_jit: f32, lag_max: f32,
  f_ref: f32, h_ref: f32, len_exp: f32, f_min: f32,
  f_max: f32, zeta: f32, drag: f32, freq_jit: f32,
  inertia_gain: f32, inertia_limit_g: f32, gravity_gain: f32, coh_rate: f32,
  gx: f32, gy: f32, gz: f32, sat: f32,
  give: f32, vc: f32, catch_jit: f32, weight_jit: f32,
  glo: vec4<f32>,      // grid origin (rest frame) xyz, cell size
  gdim: vec4<u32>,     // cells along x, y, z, 0
};
const SSCALE: f32 = 10000.0;          // fixed point of the grid velocities (0.1 mm/s)
`;

// rest chord and arc length of every strand, once
export const SCHORD = SD_STRUCT + /* wgsl */`
@group(0) @binding(0) var<uniform> P: SD;
@group(0) @binding(1) var<storage, read> sdata: array<u32>;
@group(0) @binding(2) var<storage, read> gauss: array<u32>;
@group(0) @binding(3) var<storage, read_write> schord: array<vec4<f32>>;
fn oct_dec(u: u32) -> vec3<f32> {
  var a = f32(u & 0xffffu) / 65535.0 * 2.0 - 1.0;
  var b = f32(u >> 16u) / 65535.0 * 2.0 - 1.0;
  let z = 1.0 - abs(a) - abs(b);
  let t = max(-z, 0.0);
  a += select(t, -t, a >= 0.0);
  b += select(t, -t, b >= 0.0);
  return normalize(vec3<f32>(a, b, z));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= P.R) { return; }
  let sb = r * ${STRAND_STRIDE}u;
  let g0 = sdata[sb]; let cnt = sdata[sb + 1u];
  var c = vec3<f32>(0.0); var L = 0.0;
  for (var i = 0u; i < cnt; i++) {
    let g = g0 + i;
    let lw = unpack2x16float(gauss[3u * g + 1u]);
    c += oct_dec(gauss[3u * g]) * lw.x;
    L += lw.x;
  }
  schord[r] = vec4<f32>(c, L);
}
`;

// mode 2: last frame's strand velocities summed into a grid over the rest roots (nearest cell, fixed point)
export const SGRID = SD_STRUCT + /* wgsl */`
@group(0) @binding(0) var<uniform> P: SD;
@group(0) @binding(1) var<storage, read> sdata: array<u32>;
@group(0) @binding(2) var<storage, read> sstate: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> grid: array<atomic<i32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= P.R) { return; }
  let s1 = sstate[5u * r + 1u];
  if (sstate[5u * r].w == 0.0) { return; }
  let sb = r * ${STRAND_STRIDE}u;
  let q = vec3<f32>(bitcast<f32>(sdata[sb + 2u]), bitcast<f32>(sdata[sb + 3u]), bitcast<f32>(sdata[sb + 4u]));
  let hi = vec3<i32>(P.gdim.xyz) - vec3<i32>(1);
  let c = clamp(vec3<i32>(floor((q - P.glo.xyz) / P.glo.w)), vec3<i32>(0), hi);
  let ci = (u32(c.x) * P.gdim.y + u32(c.y)) * P.gdim.z + u32(c.z);
  let v = vec3<i32>(round(clamp(s1.xyz, vec3<f32>(-20.0), vec3<f32>(20.0)) * SSCALE));
  atomicAdd(&grid[4u * ci], v.x);
  atomicAdd(&grid[4u * ci + 1u], v.y);
  atomicAdd(&grid[4u * ci + 2u], v.z);
  atomicAdd(&grid[4u * ci + 3u], 1);
}
`;

export const SDYN = COMMON + SD_STRUCT + TOUCH_STRUCT + /* wgsl */`
@group(0) @binding(0) var<uniform> P: SD;
@group(0) @binding(1) var<storage, read> joints: array<f32>;
@group(0) @binding(2) var<storage, read> sdata: array<u32>;
@group(0) @binding(3) var<storage, read> gst: array<f32>;
@group(0) @binding(4) var<storage, read> gstatic: array<f32>;
@group(0) @binding(5) var<storage, read> schord: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> sstate: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> sout: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> grid: array<i32>;
@group(0) @binding(9) var<uniform> T: Touch;
` + JOINTS + /* wgsl */`
fn h01(x: u32) -> f32 { return f32(hash32(x)) * (1.0 / 4294967296.0); }
// rotation vector turning direction a into direction b
fn rot_between(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
  let an = a / max(length(a), 1e-12);
  let bn = b / max(length(b), 1e-12);
  let cx = cross(an, bn);
  let s = length(cx);
  if (s < 1e-9) { return vec3<f32>(0.0); }
  return cx / s * atan2(s, dot(an, bn));
}
fn sat_rot(p: vec3<f32>, sat: f32) -> vec3<f32> {
  let a = length(p);
  if (a < 1e-9 || sat <= 0.0) { return p; }
  return p * (sat * tanh(a / sat) / a);
}
// average strand velocity around rest point q (trilinear over the grid cells, weighted by their counts)
fn grid_avg(q: vec3<f32>) -> vec4<f32> {
  let x = (q - P.glo.xyz) / P.glo.w - vec3<f32>(0.5);
  let b = floor(x);
  let fr = x - b;
  let hi = vec3<i32>(P.gdim.xyz);
  var sv = vec3<f32>(0.0);
  var sn = 0.0;
  for (var k = 0u; k < 8u; k++) {
    let o = vec3<i32>(i32(k & 1u), i32((k >> 1u) & 1u), i32((k >> 2u) & 1u));
    let c = vec3<i32>(b) + o;
    if (any(c < vec3<i32>(0)) || any(c >= hi)) { continue; }
    let w = select(1.0 - fr.x, fr.x, o.x == 1) * select(1.0 - fr.y, fr.y, o.y == 1) * select(1.0 - fr.z, fr.z, o.z == 1);
    let ci = (u32(c.x) * P.gdim.y + u32(c.y)) * P.gdim.z + u32(c.z);
    sv += w * vec3<f32>(f32(grid[4u * ci]), f32(grid[4u * ci + 1u]), f32(grid[4u * ci + 2u])) / SSCALE;
    sn += w * f32(grid[4u * ci + 3u]);
  }
  if (sn <= 1e-6) { return vec4<f32>(0.0); }
  return vec4<f32>(sv / sn, sn);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= P.R) { return; }
  let sb = r * ${STRAND_STRIDE}u;
  let q = vec3<f32>(bitcast<f32>(sdata[sb + 2u]), bitcast<f32>(sdata[sb + 3u]), bitcast<f32>(sdata[sb + 4u]));
  let sk = skin8(P.slot, sdata[sb + 5u], sdata[sb + 6u], sdata[sb + 7u], sdata[sb + 8u], sdata[sb + 9u], sdata[sb + 10u], q);
  let M = polar3(sk.M);
  let root = sk.base;
  let ch = schord[r];
  let L = max(ch.w, 1e-4);
  let cw = rmul(M, ch.xyz);                     // rest chord, posed (world)
  let clen = max(length(cw), 1e-6);
  let xref = root + cw;                         // rigid tip (world)
  // this strand's own constants: frequency from its length (the kernels' rule), jittered; damping; catch; side
  let hA = h01(r * 8u + 1u); let hB = h01(r * 8u + 2u); let hC = h01(r * 8u + 3u);
  let hD = h01(r * 8u + 4u); let hE = h01(r * 8u + 5u); let hF = h01(r * 8u + 6u);
  let f = clamp(P.f_ref * pow(P.h_ref / L, P.len_exp), P.f_min, P.f_max) * (1.0 + P.freq_jit * (2.0 * hA - 1.0));
  let om = 6.2831853 * f;
  let o2 = om * om;
  let zeta = clamp(P.zeta * (1.0 + P.freq_jit * (2.0 * hB - 1.0)) + P.drag / (2.0 * om), 0.02, 0.9);
  let topg = sdata[sb + 11u] & 0xffffu;
  let n0 = vec3<f32>(gstatic[topg * 28u + 12u], gstatic[topg * 28u + 13u], gstatic[topg * 28u + 14u]);
  let nw = normalize(rmul(M, n0));              // surface normal at the strand (its first kernel's), world
  // state
  let i5 = 5u * r;
  var s0 = sstate[i5];
  var s1 = sstate[i5 + 1u];
  var s2 = sstate[i5 + 2u];
  var s3 = sstate[i5 + 3u];
  var s4 = sstate[i5 + 4u];
  // mode 1: the kernels' rotation for this strand with jittered blend weights (rest frame)
  var fg = vec3<f32>(0.0);
  if (P.mode == 1u) {
    var s = vec3<f32>(0.0);
    var ws = 0.0;
    for (var i = 0u; i < 16u; i++) {
      var w = f32(byte(sdata[sb + 19u + i / 4u], i % 4u));
      if (w > 0.0) {
        w *= max(1.0 + P.weight_jit * (2.0 * h01(r * 16u + i + 977u) - 1.0), 0.05);
        let gi = (sdata[sb + 11u + i / 2u] >> (16u * (i % 2u))) & 0xffffu;
        let o = ${GSTATE.phiw}u * P.K + 3u * gi;
        s += w * vec3<f32>(gst[o], gst[o + 1u], gst[o + 2u]);
        ws += w;
      }
    }
    fg = rtmul(M, s / max(ws, 1e-9));
  }
  if (s0.w == 0.0) {                            // first step after a reset: at rest
    s0 = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    s1 = vec4<f32>(0.0);
    s2 = vec4<f32>(xref, 0.0);
    s3 = vec4<f32>(0.0);
    s4 = vec4<f32>(fg, 0.0);
  }
  let dt = P.dt;
  let e = s0.xyz;
  var v = s1.xyz;
  var ep = vec3<f32>(0.0);
  let vs = (xref - s2.xyz) * P.inv_dt;
  if (P.mode == 2u) {
    // inertia of the rigid tip (the kernels' drive), capped; the turn of gravity in the body frame
    var d = vs - s3.xyz;
    let acc = length(d) * P.inv_dt / 9.81;
    if (P.inertia_limit_g > 0.0) { d *= min(P.inertia_limit_g / max(acc, 1e-9), 1.0); }
    v -= P.inertia_gain * d;
    let g = vec3<f32>(P.gx, P.gy, P.gz);
    ep = P.gravity_gain * (g - rmul(M, g)) / o2;
    // neighbours: velocity friction toward the strands around (last frame's velocities on the grid)
    if (P.coh_rate > 0.0) {
      let va = grid_avg(q);
      if (va.w > 0.0) { v += (1.0 - exp(-P.coh_rate * dt)) * (va.xyz - v); }
    }
  }
  // the hand, for this strand alone: each strand is caught a little differently and combed a little to its own side
  var tw = s1.w * pow(T.s.w, dt * 120.0);
  if (T.touching == 1u && T.mode == 1u) {
    let w = (1.0 - smoothstep(0.45 * T.p.w, T.p.w, distance(root, T.p.xyz))) * (1.0 - P.catch_jit * hC);
    if (w > 0.0) {
      var t = T.dir.xyz - dot(T.dir.xyz, nw) * nw;
      t = t / max(length(t), 1e-6);
      t = normalize(t + P.catch_jit * 0.6 * (2.0 * hD - 1.0) * cross(nw, t) + vec3<f32>(1e-7));
      let a = T.s.x * w * (T.dir.w * (1.0 + P.catch_jit * (hE - 0.5)) * t - T.s.y * nw);
      var off = a / o2;
      let lo = length(off);
      let cap = 0.75 * L;
      if (lo > 1e-9) { off *= cap * tanh(lo / cap) / lo; }
      ep += off;
      tw = max(tw, w);
    }
  }
  // damped oscillator over dt in closed form, around the equilibrium ep
  let wd = om * sqrt(1.0 - zeta * zeta);
  let E = exp(-zeta * om * dt);
  let cs = cos(wd * dt);
  let sn = sin(wd * dt);
  let de = e - ep;
  let e1 = E * ((cs + zeta * om / wd * sn) * de + (sn / wd) * v) + ep;
  var v1 = E * ((-om * om / wd * sn) * de + (cs - zeta * om / wd * sn) * v);
  // keep the length; not below the rest height above the surface (plus give; touched fur may lie flatter)
  var c = (cw + e1) / max(length(cw + e1), 1e-12);
  let c0 = cw / clen;
  let lim = min(acos(clamp(dot(c0, nw), -1.0, 1.0)) + P.give, 1.5707963);
  let ca = mix(cos(lim), min(cos(lim), 0.06), tw);
  let el = dot(c, nw);
  if (el < ca) {
    var tt = c - el * nw;
    let tl = length(tt);
    if (tl > 1e-6) { tt = tt / tl; } else { tt = c0; }
    c = ca * nw + sqrt(max(1.0 - ca * ca, 0.0)) * tt;
  }
  let eN = c * clen - cw;
  v1 += P.vc * (eN - e1) * P.inv_dt;
  // the strand's rotation in its root rest frame
  let p = sat_rot(rot_between(ch.xyz, rtmul(M, c)), P.sat * (1.0 + 0.8 * tw));
  var outr = p;
  if (P.mode == 1u) {
    let tau = P.lag_max * hF;
    let al = select(1.0 - exp(-dt / max(tau, 1e-6)), 1.0, tau < 1e-4);
    let lagged = mix(s4.xyz, fg, al);
    s4 = vec4<f32>(lagged, 0.0);
    outr = (1.0 + P.gain_jit * (2.0 * hE - 1.0)) * lagged + p;
  }
  sout[r] = vec4<f32>(outr, tw);
  sstate[i5] = vec4<f32>(eN, 1.0);
  sstate[i5 + 1u] = vec4<f32>(v1, tw);
  sstate[i5 + 2u] = vec4<f32>(xref, 0.0);
  sstate[i5 + 3u] = vec4<f32>(vs, 0.0);
  sstate[i5 + 4u] = s4;
}
`;

// ----------------------------------------------------------------------------------------------------------------
// frame uniforms shared by the strand pass, the splat pass, the body pass and picking
export const FRAME_STRUCT = /* wgsl */`
struct Frame {
  view: mat4x4<f32>,            // world -> camera (OpenCV: x right, y down, z forward), column-major
  cam: vec4<f32>,               // camera position (xyz), 0
  intr: vec4<f32>,              // fx, fy, cx, cy
  vp: vec4<f32>,                // width, height, 1/width, 1/height
  sortz: vec4<f32>,             // znear_sort, zfar_sort, near clip, far clip
  R: u32, N: u32, K: u32, slot: u32,
  bend: u32, sh_on: u32, under: u32, colour_mode: u32,
  style: u32, selected: u32, solo: u32, lod: f32,
  splat_scale: f32, overlap: f32, length_floor: f32, sh_scale: f32,
  abs_tol: f32, rel_tol: f32, pick_x: f32, pick_y: f32,
  bg: vec4<f32>,
  highlight: u32, show_fur: u32, body_rgb: u32, smode: u32,   // body_rgb: plain body colour, RGBA8 (under == 3)   // smode 1: rotation per strand from SDYN (sout)
  region: vec4<f32>,            // x0, x1 of the part of the screen this pass draws (compare divider), 0, 0
  shadow: vec4<f32>,            // ground shadow: centre x, y, ground z, radius
  shadow2: vec4<f32>,           // strength, heading cos, heading sin, 0
  bg_top: vec4<f32>,
  bg_bot: vec4<f32>,
};
`;

export const STRANDS = COMMON + FRAME_STRUCT + /* wgsl */`
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<storage, read> joints: array<f32>;
@group(0) @binding(2) var<storage, read> sdata: array<u32>;
@group(0) @binding(3) var<storage, read> gauss: array<u32>;
@group(0) @binding(4) var<storage, read> gst: array<f32>;
@group(0) @binding(5) var<storage, read_write> recs: array<u32>;
@group(0) @binding(6) var<storage, read_write> keys: array<u32>;
@group(0) @binding(7) var<storage, read_write> scratch: array<atomic<u32>>;
@group(0) @binding(8) var bodyz: texture_2d<f32>;
@group(0) @binding(9) var<storage, read> sout: array<vec4<f32>>;
` + JOINTS + /* wgsl */`
const NB: u32 = ${NBINS}u;
const SEL0: u32 = ${NBINS + 8}u;          // selected-strand record in scratch
fn putf(i: u32, v: f32) { atomicStore(&scratch[i], bitcast<u32>(v)); }
fn quatmat(q: vec4<f32>) -> mat3x3<f32> {     // rows, (x, y, z, w)
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  return mat3x3<f32>(vec3<f32>(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
                     vec3<f32>(2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
                     vec3<f32>(2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)));
}
fn oct_dec(u: u32) -> vec3<f32> {
  var a = f32(u & 0xffffu) / 65535.0 * 2.0 - 1.0;
  var b = f32(u >> 16u) / 65535.0 * 2.0 - 1.0;
  let z = 1.0 - abs(a) - abs(b);
  let t = max(-z, 0.0);
  a += select(t, -t, a >= 0.0);
  b += select(t, -t, b >= 0.0);
  return normalize(vec3<f32>(a, b, z));
}
fn sh_gain(base: u32, l: vec3<f32>) -> f32 {
  var c: array<f32, 15>;
  for (var i = 0u; i < 15u; i++) { c[i] = sbyte(sdata[base + 23u + i / 4u], i % 4u) * F.sh_scale; }
  let x = l.x; let y = l.y; let z = l.z;
  let C1 = 0.4886025119029199;
  var g = c[0] * (x * C1) + c[1] * (y * C1) + c[2] * (z * C1);
  let xx = x * x; let yy = y * y; let zz = z * z; let xy = x * y; let yz = y * z; let xz = x * z;
  g += c[3] * (1.0925484305920792 * xy) + c[4] * (-1.0925484305920792 * yz) + c[5] * (0.31539156525252005 * (2.0 * zz - xx - yy))
     + c[6] * (-1.0925484305920792 * xz) + c[7] * (0.5462742152960396 * (xx - yy));
  g += c[8] * (-0.5900435899266435 * y * (3.0 * xx - yy)) + c[9] * (2.890611442640554 * xy * z)
     + c[10] * (-0.4570457994644658 * y * (4.0 * zz - xx - yy)) + c[11] * (0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy))
     + c[12] * (-0.4570457994644658 * x * (4.0 * zz - xx - yy)) + c[13] * (1.445305721320277 * z * (xx - yy))
     + c[14] * (-0.5900435899266435 * x * (xx - 3.0 * yy));
  return g;
}
fn cull(g0: u32, g1: u32) { for (var g = g0; g < g1; g++) { keys[g] = ${SENTINEL}u; } }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let r = gid3.x + gid3.y * nwg.x * 64u;
  if (r >= F.R) { return; }
  let sb = r * ${STRAND_STRIDE}u;
  let g0 = sdata[sb];
  let cnt = sdata[sb + 1u];
  let isSel = r == F.selected;
  if (F.show_fur == 0u || (F.solo == 1u && !isSel) || (f32(hash32(r ^ 0x9e3779b9u)) * (1.0 / 4294967296.0) >= F.lod && !isSel)) {
    cull(g0, g0 + cnt); return;
  }
  let q = vec3<f32>(bitcast<f32>(sdata[sb + 2u]), bitcast<f32>(sdata[sb + 3u]), bitcast<f32>(sdata[sb + 4u]));
  let sk = skin8(F.slot, sdata[sb + 5u], sdata[sb + 6u], sdata[sb + 7u], sdata[sb + 8u], sdata[sb + 9u], sdata[sb + 10u], q);
  let M = polar3(sk.M);
  let P = sk.base;
  // bend: weighted guide rotation vectors (world) -> root rest frame
  var fr = vec3<f32>(0.0);
  var topg = 0u;
  if (F.bend == 1u && F.smode == 1u) {
    fr = sout[r].xyz;                                   // per-strand dynamics (SDYN), already in the root rest frame
  } else if (F.bend == 1u) {
    var s = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var i = 0u; i < 16u; i++) {
      let w = f32(byte(sdata[sb + 19u + i / 4u], i % 4u));
      if (w > 0.0) {
        let gi = (sdata[sb + 11u + i / 2u] >> (16u * (i % 2u))) & 0xffffu;
        let o = ${GSTATE.phiw}u * F.K + 3u * gi;
        s += w * vec3<f32>(gst[o], gst[o + 1u], gst[o + 2u]);
        wsum += w;
      }
    }
    fr = rtmul(M, s / max(wsum, 1e-9));
  }
  topg = sdata[sb + 11u] & 0xffffu;
  let ang = length(fr);
  let kax = fr / max(ang, 1e-12);
  // view-dependent gain (posed_parameters + real_sh_basis_non_dc), in the root's rest frame
  var gain = 0.0;
  if (F.sh_on == 1u) {
    let vw = normalize(F.cam.xyz - P);
    let lv = rtmul(M, vw);
    let qf = vec4<f32>(shalf(sdata[sb + 27u], 0u), shalf(sdata[sb + 27u], 1u), shalf(sdata[sb + 28u], 0u), shalf(sdata[sb + 28u], 1u)) / 32767.0;
    let Mf = quatmat(normalize(qf));
    let l = rmul(Mf, lv);
    gain = sh_gain(sb, l / max(length(l), 1e-8));
  }
  if (isSel) {
    putf(SEL0, P.x); putf(SEL0 + 1u, P.y); putf(SEL0 + 2u, P.z);
    putf(SEL0 + 3u, fr.x); putf(SEL0 + 4u, fr.y); putf(SEL0 + 5u, fr.z);
    putf(SEL0 + 6u, gain); putf(SEL0 + 7u, f32(cnt));
  }
  let V = F.view;
  let W3 = mat3x3<f32>(V[0].xyz, V[1].xyz, V[2].xyz);     // columns of the view rotation
  let dims = textureDimensions(bodyz);
  var pos = vec3<f32>(0.0);
  for (var i = 0u; i < cnt; i++) {
    let g = g0 + i;
    let w0 = gauss[3u * g]; let w1 = gauss[3u * g + 1u]; let w2 = gauss[3u * g + 2u];
    let lw = unpack2x16float(w1);
    var ch = oct_dec(w0) * lw.x;
    if (ang > 0.0) {
      let u = (f32(i) + 0.5) / f32(cnt);
      let bf = 1.5 * (1.0 - (1.0 - u) * (1.0 - u));
      ch = rodrigues(kax, ang * bf, ch);
    }
    let mid = pos + 0.5 * ch;
    pos += ch;
    let m = rmul(M, mid) + P;
    let len = max(length(ch), F.length_floor);
    let dirw = rmul(M, ch / len);
    // camera space
    let t = (V * vec4<f32>(m, 1.0)).xyz;
    if (t.z < F.sortz.z) { keys[g] = ${SENTINEL}u; continue; }
    let px = F.intr.x * t.x / t.z + F.intr.z;
    let py = F.intr.y * t.y / t.z + F.intr.w;
    // occlusion gate against the body's view depth (render_posed rule)
    if (px >= 0.0 && py >= 0.0 && px < f32(dims.x) && py < f32(dims.y)) {
      let md = textureLoad(bodyz, vec2<i32>(i32(px), i32(py)), 0).x;
      if (t.z > md + F.abs_tol + abs(t.z) * F.rel_tol) { keys[g] = ${SENTINEL}u; continue; }
    }
    // covariance: axisymmetric about the strand direction
    let sx = 0.5 * len * F.overlap * F.splat_scale;
    let wd = lw.y * F.splat_scale;
    let dc = W3 * dirw;                                    // direction in camera space
    let Sc = mat3x3<f32>(wd * wd * vec3<f32>(1.0, 0.0, 0.0) + (sx * sx - wd * wd) * dc.x * dc,
                         wd * wd * vec3<f32>(0.0, 1.0, 0.0) + (sx * sx - wd * wd) * dc.y * dc,
                         wd * wd * vec3<f32>(0.0, 0.0, 1.0) + (sx * sx - wd * wd) * dc.z * dc);
    // gsplat persp_proj with the clamped Jacobian
    let tanx = 0.5 * F.vp.x / F.intr.x; let tany = 0.5 * F.vp.y / F.intr.y;
    let limxp = (F.vp.x - F.intr.z) / F.intr.x + 0.3 * tanx; let limxn = F.intr.z / F.intr.x + 0.3 * tanx;
    let limyp = (F.vp.y - F.intr.w) / F.intr.y + 0.3 * tany; let limyn = F.intr.w / F.intr.y + 0.3 * tany;
    let tx = t.z * min(limxp, max(-limxn, t.x / t.z));
    let ty = t.z * min(limyp, max(-limyn, t.y / t.z));
    let iz = 1.0 / t.z;
    let j0 = vec3<f32>(F.intr.x * iz, 0.0, -F.intr.x * tx * iz * iz);
    let j1 = vec3<f32>(0.0, F.intr.y * iz, -F.intr.y * ty * iz * iz);
    let Sj0 = Sc * j0; let Sj1 = Sc * j1;
    var a = dot(j0, Sj0); var b = dot(j0, Sj1); var c = dot(j1, Sj1);
    let det0 = a * c - b * b;
    a += 0.3; c += 0.3;
    let det = a * c - b * b;
    if (det <= 0.0) { keys[g] = ${SENTINEL}u; continue; }
    let comp = sqrt(max(det0 / det, 0.0));
    let mh = 0.5 * (a + c);
    let lam = mh + sqrt(max(0.01, mh * mh - det));
    let rad = ceil(3.0 * sqrt(lam));
    if (px + rad <= F.region.x || py + rad <= 0.0 || px - rad >= F.region.y || py - rad >= F.vp.y) { keys[g] = ${SENTINEL}u; continue; }
    var op = f32(w2 >> 24u) / 255.0;
    let base = vec3<f32>(f32(w2 & 255u), f32((w2 >> 8u) & 255u), f32((w2 >> 16u) & 255u)) / 255.0;
    var col = clamp(base * (1.0 + gain), vec3<f32>(0.0), vec3<f32>(1.0));
    switch (F.colour_mode) {
      case 1u: { col = base; }
      case 2u: { col = 0.5 * (dirw + vec3<f32>(1.0)); }
      case 3u: { col = turbo(ang / 0.6); }
      case 4u: { col = vec3<f32>(op); op = max(op, 0.6); }
      case 5u: { col = hashcol(r); }
      case 6u: { col = hashcol(topg * 7919u + 13u); }
      case 7u: { col = turbo((f32(i) + 0.5) / f32(cnt)); }
      case 8u: { col = turbo(len / 0.006); }
      default: {}
    }
    if (F.highlight == 1u && F.selected < F.R) {
      if (isSel) { col = vec3<f32>(1.0, 0.15, 0.55); op = max(op, 0.9); }
    }
    op *= comp;
    if (op < 1.0 / 255.0) { keys[g] = ${SENTINEL}u; continue; }
    let rb = 8u * g;
    recs[rb] = bitcast<u32>(px); recs[rb + 1u] = bitcast<u32>(py);
    recs[rb + 2u] = bitcast<u32>(c / det); recs[rb + 3u] = bitcast<u32>(-b / det); recs[rb + 4u] = bitcast<u32>(a / det);
    recs[rb + 5u] = pack2x16float(col.xy); recs[rb + 6u] = pack2x16float(vec2<f32>(col.z, op));
    recs[rb + 7u] = bitcast<u32>(t.z);
    let zr = (F.sortz.y - t.z) / (F.sortz.y - F.sortz.x);
    let bin = u32(clamp(zr, 0.0, 0.99999994) * f32(NB));
    keys[g] = bin;
    atomicAdd(&scratch[bin], 1u);
  }
}
`;

// exclusive scan of the NB bins in scratch[0..NB): three passes; total -> indirect draw args
export const SCAN = /* wgsl */`
@group(0) @binding(0) var<storage, read_write> scratch: array<u32>;
@group(0) @binding(1) var<storage, read_write> blocks: array<u32>;
@group(0) @binding(2) var<storage, read_write> indirect: array<u32>;
var<workgroup> sh: array<u32, 256>;
const PER: u32 = ${NBINS / 256 / 256}u;    // bins per thread
fn wg_excl(lid: u32, v: u32) -> u32 {      // exclusive scan over the workgroup (Hillis-Steele on shared memory)
  sh[lid] = v;
  workgroupBarrier();
  for (var o = 1u; o < 256u; o = o << 1u) {
    var t = 0u;
    if (lid >= o) { t = sh[lid - o]; }
    workgroupBarrier();
    sh[lid] += t;
    workgroupBarrier();
  }
  let incl = sh[lid];
  workgroupBarrier();
  return incl - v;
}
@compute @workgroup_size(256)
fn reduce(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let base = (wg.x * 256u + li.x) * PER;
  var s = 0u;
  for (var i = 0u; i < PER; i++) { s += scratch[base + i]; }
  sh[li.x] = s;
  workgroupBarrier();
  for (var o = 128u; o > 0u; o = o >> 1u) {
    if (li.x < o) { sh[li.x] += sh[li.x + o]; }
    workgroupBarrier();
  }
  if (li.x == 0u) { blocks[wg.x] = sh[0]; }
}
@compute @workgroup_size(256)
fn top(@builtin(local_invocation_id) li: vec3<u32>) {
  let v = blocks[li.x];
  let e = wg_excl(li.x, v);
  blocks[li.x] = e;
  if (li.x == 255u) {
    let total = e + v;
    indirect[0] = 4u; indirect[1] = total; indirect[2] = 0u; indirect[3] = 0u;
    indirect[4] = total;
  }
}
@compute @workgroup_size(256)
fn down(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let base = (wg.x * 256u + li.x) * PER;
  var s = 0u;
  for (var i = 0u; i < PER; i++) { s += scratch[base + i]; }
  var run = wg_excl(li.x, s) + blocks[wg.x];
  for (var i = 0u; i < PER; i++) { let v = scratch[base + i]; scratch[base + i] = run; run += v; }
}
`;

export const SCATTER = /* wgsl */`
struct Count { N: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<uniform> C: Count;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> scratch: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sorted: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid3.x + gid3.y * nwg.x * 256u;
  if (i >= C.N) { return; }
  let k = keys[i];
  if (k == ${SENTINEL}u) { return; }
  let p = atomicAdd(&scratch[k], 1u);
  sorted[p] = i;
}
`;

// splats: one instanced quad per sorted Gaussian, back to front, premultiplied "over" onto the body pass
export const SPLAT = FRAME_STRUCT + /* wgsl */`
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<storage, read> recs: array<u32>;
@group(0) @binding(2) var<storage, read> sorted: array<u32>;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) d: vec2<f32>, @location(1) conic: vec3<f32>,
              @location(2) col: vec4<f32> };
@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  let g = sorted[ii];
  let b = 8u * g;
  let mx = bitcast<f32>(recs[b]); let my = bitcast<f32>(recs[b + 1u]);
  let ca = bitcast<f32>(recs[b + 2u]); let cb = bitcast<f32>(recs[b + 3u]); let cc = bitcast<f32>(recs[b + 4u]);
  let rg = unpack2x16float(recs[b + 5u]); let ba = unpack2x16float(recs[b + 6u]);
  // covariance from the conic, then the ellipse axes (3 sigma)
  let det = ca * cc - cb * cb;
  let sa = cc / det; let sb = -cb / det; let sc = ca / det;
  let mh = 0.5 * (sa + sc);
  let disc = sqrt(max(mh * mh - (sa * sc - sb * sb), 0.0));
  let l1 = mh + disc; let l2 = max(mh - disc, 0.0);
  var v1 = vec2<f32>(sb, l1 - sa);
  if (length(v1) < 1e-12) { v1 = vec2<f32>(1.0, 0.0); }
  v1 = normalize(v1);
  let v2 = vec2<f32>(-v1.y, v1.x);
  let corner = vec2<f32>(f32(vi & 1u) * 2.0 - 1.0, f32((vi >> 1u) & 1u) * 2.0 - 1.0);
  var r1 = 3.0 * sqrt(l1) + 1.0; var r2 = 3.0 * sqrt(l2) + 1.0;
  if (F.style == 2u) { r1 = 2.5; r2 = 2.5; }
  let off = corner.x * r1 * v1 + corner.y * r2 * v2;
  let p = vec2<f32>(mx, my) + off;
  o.pos = vec4<f32>(p.x * F.vp.z * 2.0 - 1.0, 1.0 - p.y * F.vp.w * 2.0, 0.0, 1.0);
  o.d = off;
  o.conic = vec3<f32>(ca, cb, cc);
  o.col = vec4<f32>(rg.x, rg.y, ba.x, ba.y);
  return o;
}
@fragment
fn fs(v: VOut) -> @location(0) vec4<f32> {
  let d = -v.d;    // pixel centre minus mean, sign irrelevant for the quadratic form
  let sigma = 0.5 * (v.conic.x * d.x * d.x + v.conic.z * d.y * d.y) + v.conic.y * d.x * d.y;
  var alpha: f32;
  if (F.style == 2u) {                        // centres: small opaque discs
    if (dot(d, d) > 2.0 * 2.0) { discard; }
    alpha = 1.0;
  } else if (F.style == 3u) {                 // ellipses: each Gaussian as a solid 2-sigma ellipse with a darker rim
    let m = sqrt(max(2.0 * sigma, 0.0));
    if (m > 2.0) { discard; }
    let rim = select(1.0, 0.55, m > 1.55);
    return vec4<f32>(v.col.rgb * rim * 0.95, 0.95);
  } else if (F.style == 1u) {                 // rings: the 2-sigma ellipse of every Gaussian
    let m = sqrt(max(2.0 * sigma, 0.0));
    if (abs(m - 2.0) > 0.35) { discard; }
    alpha = 0.85;
  } else {
    alpha = min(0.999, v.col.a * exp(-sigma));
    if (sigma < 0.0 || alpha < 1.0 / 255.0) { discard; }
  }
  return vec4<f32>(v.col.rgb * alpha, alpha);
}
`;

// body: LBS of the simplified body. under = 1: colour of the fur rooted there (under-colour), 0: background colour,
// 2: neutral grey with a flat head-light shade (Gaussian and kernel views). Second target = camera z (occlusion test).
export const BODY = FRAME_STRUCT + /* wgsl */`
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<storage, read> joints: array<f32>;
@group(0) @binding(2) var<storage, read> bv: array<f32>;
@group(0) @binding(3) var<storage, read> bj: array<u32>;
@group(0) @binding(4) var<storage, read> bf: array<u32>;
@group(0) @binding(5) var<storage, read> brgb: array<u32>;
fn unorm16(w: u32, half: u32) -> f32 { return f32((w >> (16u * half)) & 0xffffu) / 65535.0; }
fn byte(w: u32, i: u32) -> u32 { return (w >> (8u * i)) & 0xffu; }
fn rmul(R: mat3x3<f32>, v: vec3<f32>) -> vec3<f32> { return vec3<f32>(dot(R[0], v), dot(R[1], v), dot(R[2], v)); }
` + JOINTS + /* wgsl */`
fn skinned(v: u32) -> vec3<f32> {
  let q = vec3<f32>(bv[3u * v], bv[3u * v + 1u], bv[3u * v + 2u]);
  let b = 6u * v;
  var p = vec3<f32>(0.0);
  let ws = array<u32, 4>(bj[b + 2u], bj[b + 3u], bj[b + 4u], bj[b + 5u]);
  for (var i = 0u; i < 8u; i++) {
    let id = select(byte(bj[b + 1u], i - 4u), byte(bj[b], i), i < 4u);
    let w = unorm16(ws[i / 2u], i % 2u);
    if (w > 0.0) { p += w * (rmul(jrot(F.slot, id), q) + jtr(F.slot, id)); }
  }
  return p;
}
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) col: vec3<f32>, @location(1) z: f32 };
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  let face = vi / 3u;
  let p = skinned(bf[vi]);
  let t = (F.view * vec4<f32>(p, 1.0)).xyz;
  var o: VOut;
  let n = F.sortz.z; let f = F.sortz.w;
  o.pos = vec4<f32>((2.0 * F.intr.x * F.vp.z) * t.x + (2.0 * F.intr.z * F.vp.z - 1.0) * t.z,
                    -(2.0 * F.intr.y * F.vp.w) * t.y - (2.0 * F.intr.w * F.vp.w - 1.0) * t.z,
                    f / (f - n) * (t.z - n), t.z);
  if (F.under == 2u || F.under == 3u) {
    let a = skinned(bf[3u * face]); let b = skinned(bf[3u * face + 1u]); let c = skinned(bf[3u * face + 2u]);
    let nn = normalize(cross(b - a, c - a));
    let vdir = normalize(F.cam.xyz - p);
    let ndv = abs(dot(nn, vdir));
    if (F.under == 3u) {
      o.col = unpack4x8unorm(F.body_rgb).rgb * (0.62 + 0.38 * ndv);    // the plain colour we set, lightly shaded
    } else {
      o.col = vec3<f32>(0.80, 0.80, 0.78) * (0.35 + 0.65 * ndv);
    }
  } else {
    let c = brgb[face];
    let rgb = vec3<f32>(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u)) / 255.0;
    o.col = select(F.bg.rgb, rgb, F.under == 1u);
  }
  o.z = t.z;
  return o;
}
struct FOut { @location(0) col: vec4<f32>, @location(1) z: vec4<f32> };
@fragment
fn fs(v: VOut) -> FOut {
  var o: FOut;
  o.col = vec4<f32>(v.col, 1.0);
  o.z = vec4<f32>(v.z, 0.0, 0.0, 0.0);
  return o;
}
`;

// backdrop: vertical gradient over the pass region, then a soft contact shadow on the ground plane
export const BACKDROP = FRAME_STRUCT + /* wgsl */`
@group(0) @binding(0) var<uniform> F: Frame;
struct BOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs_bg(@builtin(vertex_index) vi: u32) -> BOut {
  let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
  var o: BOut;
  o.pos = vec4<f32>(p, 0.999, 1.0);
  o.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return o;
}
struct FOut { @location(0) col: vec4<f32>, @location(1) z: vec4<f32> };
@fragment
fn fs_bg(v: BOut) -> FOut {
  var o: FOut;
  let t = clamp(v.uv.y, 0.0, 1.0);
  let r = length((v.uv - vec2<f32>(0.5, 0.45)) * vec2<f32>(1.0, 0.8));
  o.col = vec4<f32>(mix(F.bg_top.rgb, F.bg_bot.rgb, t * t) * (1.0 - 0.05 * F.bg_bot.w * r * r), 1.0);
  o.z = vec4<f32>(1.0e30, 0.0, 0.0, 0.0);
  return o;
}
struct SOut { @builtin(position) pos: vec4<f32>, @location(0) q: vec2<f32> };
@vertex
fn vs_shadow(@builtin(vertex_index) vi: u32) -> SOut {
  let corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
                                     vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
  let c = corners[vi];
  let fw = vec2<f32>(F.shadow2.y, F.shadow2.z);
  let sd = vec2<f32>(-fw.y, fw.x);
  let r = F.shadow.w;
  let xy = F.shadow.xy + fw * (c.x * r * 1.4) + sd * (c.y * r * 0.8);
  let t = (F.view * vec4<f32>(xy, F.shadow.z, 1.0)).xyz;
  var o: SOut;
  let n = F.sortz.z; let f = F.sortz.w;
  o.pos = vec4<f32>((2.0 * F.intr.x * F.vp.z) * t.x + (2.0 * F.intr.z * F.vp.z - 1.0) * t.z,
                    -(2.0 * F.intr.y * F.vp.w) * t.y - (2.0 * F.intr.w * F.vp.w - 1.0) * t.z,
                    f / (f - n) * (t.z - n), t.z);
  o.q = c;
  return o;
}
struct SFOut { @location(0) col: vec4<f32>, @location(1) z: vec4<f32> };
@fragment
fn fs_shadow(v: SOut) -> SFOut {
  let d2 = dot(v.q, v.q);
  let a = F.shadow2.x * exp(-4.0 * d2) * (1.0 - smoothstep(0.7, 1.0, sqrt(d2)));
  var o: SFOut;
  o.col = vec4<f32>(vec3<f32>(0.18, 0.17, 0.16) * a, a);
  o.z = vec4<f32>(1.0e30, 0.0, 0.0, 0.0);
  return o;
}
`;

// picking: front-most Gaussian whose alpha at the pick pixel is at least 0.05
export const PICK = /* wgsl */`
struct Pick { N: u32, px: f32, py: f32, pass_: u32 };
@group(0) @binding(0) var<uniform> P: Pick;
@group(0) @binding(1) var<storage, read> recs: array<u32>;
@group(0) @binding(2) var<storage, read> keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> best: array<atomic<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid3: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid3.x + gid3.y * nwg.x * 256u;
  if (i >= P.N || keys[i] == ${SENTINEL}u) { return; }
  let b = 8u * i;
  let d = vec2<f32>(P.px - bitcast<f32>(recs[b]), P.py - bitcast<f32>(recs[b + 1u]));
  let sigma = 0.5 * (bitcast<f32>(recs[b + 2u]) * d.x * d.x + bitcast<f32>(recs[b + 4u]) * d.y * d.y) + bitcast<f32>(recs[b + 3u]) * d.x * d.y;
  let op = unpack2x16float(recs[b + 6u]).y;
  if (sigma < 0.0 || op * exp(-sigma) < 0.05) { return; }
  let z = bitcast<f32>(recs[b + 7u]);
  let zk = 0xffffffffu - u32(clamp(z, 0.0, 400.0) * 1.0e7);   // larger = nearer
  if (P.pass_ == 0u) { atomicMax(&best[0], zk); }
  else if (zk == atomicLoad(&best[0])) { atomicMin(&best[1], i); }
}
`;

// fine kernels of the field as lines (anchor -> current tip), depth-tested against the body, coloured by the tip's
// offset from rest relative to the kernel's length (blue = at rest, red = offset of its full length or more)
export const LINES = FRAME_STRUCT + /* wgsl */`
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<storage, read> joints: array<f32>;
@group(0) @binding(2) var<storage, read> gjoints: array<u32>;
@group(0) @binding(3) var<storage, read> gstatic: array<f32>;
@group(0) @binding(4) var<storage, read> gst: array<f32>;
` + COMMON + JOINTS + /* wgsl */`
fn gs3(k: u32, o: u32) -> vec3<f32> { let b = k * 28u + o; return vec3<f32>(gstatic[b], gstatic[b + 1u], gstatic[b + 2u]); }
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) col: vec3<f32> };
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  let k = vi / 2u;
  let g6 = k * 6u;
  let q = gs3(k, 0u);
  let sk = skin8(F.slot, gjoints[g6], gjoints[g6 + 1u], gjoints[g6 + 2u], gjoints[g6 + 3u], gjoints[g6 + 4u], gjoints[g6 + 5u], q);
  let R = polar3(sk.M);
  let tb = ${GSTATE.tip}u * F.K + 3u * k;
  let tip = vec3<f32>(gst[tb], gst[tb + 1u], gst[tb + 2u]);
  let pr = select(tip, gs3(k, 3u), (vi & 1u) == 0u);
  let p = sk.base + rmul(R, pr - q);
  let t = (F.view * vec4<f32>(p, 1.0)).xyz * (1.0 - 0.004);    // a little toward the camera: anchors sit on the body
  var o: VOut;
  let n = F.sortz.z; let f = F.sortz.w;
  o.pos = vec4<f32>((2.0 * F.intr.x * F.vp.z) * t.x + (2.0 * F.intr.z * F.vp.z - 1.0) * t.z,
                    -(2.0 * F.intr.y * F.vp.w) * t.y - (2.0 * F.intr.w * F.vp.w - 1.0) * t.z,
                    f / (f - n) * (t.z - n), t.z);
  let dev = length(tip - gs3(k, 6u)) / max(gstatic[k * 28u + 24u], 1e-6);
  o.col = turbo(0.12 + 0.88 * clamp(dev, 0.0, 1.0));
  return o;
}
@fragment
fn fs(v: VOut) -> @location(0) vec4<f32> { return vec4<f32>(v.col, 1.0); }
`;
