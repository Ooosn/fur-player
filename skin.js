// Loose skin of the trunk, computed when a clip is loaded (the same rule as scripts/loose_skin.py, 11-20).
// The author saw the short body fur hardly move in the shake: the strands do bend with their guides, but they are
// short (median 4.8 mm), so their tips move 1-3 mm. A dog's coat moves in a shake because its loose skin rolls much
// further than the vertebrae (about 90 against 30 degrees, Dickerson, Mills, Hu 2012). Here each trunk segment
// (pelvis 5, spine 6-8) gets one extra degree of freedom: the roll of its skin about the rest spine line, relative to
// the bone, as a damped oscillator driven by the bone's roll acceleration (the skin keeps its absolute roll rate when
// the bone's changes; a spring of natural frequency hz and damping ratio zeta pulls it back). The skinning transform
// of the segment and of the helper joints that carry trunk skin becomes A_j T(theta), T a rotation about the spine
// line in the rest space. Bones, children and the camera track are unchanged; body, roots and guides follow by
// linear-blend skinning. The relative roll is softly limited to cap_deg (tanh), then scaled by amount.

const SEGMENTS = [5, 6, 7, 8];
const CARRIERS = {
  5: [[5, 1], [27, 1], [28, 1], [37, 1], [38, 1]],     // pelvis, hips and flanks
  6: [[6, 1]],
  7: [[7, 1]],
  8: [[8, 1], [25, 1], [26, 1], [15, 0.5], [20, 0.5]], // chest, chest underside, shoulders (half)
};
export const SKIN_DEFAULT = { on: true, amount: 1.0, hz: 4.0, zeta: 0.25, cap_deg: 40 };

// jm: [F * S * 39 * 12] row-major 3x4 joint transforms (rest -> posed); rig: rest joint positions. Changes jm in place.
export function looseSkin(jm, F, S, rig, fps, p) {
  const J = 39, T = F * S, h = 1 / fps / S;
  const a = [rig[8][0] - rig[5][0], rig[8][1] - rig[5][1], rig[8][2] - rig[5][2]];
  const an = Math.hypot(a[0], a[1], a[2]); a[0] /= an; a[1] /= an; a[2] /= an;
  const pv = rig[5];
  const om = 2 * Math.PI * p.hz, z = p.zeta, wd = om * Math.sqrt(1 - z * z);
  const E = Math.exp(-z * om * h), cs = Math.cos(wd * h), sn = Math.sin(wd * h);
  const A11 = E * (cs + z * om / wd * sn), A12 = E * sn / wd, A21 = -E * om * om / wd * sn, A22 = E * (cs - z * om / wd * sn);
  const cap = p.cap_deg * Math.PI / 180;
  const theta = new Float64Array(T * SEGMENTS.length);
  const stats = {};
  SEGMENTS.forEach((s, i) => {
    let th = 0, thd = 0, prev = 0, mx = 0;
    for (let t = 1; t < T; t++) {
      const o0 = ((t - 1) * J + s) * 12, o1 = (t * J + s) * 12;
      // d = R0^T R1 (body-frame increment); its rotation vector's component along a
      const d = (r, c) => jm[o0 + r] * jm[o1 + c] + jm[o0 + 4 + r] * jm[o1 + 4 + c] + jm[o0 + 8 + r] * jm[o1 + 8 + c];
      const d00 = d(0, 0), d11 = d(1, 1), d22 = d(2, 2);
      const wx = d(2, 1) - d(1, 2), wy = d(0, 2) - d(2, 0), wz = d(1, 0) - d(0, 1);
      const sh = 0.5 * Math.hypot(wx, wy, wz), ch = Math.max(-1, Math.min(1, 0.5 * (d00 + d11 + d22 - 1)));
      const ang = Math.atan2(sh, ch), k = 1 / Math.max(2 * sh, 1e-12);
      const rate = ang * k * (wx * a[0] + wy * a[1] + wz * a[2]) / h;
      thd -= rate - prev; prev = rate;
      const th2 = A11 * th + A12 * thd; thd = A21 * th + A22 * thd; th = th2;
      const v = p.amount * cap * Math.tanh(th / cap);
      theta[t * SEGMENTS.length + i] = v;
      mx = Math.max(mx, Math.abs(v));
    }
    stats[s] = mx * 180 / Math.PI;
  });
  // A_j <- A_j T(g theta): R' = R Rt, t' = R (pv - Rt pv) + t
  for (let t = 0; t < T; t++) {
    SEGMENTS.forEach((s, i) => {
      const th0 = theta[t * SEGMENTS.length + i];
      if (th0 === 0) return;
      for (const [j, g] of CARRIERS[s]) {
        const th = g * th0, c = Math.cos(th), si = Math.sin(th), C1 = 1 - c;
        const Rt = [c + a[0] * a[0] * C1, a[0] * a[1] * C1 - a[2] * si, a[0] * a[2] * C1 + a[1] * si,
                    a[1] * a[0] * C1 + a[2] * si, c + a[1] * a[1] * C1, a[1] * a[2] * C1 - a[0] * si,
                    a[2] * a[0] * C1 - a[1] * si, a[2] * a[1] * C1 + a[0] * si, c + a[2] * a[2] * C1];
        const tt = [0, 1, 2].map((r) => pv[r] - (Rt[3 * r] * pv[0] + Rt[3 * r + 1] * pv[1] + Rt[3 * r + 2] * pv[2]));
        const o = (t * J + j) * 12;
        const M = Array.from(jm.subarray(o, o + 12));
        for (let r = 0; r < 3; r++) {
          for (let cc = 0; cc < 3; cc++) jm[o + 4 * r + cc] = M[4 * r] * Rt[cc] + M[4 * r + 1] * Rt[3 + cc] + M[4 * r + 2] * Rt[6 + cc];
          jm[o + 4 * r + 3] = M[4 * r] * tt[0] + M[4 * r + 1] * tt[1] + M[4 * r + 2] * tt[2] + M[4 * r + 3];
        }
      }
    });
  }
  return stats;
}
