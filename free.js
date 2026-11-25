// "Free" motion (v23, the author 11-25: the dog should keep moving, choosing freely among the motions, and not stand
// still for long): the motion-capture clips chained in random order into one long clip. Each clip is trimmed to its
// active frames (every clip ends with about 1.6 s of standing still), placed where the previous one is (position on
// the ground and heading of the pelvis -> chest line), and cross-faded over a few frames (per joint: quaternions
// normalised-lerped, translations lerped). The result has the layout of a clip file (per frame, per substep, per joint:
// quaternion x y z w, translation x y z, 0), so the player treats it like any clip.
const J = 39;

const qmul = (a, b) => [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
  a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
  a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
  a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
function qrot(q, v) {
  const x = q[0], y = q[1], z = q[2], w = q[3];
  const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
}
const smooth = (u) => u * u * (3 - 2 * u);

// world position of joint j in frame f (last substep) of a record array
function jointPos(rec, S, f, j, rig) {
  const b = ((f * S + S - 1) * J + j) * 8;
  const p = qrot([rec[b], rec[b + 1], rec[b + 2], rec[b + 3]], rig[j]);
  return [p[0] + rec[b + 4], p[1] + rec[b + 5], p[2] + rec[b + 6]];
}
// middle of pelvis (5) and chest (8) on the ground plane, and the heading angle of pelvis -> chest
function placeOf(rec, S, f, rig) {
  const a = jointPos(rec, S, f, 5, rig), c = jointPos(rec, S, f, 8, rig);
  return { x: 0.5 * (a[0] + c[0]), y: 0.5 * (a[1] + c[1]), h: Math.atan2(c[1] - a[1], c[0] - a[0]) };
}
// the clip's active frames: from just before the first to a little after the last frame whose mean joint speed is
// above `still` (m/s)
function activeRange(rec, F, S, rig, fps, still) {
  let first = -1, last = -1, prev = null;
  for (let f = 0; f < F; f++) {
    const P = [];
    for (let j = 0; j < J; j++) P.push(jointPos(rec, S, f, j, rig));
    if (prev) {
      let s = 0;
      for (let j = 0; j < J; j++) s += Math.hypot(P[j][0] - prev[j][0], P[j][1] - prev[j][1], P[j][2] - prev[j][2]);
      if ((s / J) * fps > still) { if (first < 0) first = f - 1; last = f; }
    }
    prev = P;
  }
  if (first < 0) return [0, F - 1];
  return [Math.max(0, first - 2), Math.min(F - 1, last + 6)];
}

// clips: [{ name, rec (Float32Array), F, S, w }]; returns { rec, F, S, order }
export function buildFree(clips, rig, fps, seconds, fade = 12, rand = Math.random) {
  const S = clips[0].S, target = Math.round(seconds * fps);
  for (const c of clips) [c.a, c.b] = activeRange(c.rec, c.F, c.S, rig, fps, 0.06);
  const maxLen = Math.max(...clips.map((c) => c.b - c.a + 1));
  const per = S * J * 8;
  const out = new Float32Array((target + maxLen + 1) * per);
  let used = 0, last = null;
  const order = [];
  while (used < target) {
    const pool = clips.filter((c) => c.name !== last);
    let r = rand() * pool.reduce((s, c) => s + c.w, 0), c = pool[0];
    for (const x of pool) { r -= x.w; if (r <= 0) { c = x; break; } }
    const len = c.b - c.a + 1;
    order.push(c.name);
    if (used === 0) {                                   // the first clip where it was recorded
      out.set(c.rec.subarray(c.a * per, (c.b + 1) * per), 0);
      used = len; last = c.name; continue;
    }
    const ov = Math.min(fade, len - 1, used);
    // place clip frame a where the sequence is at frame used - ov (same spot on the ground, same heading)
    const A = placeOf(out, S, used - ov, rig), B = placeOf(c.rec, S, c.a, rig);
    const ang = A.h - B.h, cs = Math.cos(ang), sn = Math.sin(ang);
    const qz = [0, 0, Math.sin(ang / 2), Math.cos(ang / 2)];
    const gx = A.x - (cs * B.x - sn * B.y), gy = A.y - (sn * B.x + cs * B.y);
    const src = c.rec;
    for (let k = 0; k < len; k++) {
      const dst = used - ov + k, wgt = k < ov ? smooth((k + 1) / (ov + 1)) : 1;
      for (let i = 0; i < S * J; i++) {
        const sb = (c.a + k) * per + i * 8, db = dst * per + i * 8;
        let q = qmul(qz, [src[sb], src[sb + 1], src[sb + 2], src[sb + 3]]);
        const t = [cs * src[sb + 4] - sn * src[sb + 5] + gx, sn * src[sb + 4] + cs * src[sb + 5] + gy, src[sb + 6]];
        if (wgt < 1) {                                  // cross-fade with what the previous clip left there
          const p = [out[db], out[db + 1], out[db + 2], out[db + 3]];
          if (p[0] * q[0] + p[1] * q[1] + p[2] * q[2] + p[3] * q[3] < 0) q = q.map((v) => -v);
          const m = [0, 1, 2, 3].map((d) => (1 - wgt) * p[d] + wgt * q[d]);
          const n = Math.hypot(m[0], m[1], m[2], m[3]) || 1;
          q = m.map((v) => v / n);
          for (let d = 0; d < 3; d++) t[d] = (1 - wgt) * out[db + 4 + d] + wgt * t[d];
        }
        out[db] = q[0]; out[db + 1] = q[1]; out[db + 2] = q[2]; out[db + 3] = q[3];
        out[db + 4] = t[0]; out[db + 5] = t[1]; out[db + 6] = t[2]; out[db + 7] = 0;
      }
    }
    used = used - ov + len;
    last = c.name;
  }
  return { rec: out.slice(0, used * per), F: used, S, order };
}
