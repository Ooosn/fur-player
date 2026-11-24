// Orbit camera around a target with inertia; z is up. View matrix in the OpenCV convention (x right, y down,
// z forward), column-major, as the shaders expect.
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scl = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (a) => scl(a, 1 / Math.hypot(a[0], a[1], a[2]));

export class Orbit {
  constructor() {
    this.target = [0, 0, 0]; this.yaw = 0; this.pitch = 0; this.dist = 2;
    this.vyaw = 0; this.vpitch = 0; this.goalDist = null;
  }
  // orientation of a given world->camera matrix (rows of a 4x4), looking at the point closest to `centre`
  fromView(V, centre) {
    const R = [V[0].slice(0, 3), V[1].slice(0, 3), V[2].slice(0, 3)];
    const t = [V[0][3], V[1][3], V[2][3]];
    const pos = scl(add(add(scl(R[0], t[0]), scl(R[1], t[1])), scl(R[2], t[2])), -1);
    const f = R[2];
    const v = scl(f, -1);
    this.pitch = Math.asin(v[2]); this.yaw = Math.atan2(v[1], v[0]);
    this.target = centre.slice();
    this.vyaw = 0; this.vpitch = 0;
  }
  // distance at which a sphere of `radius` around the target fills `fill` of the smaller field of view
  fit(radius, W, H, fy, fill = 0.82) {
    const half = Math.min(Math.atan(0.5 * H / fy), Math.atan(0.5 * W / fy));
    this.dist = radius / Math.sin(half) / fill;
  }
  step() {                                        // inertia; returns true while still moving
    if (Math.abs(this.vyaw) < 1e-5 && Math.abs(this.vpitch) < 1e-5) { this.vyaw = 0; this.vpitch = 0; return false; }
    this.yaw += this.vyaw; this.pitch = Math.max(-0.2, Math.min(1.35, this.pitch + this.vpitch));
    this.vyaw *= 0.86; this.vpitch *= 0.86;
    return true;
  }
  pose(offset) {
    const tg = add(this.target, offset);
    const cp = Math.cos(this.pitch);
    const pos = add(tg, scl([cp * Math.cos(this.yaw), cp * Math.sin(this.yaw), Math.sin(this.pitch)], this.dist));
    const f = norm(sub(tg, pos));
    const x = norm(cross(f, [0, 0, 1]));
    const y = cross(f, x);
    const t = [-dot(x, pos), -dot(y, pos), -dot(f, pos)];
    const m = new Float32Array(16);
    m[0] = x[0]; m[4] = x[1]; m[8] = x[2]; m[12] = t[0];
    m[1] = y[0]; m[5] = y[1]; m[9] = y[2]; m[13] = t[1];
    m[2] = f[0]; m[6] = f[1]; m[10] = f[2]; m[14] = t[2];
    m[15] = 1;
    return { m, pos, x, y, f, target: tg };
  }
}
