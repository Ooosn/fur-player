// Per-guide coefficients of the dynamics for given parameters: the formulas of scripts/dyn_params.rebuild
// (length smoothed over neighbours -> tip frequency -> omega; damping ratio + drag; closed-form oscillator step for the
// substep; sleep thresholds; cohesion fraction; cone; floor give).

export function rebuild(params, g, nb, fps, substeps) {
  const K = g.K;
  const { row, ids, w } = nb;
  let hs = Float64Array.from(g.hlen);
  for (let it = 0; it < params.len_smooth_iters; it++) {
    const out = new Float64Array(K);
    for (let k = 0; k < K; k++) {
      let s = 0;
      for (let i = row[k]; i < row[k + 1]; i++) s += w[i] * hs[ids[i]];
      out[k] = s;
    }
    hs = out;
  }
  const h = 1 / fps / substeps;
  const coef = new Float32Array(K * 10);
  const give = (params.floor_give_deg || 0) * Math.PI / 180;
  let fmin = Infinity, fmax = -Infinity, zmin = Infinity, zmax = -Infinity;
  for (let k = 0; k < K; k++) {
    const f = Math.min(params.f_max, Math.max(params.f_min, params.f_ref * Math.pow(params.h_ref / hs[k], params.len_exp)));
    const om = 2 * Math.PI * f;
    const zeta = Math.min(params.zeta + params.drag / (2 * om), 0.9);
    const wd = om * Math.sqrt(1 - zeta * zeta);
    const E = Math.exp(-zeta * om * h), cs = Math.cos(wd * h), sn = Math.sin(wd * h);
    const b = k * 10;
    coef[b] = om;
    coef[b + 1] = E * (cs + zeta * om / wd * sn);
    coef[b + 2] = E * sn / wd;
    coef[b + 3] = -E * om * om / wd * sn;
    coef[b + 4] = E * (cs - zeta * om / wd * sn);
    coef[b + 5] = params.sleep_offset * g.hlen[k];
    coef[b + 6] = params.sleep_speed * g.hlen[k] * om;
    let ca = g.ca0[k], sa = g.sa0[k];
    if (give > 0) {
      const th = Math.atan2(sa, ca);
      const lim = Math.max(th, Math.min(th + give, Math.PI / 2));
      ca = Math.cos(lim); sa = Math.sin(lim);
    }
    coef[b + 7] = ca;
    coef[b + 8] = sa;
    fmin = Math.min(fmin, f); fmax = Math.max(fmax, f); zmin = Math.min(zmin, zeta); zmax = Math.max(zmax, zeta);
  }
  const coh = 1 - Math.exp(-params.cohesion * h);
  return {
    coef, h, coh,
    cone_c: Math.cos(params.cone_max_deg * Math.PI / 180), cone_s: Math.sin(params.cone_max_deg * Math.PI / 180),
    info: { f_tip_hz: [fmin, fmax], damping_ratio: [zmin, zmax], cohesion_per_substep: coh },
  };
}
