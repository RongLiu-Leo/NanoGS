import * as THREE from "three";
import createKDTree from "https://esm.sh/static-kdtree";
import { PackedSplats } from "@sparkjsdev/spark";

const TWO_PI_POW_1P5 = Math.pow(2.0 * Math.PI, 1.5);
const LOG2PI = Math.log(2.0 * Math.PI);

export async function simplifyMesh(mesh, params = {}, onStatus = () => {}, onProgress = () => {}) {
  if (!mesh) throw new Error("simplifyMesh: mesh is required.");
  if (mesh.initialized) await mesh.initialized;

  const rp = {
    ratio: clamp(Number(params.ratio ?? 0.5), 1e-6, 0.999999),
    pCapRatio: clamp(Number(params.pCapRatio ?? 0.5), 0.01, 0.5),
    k: 16,
    opacityThreshold: 0.1,
  };

  const cp = {
    lamGeo: 1.0,
    lamSh: 1.0,
    nMc: Math.max(1, Math.floor(Number(params.nMc ?? 1))),
    seed: Math.floor(Number(params.seed ?? 0)),
    epsCov: Number(params.epsCov ?? 1e-8),
  };
  const keepHistory = params.keepHistory !== false;

  if (!(rp.ratio > 0 && rp.ratio < 1)) {
    throw new Error("ratio must be in (0, 1).");
  }

  onStatus("Reading splats with full SH...");
  const state = await extractSplatsWithSH(mesh, onStatus);

  const N0 = state.count;
  const target = Math.max(Math.ceil(N0 * rp.ratio), 1);
  onStatus(`Loaded ${N0} splats. Target: ${target}.`);

  onProgress({
    type: "start",
    iteration: 0,
    originalCount: N0,
    currentCount: N0,
    targetCount: target,
    progress: 0,
  });

  let cur = pruneByOpacity(state, rp.opacityThreshold, onStatus);

  emitSnapshot(onProgress, keepHistory, {
    type: "snapshot",
    stage: "prune",
    label: "Prune",
    iteration: 0,
    originalCount: N0,
    currentCount: cur.count,
    targetCount: target,
    progress: reductionProgress(N0, cur.count, target),
    state: cur,
  });

  const Z = makeGaussianSamples(cp.nMc, cp.seed);

  let iteration = 0;
  while (cur.count > target) {
    iteration += 1;
    const N = cur.count;
    onStatus(`Pass ${iteration}: ${N} splats`);

    onProgress({
      type: "pass-start",
      stage: "pass-start",
      label: `Pass ${iteration}`,
      iteration,
      originalCount: N0,
      currentCount: N,
      targetCount: target,
      progress: reductionProgress(N0, N, target),
    });

    const kEff = Math.min(Math.max(1, rp.k), Math.max(1, N - 1));
    const cache = buildPerSplatCache(cur, cp.epsCov);

    onStatus(`Pass ${iteration}: building KD-tree`);
    const tree = createKDTree(buildPointArray(cur));

    onStatus(`Pass ${iteration}: querying kNN`);
    const edges = knnUndirectedEdgesKD(tree, cur, kEff);
    tree.dispose?.();

    if (edges.length === 0) {
      onStatus("No candidate edges remain. Stopping.");
      break;
    }

    onStatus(`Pass ${iteration}: scoring ${edges.length} edges`);
    const w = await edgeCosts(edges, cur, cache, cp, Z, onStatus);

    const mergesNeeded = N - target;
    const pCap = Math.max(1, Math.floor(rp.pCapRatio * N0));
    let P = mergesNeeded > 0 ? Math.min(mergesNeeded, pCap) : null;
    const pairs = greedyPairsFromEdges(edges, w, N, P);

    onStatus(
      `Pass ${iteration}: ${edges.length} edges, ${pairs.length} pairs selected (need ${mergesNeeded})`
    );

    if (pairs.length === 0) {
      onStatus("No merge pairs selected. Stopping.");
      break;
    }

    cur = mergePairs(cur, pairs);

    emitSnapshot(onProgress, keepHistory, {
      type: "snapshot",
      stage: "pass",
      label: `Pass ${iteration}`,
      iteration,
      originalCount: N0,
      currentCount: cur.count,
      targetCount: target,
      progress: reductionProgress(N0, cur.count, target),
      state: cur,
    });

    await microYield();
  }

  onStatus(`Final splats: ${cur.count}`);

  emitSnapshot(onProgress, keepHistory, {
    type: "done",
    stage: "done",
    label: "Final",
    iteration,
    originalCount: N0,
    currentCount: cur.count,
    targetCount: target,
    progress: 1,
    state: cur,
  });

  const packed = buildPackedSplatsFromState(cur);
  attachSparkSHExtras(packed, cur);

  return {
    packed,
    originalCount: N0,
    finalCount: cur.count,
    state: cur,
  };
}

async function getSplatCount(mesh) {
  if (!mesh) return 0;
  if (mesh.initialized) await mesh.initialized;
  let count = 0;
  mesh.forEachSplat(() => { count++; });
  return count;
}

/* -------------------------------------------------------------------------- */
/* State layout                                                               */
/* -------------------------------------------------------------------------- */

function makeState(count, shDim) {
  return {
    count,
    shDim,
    mu: new Float32Array(count * 3),
    sc: new Float32Array(count * 3),
    q: new Float32Array(count * 4),
    op: new Float32Array(count),
    sh: new Float32Array(count * shDim),

    sh1Min: -1, sh1Max: 1,
    sh2Min: -1, sh2Max: 1,
    sh3Min: -1, sh3Max: 1,
  };
}


async function extractSplatsWithSH(mesh, onStatus = () => {}) {

  const tmp = [];
  mesh.forEachSplat((index, center, scales, quaternion, opacity, color) => {
    tmp.push({
      cx: Math.fround(center.x),
      cy: Math.fround(center.y),
      cz: Math.fround(center.z),
      sx: Math.max(Math.fround(scales.x), 1e-12),
      sy: Math.max(Math.fround(scales.y), 1e-12),
      sz: Math.max(Math.fround(scales.z), 1e-12),
      qw: Math.fround(quaternion.w),
      qx: Math.fround(quaternion.x),
      qy: Math.fround(quaternion.y),
      qz: Math.fround(quaternion.z),
      op: clamp(Math.fround(opacity), 0, 1),
      // SH0-only fallback
      sh0r: Math.fround(color.r),
      sh0g: Math.fround(color.g),
      sh0b: Math.fround(color.b),
    });
  });

  let decoded = null;
  try {
    decoded = tryDecodeSparkSHExtrasFromMesh(mesh, tmp.length);
  } catch (err) {
    console.warn("SH decode unavailable, falling back to SH0-only:", err);
  }

  const extraDim = decoded?.shDim ?? 0;
  const shDim = 3 + extraDim;
  const out = makeState(tmp.length, shDim);

  // If we decoded packed SH extras, propagate their ranges once.
  if (decoded) {
    out.sh1Min = decoded.sh1Min;
    out.sh1Max = decoded.sh1Max;
    out.sh2Min = decoded.sh2Min;
    out.sh2Max = decoded.sh2Max;
    out.sh3Min = decoded.sh3Min;
    out.sh3Max = decoded.sh3Max;
  }

  for (let i = 0; i < tmp.length; i++) {
    const s = tmp[i];
    const i3 = 3 * i;
    const i4 = 4 * i;
    const is = shDim * i;

    let qw = s.qw, qx = s.qx, qy = s.qy, qz = s.qz;
    const qn = Math.hypot(qw, qx, qy, qz);
    const invq = 1.0 / Math.max(qn, 1e-12);
    qw *= invq; qx *= invq; qy *= invq; qz *= invq;

    out.mu[i3] = s.cx; out.mu[i3 + 1] = s.cy; out.mu[i3 + 2] = s.cz;
    out.sc[i3] = s.sx; out.sc[i3 + 1] = s.sy; out.sc[i3 + 2] = s.sz;
    out.q[i4] = qw; out.q[i4 + 1] = qx; out.q[i4 + 2] = qy; out.q[i4 + 3] = qz;
    out.op[i] = s.op;

    out.sh[is] = s.sh0r;
    out.sh[is + 1] = s.sh0g;
    out.sh[is + 2] = s.sh0b;

    if (decoded) {
      for (let k = 0; k < extraDim; k++) {
        out.sh[is + 3 + k] = decoded.sh[i * extraDim + k];
      }
    }
  }

  onStatus(
    decoded
      ? `Decoded full SH: ${shDim} coeffs/splat`
      : `Using SH0-only fallback: ${shDim} coeffs/splat`
  );

  return out;
}

function subsetState(src, keepIdx) {
  const out = makeState(keepIdx.length, src.shDim);
  out.sh1Min = src.sh1Min; out.sh1Max = src.sh1Max;
  out.sh2Min = src.sh2Min; out.sh2Max = src.sh2Max;
  out.sh3Min = src.sh3Min; out.sh3Max = src.sh3Max;
  for (let t = 0; t < keepIdx.length; t++) {
    const i = keepIdx[t];
    copySplat(src, i, out, t);
  }
  return out;
}

function copySplat(src, i, dst, j) {
  const si3 = 3 * i, sj3 = 3 * j;
  const si4 = 4 * i, sj4 = 4 * j;
  const sis = src.shDim * i, sjs = dst.shDim * j;

  dst.mu[sj3] = src.mu[si3];
  dst.mu[sj3 + 1] = src.mu[si3 + 1];
  dst.mu[sj3 + 2] = src.mu[si3 + 2];

  dst.sc[sj3] = src.sc[si3];
  dst.sc[sj3 + 1] = src.sc[si3 + 1];
  dst.sc[sj3 + 2] = src.sc[si3 + 2];

  dst.q[sj4] = src.q[si4];
  dst.q[sj4 + 1] = src.q[si4 + 1];
  dst.q[sj4 + 2] = src.q[si4 + 2];
  dst.q[sj4 + 3] = src.q[si4 + 3];

  dst.op[j] = src.op[i];

  for (let k = 0; k < src.shDim; k++) {
    dst.sh[sjs + k] = src.sh[sis + k];
  }
}

/* -------------------------------------------------------------------------- */
/* Pruning                                                                    */
/* -------------------------------------------------------------------------- */

function pruneByOpacity(state, threshold, onStatus = () => {}) {
  const N = state.count;
  if (N === 0) return state;

  let mean = 0;
  const ops = new Array(N);
  for (let i = 0; i < N; i++) {
    const v = state.op[i];
    mean += v;
    ops[i] = v;
  }
  mean /= N;

  const median = percentileInPlace(ops, 0.5);
  const thr = Math.min(threshold, median);

  const keep = [];
  for (let i = 0; i < N; i++) {
    if (state.op[i] >= thr) keep.push(i);
  }

  onStatus(
    `Opacity mean=${mean.toFixed(4)} median=${median.toFixed(4)} threshold=${thr.toFixed(4)} kept=${keep.length}/${N}`
  );

  return subsetState(state, keep);
}

/* -------------------------------------------------------------------------- */
/* KD-tree kNN                                                                */
/* -------------------------------------------------------------------------- */

function buildPointArray(state) {
  const pts = new Array(state.count);
  for (let i = 0; i < state.count; i++) {
    const k = 3 * i;
    pts[i] = [state.mu[k], state.mu[k + 1], state.mu[k + 2]];
  }
  return pts;
}

function knnUndirectedEdgesKD(tree, state, k) {
  const N = state.count;
  const set = new Set();
  const edges = [];

  for (let i = 0; i < N; i++) {
    const m = 3 * i;
    const q = [state.mu[m], state.mu[m + 1], state.mu[m + 2]];
    const idx = tree.knn(q, k + 1); // include self
    for (let t = 0; t < idx.length; t++) {
      const j = idx[t];
      if (j === i || j < 0) continue;
      const u = i < j ? i : j;
      const v = i < j ? j : i;
      const key = `${u},${v}`;
      if (!set.has(key)) {
        set.add(key);
        edges.push([u, v]);
      }
    }
  }

  return edges;
}

/* -------------------------------------------------------------------------- */
/* Per-splat cache                                                            */
/* -------------------------------------------------------------------------- */

function buildPerSplatCache(state, epsCov) {
  const N = state.count;

  const R = new Float32Array(N * 9);
  const Rt = new Float32Array(N * 9);
  const v = new Float32Array(N * 3);
  const invdiag = new Float32Array(N * 3);
  const logdet = new Float32Array(N);
  const sigma = new Float32Array(N * 9);
  const mass = new Float32Array(N);

  for (let i = 0; i < N; i++) {
    const i3 = 3 * i;
    const i4 = 4 * i;
    const i9 = 9 * i;

    const sx = state.sc[i3];
    const sy = state.sc[i3 + 1];
    const sz = state.sc[i3 + 2];

    const vx = sx * sx + epsCov;
    const vy = sy * sy + epsCov;
    const vz = sz * sz + epsCov;

    v[i3] = vx; v[i3 + 1] = vy; v[i3 + 2] = vz;
    invdiag[i3] = 1.0 / Math.max(vx, 1e-30);
    invdiag[i3 + 1] = 1.0 / Math.max(vy, 1e-30);
    invdiag[i3 + 2] = 1.0 / Math.max(vz, 1e-30);
    logdet[i] =
      Math.log(Math.max(vx, 1e-30)) +
      Math.log(Math.max(vy, 1e-30)) +
      Math.log(Math.max(vz, 1e-30));

    quatToRotmatInto(
      state.q[i4], state.q[i4 + 1], state.q[i4 + 2], state.q[i4 + 3],
      R, i9
    );
    transpose3Into(R, i9, Rt, i9);
    sigmaFromRotVarInto(R, i9, vx, vy, vz, sigma, i9);

    mass[i] = Math.fround(TWO_PI_POW_1P5 * state.op[i] * sx * sy * sz + 1e-12);
  }

  return { R, Rt, v, invdiag, logdet, sigma, mass };
}

/* -------------------------------------------------------------------------- */
/* Edge costs                                                                 */
/* -------------------------------------------------------------------------- */

async function edgeCosts(edges, state, cache, cp, Z, onStatus = () => {}, blockEdges = 8192) {
  const M = edges.length;
  const w = new Float32Array(M);

  for (let e0 = 0; e0 < M; e0 += blockEdges) {
    const e1 = Math.min(M, e0 + blockEdges);
    for (let e = e0; e < e1; e++) {
      const [u, v] = edges[e];
      w[e] = fullCostPairCached(u, v, state, cache, cp, Z);
    }
    onStatus(`Edge costs: ${e1}/${M}`);

    // Yield periodically so the main thread can render and
    // UI (including progress snapshots) stays responsive.
    await microYield();
  }

  return w;
}

function fullCostPairCached(i, j, state, cache, cp, Z) {
  const i3 = 3 * i, j3 = 3 * j;
  const i9 = 9 * i, j9 = 9 * j;
  const is = state.shDim * i, js = state.shDim * j;

  const mux = state.mu[i3], muy = state.mu[i3 + 1], muz = state.mu[i3 + 2];
  const mvx = state.mu[j3], mvy = state.mu[j3 + 1], mvz = state.mu[j3 + 2];

  const wi = cache.mass[i];
  const wj = cache.mass[j];
  const W = wi + wj;
  const Wsafe = W > 0 ? W : 1.0;

  let pi = wi / Wsafe;
  pi = clamp(pi, 1e-12, 1 - 1e-12);
  const pj = 1.0 - pi;
  const logPi = Math.log(pi);
  const logPj = Math.log(pj);

  const mmx = pi * mux + pj * mvx;
  const mmy = pi * muy + pj * mvy;
  const mmz = pi * muz + pj * mvz;

  const dix = mux - mmx, diy = muy - mmy, diz = muz - mmz;
  const djx = mvx - mmx, djy = mvy - mmy, djz = mvz - mmz;

  const Sigm = new Float64Array(9);

  for (let a = 0; a < 9; a++) {
    Sigm[a] = pi * cache.sigma[i9 + a] + pj * cache.sigma[j9 + a];
  }

  Sigm[0] += pi * dix * dix + pj * djx * djx;
  Sigm[1] += pi * dix * diy + pj * djx * djy;
  Sigm[2] += pi * dix * diz + pj * djx * djz;
  Sigm[3] += pi * diy * dix + pj * djy * djx;
  Sigm[4] += pi * diy * diy + pj * djy * djy;
  Sigm[5] += pi * diy * diz + pj * djy * djz;
  Sigm[6] += pi * diz * dix + pj * djz * djx;
  Sigm[7] += pi * diz * diy + pj * djz * djy;
  Sigm[8] += pi * diz * diz + pj * djz * djz;

  const s01 = 0.5 * (Sigm[1] + Sigm[3]);
  const s02 = 0.5 * (Sigm[2] + Sigm[6]);
  const s12 = 0.5 * (Sigm[5] + Sigm[7]);
  Sigm[1] = Sigm[3] = s01;
  Sigm[2] = Sigm[6] = s02;
  Sigm[5] = Sigm[7] = s12;
  Sigm[0] += cp.epsCov;
  Sigm[4] += cp.epsCov;
  Sigm[8] += cp.epsCov;

  const detm = Math.max(det3Flat(Sigm, 0), 1e-30);
  const logdetm = Math.log(detm);

  const EpNegLogQ = 0.5 * (3.0 * LOG2PI + logdetm + 3.0);

  const stdix = Math.sqrt(Math.max(cache.v[i3], 0));
  const stdiy = Math.sqrt(Math.max(cache.v[i3 + 1], 0));
  const stdiz = Math.sqrt(Math.max(cache.v[i3 + 2], 0));

  const stdjx = Math.sqrt(Math.max(cache.v[j3], 0));
  const stdjy = Math.sqrt(Math.max(cache.v[j3 + 1], 0));
  const stdjz = Math.sqrt(Math.max(cache.v[j3 + 2], 0));

  let sumLogpOnI = 0.0;
  let sumLogpOnJ = 0.0;

  for (let s = 0; s < Z.length; s++) {
    const z0 = Z[s][0], z1 = Z[s][1], z2 = Z[s][2];

    const xix = mux + z0 * stdix * cache.Rt[i9] + z1 * stdiy * cache.Rt[i9 + 3] + z2 * stdiz * cache.Rt[i9 + 6];
    const xiy = muy + z0 * stdix * cache.Rt[i9 + 1] + z1 * stdiy * cache.Rt[i9 + 4] + z2 * stdiz * cache.Rt[i9 + 7];
    const xiz = muz + z0 * stdix * cache.Rt[i9 + 2] + z1 * stdiy * cache.Rt[i9 + 5] + z2 * stdiz * cache.Rt[i9 + 8];

    const xjx = mvx + z0 * stdjx * cache.Rt[j9] + z1 * stdjy * cache.Rt[j9 + 3] + z2 * stdjz * cache.Rt[j9 + 6];
    const xjy = mvy + z0 * stdjx * cache.Rt[j9 + 1] + z1 * stdjy * cache.Rt[j9 + 4] + z2 * stdjz * cache.Rt[j9 + 7];
    const xjz = mvz + z0 * stdjx * cache.Rt[j9 + 2] + z1 * stdjy * cache.Rt[j9 + 5] + z2 * stdjz * cache.Rt[j9 + 8];

    const logNiOnI = gaussLogpdfDiagrotFlat(
      xix, xiy, xiz,
      mux, muy, muz,
      cache.R, i9,
      cache.invdiag[i3], cache.invdiag[i3 + 1], cache.invdiag[i3 + 2],
      cache.logdet[i]
    );
    const logNjOnI = gaussLogpdfDiagrotFlat(
      xix, xiy, xiz,
      mvx, mvy, mvz,
      cache.R, j9,
      cache.invdiag[j3], cache.invdiag[j3 + 1], cache.invdiag[j3 + 2],
      cache.logdet[j]
    );
    sumLogpOnI += logAddExp(logPi + logNiOnI, logPj + logNjOnI);

    const logNiOnJ = gaussLogpdfDiagrotFlat(
      xjx, xjy, xjz,
      mux, muy, muz,
      cache.R, i9,
      cache.invdiag[i3], cache.invdiag[i3 + 1], cache.invdiag[i3 + 2],
      cache.logdet[i]
    );
    const logNjOnJ = gaussLogpdfDiagrotFlat(
      xjx, xjy, xjz,
      mvx, mvy, mvz,
      cache.R, j9,
      cache.invdiag[j3], cache.invdiag[j3 + 1], cache.invdiag[j3 + 2],
      cache.logdet[j]
    );
    sumLogpOnJ += logAddExp(logPi + logNiOnJ, logPj + logNjOnJ);
  }

  const Ei = sumLogpOnI / Z.length;
  const Ej = sumLogpOnJ / Z.length;
  const EpLogp = pi * Ei + pj * Ej;
  const geo = EpLogp + EpNegLogQ;

  let cSh = 0.0;
  for (let k = 0; k < state.shDim; k++) {
    const d = state.sh[is + k] - state.sh[js + k];
    cSh += d * d;
  }

  return Math.fround(cp.lamGeo * geo + cp.lamSh * cSh);
}

/* -------------------------------------------------------------------------- */
/* Pair selection                                                             */
/* -------------------------------------------------------------------------- */

function greedyPairsFromEdges(edges, w, N, P) {
  if (edges.length === 0) return [];

  const valid = [];
  for (let i = 0; i < edges.length; i++) {
    if (Number.isFinite(w[i])) valid.push(i);
  }
  if (valid.length === 0) return [];

  valid.sort((a, b) => {
    const da = w[a];
    const db = w[b];
    if (da < db) return -1;
    if (da > db) return 1;
    return a - b;
  });

  const used = new Uint8Array(N);
  const pairs = [];

  for (let t = 0; t < valid.length; t++) {
    const e = valid[t];
    const u = edges[e][0];
    const v = edges[e][1];
    if (used[u] || used[v]) continue;
    used[u] = 1;
    used[v] = 1;
    pairs.push([u, v]);
    if (P !== null && pairs.length >= P) break;
  }

  return pairs;
}

/* -------------------------------------------------------------------------- */
/* Merge                                                                      */
/* -------------------------------------------------------------------------- */

function mergePairs(state, pairs) {
  if (pairs.length === 0) return state;

  const used = new Uint8Array(state.count);
  for (let p = 0; p < pairs.length; p++) {
    used[pairs[p][0]] = 1;
    used[pairs[p][1]] = 1;
  }

  const keep = [];
  for (let i = 0; i < state.count; i++) {
    if (!used[i]) keep.push(i);
  }

  const out = makeState(keep.length + pairs.length, state.shDim);
  out.sh1Min = state.sh1Min; out.sh1Max = state.sh1Max;
  out.sh2Min = state.sh2Min; out.sh2Max = state.sh2Max;
  out.sh3Min = state.sh3Min; out.sh3Max = state.sh3Max;

  let dst = 0;
  for (let t = 0; t < keep.length; t++, dst++) {
    copySplat(state, keep[t], out, dst);
  }

  for (let p = 0; p < pairs.length; p++, dst++) {
    const i = pairs[p][0];
    const j = pairs[p][1];
    momentMatchInto(state, i, j, out, dst);
  }

  return out;
}

function momentMatchInto(state, i, j, out, dst) {
  const i3 = 3 * i, j3 = 3 * j;
  const i4 = 4 * i, j4 = 4 * j;
  const d3 = 3 * dst, d4 = 4 * dst;
  const is = state.shDim * i, js = state.shDim * j, ds = state.shDim * dst;

  const sxi = state.sc[i3], syi = state.sc[i3 + 1], szi = state.sc[i3 + 2];
  const sxj = state.sc[j3], syj = state.sc[j3 + 1], szj = state.sc[j3 + 2];

  const wi = TWO_PI_POW_1P5 * state.op[i] * sxi * syi * szi + 1e-12;
  const wj = TWO_PI_POW_1P5 * state.op[j] * sxj * syj * szj + 1e-12;
  const W = Math.max(wi + wj, 1e-12);

  const mux = (wi * state.mu[i3] + wj * state.mu[j3]) / W;
  const muy = (wi * state.mu[i3 + 1] + wj * state.mu[j3 + 1]) / W;
  const muz = (wi * state.mu[i3 + 2] + wj * state.mu[j3 + 2]) / W;

  const SigI = new Float64Array(9);
  const SigJ = new Float64Array(9);
  sigmaFromQuatScaleFlatInto(state.q, i4, sxi, syi, szi, SigI, 0);
  sigmaFromQuatScaleFlatInto(state.q, j4, sxj, syj, szj, SigJ, 0);

  const dix = state.mu[i3] - mux;
  const diy = state.mu[i3 + 1] - muy;
  const diz = state.mu[i3 + 2] - muz;

  const djx = state.mu[j3] - mux;
  const djy = state.mu[j3 + 1] - muy;
  const djz = state.mu[j3 + 2] - muz;

  const Sig = new Float64Array(9);

  for (let a = 0; a < 9; a++) {
    Sig[a] = (wi * SigI[a] + wj * SigJ[a]) / W;
  }

  Sig[0] += (wi * dix * dix + wj * djx * djx) / W;
  Sig[1] += (wi * dix * diy + wj * djx * djy) / W;
  Sig[2] += (wi * dix * diz + wj * djx * djz) / W;
  Sig[3] += (wi * diy * dix + wj * djy * djx) / W;
  Sig[4] += (wi * diy * diy + wj * djy * djy) / W;
  Sig[5] += (wi * diy * diz + wj * djy * djz) / W;
  Sig[6] += (wi * diz * dix + wj * djz * djx) / W;
  Sig[7] += (wi * diz * diy + wj * djz * djy) / W;
  Sig[8] += (wi * diz * diz + wj * djz * djz) / W;

  const s01 = 0.5 * (Sig[1] + Sig[3]);
  const s02 = 0.5 * (Sig[2] + Sig[6]);
  const s12 = 0.5 * (Sig[5] + Sig[7]);
  Sig[1] = Sig[3] = s01;
  Sig[2] = Sig[6] = s02;
  Sig[5] = Sig[7] = s12;
  Sig[0] += 1e-8;
  Sig[4] += 1e-8;
  Sig[8] += 1e-8;

  const ev = eigenSymmetric3x3Flat(Sig);
  let vals = ev.values;
  let vecs = ev.vectors;

  const order = [0, 1, 2].sort((a, b) => vals[b] - vals[a]);
  vals = order.map((k) => Math.max(vals[k], 1e-18));

  const R = new Float64Array(9);
  for (let c = 0; c < 3; c++) {
    const src = order[c];
    R[0 + c] = vecs[0 + src];
    R[3 + c] = vecs[3 + src];
    R[6 + c] = vecs[6 + src];
  }

  if (det3Flat(R, 0) < 0) {
    R[2] *= -1; R[5] *= -1; R[8] *= -1;
  }

  const q = rotmatToQuatFlat(R, 0);

  out.mu[d3] = Math.fround(mux);
  out.mu[d3 + 1] = Math.fround(muy);
  out.mu[d3 + 2] = Math.fround(muz);

  out.sc[d3] = Math.fround(Math.sqrt(vals[0]));
  out.sc[d3 + 1] = Math.fround(Math.sqrt(vals[1]));
  out.sc[d3 + 2] = Math.fround(Math.sqrt(vals[2]));

  out.q[d4] = Math.fround(q[0]);
  out.q[d4 + 1] = Math.fround(q[1]);
  out.q[d4 + 2] = Math.fround(q[2]);
  out.q[d4 + 3] = Math.fround(q[3]);

  out.op[dst] = Math.fround(clamp(
    state.op[i] + state.op[j] - state.op[i] * state.op[j],
    0, 1
  ));

  for (let k = 0; k < state.shDim; k++) {
    out.sh[ds + k] = Math.fround((wi * state.sh[is + k] + wj * state.sh[js + k]) / W);
  }
}

/* -------------------------------------------------------------------------- */
/* Output                                                                     */
/* -------------------------------------------------------------------------- */

function buildPackedSplatsFromState(state) {
  const packed = new PackedSplats({ maxSplats: state.count });
  const center = new THREE.Vector3();
  const scales = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const color = new THREE.Color();

  for (let i = 0; i < state.count; i++) {
    const i3 = 3 * i;
    const i4 = 4 * i;
    const is = state.shDim * i;

    const r = clamp(state.sh[is] ?? 0, 0, 1);
    const g = clamp(state.sh[is + 1] ?? 0, 0, 1);
    const b = clamp(state.sh[is + 2] ?? 0, 0, 1);

    center.set(state.mu[i3], state.mu[i3 + 1], state.mu[i3 + 2]);
    scales.set(state.sc[i3], state.sc[i3 + 1], state.sc[i3 + 2]);
    quat.set(state.q[i4 + 1], state.q[i4 + 2], state.q[i4 + 3], state.q[i4]); // xyzw
    color.setRGB(r, g, b);

    packed.pushSplat(center, scales, quat, state.op[i], color);
  }

  packed.needsUpdate = true;
  return packed;
}

function attachSparkSHExtras(packed, state) {
  const count = state.count;
  const shDim = state.shDim;
  const remaining = shDim - 3;
  const hasSh1 = remaining >= 9;
  const hasSh2 = remaining >= 9 + 15;
  const hasSh3 = remaining >= 9 + 15 + 21;
  if (!hasSh1 && !hasSh2 && !hasSh3) {
    if (!packed.extra) packed.extra = {};
    delete packed.extra.sh1;
    delete packed.extra.sh2;
    delete packed.extra.sh3;
    delete packed.extra.encoding;
    return;
  }
  const sh1Min = typeof state.sh1Min === 'number' ? state.sh1Min : -1;
  const sh1Max = typeof state.sh1Max === 'number' ? state.sh1Max : 1;
  const sh2Min = typeof state.sh2Min === 'number' ? state.sh2Min : -1;
  const sh2Max = typeof state.sh2Max === 'number' ? state.sh2Max : 1;
  const sh3Min = typeof state.sh3Min === 'number' ? state.sh3Min : -1;
  const sh3Max = typeof state.sh3Max === 'number' ? state.sh3Max : 1;

  const sh1Scale = (sh1Max - sh1Min) !== 0 ? 126 / (sh1Max - sh1Min) : 0;
  const sh2Scale = (sh2Max - sh2Min) !== 0 ? 254 / (sh2Max - sh2Min) : 0;
  const sh3Scale = (sh3Max - sh3Min) !== 0 ?  62 / (sh3Max - sh3Min) : 0;
  const sh1Mid   = (sh1Min + sh1Max) * 0.5;
  const sh2Mid   = (sh2Min + sh2Max) * 0.5;
  const sh3Mid   = (sh3Min + sh3Max) * 0.5;

  if (!packed.extra) packed.extra = {};

  if (hasSh1) {
    const sh1Arr = new Uint32Array(count * 2);
    const offset = 3;
    for (let i = 0; i < count; i++) {
      const base = i * shDim + offset;
      const wordBase = i * 2;
      for (let j = 0; j < 9; j++) {
        let s;
        const val = state.sh[base + j];
        if (Number.isNaN(val)) s = 0;
        else if (sh1Scale !== 0) s = Math.round((val - sh1Mid) * sh1Scale);
        else s = 0;
        if (s > 63) s = 63;
        if (s < -63) s = -63;
        const encoded  = s & 0x7f;
        const bitStart = j * 7;
        const wIndex   = Math.floor(bitStart / 32);
        const bitOff   = bitStart - wIndex * 32;
        if (bitOff <= 25) {
          sh1Arr[wordBase + wIndex] |= encoded << bitOff;
        } else {
          const bitsInWord = 32 - bitOff;
          sh1Arr[wordBase + wIndex]     |= (encoded << bitOff) >>> 0;
          sh1Arr[wordBase + wIndex + 1] |=  encoded >>> bitsInWord;
        }
      }
    }
    packed.extra.sh1 = sh1Arr;
  } else delete packed.extra.sh1;

  if (hasSh2) {
    const sh2Arr = new Uint32Array(count * 4);
    const offset = 3 + (hasSh1 ? 9 : 0);
    for (let i = 0; i < count; i++) {
      const base = i * shDim + offset;
      const wordBase = i * 4;
      let coeffIdx = 0;
      for (let w = 0; w < 4; w++) {
        let word = 0;
        for (let b = 0; b < 4; b++) {
          let s;
          if (coeffIdx < 15) {
            const val = state.sh[base + coeffIdx];
            if (Number.isNaN(val)) s = 0;
            else if (sh2Scale !== 0) s = Math.round((val - sh2Mid) * sh2Scale);
            else s = 0;
            if (s > 127) s = 127;
            if (s < -127) s = -127;
          } else s = 0;
          const byteVal = s & 0xff;
          word |= byteVal << (8 * b);
          coeffIdx++;
        }
        sh2Arr[wordBase + w] = word >>> 0;
      }
    }
    packed.extra.sh2 = sh2Arr;
  } else delete packed.extra.sh2;

  if (hasSh3) {
    const sh3Arr = new Uint32Array(count * 4);
    const offset = 3 + (hasSh1 ? 9 : 0) + (hasSh2 ? 15 : 0);
    for (let i = 0; i < count; i++) {
      const base = i * shDim + offset;
      const wordBase = i * 4;
      for (let j = 0; j < 21; j++) {
        let s;
        const val = state.sh[base + j];
        if (Number.isNaN(val)) s = 0;
        else if (sh3Scale !== 0) s = Math.round((val - sh3Mid) * sh3Scale);
        else s = 0;
        if (s > 31) s = 31;
        if (s < -31) s = -31;
        const encoded  = s & 0x3f;
        const bitStart = j * 6;
        const wIndex   = Math.floor(bitStart / 32);
        const bitOff   = bitStart - wIndex * 32;
        if (bitOff <= 26) {
          sh3Arr[wordBase + wIndex] |= encoded << bitOff;
        } else {
          const bitsInWord = 32 - bitOff;
          sh3Arr[wordBase + wIndex]     |= (encoded << bitOff) >>> 0;
          sh3Arr[wordBase + wIndex + 1] |=  encoded >>> bitsInWord;
        }
      }
    }
    packed.extra.sh3 = sh3Arr;
  } else delete packed.extra.sh3;

  const encoding = {};
  if (hasSh1) { encoding.sh1Min = sh1Min; encoding.sh1Max = sh1Max; }
  if (hasSh2) { encoding.sh2Min = sh2Min; encoding.sh2Max = sh2Max; }
  if (hasSh3) { encoding.sh3Min = sh3Min; encoding.sh3Max = sh3Max; }
  packed.extra.encoding = encoding;
}

function tryDecodeSparkSHExtrasFromMesh(mesh, count) {
  const packed = mesh?.packedSplats ?? mesh?.splats ?? mesh;
  if (!packed || !packed.extra) return null;
  const extra = packed.extra;
  const hasSh1 = extra.sh1 instanceof Uint32Array;
  const hasSh2 = extra.sh2 instanceof Uint32Array;
  const hasSh3 = extra.sh3 instanceof Uint32Array;
  if (!hasSh1 && !hasSh2 && !hasSh3) return null;

  const enc = extra.encoding ?? {};
  const sh1Min = typeof enc.sh1Min === 'number' ? enc.sh1Min : -1;
  const sh1Max = typeof enc.sh1Max === 'number' ? enc.sh1Max : 1;
  const sh2Min = typeof enc.sh2Min === 'number' ? enc.sh2Min : -1;
  const sh2Max = typeof enc.sh2Max === 'number' ? enc.sh2Max : 1;
  const sh3Min = typeof enc.sh3Min === 'number' ? enc.sh3Min : -1;
  const sh3Max = typeof enc.sh3Max === 'number' ? enc.sh3Max : 1;
  const sh1ScaleInv = (sh1Max - sh1Min) / 126;
  const sh2ScaleInv = (sh2Max - sh2Min) / 254;
  const sh3ScaleInv = (sh3Max - sh3Min) / 62;
  const sh1Mid = (sh1Min + sh1Max) * 0.5;
  const sh2Mid = (sh2Min + sh2Max) * 0.5;
  const sh3Mid = (sh3Min + sh3Max) * 0.5;

  const dims = [];
  if (hasSh1) dims.push(9);
  if (hasSh2) dims.push(15);
  if (hasSh3) dims.push(21);
  const shDim = dims.reduce((a, b) => a + b, 0);
  const out = new Float32Array(count * shDim);

  function decode7bits(val) { return (val & 0x40) ? (val | ~0x7f) : val; }
  function decode6bits(val) { return (val & 0x20) ? (val | ~0x3f) : val; }

  if (hasSh1) {
    const sh1 = extra.sh1;
    for (let i = 0; i < count; i++) {
      const baseOut = i * shDim;
      const baseWord = i * 2;
      for (let j = 0; j < 9; j++) {
        const bitStart = j * 7;
        const wordIndex = Math.floor(bitStart / 32);
        const bitOffset = bitStart - wordIndex * 32;
        let bits;
        if (bitOffset <= 32 - 7) {
          bits = (sh1[baseWord + wordIndex] >>> bitOffset) & 0x7f;
        } else {
          const bitsInWord = 32 - bitOffset;
          const lowPart  = (sh1[baseWord + wordIndex] >>> bitOffset) & ((1 << bitsInWord) - 1);
          const highPart = sh1[baseWord + wordIndex + 1] & ((1 << (7 - bitsInWord)) - 1);
          bits = (highPart << bitsInWord) | lowPart;
        }
        const signed = decode7bits(bits);
        out[baseOut + j] = signed * sh1ScaleInv + sh1Mid;
      }
    }
  }

  if (hasSh2) {
    const sh2 = extra.sh2;
    const off = hasSh1 ? 9 : 0;
    for (let i = 0; i < count; i++) {
      const baseOut = i * shDim + off;
      const baseWord = i * 4;
      let outIdx = 0;
      for (let w = 0; w < 4; w++) {
        const word = sh2[baseWord + w];
        for (let b = 0; b < 4; b++) {
          let byteVal = (word >> (8 * b)) & 0xff;
          if (byteVal & 0x80) byteVal |= ~0xff;
          if (outIdx < 15) {
            out[baseOut + outIdx] = byteVal * sh2ScaleInv + sh2Mid;
          }
          outIdx++;
        }
      }
    }
  }

  if (hasSh3) {
    const sh3 = extra.sh3;
    const off = (hasSh1 ? 9 : 0) + (hasSh2 ? 15 : 0);
    for (let i = 0; i < count; i++) {
      const baseOut = i * shDim + off;
      const baseWord = i * 4;
      for (let j = 0; j < 21; j++) {
        const bitStart  = j * 6;
        const wordIndex = Math.floor(bitStart / 32);
        const bitOffset = bitStart - wordIndex * 32;
        let bits;
        if (bitOffset <= 32 - 6) {
          bits = (sh3[baseWord + wordIndex] >>> bitOffset) & 0x3f;
        } else {
          const bitsInWord = 32 - bitOffset;
          const lowPart  = (sh3[baseWord + wordIndex] >>> bitOffset) & ((1 << bitsInWord) - 1);
          const highPart = sh3[baseWord + wordIndex + 1] & ((1 << (6 - bitsInWord)) - 1);
          bits = (highPart << bitsInWord) | lowPart;
        }
        const signed = decode6bits(bits);
        out[baseOut + j] = signed * sh3ScaleInv + sh3Mid;
      }
    }
  }

  return { shDim, sh: out, hasSh1, hasSh2, hasSh3, sh1Min, sh1Max, sh2Min, sh2Max, sh3Min, sh3Max };
}

/* -------------------------------------------------------------------------- */
/* Math                                                                       */
/* -------------------------------------------------------------------------- */

function quatToRotmatInto(w, x, y, z, out, o) {
  const ww = w * w, xx = x * x, yy = y * y, zz = z * z;
  const wx = w * x, wy = w * y, wz = w * z;
  const xy = x * y, xz = x * z, yz = y * z;

  out[o] = 1 - 2 * (yy + zz);
  out[o + 1] = 2 * (xy - wz);
  out[o + 2] = 2 * (xz + wy);

  out[o + 3] = 2 * (xy + wz);
  out[o + 4] = 1 - 2 * (xx + zz);
  out[o + 5] = 2 * (yz - wx);

  out[o + 6] = 2 * (xz - wy);
  out[o + 7] = 2 * (yz + wx);
  out[o + 8] = 1 - 2 * (xx + yy);
}

function transpose3Into(src, so, dst, doff) {
  dst[doff] = src[so];
  dst[doff + 1] = src[so + 3];
  dst[doff + 2] = src[so + 6];
  dst[doff + 3] = src[so + 1];
  dst[doff + 4] = src[so + 4];
  dst[doff + 5] = src[so + 7];
  dst[doff + 6] = src[so + 2];
  dst[doff + 7] = src[so + 5];
  dst[doff + 8] = src[so + 8];
}

function sigmaFromRotVarInto(R, r, vx, vy, vz, out, o) {
  const r00 = R[r], r01 = R[r + 1], r02 = R[r + 2];
  const r10 = R[r + 3], r11 = R[r + 4], r12 = R[r + 5];
  const r20 = R[r + 6], r21 = R[r + 7], r22 = R[r + 8];

  out[o] = r00 * r00 * vx + r01 * r01 * vy + r02 * r02 * vz;
  out[o + 1] = r00 * r10 * vx + r01 * r11 * vy + r02 * r12 * vz;
  out[o + 2] = r00 * r20 * vx + r01 * r21 * vy + r02 * r22 * vz;

  out[o + 3] = out[o + 1];
  out[o + 4] = r10 * r10 * vx + r11 * r11 * vy + r12 * r12 * vz;
  out[o + 5] = r10 * r20 * vx + r11 * r21 * vy + r12 * r22 * vz;

  out[o + 6] = out[o + 2];
  out[o + 7] = out[o + 5];
  out[o + 8] = r20 * r20 * vx + r21 * r21 * vy + r22 * r22 * vz;
}

function sigmaFromQuatScaleFlatInto(q, qo, sx, sy, sz, out, oo) {
  const R = new Float64Array(9);
  quatToRotmatInto(q[qo], q[qo + 1], q[qo + 2], q[qo + 3], R, 0);
  sigmaFromRotVarInto(R, 0, sx * sx, sy * sy, sz * sz, out, oo);
}

function gaussLogpdfDiagrotFlat(x, y, z, mx, my, mz, R, ro, invx, invy, invz, logdet) {
  const dx = x - mx;
  const dy = y - my;
  const dz = z - mz;

  const y0 = dx * R[ro] + dy * R[ro + 3] + dz * R[ro + 6];
  const y1 = dx * R[ro + 1] + dy * R[ro + 4] + dz * R[ro + 7];
  const y2 = dx * R[ro + 2] + dy * R[ro + 5] + dz * R[ro + 8];

  const quad = y0 * y0 * invx + y1 * y1 * invy + y2 * y2 * invz;
  return -0.5 * (3.0 * LOG2PI + logdet + quad);
}

function rotmatToQuatFlat(R, o) {
  const m00 = R[o], m11 = R[o + 4], m22 = R[o + 8];
  const tr = m00 + m11 + m22;
  let qw, qx, qy, qz;

  if (tr > 0) {
    const S = Math.sqrt(tr + 1.0) * 2.0;
    qw = 0.25 * S;
    qx = (R[o + 7] - R[o + 5]) / S;
    qy = (R[o + 2] - R[o + 6]) / S;
    qz = (R[o + 3] - R[o + 1]) / S;
  } else if (R[o] > R[o + 4] && R[o] > R[o + 8]) {
    const S = Math.sqrt(1.0 + R[o] - R[o + 4] - R[o + 8]) * 2.0;
    qw = (R[o + 7] - R[o + 5]) / S;
    qx = 0.25 * S;
    qy = (R[o + 1] + R[o + 3]) / S;
    qz = (R[o + 2] + R[o + 6]) / S;
  } else if (R[o + 4] > R[o + 8]) {
    const S = Math.sqrt(1.0 + R[o + 4] - R[o] - R[o + 8]) * 2.0;
    qw = (R[o + 2] - R[o + 6]) / S;
    qx = (R[o + 1] + R[o + 3]) / S;
    qy = 0.25 * S;
    qz = (R[o + 5] + R[o + 7]) / S;
  } else {
    const S = Math.sqrt(1.0 + R[o + 8] - R[o] - R[o + 4]) * 2.0;
    qw = (R[o + 3] - R[o + 1]) / S;
    qx = (R[o + 2] + R[o + 6]) / S;
    qy = (R[o + 5] + R[o + 7]) / S;
    qz = 0.25 * S;
  }

  const n = Math.hypot(qw, qx, qy, qz);
  const inv = 1.0 / Math.max(n, 1e-12);
  return [qw * inv, qx * inv, qy * inv, qz * inv];
}

function det3Flat(A, o) {
  const a00 = A[o], a01 = A[o + 1], a02 = A[o + 2];
  const a10 = A[o + 3], a11 = A[o + 4], a12 = A[o + 5];
  const a20 = A[o + 6], a21 = A[o + 7], a22 = A[o + 8];
  return (
    a00 * (a11 * a22 - a12 * a21) -
    a01 * (a10 * a22 - a12 * a20) +
    a02 * (a10 * a21 - a11 * a20)
  );
}

function eigenSymmetric3x3Flat(Ain) {
  const A = new Float64Array(Ain);
  const V = new Float64Array([
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
  ]);

  for (let iter = 0; iter < 24; iter++) {
    let p = 0, q = 1;
    let maxAbs = Math.abs(A[1]);

    if (Math.abs(A[2]) > maxAbs) { p = 0; q = 2; maxAbs = Math.abs(A[2]); }
    if (Math.abs(A[5]) > maxAbs) { p = 1; q = 2; maxAbs = Math.abs(A[5]); }

    if (maxAbs < 1e-12) break;

    const pp = 3 * p + p;
    const qq = 3 * q + q;
    const pq = 3 * p + q;
    const qp = 3 * q + p;

    const app = A[pp];
    const aqq = A[qq];
    const apq = A[pq];

    const tau = (aqq - app) / (2 * apq);
    const t = Math.sign(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    for (let k = 0; k < 3; k++) {
      if (k === p || k === q) continue;
      const kp = 3 * k + p;
      const kq = 3 * k + q;
      const pk = 3 * p + k;
      const qk = 3 * q + k;
      const akp = A[kp];
      const akq = A[kq];
      A[kp] = c * akp - s * akq;
      A[pk] = A[kp];
      A[kq] = s * akp + c * akq;
      A[qk] = A[kq];
    }

    A[pp] = c * c * app - 2 * s * c * apq + s * s * aqq;
    A[qq] = s * s * app + 2 * s * c * apq + c * c * aqq;
    A[pq] = 0;
    A[qp] = 0;

    for (let k = 0; k < 3; k++) {
      const kp = 3 * k + p;
      const kq = 3 * k + q;
      const vkp = V[kp];
      const vkq = V[kq];
      V[kp] = c * vkp - s * vkq;
      V[kq] = s * vkp + c * vkq;
    }
  }

  return {
    values: [A[0], A[4], A[8]],
    vectors: V, // columns
  };
}

/* -------------------------------------------------------------------------- */
/* Utils                                                                      */
/* -------------------------------------------------------------------------- */

function makeGaussianSamples(n, seed) {
  const rand = mulberry32(seed >>> 0);
  const out = [];
  while (out.length < n) {
    const u1 = Math.max(rand(), 1e-12);
    const u2 = rand();
    const u3 = Math.max(rand(), 1e-12);
    const u4 = rand();

    const r1 = Math.sqrt(-2.0 * Math.log(u1));
    const t1 = 2.0 * Math.PI * u2;
    const r2 = Math.sqrt(-2.0 * Math.log(u3));
    const t2 = 2.0 * Math.PI * u4;

    out.push([
      Math.fround(r1 * Math.cos(t1)),
      Math.fround(r1 * Math.sin(t1)),
      Math.fround(r2 * Math.cos(t2)),
    ]);
  }
  return out;
}

function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function percentileInPlace(xs, p) {
  xs.sort((a, b) => a - b);
  if (xs.length === 0) return 0;
  const t = (xs.length - 1) * p;
  const i = Math.floor(t);
  const j = Math.min(i + 1, xs.length - 1);
  const w = t - i;
  return xs[i] * (1 - w) + xs[j] * w;
}

function logAddExp(a, b) {
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

function clamp(x, lo, hi) {
  return Math.min(hi, Math.max(lo, x));
}

function microYield() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function cloneState(src) {
  const out = makeState(src.count, src.shDim);
  out.mu.set(src.mu);
  out.sc.set(src.sc);
  out.q.set(src.q);
  out.op.set(src.op);
  out.sh.set(src.sh);
  out.sh1Min = src.sh1Min;
  out.sh1Max = src.sh1Max;
  out.sh2Min = src.sh2Min;
  out.sh2Max = src.sh2Max;
  out.sh3Min = src.sh3Min;
  out.sh3Max = src.sh3Max;
  return out;
}

function reductionProgress(originalCount, currentCount, targetCount) {
  const totalRemovedNeeded = Math.max(originalCount - targetCount, 1);
  const removedSoFar = Math.max(0, originalCount - currentCount);
  return clamp(removedSoFar / totalRemovedNeeded, 0, 1);
}

function emitSnapshot(onProgress, keepHistory, payload) {
  onProgress({
    ...payload,
    snapshot: keepHistory ? cloneState(payload.state) : null,
  });
}

export function packedFromState(state) {
  const packed = buildPackedSplatsFromState(state);
  attachSparkSHExtras(packed, state);
  return packed;
}

/* -------------------------------------------------------------------------- */
/* PLY export (same format as loaded PLY: binary, 3DGS properties)             */
/* -------------------------------------------------------------------------- */

const SH_C0 = 0.28209479177387814;

function logit(p) {
  const eps = 1e-8;
  p = clamp(p, eps, 1 - eps);
  return Math.log(p / (1 - p));
}

/**
 * Number of f_rest coefficients to export so PLY matches loaded SH degree.
 * shDim 3 → 0 (SH0), 12 → 9 (SH1), 27 → 24 (SH2), 48 → 45 (SH3).
 */
function fRestCountFromShDim(shDim) {
  const rest = Math.max(0, (shDim ?? 3) - 3);
  return rest;
}

/**
 * Build binary little-endian PLY bytes from simplification state so the file
 * can be re-loaded with the same format as the original PLY (3D Gaussian Splatting).
 * Header and vertex layout match the loaded PLY: SH0 → no f_rest, SH1 → f_rest_0..8, etc.
 * @param {object} state - State from simplify (mu, sc, q, op, sh, shDim)
 * @returns {ArrayBuffer} PLY file bytes
 */
export function stateToPlyBytes(state) {
  if (!state || !state.count) {
    return new ArrayBuffer(0);
  }

  const N = state.count;
  const shDim = state.shDim ?? 3;
  const numFRest = fRestCountFromShDim(shDim);
  const vertexSize =
    3 + 3 + 1 + 1 + 3 + 4 + numFRest; // xyz, f_dc_0..2, opacity, scale_0..2, rot_0..3, f_rest (0/9/24/45)
  const vertexBytes = vertexSize * 4; // 4 bytes per float
  const headerLines = [
    "ply",
    "format binary_little_endian 1.0",
    "element vertex " + N,
    "property float x",
    "property float y",
    "property float z",
    "property float f_dc_0",
    "property float f_dc_1",
    "property float f_dc_2",
    "property float opacity",
    "property float scale_0",
    "property float scale_1",
    "property float scale_2",
    "property float rot_0",
    "property float rot_1",
    "property float rot_2",
    "property float rot_3",
  ];
  for (let i = 0; i < numFRest; i++) {
    headerLines.push("property float f_rest_" + i);
  }
  headerLines.push("end_header");

  const header = headerLines.join("\n") + "\n";
  const headerBytes = new TextEncoder().encode(header);
  const totalBytes = headerBytes.length + N * vertexBytes;
  const buffer = new ArrayBuffer(totalBytes);
  const view = new DataView(buffer);
  const u8 = new Uint8Array(buffer);
  u8.set(headerBytes, 0);

  let offset = headerBytes.length;

  for (let i = 0; i < N; i++) {
    const i3 = 3 * i;
    const i4 = 4 * i;
    const is = shDim * i;

    // x, y, z
    view.setFloat32(offset, state.mu[i3], true);
    offset += 4;
    view.setFloat32(offset, state.mu[i3 + 1], true);
    offset += 4;
    view.setFloat32(offset, state.mu[i3 + 2], true);
    offset += 4;

    // f_dc_0, f_dc_1, f_dc_2  (file value = (rgb - 0.5) / SH_C0)
    const r = state.sh[is] ?? 0;
    const g = state.sh[is + 1] ?? 0;
    const b = state.sh[is + 2] ?? 0;
    view.setFloat32(offset, (r - 0.5) / SH_C0, true);
    offset += 4;
    view.setFloat32(offset, (g - 0.5) / SH_C0, true);
    offset += 4;
    view.setFloat32(offset, (b - 0.5) / SH_C0, true);
    offset += 4;

    // opacity (logit so loader's sigmoid gives back [0,1])
    view.setFloat32(offset, logit(state.op[i]), true);
    offset += 4;

    // scale_0, scale_1, scale_2 (PLY stores log(scale); loader applies exp() when loading)
    const scaleEps = 1e-12;
    view.setFloat32(offset, Math.log(Math.max(state.sc[i3], scaleEps)), true);
    offset += 4;
    view.setFloat32(offset, Math.log(Math.max(state.sc[i3 + 1], scaleEps)), true);
    offset += 4;
    view.setFloat32(offset, Math.log(Math.max(state.sc[i3 + 2], scaleEps)), true);
    offset += 4;

    // rot_0, rot_1, rot_2, rot_3 (w, x, y, z)
    view.setFloat32(offset, state.q[i4], true);
    offset += 4;
    view.setFloat32(offset, state.q[i4 + 1], true);
    offset += 4;
    view.setFloat32(offset, state.q[i4 + 2], true);
    offset += 4;
    view.setFloat32(offset, state.q[i4 + 3], true);
    offset += 4;

    // f_rest_0 .. f_rest_(numFRest-1) — only as many as loaded (SH0: none, SH1: 9, SH2: 24, SH3: 45)
    for (let k = 0; k < numFRest; k++) {
      const val = state.sh[is + 3 + k] ?? 0;
      view.setFloat32(offset, val, true);
      offset += 4;
    }
  }

  return buffer;
}

export { getSplatCount };