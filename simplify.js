import * as THREE from "three";
import createKDTree from "https://esm.sh/static-kdtree";
import { PackedSplats } from "@sparkjsdev/spark";

const SH_C0 = 0.28209479177387814;
const TWO_PI_POW_1P5 = Math.pow(2.0 * Math.PI, 1.5);
const LOG2PI = Math.log(2.0 * Math.PI);

export async function simplifyMesh(mesh, params = {}, onStatus = () => {}) {
  if (!mesh) throw new Error("simplifyMesh: mesh is required.");
  if (mesh.initialized) await mesh.initialized;

  const rp = {
    ratio: clamp(Number(params.ratio ?? 0.5), 1e-6, 0.999999),
    k: Math.max(1, Math.floor(Number(params.k ?? 16))),
    opacityThreshold: clamp(Number(params.opacityThreshold ?? 0.1), 0.0, 1.0),
  };

  const cp = {
    lamGeo: Math.max(0, Number(params.lamGeo ?? 1.0)),
    lamColor: Math.max(0, Number(params.lamColor ?? 1.0)),
    nMc: Math.max(1, Math.floor(Number(params.nMc ?? 1))),
    seed: Math.floor(Number(params.seed ?? 0)),
    epsCov: Number(params.epsCov ?? 1e-8),
  };

  if (!(rp.ratio > 0 && rp.ratio < 1)) {
    throw new Error("ratio must be in (0, 1).");
  }

  onStatus("Reading splats...");
  const state = extractSplatsFromMesh(mesh);

  const N0 = state.count;
  const target = Math.max(Math.ceil(N0 * rp.ratio), 1);
  onStatus(`Loaded ${N0} splats. Target: ${target}.`);

  let cur = pruneByOpacity(state, rp.opacityThreshold, onStatus);

  const Z = makeGaussianSamples(cp.nMc, cp.seed);

  let iteration = 0;
  while (cur.count > target) {
    iteration += 1;
    const N = cur.count;
    onStatus(`Pass ${iteration}: ${N} splats`);

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
    const w = edgeCosts(edges, cur, cache, cp, Z, onStatus);

    const mergesNeeded = N - target;
    let P = mergesNeeded > 0 ? mergesNeeded : null;
    if (P !== null) {
      P = Math.min(P, Math.max(1, Math.floor((N0 - target) / 8)));
    }

    const pairs = greedyPairsFromEdges(edges, w, N, P);

    onStatus(
      `Pass ${iteration}: ${edges.length} edges, ${pairs.length} pairs selected (need ${mergesNeeded})`
    );

    if (pairs.length === 0) {
      onStatus("No merge pairs selected. Stopping.");
      break;
    }

    cur = mergePairs(cur, pairs);
    await microYield();
  }

  onStatus(`Final splats: ${cur.count}`);

  const packed = buildPackedSplats(cur);
  const blob = buildGsplatPlyBlob(cur);

  return {
    packed,
    blob,
    originalCount: N0,
    finalCount: cur.count,
  };
}

/* -------------------------------------------------------------------------- */
/* State layout                                                               */
/* -------------------------------------------------------------------------- */

function makeState(count) {
  return {
    count,
    mu: new Float32Array(count * 3),
    sc: new Float32Array(count * 3),
    q: new Float32Array(count * 4), // wxyz
    op: new Float32Array(count),
    color: new Float32Array(count * 3),
  };
}

function extractSplatsFromMesh(mesh) {
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
      r: clamp(Math.fround(color.r), 0, 1),
      g: clamp(Math.fround(color.g), 0, 1),
      b: clamp(Math.fround(color.b), 0, 1),
    });
  });

  const out = makeState(tmp.length);
  for (let i = 0; i < tmp.length; i++) {
    const s = tmp[i];
    const qi = 4 * i;
    const mi = 3 * i;

    let qw = s.qw, qx = s.qx, qy = s.qy, qz = s.qz;
    const qn = Math.hypot(qw, qx, qy, qz);
    const invq = 1.0 / Math.max(qn, 1e-12);
    qw *= invq; qx *= invq; qy *= invq; qz *= invq;

    out.mu[mi] = s.cx; out.mu[mi + 1] = s.cy; out.mu[mi + 2] = s.cz;
    out.sc[mi] = s.sx; out.sc[mi + 1] = s.sy; out.sc[mi + 2] = s.sz;
    out.q[qi] = qw; out.q[qi + 1] = qx; out.q[qi + 2] = qy; out.q[qi + 3] = qz;
    out.op[i] = s.op;
    out.color[mi] = s.r; out.color[mi + 1] = s.g; out.color[mi + 2] = s.b;
  }
  return out;
}

function subsetState(src, keepIdx) {
  const out = makeState(keepIdx.length);
  for (let t = 0; t < keepIdx.length; t++) {
    const i = keepIdx[t];
    copySplat(src, i, out, t);
  }
  return out;
}

function copySplat(src, i, dst, j) {
  const si3 = 3 * i, sj3 = 3 * j;
  const si4 = 4 * i, sj4 = 4 * j;

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

  dst.color[sj3] = src.color[si3];
  dst.color[sj3 + 1] = src.color[si3 + 1];
  dst.color[sj3 + 2] = src.color[si3 + 2];
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
      const key = u * N + v;
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
    logdet[i] = Math.log(Math.max(vx, 1e-30)) + Math.log(Math.max(vy, 1e-30)) + Math.log(Math.max(vz, 1e-30));

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

function edgeCosts(edges, state, cache, cp, Z, onStatus = () => {}, blockEdges = 8192) {
  const M = edges.length;
  const w = new Float32Array(M);

  for (let e0 = 0; e0 < M; e0 += blockEdges) {
    const e1 = Math.min(M, e0 + blockEdges);
    for (let e = e0; e < e1; e++) {
      const [u, v] = edges[e];
      w[e] = fullCostPairCached(u, v, state, cache, cp, Z);
    }
    onStatus(`Edge costs: ${e1}/${M}`);
  }

  return w;
}

function fullCostPairCached(i, j, state, cache, cp, Z) {
  const i3 = 3 * i, j3 = 3 * j;
  const i9 = 9 * i, j9 = 9 * j;

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
    Sigm[a] =
      pi * cache.sigma[i9 + a] +
      pj * cache.sigma[j9 + a];
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

  // sym + epsI
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

  const dr = state.color[i3] - state.color[j3];
  const dg = state.color[i3 + 1] - state.color[j3 + 1];
  const db = state.color[i3 + 2] - state.color[j3 + 2];
  const cColor = dr * dr + dg * dg + db * db;

  return Math.fround(cp.lamGeo * geo + cp.lamColor * cColor);
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

  const out = makeState(keep.length + pairs.length);

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
  let vecs = ev.vectors; // columns

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

  out.op[dst] = Math.fround(clamp(state.op[i] + state.op[j] - state.op[i] * state.op[j], 0, 1));

  out.color[d3] = Math.fround((wi * state.color[i3] + wj * state.color[j3]) / W);
  out.color[d3 + 1] = Math.fround((wi * state.color[i3 + 1] + wj * state.color[j3 + 1]) / W);
  out.color[d3 + 2] = Math.fround((wi * state.color[i3 + 2] + wj * state.color[j3 + 2]) / W);
}

/* -------------------------------------------------------------------------- */
/* Output                                                                     */
/* -------------------------------------------------------------------------- */

function buildPackedSplats(state) {
  const packed = new PackedSplats({ maxSplats: state.count });
  const center = new THREE.Vector3();
  const scales = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const color = new THREE.Color();

  for (let i = 0; i < state.count; i++) {
    const i3 = 3 * i;
    const i4 = 4 * i;

    center.set(state.mu[i3], state.mu[i3 + 1], state.mu[i3 + 2]);
    scales.set(state.sc[i3], state.sc[i3 + 1], state.sc[i3 + 2]);
    quat.set(state.q[i4 + 1], state.q[i4 + 2], state.q[i4 + 3], state.q[i4]); // xyzw
    color.setRGB(state.color[i3], state.color[i3 + 1], state.color[i3 + 2]);

    packed.pushSplat(center, scales, quat, state.op[i], color);
  }

  packed.needsUpdate = true;
  return packed;
}

function buildGsplatPlyBlob(state) {
  const headerLines = [
    "ply",
    "format binary_little_endian 1.0",
    `element vertex ${state.count}`,
    "property float x",
    "property float y",
    "property float z",
    "property float nx",
    "property float ny",
    "property float nz",
    "property float f_dc_0",
    "property float f_dc_1",
    "property float f_dc_2",
    ...Array.from({ length: 45 }, (_, i) => `property float f_rest_${i}`),
    "property float opacity",
    "property float scale_0",
    "property float scale_1",
    "property float scale_2",
    "property float rot_0",
    "property float rot_1",
    "property float rot_2",
    "property float rot_3",
    "end_header\n",
  ];
  const header = new TextEncoder().encode(headerLines.join("\n"));

  const floatsPerVertex = 62;
  const body = new ArrayBuffer(state.count * floatsPerVertex * 4);
  const dv = new DataView(body);

  let off = 0;
  for (let i = 0; i < state.count; i++) {
    const i3 = 3 * i;
    const i4 = 4 * i;

    const fdc0 = (state.color[i3] - 0.5) / SH_C0;
    const fdc1 = (state.color[i3 + 1] - 0.5) / SH_C0;
    const fdc2 = (state.color[i3 + 2] - 0.5) / SH_C0;

    writeF32(dv, off, state.mu[i3]); off += 4;
    writeF32(dv, off, state.mu[i3 + 1]); off += 4;
    writeF32(dv, off, state.mu[i3 + 2]); off += 4;

    writeF32(dv, off, 0); off += 4;
    writeF32(dv, off, 0); off += 4;
    writeF32(dv, off, 0); off += 4;

    writeF32(dv, off, fdc0); off += 4;
    writeF32(dv, off, fdc1); off += 4;
    writeF32(dv, off, fdc2); off += 4;

    for (let k = 0; k < 45; k++) {
      writeF32(dv, off, 0);
      off += 4;
    }

    writeF32(dv, off, logit(clamp(state.op[i], 1e-6, 1 - 1e-6))); off += 4;
    writeF32(dv, off, Math.log(Math.max(state.sc[i3], 1e-12))); off += 4;
    writeF32(dv, off, Math.log(Math.max(state.sc[i3 + 1], 1e-12))); off += 4;
    writeF32(dv, off, Math.log(Math.max(state.sc[i3 + 2], 1e-12))); off += 4;

    writeF32(dv, off, state.q[i4]); off += 4;
    writeF32(dv, off, state.q[i4 + 1]); off += 4;
    writeF32(dv, off, state.q[i4 + 2]); off += 4;
    writeF32(dv, off, state.q[i4 + 3]); off += 4;
  }

  return new Blob([header, body], { type: "application/octet-stream" });
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
    let u1 = Math.max(rand(), 1e-12);
    let u2 = rand();
    let u3 = Math.max(rand(), 1e-12);
    let u4 = rand();

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

function logit(p) {
  const x = clamp(p, 1e-6, 1 - 1e-6);
  return Math.log(x / (1 - x));
}

function writeF32(dv, off, x) {
  dv.setFloat32(off, x, true);
}

function microYield() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}