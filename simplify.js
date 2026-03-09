import * as THREE from "three";
import { PackedSplats } from "@sparkjsdev/spark";

export function quatNormalize(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1.0;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

export function quatToMat3(qIn) {
  const [x, y, z, w] = quatNormalize(qIn);
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const wx = w * x;
  const wy = w * y;
  const wz = w * z;
  return [
    [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
    [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
    [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
  ];
}

export function mat3Transpose(A) {
  return [
    [A[0][0], A[1][0], A[2][0]],
    [A[0][1], A[1][1], A[2][1]],
    [A[0][2], A[1][2], A[2][2]],
  ];
}

export function mat3Mul(A, B) {
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
    }
  }
  return C;
}

export function mat3Diag(v) {
  return [
    [v[0], 0, 0],
    [0, v[1], 0],
    [0, 0, v[2]],
  ];
}

export function mat3Add(A, B) {
  return [
    [A[0][0] + B[0][0], A[0][1] + B[0][1], A[0][2] + B[0][2]],
    [A[1][0] + B[1][0], A[1][1] + B[1][1], A[1][2] + B[1][2]],
    [A[2][0] + B[2][0], A[2][1] + B[2][1], A[2][2] + B[2][2]],
  ];
}

export function mat3Scale(A, s) {
  return [
    [A[0][0] * s, A[0][1] * s, A[0][2] * s],
    [A[1][0] * s, A[1][1] * s, A[1][2] * s],
    [A[2][0] * s, A[2][1] * s, A[2][2] * s],
  ];
}

export function outer3(v) {
  return [
    [v[0] * v[0], v[0] * v[1], v[0] * v[2]],
    [v[1] * v[0], v[1] * v[1], v[1] * v[2]],
    [v[2] * v[0], v[2] * v[1], v[2] * v[2]],
  ];
}

export function covFromScaleQuat(scales, quat) {
  const R = quatToMat3(quat);
  const Rt = mat3Transpose(R);
  const D = mat3Diag([
    scales[0] * scales[0],
    scales[1] * scales[1],
    scales[2] * scales[2],
  ]);
  return mat3Mul(mat3Mul(R, D), Rt);
}

export function symmetricEigenJacobi(Ain) {
  let A = [
    [...Ain[0]],
    [...Ain[1]],
    [...Ain[2]],
  ];
  let V = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];

  for (let iter = 0; iter < 16; iter++) {
    let p = 0;
    let q = 1;
    let maxOff = Math.abs(A[0][1]);

    if (Math.abs(A[0][2]) > maxOff) {
      maxOff = Math.abs(A[0][2]);
      p = 0;
      q = 2;
    }
    if (Math.abs(A[1][2]) > maxOff) {
      maxOff = Math.abs(A[1][2]);
      p = 1;
      q = 2;
    }
    if (maxOff < 1e-10) break;

    const appVal = A[p][p];
    const aqq = A[q][q];
    const apq = A[p][q];
    const tau = (aqq - appVal) / (2 * apq);
    const t = Math.sign(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    for (let k = 0; k < 3; k++) {
      if (k !== p && k !== q) {
        const akp = A[k][p];
        const akq = A[k][q];
        A[k][p] = c * akp - s * akq;
        A[p][k] = A[k][p];
        A[k][q] = s * akp + c * akq;
        A[q][k] = A[k][q];
      }
    }

    const newApp = c * c * appVal - 2 * s * c * apq + s * s * aqq;
    const newAqq = s * s * appVal + 2 * s * c * apq + c * c * aqq;
    A[p][p] = newApp;
    A[q][q] = newAqq;
    A[p][q] = 0;
    A[q][p] = 0;

    for (let k = 0; k < 3; k++) {
      const vkp = V[k][p];
      const vkq = V[k][q];
      V[k][p] = c * vkp - s * vkq;
      V[k][q] = s * vkp + c * vkq;
    }
  }

  let evals = [A[0][0], A[1][1], A[2][2]];
  let evecs = [
    [V[0][0], V[0][1], V[0][2]],
    [V[1][0], V[1][1], V[1][2]],
    [V[2][0], V[2][1], V[2][2]],
  ];

  const order = [0, 1, 2].sort((a, b) => evals[b] - evals[a]);
  evals = order.map((i) => Math.max(evals[i], 1e-12));
  evecs = [
    [evecs[0][order[0]], evecs[0][order[1]], evecs[0][order[2]]],
    [evecs[1][order[0]], evecs[1][order[1]], evecs[1][order[2]]],
    [evecs[2][order[0]], evecs[2][order[1]], evecs[2][order[2]]],
  ];

  const det =
    evecs[0][0] * (evecs[1][1] * evecs[2][2] - evecs[1][2] * evecs[2][1]) -
    evecs[0][1] * (evecs[1][0] * evecs[2][2] - evecs[1][2] * evecs[2][0]) +
    evecs[0][2] * (evecs[1][0] * evecs[2][1] - evecs[1][1] * evecs[2][0]);

  if (det < 0) {
    evecs[0][2] *= -1;
    evecs[1][2] *= -1;
    evecs[2][2] *= -1;
  }

  return { evals, evecs };
}

export function mat3ToQuat(R) {
  const m00 = R[0][0];
  const m01 = R[0][1];
  const m02 = R[0][2];
  const m10 = R[1][0];
  const m11 = R[1][1];
  const m12 = R[1][2];
  const m20 = R[2][0];
  const m21 = R[2][1];
  const m22 = R[2][2];
  let x;
  let y;
  let z;
  let w;
  const trace = m00 + m11 + m22;

  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1.0);
    w = 0.25 / s;
    x = (m21 - m12) * s;
    y = (m02 - m20) * s;
    z = (m10 - m01) * s;
  } else if (m00 > m11 && m00 > m22) {
    const s = 2.0 * Math.sqrt(1.0 + m00 - m11 - m22);
    w = (m21 - m12) / s;
    x = 0.25 * s;
    y = (m01 + m10) / s;
    z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = 2.0 * Math.sqrt(1.0 + m11 - m00 - m22);
    w = (m02 - m20) / s;
    x = (m01 + m10) / s;
    y = 0.25 * s;
    z = (m12 + m21) / s;
  } else {
    const s = 2.0 * Math.sqrt(1.0 + m22 - m00 - m11);
    w = (m10 - m01) / s;
    x = (m02 + m20) / s;
    y = (m12 + m21) / s;
    z = 0.25 * s;
  }

  return quatNormalize([x, y, z, w]);
}

export function opacityMedian(splats) {
  const vals = splats.map((s) => s.opacity).sort((a, b) => a - b);
  const m = Math.floor(vals.length / 2);
  return vals.length % 2 ? vals[m] : 0.5 * (vals[m - 1] + vals[m]);
}

export function unpackSplatsFromMesh(mesh) {
  const out = [];
  mesh.packedSplats.forEachSplat((index, center, scales, quaternion, opacity, color) => {
    out.push({
      center: [center.x, center.y, center.z],
      scales: [
        Math.max(scales.x, 1e-5),
        Math.max(scales.y, 1e-5),
        Math.max(scales.z, 1e-5),
      ],
      quat: quatNormalize([quaternion.x, quaternion.y, quaternion.z, quaternion.w]),
      opacity: Math.min(Math.max(opacity, 0), 1),
      color: [
        Math.min(Math.max(color.r, 0), 1),
        Math.min(Math.max(color.g, 0), 1),
        Math.min(Math.max(color.b, 0), 1),
      ],
    });
  });
  return out;
}

export function approximateKnnIndices(splats, k) {
  const N = splats.length;
  const ids = Array.from({ length: N }, (_, i) => i);
  ids.sort((a, b) => splats[a].center[0] - splats[b].center[0]);

  const posInSorted = new Array(N);
  for (let p = 0; p < N; p++) posInSorted[ids[p]] = p;

  const windowRadius = Math.max(16, 4 * k);
  const nbr = Array.from({ length: N }, () => []);

  for (let i = 0; i < N; i++) {
    const p = posInSorted[i];
    const c = splats[i].center;
    const cand = [];

    const lo = Math.max(0, p - windowRadius);
    const hi = Math.min(N - 1, p + windowRadius);

    for (let t = lo; t <= hi; t++) {
      const j = ids[t];
      if (j === i) continue;
      const d0 = c[0] - splats[j].center[0];
      const d1 = c[1] - splats[j].center[1];
      const d2 = c[2] - splats[j].center[2];
      const dist2 = d0 * d0 + d1 * d1 + d2 * d2;
      cand.push({ j, dist2 });
    }

    cand.sort((a, b) => a.dist2 - b.dist2);
    nbr[i] = cand.slice(0, k).map((x) => x.j);
  }

  return nbr;
}

export function knnUndirectedEdges(nbr) {
  const edges = [];
  const seen = new Set();
  for (let i = 0; i < nbr.length; i++) {
    for (const j of nbr[i]) {
      const u = Math.min(i, j);
      const v = Math.max(i, j);
      if (u === v) continue;
      const key = `${u},${v}`;
      if (!seen.has(key)) {
        seen.add(key);
        edges.push([u, v]);
      }
    }
  }
  return edges;
}

export function edgeCost(si, sj, lamGeo, lamColor) {
  const dx = si.center[0] - sj.center[0];
  const dy = si.center[1] - sj.center[1];
  const dz = si.center[2] - sj.center[2];
  const d2 = dx * dx + dy * dy + dz * dz;

  const sx = 0.5 * (si.scales[0] + sj.scales[0]);
  const sy = 0.5 * (si.scales[1] + sj.scales[1]);
  const sz = 0.5 * (si.scales[2] + sj.scales[2]);
  const norm = (sx * sx + sy * sy + sz * sz) + 1e-8;

  const dsx = Math.log(si.scales[0]) - Math.log(sj.scales[0]);
  const dsy = Math.log(si.scales[1]) - Math.log(sj.scales[1]);
  const dsz = Math.log(si.scales[2]) - Math.log(sj.scales[2]);
  const scalePenalty = dsx * dsx + dsy * dsy + dsz * dsz;

  const dc0 = si.color[0] - sj.color[0];
  const dc1 = si.color[1] - sj.color[1];
  const dc2 = si.color[2] - sj.color[2];
  const colorPenalty = dc0 * dc0 + dc1 * dc1 + dc2 * dc2;

  const opacityPenalty = Math.abs(si.opacity - sj.opacity);

  return lamGeo * (d2 / norm + 0.15 * scalePenalty + 0.05 * opacityPenalty)
       + lamColor * colorPenalty;
}

export function greedyPairsFromEdges(edges, costs, N, P) {
  const order = Array.from({ length: edges.length }, (_, i) => i)
    .filter((i) => Number.isFinite(costs[i]))
    .sort((a, b) => costs[a] - costs[b]);

  const used = new Array(N).fill(false);
  const pairs = [];

  for (const ei of order) {
    const [u, v] = edges[ei];
    if (used[u] || used[v]) continue;
    used[u] = true;
    used[v] = true;
    pairs.push([u, v]);
    if (P != null && pairs.length >= P) break;
  }
  return pairs;
}

export function momentMatch(a, b) {
  const weightA = Math.pow(2 * Math.PI, 1.5) * a.opacity * a.scales[0] * a.scales[1] * a.scales[2] + 1e-12;
  const weightB = Math.pow(2 * Math.PI, 1.5) * b.opacity * b.scales[0] * b.scales[1] * b.scales[2] + 1e-12;
  const W = weightA + weightB;

  const center = [
    (weightA * a.center[0] + weightB * b.center[0]) / W,
    (weightA * a.center[1] + weightB * b.center[1]) / W,
    (weightA * a.center[2] + weightB * b.center[2]) / W,
  ];

  const SigA = covFromScaleQuat(a.scales, a.quat);
  const SigB = covFromScaleQuat(b.scales, b.quat);

  const da = [
    a.center[0] - center[0],
    a.center[1] - center[1],
    a.center[2] - center[2],
  ];
  const db = [
    b.center[0] - center[0],
    b.center[1] - center[1],
    b.center[2] - center[2],
  ];

  let Sig = mat3Add(
    mat3Scale(mat3Add(SigA, outer3(da)), weightA / W),
    mat3Scale(mat3Add(SigB, outer3(db)), weightB / W),
  );

  Sig = [
    [0.5 * (Sig[0][0] + Sig[0][0]) + 1e-8, 0.5 * (Sig[0][1] + Sig[1][0]),        0.5 * (Sig[0][2] + Sig[2][0])],
    [0.5 * (Sig[1][0] + Sig[0][1]),        0.5 * (Sig[1][1] + Sig[1][1]) + 1e-8, 0.5 * (Sig[1][2] + Sig[2][1])],
    [0.5 * (Sig[2][0] + Sig[0][2]),        0.5 * (Sig[2][1] + Sig[1][2]),        0.5 * (Sig[2][2] + Sig[2][2]) + 1e-8],
  ];

  const { evals, evecs } = symmetricEigenJacobi(Sig);
  const scales = [
    Math.sqrt(Math.max(evals[0], 1e-12)),
    Math.sqrt(Math.max(evals[1], 1e-12)),
    Math.sqrt(Math.max(evals[2], 1e-12)),
  ];
  const quat = mat3ToQuat(evecs);
  const opacity = Math.min(1, Math.max(0, a.opacity + b.opacity - a.opacity * b.opacity));

  const color = [
    (weightA * a.color[0] + weightB * b.color[0]) / W,
    (weightA * a.color[1] + weightB * b.color[1]) / W,
    (weightA * a.color[2] + weightB * b.color[2]) / W,
  ];

  return { center, scales, quat, opacity, color };
}

export function mergePairs(splats, pairs) {
  if (!pairs.length) return splats;

  const used = new Array(splats.length).fill(false);
  const merged = [];

  for (const [i, j] of pairs) {
    used[i] = true;
    used[j] = true;
    merged.push(momentMatch(splats[i], splats[j]));
  }

  const kept = [];
  for (let i = 0; i < splats.length; i++) {
    if (!used[i]) kept.push(splats[i]);
  }

  return kept.concat(merged);
}

export function buildPackedSplats(splats) {
  const packed = new PackedSplats({ maxSplats: splats.length });
  for (const s of splats) {
    packed.pushSplat(
      new THREE.Vector3(s.center[0], s.center[1], s.center[2]),
      new THREE.Vector3(s.scales[0], s.scales[1], s.scales[2]),
      new THREE.Quaternion(s.quat[0], s.quat[1], s.quat[2], s.quat[3]),
      s.opacity,
      new THREE.Color(s.color[0], s.color[1], s.color[2]),
    );
  }
  return packed;
}

export function sh0FromRgb(rgb) {
  const SH_C0 = 0.28209479177387814;
  return [
    (rgb[0] - 0.5) / SH_C0,
    (rgb[1] - 0.5) / SH_C0,
    (rgb[2] - 0.5) / SH_C0,
  ];
}

export function logit(x) {
  const v = Math.min(1 - 1e-6, Math.max(1e-6, x));
  return Math.log(v / (1 - v));
}

export function exportMinimalPly(splats) {
  const lines = [];
  lines.push("ply");
  lines.push("format ascii 1.0");
  lines.push(`element vertex ${splats.length}`);
  lines.push("property float x");
  lines.push("property float y");
  lines.push("property float z");
  lines.push("property float nx");
  lines.push("property float ny");
  lines.push("property float nz");
  lines.push("property float f_dc_0");
  lines.push("property float f_dc_1");
  lines.push("property float f_dc_2");
  lines.push("property float opacity");
  lines.push("property float scale_0");
  lines.push("property float scale_1");
  lines.push("property float scale_2");
  lines.push("property float rot_0");
  lines.push("property float rot_1");
  lines.push("property float rot_2");
  lines.push("property float rot_3");
  lines.push("end_header");

  for (const s of splats) {
    const dc = sh0FromRgb(s.color);
    lines.push([
      s.center[0], s.center[1], s.center[2],
      0, 0, 0,
      dc[0], dc[1], dc[2],
      logit(s.opacity),
      Math.log(Math.max(s.scales[0], 1e-8)),
      Math.log(Math.max(s.scales[1], 1e-8)),
      Math.log(Math.max(s.scales[2], 1e-8)),
      s.quat[0], s.quat[1], s.quat[2], s.quat[3],
    ].join(" "));
  }

  return new Blob([lines.join("\n")], { type: "application/octet-stream" });
}

export async function simplifyMesh(mesh, params, onProgress = () => {}) {
  onProgress("Decoding splats from Spark container...");

  let splats = unpackSplatsFromMesh(mesh);
  const N0 = splats.length;
  const target = Math.max(1, Math.ceil(N0 * params.ratio));

  if (!splats.length) {
    throw new Error("Empty splat set.");
  }

  const medianOpacity = opacityMedian(splats);
  const pruneThreshold = Math.min(params.opacityThreshold, medianOpacity);
  splats = splats.filter((s) => s.opacity >= pruneThreshold);

  onProgress(`Pruned by opacity: ${N0} → ${splats.length}. Simplifying...`);

  let iteration = 0;
  const maxPasses = 32;
  const initialPostPrune = splats.length;

  while (splats.length > target && iteration < maxPasses) {
    const N = splats.length;
    const kEff = Math.min(Math.max(1, params.k), Math.max(1, N - 1));

    onProgress(`Pass ${iteration + 1}: ${N} splats, k=${kEff}`);

    const nbr = approximateKnnIndices(splats, kEff);
    const edges = knnUndirectedEdges(nbr);
    if (!edges.length) break;

    const costs = edges.map(([u, v]) =>
      edgeCost(splats[u], splats[v], params.lamGeo, params.lamColor)
    );

    const mergesNeeded = N - target;
    let P = mergesNeeded;
    P = Math.min(P, Math.max(1, Math.floor((initialPostPrune - target) / 8) || 1));

    const pairs = greedyPairsFromEdges(edges, costs, N, P);
    if (!pairs.length) break;

    splats = mergePairs(splats, pairs);
    iteration++;
  }

  onProgress(`Repacking ${splats.length} simplified splats...`);

  return {
    originalCount: N0,
    finalCount: splats.length,
    splats,
    packed: buildPackedSplats(splats),
    blob: exportMinimalPly(splats),
  };
}