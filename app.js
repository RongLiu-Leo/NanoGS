import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SplatMesh } from "@sparkjsdev/spark";
import { simplifyMesh, packedFromState } from "./simplify.js";

const app = document.getElementById("app");
const layerLeft = document.getElementById("layer-left");
const layerRight = document.getElementById("layer-right");
const interactionLayer = document.getElementById("interaction-layer");
const divider = document.getElementById("divider");
const handle = document.getElementById("handle");

const labelLeft = document.getElementById("label-left");
const labelRight = document.getElementById("label-right");
const statusEl = document.getElementById("status");

const inputOriginal = document.getElementById("input-original");
const btnSimplify = document.getElementById("btn-simplify");
const btnDownload = document.getElementById("btn-download");

const ratioEl = document.getElementById("ratio");
const ratioNumberEl = document.getElementById("ratio-number");
const kEl = document.getElementById("k");
const opacityThresholdEl = document.getElementById("opacity-threshold");
const lamGeoEl = document.getElementById("lam-geo");
const lamShEl = document.getElementById("lam-sh");

const progressPercentEl = document.getElementById("progress-percent");
const progressBarFillEl = document.getElementById("progress-bar-fill");
const progressBarEl = document.getElementById("progress-bar");
const progressBarHandleEl = document.getElementById("progress-bar-handle");
const statOriginalEl = document.getElementById("stat-original");
const statTargetEl = document.getElementById("stat-target");
const statCurrentEl = document.getElementById("stat-current");
const statShownEl = document.getElementById("stat-shown");
const trackListEl = document.getElementById("track-list");

const rendererLeft = new THREE.WebGLRenderer({ antialias: true, alpha: false });
const rendererRight = new THREE.WebGLRenderer({ antialias: true, alpha: false });

rendererLeft.setPixelRatio(Math.min(window.devicePixelRatio, 2));
rendererRight.setPixelRatio(Math.min(window.devicePixelRatio, 2));
rendererLeft.setClearColor(0x111111, 1);
rendererRight.setClearColor(0x111111, 1);

layerLeft.appendChild(rendererLeft.domElement);
layerRight.appendChild(rendererRight.domElement);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 3);

const controls = new OrbitControls(camera, interactionLayer);
controls.enableDamping = true;
controls.target.set(0, 0, 0);

const sceneLeft = new THREE.Scene();
const sceneRight = new THREE.Scene();

let leftMesh = null;
let rightMesh = null;

let originalName = "example.ply";
let simplifiedName = "example_0.1.ply";

let originalBytes = null;
let latestSimplifyResult = null;

let historyEntries = [];
let activeHistoryIndex = -1;

let isSimplifying = false;
let autoFollowProgress = true;

ratioEl.addEventListener("input", () => {
  ratioNumberEl.value = ratioEl.value;
});

ratioNumberEl.addEventListener("input", () => {
  ratioEl.value = ratioNumberEl.value;
});

function setStatus(text) {
  statusEl.textContent = text;
}

function updateLabels() {
  labelLeft.textContent = `Original: ${originalName}`;
  labelRight.textContent = `NanoGS: ${simplifiedName}`;
}

function clearMesh(scene, mesh) {
  if (!mesh) return;
  scene.remove(mesh);
  if (typeof mesh.dispose === "function") mesh.dispose();
}

async function buildMeshFromBytes(bytes) {
  const mesh = new SplatMesh({ fileBytes: bytes });
  await mesh.initialized;
  mesh.position.set(0, 0, 0);
  return mesh;
}

async function buildMeshFromPacked(packed) {
  const mesh = new SplatMesh({ packedSplats: packed });
  await mesh.initialized;
  mesh.position.set(0, 0, 0);
  return mesh;
}

async function loadOriginalBytes(bytes, name) {
  setStatus(`Loading original: ${name}`);
  const mesh = await buildMeshFromBytes(bytes);
  clearMesh(sceneLeft, leftMesh);
  leftMesh = mesh;
  sceneLeft.add(leftMesh);
  originalBytes = bytes;
  originalName = name;
  updateLabels();

  latestSimplifyResult = null;
  btnDownload.disabled = true;
  clearHistory();

  setStatus(`Loaded original: ${name}`);
}

async function loadSimplifiedBytes(bytes, name) {
  setStatus(`Loading simplified: ${name}`);
  const mesh = await buildMeshFromBytes(bytes);
  clearMesh(sceneRight, rightMesh);
  rightMesh = mesh;
  sceneRight.add(rightMesh);
  simplifiedName = name;
  updateLabels();
  setStatus(`Loaded simplified: ${name}`);
}

async function loadSimplifiedPacked(packed, name) {
  setStatus(`Building simplified view: ${name}`);
  const mesh = await buildMeshFromPacked(packed);
  clearMesh(sceneRight, rightMesh);
  rightMesh = mesh;
  sceneRight.add(rightMesh);
  simplifiedName = name;
  updateLabels();
  setStatus(`Simplified view ready: ${name}`);
}

function resize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  rendererLeft.setSize(w, h, false);
  rendererRight.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

let split = 0.5;

function setSplit(t) {
  split = Math.max(0, Math.min(1, t));
  const pct = split * 100;
  divider.style.left = `${pct}%`;
  handle.style.left = `${pct}%`;

  const rightInset = 100 - pct;
  const clip = `inset(0 ${rightInset}% 0 0)`;
  layerLeft.style.clipPath = clip;
  layerLeft.style.webkitClipPath = clip;
}

function clientXToSplit(clientX) {
  const rect = app.getBoundingClientRect();
  return (clientX - rect.left) / rect.width;
}

let dragging = false;

handle.addEventListener("pointerdown", (e) => {
  dragging = true;
  handle.setPointerCapture(e.pointerId);
  controls.enabled = false;
  setSplit(clientXToSplit(e.clientX));
  e.preventDefault();
  e.stopPropagation();
});

handle.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  setSplit(clientXToSplit(e.clientX));
  e.preventDefault();
  e.stopPropagation();
});

function stopDrag(e) {
  if (!dragging) return;
  dragging = false;
  controls.enabled = true;
  try {
    handle.releasePointerCapture(e.pointerId);
  } catch (_) {}
  e.preventDefault();
  e.stopPropagation();
}

handle.addEventListener("pointerup", stopDrag);
handle.addEventListener("pointercancel", stopDrag);

function clearHistory() {
  historyEntries = [];
  activeHistoryIndex = -1;
  trackListEl.innerHTML = "";
  updateProgressVisual(0);
  statOriginalEl.textContent = "-";
  statTargetEl.textContent = "-";
  statCurrentEl.textContent = "-";
  statShownEl.textContent = "-";
}

function formatInt(n) {
  return Number.isFinite(n) ? n.toLocaleString() : "-";
}

function updateProgressHeader(entry, options = {}) {
  if (!entry) return;

  const {
    mode = "auto",          // "auto" | "live" | "preview"
    previewIndex = -1,
  } = options;

  let progress01 = entry.progress ?? 0;

  if (mode === "live") {
    progress01 = entry.progress ?? 0;
  } else if (mode === "preview") {
    progress01 = displayedProgressFromIndex(previewIndex);
  } else {
    if (isSimplifying && autoFollowProgress) {
      progress01 = entry.progress ?? 0;
    } else if (previewIndex >= 0) {
      progress01 = displayedProgressFromIndex(previewIndex);
    } else if (activeHistoryIndex >= 0) {
      progress01 = displayedProgressFromIndex(activeHistoryIndex);
    }
  }

  updateProgressVisual(progress01);

  statOriginalEl.textContent = formatInt(entry.originalCount);
  statTargetEl.textContent = formatInt(entry.targetCount);
  statCurrentEl.textContent = formatInt(entry.currentCount);
}

function renderHistory() {
  trackListEl.innerHTML = "";

  historyEntries.forEach((entry, index) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "track-item" + (index === activeHistoryIndex ? " active" : "");

    const pct = Math.round((entry.progress ?? 0) * 100);
    btn.innerHTML = `
      <div class="track-row-top">
        <div class="track-title">${entry.label}</div>
        <div class="track-count">${formatInt(entry.currentCount)} splats</div>
      </div>
      <div class="track-row-bottom">
        ${entry.stage}${entry.iteration ? ` · iter ${entry.iteration}` : ""} · ${pct}%
      </div>
    `;

    btn.addEventListener("click", async () => {
      await previewHistoryEntry(index);
    });

    trackListEl.appendChild(btn);
  });
}

async function previewHistoryEntry(index) {
  await scrubToProgressIndex(index);
}

function progressClientXToIndex(clientX) {
  const rect = progressBarEl.getBoundingClientRect();
  const t = Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(rect.width, 1)));
  if (historyEntries.length <= 1) return 0;
  return Math.round(t * (historyEntries.length - 1));
}

let progressDragging = false;
let progressPreviewToken = 0;

async function scrubToProgressIndex(index) {
  index = Math.max(0, Math.min(index, historyEntries.length - 1));
  if (index === activeHistoryIndex) return;

  autoFollowProgress = false;
  const token = ++progressPreviewToken;
  const entry = historyEntries[index];
  if (!entry) return;

  activeHistoryIndex = index;
  renderHistory();
  updateProgressHeader(entry, {
    mode: "preview",
    previewIndex: index,
  });
  statShownEl.textContent = formatInt(entry.currentCount);

  try {
    const packed = packedFromState(entry.snapshot);
    const stageName = `${originalName.replace(/\.ply$/i, "")}_${entry.label.toLowerCase().replace(/\s+/g, "_")}.ply`;
    const mesh = await buildMeshFromPacked(packed);

    if (token !== progressPreviewToken) {
      if (typeof mesh.dispose === "function") mesh.dispose();
      return;
    }

    clearMesh(sceneRight, rightMesh);
    rightMesh = mesh;
    sceneRight.add(rightMesh);

    simplifiedName = stageName;
    updateLabels();
    setStatus(`Showing ${entry.label}: ${entry.currentCount} splats`);
  } catch (err) {
    console.error(err);
    setStatus(`Failed to preview ${entry.label}: ${err.message}`);
  }
}

progressBarEl.addEventListener("pointerdown", async (e) => {
  if (!historyEntries.length) return;
  progressDragging = true;
  progressBarEl.setPointerCapture(e.pointerId);
  const index = progressClientXToIndex(e.clientX);
  await scrubToProgressIndex(index);
  e.preventDefault();
  e.stopPropagation();
});

progressBarEl.addEventListener("pointermove", async (e) => {
  if (!progressDragging || !historyEntries.length) return;
  const index = progressClientXToIndex(e.clientX);
  await scrubToProgressIndex(index);
  e.preventDefault();
  e.stopPropagation();
});

function stopProgressDrag(e) {
  if (!progressDragging) return;
  progressDragging = false;
  try {
    progressBarEl.releasePointerCapture(e.pointerId);
  } catch (_) {}
  e.preventDefault();
  e.stopPropagation();
}

progressBarEl.addEventListener("pointerup", stopProgressDrag);
progressBarEl.addEventListener("pointercancel", stopProgressDrag);

function handleSimplifyProgress(evt) {
  if (!evt) return;

  if (evt.type === "start") {
    statOriginalEl.textContent = formatInt(evt.originalCount);
    statTargetEl.textContent = formatInt(evt.targetCount);
    statCurrentEl.textContent = formatInt(evt.currentCount);
    statShownEl.textContent = "-";
    updateProgressVisual(0);
    return;
  }

  if (evt.type === "snapshot" || evt.type === "done") {
    historyEntries.push({
      label: evt.label,
      stage: evt.stage,
      iteration: evt.iteration,
      originalCount: evt.originalCount,
      currentCount: evt.currentCount,
      targetCount: evt.targetCount,
      progress: evt.progress,
      snapshot: evt.snapshot,
    });

    statCurrentEl.textContent = formatInt(evt.currentCount);

    if (autoFollowProgress) {
      activeHistoryIndex = historyEntries.length - 1;
      statShownEl.textContent = formatInt(evt.currentCount);
      updateProgressHeader(evt, { mode: "live" });
    } else {
      updateProgressHeader(evt, {
        mode: "preview",
        previewIndex: activeHistoryIndex,
      });
    }

    renderHistory();
  }
}

async function simplifyCurrent() {
  if (!leftMesh) {
    setStatus("No original splat loaded.");
    return;
  }

  const params = {
    ratio: Math.min(0.95, Math.max(0.02, Number(ratioEl.value))),
    k: Math.max(2, Math.floor(Number(kEl.value))),
    opacityThreshold: Math.min(1, Math.max(0, Number(opacityThresholdEl.value))),
    lamGeo: Math.max(0, Number(lamGeoEl.value)),
    lamSh: Math.max(0, Number(lamShEl.value)),
    keepHistory: true,
  };

  btnSimplify.disabled = true;
  btnDownload.disabled = true;
  clearHistory();

  isSimplifying = true;
  autoFollowProgress = true;
  activeHistoryIndex = -1;

    try {
    const t0 = performance.now();

    const result = await simplifyMesh(leftMesh, params, setStatus, handleSimplifyProgress);
    latestSimplifyResult = result;

    const ratioTag = params.ratio.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
    simplifiedName = originalName.replace(/\.ply$/i, `_${ratioTag}.ply`);

    await loadSimplifiedPacked(result.packed, simplifiedName);

    if (historyEntries.length > 0) {
      activeHistoryIndex = historyEntries.length - 1;
      renderHistory();
      statShownEl.textContent = formatInt(historyEntries[historyEntries.length - 1].currentCount);
      updateProgressHeader(historyEntries[historyEntries.length - 1], {
        mode: "preview",
        previewIndex: activeHistoryIndex,
      });
    }

    isSimplifying = false;
    autoFollowProgress = true;

    const t1 = performance.now();
    setStatus(
      `Done. ${result.originalCount} → ${result.finalCount} splats in ${((t1 - t0) / 1000).toFixed(2)} s.`
    );

    btnDownload.disabled = false;
  } catch (err) {
    isSimplifying = false;
    console.error(err);
    setStatus(`Simplify failed: ${err.message}`);
  } finally {
    isSimplifying = false;
    btnSimplify.disabled = false;
  }
}

inputOriginal.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  try {
    const bytes = await file.arrayBuffer();
    await loadOriginalBytes(bytes, file.name);
  } catch (err) {
    console.error(err);
    setStatus(`Failed to load original file: ${file.name}`);
  }
});

btnSimplify.addEventListener("click", simplifyCurrent);

btnDownload.addEventListener("click", () => {
  if (!latestSimplifyResult?.packed) {
    setStatus("Download export is not implemented yet for in-browser packed results.");
    return;
  }
  setStatus("Download export is not implemented yet for in-browser packed results.");
});

window.addEventListener("resize", resize);

function animate() {
  controls.update();
  rendererRight.render(sceneRight, camera);
  rendererLeft.render(sceneLeft, camera);
  requestAnimationFrame(animate);
}

async function fetchBytes(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch ${url}`);
  return await resp.arrayBuffer();
}

async function boot() {
  try {
    const [demoOriginal, demoSimplified] = await Promise.all([
      fetchBytes("./example.ply"),
      fetchBytes("./example_0.1.ply"),
    ]);

    await loadOriginalBytes(demoOriginal, "example.ply");
    await loadSimplifiedBytes(demoSimplified, "example_0.1.ply");
    updateLabels();
    resize();
    setSplit(0.5);
    animate();
  } catch (err) {
    console.error(err);
    setStatus("Boot failed. Check that example.ply and example_0.1.ply exist next to index.html.");
    resize();
    setSplit(0.5);
    animate();
  }
}

function displayedProgressFromIndex(index) {
  if (!historyEntries.length || index < 0) return 0;
  if (historyEntries.length === 1) return 1;
  return index / (historyEntries.length - 1);
}

function updateProgressVisual(progress01) {
  const pct = Math.round(progress01 * 100);
  progressPercentEl.textContent = `${pct}%`;
  progressBarFillEl.style.width = `${pct}%`;
  if (progressBarHandleEl) {
    progressBarHandleEl.style.left = `${pct}%`;
  }
}

boot();