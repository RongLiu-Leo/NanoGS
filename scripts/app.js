import * as THREE from "three";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";
import { SplatMesh } from "@sparkjsdev/spark";
import { simplifyMesh, packedFromState, getSplatCount, stateToPlyBytes } from "./simplify.js";

const app = document.getElementById("app");
const viewer = document.getElementById("viewer");
const layerLeft = document.getElementById("layer-left");
const layerRight = document.getElementById("layer-right");
const interactionLayer = document.getElementById("interaction-layer");
const divider = document.getElementById("divider");
const handle = document.getElementById("handle");

const labelLeft = document.getElementById("label-left");
const labelRight = document.getElementById("label-right");

const inputOriginal = document.getElementById("input-original");
const exampleDropdownEl = document.getElementById("example-dropdown");
const exampleDropdownTrigger = document.getElementById("example-dropdown-trigger");
const exampleDropdownLabel = exampleDropdownTrigger?.querySelector(".example-dropdown-label");
const exampleDropdownList = document.getElementById("example-dropdown-list");
const btnSimplify = document.getElementById("btn-simplify");
const btnDownload = null;

const ratioEl = document.getElementById("ratio");
const ratioNumberEl = document.getElementById("ratio-number");
const pCapRatioEl = document.getElementById("p-cap-ratio");
const pCapRatioNumberEl = document.getElementById("p-cap-ratio-number");

const progressPercentEl = document.getElementById("progress-percent");
const progressBarFillEl = document.getElementById("progress-bar-fill");
const progressBarEl = document.getElementById("progress-bar");
const progressBarHandleEl = document.getElementById("progress-bar-handle");
const statOriginalEl = document.getElementById("stat-original");
const statTargetEl = document.getElementById("stat-target");
const trackListEl = document.getElementById("track-list");

const rendererLeft = new THREE.WebGLRenderer({ antialias: true, alpha: false });
const rendererRight = new THREE.WebGLRenderer({ antialias: true, alpha: false });

rendererLeft.setPixelRatio(Math.min(window.devicePixelRatio, 2));
rendererRight.setPixelRatio(Math.min(window.devicePixelRatio, 2));
rendererLeft.setClearColor(0x111111, 1);
rendererRight.setClearColor(0x111111, 1);

layerLeft.appendChild(rendererLeft.domElement);
layerRight.appendChild(rendererRight.domElement);

const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
camera.position.set(-1.5, -1.5, 1.5);
camera.up.set(0, -1, 0);

const controls = new TrackballControls(camera, interactionLayer);
controls.target.set(0, 0, 0);
controls.dynamicDampingFactor = 0.1;

const sceneLeft = new THREE.Scene();
const sceneRight = new THREE.Scene();

let leftMesh = null;
let rightMesh = null;

let originalName = "-";
let simplifiedName = "-";

let originalBytes = null;
let originalSplatCount = null;
let latestSimplifyResult = null;

let historyEntries = [];
let activeHistoryIndex = -1;

let isSimplifying = false;
let autoFollowProgress = true;

ratioEl.addEventListener("input", () => {
  ratioNumberEl.value = ratioEl.value;
  updateStatsFromOriginalAndRatio();
});

ratioNumberEl.addEventListener("input", () => {
  ratioEl.value = ratioNumberEl.value;
  updateStatsFromOriginalAndRatio();
});

pCapRatioEl.addEventListener("input", () => {
  pCapRatioNumberEl.value = pCapRatioEl.value;
});

pCapRatioNumberEl.addEventListener("input", () => {
  pCapRatioEl.value = pCapRatioNumberEl.value;
});

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
  const mesh = await buildMeshFromBytes(bytes);
  clearMesh(sceneLeft, leftMesh);
  leftMesh = mesh;
  sceneLeft.add(leftMesh);
  originalBytes = bytes;
  originalName = name;

  // Reset simplified view/state when a new model is loaded.
  clearMesh(sceneRight, rightMesh);
  rightMesh = null;
  simplifiedName = "-";

  latestSimplifyResult = null;
  clearHistory();
  originalSplatCount = await getSplatCount(mesh);
  updateStatsFromOriginalAndRatio();

  // With no simplified model, show the full original.
  // Smoothly move the divider/handle to the far right.
  animateSplitTo(1.0);
  updateLabels();
  syncExampleSelectToCurrent();
}

async function loadSimplifiedBytes(bytes, name) {
  const mesh = await buildMeshFromBytes(bytes);
  clearMesh(sceneRight, rightMesh);
  rightMesh = mesh;
  sceneRight.add(rightMesh);
  simplifiedName = name;
  updateLabels();
}

async function loadSimplifiedPacked(packed, name) {
  const mesh = await buildMeshFromPacked(packed);
  clearMesh(sceneRight, rightMesh);
  rightMesh = mesh;
  sceneRight.add(rightMesh);
  simplifiedName = name;
  updateLabels();
}

function resize() {
  const rect = viewer?.getBoundingClientRect?.();
  const w = Math.max(1, Math.floor(rect?.width ?? window.innerWidth));
  const h = Math.max(1, Math.floor(rect?.height ?? window.innerHeight));
  rendererLeft.setSize(w, h, false);
  rendererRight.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  controls.handleResize();
}

let split = 0.5;
let splitAnimationId = 0;

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

function animateSplitTo(target, durationMs = 900) {
  const start = split;
  const end = Math.max(0, Math.min(1, target));
  const startTime = performance.now();
  const myId = ++splitAnimationId;

  function frame(now) {
    if (myId !== splitAnimationId) return; // cancelled by a new animation or user drag
    const t = Math.min(1, (now - startTime) / durationMs);
    // Ease-out for a smoother finish.
    const eased = 1 - Math.pow(1 - t, 3);
    const value = start + (end - start) * eased;
    setSplit(value);
    if (t < 1) {
      requestAnimationFrame(frame);
    }
  }

  requestAnimationFrame(frame);
}

function clientXToSplit(clientX) {
  const rect = viewer?.getBoundingClientRect?.() ?? app.getBoundingClientRect();
  return (clientX - rect.left) / rect.width;
}

let dragging = false;

handle.addEventListener("pointerdown", (e) => {
  // Cancel any ongoing automatic animation when the user grabs the handle.
  splitAnimationId++;
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
}

function formatInt(n) {
  return Number.isFinite(n) ? n.toLocaleString() : "-";
}

function updateStatsFromOriginalAndRatio() {
  statOriginalEl.textContent = formatInt(originalSplatCount);
  if (originalSplatCount == null) {
    statTargetEl.textContent = "-";
    return;
  }
  const ratio = Math.min(0.99, Math.max(0.01, Number(ratioEl.value)));
  const target = Math.max(1, Math.ceil(originalSplatCount * ratio));
  statTargetEl.textContent = formatInt(target);
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
}

function downloadPlyForEntry(entry, filename) {
  if (!entry?.snapshot) return;
  const bytes = stateToPlyBytes(entry.snapshot);
  const blob = new Blob([bytes], { type: "application/octet-stream" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function renderHistory() {
  trackListEl.innerHTML = "";

  const baseName = originalName.replace(/\.ply$/i, "");

  historyEntries.forEach((entry, index) => {
    const wrap = document.createElement("div");
    wrap.className = "track-item-wrap";

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "track-item" + (index === activeHistoryIndex ? " active" : "");

    const pct = Math.round((entry.progress ?? 0) * 100);
    const iteration = typeof entry.iteration === "number" ? entry.iteration : 0;
    const isPrune = entry.stage === "prune";
    const title = `Iteration ${iteration}`;
    const stageText = isPrune ? "prune" : "merge";

    btn.innerHTML = `
      <div class="track-row-top">
        <div class="track-title">${title}</div>
        <div class="track-count">${formatInt(entry.currentCount)} splats</div>
      </div>
      <div class="track-row-bottom">
        ${stageText} · ${pct}%
      </div>
    `;

    btn.addEventListener("click", async () => {
      await previewHistoryEntry(index);
    });

    const downloadBtn = document.createElement("button");
    downloadBtn.type = "button";
    downloadBtn.className = "track-download";
    downloadBtn.title = "Download as PLY";
    downloadBtn.innerHTML = "↓";
    downloadBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const filename = `${baseName}_iteration_${iteration}.ply`;
      downloadPlyForEntry(entry, filename);
    });

    wrap.appendChild(btn);
    wrap.appendChild(downloadBtn);
    trackListEl.appendChild(wrap);
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

  try {
    const packed = packedFromState(entry.snapshot);
    const baseName = originalName.replace(/\.ply$/i, "");
    const iteration =
      typeof entry.iteration === "number" ? entry.iteration : index;
    const stageName = `${baseName}_iteration_${iteration}.ply`;
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
  } catch (err) {
    console.error(err);
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
    updateProgressVisual(0);
    return;
  }

  if (evt.type === "snapshot") {
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

    if (autoFollowProgress) {
      activeHistoryIndex = historyEntries.length - 1;
      updateProgressHeader(evt, { mode: "live" });
    } else {
      updateProgressHeader(evt, {
        mode: "preview",
        previewIndex: activeHistoryIndex,
      });
    }

    renderHistory();

    // While simplifying and auto-following, also live-update the right view
    // so that layer-right and label-right reflect the current progress stage.
    if (isSimplifying && autoFollowProgress && evt.snapshot) {
      const snapshot = evt.snapshot;
      (async () => {
        try {
          const packed = packedFromState(snapshot);
          const baseName = originalName.replace(/\.ply$/i, "");
          const iteration =
            typeof evt.iteration === "number"
              ? evt.iteration
              : historyEntries.length - 1;
          const stageName = `${baseName}_iteration_${iteration}.ply`;
          const mesh = await buildMeshFromPacked(packed);

          // If simplification has stopped or the user took manual control,
          // avoid overwriting their chosen view.
          if (!isSimplifying || !autoFollowProgress) {
            if (typeof mesh.dispose === "function") mesh.dispose();
            return;
          }

          clearMesh(sceneRight, rightMesh);
          rightMesh = mesh;
          simplifiedName = stageName;
          updateLabels();
        } catch (err) {
          console.error(err);
        }
      })();
    }

    return;
  }

  if (evt.type === "done") {
    // Keep the existing stages, but treat "Final" as a completion
    // event that updates progress to 100% without adding a new row.
    if (historyEntries.length > 0) {
      const last = historyEntries[historyEntries.length - 1];
      last.progress = evt.progress ?? 1;
      renderHistory();
    }
    updateProgressHeader(evt, { mode: "live" });
  }
}

function setSimplifyButtonState(state) {
  btnSimplify.classList.remove("btn-simplify-loading", "btn-simplify-success");
  btnSimplify.disabled = false;
  if (state === "loading") {
    btnSimplify.classList.add("btn-simplify-loading");
  } else if (state === "success") {
    btnSimplify.classList.add("btn-simplify-success");
  }
}

async function simplifyCurrent() {
  if (!leftMesh || isSimplifying) {
    return;
  }

  const params = {
    ratio: Math.min(0.99, Math.max(0.01, Number(ratioEl.value))),
    pCapRatio: Math.min(0.5, Math.max(0.01, Number(pCapRatioEl.value))),
    keepHistory: true,
  };

  setSimplifyButtonState("loading");
  // Yield so the browser paints the spinner before we do blocking work
  await new Promise((r) => requestAnimationFrame(r));
  await new Promise((r) => requestAnimationFrame(r));

  clearHistory();

  // When starting a new simplification, clear the previous progress stage
  // from the right view so the upcoming run can fully own the UI state.
  clearMesh(sceneRight, rightMesh);
  rightMesh = null;
  simplifiedName = originalName.replace(/\.ply$/i, "_simplifying.ply");
  updateLabels();

  isSimplifying = true;
  autoFollowProgress = true;
  activeHistoryIndex = -1;

    try {
    const t0 = performance.now();

    const result = await simplifyMesh(leftMesh, params, undefined, handleSimplifyProgress);
    latestSimplifyResult = result;

    let finalName;
    if (historyEntries.length > 0) {
      const lastEntry = historyEntries[historyEntries.length - 1];
      const iteration =
        typeof lastEntry.iteration === "number"
          ? lastEntry.iteration
          : historyEntries.length - 1;
      const baseName = originalName.replace(/\.ply$/i, "");
      finalName = `${baseName}_iteration_${iteration}.ply`;
    } else {
      const ratioTag = params.ratio.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
      finalName = originalName.replace(/\.ply$/i, `_${ratioTag}.ply`);
    }

    simplifiedName = finalName;

    await loadSimplifiedPacked(result.packed, simplifiedName);

    // Once the simplified model is ready, smoothly reveal the comparison
    // by animating the divider/handle from the right edge to the center.
    animateSplitTo(0.5);

    if (historyEntries.length > 0) {
      activeHistoryIndex = historyEntries.length - 1;
      renderHistory();
      updateProgressHeader(historyEntries[historyEntries.length - 1], {
        mode: "preview",
        previewIndex: activeHistoryIndex,
      });
    }

    isSimplifying = false;
    autoFollowProgress = true;

    const t1 = performance.now();

    setSimplifyButtonState("success");
    setTimeout(() => setSimplifyButtonState("idle"), 1200);
  } catch (err) {
    isSimplifying = false;
    console.error(err);
    setSimplifyButtonState("idle");
  } finally {
    isSimplifying = false;
    if (!btnSimplify.classList.contains("btn-simplify-success")) {
      setSimplifyButtonState("idle");
    }
  }
}

async function loadPlyFile(file) {
  if (!file) return;
  try {
    const bytes = await file.arrayBuffer();
    await loadOriginalBytes(bytes, file.name);
    populateExampleSelect();
  } catch (err) {
    console.error(err);
  }
}

inputOriginal.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  await loadPlyFile(file);
});

viewer.addEventListener("dragover", (e) => {
  e.preventDefault();
  const hasPly = Array.from(e.dataTransfer?.items || []).some(
    (item) => item.kind === "file" && /\.ply$/i.test(item.getAsFile()?.name ?? "")
  );
  if (hasPly) {
    e.dataTransfer.dropEffect = "copy";
    viewer.classList.add("drag-over");
  }
});

viewer.addEventListener("dragleave", (e) => {
  if (!viewer.contains(e.relatedTarget)) viewer.classList.remove("drag-over");
});

viewer.addEventListener("drop", async (e) => {
  e.preventDefault();
  viewer.classList.remove("drag-over");
  const file = Array.from(e.dataTransfer?.files || []).find((f) => /\.ply$/i.test(f.name));
  await loadPlyFile(file);
});

btnSimplify.addEventListener("click", simplifyCurrent);

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

const EXAMPLE_FOLDER = "./examples";

let exampleFileList = [];

/** Fetch folder URL and parse server directory listing for .ply filenames. */
async function listPlyFilesInFolder(folderUrl) {
  const resp = await fetch(folderUrl, { method: "GET" });
  if (!resp.ok) throw new Error(`Failed to list folder: ${folderUrl}`);
  const html = await resp.text();
  const doc = new DOMParser().parseFromString(html, "text/html");
  const links = doc.querySelectorAll("a[href]");
  const plyFiles = [];
  for (const a of links) {
    const href = (a.getAttribute("href") || "").trim();
    if (href.endsWith(".ply") && !href.includes("/")) plyFiles.push(href);
  }
  return plyFiles;
}

function setExampleDropdownOpen(open) {
  if (!exampleDropdownEl || !exampleDropdownList) return;
  exampleDropdownEl.dataset.open = open ? "true" : "false";
  exampleDropdownList.hidden = !open;
  exampleDropdownTrigger?.setAttribute("aria-expanded", String(open));
}

function populateExampleSelect() {
  if (!exampleDropdownList) return;
  exampleDropdownList.innerHTML = "";
  const isExample = exampleFileList.includes(originalName);
  if (!isExample && originalName !== "-") {
    const opt = document.createElement("button");
    opt.type = "button";
    opt.className = "example-dropdown-option";
    opt.setAttribute("role", "option");
    opt.dataset.value = "__custom__";
    opt.setAttribute("aria-selected", "true");
    opt.textContent = `Current: ${originalName}`;
    exampleDropdownList.appendChild(opt);
  }
  for (const name of exampleFileList) {
    const opt = document.createElement("button");
    opt.type = "button";
    opt.className = "example-dropdown-option";
    opt.setAttribute("role", "option");
    opt.dataset.value = name;
    opt.setAttribute("aria-selected", name === originalName ? "true" : "false");
    opt.textContent = name;
    exampleDropdownList.appendChild(opt);
  }
  syncExampleSelectToCurrent();
}

function syncExampleSelectToCurrent() {
  if (!exampleDropdownLabel || !exampleDropdownList) return;
  const isExample = exampleFileList.includes(originalName);
  exampleDropdownLabel.textContent = isExample ? originalName : (originalName !== "-" ? `Current: ${originalName}` : "—");
  exampleDropdownList.querySelectorAll(".example-dropdown-option").forEach((el) => {
    el.setAttribute("aria-selected", el.dataset.value === originalName || (el.dataset.value === "__custom__" && !isExample) ? "true" : "false");
  });
}

async function loadExampleByName(filename) {
  const url = `${EXAMPLE_FOLDER}/${filename}`;
  const bytes = await fetchBytes(url);
  await loadOriginalBytes(bytes, filename);
  setExampleDropdownOpen(false);
  populateExampleSelect();
}

async function boot() {
  try {
    exampleFileList = await listPlyFilesInFolder(EXAMPLE_FOLDER);
    if (exampleFileList.length === 0) throw new Error(`No .ply files in ${EXAMPLE_FOLDER}`);
    populateExampleSelect();
    const chosen = exampleFileList[Math.floor(Math.random() * exampleFileList.length)];
    const demoOriginal = await fetchBytes(`${EXAMPLE_FOLDER}/${chosen}`);
    await loadOriginalBytes(demoOriginal, chosen);
    syncExampleSelectToCurrent();
    resize();
    animate();
  } catch (err) {
    console.error(err);
    resize();
    setSplit(0.5);
    animate();
  }
}

exampleDropdownTrigger?.addEventListener("click", (e) => {
  e.preventDefault();
  const open = exampleDropdownEl?.dataset.open === "true";
  setExampleDropdownOpen(!open);
});

exampleDropdownList?.addEventListener("click", (e) => {
  const opt = e.target.closest(".example-dropdown-option");
  if (!opt) return;
  const value = opt.dataset.value;
  if (!value || value === "__custom__") return;
  (async () => {
    try {
      await loadExampleByName(value);
    } catch (err) {
      console.error(err);
    }
  })();
});

document.addEventListener("click", (e) => {
  if (exampleDropdownEl?.dataset.open === "true" && !exampleDropdownEl.contains(e.target)) {
    setExampleDropdownOpen(false);
  }
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && exampleDropdownEl?.dataset.open === "true") {
    setExampleDropdownOpen(false);
  }
});

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