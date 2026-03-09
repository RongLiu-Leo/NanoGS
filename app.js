import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SplatMesh } from "@sparkjsdev/spark";
import { simplifyMesh } from "./simplify.js";

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
const lamColorEl = document.getElementById("lam-color");

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
let simplifiedBlob = null;

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
  btnDownload.disabled = false;
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
  btnDownload.disabled = false;
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
    lamColor: Math.max(0, Number(lamColorEl.value)),
  };

  btnSimplify.disabled = true;
  btnDownload.disabled = true;

  try {
    const t0 = performance.now();

    const result = await simplifyMesh(leftMesh, params, setStatus);

    const ratioTag = params.ratio.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
    simplifiedName = originalName.replace(/\.ply$/i, `_${ratioTag}.ply`);
    simplifiedBlob = result.blob;

    await loadSimplifiedPacked(result.packed, simplifiedName);

    const t1 = performance.now();
    setStatus(
      `Done. ${result.originalCount} → ${result.finalCount} splats in ${((t1 - t0) / 1000).toFixed(2)} s.`
    );
  } catch (err) {
    console.error(err);
    setStatus(`Simplify failed: ${err.message}`);
  } finally {
    btnSimplify.disabled = false;
    btnDownload.disabled = !simplifiedBlob;
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
  if (!simplifiedBlob) return;
  const url = URL.createObjectURL(simplifiedBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = simplifiedName || "simplified.ply";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  setStatus(`Downloaded: ${simplifiedName}`);
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

boot();