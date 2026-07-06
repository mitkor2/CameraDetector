/* Crime Scene Camera Detector — static web app
 *
 * Pipeline: getUserMedia camera feed -> classifier (on-device YOLOv8-cls ONNX,
 * or the project's Roboflow-hosted model as fallback) -> confirmed detections
 * are pinned on a Leaflet map at the device's GPS position with a snapshot.
 * Sightings persist in localStorage and can be exported as GeoJSON / CSV.
 */

'use strict';

// ────────────────────────── configuration ──────────────────────────

const DEFAULTS = {
  backend: 'onnx',            // onnx | auto | roboflow
  threshold: 0.75,            // min P(camera) to count a frame as positive
  confirmFrames: 2,           // consecutive positive frames before capturing
  dupRadius: 15,              // metres: closer detections update the existing pin
  rfModel: 'camerav2-u58ps/2',
  rfKey: '',
};

const ONNX_MODEL_URL = 'models/camera-classifier.onnx';
const ONNX_INPUT_SIZES = [224, 640, 320];   // probed in this order
const CAMERA_CLASS_INDEX = 0;               // ultralytics sorts folders: camera=0, no_camera=1
const INFER_INTERVAL_MS = { onnx: 700, roboflow: 1600 };
const CAPTURE_COOLDOWN_MS = 8000;
const GPS_STALE_MS = 30000;
const SNAPSHOT_WIDTH = 480;

const LS_SETTINGS = 'ccd_settings_v1';
const LS_SIGHTINGS = 'ccd_sightings_v1';

// ────────────────────────── state ──────────────────────────

let settings = loadSettings();
let sightings = loadSightings();
let running = false;
let stream = null;
let gpsWatchId = null;
let lastFix = null;           // {lat, lon, acc, t}
let ortSession = null;
let onnxInputSize = null;
let onnxInputName = null;
let onnxModelSource = null;   // 'device' (loaded via file picker, kept in IndexedDB) | 'repo'
let wakeLock = null;
let activeBackend = null;     // 'onnx' | 'roboflow' | null
let positiveStreak = 0;
let lastCaptureAt = 0;
let inferBusy = false;
let inferTimer = null;
let centeredOnce = false;

const markers = new Map();    // sighting id -> Leaflet marker

// ────────────────────────── dom ──────────────────────────

const $ = (id) => document.getElementById(id);
const video = $('video');
const hudLabel = $('hud-label');
const confFill = $('conf-fill');

// ────────────────────────── map ──────────────────────────

const map = L.map('map').setView([52.2215, 6.8937], 5); // Enschede, NL by default
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);

const camIcon = (manual) => L.divIcon({
  className: '',
  html: `<div style="width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;
         font-size:17px;background:${manual ? '#d29922' : '#f85149'};border:2px solid #fff;
         box-shadow:0 1px 4px rgba(0,0,0,.5)">📷</div>`,
  iconSize: [32, 32],
  iconAnchor: [16, 16],
  popupAnchor: [0, -16],
});

let posMarker = null;
let posCircle = null;

function updatePositionOnMap(fix) {
  const ll = [fix.lat, fix.lon];
  if (!posMarker) {
    posMarker = L.circleMarker(ll, { radius: 7, color: '#fff', weight: 2, fillColor: '#2f81f7', fillOpacity: 1 }).addTo(map);
    posCircle = L.circle(ll, { radius: fix.acc, color: '#2f81f7', weight: 1, fillOpacity: 0.08 }).addTo(map);
  } else {
    posMarker.setLatLng(ll);
    posCircle.setLatLng(ll).setRadius(fix.acc);
  }
  if (!centeredOnce) {
    map.setView(ll, 17);
    centeredOnce = true;
  }
}

// ────────────────────────── persistence ──────────────────────────

// IndexedDB key-value store — used to keep a YOLO .onnx model loaded from the
// device (localStorage is too small for model files)
function idbOpen() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('ccd', 1);
    req.onupgradeneeded = () => req.result.createObjectStore('kv');
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function idbGet(key) {
  const db = await idbOpen();
  return new Promise((resolve, reject) => {
    const req = db.transaction('kv').objectStore('kv').get(key);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function idbSet(key, value) {
  const db = await idbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('kv', 'readwrite');
    tx.objectStore('kv').put(value, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function idbDel(key) {
  const db = await idbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('kv', 'readwrite');
    tx.objectStore('kv').delete(key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

function loadSettings() {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(LS_SETTINGS) || '{}') }; }
  catch { return { ...DEFAULTS }; }
}

function saveSettings() {
  localStorage.setItem(LS_SETTINGS, JSON.stringify(settings));
}

function loadSightings() {
  try { return JSON.parse(localStorage.getItem(LS_SIGHTINGS) || '[]'); }
  catch { return []; }
}

function saveSightings() {
  try {
    localStorage.setItem(LS_SIGHTINGS, JSON.stringify(sightings));
  } catch (e) {
    // storage full — drop the oldest photos (keep coordinates) and retry once
    const withPhotos = sightings.filter(s => s.photo);
    withPhotos.slice(0, Math.ceil(withPhotos.length / 2)).forEach(s => { s.photo = null; });
    try { localStorage.setItem(LS_SIGHTINGS, JSON.stringify(sightings)); }
    catch { showBanner('Storage is full — new sightings may not persist. Export and clear old data.', true); }
  }
}

// ────────────────────────── ui helpers ──────────────────────────

function showBanner(msg, isError) {
  const b = $('banner');
  b.textContent = msg;
  b.classList.toggle('error', !!isError);
  b.classList.remove('hidden');
}

function hideBanner() { $('banner').classList.add('hidden'); }

function setPill(id, text, cls) {
  const p = $(id);
  p.textContent = text;
  p.className = 'pill' + (cls ? ' ' + cls : '');
}

function setHud(label, prob) {
  hudLabel.textContent = label;
  const pct = Math.round((prob || 0) * 100);
  confFill.style.width = pct + '%';
  confFill.style.background = prob >= settings.threshold ? 'var(--danger)' : 'var(--ok)';
}

function flash() {
  const f = $('flash');
  f.classList.add('on');
  requestAnimationFrame(() => requestAnimationFrame(() => f.classList.remove('on')));
}

function fmtTime(t) { return new Date(t).toLocaleString(); }

// ────────────────────────── geometry ──────────────────────────

function haversineMeters(lat1, lon1, lat2, lon2) {
  const R = 6371000, rad = Math.PI / 180;
  const dLat = (lat2 - lat1) * rad, dLon = (lon2 - lon1) * rad;
  const a = Math.sin(dLat / 2) ** 2 +
            Math.cos(lat1 * rad) * Math.cos(lat2 * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

// ────────────────────────── backends ──────────────────────────

async function initBackend() {
  activeBackend = null;
  const want = settings.backend;

  if (want === 'onnx' || want === 'auto') {
    try {
      await initOnnx();
      activeBackend = 'onnx';
      setPill('pill-model', 'YOLO on-device · ' + onnxInputSize + 'px · ' +
        (onnxModelSource === 'device' ? 'loaded from device' : 'from repo'), 'ok');
      return;
    } catch (e) {
      console.warn('ONNX backend unavailable:', e.message);
      if (want === 'onnx') {
        setPill('pill-model', 'YOLO model missing', 'err');
        showBanner('The camera feed is live, but no YOLO model is loaded yet. Open ⚙ Settings → “Load YOLO model” and pick your exported best.onnx (it is stored on this device for next time), or commit it as docs/models/camera-classifier.onnx.', true);
        return;
      }
    }
  }

  if (settings.rfKey && settings.rfModel) {
    activeBackend = 'roboflow';
    setPill('pill-model', 'model: Roboflow API', 'warn');
  } else {
    setPill('pill-model', 'model: none', 'err');
    showBanner('The camera feed is live, but no detection model is configured. Open ⚙ Settings → “Load YOLO model” and pick your exported best.onnx. Manual 📌 pins work meanwhile.', true);
  }
}

async function initOnnx() {
  if (typeof ort === 'undefined') throw new Error('onnxruntime-web failed to load');
  ort.env.wasm.numThreads = 1; // GitHub Pages lacks COOP/COEP headers needed for threads
  ort.env.wasm.wasmPaths = new URL('vendor/ort/', document.baseURI).href;

  // a model loaded from the device (kept in IndexedDB) wins over the repo file
  const stored = await idbGet('model').catch(() => null);
  if (stored) {
    ortSession = await ort.InferenceSession.create(new Uint8Array(stored), { executionProviders: ['wasm'] });
    onnxModelSource = 'device';
  } else {
    const head = await fetch(ONNX_MODEL_URL, { method: 'HEAD' });
    if (!head.ok) throw new Error('model file not found (' + head.status + ')');
    ortSession = await ort.InferenceSession.create(ONNX_MODEL_URL, { executionProviders: ['wasm'] });
    onnxModelSource = 'repo';
  }
  onnxInputName = ortSession.inputNames[0];

  // Probe which input resolution this export uses (their training used 640, default export is 224)
  for (const size of ONNX_INPUT_SIZES) {
    try {
      const dummy = new ort.Tensor('float32', new Float32Array(3 * size * size), [1, 3, size, size]);
      await ortSession.run({ [onnxInputName]: dummy });
      onnxInputSize = size;
      return;
    } catch { /* try next size */ }
  }
  ortSession = null;
  throw new Error('could not determine model input size');
}

function grabFrame(size) {
  // centre-crop the video frame to a square canvas of the given size
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const vw = video.videoWidth, vh = video.videoHeight;
  const side = Math.min(vw, vh);
  ctx.drawImage(video, (vw - side) / 2, (vh - side) / 2, side, side, 0, 0, size, size);
  return canvas;
}

function softmax(arr) {
  const m = Math.max(...arr);
  const ex = arr.map(v => Math.exp(v - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map(v => v / s);
}

async function classifyOnnx() {
  const size = onnxInputSize;
  const canvas = grabFrame(size);
  const { data } = canvas.getContext('2d').getImageData(0, 0, size, size);
  const n = size * size;
  const input = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    input[i] = data[i * 4] / 255;             // R
    input[n + i] = data[i * 4 + 1] / 255;     // G
    input[2 * n + i] = data[i * 4 + 2] / 255; // B
  }
  const tensor = new ort.Tensor('float32', input, [1, 3, size, size]);
  const out = await ortSession.run({ [onnxInputName]: tensor });
  let probs = Array.from(out[ortSession.outputNames[0]].data);
  const sum = probs.reduce((a, b) => a + b, 0);
  if (probs.some(v => v < 0 || v > 1) || Math.abs(sum - 1) > 0.01) probs = softmax(probs);
  return probs[CAMERA_CLASS_INDEX];
}

async function classifyRoboflow() {
  const canvas = grabFrame(416);
  const base64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
  const url = 'https://classify.roboflow.com/' + settings.rfModel + '?api_key=' + encodeURIComponent(settings.rfKey);
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: base64,
  });
  if (!res.ok) throw new Error('Roboflow API error ' + res.status);
  const json = await res.json();
  const preds = json.predictions || [];
  const isCameraClass = (c) => {
    const s = String(c).toLowerCase();
    return s.includes('camera') && !s.includes('no');
  };
  for (const p of preds) {
    const cls = p.class ?? p.class_name ?? '';
    if (isCameraClass(cls)) return p.confidence ?? 0;
  }
  if (json.top !== undefined) return isCameraClass(json.top) ? (json.confidence ?? 0) : 1 - (json.confidence ?? 0);
  return 0;
}

// ────────────────────────── detection loop ──────────────────────────

async function inferOnce() {
  if (!running || inferBusy || video.readyState < 2 || !activeBackend) return;
  inferBusy = true;
  try {
    const pCamera = activeBackend === 'onnx' ? await classifyOnnx() : await classifyRoboflow();
    hideBanner();

    if (pCamera >= settings.threshold) {
      positiveStreak++;
      setHud('CAMERA ' + Math.round(pCamera * 100) + '%', pCamera);
      const cooledDown = Date.now() - lastCaptureAt > CAPTURE_COOLDOWN_MS;
      if (positiveStreak >= settings.confirmFrames && cooledDown) {
        captureSighting(pCamera, 'detector');
        positiveStreak = 0;
      }
    } else {
      positiveStreak = 0;
      setHud('watching… ' + Math.round(pCamera * 100) + '%', pCamera);
    }
  } catch (e) {
    console.error(e);
    setHud('error', 0);
    showBanner('Detection error: ' + e.message, true);
  } finally {
    inferBusy = false;
  }
}

function startLoop() {
  stopLoop();
  inferTimer = setInterval(inferOnce, INFER_INTERVAL_MS[activeBackend] || 1000);
}

function stopLoop() {
  if (inferTimer) { clearInterval(inferTimer); inferTimer = null; }
}

// ────────────────────────── sightings ──────────────────────────

function currentFix() {
  if (!lastFix) return null;
  if (Date.now() - lastFix.t > GPS_STALE_MS) return { ...lastFix, stale: true };
  return lastFix;
}

function captureSighting(confidence, source) {
  const fix = currentFix();
  if (!fix) {
    showBanner('📷 Camera detected but there is no GPS fix yet — pin skipped. Waiting for location…');
    return;
  }

  lastCaptureAt = Date.now();
  flash();

  let photo = null;
  if (video.readyState >= 2 && video.videoWidth > 0) {
    const snap = document.createElement('canvas');
    snap.width = SNAPSHOT_WIDTH;
    snap.height = Math.round(video.videoHeight * (SNAPSHOT_WIDTH / video.videoWidth));
    snap.getContext('2d').drawImage(video, 0, 0, snap.width, snap.height);
    photo = snap.toDataURL('image/jpeg', 0.7);
  }

  // near an existing pin? update it instead of duplicating
  const near = sightings.find(s => haversineMeters(s.lat, s.lon, fix.lat, fix.lon) <= settings.dupRadius);
  if (near && source === 'detector') {
    near.hits = (near.hits || 1) + 1;
    near.lastSeen = Date.now();
    if (confidence >= (near.confidence || 0) && photo) {
      near.confidence = confidence;
      near.photo = photo;
    }
    saveSightings();
    upsertMarker(near);
    renderSightingsList();
    if (fix.stale) showBanner('Pin updated with a stale GPS fix (last update > 30 s ago) — accuracy may be off.');
    return;
  }

  const s = {
    id: 'sig_' + Date.now().toString(36) + Math.floor(Math.random() * 1e4).toString(36),
    lat: fix.lat,
    lon: fix.lon,
    accuracy: Math.round(fix.acc),
    confidence,
    hits: 1,
    firstSeen: Date.now(),
    lastSeen: Date.now(),
    source,
    photo,
  };
  sightings.push(s);
  saveSightings();
  upsertMarker(s);
  renderSightingsList();
  updateCountPill();
  if (fix.stale) showBanner('Pin placed with a stale GPS fix (last update > 30 s ago) — accuracy may be off.');
}

function popupHtml(s) {
  return `<div class="popup">
    ${s.photo ? `<img src="${s.photo}" alt="detected camera snapshot">` : ''}
    <div class="meta">
      ${s.source === 'manual' ? '📌 manual pin' : '🤖 confidence ' + Math.round(s.confidence * 100) + '%'}
      &middot; seen ${s.hits}&times;<br>
      first: ${fmtTime(s.firstSeen)}<br>
      ${s.lat.toFixed(6)}, ${s.lon.toFixed(6)} (±${s.accuracy} m)
    </div>
    <button class="del" onclick="window._deleteSighting('${s.id}')">delete pin</button>
  </div>`;
}

function upsertMarker(s) {
  let m = markers.get(s.id);
  if (!m) {
    m = L.marker([s.lat, s.lon], { icon: camIcon(s.source === 'manual') }).addTo(map);
    markers.set(s.id, m);
  }
  m.bindPopup(popupHtml(s));
}

window._deleteSighting = function (id) {
  sightings = sightings.filter(s => s.id !== id);
  saveSightings();
  const m = markers.get(id);
  if (m) { map.removeLayer(m); markers.delete(id); }
  renderSightingsList();
  updateCountPill();
};

function renderAllMarkers() {
  sightings.forEach(upsertMarker);
}

function updateCountPill() {
  setPill('pill-count', sightings.length + ' sighting' + (sightings.length === 1 ? '' : 's'), sightings.length ? 'ok' : '');
  const badge = $('tab-badge');
  badge.textContent = sightings.length;
  badge.classList.toggle('hidden', !sightings.length);
}

function renderSightingsList() {
  const el = $('sightings-list');
  if (!sightings.length) {
    el.innerHTML = '<p class="empty">No cameras recorded yet. Start detecting and walk the scene perimeter.</p>';
    return;
  }
  el.innerHTML = [...sightings].reverse().map(s => `
    <div class="sighting" data-id="${s.id}">
      ${s.photo ? `<img src="${s.photo}" alt="camera snapshot">` : '<div class="no-photo">📌</div>'}
      <div class="info">
        <b>${s.source === 'manual' ? 'Manual pin' : Math.round(s.confidence * 100) + '% camera'}</b><br>
        ${fmtTime(s.firstSeen)}<br>
        ${s.lat.toFixed(5)}, ${s.lon.toFixed(5)}
      </div>
    </div>`).join('');
  el.querySelectorAll('.sighting').forEach(card => {
    card.addEventListener('click', () => {
      const s = sightings.find(x => x.id === card.dataset.id);
      if (!s) return;
      switchView('map');
      map.setView([s.lat, s.lon], 18);
      markers.get(s.id)?.openPopup();
    });
  });
}

// ────────────────────────── exports ──────────────────────────

function download(filename, text, mime) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: mime }));
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function exportGeoJSON() {
  const fc = {
    type: 'FeatureCollection',
    features: sightings.map(s => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [s.lon, s.lat] },
      properties: {
        id: s.id,
        source: s.source,
        confidence: s.confidence,
        hits: s.hits,
        accuracy_m: s.accuracy,
        firstSeen: new Date(s.firstSeen).toISOString(),
        lastSeen: new Date(s.lastSeen).toISOString(),
        photo: s.photo, // JPEG data URI
      },
    })),
  };
  download('camera-sightings.geojson', JSON.stringify(fc, null, 2), 'application/geo+json');
}

function exportCSV() {
  const rows = [['id', 'lat', 'lon', 'accuracy_m', 'confidence', 'hits', 'first_seen', 'last_seen', 'source']];
  sightings.forEach(s => rows.push([
    s.id, s.lat, s.lon, s.accuracy, s.confidence, s.hits,
    new Date(s.firstSeen).toISOString(), new Date(s.lastSeen).toISOString(), s.source,
  ]));
  download('camera-sightings.csv', rows.map(r => r.join(',')).join('\n'), 'text/csv');
}

// ────────────────────────── camera & gps ──────────────────────────

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 960 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

function stopCamera() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  video.srcObject = null;
}

function startGPS() {
  if (!('geolocation' in navigator)) {
    setPill('pill-gps', 'GPS: unsupported', 'err');
    return;
  }
  setPill('pill-gps', 'GPS: locating…', 'warn');
  gpsWatchId = navigator.geolocation.watchPosition(
    (pos) => {
      lastFix = {
        lat: pos.coords.latitude,
        lon: pos.coords.longitude,
        acc: pos.coords.accuracy,
        t: Date.now(),
      };
      setPill('pill-gps', 'GPS: ±' + Math.round(lastFix.acc) + ' m', lastFix.acc <= 25 ? 'ok' : 'warn');
      updatePositionOnMap(lastFix);
    },
    (err) => {
      setPill('pill-gps', 'GPS: ' + (err.code === 1 ? 'denied' : 'error'), 'err');
      showBanner('Location unavailable: ' + err.message + '. Detections cannot be pinned without a GPS fix.', true);
    },
    { enableHighAccuracy: true, maximumAge: 5000, timeout: 20000 },
  );
}

// ────────────────────────── start / stop ──────────────────────────

async function start() {
  hideBanner();
  $('btn-start').disabled = true;
  try {
    setHud('starting camera…', 0);
    await startCamera();
  } catch (e) {
    console.error(e);
    showBanner(e.name === 'NotAllowedError'
      ? 'Camera access was denied. Allow camera access for this site in your browser settings, then press ▶ Start detecting.'
      : 'Could not start the camera: ' + e.message, true);
    setHud('idle', 0);
    stopCamera();
    $('btn-start').disabled = false;
    return;
  }
  // the feed stays live from here on, even if no detection model is available
  running = true;
  $('btn-stop').disabled = false;
  startGPS();
  acquireWakeLock();
  await refreshBackend();
}

// keep the phone screen awake while detecting; the lock is dropped by the OS
// whenever the tab is hidden, so re-acquire it on return
async function acquireWakeLock() {
  try { wakeLock = await navigator.wakeLock?.request('screen'); } catch { /* unsupported or denied — not fatal */ }
}

document.addEventListener('visibilitychange', () => {
  if (running && document.visibilityState === 'visible') acquireWakeLock();
});

async function refreshBackend() {
  stopLoop();
  setHud('loading model…', 0);
  await initBackend(); // reports its own errors via pill + banner
  if (!running) return;
  if (activeBackend) {
    positiveStreak = 0;
    startLoop();
    setHud('watching…', 0);
  } else {
    setHud('feed only — no model', 0);
  }
}

function stop() {
  running = false;
  stopLoop();
  stopCamera();
  wakeLock?.release().catch(() => {});
  wakeLock = null;
  if (gpsWatchId !== null) { navigator.geolocation.clearWatch(gpsWatchId); gpsWatchId = null; }
  setPill('pill-gps', 'GPS: off');
  setHud('idle', 0);
  $('btn-start').disabled = false;
  $('btn-stop').disabled = true;
}

// ────────────────────────── settings ui ──────────────────────────

async function updateModelFileStatus() {
  const stored = await idbGet('model').catch(() => null);
  const mb = stored ? stored.byteLength / 1024 / 1024 : 0;
  $('model-file-status').textContent = stored
    ? 'Model on this device: ' + (mb >= 1 ? mb.toFixed(1) + ' MB' : Math.max(1, Math.round(stored.byteLength / 1024)) + ' KB') + ' ✔'
    : 'No model stored on this device yet.';
  $('btn-model-clear').classList.toggle('hidden', !stored);
}

function bindSettings() {
  const backend = $('set-backend'), thresh = $('set-thresh'), threshVal = $('set-thresh-val'),
        frames = $('set-frames'), radius = $('set-radius'),
        rfModel = $('set-rf-model'), rfKey = $('set-rf-key'),
        modelFile = $('set-model-file');

  modelFile.addEventListener('change', async () => {
    const file = modelFile.files[0];
    if (!file) return;
    $('model-file-status').textContent = 'Storing model…';
    try {
      await idbSet('model', await file.arrayBuffer());
      settings.backend = 'onnx';
      backend.value = 'onnx';
      saveSettings();
      await updateModelFileStatus();
      if (running) await refreshBackend();
      else showBanner('Model stored. Press ▶ Start detecting.');
    } catch (e) {
      $('model-file-status').textContent = 'Could not store the model: ' + e.message;
    }
    modelFile.value = '';
  });

  $('btn-model-clear').addEventListener('click', async () => {
    await idbDel('model').catch(() => {});
    await updateModelFileStatus();
    if (running && activeBackend === 'onnx') await refreshBackend();
  });
  updateModelFileStatus();

  backend.value = settings.backend;
  thresh.value = settings.threshold;
  threshVal.textContent = Math.round(settings.threshold * 100) + '%';
  frames.value = settings.confirmFrames;
  radius.value = settings.dupRadius;
  rfModel.value = settings.rfModel;
  rfKey.value = settings.rfKey;

  backend.addEventListener('change', async () => {
    settings.backend = backend.value; saveSettings();
    if (running) await refreshBackend();
  });
  thresh.addEventListener('input', () => {
    settings.threshold = parseFloat(thresh.value);
    threshVal.textContent = Math.round(settings.threshold * 100) + '%';
    saveSettings();
  });
  frames.addEventListener('change', () => { settings.confirmFrames = Math.max(1, parseInt(frames.value) || 2); saveSettings(); });
  radius.addEventListener('change', () => { settings.dupRadius = Math.max(0, parseFloat(radius.value) || 0); saveSettings(); });
  rfModel.addEventListener('change', async () => {
    settings.rfModel = rfModel.value.trim(); saveSettings();
    if (running && activeBackend !== 'onnx') await refreshBackend();
  });
  rfKey.addEventListener('change', async () => {
    settings.rfKey = rfKey.value.trim(); saveSettings();
    if (running && activeBackend !== 'onnx') await refreshBackend();
  });
}

// ────────────────────────── views (mobile tab bar) ──────────────────────────

function switchView(name) {
  document.querySelectorAll('main .view').forEach(v => v.classList.toggle('active', v.dataset.view === name));
  document.querySelectorAll('#tabbar button').forEach(b => b.classList.toggle('active', b.dataset.view === name));
  // Leaflet needs a size recalculation after its container becomes visible
  if (name === 'map') setTimeout(() => map.invalidateSize(), 60);
}

document.querySelectorAll('#tabbar button').forEach(b =>
  b.addEventListener('click', () => switchView(b.dataset.view)));

function openSettings() {
  $('settings').classList.remove('hidden');
  $('sheet-backdrop').classList.remove('hidden');
}

function closeSettings() {
  $('settings').classList.add('hidden');
  $('sheet-backdrop').classList.add('hidden');
}

// ────────────────────────── wire up ──────────────────────────

$('btn-start').addEventListener('click', start);
$('btn-stop').addEventListener('click', stop);
$('btn-settings').addEventListener('click', openSettings);
$('btn-settings-close').addEventListener('click', closeSettings);
$('sheet-backdrop').addEventListener('click', closeSettings);
$('btn-manual').addEventListener('click', () => {
  if (!currentFix()) { showBanner('No GPS fix yet — cannot place a manual pin.', true); return; }
  captureSighting(1, 'manual');
});
$('btn-locate').addEventListener('click', () => {
  const fix = currentFix();
  if (fix) map.setView([fix.lat, fix.lon], 17);
  else showBanner('No GPS fix yet.');
});
$('btn-export-geojson').addEventListener('click', exportGeoJSON);
$('btn-export-csv').addEventListener('click', exportCSV);
$('btn-clear').addEventListener('click', () => {
  if (!sightings.length || !confirm('Delete all ' + sightings.length + ' recorded sightings?')) return;
  sightings = [];
  saveSightings();
  markers.forEach(m => map.removeLayer(m));
  markers.clear();
  renderSightingsList();
  updateCountPill();
});

bindSettings();
renderAllMarkers();
renderSightingsList();
updateCountPill();

if (sightings.length) {
  map.fitBounds(L.latLngBounds(sightings.map(s => [s.lat, s.lon])).pad(0.3));
  centeredOnce = true;
}

if (!window.isSecureContext) {
  showBanner('This page is not served over HTTPS — camera and GPS access will be blocked by the browser.', true);
} else {
  // start detecting immediately — no button press needed. The browser asks for
  // camera/location permission on first visit; if that is denied or dismissed,
  // the ▶ button remains as a manual retry.
  start(); // errors are surfaced in the banner by start() itself
}
