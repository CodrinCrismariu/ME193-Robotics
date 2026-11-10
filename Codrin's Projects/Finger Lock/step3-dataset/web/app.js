'use strict';
// Sign dataset collector: webcam -> labeled photos, saved by collect_server.py. Hand boxes are added in label.html.

const LABELS = ['C', 'O', 'D', 'R', 'I', 'N', 'none'];
const KEY_TO_LABEL = { c: 'C', o: 'O', d: 'D', r: 'R', i: 'I', n: 'N', x: 'none' };
const HINTS = {
  C: 'Curve the fingers and thumb into a C, as if holding a cup from the side.',
  O: 'Bring all the fingertips to the tip of the thumb so the hand makes an O.',
  D: 'Index finger straight up; the other fingertips touch the thumb, making a small circle.',
  R: 'Index and middle fingers up and crossed; ring finger and pinky folded down under the thumb.',
  I: 'Pinky straight up; the other fingers make a fist with the thumb across them.',
  N: 'Fist with the index and middle fingers folded over the thumb; the thumb tip peeks out between the middle and ring fingers.',
  none: 'Anything that is not one of the six letters: other signs, a relaxed hand, a fist, a hand halfway between letters, or no hand at all.',
};
const TARGET = { C: 200, O: 200, D: 200, R: 200, I: 200, N: 200, none: 300 };
const W = 320, H = 240;        // saved image size
const VW = 640, VH = 480;      // on-screen view size

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const canvas = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
const store = { get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch { return d; } },
                set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } } };

const S = {
  stream: null, track: null, locked: false, label: 'C', session: sessionName(),
  samples: [], counts: Object.fromEntries(LABELS.map((l) => [l, 0])), mine: [], busy: false, frames: 0, fpsT: performance.now(),
};

const video = document.createElement('video');
video.playsInline = true;
video.muted = true;
const grab = canvas(W, H), gctx = grab.getContext('2d', { willReadFrequently: true });
const view = $('view'), vctx = view.getContext('2d');

function sessionName() {
  const d = new Date(), p = (n) => String(n).padStart(2, '0');
  return `s${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
}
function status(msg, kind) { const el = $('status'); el.textContent = msg; el.className = 'status' + (kind ? ' ' + kind : ''); }
const mirror = () => $('mirror').checked;

// ── drawing a camera frame: center-cropped to 4:3 so nothing is squashed ────
function drawFrame(ctx, w, h) {
  const vw = video.videoWidth, vh = video.videoHeight;
  let sw = vw, sh = vh;
  if (vw / vh > 4 / 3) sw = vh * 4 / 3; else sh = vw * 3 / 4;
  ctx.drawImage(video, (vw - sw) / 2, (vh - sh) / 2, sw, sh, 0, 0, w, h);
}

function strokeBox(ctx, box, sx, sy, lw) {
  if (!box) return;
  ctx.lineWidth = lw;
  ctx.strokeStyle = 'rgba(0,0,0,.45)';
  ctx.strokeRect(box[0] * sx + 1, box[1] * sy + 1, (box[2] - box[0]) * sx, (box[3] - box[1]) * sy);
  ctx.strokeStyle = '#e8743b';
  ctx.strokeRect(box[0] * sx, box[1] * sy, (box[2] - box[0]) * sx, (box[3] - box[1]) * sy);
}


// ── live view ───────────────────────────────────────────────────────────────
function loop() {
  if (video.readyState >= 2 && video.videoWidth) {
    drawFrame(gctx, W, H);
    vctx.save();
    if (mirror()) { vctx.translate(VW, 0); vctx.scale(-1, 1); }
    vctx.drawImage(grab, 0, 0, VW, VH);
    vctx.restore();
    S.frames++;
    const now = performance.now();
    if (now - S.fpsT > 1000) { $('fps').textContent = `${Math.round(S.frames * 1000 / (now - S.fpsT))} fps`; S.frames = 0; S.fpsT = now; }
  }
  requestAnimationFrame(loop);
}

// ── camera ──────────────────────────────────────────────────────────────────
async function startCamera(deviceId) {
  if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
  const video0 = { width: { ideal: 640 }, height: { ideal: 480 } };
  if (deviceId) video0.deviceId = { exact: deviceId };
  S.stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: video0 });
  S.track = S.stream.getVideoTracks()[0];
  S.locked = false;
  $('lockExp').textContent = 'Lock exposure';
  video.srcObject = S.stream;
  await video.play();
  clearBackground('New camera: capture the empty scene again.');
  const st = S.track.getSettings();
  status(`Camera: ${S.track.label || 'webcam'} (${st.width}×${st.height})`, 'ok');
  return S.track.getSettings().deviceId;
}

async function setupCameras() {
  let current = await startCamera(store.get('camera', '') || undefined).catch(() => startCamera());
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
  const builtIn = /integrated|built-?in|internal|front|facetime/i;
  const curCam = cams.find((c) => c.deviceId === current);
  if (!store.get('camera', '') && curCam && builtIn.test(curCam.label)) {
    const usb = cams.find((c) => !builtIn.test(c.label));
    if (usb) { current = await startCamera(usb.deviceId); $('camNote').textContent = `Switched to ${usb.label}: the assignment asks for the USB webcam.`; }
  }
  $('camera').innerHTML = cams.map((c, i) => `<option value="${c.deviceId}">${c.label || 'Camera ' + (i + 1)}</option>`).join('');
  $('camera').value = current;
}

async function toggleExposure() {
  const note = $('camNote');
  if (!S.track || !S.track.getCapabilities) { note.textContent = 'This browser cannot change the camera exposure.'; return; }
  const caps = S.track.getCapabilities(), want = S.locked ? 'continuous' : 'manual', adv = {};
  for (const k of ['exposureMode', 'whiteBalanceMode', 'focusMode']) if (caps[k] && caps[k].includes(want)) adv[k] = want;
  if (!Object.keys(adv).length) { note.textContent = 'This camera does not let the browser lock exposure. Keep the lighting steady instead.'; return; }
  try {
    await S.track.applyConstraints({ advanced: [adv] });
    S.locked = !S.locked;
    $('lockExp').textContent = S.locked ? 'Unlock exposure' : 'Lock exposure';
    note.textContent = `${S.locked ? 'Locked' : 'Unlocked'}: ${Object.keys(adv).join(', ')}.`;
    clearBackground();
  } catch (e) { note.textContent = 'The camera refused: ' + e.message; }
}

function clearBackground() { /* nothing to reset: kept so the camera code above stays unchanged */ }

// ── capture ─────────────────────────────────────────────────────────────────
function ready() {
  if (!video.videoWidth) { status('The camera is not running.', 'err'); return false; }
  return !S.busy;
}

async function saveOne(burstId) {
  drawFrame(gctx, W, H);
  const res = await api('/api/save', { label: S.label, session: S.session, burst: burstId, image: grab.toDataURL('image/jpeg', 0.92) });
  S.samples.push(res.sample); S.counts = res.counts; S.mine.push(res.sample.id);
  renderLabels(); addThumb(res.sample);
}

async function single() {
  if (!ready()) return;
  S.busy = true; setButtons();
  try { await saveOne('single'); status(`Saved ${S.label} (${S.counts[S.label]} so far)`, 'ok'); }
  catch (e) { status('Save failed: ' + e.message, 'err'); }
  S.busy = false; setButtons();
}

async function burst() {
  if (!ready()) return;
  S.busy = true; setButtons();
  const n = Math.min(60, Math.max(2, +$('burstN').value || 20)), id = 'b' + Date.now().toString(36);
  const cd = $('countdown');
  for (const k of [3, 2, 1]) { cd.textContent = k; cd.hidden = false; await sleep(700); }
  cd.hidden = true;
  $('viewwrap').classList.add('recording');
  try {
    for (let i = 0; i < n; i++) { await saveOne(id); status(`Burst: ${i + 1} of ${n} (${S.label})`); await sleep(250); }
    status(`Burst done: ${n} × ${S.label} (${S.counts[S.label]} so far)`, 'ok');
  } catch (e) { status('Burst stopped: ' + e.message, 'err'); }
  $('viewwrap').classList.remove('recording');
  S.busy = false; setButtons();
}

async function deleteSample(id) {
  await api('/api/delete', { id });
  const s = S.samples.find((x) => x.id === id);
  S.samples = S.samples.filter((x) => x.id !== id);
  S.mine = S.mine.filter((x) => x !== id);
  if (s) S.counts[s.label] = Math.max(0, S.counts[s.label] - 1);
  renderLabels(); renderGallery();
}

async function undo() {
  if (S.busy || !S.mine.length) return;
  const id = S.mine[S.mine.length - 1];
  try { await deleteSample(id); status(`Removed ${id}`); } catch (e) { status('Undo failed: ' + e.message, 'err'); }
}

function newSession() {
  S.session = sessionName() + '-' + Math.random().toString(36).slice(2, 5);
  $('sessionInfo').textContent = `session ${S.session}`;
  status(`New session ${S.session}`, 'ok');
}

// ── panels ──────────────────────────────────────────────────────────────────
function renderLabels() {
  $('labels').innerHTML = LABELS.map((l) => {
    const c = S.counts[l] || 0, pct = Math.min(100, Math.round(100 * c / TARGET[l]));
    return `<button type="button" class="lab${l === S.label ? ' on' : ''}${l === 'none' ? ' wide' : ''}" data-label="${l}">` +
      `<b>${l}</b><small>${c} / ${TARGET[l]}</small><i><em style="width:${pct}%"></em></i></button>`;
  }).join('');
  $('hintTitle').textContent = S.label;
  $('hintText').textContent = HINTS[S.label];
  const checked = S.samples.filter((s) => s.checked).length;
  $('total').textContent = `${S.samples.length} photos, ${checked} with a checked box`;
}

function setLabel(l) { S.label = l; renderLabels(); }

function setButtons() {
  $('shot').disabled = S.busy; $('burst').disabled = S.busy;
  $('undo').disabled = S.busy || !S.mine.length;
}

const imgUrl = (s) => `/dataset/images/${s.id}.jpg`;
const loadImg = (src) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });

function thumbNode(s) {
  const d = document.createElement('div');
  d.className = 'thumb';
  d.title = 'Open in the box labeler';
  d.innerHTML = `<canvas width="160" height="120"></canvas><div class="cap"><b>${s.label}</b><span>${s.checked ? '<span class="ed">box checked</span>' : 'box not checked'}</span></div><button type="button" class="x" title="Delete">✕</button>`;
  d.querySelector('.x').addEventListener('click', (e) => { e.stopPropagation(); deleteSample(s.id).catch((er) => status(er.message, 'err')); });
  d.addEventListener('click', () => window.open(`label.html#${s.id}`, 'labeler'));
  drawThumb(d.querySelector('canvas'), s);
  return d;
}

function renderGallery() {
  const f = $('filter').value || 'all';
  const list = S.samples.filter((s) => f === 'all' || s.label === f).slice(-60).reverse();
  $('grid').replaceChildren(...list.map(thumbNode));
}

function addThumb(s) {                       // a new capture: add one tile instead of redrawing the gallery
  const f = $('filter').value || 'all';
  if (f !== 'all' && s.label !== f) return;
  const grid = $('grid');
  grid.prepend(thumbNode(s));
  while (grid.children.length > 60) grid.lastChild.remove();
}

async function drawThumb(c, s) {
  try {
    const img = await loadImg(imgUrl(s)), ctx = c.getContext('2d');
    ctx.save();
    if (mirror()) { ctx.translate(160, 0); ctx.scale(-1, 1); }
    ctx.drawImage(img, 0, 0, 160, 120);
    strokeBox(ctx, s.box, 160 / W, 120 / H, 3);
    ctx.restore();
  } catch { /* file missing: leave the tile blank */ }
}

// ── wiring ──────────────────────────────────────────────────────────────────
async function api(path, body) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

$('labels').addEventListener('click', (e) => { const b = e.target.closest('.lab'); if (b) setLabel(b.dataset.label); });
$('shot').addEventListener('click', single);
$('burst').addEventListener('click', burst);
$('undo').addEventListener('click', undo);
$('newSession').addEventListener('click', newSession);
$('lockExp').addEventListener('click', toggleExposure);
$('mirror').addEventListener('change', () => { store.set('mirror', mirror() ? '1' : '0'); renderGallery(); });
$('filter').addEventListener('change', renderGallery);
$('camera').addEventListener('change', async () => {
  try { const id = await startCamera($('camera').value); store.set('camera', id); } catch (e) { status('Could not open that camera: ' + e.message, 'err'); }
});


document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName) && e.target.type !== 'range' && e.target.type !== 'checkbox') return;
  const k = e.key.toLowerCase();
  if (k === ' ') {
    e.preventDefault();
    if (document.activeElement && document.activeElement.tagName === 'BUTTON') document.activeElement.blur();   // no double capture
    single();
  }
  else if (k === 'b') burst();
  else if (k === 'z') undo();
  else if (KEY_TO_LABEL[k]) setLabel(KEY_TO_LABEL[k]);
});

async function init() {
  $('mirror').checked = store.get('mirror', '1') === '1';
  $('filter').innerHTML = '<option value="all">All labels</option>' + LABELS.map((l) => `<option value="${l}">${l}</option>`).join('');
  $('sessionInfo').textContent = `session ${S.session}`;
  try {
    const r = await fetch('/api/samples').then((x) => x.json());
    S.samples = r.samples;
    for (const s of S.samples) S.counts[s.label] = (S.counts[s.label] || 0) + 1;
  } catch { status('Cannot reach collect_server.py. Is it running?', 'err'); }
  renderLabels(); renderGallery(); setButtons();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { status('This browser cannot use the webcam here. Open http://localhost:8000 in Chrome or Edge.', 'err'); return; }
  try { await setupCameras(); } catch (e) { status('Camera error: ' + e.message + '. Allow camera access and reload.', 'err'); }
  requestAnimationFrame(loop);
}
init();
