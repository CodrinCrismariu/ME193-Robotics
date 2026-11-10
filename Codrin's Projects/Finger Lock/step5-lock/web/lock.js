'use strict';
// Sign lock page: webcam frames -> lock_server.py (hand detector + letter classifier) -> lock policy.
//
// The lock policy
//   idle     LEDs off. Pressing the button (Space) wakes the lock.
//   listen   LEDs glow. A letter is accepted when the same letter (not "none") stays above the confidence
//            threshold, averaged over the last few frames, for the hold time (0.8 s by default). After each letter
//            the hand has to change or leave before the next one counts, so one long "C" is not "C, C, C".
//            Nothing accepted for 20 s: back to idle.
//   check    after as many letters as the code has (CODRIN: 6): the right code opens the lock for 6 s, then it locks again by itself;
//            a wrong code flashes red. Three wrong codes in a row lock the button for 30 s.

const LABELS = ['C', 'O', 'D', 'R', 'I', 'N', 'none'];
const LETTERS = ['C', 'O', 'D', 'R', 'I', 'N'];
const W = 320, H = 240;          // frame size sent to the networks (what they were trained on)
const VW = 640, VH = 480;        // on-screen view
const MIN_LEN = 3, MAX_LEN = 8;  // secret code length (default CODRIN: 6 letters)
const TIMEOUT_MS = 20000, OPEN_MS = 6000, WRONG_MS = 2500, LOCKOUT_MS = 30000, MAX_WRONG = 3, RELEASE_MS = 300;

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const canvas = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
const store = { get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch { return d; } },
                set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } } };

const S = { stream: null, track: null, locked: false, last: null, hist: [], avg: null, frames: 0, fpsT: performance.now(), fps: 0, ms: 0 };
const video = document.createElement('video');
video.playsInline = true;
video.muted = true;
const grab = canvas(W, H), gctx = grab.getContext('2d', { willReadFrequently: true });
const inf = canvas(W, H), ictx = inf.getContext('2d');
const view = $('view'), vctx = view.getContext('2d');

function status(msg, kind) { const el = $('status'); el.textContent = msg; el.className = 'status' + (kind ? ' ' + kind : ''); }
const mirror = () => $('mirror').checked;

// ── drawing a camera frame: center-cropped to 4:3 so nothing is squashed ────
function drawFrame(ctx, w, h) {
  const vw = video.videoWidth, vh = video.videoHeight;
  let sw = vw, sh = vh;
  if (vw / vh > 4 / 3) sw = vh * 4 / 3; else sh = vw * 3 / 4;
  ctx.drawImage(video, (vw - sw) / 2, (vh - sh) / 2, sw, sh, 0, 0, w, h);
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
  let current = '';
  try {
    current = await startCamera(store.get('camera', '') || undefined).catch(() => startCamera());
  } catch (e) {
    status(`Could not open the camera (${e.message}). Choose one in the Camera box.`, 'err');
  }
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
  const builtIn = /integrated|built-?in|internal|front|facetime/i;
  const curCam = cams.find((c) => c.deviceId === current);
  if (!store.get('camera', '') && curCam && builtIn.test(curCam.label)) {
    const usb = cams.find((c) => !builtIn.test(c.label));
    if (usb) {
      try { current = await startCamera(usb.deviceId); $('camNote').textContent = `Switched to ${usb.label}: the assignment asks for the USB webcam.`; }
      catch { current = await startCamera(current).catch(() => ''); }          // the USB webcam failed: back to the one that worked
    }
  }
  // always fill the list, even if no camera started; with none running, "Choose a camera…" comes first so any pick counts
  $('camera').innerHTML = (current ? '' : '<option value="" disabled>Choose a camera…</option>') +
    cams.map((c, i) => `<option value="${c.deviceId}">${c.label || 'Camera ' + (i + 1)}</option>`).join('');
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

// ── live view: video, the detector's box and the current letter ─────────────
function topLetter() {
  if (!S.avg) return null;
  let b = 0;
  S.avg.forEach((v, i) => { if (v > S.avg[b]) b = i; });
  return { label: LABELS[b], p: S.avg[b] };
}

function loop() {
  if (video.readyState >= 2 && video.videoWidth) {
    drawFrame(gctx, W, H);
    vctx.save();
    if (mirror()) { vctx.translate(VW, 0); vctx.scale(-1, 1); }
    vctx.drawImage(grab, 0, 0, VW, VH);
    vctx.restore();
    const p = S.last;
    if (p && p.probs) {
      const sx = VW / W, sy = VH / H;
      let [x0, y0, x1, y1] = p.box;
      if (mirror()) [x0, x1] = [W - x1, W - x0];
      vctx.lineWidth = 5;
      vctx.strokeStyle = '#e8743b';
      vctx.strokeRect(x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy);
      const t = topLetter();
      if (t) {
        const txt = `${t.label} ${Math.round(t.p * 100)}%`;
        vctx.font = '700 26px Segoe UI, sans-serif';
        const tw = vctx.measureText(txt).width + 16, ty = Math.max(0, y0 * sy - 38);
        vctx.fillStyle = '#e8743b';
        vctx.fillRect(x0 * sx, ty, tw, 34);
        vctx.fillStyle = '#fff';
        vctx.fillText(txt, x0 * sx + 8, ty + 26);
      }
    }
  }
  requestAnimationFrame(loop);
}

// ── inference: one frame at a time, so frames never queue up ────────────────
async function api(path, body) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

async function inferLoop() {
  for (;;) {
    if (!video.videoWidth) { await sleep(200); continue; }
    drawFrame(ictx, W, H);                           // the raw frame (never mirrored), as in training
    try {
      onPrediction(await api('/api/predict', { image: inf.toDataURL('image/jpeg', 0.85) }));
    } catch (e) {
      status('Cannot reach lock_server.py: ' + e.message, 'err');
      await sleep(1000);
    }
  }
}

function onPrediction(r) {
  S.last = r;
  S.ms = r.ms;
  const none = LABELS.indexOf('none');
  S.hist.push(r.probs || LABELS.map((_, i) => (i === none ? 1 : 0)));   // no hand counts as 100% "none"
  if (S.hist.length > 20) S.hist.shift();
  const k = +$('smooth').value, recent = S.hist.slice(-k);
  S.avg = LABELS.map((_, i) => recent.reduce((s, p) => s + p[i], 0) / recent.length);
  const t = topLetter();
  const cand = t.p >= +$('minConf').value && t.label !== 'none' ? t.label : null;
  policy(cand);
  renderBars();
  $('handInfo').textContent = r.probs ? `hand found (${Math.round(r.hand * 100)}% sure)` : 'no hand';
  S.frames++;
  const now = performance.now();
  if (now - S.fpsT > 1000) { S.fps = S.frames * 1000 / (now - S.fpsT); S.frames = 0; S.fpsT = now; }
  $('speed').textContent = `${S.fps.toFixed(1)} frames/s · networks ${S.ms} ms per frame`;
}

// ── the lock policy ─────────────────────────────────────────────────────────
const P = { state: 'idle', entered: [], wrong: 0, until: 0, last: 0, holdL: null, holdT: 0, release: false, relL: null, relT: 0, note: '' };
const code = () => $('code').value.toUpperCase();
const codeLen = () => code().length;

function press() {
  ensureAudio();
  if (P.state === 'lockout' || P.state === 'open') return;
  if (!validCode()) { status(`Set a code of ${MIN_LEN} to ${MAX_LEN} letters from C O D R I N in Settings first.`, 'err'); return; }
  Object.assign(P, { state: 'listen', entered: [], last: performance.now(), holdL: null, release: false, note: '' });
  tone(660, 0.08);
  render();
}

function policy(cand) {
  const now = performance.now();
  if (P.state === 'open' && now > P.until) Object.assign(P, { state: 'idle', entered: [], note: 'Locked again.' });
  if (P.state === 'wrong' && now > P.until) {
    if (P.wrong >= MAX_WRONG) Object.assign(P, { state: 'lockout', until: now + LOCKOUT_MS, entered: [] });
    else Object.assign(P, { state: 'idle', entered: [] });
  }
  if (P.state === 'lockout' && now > P.until) Object.assign(P, { state: 'idle', wrong: 0, note: '' });
  if (P.state !== 'listen') { setRing(0, null); render(); return; }
  if (now - P.last > TIMEOUT_MS) { Object.assign(P, { state: 'idle', entered: [], note: 'Timed out. Press the button to try again.' }); render(); return; }
  if (P.release) {                                  // after a letter: the same letter can't count again until the hand changes
    if (cand && cand !== P.relL) Object.assign(P, { release: false, relT: 0 });     // a different letter: go on at once
    else if (!cand) {                                                               // hand down or unclear: wait a moment
      if (!P.relT) P.relT = now;
      if (now - P.relT >= RELEASE_MS) Object.assign(P, { release: false, relT: 0 });
    } else P.relT = 0;
    if (P.release) { setRing(0, null); render(); return; }
  }
  if (!cand) { P.holdL = null; setRing(0, null); render(); return; }
  if (cand !== P.holdL) { P.holdL = cand; P.holdT = now; }
  const frac = (now - P.holdT) / +$('holdMs').value;
  setRing(frac, cand);
  if (frac >= 1) {
    P.entered.push(cand);
    Object.assign(P, { last: now, holdL: null, release: true, relL: cand, relT: 0 });
    tone(880, 0.09);
    if (P.entered.length === codeLen()) check();
  }
  render();
}

function check() {
  const now = performance.now();
  if (P.entered.join('') === code()) {
    Object.assign(P, { state: 'open', wrong: 0, until: now + OPEN_MS });
    [523, 659, 784].forEach((f, i) => tone(f, 0.6, 'sine', i * 0.09));
  } else {
    P.wrong++;
    Object.assign(P, { state: 'wrong', until: now + WRONG_MS });
    tone(140, 0.45, 'square', 0, 0.06);
  }
}

function render() {
  const n = P.entered.length;
  for (let i = 0; i < codeLen() && $('led' + i); i++) {
    const led = $('led' + i);
    led.className = P.state === 'open' ? 'ok' : P.state === 'wrong' || P.state === 'lockout' ? 'bad'
      : P.state === 'listen' ? (i < n ? 'on' : 'listen') : '';
    $('slot' + i).textContent = P.entered[i] || '·';
  }
  $('lockcard').classList.toggle('open', P.state === 'open');
  $('lockcard').classList.toggle('bad', P.state === 'wrong');
  const left = Math.max(0, Math.ceil((P.until - performance.now()) / 1000));
  $('msg').textContent = {
    idle: P.note || 'Press the button to start',
    listen: P.release ? 'Got it. Change or lower your hand…' : `Sign letter ${n + 1} of ${codeLen()} and hold it`,
    open: 'Unlocked!',
    wrong: `Wrong code (${P.wrong} of ${MAX_WRONG} tries)`,
    lockout: `Too many wrong codes. Try again in ${left} s`,
  }[P.state];
  const btn = $('wake');
  btn.disabled = P.state === 'lockout' || P.state === 'open';
  btn.firstChild.textContent = P.state === 'listen' ? 'Start over ' : 'Press the button ';
}

function setRing(frac, cand) {
  $('ring').style.strokeDashoffset = String(264 * (1 - Math.max(0, Math.min(1, frac))));
  $('cand').textContent = cand || '–';
}

// ── sounds (made on the fly, no files) ──────────────────────────────────────
let AC = null;
function ensureAudio() { if (!AC) { try { AC = new (window.AudioContext || window.webkitAudioContext)(); } catch { AC = null; } } }
function tone(freq, dur, type = 'sine', when = 0, vol = 0.15) {
  if (!AC) return;
  const o = AC.createOscillator(), g = AC.createGain(), t = AC.currentTime + when;
  o.type = type;
  o.frequency.value = freq;
  o.connect(g);
  g.connect(AC.destination);
  g.gain.setValueAtTime(vol, t);
  g.gain.exponentialRampToValueAtTime(0.001, t + dur);
  o.start(t);
  o.stop(t + dur);
}

// ── probability bars and settings ───────────────────────────────────────────
function buildBars() {
  $('bars').innerHTML = LABELS.map((l, i) => `<b>${l}</b><i><em id="bar${i}"></em></i><span id="pct${i}">0%</span>`).join('');
}

function renderBars() {
  const t = topLetter();
  S.avg.forEach((v, i) => {
    const bar = $('bar' + i);
    bar.style.width = `${(v * 100).toFixed(1)}%`;
    bar.className = LABELS[i] === t.label ? 'top' : '';
    $('pct' + i).textContent = `${Math.round(v * 100)}%`;
  });
}

function validCode() { const c = code(); return c.length >= MIN_LEN && c.length <= MAX_LEN && [...c].every((ch) => LETTERS.includes(ch)); }

const KEYS = { code: 'lock_secret', holdMs: 'lock_holdMs', minConf: 'lock_minConf', smooth: 'lock_smooth' };

function buildLock() {                               // one LED and one slot per letter of the code
  const n = validCode() ? codeLen() : 6;
  $('leds').innerHTML = Array.from({ length: n }, (_, i) => `<i id="led${i}"></i>`).join('');
  $('slots').innerHTML = Array.from({ length: n }, (_, i) => `<b id="slot${i}">·</b>`).join('');
}

function settings() {
  $('holdOut').value = `${(+$('holdMs').value / 1000).toFixed(1)} s`;
  $('confOut').value = `${Math.round(+$('minConf').value * 100)}%`;
  $('smoothOut').value = `${$('smooth').value} frame${$('smooth').value === '1' ? '' : 's'}`;
  $('code').style.borderColor = validCode() ? '' : '#e34948';
  for (const [id, key] of Object.entries(KEYS)) store.set(key, $(id).value);
  if ($('leds').children.length !== (validCode() ? codeLen() : 6)) {
    Object.assign(P, { state: P.state === 'listen' ? 'idle' : P.state, entered: [] });   // the code changed length: start over
    buildLock();
    render();
  }
}

// ── wiring ──────────────────────────────────────────────────────────────────
$('wake').addEventListener('click', press);
for (const id of ['code', 'holdMs', 'minConf', 'smooth']) $(id).addEventListener('input', settings);
$('mirror').addEventListener('change', () => store.set('mirror', mirror() ? '1' : '0'));
$('lockExp').addEventListener('click', toggleExposure);
$('camera').addEventListener('change', async () => {
  try { const id = await startCamera($('camera').value); store.set('camera', id); } catch (e) { status('Could not open that camera: ' + e.message, 'err'); }
});


document.addEventListener('keydown', (e) => {
  if (e.key !== ' ' || e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  e.preventDefault();
  if (document.activeElement && document.activeElement.tagName === 'BUTTON') document.activeElement.blur();
  press();
});

async function init() {
  $('mirror').checked = store.get('mirror', '1') === '1';
  for (const [id, key] of Object.entries(KEYS)) $(id).value = store.get(key, $(id).value);
  buildLock();
  settings();
  buildBars();
  render();
  try {
    const info = await fetch('/api/info').then((x) => x.json());
    const m = info.metrics.classifier || {}, d = info.metrics.detector || {};
    $('modelInfo').textContent = `Detector "${info.detector}"` + (d.mean_iou ? ` (test box overlap ${d.mean_iou.toFixed(2)})` : '') +
      `, classifier "${info.classifier}"` + (m.test_real ? ` (test ${Math.round(m.test_real * 100)}% right)` : '') + '.';
  } catch { status('Cannot reach lock_server.py. Is it running?', 'err'); }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { status('This browser cannot use the webcam here. Open http://localhost:8001 in Chrome or Edge.', 'err'); return; }
  try { await setupCameras(); } catch (e) { status('Camera error: ' + e.message + '. Allow camera access and reload.', 'err'); }
  requestAnimationFrame(loop);
  inferLoop();
}
init();
