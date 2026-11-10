'use strict';
// Hand box labeler: draw, check and fix the bounding box of the hand on every saved photo.

const W = 320, H = 240;          // saved image size (boxes are in these pixels)
const VW = 640, VH = 480;        // canvas size

const $ = (id) => document.getElementById(id);
const store = { get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch { return d; } },
                set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } } };
const cv = $('canvas'), ctx = cv.getContext('2d');

const L = { all: [], list: [], i: 0, img: null, box: null, before: null, drag: null, cache: new Map() };

function status(msg, kind) { const el = $('status'); el.textContent = msg; el.className = 'status' + (kind ? ' ' + kind : ''); }
const mirror = () => $('mirror').checked;
const cur = () => L.list[L.i];

async function api(path, body) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

function loadImg(id) {
  if (!L.cache.has(id)) {
    L.cache.set(id, new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = `/dataset/images/${id}.jpg`; }));
    if (L.cache.size > 40) L.cache.delete(L.cache.keys().next().value);
  }
  return L.cache.get(id);
}

// ── which photos ────────────────────────────────────────────────────────────
function filterList() {
  const f = $('filter').value;
  return L.all.filter((s) => f === 'all' ? true
    : f === 'todo' ? !s.checked
    : f === 'nobox' ? s.label !== 'none' && !s.box
    : s.label === f);
}

function rebuild(keepId) {
  L.list = filterList();
  const j = keepId ? L.list.findIndex((s) => s.id === keepId) : -1;
  L.i = j >= 0 ? j : Math.min(L.i, Math.max(0, L.list.length - 1));
  $('scrub').max = Math.max(0, L.list.length - 1);
  show();
}

function progress() {
  const done = L.all.filter((s) => s.checked).length;
  $('progress').textContent = `${done} of ${L.all.length} boxes checked`;
}

// ── drawing ─────────────────────────────────────────────────────────────────
function draw() {
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, VW, VH);
  if (!L.img) return;
  ctx.save();
  if (mirror()) { ctx.translate(VW, 0); ctx.scale(-1, 1); }
  ctx.drawImage(L.img, 0, 0, VW, VH);
  const b = L.box;
  if (b) {
    const sx = VW / W, sy = VH / H, x = b[0] * sx, y = b[1] * sy, w = (b[2] - b[0]) * sx, h = (b[3] - b[1]) * sy;
    ctx.fillStyle = 'rgba(0,0,0,.35)';                          // dim everything outside the box
    ctx.fillRect(0, 0, VW, y); ctx.fillRect(0, y + h, VW, VH - y - h); ctx.fillRect(0, y, x, h); ctx.fillRect(x + w, y, VW - x - w, h);
    ctx.lineWidth = 4;
    ctx.strokeStyle = cur() && cur().checked && !L.dirty ? '#2e6b5b' : '#e8743b';
    ctx.strokeRect(x, y, w, h);
  }
  ctx.restore();
}

async function show() {
  const s = cur();
  $('where').textContent = L.list.length ? `photo ${L.i + 1} of ${L.list.length}` : '';
  $('scrub').value = L.i;
  progress();
  if (!s) {
    L.img = null; L.box = null; draw();
    $('labelBig').textContent = '–';
    $('info').textContent = '';
    status($('filter').value === 'todo' ? 'Every box is checked. Nice work.' : 'No photos here.', 'ok');
    return;
  }
  L.box = s.box ? [...s.box] : null;
  L.before = L.box; L.dirty = false;
  $('labelBig').textContent = s.label;
  $('info').textContent = `${s.id} · ${s.checked ? 'checked' : 'not checked yet'} · ${s.box ? `box ${s.box[2] - s.box[0]}×${s.box[3] - s.box[1]} px` : 'no box'}`;
  try {
    L.img = await loadImg(s.id);
    if (cur() !== s) return;                                      // moved on while it loaded
    draw();
    status(s.checked ? 'Already checked: change it or move on.' : 'Check the box, then press Enter.');
  } catch { status(`Could not load ${s.id}.jpg`, 'err'); }
  for (const k of [1, 2, 3]) if (L.list[L.i + k]) loadImg(L.list[L.i + k].id);   // preload the next few
}

// ── drawing a box with the mouse ────────────────────────────────────────────
function point(e) {
  const r = cv.getBoundingClientRect();
  let x = (e.clientX - r.left) / r.width * W;
  const y = (e.clientY - r.top) / r.height * H;
  if (mirror()) x = W - x;
  return [Math.max(0, Math.min(W, x)), Math.max(0, Math.min(H, y))];
}

cv.addEventListener('pointerdown', (e) => { if (!cur()) return; L.drag = point(e); L.prevBox = L.box; cv.setPointerCapture(e.pointerId); });
cv.addEventListener('pointermove', (e) => {
  if (!L.drag) return;
  const [x, y] = point(e), [sx, sy] = L.drag;
  L.box = [Math.round(Math.min(x, sx)), Math.round(Math.min(y, sy)), Math.round(Math.max(x, sx)), Math.round(Math.max(y, sy))];
  L.dirty = true;
  draw();
});
cv.addEventListener('pointerup', () => {
  if (!L.drag) return;
  L.drag = null;
  if (L.box && (L.box[2] - L.box[0] < 4 || L.box[3] - L.box[1] < 4)) { L.box = L.prevBox; draw(); }   // a click, not a drag
});

// ── actions ─────────────────────────────────────────────────────────────────
async function save(box) {
  const s = cur();
  if (!s) return;
  try {
    const res = await api('/api/box', { id: s.id, box, checked: true });
    const nextId = L.list[L.i + 1] && L.list[L.i + 1].id;
    Object.assign(s, res.sample);
    if (filterList().some((x) => x.id === s.id)) { rebuild(s.id); go(1); }   // still in this list: step past it
    else rebuild(nextId);                                                     // dropped out (e.g. "not checked yet"): the next one moves up
    status(`Saved ${s.id}: ${box ? 'box ' + (box[2] - box[0]) + '×' + (box[3] - box[1]) : 'no hand'}`, 'ok');
  } catch (e) { status('Could not save: ' + e.message, 'err'); }
}

function confirmBox() {
  const s = cur();
  if (!s) return;
  if (!L.box && s.label !== 'none') { status('Draw a box first, or press H if there really is no hand in this photo.', 'err'); return; }
  save(L.box);
}

async function deletePhoto() {
  const s = cur();
  if (!s || !window.confirm(`Delete ${s.id}? This removes the photo from the dataset.`)) return;
  try {
    await api('/api/delete', { id: s.id });
    L.all = L.all.filter((x) => x.id !== s.id);
    const nextId = L.list[L.i + 1] && L.list[L.i + 1].id;
    rebuild(nextId);
    status(`Deleted ${s.id}`);
  } catch (e) { status('Could not delete: ' + e.message, 'err'); }
}

function go(d) {
  if (!L.list.length) return;
  L.i = Math.max(0, Math.min(L.list.length - 1, L.i + d));
  show();
}

$('confirm').addEventListener('click', confirmBox);
$('nohand').addEventListener('click', () => save(null));
$('undoBox').addEventListener('click', () => { L.box = L.before; L.dirty = false; draw(); });
$('del').addEventListener('click', deletePhoto);
$('prev').addEventListener('click', () => go(-1));
$('next').addEventListener('click', () => go(1));
$('filter').addEventListener('change', () => { L.i = 0; rebuild(); });
$('scrub').addEventListener('input', () => { L.i = +$('scrub').value; show(); });
$('mirror').addEventListener('change', () => { store.set('mirror', mirror() ? '1' : '0'); draw(); });

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.target.tagName === 'SELECT') return;
  const k = e.key;
  if (k === 'Enter' || k === ' ') { e.preventDefault(); if (document.activeElement && document.activeElement.tagName === 'BUTTON') document.activeElement.blur(); confirmBox(); }
  else if (k === 'h' || k === 'H') save(null);
  else if (k === 'u' || k === 'U') { L.box = L.before; L.dirty = false; draw(); }
  else if (k === 'Delete') deletePhoto();
  else if (k === 'ArrowLeft' || k === 'a' || k === 'A') go(-1);
  else if (k === 'ArrowRight' || k === 'd' || k === 'D') go(1);
});

async function init() {
  $('mirror').checked = store.get('mirror', '1') === '1';
  try {
    const r = await fetch('/api/samples').then((x) => x.json());
    L.all = r.samples;
  } catch { status('Cannot reach collect_server.py. Is it running?', 'err'); return; }
  const want = decodeURIComponent(location.hash.slice(1));
  if (want && L.all.some((s) => s.id === want)) { $('filter').value = 'all'; rebuild(want); }
  else rebuild();
}
window.addEventListener('hashchange', () => {
  const want = decodeURIComponent(location.hash.slice(1));
  if (want && L.all.some((s) => s.id === want)) { $('filter').value = 'all'; rebuild(want); }
});
init();
