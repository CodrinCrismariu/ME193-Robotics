// Campus navigator page. Python sends:
//   live    4 times a second: {counts, heading (deg since the last turn, + = left), moving, turns, walk}
//   walks   on every change: {waypoints, active, walks: [summary], settings, message?}
//   status  every 5 s: {rate, gaps, mcu, work_ms, model}
// and receives settings {walker, height_m}, walk_start {loc}, walk_label {loc}, walk_undo,
// walk_stop {save}, turn_edit {id, index, op}, label_edit {id, index, op, loc}, walk_delete {id}.

const socket = io(`http://${window.location.host}`, { path: '/socket.io', transports: ['polling', 'websocket'] });
const $ = (id) => document.getElementById(id);
const STALE_MS = 2000;

let state = { waypoints: [], active: null, walks: [], settings: {} };
let lastLive = 0;
let openWalks = new Set();
let editing = null;           // {id, index} of the turn being edited
let calibrated = false;
let startLoc = null;
let navHere = null, navGoal = null;   // picked on the page before Start navigating          // starting point picked on the page (not sent until Start recording)
const RECALIB_S = 180;        // remind to stand still after this long without it

socket.on('connect', () => setStatus(true, 'Connected, waiting for data…'));
socket.on('disconnect', () => setStatus(false, 'Disconnected from the board'));

socket.on('status', (s) => {
  const fresh = Date.now() - lastLive < STALE_MS;
  const work = s.work_ms == null ? '' : ` · ${s.work_ms.toFixed(0)} ms per update`;
  const model = s.model ? '' : ' · step counter model missing!';
  setStatus(fresh, `${s.rate.toFixed(0)} Hz · ${s.gaps} gaps${work}${model} · ${s.mcu}`);
});

socket.on('live', (m) => {
  lastLive = Date.now();
  renderLive(m);
  if (m.walk) { state.active = m.walk; renderActive(); }
  renderNav(m.nav);
});

socket.on('walks', (s) => {
  state = s;
  $('message').textContent = s.message || '';
  const w = $('walker'), h = $('height');
  if (document.activeElement !== w) w.value = s.settings.walker || '';
  if (document.activeElement !== h) h.value = (s.settings.height_m || 1.75).toFixed(2);
  renderWalkCard();
  renderSaved();
  renderNavPicker();
});

function setStatus(live, text) {
  $('dot').classList.toggle('live', live);
  $('status-text').textContent = text;
}

// --- live ---------------------------------------------------------------------------

const ACTIVITY = { standing: 'Standing', walking: 'Walking', running: 'Running', backward: 'Backwards' };

function renderCalibration(m) {
  if (m.calibrated !== calibrated) {
    calibrated = m.calibrated;
    renderStartButton();
    renderNavPicker();
  }
  const c = $('calib');
  c.classList.toggle('ok', calibrated);
  c.textContent = calibrated
    ? 'Gyro drift measured. Pick where you are and press Start recording.'
    : 'Stand still for a few seconds: measuring the gyro drift…';
  const r = $('recalib');
  r.hidden = !(m.since_still != null && m.since_still > RECALIB_S);
  if (!r.hidden) r.textContent = `No still moment for ${Math.round(m.since_still / 60)} min: stand still for 3 s so turns stay accurate.`;
}

function renderLive(m) {
  renderCalibration(m);
  const c = m.counts;
  const w = m.walk;
  $('steps').textContent = w ? w.steps : c ? c.steps : '—';
  $('distance').textContent = w ? `${w.distance.toFixed(1)} m` : c ? `${c.distance_m.toFixed(1)} m` : '—';
  $('activity').textContent = c ? ACTIVITY[c.activity] || c.activity : '—';
  // Arrow: which way you face compared with the current straight (+ = turned left).
  $('arrow').style.transform = `rotate(${-m.heading}deg)`;
  $('deg').textContent = `${Math.abs(Math.round(m.heading))}° ${m.heading > 5 ? 'left' : m.heading < -5 ? 'right' : ''}`;
  const recent = $('recent');
  const now = Date.now() / 1000;
  const turns = (m.turns || []).filter((t) => now - t.at < 60);
  if (turns.length) {
    recent.innerHTML = '';
    for (const t of turns.slice().reverse()) {
      recent.appendChild(chip(turnWord(t.angle), t.angle > 0 ? 'left' : 'right',
        `${Math.round(now - t.at)} s ago`));
    }
  }
}

function turnWord(angle) {
  const side = angle > 0 ? 'left' : 'right';
  return Math.abs(angle) >= 180 ? `Turn ${side} ×2` : `Turn ${side}`;
}

// --- recording a walk ---------------------------------------------------------------

function renderWalkCard() {
  const active = !!state.active;
  $('walk-idle').hidden = active;
  $('walk-active').hidden = !active;
  // Tapping a waypoint only picks the starting point; nothing records until Start recording.
  const start = $('start-buttons');
  start.innerHTML = '';
  for (const loc of state.waypoints) {
    const b = document.createElement('button');
    b.textContent = loc;
    b.classList.toggle('selected', loc === startLoc);
    b.addEventListener('click', () => {
      startLoc = loc === startLoc ? null : loc;
      renderWalkCard();
    });
    start.appendChild(b);
  }
  renderStartButton();
  if (active) renderActive();
}

function renderStartButton() {
  const b = $('start');
  b.disabled = !calibrated || !startLoc;
  b.textContent = startLoc ? `Start recording at ${startLoc}` : 'Start recording';
  $('start-hint').textContent = !calibrated ? 'Waiting for the gyro drift to be measured.'
    : !startLoc ? 'Pick your starting point first.' : '';
}

$('start').addEventListener('click', () => {
  if (!startLoc) return;
  if (!state.settings.walker && !confirm('No walker name saved. Start anyway?')) return;
  socket.emit('walk_start', { loc: startLoc });
  startLoc = null;
});

$('reset').addEventListener('click', () => {
  if (confirm('Reset steps, distance and turns to zero?')) socket.emit('reset', {});
});

function renderActive() {
  const w = state.active;
  if (!w) return;
  const here = w.labels[w.labels.length - 1].loc;
  $('walk-from').textContent = here;
  $('walk-stats').textContent = `${fmtTime(w.elapsed)} · ${w.distance.toFixed(1)} m · ${w.steps} steps`
    + ` · ${w.labels.length - 1} waypoint${w.labels.length === 2 ? '' : 's'} reached`;
  // Directions since the last waypoint (the leg in progress): turns so far.
  const t0 = w.labels[w.labels.length - 1].t;
  const dirs = $('walk-dirs');
  dirs.innerHTML = '';
  const legTurns = w.turns.filter((t) => t.t > t0);
  if (!legTurns.length) dirs.appendChild(chip('Forward', 'forward'));
  for (const t of legTurns) dirs.appendChild(chip(turnWord(t.angle), t.angle > 0 ? 'left' : 'right'));
  const reach = $('reach-buttons');
  const key = here + state.waypoints.join();
  if (reach.dataset.key !== key) {
    reach.dataset.key = key;
    reach.innerHTML = '';
    for (const loc of state.waypoints) {
      const b = document.createElement('button');
      b.textContent = loc;
      b.className = loc === here ? 'here' : 'primary';
      b.disabled = loc === here;
      b.addEventListener('click', () => socket.emit('walk_label', { loc }));
      reach.appendChild(b);
    }
  }
}

$('settings').addEventListener('submit', (e) => {
  e.preventDefault();
  socket.emit('settings', { walker: $('walker').value.trim(), height_m: parseFloat($('height').value) });
});
$('undo').addEventListener('click', () => socket.emit('walk_undo', {}));
$('stop').addEventListener('click', () => socket.emit('walk_stop', { save: true }));
$('discard').addEventListener('click', () => {
  if (confirm('Throw this walk away?')) socket.emit('walk_stop', { save: false });
});

// --- saved walks ----------------------------------------------------------------------

function renderSaved() {
  $('walk-count').textContent = state.walks.length;
  const box = $('walks');
  box.innerHTML = '';
  if (!state.walks.length) {
    box.innerHTML = '<p class="hint">None yet.</p>';
    return;
  }
  for (const w of state.walks) {
    const d = document.createElement('details');
    d.className = 'walk';
    d.open = openWalks.has(w.id);
    d.addEventListener('toggle', () => (d.open ? openWalks.add(w.id) : openWalks.delete(w.id)));
    const when = new Date(w.created * 1000).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    const s = document.createElement('summary');
    s.textContent = `${when} · ${w.walker} · ${w.legs.length} leg${w.legs.length === 1 ? '' : 's'} · ${fmtTime(w.seconds)}`;
    d.appendChild(s);

    for (const leg of w.legs) {
      const el = document.createElement('div');
      el.className = 'leg';
      el.innerHTML = '<div class="leg-title"></div><div class="leg-meta"></div><div class="dirs"></div>';
      el.querySelector('.leg-title').textContent = `${leg.from} → ${leg.to}`;
      el.querySelector('.leg-meta').textContent = `${leg.distance.toFixed(1)} m · ${leg.steps} steps · ${fmtTime(leg.t1 - leg.t0)}`;
      const dirs = el.querySelector('.dirs');
      let lastTurn = null;
      for (const it of leg.items) {
        if (it.do === 'forward') {
          dirs.appendChild(chip(`Forward ${it.m.toFixed(0)} m`, 'forward'));
        } else if (it.turn !== lastTurn) {            // a U-turn is two chips, one button
          lastTurn = it.turn;
          const angle = w.turns[it.turn].angle;
          const b = chip(turnWord(angle), it.do, 'Tap to fix', 'button');
          b.addEventListener('click', () => {
            editing = editing && editing.id === w.id && editing.index === it.turn ? null : { id: w.id, index: it.turn };
            renderSaved();
          });
          dirs.appendChild(b);
          if (editing && editing.id === w.id && editing.index === it.turn) {
            dirs.after(editBox(w, it.turn));
          }
        }
      }
      d.appendChild(el);
    }

    const labels = document.createElement('div');
    labels.className = 'labels';
    w.labels.forEach((l, i) => {
      const row = document.createElement('div');
      row.className = 'row hint';
      const sel = document.createElement('select');
      for (const loc of state.waypoints) sel.add(new Option(loc, loc, false, loc === l.loc));
      sel.addEventListener('change', () => socket.emit('label_edit', { id: w.id, index: i, op: 'rename', loc: sel.value }));
      row.append(`${fmtTime(l.t)} `, sel);
      if (i > 0 && w.labels.length > 2) {
        const del = smallButton('Remove', () => socket.emit('label_edit', { id: w.id, index: i, op: 'delete' }));
        row.append(del);
      }
      labels.appendChild(row);
    });
    const wrap = document.createElement('details');
    wrap.innerHTML = '<summary class="hint">Waypoint labels</summary>';
    wrap.appendChild(labels);
    d.appendChild(wrap);

    const tools = document.createElement('div');
    tools.className = 'row';
    tools.style.marginTop = '8px';
    tools.appendChild(smallButton('Delete walk', () => {
      if (confirm('Delete this walk for good?')) socket.emit('walk_delete', { id: w.id });
    }, 'danger'));
    if (w.lost_samples) tools.append(` ${w.lost_samples} samples lost`);
    d.appendChild(tools);
    box.appendChild(d);
  }
}

function editBox(w, index) {
  const t = w.turns[index];
  const box = document.createElement('div');
  box.className = 'edit-box';
  box.innerHTML = `<div class="hint">Measured ${t.measured > 0 ? 'left' : 'right'} ${Math.abs(Math.round(t.measured))}° at ${fmtTime(t.t)}`
    + `${t.auto !== t.angle ? ` (detected as ${turnWord(t.auto).toLowerCase()})` : ''}. It was really:</div>`;
  const row = document.createElement('div');
  row.className = 'row';
  const send = (op) => { editing = null; socket.emit('turn_edit', { id: w.id, index, op }); };
  row.append(smallButton('Turn left', () => send('left')), smallButton('Turn right', () => send('right')),
    smallButton('U-turn', () => send('around')), smallButton('Not a turn', () => send('delete'), 'danger'));
  box.appendChild(row);
  return box;
}

function chip(text, kind, title = '', tag = 'span') {
  const el = document.createElement(tag);
  el.className = `dir ${kind}`;
  el.textContent = { forward: '↑ ', left: '↰ ', right: '↱ ' }[kind] + text;
  if (title) el.title = title;
  return el;
}

function smallButton(text, onClick, extra = '') {
  const b = document.createElement('button');
  b.className = `small ${extra}`;
  b.textContent = text;
  b.addEventListener('click', onClick);
  return b;
}

function fmtTime(s) {
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
}

// --- navigation -------------------------------------------------------------------------

const ARROW = { forward: '↑', left: '↰', right: '↱', 'around-left': '↶', 'around-right': '↷' };

function renderNavPicker() {
  const opts = state.nav_options || {};
  const here = $('nav-here'), goal = $('nav-goal');
  here.innerHTML = '';
  for (const loc of state.waypoints) {
    const b = document.createElement('button');
    b.textContent = loc;
    b.classList.toggle('selected', loc === navHere);
    b.addEventListener('click', () => { navHere = loc === navHere ? null : loc; navGoal = null; renderNavPicker(); });
    here.appendChild(b);
  }
  goal.innerHTML = '';
  const reachable = navHere ? opts[navHere] || [] : [];
  for (const loc of state.waypoints) {
    if (!reachable.includes(loc)) continue;
    const b = document.createElement('button');
    b.textContent = loc;
    b.classList.toggle('selected', loc === navGoal);
    b.addEventListener('click', () => { navGoal = loc === navGoal ? null : loc; renderNavPicker(); });
    goal.appendChild(b);
  }
  $('nav-to-hint').textContent = !navHere ? '(pick where you are first)'
    : reachable.length ? '' : '(no recorded walk goes anywhere from here yet)';
  const start = $('nav-start');
  start.disabled = !calibrated || !navHere || !navGoal;
  $('nav-start-hint').textContent = !Object.values(opts).some((v) => v.length) ? 'Record a walk first.'
    : !calibrated ? 'Waiting for the gyro drift to be measured.' : '';
}

$('nav-start').addEventListener('click', () => {
  if (navHere && navGoal) socket.emit('nav_start', { here: navHere, goal: navGoal });
});
$('nav-stop').addEventListener('click', () => socket.emit('nav_stop', {}));

let navKey = '';
function renderNav(n) {
  $('nav-idle').hidden = !!n;
  $('nav-active').hidden = !n;
  $('walk-card').hidden = !!n;               // one thing at a time: navigating or recording
  if (!n) return;
  const arrow = $('nav-arrow');
  if (n.arrived) {
    arrow.textContent = '✓';
    arrow.className = 'nav-arrow arrived';
    $('nav-text').textContent = n.text;
    $('nav-sub').textContent = '';
  } else {
    arrow.textContent = ARROW[n.arrow === 'forward' ? 'forward' : n.action] || '↑';
    arrow.className = `nav-arrow ${n.arrow}`;
    $('nav-text').textContent = n.text;
    $('nav-sub').textContent = `Leg ${n.leg + 1} of ${n.path.length - 1}: ${n.from} → ${n.to} · ${Math.max(0, n.walked).toFixed(0)} of ${n.leg_length.toFixed(0)} m`;
  }
  $('nav-message').textContent = n.message || '';
  // the route as chips, the current leg outlined
  const key = n.path.join('|') + (n.arrived ? 'A' : n.leg);
  if (key !== navKey) {
    navKey = key;
    const route = $('nav-route');
    route.innerHTML = '';
    n.path.forEach((loc, i) => {
      const done = n.arrived || i <= (n.leg ?? 0);
      const c = document.createElement('span');
      c.className = `dir forward${done && !(i === n.leg + 1) ? ' done' : ''}${!n.arrived && i === n.leg + 1 ? ' current' : ''}`;
      c.textContent = (i ? '→ ' : '') + loc;
      route.appendChild(c);
    });
    const at = $('nav-at');
    at.innerHTML = '';
    for (const loc of state.waypoints) {
      at.appendChild(smallButton(`I'm at ${loc}`, () => socket.emit('nav_at', { loc })));
    }
  }
}
