/* Runs Aero Lab's own page scripts on the real page markup, with a small dependency-free DOM and no 3D engine.
 * It checks behaviour that needs no rendering: the scripts load, steps switch panels, the solver choice shows only
 * its own settings, a saved case restores, and the request equals the one the builder makes.
 * It does not check layout, styles or the 3D view.
 *   node aero_page_dom.cjs <repository root>     prints {"checks": n} and exits 0, or throws */
const fs = require('node:fs'), vm = require('node:vm'), path = require('node:path'), assert = require('node:assert/strict');
const root = process.argv[2];
const staticDir = path.join(root, 'apps/ludwig_web/static');
let checks = 0;
const ok = (value, message) => { checks++; assert.ok(value, message); };
const equal = (a, b, message) => { checks++; assert.deepEqual(a, b, message); };

// ---- a minimal DOM -------------------------------------------------------------------------------------------
const VOID = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'source', 'track', 'wbr']);
class Event { constructor(type, options = {}) { this.type = type; this.bubbles = !!options.bubbles; this.target = null; } preventDefault() {} }
class Node {
  constructor(tag, attrs = {}) {
    this.tagName = tag.toUpperCase(); this.attrs = {...attrs}; this.children = []; this.parentElement = null; this.listeners = {}; this.text = '';
    this.style = {setProperty() {}, cssText: ''};
    const node = this;
    this.dataset = new Proxy({}, {
      get: (_, key) => node.attrs['data-' + String(key).replace(/[A-Z]/g, c => '-' + c.toLowerCase())],
      set: (_, key, value) => { node.attrs['data-' + String(key).replace(/[A-Z]/g, c => '-' + c.toLowerCase())] = String(value); return true; }});
    this.classList = {
      contains: name => (node.attrs.class || '').split(/\s+/).includes(name),
      add: name => { if (!node.classList.contains(name)) node.attrs.class = ((node.attrs.class || '') + ' ' + name).trim(); },
      remove: name => { node.attrs.class = (node.attrs.class || '').split(/\s+/).filter(c => c && c !== name).join(' '); },
      toggle: (name, force) => { const on = force === undefined ? !node.classList.contains(name) : !!force; on ? node.classList.add(name) : node.classList.remove(name); return on; }};
    if (this.tagName === 'INPUT') { this._value = attrs.value ?? ''; this._checked = 'checked' in attrs; }
  }
  get id() { return this.attrs.id || ''; } set id(v) { this.attrs.id = String(v); }
  get className() { return this.attrs.class || ''; } set className(v) { this.attrs.class = String(v); }
  get type() { return this.attrs.type || (this.tagName === 'SELECT' ? 'select-one' : ''); } set type(v) { this.attrs.type = v; }
  get title() { return this.attrs.title || ''; } set title(v) { this.attrs.title = String(v); }
  get name() { return this.attrs.name || ''; }
  get hidden() { return 'hidden' in this.attrs; } set hidden(v) { if (v) this.attrs.hidden = ''; else delete this.attrs.hidden; }
  get disabled() { return 'disabled' in this.attrs; } set disabled(v) { if (v) this.attrs.disabled = ''; else delete this.attrs.disabled; }
  get open() { return 'open' in this.attrs; } set open(v) { if (v) this.attrs.open = ''; else delete this.attrs.open; }
  get readOnly() { return 'readonly' in this.attrs; } set readOnly(v) { if (v) this.attrs.readonly = ''; else delete this.attrs.readonly; }
  get max() { return this.attrs.max; } set max(v) { this.attrs.max = String(v); }
  get href() { return this.attrs.href; } set href(v) { this.attrs.href = String(v); }
  get options() { return this.tagName === 'SELECT' ? all(this).filter(n => n.tagName === 'OPTION') : undefined; }
  get selectedOptions() { const o = this.options.find(n => n.value === this.value); return o ? [o] : []; }
  get selected() { return 'selected' in this.attrs; }
  get checked() { return !!this._checked; } set checked(v) { this._checked = !!v; }
  get value() {
    if (this.tagName === 'SELECT') { if (this._value !== undefined) return this._value; const o = this.options; return (o.find(n => n.selected) || o[0])?.value ?? ''; }
    if (this.tagName === 'OPTION') return this.attrs.value ?? this.textContent;
    return this._value ?? '';
  }
  set value(v) {
    if (this.tagName === 'SELECT') { if (this.options.some(n => n.value === String(v))) this._value = String(v); return; }   // a select keeps its value when no option matches
    if (this.tagName === 'OPTION') { this.attrs.value = String(v); return; }
    this._value = String(v);
  }
  get textContent() { return this.text + this.children.map(c => c.textContent).join(''); }
  set textContent(v) { this.children = []; this.text = String(v); }
  setAttribute(k, v) { this.attrs[k] = String(v); } getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  hasAttribute(k) { return k in this.attrs; } removeAttribute(k) { delete this.attrs[k]; }
  appendChild(child) { child.parentElement?.removeChild?.(child); child.parentElement = this; this.children.push(child); return child; }
  append(...items) { for (const item of items) typeof item === 'string' ? (this.text += item) : this.appendChild(item); }
  removeChild(child) { this.children = this.children.filter(c => c !== child); child.parentElement = null; }
  remove() { this.parentElement?.removeChild(this); }
  replaceChildren(...items) { this.children = []; this.text = ''; this.append(...items); }
  addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); }
  dispatchEvent(event) { event.target ||= this; for (let n = this; n; n = event.bubbles ? n.parentElement : null) for (const h of n.listeners[event.type] || []) h(event); return true; }
  click() { if (!this.disabled) this.dispatchEvent(Object.assign(new Event('click', {bubbles: true}), {target: this})); }
  focus() { documentNode.activeElement = this; } scrollIntoView() {} requestFullscreen() { return Promise.resolve(); }
  querySelectorAll(selector) { return all(this).filter(n => selector.split(',').some(s => matches(n, s.trim()))); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  closest(selector) { for (let n = this; n; n = n.parentElement) if (matches(n, selector)) return n; return null; }
}
const all = node => node.children.flatMap(child => [child, ...all(child)]);
function matchesSimple(node, selector) {
  for (const m of selector.matchAll(/\[([\w-]+)(?:=(?:"([^"]*)"|'([^']*)'|([^\]]*)))?\]/g)) {
    if (!(m[1] in node.attrs)) return false;
    const value = m[2] ?? m[3] ?? m[4]; if (value !== undefined && node.attrs[m[1]] !== value) return false;
  }
  const bare = selector.replace(/\[[^\]]*\]/g, '');
  const tag = bare.match(/^[a-z][\w-]*/i); if (tag && node.tagName !== tag[0].toUpperCase()) return false;
  const id = bare.match(/#([\w-]+)/); if (id && node.id !== id[1]) return false;
  for (const m of bare.matchAll(/\.([\w-]+)/g)) if (!node.classList.contains(m[1])) return false;
  return true;
}
function matches(node, selector) {
  const parts = selector.trim().split(/\s+/);
  if (!matchesSimple(node, parts[parts.length - 1])) return false;
  let ancestor = node.parentElement;
  for (let i = parts.length - 2; i >= 0; i--) {
    while (ancestor && !matchesSimple(ancestor, parts[i])) ancestor = ancestor.parentElement;
    if (!ancestor) return false; ancestor = ancestor.parentElement;
  }
  return true;
}
function parse(html) {
  const top = new Node('html-root'); let current = top;
  const token = /<!--[\s\S]*?-->|<!doctype[^>]*>|<(\/?)([a-zA-Z][\w-]*)((?:[^>"']|"[^"]*"|'[^']*')*)>|([^<]+)/gi;
  for (const m of html.matchAll(token)) {
    if (m[4] !== undefined) { if (current.children.length === 0) current.text += decode(m[4]); else { const t = new Node('#text'); t.text = decode(m[4]); current.appendChild(t); } continue; }
    if (!m[2]) continue;
    const tag = m[2].toLowerCase();
    if (m[1]) { assert.equal(current.tagName, tag.toUpperCase(), `unbalanced </${tag}> inside <${current.tagName.toLowerCase()} id="${current.id}">`); current = current.parentElement; continue; }
    const attrs = {}; for (const a of m[3].matchAll(/([\w:-]+)(?:="([^"]*)")?/g)) attrs[a[1]] = decode(a[2] ?? '');
    const node = current.appendChild(new Node(tag, attrs));
    if (!VOID.has(tag)) current = node;
  }
  assert.equal(current, top, 'the page leaves an element open');
  return top;
}
const decode = text => text.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"');

const page = parse(fs.readFileSync(path.join(staticDir, 'aero_lab.html'), 'utf8').replace(/^﻿/, ''));
const byId = id => all(page).find(n => n.id === id) || null;
const documentNode = {
  getElementById: byId, querySelectorAll: s => page.querySelectorAll(s), querySelector: s => page.querySelector(s),
  createElement: tag => new Node(tag), createTextNode: text => { const t = new Node('#text'); t.text = String(text); return t; },
  addEventListener() {}, body: page.querySelector('body'), documentElement: page.querySelector('html'), activeElement: null, fullscreenElement: null, hidden: false};

// ---- the browser globals the scripts use -----------------------------------------------------------------------
const capabilities = {ok: true, gpu: {available: true, free_gb: 10, total_gb: 12, budget_gb: 9.3},
  solvers: ['main_d3q27', 'main_d3q39', 'pond_thermal', 'q19_octree'].map(id => ({id, label: 'Solver ' + id, note: 'Note of ' + id, available: true, unavailable_reason: ''}))};
const messages = [];
const storage = new Map();
const sandbox = {
  document: documentNode, console, URL, URLSearchParams, Event, Promise, JSON, Math, Number, String, Array, Object, Set, Map, Date, Error,
  setTimeout, clearTimeout, setInterval() { return 0; }, clearInterval() {}, requestAnimationFrame() { return 0; },
  location: {href: 'http://127.0.0.1:8058/aero-lab', search: '', pathname: '/aero-lab', hash: ''},
  history: {replaceState() {}}, localStorage: {getItem: k => storage.get(k) ?? null, setItem: (k, v) => storage.set(k, String(v)), removeItem: k => storage.delete(k)},
  matchMedia: () => ({matches: false, addEventListener() {}}), addEventListener() {}, removeEventListener() {}, confirm: () => true,
  fetch: async url => String(url).startsWith('/api/aero/capabilities')
    ? {ok: true, status: 200, json: async () => capabilities}
    : {ok: false, status: 404, json: async () => ({ok: false, error: 'not available in the fixture'})}};
sandbox.window = sandbox; sandbox.globalThis = sandbox;
vm.createContext(sandbox);
const load = name => vm.runInContext(fs.readFileSync(path.join(staticDir, name), 'utf8'), sandbox, {filename: name});

(async () => {
  // the page's script order, without the 3D engine and the scripts this change did not touch
  for (const name of ['aero_request.js', 'aero_layout.js', 'aero_lab.js', 'aero_studies.js', 'aero_view_capture.js']) load(name);
  await new Promise(resolve => setTimeout(resolve, 20));     // capabilities arrive
  const $ = byId, layout = sandbox.LudwigAeroLayout, studies = sandbox.LudwigAeroStudiesContext, builder = sandbox.LudwigAeroRequest;
  const realShow = layout.showMessage; layout.showMessage = (text, options) => { messages.push(String(text)); return realShow(text, options); };
  ok(layout && studies && builder && sandbox.LudwigAeroStudies, 'the page scripts load and publish their interfaces');
  equal($('connection').textContent, 'Solver service online', 'the solver service is reported online');
  equal($('solver').options.map(o => o.textContent), capabilities.solvers.map(s => s.label), 'solver options take their labels from the backend');

  // steps: one panel at a time, in the order of the row
  const panels = page.querySelectorAll('[data-parameter-panel]'), visible = () => panels.filter(p => !p.hidden).map(p => p.dataset.parameterPanel);
  equal(visible(), ['model'], 'the page opens on step 1');
  equal([$('parameter-step').textContent, $('parameter-title').textContent], ['Step 1', 'Model'], 'the pane names the step');
  const expected = {model: 'Step 1', domain: 'Step 2', flow: 'Step 3', mesh: 'Step 4', run: 'Step 5', results: 'Step 6', study: '', display: '', help: ''};
  for (const button of page.querySelectorAll('.workflow-bar [data-parameter-target]')) {
    button.click(); const name = button.dataset.parameterTarget;
    equal(visible(), [name], `the ${name} button shows only its panel`);
    equal($('parameter-step').hidden ? '' : $('parameter-step').textContent, expected[name], `${name} is ${expected[name] || 'not a numbered step'}`);
    equal(button.getAttribute('aria-pressed'), 'true', `${name} is marked as the current step`);
  }
  // a control inside a closed Advanced group is revealed when something points at it
  ok(!$('advanced-results').open, 'Advanced groups start closed');
  page.querySelector('.result-access-bar [data-focus-control="results-path"]').click();
  ok(visible()[0] === 'results' && $('advanced-results').open && documentNode.activeElement === $('results-path'), 'Open other results reveals and focuses the path field');

  // the solver choice shows only its own settings
  const choose = id => { $('solver').value = id; $('solver').dispatchEvent(new Event('change')); };
  const shown = id => { for (let n = $(id); n; n = n.parentElement) if (n.hidden && !n.hasAttribute('data-parameter-panel')) return false; return true; };
  const state = () => ({gpu: shown('target-vram'), subdivision: shown('surface-triangle-max-edge-mm'), octree: shown('octree-band-fine'), octreeAdvanced: shown('octree-hrr-sigma'),
    thermal: shown('flow-mach'), cfl: shown('flow-cfl'), levels: $('levels').options.filter(o => !o.disabled).length, memory: $('metric-memory-label').textContent});
  equal(state(), {gpu: true, subdivision: true, octree: false, octreeAdvanced: false, thermal: false, cfl: false, levels: 7, memory: 'VRAM'}, 'D3Q27: GPU settings only');
  equal($('solver-note').textContent, 'Note of main_d3q27', 'the note under the solver follows the choice');
  choose('main_d3q39');
  equal(state(), {gpu: true, subdivision: true, octree: false, octreeAdvanced: false, thermal: false, cfl: false, levels: 7, memory: 'VRAM'}, 'D3Q39: GPU settings only');
  choose('pond_thermal');
  equal(state(), {gpu: true, subdivision: true, octree: false, octreeAdvanced: false, thermal: true, cfl: true, levels: 7, memory: 'VRAM'}, 'thermal: Mach, temperature and CFL appear');
  ok($('velocity').disabled && $('sound-speed').disabled && Math.abs(Number($('velocity').value) - 0.8 * Math.sqrt(1.4 * 287.05 * 288.15)) < 1e-3, 'thermal: speed is derived from Mach and temperature');
  choose('q19_octree');
  equal(state(), {gpu: false, subdivision: false, octree: true, octreeAdvanced: true, thermal: false, cfl: false, levels: 10, memory: 'RAM'}, 'CPU octree: its settings only, ten levels, RAM');
  ok(!$('velocity').disabled && !$('octree-wall-function').checked, 'CPU octree: speed is an input again and the wall function is off');
  equal([$('velocity').value, $('sound-speed').value], ['10', '343'], 'leaving the thermal solver brings back the entered speed and sound speed');
  for (const id of ['octree-collision', 'octree-absorbing', 'octree-ramp-steps', 'lattice', 'frames']) ok(!$(id), `${id} is not on the page`);

  // without a model there is nothing to mesh; with one, the request equals the builder's
  ok($('build-mesh').disabled && $('run-solver').disabled, 'Create mesh and Run are disabled without a model');
  studies.state.modelId = '00000000-0000-4000-8000-000000000000'; studies.updateActions();
  let refusal = ''; try { studies.caseSettings(); } catch (error) { refusal = error.message; }
  ok(!$('build-mesh').disabled && $('run-solver').disabled, `a model enables Create mesh; Run still needs a mesh (${refusal || 'settings valid'})`);
  const io = {value: id => $(id).value, checked: id => $(id).checked};
  for (const id of Object.keys(builder.SOLVERS)) {
    choose(id);
    const request = JSON.parse(JSON.stringify(studies.caseSettings()));
    equal(request, JSON.parse(JSON.stringify(builder.buildRequest(io, {modelId: studies.state.modelId, cuts: [], outputs: {}, displayEvery: 100}))), `${id}: the page sends the builder's request`);
    equal([request.solver, request.lattice], [builder.SOLVERS[id].solver, builder.SOLVERS[id].lattice], `${id}: solver and lattice keys`);
  }
  // the run length belongs to step 5: changing it keeps the mesh, changing the cell size does not
  studies.state.jobId = 'job'; studies.state.job = {status: 'mesh_ready'}; studies.state.draftDirty = false;
  $('steps').value = '2400'; $('steps').dispatchEvent(new Event('input'));
  ok(studies.state.jobId === 'job', 'changing the solver steps keeps the mesh');
  $('surface-dx').value = '0.03'; $('surface-dx').dispatchEvent(new Event('input'));
  ok(studies.state.jobId === null, 'changing the cell size asks for a new mesh');

  // a saved case restores its solver and settings; removed settings and solvers do not break it
  studies.hydrateCase({solver: 'q19_octree', lattice: 'd3q19', surface_dx: 0.05, levels: 9, steps: 500, frames: 12, target_vram_gb: 6,
    transform: {scale: [1, 1, 1], rotation_deg: [0, 0, 0], translation: [0, 0, 0]}, domain: {boundary: 'symmetry', symmetry_placement: 'center', symmetry_center_y: true, upstream: 4, downstream: 6, lateral: 3, height: 3},
    flow: {alpha: 2, beta: 0, velocity: 30, density: 1.2, kinematic_viscosity: 1.5e-5, speed_of_sound: 340},
    octree: {band_fine: 10, band: 5, wake_length: 2, wake_level_offset: 1, threads: 4, seal_cavities: true, wall_function: true, hrr_sigma: 0.95, collision: 'rr', absorbing_faces: 'outlet', ramp_steps: 10}, cuts: []});
  equal([$('solver').value, $('levels').value, $('steps').value, $('octree-band-fine').value, $('octree-hrr-sigma').value, $('octree-wall-function').checked, $('octree-seal-cavities').checked, $('symmetry-placement').value],
        ['q19_octree', '9', '500', '10', '0.95', true, true, 'center'], 'a saved octree case restores, ignoring the settings that no longer exist');
  const restored = JSON.parse(JSON.stringify(studies.caseSettings()));
  equal([restored.domain.symmetry_placement, restored.octree.wall_function, 'collision' in restored.octree, 'frames' in restored], ['center', true, false, false], 'the restored case sends current keys only');
  messages.length = 0;
  studies.hydrateCase({solver: 'lbmtau', lattice: 'd3q27', surface_dx: 0.04, levels: 3, steps: 800,
    transform: {scale: [1, 1, 1], rotation_deg: [0, 0, 0], translation: [0, 0, 0]}, domain: {boundary: 'free_air', upstream: 2, downstream: 3, lateral: 2, height: 2},
    flow: {velocity: 20, density: 1.225, kinematic_viscosity: 1.5e-5, speed_of_sound: 343}, cuts: []});
  ok($('solver').value === 'q19_octree' && messages.some(text => /lbmtau/.test(text)), 'a case of a solver no longer offered keeps the current solver and says so');
  equal([$('surface-dx').value, $('boundary').value, $('velocity').value], ['0.04', 'free_air', '20'], 'its other settings still load');
  process.stdout.write(JSON.stringify({checks}));
})().catch(error => { console.error(error); process.exit(1); });
