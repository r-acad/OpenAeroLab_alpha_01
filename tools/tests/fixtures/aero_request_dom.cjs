// Builds the requests Aero Lab sends, from the page's own default control values, with the page's own request
// builder (apps/ludwig_web/static/aero_request.js). No browser, no solver.
//   node aero_request_dom.cjs <repository root> <solver choice> [<JSON of control overrides>]
// Prints {"request": mesh request, "run": run options, "controls_read": ids the builder read}.
const fs = require('node:fs'), path = require('node:path');
const [root, choice, overridesText] = process.argv.slice(2);
if (!root || !choice) { console.error('usage: aero_request_dom.cjs <root> <solver choice> [overrides]'); process.exit(2); }
const page = fs.readFileSync(path.join(root, 'apps/ludwig_web/static/aero_lab.html'), 'utf8');
const builder = require(path.join(root, 'apps/ludwig_web/static/aero_request.js'));

// The page's controls: id -> {value, checked}, as a browser would hold them before any edit.
const controls = new Map();
const attribute = (tag, name) => { const m = tag.match(new RegExp(`\\b${name}="([^"]*)"`)); return m ? m[1] : null; };
for (const m of page.matchAll(/<input\b[^>]*>/g)) {
  const id = attribute(m[0], 'id'); if (!id) continue;
  controls.set(id, {value: attribute(m[0], 'value') ?? '', checked: /\schecked(?=[\s>])/.test(m[0])});
}
for (const m of page.matchAll(/<select\b[^>]*>([\s\S]*?)<\/select>/g)) {
  const id = attribute(m[0].slice(0, m[0].indexOf('>') + 1), 'id'); if (!id) continue;
  const options = [...m[1].matchAll(/<option\b[^>]*>/g)].map(o => ({value: attribute(o[0], 'value') ?? '', selected: /\sselected(?=[\s>])/.test(o[0])}));
  controls.set(id, {value: (options.find(o => o.selected) || options[0] || {value: ''}).value, checked: false, options: options.map(o => o.value)});
}
if (!controls.get('solver')?.options.includes(choice)) { console.error(`the page offers no solver "${choice}"`); process.exit(3); }
controls.get('solver').value = choice;
for (const [id, value] of Object.entries(overridesText ? JSON.parse(overridesText) : {})) {
  if (!controls.has(id)) { console.error(`the page has no control "${id}"`); process.exit(3); }
  if (typeof value === 'boolean') controls.get(id).checked = value; else controls.get(id).value = String(value);
}
// A reader that refuses ids the page does not have: the builder may only read real controls.
const read = new Set();
const control = id => { if (!controls.has(id)) throw new Error(`aero_request.js reads "${id}", which the page does not have`); read.add(id); return controls.get(id); };
const io = {value: id => control(id).value, checked: id => control(id).checked};
const context = {modelId: '00000000-0000-4000-8000-000000000000', cuts: [], outputs: {}, displayEvery: 100};
const request = builder.buildRequest(io, context);
const run = builder.runOptions(io, {...context, studyName: 'Aero study'});
process.stdout.write(JSON.stringify({request, run, controls_read: [...read].sort(), solvers: Object.keys(builder.SOLVERS), page_solvers: controls.get('solver').options}));
