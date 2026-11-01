"use strict";

/* The wire codecs are deliberately local: the page has no CDN or build dependency. */
const utf8 = new TextEncoder();
const fromUtf8 = new TextDecoder();

class ByteWriter {
  constructor() { this.buffer = new ArrayBuffer(1024); this.view = new DataView(this.buffer); this.at = 0; }
  room(n) {
    if (this.at + n <= this.buffer.byteLength) return;
    let size = this.buffer.byteLength;
    while (size < this.at + n) size *= 2;
    const next = new ArrayBuffer(size);
    new Uint8Array(next).set(new Uint8Array(this.buffer, 0, this.at));
    this.buffer = next; this.view = new DataView(next);
  }
  u8(n) { this.room(1); this.view.setUint8(this.at++, n); }
  i8(n) { this.room(1); this.view.setInt8(this.at++, n); }
  u16(n, le = false) { this.room(2); this.view.setUint16(this.at, n, le); this.at += 2; }
  i16(n, le = false) { this.room(2); this.view.setInt16(this.at, n, le); this.at += 2; }
  u32(n, le = false) { this.room(4); this.view.setUint32(this.at, n, le); this.at += 4; }
  i32(n, le = false) { this.room(4); this.view.setInt32(this.at, n, le); this.at += 4; }
  f32(n, le = false) { this.room(4); this.view.setFloat32(this.at, n, le); this.at += 4; }
  f64(n, le = false) { this.room(8); this.view.setFloat64(this.at, n, le); this.at += 8; }
  bytes(bytes) { this.room(bytes.length); new Uint8Array(this.buffer, this.at, bytes.length).set(bytes); this.at += bytes.length; }
  output() { return this.buffer.slice(0, this.at); }
}

function encodeMsgpack(value) {
  const w = new ByteWriter();
  const write = (v) => {
    if (v === null || v === undefined || (typeof v === "number" && !Number.isFinite(v))) { w.u8(0xc0); return; }
    if (v === false || v === true) { w.u8(v ? 0xc3 : 0xc2); return; }
    if (typeof v === "number") {
      if (Number.isInteger(v) && Math.abs(v) <= 2147483647) {
        if (v >= 0 && v < 128) w.u8(v);
        else if (v >= -32 && v < 0) w.u8(v + 256);
        else if (v >= 0 && v <= 255) { w.u8(0xcc); w.u8(v); }
        else if (v >= 0 && v <= 65535) { w.u8(0xcd); w.u16(v); }
        else if (v >= 0) { w.u8(0xce); w.u32(v); }
        else if (v >= -128) { w.u8(0xd0); w.i8(v); }
        else if (v >= -32768) { w.u8(0xd1); w.i16(v); }
        else { w.u8(0xd2); w.i32(v); }
      } else { w.u8(0xcb); w.f64(v); }
      return;
    }
    if (typeof v === "string") {
      const bytes = utf8.encode(v), n = bytes.length;
      if (n < 32) w.u8(0xa0 | n);
      else if (n < 256) { w.u8(0xd9); w.u8(n); }
      else if (n < 65536) { w.u8(0xda); w.u16(n); }
      else { w.u8(0xdb); w.u32(n); }
      w.bytes(bytes); return;
    }
    if (v instanceof Uint8Array) {
      if (v.length < 256) { w.u8(0xc4); w.u8(v.length); }
      else if (v.length < 65536) { w.u8(0xc5); w.u16(v.length); }
      else { w.u8(0xc6); w.u32(v.length); }
      w.bytes(v); return;
    }
    if (Array.isArray(v)) {
      if (v.length < 16) w.u8(0x90 | v.length);
      else if (v.length < 65536) { w.u8(0xdc); w.u16(v.length); }
      else { w.u8(0xdd); w.u32(v.length); }
      v.forEach(write); return;
    }
    if (typeof v === "object") {
      const keys = Object.keys(v).filter(k => v[k] !== undefined);
      if (keys.length < 16) w.u8(0x80 | keys.length);
      else if (keys.length < 65536) { w.u8(0xde); w.u16(keys.length); }
      else { w.u8(0xdf); w.u32(keys.length); }
      keys.forEach(k => { write(k); write(v[k]); }); return;
    }
    throw new Error(`Unsupported MessagePack value: ${typeof v}`);
  };
  write(value); return w.output();
}

function decodeMsgpack(buffer) {
  const d = new DataView(buffer), bytes = new Uint8Array(buffer);
  let at = 0;
  const need = n => { if (at + n > buffer.byteLength) throw new Error("Truncated MessagePack response"); };
  const u8 = () => { need(1); return d.getUint8(at++); };
  const readBytes = n => { need(n); const value = bytes.slice(at, at + n); at += n; return value; };
  const readString = n => fromUtf8.decode(readBytes(n));
  const readArray = n => { const a = []; for (let i = 0; i < n; i++) a.push(read()); return a; };
  const readMap = n => { const o = Object.create(null); for (let i = 0; i < n; i++) { const key = read(); o[String(key)] = read(); } return o; };
  const u16 = () => { need(2); const v = d.getUint16(at); at += 2; return v; };
  const u32 = () => { need(4); const v = d.getUint32(at); at += 4; return v; };
  const read = () => {
    const code = u8();
    if (code <= 0x7f) return code;
    if (code >= 0xe0) return code - 256;
    if ((code & 0xf0) === 0x80) return readMap(code & 0x0f);
    if ((code & 0xf0) === 0x90) return readArray(code & 0x0f);
    if ((code & 0xe0) === 0xa0) return readString(code & 0x1f);
    switch (code) {
      case 0xc0: return null;
      case 0xc2: return false;
      case 0xc3: return true;
      case 0xc4: return readBytes(u8());
      case 0xc5: return readBytes(u16());
      case 0xc6: return readBytes(u32());
      case 0xca: need(4); { const v = d.getFloat32(at); at += 4; return v; }
      case 0xcb: need(8); { const v = d.getFloat64(at); at += 8; return v; }
      case 0xcc: return u8();
      case 0xcd: return u16();
      case 0xce: return u32();
      case 0xcf: need(8); { const v = Number(d.getBigUint64(at)); at += 8; return v; }
      case 0xd0: need(1); return d.getInt8(at++);
      case 0xd1: need(2); { const v = d.getInt16(at); at += 2; return v; }
      case 0xd2: need(4); { const v = d.getInt32(at); at += 4; return v; }
      case 0xd3: need(8); { const v = Number(d.getBigInt64(at)); at += 8; return v; }
      case 0xd9: return readString(u8());
      case 0xda: return readString(u16());
      case 0xdb: return readString(u32());
      case 0xdc: return readArray(u16());
      case 0xdd: return readArray(u32());
      case 0xde: return readMap(u16());
      case 0xdf: return readMap(u32());
      default: throw new Error(`Unsupported MessagePack type 0x${code.toString(16)}`);
    }
  };
  const value = read();
  if (at !== buffer.byteLength) throw new Error("Trailing bytes in MessagePack response");
  return value;
}

function encodeBson(value) {
  if (!value || Array.isArray(value) || typeof value !== "object") throw new Error("BSON root must be a document");
  const w = new ByteWriter();
  const cstring = s => { w.bytes(utf8.encode(s)); w.u8(0); };
  const document = obj => {
    const start = w.at; w.i32(0, true);
    const entries = Array.isArray(obj) ? obj.map((v, i) => [String(i), v]) : Object.entries(obj);
    for (const [key, v] of entries) {
      if (v === undefined) continue;
      if (v === null || (typeof v === "number" && !Number.isFinite(v))) { w.u8(0x0a); cstring(key); }
      else if (typeof v === "boolean") { w.u8(0x08); cstring(key); w.u8(v ? 1 : 0); }
      else if (typeof v === "string") { const b = utf8.encode(v); w.u8(0x02); cstring(key); w.i32(b.length + 1, true); w.bytes(b); w.u8(0); }
      else if (typeof v === "number") {
        if (Number.isInteger(v) && v >= -2147483648 && v <= 2147483647) { w.u8(0x10); cstring(key); w.i32(v, true); }
        else { w.u8(0x01); cstring(key); w.f64(v, true); }
      }
      else if (v instanceof Uint8Array) { w.u8(0x05); cstring(key); w.i32(v.length, true); w.u8(0); w.bytes(v); }
      else if (Array.isArray(v)) { w.u8(0x04); cstring(key); document(v); }
      else if (typeof v === "object") { w.u8(0x03); cstring(key); document(v); }
      else throw new Error(`Unsupported BSON value for ${key}`);
    }
    w.u8(0); w.view.setInt32(start, w.at - start, true);
  };
  document(value); return w.output();
}

function decodeBson(buffer) {
  const d = new DataView(buffer), bytes = new Uint8Array(buffer);
  let at = 0;
  const need = n => { if (at + n > buffer.byteLength) throw new Error("Truncated BSON response"); };
  const i32 = () => { need(4); const v = d.getInt32(at, true); at += 4; return v; };
  const cstring = () => { const start = at; while (at < bytes.length && bytes[at] !== 0) at++; need(1); const v = fromUtf8.decode(bytes.subarray(start, at)); at++; return v; };
  const string = () => { const n = i32(); if (n < 1) throw new Error("Invalid BSON string length"); need(n); const v = fromUtf8.decode(bytes.subarray(at, at + n - 1)); at += n; return v; };
  const document = array => {
    const start = at, length = i32(), end = start + length;
    if (length < 5 || end > bytes.length) throw new Error("Invalid BSON document length");
    const o = array ? [] : Object.create(null);
    while (at < end - 1) {
      const type = bytes[at++], key = cstring(); let value;
      switch (type) {
        case 0x01: need(8); value = d.getFloat64(at, true); at += 8; break;
        case 0x02: value = string(); break;
        case 0x03: value = document(false); break;
        case 0x04: value = document(true); break;
        case 0x05: { const n = i32(); need(n + 1); at++; value = bytes.slice(at, at + n); at += n; break; }
        case 0x08: need(1); value = bytes[at++] !== 0; break;
        case 0x09: need(8); value = new Date(Number(d.getBigInt64(at, true))); at += 8; break;
        case 0x0a: value = null; break;
        case 0x10: value = i32(); break;
        case 0x12: need(8); value = Number(d.getBigInt64(at, true)); at += 8; break;
        default: throw new Error(`Unsupported BSON type 0x${type.toString(16)}`);
      }
      if (array) o[Number(key)] = value; else o[key] = value;
    }
    if (at !== end - 1 || bytes[at++] !== 0) throw new Error("Invalid BSON document terminator");
    return o;
  };
  const value = document(false);
  if (at !== bytes.length) throw new Error("Trailing bytes in BSON response");
  return value;
}

const AirfoilWire = { encodeMsgpack, decodeMsgpack, encodeBson, decodeBson };
if (typeof window !== "undefined") window.AirfoilWire = AirfoilWire;

const $ = id => document.getElementById(id);
const minimumServiceRevision = 9;
const el = Object.fromEntries([
  "cav-fs-controls", "cav-conventional-controls", "cav-speed", "cav-water-depth", "cav-air-height", "cav-fs-duration", "cav-fs-max-steps", "cav-fs-resolution", "cav-fs-vapor", "geometry-axis-label", "cav-fs-formulation", "cav-fs-formulation-note", "history-residual-title", "cfd-te-topology-field", "underwater-mesh-note",
  "cfd-time-mode-field", "cfd-time-mode", "physical-time-controls", "cfd-duration-ctu", "cfd-burnin-ctu", "cfd-block-ctu", "cfd-auto-stop-field", "cfd-sfd-field", "cfd-time-note", "cav-duration-field", "sampling-state", "diagnostics-axis-note", "cav-chord-field", "cav-chord",
  "server-state", "transport", "theme-mode", "source-kind", "naca-picker", "naca-code", "uiuc-picker", "airfoil-search", "airfoil-select", "catalog-status", "airfoil-meta", "airfoil-name", "airfoil-description", "airfoil-link", "show-reference", "curvature-mapping-field", "curvature-mapping-help", "cfd-te-topology",
  "mach", "reynolds", "alpha", "ncrit", "transition-mode", "trip-controls", "trip-upper", "trip-lower", "sigma", "sigma-field", "flow-hint", "backend", "backend-note", "advanced-cfd", "advanced-air", "advanced-cav", "cfd-steps", "cfd-gpu", "cfd-rk", "cfd-rk-field", "gas-physics-fields", "cfd-auto-stop", "cfd-mass-ceiling", "gas-mesh-field", "cfd-mesh", "cfd-resolution", "base-faces-field", "cfd-n-base", "cfd-sfd-storage", "cfd-viscosity-model", "cfd-temperature-field", "cfd-temperature", "cav-warm-steps", "cav-duration", "cav-frames", "cav-channel", "cav-channel-field", "cav-fluid", "cav-domain", "cav-domain-note", "cav-depth", "cav-depth-field", "cav-air-mode", "cav-air-mode-field", "operation-feedback", "operation-label", "curvature-scale", "curvature-scale-field",
  "cav-reference-picker", "cav-reference-select", "cav-reference-meta", "cav-reference-status", "cav-reference-note", "cav-reference-link", "run-status", "job-progress", "gas-mesh-label", "sweep-enabled", "sweep-controls", "alpha-start", "alpha-stop", "alpha-step", "sweep-count", "run-button", "form-error", "job-title", "job-message", "job-percent", "progress-bar", "job-id", "log-state", "log-count", "log-follow", "log-download", "log-view", "log-empty", "log-warning",
  "lowspeed-controls", "lowspeed-resolution", "lowspeed-tu", "lowspeed-mut-ratio", "lowspeed-note", "lowspeed-gpu", "lowspeed-gpu-note", "recent-select", "recent-open", "recent-refresh", "recent-status",
  "gurney-group", "gurney-enabled", "gurney-controls", "gurney-height", "gurney-mode", "gurney-refine", "gurney-note", "gurney-range",
  "streamline-align", "streamline-controls", "streamline-mode", "streamline-admission", "streamline-fallback", "streamline-note", "geometry-streamlines", "field-streamlines",
  "result-backend", "result-alpha", "download-result", "geometry-chart", "geometry-title", "geometry-status", "geometry-display", "native-result-geometry", "curvature-chart", "preview-resolution", "create-mesh", "refine-mesh", "trailing-edge-thickness", "cp-chart", "reference-note", "field-kind", "field-renderer", "field-grid", "field-mesh-overlay", "field-view", "babylon-view", "field-legend", "polar-cl", "polar-cd", "polar-cm", "polar-note", "points-body", "result-notes", "convergence-panel", "convergence-state", "history-residual", "history-cl", "history-cd"
].map(id => [id, $(id)]));
  const state = { serviceRevision: null, airfoils: [], catalogLoaded: false, cavitationReferences: [], cavitationReferencesLoaded: false, cavitationReferencesLoading: null, cavitationReferenceSelection: 0, cavitationReferenceAppliedId: null, job: null, result: null, pollTimer: null, historyTimer: null, pollGeneration: 0, logCursor: 0, historyCursor: 0, logLineCount: 0, jobWire: "msgpack", machEdited: false, liveHistory: Object.create(null), activeCfdAlpha: null, requestedAlpha: null, activeTab: "geometry", meshKeys: "", fullMeshes: new Map(), meshLoading: new Set(), fullFields: new Map(), fieldLoading: new Set(), preview: null, previewGeneration: 0, previewTimer: null, gridVisible: true, meshOverlay: false, showAirfoilName: true, showStreamlines: true };
const colors = { jxfoil: "#087e89", xfoil: "#bf39bd", transonic: "#df713e", cavitation: "#3e69ac", lowspeed: "#2f8f4e", lbm: "#7a4fb5" };
const backendName = { jxfoil: "JXfoil (Julia clone)", xfoil: "XFOIL executable", transonic: "Compressible CFD", cmesh: "Compressible CFD", cavitation: "Species / cavitation CFD", lowspeed: "Low-speed CFD (pressure-based, new)", lbm: "LBM 3D (LUDWIG)" };
function selectedBackends(value) {
  return ({both:["jxfoil","transonic"],cfd_both:["transonic","cavitation"],xfoil_cfd_both:["xfoil","transonic","cavitation"],xfoil_lowspeed:["xfoil","lowspeed"]})[value] || [value];
}
// ---- Gurney flap (23 Oct 2026): a zero-thickness plate perpendicular to the chord at the trailing edge, exact sharp
// root and tip corners (gurney.jl). One side: the plate extends h from that trailing-edge point; symmetric: a T, h/2 each way.
const gurneyRules={heightMin:.001,heightMax:.05,minimumServiceRevision:11,baffleServiceRevision:16,refineServiceRevision:17,refineLevels:["off","standard","fine"]};   // 28 Oct 2026: the low-speed baffle needs api revision 16 (its result shape)
// The Gurney flap is a baffle (an interior wall) of the low-speed CFD's native C+H mesh; the solver's quality gate admits only a
// range of h that depends on the section and the mesh. Create mesh returns it; the page keeps the last one it saw (and the
// last refusal) and shows them beside the height. Nothing here decides: the solver's gate is the authority.
const gurneyDemoRange="On the demo mesh of NACA 0012 the gate admits h/c from 5.75e-4 to 0.0726 on one side and, as the total h, 1.15e-3 to 0.145 symmetric.";
const gurneyUnsteadyCaveat="Steady RANS caveat: a Gurney flap sheds, and the steady march suppresses the shedding, so the plate's back-face suction is a bound, not a time average. Development evidence (apps/airfoil_web/validation/gurney_urans_20261027, a 2 %c flap on NACA 0012, time-accurate at small dt, on the earlier reshaped-mesh plate): the wake sheds and the time-mean C_D rose to 22 % above the steady value (a lower bound). Development, not validated.";
function gurneyRangeKey(value){return value?JSON.stringify([value.airfoil,value.flow.lowspeed_resolution||"demo",value.flow.gurney_mode]):null;}
function gurneyRefusalKey(value){return value?JSON.stringify([value.airfoil,value.flow.lowspeed_resolution||"demo",value.flow.gurney_mode,value.flow.gurney_height]):null;}
// Take the baffle's record (the mesh metadata's gurney_baffle, or a point's baffle) into the page's memory.
function rememberGurneyRange(value,baffle){
  const r=baffle?.range;if(!r||!value||value.flow.gurney_height==null)return;
  state.gurneyRange={key:gurneyRangeKey(value),side:value.flow.gurney_mode,lo:Number(r.lo),hi:Number(r.hi),admitted:Number(r.admitted),scanned:Number(r.scanned),contiguous:r.contiguous!==false};
  state.gurneyRefusal=null;
}
function gurneyEnabled(){return !!el["gurney-enabled"]?.checked;}
// The request's Gurney keys ({} when off). strict: refuse invalid inputs with a message; otherwise return null.
function gurneyFlow(strict=true){
  if(!gurneyEnabled())return {};
  const fail=message=>{if(strict)throw new Error(message);return null;};
  const h=Number(el["gurney-height"].value),mode=el["gurney-mode"].value||"lower";
  if(el["gurney-height"].value.trim()===""||!Number.isFinite(h)||h<gurneyRules.heightMin||h>gurneyRules.heightMax)
    return fail(`Enter a Gurney plate height h/c between ${gurneyRules.heightMin} and ${gurneyRules.heightMax}.`);
  if(!["symmetric","lower","upper"].includes(mode))return fail("Select the Gurney flap side.");
  const refine=el["gurney-refine"]?.value||"standard";
  if(!gurneyRules.refineLevels.includes(refine))return fail("Select the mesh refinement at the flap tip: off, standard or fine.");
  return {gurney_height:h,gurney_mode:mode,gurney_refine:refine};
}
// ---- Streamline-aligned meshes (24 Oct 2026, DEVELOPMENT, NOT VALIDATED): the CFD mesh lines are moved onto JXfoil's wake
// path and stagnation streamline at the analysis angle (streamline_backend.jl). Off: no key is sent and nothing changes.
const streamlineRules={minimumServiceRevision:12};
// The low-speed CFD's opt-in GPU run (24 Oct 2026): the request key lowspeed_gpu, sent only when ticked.
const lowspeedGpuRules={minimumServiceRevision:13};
function lowspeedGpuRefusal(){
  if(!el["lowspeed-gpu"].checked)return "";
  if(state.serviceRevision!=null&&state.serviceRevision<lowspeedGpuRules.minimumServiceRevision)return "Restart run.cmd and use the browser tab it opens: this Julia service predates the low-speed GPU option and would refuse it.";
  return "";
}
function streamlineEnabled(){return !!el["streamline-align"]?.checked;}
function streamlineFlow(){
  if(!streamlineEnabled())return {};
  const flow={cfd_streamline_align:true,cfd_streamline_mode:el["streamline-mode"].value||"viscous",cfd_streamline_admission:el["streamline-admission"].value||"demo"};
  if(el["streamline-fallback"].checked)flow.cfd_streamline_fallback=true;
  return flow;
}
function streamlineRefusal(flow,backends){
  if(!flow.cfd_streamline_align)return "";
  if(state.serviceRevision!=null&&state.serviceRevision<streamlineRules.minimumServiceRevision)return "Restart run.cmd and use the browser tab it opens: this Julia service predates the streamline-aligned mesh and would refuse it.";
  if(backends.includes("cavitation"))return "Streamline alignment is not offered for the Species / cavitation CFD (its driver builds its own mesh).";
  const lowspeed=backends.includes("lowspeed"),compressible=backends.includes("transonic");
  if(!lowspeed&&!compressible)return "Streamline alignment is available for the low-speed CFD (pressure-based, new) with a Gurney flap, and for the compressible CFD on a finite trailing edge (C+H mesh).";
  if(lowspeed&&flow.gurney_height==null&&flow.lowspeed_topology!=="ch_native")return "Streamline alignment runs on the low-speed CFD's C+H mesh: enable a Gurney flap (the clean sharp-trailing-edge C mesh is not aligned yet).";
  if(compressible&&flow.gurney_height!=null)return "Streamline alignment with a Gurney flap is offered on the low-speed CFD only: the Gurney flap is a baffle, which only the low-speed solver has.";
  if(compressible&&(flow.cfd_te_topology||"auto")!=="auto")return "Streamline alignment on the compressible CFD runs on the C+H mesh: select TE topology Auto (a finite trailing edge).";
  return "";
}
function updateStreamlineSetup(backends=selectedBackends(el.backend.value)){
  const on=streamlineEnabled();
  el["streamline-controls"].classList.toggle("hidden",!on);el["streamline-note"].classList.toggle("hidden",!on);
  if(!on)return;
  const refusal=streamlineRefusal({cfd_streamline_align:true,gurney_height:gurneyEnabled()?Number(el["gurney-height"].value):undefined,cfd_te_topology:el["cfd-te-topology"].value,lowspeed_topology:state.savedLowspeedTopology||undefined},backends)||
    (backends.includes("transonic")&&!backends.includes("lowspeed")&&!coupledChForControls()?"Streamline alignment runs on the C+H mesh of a finite trailing edge; this section's trailing edge is sharp, so its CFD mesh is not C+H.":"");
  el["streamline-note"].textContent="DEVELOPMENT, NOT VALIDATED. "+(gurneyEnabled()?
    "Before meshing, JXfoil solves the CLEAN section at the analysis angle (it cannot model the Gurney plate); the CFD mesh lines are then moved onto its wake path (from the native base's midpoint) and its stagnation streamline, and the Gurney baffle is cut AFTER the alignment, on the aligned mesh (it never moves the wall row, so the aligned wake cut stays). ":
    "Before meshing, JXfoil solves the section at the analysis angle; the C+H mesh lines are then moved onto its wake path (from the trailing-edge base midpoint) and its stagnation streamline. ")+
    "Create mesh and Run build the same nodes. One mesh per angle in a sweep. "+
    (el["streamline-admission"].value==="strict"?"Strict admission: the alignment is reduced, or refused, if any mesh quality gate would get worse.":"Demo admission: cells stay convex and the wall spacing unchanged; quality gate changes are reported in the notes and the mesh panel, none refuses.")+(refusal?` Refused here: ${refusal}`:"");
  el["streamline-note"].classList.toggle("warning-text",!!refusal);
}
// The JXfoil streamlines carried by a streamline-aligned mesh (mesh.streamlines): the wake path the mesh follows and the
// stagnation streamline, drawn over the mesh or the flow field.
function drawStreamlines(ctx,plot,mesh){
  const s=mesh?.streamlines;if(!state.showStreamlines||!s)return;
  const line=(p,color,width)=>{const xs=p?.x,ys=p?.y;if(!Array.isArray(xs)||xs.length<2)return;ctx.beginPath();for(let i=0;i<xs.length;i++){const x=plot.x(Number(xs[i])),y=plot.y(Number(ys[i]));if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);}ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();};
  ctx.save();ctx.setLineDash([]);
  line(s.wake,"rgba(220,30,30,0.95)",2.2);line(s.stagnation,"rgba(20,140,60,0.95)",2.2);
  if(Array.isArray(s.stagnation_point)){ctx.fillStyle="rgba(20,140,60,1)";ctx.beginPath();ctx.arc(plot.x(Number(s.stagnation_point[0])),plot.y(Number(s.stagnation_point[1])),4,0,Math.PI*2);ctx.fill();}
  ctx.font="600 11px Segoe UI, Arial, sans-serif";ctx.textBaseline="top";ctx.fillStyle="rgba(150,20,20,1)";
  ctx.fillText(`JXfoil ${String(s.source||"").startsWith("viscous")?"viscous":"inviscid"} wake (red) · stagnation streamline (green) · α ${Number(s.alpha_deg)}° · mesh aligned (${String(s.admission||"")} admission, development)`,46,8);
  ctx.restore();
}
// The geometry key of previews and analyses: unchanged when the flap is off.
function geometrySourceKey(airfoil,flow={}){
  return flow.gurney_height==null ? JSON.stringify([airfoil,flow.cfd_te_thickness]) :
    JSON.stringify([airfoil,flow.cfd_te_thickness,{h:flow.gurney_height,mode:flow.gurney_mode}]);
}
function gurneyModeText(mode){return mode==="symmetric"?"symmetric T (h/2 above, h/2 below)":mode==="upper"?"upper side":"lower (pressure) side";}
function gurneyLabel(flow){
  if(flow?.gurney_height==null)return "";
  const rf=flow.gurney_refine&&flow.gurney_refine!=="off"?`, tip refinement ${flow.gurney_refine}`:"";
  return `Gurney plate (baffle) h/c ${flow.gurney_height}, ${gurneyModeText(flow.gurney_mode)}${rf}`;
}
// Refuse the Gurney flap where the selected backend cannot mesh it (the server refuses the same cases).
function gurneyBackendRefusal(flow,backends){
  if(flow.gurney_height==null)return "";
  if(Number(flow.cfd_te_thickness)>0)return "The Gurney flap and the CFD trailing-edge thickness override both set the trailing edge. Clear the CFD trailing-edge thickness: the plate stands on the section's own trailing edge.";
  const cfd=backends.some(isCfdBackend);
  if(cfd&&(flow.cfd_te_topology||"auto")!=="auto")return "The Gurney flap is meshed in the conforming C+H topology. Select TE topology Auto: Legacy A-plus fans are deprecated for finite trailing edges.";
  if(cfd&&flow.cfd_mesh==="legacy_sharp_cmesh")return "The Gurney flap needs a finite base; the Legacy sharp C mesh closes the trailing edge. Choose the refined shared mesh profile.";
  if(backends.includes("cavitation"))return "The Gurney flap is not available for the Species (cavitation) backends. The flap is modelled as a baffle, a zero-thickness wall inside PBCav's C+H mesh; the water-tunnel driver builds its own mesh and has no interior walls, and the other Species paths do not mesh C+H. Use the low-speed CFD.";
  if(backends.includes("transonic"))return "The Gurney flap is available in the low-speed CFD only (PBCav, version 2). It is modelled as a baffle, a zero-thickness wall along a grid line of the section's own finite-trailing-edge C+H mesh, and the baffle exists only in the low-speed solver; the compressible C+H solver has no interior walls. Choose the low-speed CFD.";
  let needed=backends.includes("lowspeed")?gurneyRules.baffleServiceRevision:gurneyRules.minimumServiceRevision;
  if(backends.includes("lowspeed")&&flow.gurney_refine&&flow.gurney_refine!=="off")needed=Math.max(needed,gurneyRules.refineServiceRevision);
  if(state.serviceRevision!=null&&state.serviceRevision<needed)return "Restart run.cmd and use the browser tab it opens: this Julia service predates the Gurney plate"+(needed>gurneyRules.minimumServiceRevision?" as a baffle (api revision "+needed+")":"")+" and would refuse or ignore it.";
  return "";
}
// CFD backends with live iteration history, spatial fields and the CFD polar/status rules.
const cfdBackends = ["transonic","cavitation","lowspeed","lbm"];
function isCfdBackend(backend) { return cfdBackends.includes(backend); }
// Low-speed CFD (PBCav, version 2): converged only at the solver's own steady stop.
function lowspeedSteady(point) { return point?.backend==="lowspeed" && point?.diagnostics?.solver_outcome==="STEADY"; }
function quantityLabel(point,key) {
  const declared=point?.fields?.quantity_labels?.[key];
  return typeof declared==="string" ? declared : (fieldNames[key] || String(key).replaceAll("_"," "));
}
const fieldNames = { mach: "Mach number", pressure: "Pressure", dynamic_pressure:"Dynamic pressure", hydrostatic_pressure:"Hydrostatic pressure", beta_water:"Water volume fraction", beta_air:"Air volume fraction", beta_vapor:"Vapor volume fraction", density: "Density", speed: "Speed", velocity: "Speed", u: "Streamwise speed", ux: "Streamwise speed", v: "Normal speed", temperature: "Temperature", vorticity: "Vorticity", cp: "Pressure coefficient", alpha: "Vapor fraction", vapor_fraction: "Vapor fraction", vapour_fraction: "Vapor fraction", alpha_v: "Vapor fraction", viscosity: "Viscosity", nut: "Eddy viscosity", turbulence: "Turbulence", makie: "Flow overview" };
const fieldUnits = { mach: "dimensionless", cp: "dimensionless", pressure: "solver units", density: "solver units", speed: "solver units", u: "solver units", v: "solver units", temperature: "solver units", vorticity: "solver units", nut: "solver units" };

function labelBackend(v) { return backendName[v] || String(v || "Unknown backend"); }
function escapeHtml(v) { return String(v ?? "").replace(/[&<>"']/g, s => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[s])); }
function finite(v) { return typeof v === "number" && Number.isFinite(v); }
function numberText(v, digits = 4) { return finite(v) ? Number(v).toFixed(digits) : "—"; }
function displayError(err) { el["form-error"].textContent = err instanceof Error ? err.message : String(err); el["form-error"].classList.remove("hidden"); }
function clearError() { el["form-error"].textContent = ""; el["form-error"].classList.add("hidden"); }

// Operations are independent: a late/cancelled preview cannot clear a new mesh
// operation's progress or re-enable its buttons. Background telemetry is excluded.
const operations = new Map(), operationButtons = new Map();
function beginOperation(key, label, buttons = []) {
  const token = {};
  for (const [button] of buttons) if (!operationButtons.has(button)) operationButtons.set(button, { html: button.innerHTML, disabled: button.disabled });
  operations.set(key, { token, label, buttons });
  updateOperationFeedback();
  return () => { if (operations.get(key)?.token === token) { operations.delete(key); updateOperationFeedback(); } };
}
function updateOperationFeedback() {
  const active = [...operations.values()];
  for (const [button, original] of operationButtons) {
    const busy = active.flatMap(op=>op.buttons).find(([node])=>node===button);
    button.disabled = busy ? true : original.disabled;
    button.innerHTML = busy ? `<span class="operation-spinner" aria-hidden="true"></span>${escapeHtml(busy[1])}` : original.html;
    button.classList.toggle("is-busy", !!busy); button.setAttribute("aria-busy", String(!!busy));
    if (!busy) operationButtons.delete(button);
  }
  const latest = active[active.length-1];
  el["operation-feedback"].classList.toggle("hidden", !latest);
  el["operation-feedback"].setAttribute("aria-busy", String(!!latest));
  el["operation-label"].textContent = latest ? `${latest.label}${active.length>1 ? ` · ${active.length} operations` : ""}` : "";
}
function afterPaint() { return new Promise(resolve => requestAnimationFrame(()=>requestAnimationFrame(resolve))); }

async function request(path, { method = "GET", value, format = "msgpack", signal } = {}) {
  const headers = { Accept: format === "bson" ? "application/bson" : "application/msgpack" };
  if (value !== undefined) headers["Content-Type"] = format === "bson" ? "application/bson" : "application/msgpack";
  let response;
  try {
    response = await fetch(path, { method, headers, body: value === undefined ? undefined : (format === "bson" ? encodeBson(value) : encodeMsgpack(value)), cache: "no-store", signal });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new Error(`Cannot reach Julia at ${location.origin}. Keep the launcher window open and use the new tab opened by run.cmd.`);
  }
  const contentType = response.headers.get("Content-Type") || "";
  const data = await response.arrayBuffer();
  let payload;
  if (contentType.includes("json") && !contentType.includes("bson")) payload = JSON.parse(fromUtf8.decode(data));
  else if (contentType.includes("bson")) payload = decodeBson(data);
  else if (contentType.includes("msgpack")) payload = decodeMsgpack(data);
  else payload = format === "bson" ? decodeBson(data) : decodeMsgpack(data);
  if (!response.ok) throw new Error(payload?.error || payload?.message || `HTTP ${response.status}`);
  return payload;
}

function selectedCatalogItem() { return state.airfoils.find(x => String(x.id) === el["airfoil-select"].value); }
function updateAirfoilMeta() {
  const item = selectedCatalogItem();
  el["airfoil-meta"].classList.toggle("hidden", !item);
  if (!item) return;
  el["airfoil-name"].textContent = item.name || item.id || item.filename || "Airfoil";
  el["airfoil-description"].textContent = item.description || "UIUC coordinate file";
  const url = item.source_url || item.url;
  el["airfoil-link"].classList.toggle("hidden", !url);
  if (url && /^https?:\/\//i.test(url)) el["airfoil-link"].href = url;
  else el["airfoil-link"].removeAttribute("href");
}
function renderAirfoils() {
  const q = el["airfoil-search"].value.trim().toLowerCase();
  const current = el["airfoil-select"].value;
  const filtered = state.airfoils.filter(item => [item.id, item.name, item.description, item.filename].some(v => String(v || "").toLowerCase().includes(q)));
  el["airfoil-select"].replaceChildren(...filtered.map(item => {
    const option = document.createElement("option");
    option.value = String(item.id);
    option.textContent = `${item.name || item.id} · ${item.id}`;
    return option;
  }));
  if (filtered.length) el["airfoil-select"].value = filtered.some(x => String(x.id) === current) ? current : String(filtered[0].id);
  el["catalog-status"].textContent = state.catalogLoaded ? `${filtered.length} of ${state.airfoils.length} airfoils` : "Catalog unavailable";
  updateAirfoilMeta();
}
async function loadAirfoils() {
  const done = beginOperation("catalog", "Loading airfoil catalog…");
  el["catalog-status"].textContent = "Loading UIUC catalog…";
  try {
    const data = await request("/api/airfoils", { format: "bson" });
    state.airfoils = Array.isArray(data) ? data : (data.items || data.airfoils || []);
    state.catalogLoaded = true;
    renderAirfoils();
    scheduleGeometryPreview();
  } catch (err) {
    el["catalog-status"].textContent = `Catalog could not load: ${err.message}`;
  } finally { done(); }
}
function selectedCavitationReference() {
  if(el["source-kind"].value!=="cavitation_reference")return null;
  return state.cavitationReferences.find(item=>item.id===el["cav-reference-select"].value)||null;
}
function boundarySummary(value) {
  if(value==null)return "";
  if(Array.isArray(value))return value.map(boundarySummary).filter(Boolean).join(", ");
  if(typeof value==="object")return Object.entries(value).map(([key,item])=>`${key.replaceAll("_"," ")}: ${boundarySummary(item)}`).join("; ");
  return String(value);
}
function renderCavitationReferenceMeta(item=selectedCavitationReference()) {
  if(!item)return;
  const conditions=item.physical_conditions||item.reference_conditions;
  el["cav-reference-note"].textContent=[item.description,boundarySummary(item.boundary_conditions),
    conditions?`Reference conditions: ${boundarySummary(conditions)}`:"",...(Array.isArray(item.notes)?item.notes:[]),
    item.flow?.cav_domain==="tunnel"?"Tunnel runs use Warmup steps, Duration and Output frames; the gas CFD iteration cap does not set tunnel duration.":"",
    "Preset conditions are editable. Modified inputs are not the original reference case."].filter(Boolean).join("\n");
  const href=safeHref(item.source_url||item.provenance?.source_url||item.provenance?.url);
  el["cav-reference-link"].classList.toggle("hidden",!href);
  if(href)el["cav-reference-link"].href=href;
}
async function loadCavitationReferences() {
  if(state.cavitationReferencesLoaded)return state.cavitationReferences;
  if(state.cavitationReferencesLoading)return state.cavitationReferencesLoading;
  const done=beginOperation("cavitation-references","Loading cavitation reference cases…");
  el["cav-reference-status"].textContent="Loading reference cases…";el["cav-reference-select"].disabled=true;
  state.cavitationReferencesLoading=(async()=>{
    await requireCurrentService();
    const catalog=await request("/api/cavitation-references",{format:el.transport.value||"msgpack"}),items=catalog?.items;
    if(!Array.isArray(items)||!items.length||new Set(items.map(item=>item?.id)).size!==items.length||
        items.some(item=>typeof item?.id!=="string"||!item.id||typeof item.flow!=="object"||Array.isArray(item.flow)||!item.flow||
          !Array.isArray(item.backends)||!item.backends.length))throw new Error("Julia returned an invalid cavitation reference catalog.");
    state.cavitationReferences=items;state.cavitationReferencesLoaded=true;
    const previous=el["cav-reference-select"].value;
    el["cav-reference-select"].replaceChildren(...items.map(item=>{const option=document.createElement("option");option.value=item.id;option.textContent=item.name||item.id;return option;}));
    el["cav-reference-select"].value=items.some(item=>item.id===previous)?previous:items[0].id;
    el["cav-reference-select"].disabled=false;
    return items;
  })();
  try{return await state.cavitationReferencesLoading;}
  catch(err){el["cav-reference-status"].textContent=`Reference cases unavailable: ${err.message}`;throw err;}
  finally{state.cavitationReferencesLoading=null;done();}
}
function applyCavitationReference(item) {
  const backend=["jxfoil","xfoil","transonic","cavitation","both","cfd_both","xfoil_cfd_both"].find(value=>
    selectedBackends(value).length===item.backends.length&&selectedBackends(value).every(key=>item.backends.includes(key)));
  if(!backend)throw new Error("This reference uses a backend selection unavailable in this app.");
  el.backend.value=backend;
  const controls={mach:"mach",reynolds:"reynolds",alpha_deg:"alpha",ncrit:"ncrit",sigma:"sigma",
    cfd_steps:"cfd-steps",cfd_gpu:"cfd-gpu",cfd_auto_stop:"cfd-auto-stop",cfd_mass_ceiling:"cfd-mass-ceiling",
    cfd_mesh:"cfd-mesh",cfd_resolution:"cfd-resolution",cfd_n_base:"cfd-n-base",cfd_te_topology:"cfd-te-topology",
    cfd_te_thickness:"trailing-edge-thickness",cfd_time_mode:"cfd-time-mode",cfd_sfd_storage:"cfd-sfd-storage",
    cfd_viscosity_model:"cfd-viscosity-model",cfd_freestream_temperature_k:"cfd-temperature",cfd_rk_weight_closure:"cfd-rk",
    cav_fluid:"cav-fluid",cav_domain:"cav-domain",cav_air_mode:"cav-air-mode",cav_depth_chords:"cav-depth",cav_chord_m:"cav-chord",
    cav_warm_steps:"cav-warm-steps",cav_duration_ctu:"cav-duration",cav_frames:"cav-frames",cav_channel_height:"cav-channel",
    cfd_duration_ctu:"cfd-duration-ctu",cfd_burnin_ctu:"cfd-burnin-ctu",cfd_block_ctu:"cfd-block-ctu",
    cav_speed_m_s:"cav-speed",cav_water_depth_chords:"cav-water-depth",cav_air_height_chords:"cav-air-height",
    cav_fs_formulation:"cav-fs-formulation",cav_fs_duration_seconds:"cav-fs-duration",cav_fs_max_steps:"cav-fs-max-steps",
    cav_fs_initial_vapor_fraction:"cav-fs-vapor"};
  for(const [key,id]of Object.entries(controls))if(Object.hasOwn(item.flow,key)){
    const value=item.flow[key];if(typeof value==="boolean")el[id].checked=value;else if(value!=null)el[id].value=String(value);
  }
  if(Object.hasOwn(item.flow,"cfd_resolution"))el["preview-resolution"].value=String(item.flow.cfd_resolution);
  if(!Object.hasOwn(item.flow,"cfd_te_thickness"))el["trailing-edge-thickness"].value="";
  el["transition-mode"].value=Number(item.flow.trip_upper||0)>0||Number(item.flow.trip_lower||0)>0||Number(item.flow.cav_trip_x||0)>0?"forced":"default";
  for(const side of ["upper","lower"]){const value=item.flow[`trip_${side}`]??item.flow.cav_trip_x;if(value!=null)el[`trip-${side}`].value=String(value);}
  const sweep=item.sweep||{enabled:false};el["sweep-enabled"].checked=sweep.enabled===true;
  for(const key of ["start","stop","step"])if(Object.hasOwn(sweep,key))el[`alpha-${key}`].value=String(sweep[key]);
  state.machEdited=true;state.cavitationReferenceAppliedId=item.id;el["advanced-cfd"].open=true;
  el["cav-reference-select"].title=item.name||item.id;
  renderCavitationReferenceMeta(item);el["cav-reference-status"].textContent="Reference preset loaded · inputs remain editable";
}
function clearCavitationReferenceGeometry() {
  state.previewAbort?.abort();clearTimeout(state.previewTimer);state.previewGeneration++;
  state.preview=null;state.cavitationReferenceAppliedId=null;spatialMemory.geometry=null;spatialMemory.curvature=null;
  spatialViews.geometry?.destroy();spatialViews.geometry=null;spatialViews.curvature?.destroy();spatialViews.curvature=null;
  putChart("geometry-chart",null,"Loading selected reference geometry…");
  putChart("curvature-chart",null,"Loading selected reference geometry…");
  el["geometry-title"].textContent="Cavitation reference";el["geometry-status"].textContent="Loading selected reference geometry…";
}
async function selectCavitationReference() {
  const generation=++state.cavitationReferenceSelection;clearCavitationReferenceGeometry();
  try{
    if(!state.cavitationReferencesLoaded)await loadCavitationReferences();
    if(generation!==state.cavitationReferenceSelection||el["source-kind"].value!=="cavitation_reference")return;
    const item=selectedCavitationReference();if(!item)throw new Error("Select a cavitation reference case.");
    applyCavitationReference(item);clearError();updateSetup();scheduleGeometryPreview();
  }catch(err){if(generation===state.cavitationReferenceSelection&&el["source-kind"].value==="cavitation_reference")displayError(err);}
}
async function loadHealth() {
  try {
    const data = await request("/api/health", { format: "bson" });
    if (data.ok === false) throw new Error(data.message || "Server unavailable");
    state.serviceRevision=Number(data.api_revision)||0;
    if(data.native_solver_platform_supported===false){
      for(const option of el.backend.options)if(!['jxfoil','xfoil'].includes(option.value)){option.disabled=true;option.title='This CUDA runtime requires Windows or Linux. JXfoil and saved result viewing are available on Mac.';}
      if(el.backend.selectedOptions[0]?.disabled){el.backend.value='jxfoil';updateSetup();}
    }
    window.AirfoilLBM?.onHealth?.(data.lbm);
    window.LudwigOutputControls?.setCapabilities(data.output_capabilities,selectedBackends(el.backend.value));
    if(state.serviceRevision<minimumServiceRevision) {
      el["server-state"].className="status-pill offline";
      el["server-state"].textContent="Julia restart needed";
      el["job-message"].textContent="Restart run.cmd, then use the browser tab it opens (the port may change). This Julia service predates the current mesh, C+H / nonlinear free-surface formulation controls or cavitation reference cases.";
      return;
    }
    el["server-state"].className = "status-pill online";
    el["server-state"].innerHTML = "<i></i> Julia service online";
  } catch (err) {
    state.serviceRevision=0;
    el["server-state"].className = "status-pill offline";
    el["server-state"].innerHTML = "<i></i> Julia service offline";
    el["job-message"].textContent = `Connection check failed: ${err.message}`;
  }
}
async function requireCurrentService() {
  if(state.serviceRevision===null || state.serviceRevision<minimumServiceRevision)await loadHealth();
  if(state.serviceRevision<minimumServiceRevision)throw new Error("Restart run.cmd and use the browser tab it opens before creating a mesh or running an analysis. This Julia service cannot apply the current mesh, C+H / nonlinear free-surface formulation controls or cavitation reference cases; its port may differ from the new service.");
}
function usesCoupledCh({airfoil,flow={},backends=[],preview=null}) {
  if(!backends.includes("transonic")||(flow.cfd_te_topology||"auto")!=="auto")return false;
  if(Number(flow.cfd_te_thickness)>1e-8)return true;
  if(flow.gurney_height!=null)return true;   // the Gurney flap always has a finite base
  if(airfoil?.kind==="naca4"&&/^\d{4}$/.test(airfoil.id)&&Number(airfoil.id.slice(-2))>0)return true;
  if(airfoil?.kind==="oat15a")return true;
  const declared=Number(preview?.geometry_metadata?.cfd_te_gap);
  if(finite(declared))return declared>1e-8;
  const geometry=preview?.geometry,x=geometry?.x,y=geometry?.y;
  return Array.isArray(x)&&Array.isArray(y)&&x.length>2&&x.length===y.length&&Math.hypot(x[0]-x.at(-1),y[0]-y.at(-1))>1e-8;
}
function coupledChForControls() {
  const kind=el["source-kind"].value,id=kind==="naca4"?el["naca-code"].value.trim():kind==="uiuc"?el["airfoil-select"].value:kind==="cavitation_reference"?el["cav-reference-select"].value:kind;
  const airfoil={kind,id},thickness=el["trailing-edge-thickness"].value.trim()===""?undefined:Number(el["trailing-edge-thickness"].value);
  const gurney=gurneyFlow(false)||{},sourceKey=geometrySourceKey(airfoil,{cfd_te_thickness:thickness,...gurney});
  return usesCoupledCh({airfoil,flow:{cfd_te_topology:el["cfd-te-topology"].value||"auto",cfd_te_thickness:thickness,...gurney},
    backends:selectedBackends(el.backend.value),preview:state.preview?.sourceKey===sourceKey?state.preview:null});
}
function updateSetup() {
  const sourceKind = el["source-kind"].value;
  const isUiuc = sourceKind === "uiuc";
  const isCavitationReference=sourceKind==="cavitation_reference";
  el["naca-picker"].classList.toggle("hidden", sourceKind !== "naca4");
  el["uiuc-picker"].classList.toggle("hidden", !isUiuc);
  el["cav-reference-picker"].classList.toggle("hidden",!isCavitationReference);
  el["cav-reference-meta"].classList.toggle("hidden",!isCavitationReference);
  if (isUiuc && !state.catalogLoaded) loadAirfoils();
  const backends=selectedBackends(el.backend.value),gas=backends.includes("transonic"),cav=backends.includes("cavitation"),cavOnly=cav&&!gas;
  window.LudwigOutputControls?.setCapabilities(null,backends);
  const coupledCh=coupledChForControls();
  const chMeshConflict=coupledCh&&el["cfd-mesh"].value==="legacy_sharp_cmesh";
  if(coupledCh){el["cfd-time-mode"].value="steady";el["cfd-viscosity-model"].value="constant_kinematic";el["cfd-rk"].checked=true;}
  el["cfd-time-mode"].disabled=coupledCh;el["cfd-viscosity-model"].disabled=coupledCh;
  el["cfd-time-mode"].title=coupledCh?"The coupled C+H driver currently supports steady pseudo-time only.":"";
  el["cfd-viscosity-model"].title=coupledCh?"The coupled C+H driver currently uses constant kinematic viscosity with fully turbulent SST.":"";
  const comparison=el.backend.value==="xfoil_cfd_both";
  el["cav-fluid"].disabled=comparison;el["cav-domain"].disabled=comparison;
  const cavDomain=el["cav-domain"].value||"tunnel",underwater=cav&&cavDomain==="underwater";
  if(underwater)el["cav-fluid"].value="water";
  el["cav-fluid"].disabled=comparison||underwater;
  el.mach.disabled=underwater;el.ncrit.disabled=underwater;
  el["cfd-te-topology"].disabled=underwater;el["trailing-edge-thickness"].disabled=underwater;
  el["cfd-te-topology-field"].classList.toggle("hidden",underwater);el["underwater-mesh-note"].classList.toggle("hidden",!underwater);
  el["trailing-edge-thickness"].title=underwater?"Nonlinear free surface uses the native trailing edge on an annular development mesh. The override is sent as zero; this saved gas value is restored when returning to a gas case.":"Optional CFD trailing-edge thickness override. Leave blank or zero for native geometry.";
  el.reynolds.title=underwater?"Informational only: the experimental nonlinear free-surface worker is inviscid.":"";
  const tripCapable=!underwater&&((gas&&!coupledCh)||cav||backends.includes("xfoil"));
  if(!tripCapable)el["transition-mode"].value="default";
  el["transition-mode"].disabled=comparison||!tripCapable;
  el["trip-controls"].classList.toggle("hidden",el["transition-mode"].value!=="forced");
  const cavAir=cav&&el["cav-fluid"].value==="air";
  const sharedMesh=cav&&cavDomain==="free_fluid";
  const speciesCav=cav&&(cavAir||sharedMesh);
  const physical=(gas&&el["cfd-time-mode"].value==="physical") || (speciesCav&&(!cavAir||el["cav-air-mode"].value==="physical"));
  el["cfd-time-mode-field"].classList.toggle("hidden",!gas);
  el["physical-time-controls"].classList.toggle("hidden",!physical);
  el["cfd-auto-stop-field"].classList.toggle("hidden",physical);
  el["cfd-sfd-field"].classList.toggle("hidden",physical||coupledCh);
  el["cav-duration-field"].classList.toggle("hidden",physical||underwater);
  el["cfd-time-note"].textContent=physical ? "CTU = time × speed / chord. Physical time uses one global timestep without SFD. The iteration limit can end a run before the requested duration; adequate sampling also requires ten blocks after burn-in. Timestep and stall validation remain separate." : "Smart stop checks residuals and load history over two windows, then waits 20,000 more iterations.";
  if(coupledCh)el["cfd-time-note"].textContent="C+H compressible CFD: steady pseudo-time, conservative SSPRK, constant kinematic viscosity and fully turbulent SST. SFD is disabled. Smart stopping checks native residuals, mass balance and load stability; a stopped diagnostic run is not proof of convergence.";
  const cfd = cav || gas;
  el["sigma-field"].classList.toggle("hidden", !cav||cavAir||underwater);
  el.sigma.disabled=underwater;
  el["cav-fs-controls"].classList.toggle("hidden",!underwater);
  el["cav-fs-formulation-note"].textContent=(el["cav-fs-formulation"].value||"incompressible_projection")==="incompressible_projection"?"Incompressible projection transports water, vapor nuclei and air on a convective/gravity timestep. Mach is unused. Inviscid startup only; coarse-grid pressure oscillations remain unresolved.":"Barotropic explicit transports water, vapor nuclei and air with gravity on its acoustic timestep. The short default is an acoustic startup diagnostic.";
  el["cav-conventional-controls"].classList.toggle("hidden",underwater);
  el["cav-channel-field"].classList.toggle("hidden",cavDomain!=="tunnel");
  el["cav-depth-field"].classList.toggle("hidden",cavDomain!=="underwater");
  el["cav-chord-field"].classList.toggle("hidden",cavDomain!=="underwater");
  el["cav-air-mode-field"].classList.toggle("hidden",!cavAir);
  el["cav-domain-note"].textContent=cavDomain==="free_fluid" ? "Far boundaries use the shared airfoil mesh. This domain has no air/water free surface; select Nonlinear free surface for a submerged airfoil." : underwater ? "Nonlinear free surface uses actual physical seconds, a submerged airfoil and gravity. The short default is an inviscid startup experiment; inspect reached time and conservation before interpreting fields." : "Tunnel walls reproduce the cavitation validation setup. There is no air/water free surface in this domain.";
  el["advanced-cav"].classList.toggle("hidden", !cav);
  el["advanced-cfd"].classList.toggle("hidden", !cfd);
  el["advanced-air"].classList.toggle("hidden",cavOnly&&!speciesCav);
  el["gas-physics-fields"].classList.toggle("hidden",!gas);
  el["cfd-rk-field"].classList.toggle("hidden",!gas||physical);
  el["cfd-rk"].disabled = coupledCh||!el["cfd-gpu"].checked;
  el["cfd-rk"].title=coupledCh?"The native coupled C+H driver always uses conservative SSPRK weights.":"";
  el["cfd-temperature-field"].classList.toggle("hidden", !gas || el["cfd-viscosity-model"].value !== "sutherland");
  el["gas-mesh-field"].classList.toggle("hidden", (cavOnly&&!sharedMesh)||(coupledCh&&!cav&&!chMeshConflict));
  el["gas-mesh-label"].textContent=coupledCh&&cav?"Species mesh profile (legacy spacing)":"CFD mesh profile";
  el["gas-mesh-field"].title=chMeshConflict?"Auto C+H for a finite trailing edge conflicts with Legacy sharp C mesh. Choose Refined shared mesh or explicitly change TE topology to Legacy A-plus.":"";
  const manualBase=cfd&&!underwater&&(el["cfd-te-topology"].value||"auto")==="aplus"&&
    !(cavOnly&&!speciesCav)&&el["cfd-mesh"].value!=="legacy_sharp_cmesh";
  el["base-faces-field"].classList.toggle("hidden",!manualBase);
  el["cfd-n-base"].disabled=!manualBase;
  el["flow-hint"].textContent = cav ? (cavAir ? "Isothermal air runs at low Mach with cavitation disabled. Ncrit is unused." : "Water/vapor CFD uses the selected domain. Mach sets reference speed; Ncrit is unused.") : sourceKind === "rae2822" ? "RAE Case 9: M 0.734, Re 6.5 million, α 2.79°, fixed trips x/c 0.03 on both sides." : sourceKind === "oat15a" ? "ONERA OAT15A: M 0.73, Re 3 million, fixed trips x/c 0.07. The 3.5° case is unsteady." : "Ncrit applies to the JXfoil boundary layer calculation.";
  const positiveMach = Number(el.mach.value) > 0;
  el["backend-note"].textContent = cav ? (cavAir ? "Species CFD with an isothermal air equation of state; compare at low Mach. Stall parity requires validation." : "Water/vapor CFD supports Mach 0.02–0.30 and reports available vapor and flow quantities.") : el.backend.value === "jxfoil" ? (positiveMach ? "The current JXfoil clone supports Mach 0 only. Select compressible CFD for nonzero Mach." : "JXfoil provides section coefficients and surface pressure. Flow fields require CFD.") : el.backend.value === "both" ? "JXfoil runs as an incompressible reference at Mach 0; CFD uses the requested Mach (0.02–0.98). Gas CFD is experimental: inspect residuals and Cp before using coefficients." : "Compressible CFD supports Mach 0.02–0.98 and reports spatial flow quantities. This generic-airfoil path is experimental: inspect residuals and Cp before using coefficients.";
  if(el.backend.value==="xfoil")el["backend-note"].textContent="Runs the actual XFOIL executable at the requested subsonic Mach, Re, Ncrit and transition locations. It provides Cp and coefficients; volume fields require CFD. Failed angles remain explicit.";
  if(el.backend.value==="cfd_both")el["backend-note"].textContent=`Runs both CFD backends with the selected Species fluid (${el["cav-fluid"].value}) and domain (${cavDomain}). Different fluids or domains are separate analyses, not a matched air parity comparison.`;
  if(comparison)el["backend-note"].textContent="Air comparison visibly selects Species air + free-fluid and prescribed trips. XFOIL and both CFD use the requested Mach (0.02–0.30). Energy, transition and boundary models differ; inspect convergence before comparing. A CFD trailing-edge override changes only CFD geometry.";
  if(backends.includes("xfoil"))el["flow-hint"].textContent="Ncrit controls the actual XFOIL boundary layer. Prescribed trip coordinates are passed to all selected solvers; their transition models differ. XFOIL uses its subsonic compressibility correction, not resolved shock CFD.";
  if(cavAir)el["backend-note"].textContent+=" Species air uses fully turbulent SST with an optional symmetric k-floor trip. It does not implement XFOIL natural transition or independent upper/lower trip locations.";
  if(backends.includes("jxfoil")&&el["transition-mode"].value==="forced")el["backend-note"].textContent+=" Prescribed trip controls apply to CFD; the JXfoil clone retains its native transition model.";
  if(coupledCh){
    el["backend-note"].textContent+=" The finite trailing edge uses the connected C+H gas driver. This experimental path uses fully turbulent SST; coefficients remain unqualified until convergence and parity checks pass.";
    el["flow-hint"].textContent=backends.includes("xfoil")||cav?"Trip controls apply to XFOIL / Species; C+H compressible CFD uses fully turbulent SST without prescribed trips.":"C+H compressible CFD uses fully turbulent SST without prescribed trips. Ncrit does not control this CFD model.";
    if(chMeshConflict)el["backend-note"].textContent+=" Mesh selection conflict: choose Refined shared mesh, or explicitly select Legacy A-plus TE topology. Auto C+H cannot use the Legacy sharp C geometry derivative.";
  }
  if(underwater){el["flow-hint"].textContent="Physical speed and chord set the nonlinear free-surface case. Mach, Ncrit and cavitation number are unused. Reynolds number is informational: this experimental worker has no viscosity.";el["backend-note"].textContent="Experimental nonlinear water/air/vapor interface with gravity and an inviscid airfoil wall. No SST, energy equation or phase change. Instantaneous startup fields and loads are diagnostic only; cavitation, drag and stall parity are unqualified.";}
  if(isCavitationReference&&cav&&!underwater)el["flow-hint"].textContent="The preset Mach is the numerical carrier value. Inspect Case and boundary conditions for the physical experiment and model differences. Inputs remain editable.";
  const sweep = el["sweep-enabled"].checked;
  el["sweep-controls"].classList.toggle("hidden", !sweep);
  el["sweep-count"].classList.toggle("hidden", !sweep);
  if (sweep) {
    const start = Number(el["alpha-start"].value), stop = Number(el["alpha-stop"].value), step = Number(el["alpha-step"].value);
    const ratio = (stop - start) / step;
    const count = Number.isFinite(ratio) && ratio > 0 ? Math.floor(ratio + 1e-8) + 1 : 0;
    el["sweep-count"].textContent = count ? `${count} requested angles · limit ${cfd || backends.includes("lowspeed") ? 31 : 121}` : "Set a nonzero step pointing toward the end angle.";
    if (backends.includes("lowspeed") && count) el["sweep-count"].textContent += ` · low-speed CFD about ${Math.round(count*20)} min`;
  }
  updateLowspeedSetup(backends);
  updateGurneySetup(backends,underwater);
  updateStreamlineSetup(backends);
  window.AirfoilLBM?.updateSetup(backends);   // LBM 3D (LUDWIG, lbm.js): its controls, notes and estimate
}
function updateGurneySetup(backends=selectedBackends(el.backend.value),underwater=false) {
  const on=gurneyEnabled();
  el["gurney-controls"].classList.toggle("hidden",!on);el["gurney-note"].classList.toggle("hidden",!on);el["gurney-range"].classList.toggle("hidden",!on);
  el["gurney-enabled"].disabled=underwater;
  if(underwater&&on){el["gurney-note"].textContent="The nonlinear free surface uses the native trailing edge; the Gurney flap is refused there.";el["gurney-range"].classList.add("hidden");return;}
  if(!on)return;
  const h=Number(el["gurney-height"].value),mode=el["gurney-mode"].value||"lower",valid=Number.isFinite(h)&&h>=gurneyRules.heightMin&&h<=gurneyRules.heightMax;
  const where=backends.includes("lowspeed")?" The low-speed CFD (PBCav C+H free-air mesh, snapshot S1030_FLAPREF) models it as a baffle: a wall inside the C+H mesh along the grid line from the trailing-edge corner, straightened onto the chord's normal with its tip at exactly h. Standard and Fine redistribute mesh rows around the flap and wake; Off uses the section's own rows."
    :"";
  const panel=backends.some(b=>b==="xfoil"||b==="jxfoil")?" XFOIL/JXfoil cannot model the plate: they run the clean section, labelled as the no-flap reference.":"";
  const refusal=gurneyBackendRefusal({cfd_te_thickness:el["trailing-edge-thickness"].value.trim()===""?undefined:Number(el["trailing-edge-thickness"].value),
    cfd_te_topology:el["cfd-te-topology"].value||"auto",cfd_mesh:el["cfd-mesh"].value,gurney_height:valid?h:0},backends);
  el["gurney-note"].textContent=`Plate height h/c, perpendicular to the chord at the TE, zero thickness, sharp: `+
    (mode==="symmetric"?"a T from h/2 above the upper to h/2 below the lower trailing-edge point":mode==="upper"?"from the upper trailing-edge point up h":"from the lower trailing-edge point down h")+
    `. The airfoil is unchanged; its native base stays.`+where+panel+" "+gurneyUnsteadyCaveat+(refusal?` Refused here: ${refusal}`:"");
  el["gurney-note"].classList.toggle("warning-text",!!refusal);
  // the solver gate's admitted range of h for this section, mesh and side (from Create mesh), and its last refusal
  const value=previewCase(false),range=state.gurneyRange,key=gurneyRangeKey(value);
  const refused=state.gurneyRefusal&&state.gurneyRefusal.key===gurneyRefusalKey(value)?state.gurneyRefusal.message:"";
  let text,warn=false;
  if(refused){text=refused;warn=true;}
  else if(range&&range.key===key&&Number.isFinite(range.lo)&&Number.isFinite(range.hi)){
    const sideText=mode==="symmetric"?"total h/c":"h/c",inside=valid&&h>=range.lo*(1-1e-9)&&h<=range.hi*(1+1e-9);
    text=`The solver's gate admits ${sideText} from ${range.lo.toPrecision(3)} to ${range.hi.toPrecision(3)} on this mesh (${range.admitted} of ${range.scanned} heights of the corner line${range.contiguous?"":", not contiguous"}), from the last Create mesh.`+
      (valid&&!inside?` The entered h/c ${h} is outside it: the solver will refuse it.`:"");
    warn=valid&&!inside;
  }else if(range&&range.key===key)text="No height of this mesh's corner line passes the solver's gate.";
  else text=`The solver's quality gate admits only a range of h that depends on the section and the mesh; Create mesh returns it. ${gurneyDemoRange}`;
  el["gurney-range"].textContent=text;el["gurney-range"].classList.toggle("warning-text",warn);
}
// Low-speed CFD (PBCav, version 2): incompressible, fully turbulent SST, CPU only. Mach is not one of its inputs.
function updateLowspeedSetup(backends=selectedBackends(el.backend.value)) {
  const lowspeed=backends.includes("lowspeed"),alone=lowspeed&&backends.length===1;
  el["lowspeed-controls"].classList.toggle("hidden",!lowspeed);
  el["lowspeed-gpu-note"].classList.toggle("hidden",!(lowspeed&&el["lowspeed-gpu"].checked));
  el.mach.title=alone?"Ignored: the low-speed CFD is incompressible.":lowspeed?"XFOIL uses this Mach (0 = incompressible); the low-speed CFD ignores it.":"";
  if(!lowspeed)return;
  if(alone)el.mach.disabled=true;
  const gpu=el["lowspeed-gpu"].checked,where=gpu?"on the GPU (CUDA, opt-in) once the shared GPU lock is free":"on 8 CPU threads";
  if(gurneyEnabled()){
    // Gurney flap: the low-speed CFD runs PBCav's native C+H mesh with the plate as a baffle; XFOIL runs the clean section (no-flap reference).
    el["backend-note"].textContent=(alone?"Low-speed CFD":"XFOIL + low-speed CFD: XFOIL runs first, on the clean section (no-flap reference: XFOIL cannot model the plate); then the low-speed CFD")+
      " on PBCav's C+H free-air mesh (conforming C block + H block behind the finite base) with the Gurney plate as a baffle (snapshot S1030_FLAPREF). The selected flap-tip mesh refinement also redistributes rows across the flap-height wake; Off uses the section's own mesh. Fully turbulent SST, ramp start, "+(gpu?where:"about 15-60 minutes per angle "+where)+". The solver refuses a plate height outside the mesh's admitted range, and says which.";
    el["flow-hint"].textContent=alone?"Reynolds and angle set the low-speed CFD case. Mach, Ncrit and trips are not used: the CFD is incompressible and fully turbulent.":"Ncrit and the prescribed trips apply to XFOIL only; the low-speed CFD is fully turbulent and ignores Mach.";
    return;
  }
  el["backend-note"].textContent=alone ?
    "Low-speed CFD: the new incompressible pressure-based solver (version 2, in development, not yet validated). Fully turbulent k-omega SST in free air on the demo mesh, "+(gpu?where:"about 15-25 minutes per angle "+where+" (alpha 0 longer)")+"; the live charts show CL, CD and the solver's steady residual. Mach is ignored." :
    "XFOIL + low-speed CFD: XFOIL runs first, on the same closed-trailing-edge section as the low-speed CFD (like for like), at the entered Mach (0 = incompressible) with the transition setting below; the low-speed CFD (version 2, not yet validated) then runs every angle, fully turbulent, about 15-25 minutes per angle. Compare against XFOIL with fixed transition near the leading edge.";
  el["flow-hint"].textContent=alone ?
    "Reynolds and angle set the low-speed CFD case. Mach, Ncrit and trips are not used: the CFD is incompressible and fully turbulent." :
    "Ncrit and the prescribed trips apply to XFOIL only; the low-speed CFD is fully turbulent and ignores Mach.";
}
function inputNumber(id, label, min = -Infinity, max = Infinity) {
  const value = Number(el[id].value);
  if (el[id].value.trim() === "" || !Number.isFinite(value) || value < min || value > max) throw new Error(`Enter a valid ${label}.`);
  return value;
}
function applyReferenceTripControls(flow) {
  if(el["source-kind"].value!=="cavitation_reference"||flow.cav_fluid!=="water"||flow.cav_domain!=="tunnel")return;
  if(el["transition-mode"].value==="forced"){
    const upper=inputNumber("trip-upper","upper transition location",0,1),lower=inputNumber("trip-lower","lower transition location",0,1);
    if(upper!==lower)throw new Error("Reference tunnel Species uses one symmetric turbulence-floor trip. Set the same upper and lower trip location.");
    flow.trip_upper=upper;flow.trip_lower=lower;flow.cav_trip_x=upper;
  }else{flow.trip_upper=0;flow.trip_lower=0;flow.cav_trip_x=0;}
}
function collectCase() {
  const kind = el["source-kind"].value;
  const preset=selectedCavitationReference();
  if(kind==="cavitation_reference"&&(!preset||state.cavitationReferenceAppliedId!==preset.id))throw new Error("Wait for the selected cavitation reference preset to load.");
  const selected=el.backend.value,backends=selectedBackends(selected),hasCfd=backends.some(isCfdBackend);
  const underwater=backends.includes("cavitation")&&el["cav-domain"].value==="underwater";
  if(underwater&&(backends.length!==1||backends[0]!=="cavitation"))throw new Error("Nonlinear free surface currently requires the standalone Species / cavitation backend. Other backends use different flow conditions.");
  const id = kind === "naca4" ? el["naca-code"].value.trim() : kind === "uiuc" ? el["airfoil-select"].value : kind === "cavitation_reference" ? el["cav-reference-select"].value : kind;
  if (kind === "naca4" && !/^\d{4}$/.test(id)) throw new Error("Enter a four digit NACA designation.");
  if (kind === "uiuc" && !id) throw new Error("Select an airfoil from the UIUC catalog.");
  const flow = {
    ...(preset?.flow||{}),
    mach: underwater ? undefined : inputNumber("mach", "Mach number", 0, 5),
    reynolds: inputNumber("reynolds", "Reynolds number", 1),
    alpha_deg: inputNumber("alpha", "angle of attack", -90, 90),
    ncrit: underwater ? undefined : inputNumber("ncrit", "Ncrit", 0.01, 100)
  };
  if(underwater)flow.cfd_te_thickness=0;
  else if (el["trailing-edge-thickness"].value.trim() !== "") flow.cfd_te_thickness = inputNumber("trailing-edge-thickness", "trailing-edge thickness", 0, 0.05);
  if(hasCfd&&!underwater)flow.cfd_te_topology=el["cfd-te-topology"].value||"auto";
  if(!underwater)Object.assign(flow,gurneyFlow(true));
  if(!underwater)Object.assign(flow,streamlineFlow());
  else if(gurneyEnabled())throw new Error("The Gurney flap is not available for the nonlinear free surface (an annular development mesh with the native trailing edge).");
  const coupledCh=usesCoupledCh({airfoil:{kind,id},flow,backends,
    preview:state.preview?.sourceKey===geometrySourceKey({kind,id},flow)?state.preview:null});
  const tripCapable=!coupledCh||backends.includes("xfoil")||backends.includes("cavitation");
  // The published pressure cases used fixed transition on both surfaces.
  // Keep the request and the reference eligibility check on the same setup.
  if (!underwater&&tripCapable&&(kind === "rae2822" || kind === "oat15a")) {
    flow.trip_upper = kind === "rae2822" ? 0.03 : 0.07;
    flow.trip_lower = flow.trip_upper;
  }
  if(!underwater&&tripCapable&&el["transition-mode"].value==="forced") {
    flow.trip_upper=inputNumber("trip-upper","upper transition location",0,1);
    flow.trip_lower=inputNumber("trip-lower","lower transition location",0,1);
  }
  if(backends.includes("xfoil")&&flow.mach>=1)throw new Error("XFOIL executable requires a subsonic Mach below1; it does not resolve shocks.");
  if (backends.includes("transonic") && (flow.mach < 0.02 || flow.mach > 0.98)) throw new Error("Compressible CFD currently supports Mach 0.02–0.98.");
  if (!underwater&&backends.includes("cavitation") && (flow.mach < 0.02 || flow.mach > 0.3)) throw new Error("Species / cavitation CFD currently supports Mach 0.02–0.30.");
  if (!underwater&&hasCfd && (flow.reynolds < 1000 || flow.reynolds > 1e9)) throw new Error("CFD currently supports Reynolds numbers from 1,000 to 1 billion.");
  if (backends.includes("transonic")) {
    flow.cfd_time_mode=coupledCh?"steady":el["cfd-time-mode"].value||"steady";
    const physical=flow.cfd_time_mode==="physical";
    flow.cfd_steps = inputNumber("cfd-steps", "CFD step count", 100);
    flow.cfd_gpu = el["cfd-gpu"].checked;
    flow.cfd_rk_weight_closure = coupledCh||(!physical && flow.cfd_gpu && el["cfd-rk"].checked);
    flow.cfd_auto_stop = !physical && el["cfd-auto-stop"].checked;
    flow.cfd_mass_ceiling = inputNumber("cfd-mass-ceiling", "scaled mass residual ceiling", 1e-6, 1000);
    if(!coupledCh||backends.includes("cavitation")||el["cfd-mesh"].value==="legacy_sharp_cmesh")flow.cfd_mesh = el["cfd-mesh"].value;
    flow.cfd_resolution = el["cfd-resolution"].value;
    if(coupledCh){flow.cfd_sfd_enabled=false;flow.cfd_transition_model="fully_turbulent";}
    else flow.cfd_sfd_storage = physical ? "float32" : el["cfd-sfd-storage"].value;
    flow.cfd_viscosity_model = coupledCh?"constant_kinematic":el["cfd-viscosity-model"].value;
    if (flow.cfd_viscosity_model === "sutherland") flow.cfd_freestream_temperature_k = inputNumber("cfd-temperature", "freestream temperature", 100, 2000);
    if (!physical && !flow.cfd_gpu && (flow.cfd_sfd_storage === "float64" || flow.cfd_viscosity_model !== "constant_kinematic")) throw new Error("Float64 SFD and dynamic air viscosity require the GPU solver.");
    if (!coupledCh&&flow.cfd_mesh !== "legacy_sharp_cmesh") {
      flow.cfd_n_base = inputNumber("cfd-n-base", "Trailing-edge base faces per side", 1, 8);
      if (!Number.isInteger(flow.cfd_n_base)) throw new Error("Trailing-edge base faces per side must be a whole number.");
    }
  }
  if (backends.includes("cavitation")) {
    flow.cav_fluid=el["cav-fluid"].value||"water";
    flow.cav_domain=el["cav-domain"].value||"tunnel";
    if(underwater){
      if(flow.cav_fluid!=="water")throw new Error("Nonlinear free surface requires the water/air/vapor fluid model.");
      Object.assign(flow,underwaterControls());
      flow.cfd_gpu=el["cfd-gpu"].checked;flow.cfd_auto_stop=false;
    }else{
    if(flow.cav_fluid==="air"&&el["transition-mode"].value==="forced"&&flow.trip_upper!==flow.trip_lower)throw new Error("Species air supports only a symmetric SST k-floor trip; use the same upper/lower location. Natural XFOIL transition is a different model.");
    if(selected==="xfoil_cfd_both"&&(flow.cav_fluid!=="air"||flow.cav_domain!=="free_fluid"))throw new Error("Air comparison requires the visible Species fluid=air and domain=free fluid controls.");
    flow.cfd_steps=inputNumber("cfd-steps","CFD step count",100);
    flow.cfd_gpu=el["cfd-gpu"].checked;
    flow.cfd_auto_stop=el["cfd-auto-stop"].checked;
    flow.cfd_mass_ceiling=inputNumber("cfd-mass-ceiling", "scaled mass residual ceiling",1e-6,1000);
    if(flow.cav_fluid==="air")flow.cav_air_mode=el["cav-air-mode"].value||"steady";
    if(flow.cav_fluid==="water")flow.sigma = inputNumber("sigma", "cavitation number", 0.01, 20);
    flow.cav_warm_steps = inputNumber("cav-warm-steps", "cavitation warmup steps", 0);
    flow.cav_duration_ctu = inputNumber("cav-duration", "cavitation duration", 0.001);
    flow.cav_frames = inputNumber("cav-frames", "cavitation output frames", 2, 1000);
    if(flow.cav_domain==="tunnel")flow.cav_channel_height = inputNumber("cav-channel", "channel height", 0.2);
    if(flow.cav_domain==="free_fluid"||flow.cav_fluid==="air") {
      flow.cfd_mesh=el["cfd-mesh"].value;flow.cfd_resolution=el["cfd-resolution"].value;
      if(flow.cav_domain==="free_fluid"&&flow.cfd_mesh==="legacy_sharp_cmesh")throw new Error("Free-fluid Species CFD requires a shared A-plus mesh.");
      flow.cfd_n_base=inputNumber("cfd-n-base", "Trailing-edge base faces per side",1,8);
      if(!Number.isInteger(flow.cfd_n_base))throw new Error("Trailing-edge base faces per side must be a whole number.");
    }
    }
  }
  if (backends.includes("lowspeed")) {
    if (Number(flow.cfd_te_thickness) > 0) throw new Error(flow.gurney_height!=null ? "The Gurney flap and the CFD trailing-edge thickness override both set the trailing edge. Clear the CFD trailing-edge thickness." : "Low-speed CFD uses a sharp-trailing-edge C-mesh. Clear the CFD trailing-edge thickness override to run it.");
    if (flow.reynolds < 1e4 || flow.reynolds > 1e8) throw new Error("Low-speed CFD supports Reynolds numbers from 10,000 to 100 million.");
    flow.lowspeed_resolution = el["lowspeed-resolution"].value || "demo";
    flow.lowspeed_tu = inputNumber("lowspeed-tu", "freestream turbulence intensity (%)", 0.0001, 19) / 100;
    flow.lowspeed_mut_ratio = inputNumber("lowspeed-mut-ratio", "freestream eddy-viscosity ratio", 0.0001, 10000);
    if (el["lowspeed-gpu"].checked) { const gpuRefusal=lowspeedGpuRefusal(); if(gpuRefusal)throw new Error(gpuRefusal); flow.lowspeed_gpu = true; }
    if (backends.length === 1) flow.mach = 0;
  }
  if (backends.includes("lbm")) {
    if (!window.AirfoilLBM) throw new Error("The LBM 3D page code (lbm.js) is not loaded; reload the page.");
    Object.assign(flow, window.AirfoilLBM.flowKeys(flow, true));
  }
  applyReferenceTripControls(flow);
  const gurneyRefusal=gurneyBackendRefusal(flow,backends);if(gurneyRefusal)throw new Error(gurneyRefusal);
  const alignRefusal=streamlineRefusal(flow,backends);if(alignRefusal)throw new Error(alignRefusal);
  const speciesPhysical=backends.includes("cavitation")&&(flow.cav_fluid==="air" ? flow.cav_air_mode==="physical" : flow.cav_domain==="free_fluid");
  const physical=flow.cfd_time_mode==="physical" || speciesPhysical;
  if(physical){
    const duration=inputNumber("cfd-duration-ctu","physical duration",.001,10000);
    flow.cfd_burnin_ctu=inputNumber("cfd-burnin-ctu","burn-in time",0,duration);
    if(flow.cfd_burnin_ctu>=duration)throw new Error("Burn-in time must be shorter than physical duration.");
    flow.cfd_block_ctu=inputNumber("cfd-block-ctu","statistics block duration",.001,1000);
    if(speciesPhysical)flow.cav_duration_ctu=duration;
    if(flow.cfd_time_mode==="physical")flow.cfd_duration_ctu=duration;
    flow.cfd_auto_stop=false;
  }
  const sweep = { ...(preset?.sweep||{}), enabled: el["sweep-enabled"].checked };
  if (sweep.enabled) {
    sweep.start = inputNumber("alpha-start", "sweep start", -90, 90);
    sweep.stop = inputNumber("alpha-stop", "sweep stop", -90, 90);
    sweep.step = inputNumber("alpha-step", "sweep step", -90, 90);
    const intervals = (sweep.stop - sweep.start) / sweep.step;
    if (!Number.isFinite(intervals) || intervals <= 0) throw new Error("Sweep step must point toward a different end angle.");
    if (Math.abs(intervals - Math.round(intervals)) > 1e-7) throw new Error("Sweep step must land exactly on the end angle.");
    const count = Math.round(intervals) + 1;
    if (count > (hasCfd ? 31 : 121)) throw new Error(`Use at most ${hasCfd ? 31 : 121} angles in this sweep.`);
  }
  const display_every=window.LudwigOutputControls?.displayEvery()||100;
  flow.display_every=display_every;
  const value={airfoil:{kind,id},flow,sweep,backends,display_every,outputs:window.LudwigOutputControls?.read()||{}};
  if(flow.lbm_target_vram_percent!=null)value.target_vram_percent=flow.lbm_target_vram_percent;
  return value;
}
function underwaterControls(){
  const formulation=el["cav-fs-formulation"].value||"incompressible_projection";
  if(!["incompressible_projection","barotropic_explicit"].includes(formulation))throw new Error("Select a nonlinear free-surface formulation.");
  const presets={coarse:[64,16],standard:[96,24],fine:[128,32]},resolution=presets[el["cav-fs-resolution"].value||"standard"];
  if(!resolution)throw new Error("Select a nonlinear free-surface grid resolution.");
  const flow={cav_fs_formulation:formulation,cav_chord_m:inputNumber("cav-chord","physical chord",.001),cav_depth_chords:inputNumber("cav-depth","airfoil depth / chord",.001),
    cav_speed_m_s:inputNumber("cav-speed","physical speed",.001),cav_water_depth_chords:inputNumber("cav-water-depth","water depth / chord",.1),
    cav_air_height_chords:inputNumber("cav-air-height","air height / chord",.1),cav_fs_duration_seconds:inputNumber("cav-fs-duration","physical duration in seconds",1e-8),
    cav_fs_max_steps:inputNumber("cav-fs-max-steps","free-surface maximum step count",1,1e8),cav_fs_initial_vapor_fraction:inputNumber("cav-fs-vapor","initial vapor nuclei fraction",0,.01),cav_fs_n_foil:resolution[0],cav_fs_n_normal:resolution[1]};
  if(!Number.isInteger(flow.cav_fs_max_steps))throw new Error("Free-surface maximum step count must be a whole number.");
  if(flow.cav_depth_chords>=flow.cav_water_depth_chords)throw new Error("Airfoil depth must be smaller than the total water depth.");
  return flow;
}
function previewCase(mesh) {
  const kind = el["source-kind"].value;
  const preset=selectedCavitationReference();
  if(kind==="cavitation_reference"&&(!preset||state.cavitationReferenceAppliedId!==preset.id))return null;
  const id = kind === "naca4" ? el["naca-code"].value.trim() : kind === "uiuc" ? el["airfoil-select"].value : kind === "cavitation_reference" ? el["cav-reference-select"].value : kind;
  if ((kind === "naca4" && !/^\d{4}$/.test(id)) || !id) return null;
  const valueOr = (id, fallback) => finite(Number(el[id].value)) ? Number(el[id].value) : fallback;
  const flow = { ...(preset?.flow||{}), mach: valueOr("mach", 0.3), reynolds: valueOr("reynolds", 300000), alpha_deg: valueOr("alpha", 0),
    cfd_mesh: el["cfd-mesh"].value, cfd_resolution: el["preview-resolution"].value, cfd_te_topology:el["cfd-te-topology"].value||"auto",
    cfd_n_base: valueOr("cfd-n-base", 4) };
  if(selectedBackends(el.backend.value).includes("lowspeed"))flow.lowspeed_resolution=el["lowspeed-resolution"].value||"demo";
  if(selectedBackends(el.backend.value).includes("lbm")){const keys=window.AirfoilLBM?.flowKeys(flow,false);if(!keys)return null;Object.assign(flow,keys);}
  if(selectedBackends(el.backend.value).includes("cavitation")){
    flow.cav_fluid=el["cav-fluid"].value||"water";flow.cav_domain=el["cav-domain"].value||"tunnel";
    if(flow.cav_fluid==="water"&&flow.cav_domain!=="underwater")flow.sigma=valueOr("sigma",1);
    flow.cav_channel_height=valueOr("cav-channel",1.28);flow.cav_depth_chords=valueOr("cav-depth",1);
    if(flow.cav_domain==="underwater"){
      try{Object.assign(flow,underwaterControls());}catch{return null;}flow.cav_fluid="water";
      delete flow.mach;delete flow.cfd_te_topology;
    }
  }
  if(el["transition-mode"].value==="forced"){
    flow.trip_upper=valueOr("trip-upper",.05);flow.trip_lower=valueOr("trip-lower",.05);
  }
  if(flow.cav_domain==="underwater")flow.cfd_te_thickness=0;
  else if (el["trailing-edge-thickness"].value.trim() !== "") flow.cfd_te_thickness = valueOr("trailing-edge-thickness", 0);
  if(flow.cav_domain!=="underwater"&&gurneyEnabled()){const g=gurneyFlow(false);if(!g)return null;Object.assign(flow,g);}
  if(flow.cav_domain!=="underwater")Object.assign(flow,streamlineFlow());
  try{applyReferenceTripControls(flow);}catch{return null;}
  return { airfoil: { kind, id }, flow, mesh, backends:selectedBackends(el.backend.value) };
}
function scheduleGeometryPreview() {
  clearTimeout(state.previewTimer);
  state.previewAbort?.abort();
  const generation = ++state.previewGeneration;
  state.previewTimer = setTimeout(() => loadGeometryPreview(false, generation), 160);
}
function meshSettingsKey(value = previewCase(false)) {
  return value ? JSON.stringify([value.airfoil,value.flow]) : null;
}
function invalidatePreviewMesh() {
  if (state.preview) {
    state.preview.mesh = null;
    state.preview.settingsKey = meshSettingsKey();
    if (el["geometry-display"].value === "mesh") renderSelectedGeometry();
  }
  scheduleGeometryPreview();
}
async function loadGeometryPreview(mesh = false, generation = ++state.previewGeneration) {
  clearTimeout(state.previewTimer);
  const lowspeedMesh = mesh && selectedBackends(el.backend.value).includes("lowspeed");
  if (lowspeedMesh && Number(el["trailing-edge-thickness"].value) > 0) {
    displayError(gurneyEnabled() ? "The Gurney flap and the CFD trailing-edge thickness override both set the trailing edge. Clear the CFD trailing-edge thickness." : "Low-speed CFD uses a sharp-trailing-edge C-mesh. Clear the CFD trailing-edge thickness override to create its mesh.");
    return;
  }
  if (mesh && gurneyEnabled()) {try{gurneyFlow(true);}catch(err){displayError(err);return;}}
  const value = previewCase(mesh);
  if (!value) {if(mesh)displayError("Check the selected airfoil, transition and flow settings before creating the mesh. Nonlinear free surface also requires valid dimensions, duration and step limit.");return;}
  if (mesh) {const refusal=gurneyBackendRefusal(value.flow,value.backends)||streamlineRefusal(value.flow,value.backends);if(refusal){displayError(refusal);return;}}
  state.previewAbort?.abort();
  const controller = new AbortController(); state.previewAbort = controller;
  const done = beginOperation("preview", lowspeedMesh ? "Creating the low-speed CFD mesh in Julia (about a minute)…" : mesh ? "Creating mesh in Julia…" : "Loading selected airfoil…", mesh ? [[el["create-mesh"], "Creating…"], [el["refine-mesh"], "Meshing…"]] : []);
  const sourceKey = geometrySourceKey(value.airfoil, value.flow);
  const settingsKey = meshSettingsKey(value);
  if (state.preview?.sourceKey !== sourceKey) {
    state.preview = null; spatialMemory.geometry = null; spatialMemory.curvature = null;
    spatialViews.geometry?.destroy(); spatialViews.geometry = null;
    putChart("geometry-chart", null, "Loading selected airfoil…");
  }
  if (state.preview?.mesh && state.preview.settingsKey !== settingsKey) {
    state.preview.mesh = null;
    state.preview.settingsKey = settingsKey;
    renderSelectedGeometry();
  }
  el["geometry-status"].textContent = lowspeedMesh ? "Creating the low-speed CFD mesh in Julia (about a minute)…" : mesh ? "Creating mesh in Julia…" : "Loading selected airfoil…";
  try {
    if(mesh && !(state.serviceRevision>=minimumServiceRevision))await requireCurrentService();
    const payload = await request("/api/preview", { method: "POST", value, format: el.transport.value, signal: controller.signal });
    if (generation !== state.previewGeneration) return;
    if (!Array.isArray(payload.geometry?.x) || payload.geometry.x.length < 3) throw new Error("Geometry preview is missing coordinates");
    if(payload.mesh&&!validMesh(payload.mesh))throw new Error("Mesh preview contains an incomplete or invalid block");
    state.preview = { ...payload, sourceKey, settingsKey };
    if(mesh)rememberGurneyRange(value,payload.geometry_metadata?.gurney_baffle);
    updateSetup();
    el["geometry-title"].textContent = `${payload.geometry.name || "Airfoil"}${Number(value.flow.cfd_te_thickness)>0 ? " · CFD geometry" : ""}`;
    const gurneyText = value.flow.gurney_height!=null ? ` · ${gurneyLabel(value.flow)}${payload.gurney?` · wall at the TE station ${Number(payload.gurney.base_height).toFixed(5)} c (gap + h)`:""}` : "";
    const cells = payload.lbm ? (window.AirfoilLBM?.statusText(payload.lbm) || "LBM lattice") : payload.mesh ? `${meshCellCount(payload.mesh,true).toLocaleString()} mesh cells${meshTopologyText(payload.mesh)}` : `${payload.geometry.x.length} contour points`;
    el["geometry-status"].textContent = `${cells} · ${[payload.geometry_metadata?.geometry_support,payload.geometry_metadata?.conditioning].find(v=>typeof v==="string"&&v) || "conditioned source geometry"}${Number(value.flow.cfd_te_thickness)>0 ? " · trailing-edge thickness override applies to CFD" : ""}${gurneyText}`;
    el["geometry-status"].title=el["geometry-status"].textContent;
    state.preview.displayTitle=el["geometry-title"].textContent;
    state.preview.displayStatus=el["geometry-status"].textContent;
    if (mesh) el["geometry-display"].value = payload.lbm ? (window.AirfoilLBM?.displayAfterMesh(state.preview) || "outline") : "mesh";
    renderSelectedGeometry();
    if (!state.result && payload.mesh) renderFieldChoices(null);
  } catch (err) {
    if (generation === state.previewGeneration && err.name !== "AbortError") {
      el["geometry-status"].textContent = `Preview failed: ${err.message}`;
      if(mesh&&value.flow.gurney_height!=null&&/refused by the solver's plate quality gate/.test(String(err.message))){state.gurneyRefusal={key:gurneyRefusalKey(value),message:String(err.message)};updateSetup();}
    }
  } finally {
    done();
    if (state.previewAbort === controller) state.previewAbort = null;
  }
}
function renderSelectedGeometry() {
  if(el["source-kind"].value==="cavitation_reference"&&!state.preview){
    putChart("geometry-chart",null,"Loading selected reference geometry…");return;
  }
  if(window.AirfoilLBM?.renderGeometry(el["geometry-display"].value,state.preview))return;   // LBM 3D geometry/lattice views (lbm.js)
  const point=currentResult();
  const native=point?.native_geometry&&(!state.preview||state.preview.sourceKey===state.analysisGeometryKey) ? point.native_geometry : null;
  el["native-result-geometry"].disabled=!native;el["native-result-geometry"].hidden=!native;
  if(el["geometry-display"].value==="native"&&!native)el["geometry-display"].value="outline";
  const mode = el["geometry-display"].value;
  const geometry = mode==="native" ? native : state.preview?.geometry || state.result?.geometry;
  if(mode==="native"){el["geometry-title"].textContent=`${native.name || "Airfoil"} · native result geometry`;el["geometry-status"].textContent=`Source geometry used by ${labelBackend(point.backend)}; CFD trailing-edge overrides are shown in the other displays.`;}
  else if(state.preview){el["geometry-title"].textContent=state.preview.displayTitle || state.preview.geometry.name;el["geometry-status"].textContent=state.preview.displayStatus || "Conditioned geometry";}
  const curvature = mode === "curvature";
  el["curvature-scale-field"].classList.toggle("hidden", !curvature);
  el["curvature-mapping-field"].classList.toggle("hidden", !curvature);
  el["curvature-mapping-help"].classList.toggle("hidden", !curvature);
  el["geometry-chart"].classList.toggle("hidden", curvature);
  el["curvature-chart"].classList.toggle("hidden", !curvature);
  if (!geometry) return;
  if (curvature) {
    renderPorcupine(geometry);
  } else {
    const matchingRunMesh=(!state.preview || (state.preview.sourceKey===state.analysisGeometryKey && state.preview.settingsKey===state.analysisMeshKey)) ? currentMesh() : null;
    renderGeometry(geometry.x, geometry.y, geometry, mode === "mesh" ? state.preview?.mesh || matchingRunMesh : null);
  }
}
function jobStatus(job) {
  window.LudwigOutputControls?.render(job);
  const status = String(job.status || "queued");
  state.displayJobStatus=status;
  const stalled = status === "failed" && /stopped without convergence/i.test(String(job.message || ""));
  el["job-title"].textContent = stalled ? "Stopped without convergence" : ({ queued: "Queued", running: "Julia solvers running", completed: "Analysis complete", partial: "Analysis finished with partial results", failed: "Analysis failed" })[status] || status;
  el["job-message"].textContent = job.error || job.message || ({ queued: "Waiting for the worker.", running: "Computing the requested operating points.", completed: "Results are ready below." })[status] || "";
  let progress = Number(job.progress);
  if (!Number.isFinite(progress)) progress = ["completed","partial","failed"].includes(status) ? 100 : 0;
  else if (progress > 0 && progress <= 1) progress *= 100;
  progress = Math.max(0, Math.min(100, progress));
  el["job-percent"].textContent = `${Math.round(progress)}%`;
  el["progress-bar"].style.width = `${progress}%`;
  el["job-progress"].setAttribute("aria-valuenow",String(Math.round(progress)));
  el["job-id"].textContent = job.id ? `Job ${job.id}` : "";
  syncRunStatusDetails(job);refreshPointOverlays();
}
function syncRunStatusDetails(job=state.job) {
  const failed=(state.result?.results||[]).filter(point=>!["completed","converged","ok","experimental_physical_state"].includes(point.status||"completed"));
  const messages=[job?.error,["failed","partial"].includes(job?.status)?job.message:null,...failed.map(point=>point.error||point.message||point.convergence)].filter(value=>typeof value==="string"&&value.trim());
  for(const message of new Set(messages))if(!el["job-message"].textContent.includes(message))el["job-message"].textContent+=`\n${message}`;
  const failure=job?.status==="failed"||job?.status==="partial"&&failed.length>0;
  el["run-status"].classList.toggle("has-failure",!!failure);
  if(failure)el["run-status"].open=true;
  if(/waiting for the GPU/i.test(String(job?.message||"")))el["run-status"].open=true;   // a GPU job queued behind the shared lock: show who holds it
}
const pointOverlays=new Map(),pointOverlayHosts=["geometry-chart","curvature-chart","field-view","cp-chart"];
function pointOverlayData(hostId) {
  const geometryView=["geometry-chart","curvature-chart"].includes(hostId);
  const previewMismatch=geometryView&&state.preview&&
    (state.preview.sourceKey!==state.analysisGeometryKey||state.preview.settingsKey!==state.analysisMeshKey);
  const previewOnly=geometryView&&(!state.preview||!state.analysisCase||previewMismatch);
  let point=previewOnly?null:currentResult(),flow=state.analysisCase?.flow||{},name=state.result?.geometry?.name||state.analysisCase?.airfoil?.id||"Airfoil";
  const activeBackend=state.activeCfdAlpha!=null&&(state.analysisCase?.backends||[]).includes(state.activeCfdBackend)?state.activeCfdBackend:null;
  let backend=point?.backend||activeBackend||state.analysisCase?.backends?.[0],alpha=point?.alpha_deg??(activeBackend?state.activeCfdAlpha:state.requestedAlpha);
  let values=point,label="No coefficients",kind="empty";
  if(previewOnly){
    flow=previewCase(false)?.flow||{};name=state.preview?.geometry?.name||el["geometry-title"].textContent||"Geometry preview";
    backend=null;alpha=flow.alpha_deg;values=null;label="Geometry preview";
  }else if(point){
    if(["failed","invalid_physical_state","unsupported","error"].includes(point.status)){values=null;label="Failed · no coefficients";kind="failed";}
    else if(isCfdBackend(backend)){label=isPhysicalPoint(point)?"Instantaneous · unvalidated":validCfdPolarPoint(point)?"CFD · unvalidated":"Diagnostic · unverified";kind="diagnostic";}
    else{label=point.converged===false?"Unconverged":"Computed";kind=point.converged===false?"diagnostic":"saved";}
  }else{
    const running=["queued","running"].includes(state.displayJobStatus||state.job?.status);
    const key=`${backend}:${Number(alpha)}`,rows=state.liveHistory[key]||[],row=orderedHistory(rows).at(-1);
    if(running&&row){values=row;label=`Live n=${row.iteration} · mesh preview · unverified`;kind="live";}
    else if(["failed","partial"].includes(state.displayJobStatus||state.job?.status)){values=null;label="No saved coefficients";kind="failed";}
    else if(running)label="Waiting for solver";
  }
  const d=point?.diagnostics||{};
  const mach=previewOnly?flow.mach:point?.analysis_mach??d.mach??(backend==="jxfoil"?0:flow.mach);
  const re=previewOnly?flow.reynolds:point?.reynolds??d.reynolds??flow.reynolds;
  const format=value=>finite(Number(value))&&value!=null?Number(value):null;
  return {name,backend,alpha:format(alpha),mach:format(mach),re:format(re),cl:pointValue(values,"cl"),cd:pointValue(values,"cd"),cm:pointValue(values,"cm"),label,kind};
}
function attachPointOverlay(hostId) {
  const node=el[hostId];if(!node)return;
  let overlay=pointOverlays.get(hostId);
  if(!overlay){overlay=document.createElement("div");overlay.className="point-overlay";overlay.setAttribute("aria-label","Displayed operating point");pointOverlays.set(hostId,overlay);}
  const value=pointOverlayData(hostId),re=finite(value.re)?value.re>=1e6?`${(value.re/1e6).toPrecision(4)}M`:value.re.toPrecision(4):"—";
  overlay.textContent=`${value.name}${value.backend?` · ${labelBackend(value.backend)}`:""}\nα ${numberText(value.alpha,2)}° · Re ${re} · M ${numberText(value.mach,3)}\nCL ${numberText(value.cl,4)} · CD ${numberText(value.cd,5)} · CM ${numberText(value.cm,4)}\n${value.label}`;
  overlay.dataset.state=value.kind;
  if(!Array.from(node.children).includes(overlay))node.append(overlay);
}
function refreshPointOverlays(){for(const id of pointOverlayHosts)if(pointOverlays.has(id))attachPointOverlay(id);}
function resetLog() {
  state.logCursor = 0;
  state.historyCursor = 0;
  state.logLineCount = 0;
  state.liveHistory = Object.create(null);
  state.activeCfdAlpha = null;
  state.activeCfdBackend = "transonic";
  state.physicalProgress = Object.create(null);
  el["log-view"].replaceChildren(el["log-empty"]);
  el["log-empty"].textContent = "Waiting for process messages…";
  el["log-count"].textContent = "0 lines";
  el["log-state"].textContent = "Submitting case to Julia…";
  el["log-warning"].classList.add("hidden");
  el["log-warning"].textContent = "";
  el["log-download"].classList.add("hidden");
  el["log-download"].removeAttribute("href");
  el["log-follow"].classList.add("hidden");
  el["log-view"].scrollTop = 0;
  renderIterationHistory();
}
function updateLogFollow() {
  const view = el["log-view"];
  const awayFromBottom = view.scrollHeight - view.scrollTop - view.clientHeight > 35;
  el["log-follow"].classList.toggle("hidden", !state.logLineCount || !awayFromBottom);
}
function logTime(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value || "") : date.toLocaleTimeString([], { hour12: false });
}
function appendLog(payload, id) {
  const entries = Array.isArray(payload?.entries) ? payload.entries : [];
  const view = el["log-view"];
  const follow = !state.logLineCount || view.scrollHeight - view.scrollTop - view.clientHeight <= 35;
  const fragment = document.createDocumentFragment();
  let added = 0;
  let historyChanged = false;
  for (const entry of entries) {
    const seq = Number(entry?.seq);
    if (!Number.isSafeInteger(seq) || seq <= state.logCursor) continue;
    const row = document.createElement("div");
    row.className = "log-entry";
    const time = document.createElement("span");
    time.className = "log-entry-time";
    time.textContent = logTime(entry.at);
    time.title = String(entry.at || "");
    const source = document.createElement("span");
    source.className = "log-entry-source";
    source.textContent = `[${String(entry.source || "Julia")}]`;
    const line = document.createElement("span");
    line.className = "log-entry-line";
    line.textContent = String(entry.line ?? "");
    historyChanged = captureIterationLog(line.textContent, entry.source) || historyChanged;
    row.append(time, source, line);
    fragment.append(row);
    state.logCursor = seq;
    added++;
  }
  if (added) {
    el["log-empty"].remove();
    view.append(fragment);
    state.logLineCount += added;
    el["log-count"].textContent = `${state.logLineCount} line${state.logLineCount === 1 ? "" : "s"}`;
    if (follow) view.scrollTop = view.scrollHeight;
    updateLogFollow();
  }
  if (historyChanged) renderIterationHistory(currentResult());
  if (Number(payload?.dropped) > 0) {
    el["log-warning"].textContent = `${Number(payload.dropped)} older process line${Number(payload.dropped) === 1 ? " was" : "s were"} omitted from the live view. Download the full log to inspect them.`;
    el["log-warning"].classList.remove("hidden");
  }
  if (payload?.log_url) {
    const url = new URL(String(payload.log_url), location.origin);
    if (url.origin === location.origin && url.pathname.includes(`/api/jobs/${encodeURIComponent(id)}/files/`)) {
      el["log-download"].href = url.href;
      el["log-download"].download = `airfoil-${id}.log`;
      el["log-download"].classList.remove("hidden");
    }
  }
}
async function pollLogs(id, generation, drain = false) {
  try {
    for (let page = 0; page < (drain ? 20 : 1); page++) {
      const payload = await request(`/api/jobs/${encodeURIComponent(id)}/logs?after=${state.logCursor}`, { format: state.jobWire });
      if (generation !== state.pollGeneration) return false;
      appendLog(payload, id);
      el["log-state"].textContent = payload.status === "completed" ? "Analysis complete · final log available" : payload.status === "failed" ? "Analysis failed · inspect the messages below" : state.logLineCount ? "Receiving process output" : "Waiting for process output…";
      if (!drain || !Array.isArray(payload.entries) || !payload.entries.length) break;
    }
    return true;
  } catch (err) {
    if (generation === state.pollGeneration) el["log-state"].textContent = drain ? `Final log refresh failed: ${err.message}` : `Log connection interrupted: ${err.message}. Retrying…`;
    return false;
  }
}
async function pollHistory(id, generation, drain = false) {
  try {
    for (let page = 0; page < (drain ? 20 : 1); page++) {
      const payload = await request(`/api/jobs/${encodeURIComponent(id)}/history?after=${state.historyCursor}`, { format: state.jobWire });
      if (generation !== state.pollGeneration) return;
      const entries = Array.isArray(payload.entries) ? payload.entries : [];
      for (const entry of entries) {
        const seq = Number(entry.seq);
        if (!Number.isSafeInteger(seq) || seq <= state.historyCursor) continue;
        state.historyCursor = seq;
        if (entry.backend && !isCfdBackend(entry.backend)) continue;
        const backend=entry.backend||"transonic";
        const alpha = String(Number(entry.alpha_deg));
        if (alpha === "NaN") continue;
        state.activeCfdAlpha = alpha;
        state.activeCfdBackend=backend;
        const row = normalizeIterationHistory([entry])[0];
        if (!row) continue;
        row._history_priority=2;
        const rows = state.liveHistory[`${backend}:${alpha}`] ||= [];
        const prior = rows.findIndex(x => x.iteration === row.iteration);
        if (prior >= 0) rows[prior] = mergeHistoryRows(rows[prior],row); else rows.push(row);
      }
      if (entries.length) {
        el["convergence-panel"].classList.remove("hidden");
        renderIterationHistory(currentResult());
      }
      if (!drain || !entries.length) break;
    }
  } catch (err) {
    if (generation === state.pollGeneration && !state.result) el["convergence-state"].textContent = `Live history temporarily unavailable: ${err.message}`;
  }
}
async function pollTelemetry(id, generation) {
  // Telemetry must keep arriving while a status response or mesh artifact is
  // being serialized, and while the user is looking at a different tab.
  await Promise.allSettled([pollLogs(id, generation), pollHistory(id, generation)]);
  if (generation !== state.pollGeneration || ["completed", "partial", "failed"].includes(state.job?.status)) return;
  state.historyTimer = setTimeout(() => pollTelemetry(id, generation), 500);
}
async function pollJob(id, generation) {
  const [jobOutcome] = await Promise.allSettled([
    request(`/api/jobs/${encodeURIComponent(id)}`, { format: state.jobWire })
  ]);
  if (generation !== state.pollGeneration) return;
  if (jobOutcome.status === "fulfilled") {
    const job = jobOutcome.value;
    state.job = job; jobStatus(job);
    if (job.result && typeof job.result === "object") {
      const priorCount = state.result?.results?.length || 0;
      const priorMeshes = state.meshKeys;
      state.result = job.result;
      state.meshKeys = Object.keys(job.result.meshes || {}).sort().join("|"); refreshLiveField(); window.AirfoilLBM?.onResult();
      const resultCount = job.result.results?.length || 0;
      if (priorCount !== resultCount || ["completed","partial","failed"].includes(job.status)) {
        if (resultCount) renderResult();
        else if (state.meshKeys !== priorMeshes) { renderPoint(); renderPolars(); }
      } else if (state.meshKeys !== priorMeshes) { renderPoint(); renderPolars(); }
    }
    if (["completed","partial","failed"].includes(job.status)) {
      clearTimeout(state.historyTimer);
      await Promise.allSettled([pollLogs(id, generation, true), pollHistory(id, generation, true)]); // Collect final messages and telemetry.
      if (generation !== state.pollGeneration) return;
      state.finishAnalysis?.(); state.finishAnalysis=null;
      if (!state.job?.restored) loadRecentResults();
      if(job.exports?.status==='generating'){state.pollTimer=setTimeout(()=>pollJob(id,generation),1500);return;}
      return;
    }
  } else {
    el["job-message"].textContent = `Status request failed: ${jobOutcome.reason.message}. Retrying…`;
  }
  state.pollTimer = setTimeout(() => pollJob(id, generation), jobOutcome.status === "fulfilled" ? 1000 : 2000);
}
async function runAnalysis() {
  clearError();
    const checked=beginOperation("service-check","Checking Julia service…",[[el["run-button"],"Checking Julia…"]]);
    try {await requireCurrentService();}catch(err){displayError(err);return;}finally{checked();}
  let value;
  try { value = collectCase(); } catch (err) { displayError(err); return; }
  if (state.pollTimer) clearTimeout(state.pollTimer);
  clearTimeout(state.historyTimer);
  const generation = ++state.pollGeneration;
  state.job = null; state.result = null; state.jobWire = el.transport.value;
  state.analysisCase=value;
  state.analysisGeometryKey=geometrySourceKey(value.airfoil,value.flow);
  state.analysisMeshKey=meshSettingsKey();
  state.requestedAlpha = value.flow.alpha_deg; state.meshKeys = "";
  state.fullMeshes.clear(); state.meshLoading.clear();
  state.fullFields.clear(); state.fieldLoading.clear(); state.liveField=null; state.liveFieldLoading=null; state.liveAutoShown=false;
  chartMemory.clear();
  el["convergence-panel"].classList.toggle("hidden", !value.backends.some(backend=>isCfdBackend(backend)));
  spatialMemory.geometry = null; spatialMemory.field = null;
  resetLog();
  window.AirfoilBabylon?.hide();
  el["download-result"].disabled = true;
  renderSelection(); renderPolars(); renderTable();
  state.finishAnalysis?.();
  state.finishAnalysis=beginOperation("analysis", "Julia analysis running…", [[el["run-button"], "Analysis running…"]]);
  jobStatus({ status: "queued", message: "Submitting case to Julia…", progress: 0 });
  try {
    const job = await request("/api/jobs", { method: "POST", value, format: state.jobWire });
    if (!job.id) throw new Error("The service returned no job ID.");
    state.job = job; jobStatus(job);
    pollTelemetry(job.id, generation);
    pollJob(job.id, generation);
  } catch (err) {
    if (generation !== state.pollGeneration) return;
    state.finishAnalysis?.(); state.finishAnalysis=null;
    jobStatus({ status: "failed", error: err.message, progress: 0 });
    el["log-state"].textContent = "Submission failed before a job started.";
    displayError(err);
  }
}

function pointValue(p, key) { return finite(p?.[key]) ? p[key] : null; }
function polyline(series, options = {}) {
  return series.some(s => s.points?.some(p => finite(p?.x) && finite(p?.y))) ? { series, options } : null;
}
function polylineMarkup(series, options = {}) {
  const W = options.compact ? 360 : 600, H = options.compact ? 170 : options.short ? 208 : 245, m = { left: 45, right: 15, top: 16, bottom: 31 };
  const all = series.flatMap(s => (s.points || []).filter(p => finite(p?.x) && finite(p?.y)));
  if (!all.length) return null;
  let xmin = Math.min(...all.map(p => p.x)), xmax = Math.max(...all.map(p => p.x));
  let ymin = Math.min(...all.map(p => p.y)), ymax = Math.max(...all.map(p => p.y));
  if (options.geometry) { ymin = Math.min(ymin, -0.14); ymax = Math.max(ymax, 0.14); }
  if (xmin === xmax) { xmin -= 1; xmax += 1; }
  if (ymin === ymax) { const pad=options.singlePad || Math.max(0.05,Math.abs(ymin)*0.25); ymin=Math.max(options.nonnegative?0:-Infinity,ymin-pad); ymax+=pad; }
  const xpad = (xmax - xmin) * 0.04, ypad = (ymax - ymin) * 0.08;
  xmin -= xpad; xmax += xpad; ymin -= ypad; ymax += ypad;
  if (options.compact) xmin=Math.max(0,xmin);
  if (options.nonnegative) ymin=Math.max(0,ymin);
  if (options.viewport) ({ xmin, xmax, ymin, ymax } = options.viewport);
  const px = x => m.left + (x - xmin) / (xmax - xmin) * (W - m.left - m.right);
  const py = y => m.top + (options.invertY ? (y - ymin) : (ymax - y)) / (ymax - ymin) * (H - m.top - m.bottom);
  const xticks = Array.from({ length: 5 }, (_, i) => xmin + (xmax - xmin) * i / 4);
  const yticks = Array.from({ length: 5 }, (_, i) => ymin + (ymax - ymin) * i / 4);
  const tickText=(v,span)=>{const d=Math.max(0,Math.min(15,1-Math.floor(Math.log10(span/4))));const significant=Math.max(3,Math.min(15,2+Math.ceil(Math.log10(Math.max(Math.abs(v),span)/span))));return (Math.abs(v)>1e7||span<1e-7)?v.toExponential(significant):v.toFixed(d);};
  const gridLine=attributes=>state.gridVisible?`<line class="coordinate-grid" ${attributes} stroke="var(--grid)"/>`:"";
  const grid = xticks.map(t => `${gridLine(`x1="${px(t)}" y1="${m.top}" x2="${px(t)}" y2="${H-m.bottom}"`)}<text x="${px(t)}" y="${H-8}" text-anchor="middle" fill="#8295a1" font-size="10">${tickText(t,xmax-xmin)}</text>`).join("") + yticks.map(t => `${gridLine(`x1="${m.left}" y1="${py(t)}" x2="${W-m.right}" y2="${py(t)}"`)}<text x="${m.left-7}" y="${py(t)+3}" text-anchor="end" fill="#8295a1" font-size="10">${tickText(t,ymax-ymin)}</text>`).join("");
  const lines = series.map(s => {
    const good = (s.points || []).filter(p => finite(p?.x) && finite(p?.y));
    if (!good.length) return "";
    const path = good.map((p, i) => `${i ? "L" : "M"}${px(p.x).toFixed(2)} ${py(p.y).toFixed(2)}`).join(" ");
    const line = good.length > 1 && s.connect !== false ? `<path d="${path}" fill="none" stroke="${s.color}" stroke-width="2.5" ${s.dash ? `stroke-dasharray="${escapeHtml(s.dash)}"` : ""} stroke-linecap="round" stroke-linejoin="round"/>` : "";
    const dots = options.markers === false && good.length > 1 && s.marker !== "hollow" ? "" : good.map(p => s.marker === "hollow" ? `<circle cx="${px(p.x)}" cy="${py(p.y)}" r="${options.short ? 4.5 : 3.5}" fill="var(--paper)" stroke="${s.color}" stroke-width="2.3"><title>Diagnostic only: α=${numberText(p.x,2)}°, value=${numberText(p.y,4)}</title></circle>` : `<circle cx="${px(p.x)}" cy="${py(p.y)}" r="${options.short ? 3.6 : 2.5}" fill="${s.color}"/>`).join("");
    return line + dots;
  }).join("");
  const clipId = `chart-clip-${++chartClipSequence}`;
  return { bounds: {xmin,xmax,ymin,ymax}, width: W, height: H, margin: m, html: `<svg class="plot-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${escapeHtml(options.title || "Data chart")}"><defs><clipPath id="${clipId}"><rect x="${m.left}" y="${m.top}" width="${W-m.left-m.right}" height="${H-m.top-m.bottom}"/></clipPath></defs>${grid}<g clip-path="url(#${clipId})">${lines}</g>${options.xLabel ? `<text x="${W-m.right}" y="${H-1}" text-anchor="end" font-size="9">${escapeHtml(options.xLabel)}</text>` : ""}${options.yLabel ? `<text x="${m.left}" y="10" font-size="9">${escapeHtml(options.yLabel)}</text>` : ""}</svg><div class="plot-legend">${series.filter(s => s.points?.length).map(s => `<span><i class="${s.marker === "hollow" ? "hollow" : ""}" style="${s.marker === "hollow" ? `background:var(--paper);border-color:${s.color}` : `background:${s.color}`}"></i>${escapeHtml(s.name)}</span>`).join("")}</div>` };
}
let chartClipSequence = 0;
const chartMemory = new Map(), chartViews = new Map();
class DataChart {
  constructor(id, node, spec) {
    chartViews.get(id)?.destroy();
    this.id=id; this.node=node; this.spec=spec;
    this.bounds=chartMemory.get(id) || null; this.drag=null;
    node.classList.add("interactive-chart"); node.tabIndex=0;
    node.title="Wheel: zoom · drag: box zoom · right/middle/Shift drag: pan · double click: fit";
    this.events = {
      wheel: e=>{ e.preventDefault(); const p=this.position(e); if(!p)return; this.zoomAt(Math.exp(-e.deltaY*0.0014),p); },
      pointerdown: e=>{
        if(![0,1,2].includes(e.button))return; const p=this.position(e); if(!p)return; e.preventDefault();
        this.drag={mode:e.button!==0||e.shiftKey||e.ctrlKey?"pan":"box",start:p,last:p,bounds:{...this.currentBounds}};
        node.setPointerCapture?.(e.pointerId);
        if(this.drag.mode==="box") { this.box=document.createElement("div"); this.box.className="zoom-selection"; node.append(this.box); this.drawBox(p,p); }
        node.classList.add(this.drag.mode==="pan"?"is-panning":"is-selecting");
      },
      pointermove:e=>{
        if(!this.drag)return; const p=this.position(e); if(!p)return;
        if(this.drag.mode==="box")this.drawBox(this.drag.start,p);
        else { const a=this.world(this.drag.last), b=this.world(p), v=this.currentBounds; this.bounds={xmin:v.xmin+a.x-b.x,xmax:v.xmax+a.x-b.x,ymin:v.ymin+a.y-b.y,ymax:v.ymax+a.y-b.y}; this.save(); this.draw(); }
        this.drag.last=p;
      },
      pointerup:e=>{
        if(!this.drag)return; const p=this.position(e)||this.drag.last;
        if(this.drag.mode==="box"&&Math.abs(p.x-this.drag.start.x)>5&&Math.abs(p.y-this.drag.start.y)>5){const a=this.world(this.drag.start),b=this.world(p);this.bounds={xmin:Math.min(a.x,b.x),xmax:Math.max(a.x,b.x),ymin:Math.min(a.y,b.y),ymax:Math.max(a.y,b.y)};this.save();this.draw();}
        this.endDrag();
      },
      pointercancel:()=>this.endDrag(), contextmenu:e=>e.preventDefault(),
      dblclick:e=>{e.preventDefault();this.fit();},
      keydown:e=>{if(e.key==="Escape")this.endDrag();else if(e.key==="Home"||e.key==="f"){e.preventDefault();this.fit();}}
    };
    for(const [name,fn] of Object.entries(this.events))node.addEventListener(name,fn,{passive:false});
    chartViews.set(id,this);this.draw();
  }
  destroy(){this.pendingSpec=null;this.endDrag();for(const [name,fn] of Object.entries(this.events))this.node.removeEventListener(name,fn);chartViews.delete(this.id);}
  updateSpec(spec){if(this.drag){this.pendingSpec=spec;return;}this.spec=spec;this.draw();}
  endDrag(){this.drag=null;this.box?.remove();this.box=null;this.node.classList.remove("is-panning","is-selecting");if(this.pendingSpec){this.spec=this.pendingSpec;this.pendingSpec=null;this.draw();}}
  drawBox(a,b){const r=this.node.getBoundingClientRect(),s=this.node.querySelector("svg").getBoundingClientRect();Object.assign(this.box.style,{left:`${s.left-r.left+Math.min(a.x,b.x)*s.width/this.width}px`,top:`${s.top-r.top+Math.min(a.y,b.y)*s.height/this.height}px`,width:`${Math.abs(a.x-b.x)*s.width/this.width}px`,height:`${Math.abs(a.y-b.y)*s.height/this.height}px`});}
  position(e){const svg=this.node.querySelector("svg");if(!svg)return null;const r=svg.getBoundingClientRect();return {x:Math.max(this.margin.left,Math.min(this.width-this.margin.right,(e.clientX-r.left)/r.width*this.width)),y:Math.max(this.margin.top,Math.min(this.height-this.margin.bottom,(e.clientY-r.top)/r.height*this.height))};}
  world(p){const b=this.currentBounds,m=this.margin;const fx=(p.x-m.left)/(this.width-m.left-m.right),fy=(p.y-m.top)/(this.height-m.top-m.bottom);return{x:b.xmin+fx*(b.xmax-b.xmin),y:this.spec.options.invertY?b.ymin+fy*(b.ymax-b.ymin):b.ymax-fy*(b.ymax-b.ymin)};}
  save(){chartMemory.set(this.id,{...this.bounds});}
  zoomAt(factor,p){const b=this.currentBounds,w=this.world(p),fx=(w.x-b.xmin)/(b.xmax-b.xmin),fy=(w.y-b.ymin)/(b.ymax-b.ymin);const dx=Math.max((b.xmax-b.xmin)/factor,Math.max(1,Math.abs(w.x))*Number.EPSILON*64),dy=Math.max((b.ymax-b.ymin)/factor,Math.max(1,Math.abs(w.y))*Number.EPSILON*64);if(!finite(dx)||!finite(dy))return;this.bounds={xmin:w.x-fx*dx,xmax:w.x+(1-fx)*dx,ymin:w.y-fy*dy,ymax:w.y+(1-fy)*dy};this.save();this.draw();}
  fit(){this.bounds=null;chartMemory.delete(this.id);this.draw();}
  draw(){const plot=polylineMarkup(this.spec.series,{...this.spec.options,viewport:this.bounds});if(!plot)return;this.currentBounds=plot.bounds;this.width=plot.width;this.height=plot.height;this.margin=plot.margin;this.node.innerHTML=plot.html;if(this.id==="cp-chart")attachPointOverlay(this.id);}
}
function putChart(id, content, emptyText) {
  const node = el[id];
  node.classList.toggle("empty-chart", !content);
  if (content?.series) {const view=chartViews.get(id);if(view)view.updateSpec(content);else new DataChart(id,node,content);}
  else { chartViews.get(id)?.destroy(); node.classList.remove("interactive-chart"); if (content) node.innerHTML = content; else node.textContent = emptyText; }
  if(id==="cp-chart")attachPointOverlay(id);
}

function captureIterationLog(line, source="transonic") {
  const backend=/lowspeed/i.test(String(source)) ? "lowspeed" : /^lbm$/i.test(String(source)) ? "lbm" : /cav|species/i.test(String(source)) ? "cavitation" : "transonic";
  const alpha = line.match(/(?:Starting (?:transonic|cavitation|lowspeed|lbm) point|Worker process started).*?alpha\s*[=:]\s*([-+\d.eE]+)/i);
  if (alpha) {
    state.activeCfdAlpha = String(Number(alpha[1]));
    state.activeCfdBackend=backend;
    state.liveHistory[`${backend}:${state.activeCfdAlpha}`] ||= [];
  }
  const physical=line.match(/PHYSICAL_TIME,(\d+),([-+\d.eE]+),([-+\d.eE]+),([-+\d.eE]+)/);
  const iteration = physical || line.match(/\bn\s*=\s*(\d+)(?:\s*\/\s*\d+)?/i);
  if (!iteration || state.activeCfdAlpha === null) return false;
  const number = key => {
    const match = line.match(new RegExp(`\\b${key}\\s*=\\s*([-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?)`, "i"));
    return match ? Number(match[1]) : null;
  };
  const record = { iteration: Number(iteration[1]), cl: number("cl"), cd: number("cd"), residual_scaled: number("(?:Rscaled|residual_scaled|residual|res|r)"), mass_net_frac: number("(?:mass_net_frac|mass_net)"), _history_priority:1 };
  const key=`${backend}:${state.activeCfdAlpha}`;
  state.activeCfdBackend=backend;
  const rows = state.liveHistory[key] ||= [];
  if(physical){
    record.t_ctu=Number(physical[2]);
    state.physicalProgress ||= Object.create(null);
    state.physicalProgress[key]={elapsed_ctu:record.t_ctu,requested_duration_ctu:Number(physical[3]),dt_ctu:Number(physical[4])};
  }
  const old = rows.findIndex(row => row.iteration === record.iteration);
  if (old >= 0) rows[old] = mergeHistoryRows(rows[old],record); else rows.push(record);
  return true;
}
function normalizeIterationHistory(raw) {
  if (!raw) return [];
  let source = raw;
  if (!Array.isArray(source) && Array.isArray(source.records)) source = source.records;
  if (!Array.isArray(source) && typeof source === "object") {
    const steps = source.iteration || source.iterations || source.step || source.n;
    if (Array.isArray(steps)) source = steps.map((iteration, i) => ({ iteration, t_ctu: (source.t_ctu || [])[i], t_seconds:(source.t_seconds||source.elapsed_seconds||[])[i], residual_scaled: (source.residual_scaled || source.residual || source.res || [])[i], residual_linf: (source.residual_linf || [])[i], mass_net_frac: (source.mass_net_frac || [])[i], cl: (source.cl || source.Cl || [])[i], cd: (source.cd || source.Cd || [])[i] }));
  }
  if (!Array.isArray(source)) return [];
  const n = v => v == null || v === "" ? null : Number(v);
  return orderedHistory(source.map(row => Array.isArray(row) ? { iteration: n(row[0]), cl: n(row[1]), cd: n(row[2]), residual_scaled: n(row[3]), mass_net_frac: n(row[4]) } : {
    iteration: n(row.iteration ?? row.step ?? row.n), _history_priority:n(row._history_priority),
    t_ctu: n(row.t_ctu ?? row.elapsed_ctu),
    t_seconds:n(row.t_seconds ?? row.elapsed_seconds ?? row.time_seconds),
    residual_scaled: n(row.residual_scaled ?? row.residual ?? row.res ?? row.r ?? row.residual_linf),
    residual_linf: n(row.residual_linf),
    mass_net_frac: n(row.mass_net_frac),
    cl: n(row.cl ?? row.Cl ?? row.CL), cd: n(row.cd ?? row.Cd ?? row.CD), cm:n(row.cm ?? row.Cm ?? row.CM)
  }));
}
function mergeHistoryRows(old,row) {
  const numeric=value=>Object.fromEntries(Object.entries(value).filter(([,v])=>finite(v)));
  // Rounded text logs can fill missing fields while structured telemetry and
  // final CSV records retain their full numerical precision, in either order.
  return (old._history_priority||0)>(row._history_priority||0) ? {...numeric(row),...numeric(old)} : {...numeric(old),...numeric(row)};
}
function orderedHistory(rows) {
  // Log messages and the binary history endpoint arrive independently. Their
  // cursors are monotone, but delivery between the two streams is not. Never
  // connect a newly delivered older sample after a newer one on the chart.
  const byStep = new Map();
  for (const row of rows) {
    if (!Number.isSafeInteger(row?.iteration) || row.iteration < 0) continue;
    const old = byStep.get(row.iteration) || { iteration: row.iteration };
    byStep.set(row.iteration,mergeHistoryRows(old,row));
  }
  return [...byStep.values()].sort((a, b) => a.iteration - b.iteration);
}
function isPhysicalPoint(point) {return point?.status==="experimental_physical_state"||point?.diagnostics?.time_mode==="physical" || point?.diagnostics?.clock==="physical time";}
function isProjectionPoint(point){
  const diag=point?.diagnostics||{};
  return diag.formulation==="incompressible_projection"||diag.cav_fs_formulation==="incompressible_projection"||diag.residual_kind==="volume_flux_divergence"||String(diag.residual_quantity||"").includes("volume")&&String(diag.residual_quantity||"").includes("divergence");
}
function isUnderwaterPoint(point){return point?.diagnostics?.domain==="underwater"||point?.diagnostics?.cav_domain==="underwater"||point?.backend==="cavitation"&&point?.status==="experimental_physical_state";}
function availableFieldQuantity(point,key){
  if(isUnderwaterPoint(point)&&["temperature","viscosity","nut","turbulence","nu","mu","viscosity_ratio","turbulence_ratio","eddy_viscosity"].includes(key))return false;
  return !(key==="mach"&&isProjectionPoint(point));
}
function experimentalStateLabel(point){return isProjectionPoint(point)?"Experimental inviscid physical state":"Experimental inviscid startup";}
function analysisUsesPhysicalClock(backend) {
  const flow=state.analysisCase?.flow||{};
  return backend==="lowspeed" ? false : backend==="transonic" ? flow.cfd_time_mode==="physical" : flow.cav_fluid==="air" ? flow.cav_air_mode==="physical" : ["free_fluid","underwater"].includes(flow.cav_domain);
}
function renderIterationHistory(point = currentResult()) {
  refreshPointOverlays();
  if (!el["convergence-panel"]) return;
  const running = ["queued", "running"].includes(state.job?.status), cfd=isCfdBackend(point?.backend);
  if (!running && point && !cfd) {
    el["convergence-state"].textContent = "Select a CFD point to inspect its history.";
    el["sampling-state"].classList.add("hidden");
    for (const [id,label] of [["history-residual","mass residual"],["history-cl","lift"],["history-cd","drag"]]) putChart(id,null,`Select a CFD point for ${label} history.`);
    return;
  }
  const backend=running ? state.activeCfdBackend||state.analysisCase?.backends?.find(isCfdBackend)||"transonic" : point?.backend||state.activeCfdBackend||"transonic";
  const alpha = running ? state.activeCfdAlpha : cfd ? String(Number(point.alpha_deg)) : state.activeCfdAlpha, key=`${backend}:${alpha}`;
  const finalRows = !running && cfd ? normalizeIterationHistory(point.iteration_history || point.history || point.diagnostics?.iteration_history).map(row=>({...row,_history_priority:3})) : [];
  const rows = orderedHistory([...finalRows, ...(state.liveHistory[key] || [])]);
  if (rows.length) el["convergence-panel"].classList.remove("hidden");
  const label = alpha == null ? "current point" : `${numberText(Number(alpha), 2)}°`;
  const physical=running ? analysisUsesPhysicalClock(backend) || !!state.physicalProgress?.[key] : isPhysicalPoint(point);
  const guard = !running&&!physical ? point?.diagnostics?.convergence_guard : null;
  const guardText = guard?.status ? ` · ${guard.stopped_without_convergence || guard.status === "stalled_stop" ? "stalled; stopped without convergence" : "stop guard: " + String(guard.status).replaceAll("_", " ")}${guard.stopped_early ? " at n=" + rows[rows.length - 1]?.iteration : ""}` : "";
  const latestNet = rows[rows.length-1]?.mass_net_frac;
  const netText = finite(latestNet) ? ` · signed net mass ${latestNet.toExponential(2)}/(ρU∞c)` : "";
  const progress=state.physicalProgress?.[key]||{},diag=point?.diagnostics||{},stats=!running&&physical ? diag.time_statistics : null;
  const underwater=running?state.analysisCase?.flow?.cav_domain==="underwater":point?.status==="experimental_physical_state"||diag.domain==="underwater"||diag.cav_domain==="underwater";
  const projection=underwater&&(running?state.analysisCase?.flow?.cav_fs_formulation==="incompressible_projection":isProjectionPoint(point));
  const rawProjection=projection&&rows.some(row=>finite(row.residual_linf));
  const residualField=rawProjection?"residual_linf":"residual_scaled";
  const lowspeedHistory=backend==="lowspeed",lbmHistory=backend==="lbm";
  const residualLabel=lbmHistory?"force change per sample, |ΔCL|+|ΔCD|":lowspeedHistory?"solver steady residual (momentum)":projection?rawProjection?"volume-flux divergence, 1/s":"volume-flux divergence × c/U":underwater?"mass-rate diagnostic":"scaled mass residual";
  el["history-residual-title"].textContent=lbmHistory?"LBM force change per sample, |ΔCL|+|ΔCD| · log₁₀":lowspeedHistory?"Steady residual (momentum; stop at 1e-6) · log₁₀":projection?`Volume-flux divergence (${rawProjection?"L∞, 1/s":"scaled by c/U"}) · log₁₀`:underwater?"Mass-rate diagnostic · log₁₀":"Mass residual (scaled L∞) · log₁₀";
  const secondTimes=rows.map(r=>r.t_seconds).filter(finite),seconds=!running&&finite(diag.elapsed_seconds)?diag.elapsed_seconds:secondTimes.length?Math.max(...secondTimes):null;
  const requestedSeconds=!running?diag.requested_duration_seconds:state.analysisCase?.flow?.cav_fs_duration_seconds;
  const times=rows.map(r=>r.t_ctu).filter(finite),elapsed=!running&&finite(diag.elapsed_ctu) ? diag.elapsed_ctu : times.length ? Math.max(...times) : progress.elapsed_ctu;
  const requested=!running ? diag.requested_duration_ctu : progress.requested_duration_ctu ?? (backend==="transonic" ? state.analysisCase?.flow?.cfd_duration_ctu : state.analysisCase?.flow?.cav_duration_ctu);
  const timeText=physical ? underwater?` · physical time ${finite(seconds)?Number(seconds).toPrecision(5):"pending"}${finite(requestedSeconds)?" / "+requestedSeconds:""} s${finite(elapsed)?` · ${Number(elapsed).toPrecision(4)} CTU`:""}`:` · physical time ${finite(elapsed)?Number(elapsed).toPrecision(5):"pending"}${finite(requested)?" / "+requested:""} CTU` : "";
  el["convergence-state"].textContent = rows.length ? `${label} · ${labelBackend(backend)}${timeText} · ${rows.length} recorded iteration${rows.length === 1 ? "" : "s"} · latest n=${rows[rows.length-1].iteration}${netText}${guardText}` : `Waiting for CFD ${physical?"physical-time":"iteration"} output.${timeText}`;
  el["convergence-state"].title = guard?.reason || "";
  el["sampling-state"].classList.toggle("hidden",!physical);
  if(physical){
    const status=running ? "sampling in progress; means unqualified" : String(stats?.status||"insufficient_sampling").replaceAll("_"," ");
    const summary=stats?.available ? ` · retained ${stats.average_duration_ctu?.toPrecision(4)} CTU · ${stats.block_count} blocks · CL mean ${numberText(stats.cl_mean)} · CD mean ${numberText(stats.cd_mean,5)}` : "";
    el["sampling-state"].textContent=`${status}${summary} · field is instantaneous; physical stall parity unvalidated`;
    el["sampling-state"].title=stats?.reason||stats?.validation_scope||"Sampling, stationarity, block independence and timestep refinement require separate checks.";
    if(underwater){el["sampling-state"].textContent=`${projection?"Experimental inviscid physical state":"Experimental inviscid startup"} · instantaneous fields and loads only · no SST, energy or phase change · cavitation, drag and stall parity unqualified`;el["sampling-state"].title="The physical clock advances only through accepted solver steps. A step-limit stop may occur before the requested duration.";}
  }
  const secondsAxis=underwater&&secondTimes.length>0,physicalAxis=physical&&times.length>0;
  el["diagnostics-axis-note"].textContent=`Live updates every 0.5 s. Wheel: zoom · drag: box zoom · right drag: pan · double click: fit. Horizontal axis: ${secondsAxis?"actual physical time, seconds":physicalAxis?"actual physical time t U∞ / c (CTU)":"solver iteration"}.${underwater?` Residual: ${residualLabel}. Experimental startup diagnostics do not establish steady convergence or stall parity.${projection?" Pressure iteration residuals describe the inner linear solve separately.":""}`:lbmHistory?" LBM 3D: the solver's force samples (about 16 per convective time); the residual is the change of CL plus CD between samples, not an equation residual. Horizontal axis: LBM time step.":lowspeedHistory?" Residual: the low-speed solver's own steady residual of the momentum equation (ρU²/c per unit volume); its steady stop also needs the k and ω residuals below 1e-7 and the far-field circulation settled. The solver's mass balance is exact to round-off at every step.":" Residual shows volume-scaled continuity mass flux divergence."}${physical?" The iteration budget can end before the requested physical duration.":""}`;
  for (const [field, id, color, transform] of [
    [residualField, "history-residual", colors.blue || "#3e69ac", v => v > 0 ? Math.log10(v) : null],
    ["cl", "history-cl", colors.jxfoil, v => v],
    ["cd", "history-cd", colors.transonic, v => v]
  ]) {
    const points = rows.map(row => ({ x: secondsAxis?row.t_seconds:physicalAxis ? row.t_ctu : row.iteration, y: finite(row[field]) ? transform(row[field]) : null })).filter(row => finite(row.x)&&finite(row.y));
    const residual=id==="history-residual";
    putChart(id, polyline([{ name: residual ? `log₁₀ ${residualLabel}` : field.toUpperCase(), color, points }], { short: true, compact: true, markers: false, xLabel:secondsAxis?"physical time, s":physicalAxis?"t U∞ / c (CTU)":"iteration", title: `${residual?residualLabel:field} over ${physicalAxis||secondsAxis?"physical time":"CFD iterations"}` }), `Waiting for ${residual?residualLabel:field.toUpperCase()} samples.`);
  }
}

const spatialViews = { geometry: null, curvature: null, field: null };
const spatialMemory = { geometry: null, curvature: null, field: null };
class SpatialPlot {
  constructor(kind, node, bounds, painter, probe = null, options = {}) {
    const prior=spatialViews[kind];
    if(prior?.drag){prior.pendingPlot={kind,node,bounds,painter,probe,options};return prior;}
    spatialViews[kind]?.destroy();
    this.kind = kind; this.node = node; this.bounds = bounds; this.painter = painter; this.probe = probe; this.options = options;
    this.canvas = document.createElement("canvas"); this.ctx = this.canvas.getContext("2d");
    this.tooltip = document.createElement("div"); this.tooltip.className = "field-probe hidden";
    node.classList.remove("empty-chart"); node.replaceChildren(this.canvas, this.tooltip);
    this.width = 1; this.height = 1;
    const savedMemory=spatialMemory[kind],memory=savedMemory?.axisUnits===(options.axisUnits||"chord")?savedMemory:null;
    this.centerX = memory?.centerX ?? (bounds.xmin + bounds.xmax)/2;
    this.centerY = memory?.centerY ?? (bounds.ymin + bounds.ymax)/2;
    this.zoom = memory?.zoom ?? 1;
    this.drag = null; this.frame = null;
    this.onWheel = e => { e.preventDefault(); const r = this.canvas.getBoundingClientRect(); this.zoomAt(Math.exp(-e.deltaY*0.0014), e.clientX-r.left, e.clientY-r.top); };
    this.onDown = e => {
      if (![0,1,2].includes(e.button)) return;
      e.preventDefault();
      const r=this.canvas.getBoundingClientRect(), p={x:e.clientX-r.left,y:e.clientY-r.top};
      const pan=e.button!==0||e.shiftKey||e.ctrlKey||e.pointerType==="touch";
      this.drag = { mode:pan?"pan":"box", start:p, last:p };
      this.canvas.setPointerCapture?.(e.pointerId); this.canvas.classList.add(pan?"is-panning":"is-selecting");
      if(!pan){this.box=document.createElement("div");this.box.className="zoom-selection";this.node.append(this.box);this.drawBox(p,p);}
    };
    this.onMove = e => {
      const r = this.canvas.getBoundingClientRect(), px = e.clientX-r.left, py = e.clientY-r.top;
      if(this.drag){if(this.drag.mode==="pan"){this.centerX-=(px-this.drag.last.x)/this.scale;this.centerY+=(py-this.drag.last.y)/this.scale;this.remember();this.schedule();}else this.drawBox(this.drag.start,{x:px,y:py});this.drag.last={x:px,y:py};}
      if (this.probe) { const text = this.probe(this.worldX(px), this.worldY(py)); this.tooltip.textContent = text || ""; this.tooltip.classList.toggle("hidden", !text); }
    };
    this.onUp = () => { if(this.drag?.mode==="box"){const a=this.drag.start,b=this.drag.last;if(Math.abs(a.x-b.x)>5&&Math.abs(a.y-b.y)>5){this.centerX=this.worldX((a.x+b.x)/2);this.centerY=this.worldY((a.y+b.y)/2);this.zoomAt(Math.min(this.width/Math.abs(a.x-b.x),this.height/Math.abs(a.y-b.y)));}}this.endDrag(); };
    this.onCancel = () => this.endDrag();
    this.onDoubleClick = e=>{e.preventDefault();this.center();};
    this.onContextMenu = e=>e.preventDefault();
    this.onLeave = () => { if (!this.drag) this.tooltip.classList.add("hidden"); };
    this.canvas.addEventListener("wheel", this.onWheel, {passive:false}); this.canvas.addEventListener("pointerdown", this.onDown); this.canvas.addEventListener("pointermove", this.onMove); this.canvas.addEventListener("pointerup", this.onUp); this.canvas.addEventListener("pointercancel", this.onCancel); this.canvas.addEventListener("pointerleave", this.onLeave);this.canvas.addEventListener("dblclick",this.onDoubleClick);this.canvas.addEventListener("contextmenu",this.onContextMenu);
    this.canvas.tabIndex=0;this.onKey=e=>{if(e.key==="Escape")this.endDrag();if(e.key==="Home"||e.key==="f"){e.preventDefault();this.center();}};this.canvas.addEventListener("keydown",this.onKey);
    this.observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => this.resize());
    this.observer?.observe(node);
    this.onWindowResize = () => this.resize();
    if (!this.observer) window.addEventListener("resize", this.onWindowResize);
    spatialViews[kind] = this;
    this.resize();attachPointOverlay(node.id);
  }
  destroy() { this.pendingPlot=null; this.endDrag(); this.observer?.disconnect(); if (!this.observer) window.removeEventListener("resize", this.onWindowResize); if (this.frame != null) cancelAnimationFrame(this.frame); }
  endDrag(){this.drag=null;this.box?.remove();this.box=null;this.canvas.classList.remove("is-panning","is-selecting");if(this.pendingPlot){const next=this.pendingPlot;this.pendingPlot=null;new SpatialPlot(next.kind,next.node,next.bounds,next.painter,next.probe,next.options);}}
  drawBox(a,b){Object.assign(this.box.style,{left:`${Math.min(a.x,b.x)}px`,top:`${Math.min(a.y,b.y)}px`,width:`${Math.abs(a.x-b.x)}px`,height:`${Math.abs(a.y-b.y)}px`});}
  remember() { spatialMemory[this.kind] = { centerX:this.centerX, centerY:this.centerY, zoom:this.zoom,axisUnits:this.options.axisUnits||"chord" }; }
  resize() {
    const rect = this.node.getBoundingClientRect();
    this.width = Math.max(100, rect.width || (this.kind === "geometry" ? 600 : 900));
    this.height = Math.max(100, rect.height || (this.kind === "geometry" ? 260 : 430));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.dpr = dpr;
    this.canvas.width = Math.round(this.width*dpr); this.canvas.height = Math.round(this.height*dpr);
    this.ctx?.setTransform(dpr,0,0,dpr,0,0);
    this.baseScale = Math.min((this.width-50)/(this.bounds.xmax-this.bounds.xmin), (this.height-50)/(this.bounds.ymax-this.bounds.ymin));
    this.scale = this.baseScale*this.zoom;
    this.schedule();
  }
  x(v) { return this.width/2+(v-this.centerX)*this.scale; }
  y(v) { return this.height/2-(v-this.centerY)*this.scale; }
  worldX(px) { return this.centerX+(px-this.width/2)/this.scale; }
  worldY(py) { return this.centerY-(py-this.height/2)/this.scale; }
  zoomAt(factor, px=this.width/2, py=this.height/2) {
    const wx=this.worldX(px), wy=this.worldY(py);
    const maxScale=Math.min(this.width,this.height)/(64*Number.EPSILON*Math.max(1,Math.abs(wx),Math.abs(wy)));
    const nextScale=Math.min(maxScale,Math.max(Number.EPSILON,this.scale*factor));
    if(!finite(nextScale)||nextScale<=0)return;
    this.zoom=nextScale/this.baseScale; this.scale=nextScale;
    this.centerX=wx-(px-this.width/2)/this.scale; this.centerY=wy+(py-this.height/2)/this.scale;
    this.remember(); this.schedule();
  }
  focus(xmin,xmax,ymin,ymax){const s=Math.min((this.width-20)/(xmax-xmin),(this.height-20)/(ymax-ymin));if(!finite(s)||s<=0)return;this.centerX=(xmin+xmax)/2;this.centerY=(ymin+ymax)/2;this.scale=s;this.zoom=s/this.baseScale;this.remember();this.schedule();}
  center() { this.centerX=(this.bounds.xmin+this.bounds.xmax)/2; this.centerY=(this.bounds.ymin+this.bounds.ymax)/2; this.zoom=1; this.scale=this.baseScale; this.remember(); this.schedule(); }
  schedule() { if (this.frame != null) return; this.frame=requestAnimationFrame(() => { this.frame=null; this.draw(); }); }
  draw() {
    const c=this.ctx; if (!c) return;
    const css=getComputedStyle(document.documentElement);
    const variable=(name,fallback)=>css.getPropertyValue(name).trim() || fallback;
    c.clearRect(0,0,this.width,this.height); c.fillStyle=variable("--soft","#f7fafb"); c.fillRect(0,0,this.width,this.height);
    this.painter(c,this);
    const showAxes=!this.options.hideAxes;
    if (showAxes) {
      const worldStep=80/this.scale, power=Math.pow(10,Math.floor(Math.log10(worldStep)));
      const step=[1,2,5,10].map(v=>v*power).find(v=>v>=worldStep) || 10*power;
      const xmin=this.worldX(0), xmax=this.worldX(this.width), ymin=this.worldY(this.height), ymax=this.worldY(0);
      c.strokeStyle=variable("--grid","#dce8ec"); c.fillStyle=variable("--muted","#1f3e50"); c.lineWidth=1; c.font="10px Segoe UI, Arial, sans-serif";
      const tickText=v=>step<1e-4||step>1e5?v.toExponential(Math.max(3,Math.min(15,1+Math.ceil(Math.log10(Math.max(Math.abs(v),step)/step))))):v.toFixed(Math.max(0,Math.min(15,-Math.floor(Math.log10(step)))));
      const xfirst=Math.ceil(xmin/step), yfirst=Math.ceil(ymin/step);
      for (let k=0;k<1000;k++) {const x=(xfirst+k)*step;if(x>xmax)break;const px=this.x(x); if(state.gridVisible){c.beginPath(); c.moveTo(px,0); c.lineTo(px,this.height); c.stroke();} c.fillText(tickText(x),px+3,this.height-7); }
      for (let k=0;k<1000;k++) {const y=(yfirst+k)*step;if(y>ymax)break;const py=this.y(y); if(state.gridVisible){c.beginPath(); c.moveTo(0,py); c.lineTo(this.width,py); c.stroke();} c.fillText(tickText(y),6,py-4); }
      if(state.gridVisible){c.strokeStyle=variable("--line","#b2c7d0"); c.beginPath(); c.moveTo(this.x(0),0); c.lineTo(this.x(0),this.height); c.moveTo(0,this.y(0)); c.lineTo(this.width,this.y(0)); c.stroke();}
    }
    if (showAxes) { c.fillStyle=variable("--ink","#456174"); c.font="11px Segoe UI, Arial, sans-serif"; c.fillText(this.options.axisUnits==="m"?"x, m":"x/c",this.width-35,this.height-7); c.fillText(this.options.axisUnits==="m"?"y, m":"y/c",7,14); }
    if(state.showAirfoilName&&this.labelGeometry) drawFoilName(c,this,this.labelGeometry);
  }
}
function drawFoil(ctx, plot, xs, ys, fill = null, name = null) {
  if (!xs?.length || xs.length !== ys?.length) return;
  ctx.beginPath(); xs.forEach((x,i) => { const px=plot.x(Number(x)), py=plot.y(Number(ys[i])); if (i) ctx.lineTo(px,py); else ctx.moveTo(px,py); }); ctx.closePath();
  const css=getComputedStyle(document.documentElement);
  ctx.fillStyle=fill || css.getPropertyValue("--foil-fill").trim() || "#ffffff";
  ctx.fill(); ctx.strokeStyle=css.getPropertyValue("--ink").trim() || "#17354b"; ctx.lineWidth=2; ctx.stroke();
  plot.labelGeometry={xs,ys,name:name || (plot.kind==="geometry" ? state.preview?.geometry?.name : state.result?.geometry?.name) || "Airfoil"};
}
function drawFoilName(ctx,plot,{xs,ys,name}){
  let area=0,cx=0,cy=0;
  for(let i=0;i<xs.length;i++){const j=(i+1)%xs.length,q=Number(xs[i])*Number(ys[j])-Number(xs[j])*Number(ys[i]);area+=q;cx+=(Number(xs[i])+Number(xs[j]))*q;cy+=(Number(ys[i])+Number(ys[j]))*q;}
  if(Math.abs(area)<1e-18)return;cx/=3*area;cy/=3*area;
  const x=plot.x(cx),y=plot.y(cy);if(x<0||x>plot.width||y<0||y>plot.height)return;
  ctx.save();ctx.textAlign="center";ctx.textBaseline="middle";ctx.font="600 12px Segoe UI, Arial, sans-serif";ctx.lineWidth=3;ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue("--foil-fill").trim()||"#fff";ctx.strokeText(name,x,y);ctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue("--ink").trim()||"#17354b";ctx.fillText(name,x,y);ctx.restore();
}
function renderGeometry(xs, ys, geometry=null, mesh=null) {
  if (!xs?.length || xs.length !== ys?.length) { spatialViews.geometry?.destroy(); spatialViews.geometry=null; putChart("geometry-chart",null,"Geometry was not returned by the solver."); return; }
  const grid=mesh ? inspectableMesh(mesh) : null;
  const wall=meshWall(grid);
  const contour=wall || {x:xs,y:ys};
  const si=spatialCoordinatesSI(null,grid)||geometry?.units==="m";
  el["geometry-axis-label"].textContent=si?"x, m · y, m":"x/c · y/c";
  const bounds=si?coordinateBounds(grid?meshBlocks(grid):[contour]):{xmin:-0.07,xmax:1.07,ymin:-0.18,ymax:0.18};
  const gurneyBase=!si&&state.preview?.gurney&&state.preview.geometry===geometry;
  new SpatialPlot("geometry",el["geometry-chart"],bounds,(ctx,plot)=>{if(grid)drawMeshLines(ctx,plot,grid);drawFoil(ctx,plot,contour.x,contour.y,null,geometry?.name);const bm=grid?.baffles?grid:mesh;if(Array.isArray(bm?.baffles)&&bm.baffles.length)drawBaffles(ctx,plot,bm);else if(gurneyBase)drawGurneyPlate(ctx,plot,state.preview.gurney);if(!si)window.AirfoilLBM?.drawSection(ctx,plot,state.preview,geometry);if(grid)drawStreamlines(ctx,plot,grid.streamlines?grid:mesh);},null,{axisUnits:si?"m":"chord"});
}
// The Gurney plate (zero thickness, on the trailing-edge station), drawn from the server's plate record over the airfoil.
function drawGurneyPlate(ctx,plot,g){
  const lo=g?.lower_tip,up=g?.upper_tip,lte=g?.lower_te,ute=g?.upper_te;
  if(!Array.isArray(lo)||!Array.isArray(up))return;
  const seg=(p,q)=>{ctx.beginPath();ctx.moveTo(plot.x(Number(p[0])),plot.y(Number(p[1])));ctx.lineTo(plot.x(Number(q[0])),plot.y(Number(q[1])));ctx.stroke();};
  ctx.save();ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue("--orange").trim()||"#df713e";ctx.lineWidth=4;ctx.lineCap="butt";
  if(Number(g.plate_below_lower_te)>0&&Array.isArray(lte))seg(lte,lo);
  if(Number(g.plate_above_upper_te)>0&&Array.isArray(ute))seg(ute,up);
  if(Number(g.plate_below_lower_te)>0&&Number(g.plate_above_upper_te)>0&&Array.isArray(lte)&&Array.isArray(ute))seg(lte,ute);
  ctx.restore();
}
// The Gurney baffle(s) of a mesh (mesh.baffles, from the solver's mesh: the grid line from the trailing-edge corner after the
// plate's straightening and snap, root to tip), drawn as the plate: orange, over the mesh or the field; the tip is marked.
function drawBaffles(ctx,plot,mesh){
  const list=mesh?.baffles;if(!Array.isArray(list)||!list.length)return;
  ctx.save();ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue("--orange").trim()||"#df713e";ctx.lineWidth=3;ctx.lineCap="butt";
  for(const b of list){
    const xs=b?.x,ys=b?.y;if(!Array.isArray(xs)||xs.length<2||xs.length!==ys?.length)continue;
    ctx.beginPath();for(let i=0;i<xs.length;i++){const x=plot.x(Number(xs[i])),y=plot.y(Number(ys[i]));if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);}ctx.stroke();
    const tip=b.tip;if(Array.isArray(tip)){ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();ctx.arc(plot.x(Number(tip[0])),plot.y(Number(tip[1])),3,0,Math.PI*2);ctx.fill();}
  }
  ctx.restore();
}
// Keep the source contour open at the finite trailing-edge base. A periodic
// tangent there would mistake the flat base for a smooth continuation.
function curvaturePorcupine(geometry, requestedScale = null, mapping = "linear") {
  const xs=geometry?.x?.map(Number), ys=geometry?.y?.map(Number), values=geometry?.curvature;
  if (!xs || !ys || xs.length !== ys.length || xs.length < 3 || !Array.isArray(values) || values.length !== xs.length) return null;
  const count=xs.length, kappas=values.map(v=>Number(typeof v==="object" ? v.curvature ?? v.kappa ?? v.y : v));
  if (![...xs,...ys,...kappas].every(finite)) return null;
  let area=0;
  for(let i=0;i<count;i++){const j=(i+1)%count;area+=xs[i]*ys[j]-xs[j]*ys[i];}
  if (Math.abs(area)<1e-18) return null;
  const winding=Math.sign(area), chord=Math.max(...xs)-Math.min(...xs);
  // Reference curvature is one inverse chord. log1p is finite at zero, has
  // the same local slope as linear, and preserves signed concave regions.
  const referenceKappa=1/chord, logarithmic=mapping==="log";
  const mappedKappas=kappas.map(value=>logarithmic ? Math.sign(value)*referenceKappa*Math.log1p(Math.abs(value)/referenceKappa) : value);
  const maxKappa=Math.max(...mappedKappas.map(Math.abs));
  const automaticScale=maxKappa>0 ? 0.22*chord/maxKappa : 1;
  const scale=finite(requestedScale)&&requestedScale>=0 ? requestedScale : automaticScale;
  const secant=(i,j)=>{const h=Math.hypot(xs[j]-xs[i],ys[j]-ys[i]);return {x:(xs[j]-xs[i])/Math.max(h,1e-30),y:(ys[j]-ys[i])/Math.max(h,1e-30),h};};
  const points=xs.map((x,i)=>{
    let dx,dy;
    if(i===0){const a=secant(0,1),b=secant(1,2),sum=a.h+b.h;dx=((2*a.h+b.h)*a.x-a.h*b.x)/sum;dy=((2*a.h+b.h)*a.y-a.h*b.y)/sum;}
    else if(i===count-1){const a=secant(i-2,i-1),b=secant(i-1,i),sum=a.h+b.h;dx=((a.h+2*b.h)*b.x-b.h*a.x)/sum;dy=((a.h+2*b.h)*b.y-b.h*a.y)/sum;}
    else {const a=secant(i-1,i),b=secant(i,i+1),sum=a.h+b.h;dx=(b.h*a.x+a.h*b.x)/sum;dy=(b.h*a.y+a.h*b.y)/sum;}
    const length=Math.hypot(dx,dy), nx=winding*dy/length,ny=-winding*dx/length;
    // Convex curvature points outward for either contour orientation;
    // concave portions point inward. Reversing node order changes neither.
    const kappa=winding*kappas[i],mappedKappa=winding*mappedKappas[i],distance=scale*mappedKappa;
    return {x,y:ys[i],nx,ny,kappa,sourceKappa:kappas[i],mappedKappa,distance,ex:x+distance*nx,ey:ys[i]+distance*ny};
  });
  return points.every(p=>Object.values(p).every(finite)) ? {points,scale,automaticScale,chord,mapping:logarithmic ? "log" : "linear",referenceKappa} : null;
}
function renderPorcupine(geometry) {
  const input=el["curvature-scale"].value.trim(), requested=input==="" ? null : Number(input);
  const data=curvaturePorcupine(geometry,requested,state.curvatureMapping);
  if(!data){spatialViews.curvature?.destroy();spatialViews.curvature=null;putChart("curvature-chart",null,"No conditioned curvature samples were returned.");return;}
  el["curvature-scale"].placeholder=`Auto ${data.automaticScale.toPrecision(3)}`;
  el["curvature-scale"].setAttribute("aria-invalid", String(input!==""&&(!finite(requested)||requested<0)));
  const {points,scale,chord}=data, xmin=Math.min(...points.flatMap(p=>[p.x,p.ex])),xmax=Math.max(...points.flatMap(p=>[p.x,p.ex]));
  const ymin=Math.min(...points.flatMap(p=>[p.y,p.ey])),ymax=Math.max(...points.flatMap(p=>[p.y,p.ey])),pad=.05*Math.max(chord,1e-6);
  const nearest=(x,y)=>{let best=null,d=Infinity;for(const p of points){const q=(p.x-x)**2+(p.y-y)**2;if(q<d){d=q;best=p;}}return `${geometry?.units==="m"?"x, m":"x/c"} ${numberText(best.x,5)} · raw κc ${numberText(best.kappa*chord,4)} · Δ/c ${numberText(best.distance/chord,5)} · ${data.mapping} scale ${scale.toPrecision(4)}`;};
  const si=geometry?.units==="m";el["geometry-axis-label"].textContent=si?"x, m · y, m":"x/c · y/c";
  const plot=new SpatialPlot("curvature",el["curvature-chart"],{xmin:xmin-pad,xmax:xmax+pad,ymin:ymin-pad,ymax:ymax+pad},(ctx,view)=>{
    const css=getComputedStyle(document.documentElement),magenta=css.getPropertyValue("--curvature-line").trim()||"#d318ae";
    // Draw the body first so signed inward rays remain visible in concavities.
    drawFoil(ctx,view,geometry.x,geometry.y,null,geometry.name);
    ctx.save();ctx.strokeStyle=magenta;ctx.globalAlpha=.40;ctx.lineWidth=.65;ctx.beginPath();
    for(const p of points){ctx.moveTo(view.x(p.x),view.y(p.y));ctx.lineTo(view.x(p.ex),view.y(p.ey));}
    ctx.stroke();ctx.globalAlpha=1;ctx.lineWidth=1.6;ctx.beginPath();
    points.forEach((p,i)=>{if(i)ctx.lineTo(view.x(p.ex),view.y(p.ey));else ctx.moveTo(view.x(p.ex),view.y(p.ey));});ctx.stroke();ctx.restore();
  },nearest,{axisUnits:si?"m":"chord"});
  plot.porcupine=data;
  const mappingLabel=data.mapping==="log" ? "signed log1p(|κc|)" : "signed κc";
  el["curvature-scale-field"].title=`Normal distance = scale × ${data.mapping==="log" ? "sign(κ) × κref × log1p(|κ| / κref), with κref = 1/chord" : "signed curvature"}. Leave blank to fit the longest ray to 0.22 chord.`;
  el["geometry-status"].textContent=`${geometry.x.length} source-node rays · magenta ${mappingLabel} · scale ${scale.toPrecision(4)} · convex outward, concave inward`;
}
function currentResult() {
  const results = state.result?.results || [];
  const backend = el["result-backend"].value;
  const alpha = Number(el["result-alpha"].value);
  return results.find(r => r.backend === backend && Number(r.alpha_deg) === alpha) || results.find(r => r.backend === backend) || results[0] || null;
}
function currentMesh(point = currentResult()) {
  if (validMesh(point?.mesh)) return point.mesh;
  const meshes = state.result?.meshes || {};
  const alpha = point?.alpha_deg ?? state.activeCfdAlpha ?? state.requestedAlpha;
  const key = Number(alpha).toFixed(6);
  return meshes[key] || Object.values(meshes)[0] || (!state.result ? state.preview?.mesh : null) || null;
}
function inspectableMesh(preview) {
  const url = safeHref(preview?.artifact);
  if (!url || !validMesh(preview)) return preview;
  const loaded = state.fullMeshes.get(url);
  if (loaded) return loaded;
  if (!state.meshLoading.has(url)) {
    state.meshLoading.add(url);
    const done=beginOperation(`mesh:${url}`, "Loading full mesh…");
    request(url).then(full => {
      if (!fullMeshMatchesPreview(full,preview)) throw new Error("Full mesh blocks have invalid dimensions or missing connections");
      full.artifact = url;
      state.fullMeshes.set(url, full);
      if (el["field-kind"].value === "mesh:grid" && state.activeTab === "field") renderField();
      if((state.preview?.mesh?.artifact===url||currentMesh()?.artifact===url)&&el["geometry-display"].value==="mesh")renderSelectedGeometry();
      if(state.meshOverlay&&state.activeTab==="field")renderField();
    }).catch(err => { el["field-legend"].title = `Full mesh unavailable: ${err.message}`; })
      .finally(() => {state.meshLoading.delete(url);done();});
  }
  return preview;
}
// A completed CFD point may publish a separate full-resolution field artifact.
// The small field in the job result remains the live preview while it loads.
function validFullField(field) {
  if (field?.kind === "structured_blocks") {
    const blocks=field.blocks;
    return Array.isArray(blocks) && blocks.length>1 && blocks.length<=32 &&
      new Set(blocks.map(block=>block?.id)).size===blocks.length &&
      blocks.every(block=>typeof block?.id==="string" && /^[a-zA-Z0-9_-]+$/.test(block.id) &&
        block.kind==="structured_cells" && validFullField(block) &&
        block.iteration===field.iteration && block.mesh_sha256===field.mesh_sha256 &&
        block.cell_order===field.cell_order);
  }
  const ni = Number(field?.ni), nj = Number(field?.nj), samples = field?.samples;
  return field?.kind === "structured_cells" && Number.isInteger(ni) && Number.isInteger(nj) &&
    field?.cell_order === "j-major, i-fastest" &&
    Number.isInteger(field?.iteration) && field.iteration >= 1 &&
    /^[0-9a-f]{64}$/i.test(field?.mesh_sha256 || "") &&
    ni > 1 && nj > 1 && samples && Array.isArray(samples.x) && Array.isArray(samples.y) &&
    samples.x.length === ni * nj && samples.y.length === ni * nj &&
    samples.x.every(finite) && samples.y.every(finite) &&
    Object.entries(samples).every(([key, values]) => key === "x" || key === "y" ||
      (Array.isArray(values) && values.length === ni * nj && values.every(finite)));
}
async function loadFullFieldArtifact(url, provided = null, nested = false) {
  const field = provided || await request(url);
  if (field?.kind === "structured_blocks") {
    if (nested || !Array.isArray(field.blocks) || field.blocks.length<2 || field.blocks.length>32 ||
        !Number.isSafeInteger(field.iteration) || field.iteration<1 ||
        !/^[0-9a-f]{64}$/i.test(field.mesh_sha256 || "") || field.cell_order!=="j-major, i-fastest" ||
        new Set(field.blocks.map(block=>block?.id)).size!==field.blocks.length)
      throw new Error("Full CFD block field has invalid dimensions or provenance");
    const source=new URL(url,location.href),directory=source.pathname.slice(0,source.pathname.lastIndexOf("/")+1);
    const blocks=[];
    for (const block of field.blocks) {
      if (typeof block?.id!=="string" || !/^[a-zA-Z0-9_-]+$/.test(block.id))
        throw new Error("Full CFD field has an invalid block identity");
      let full;
      if (block.artifact) {
        if (typeof block.artifact!=="string" || !/^[a-zA-Z0-9_.-]+\.msgpack$/.test(block.artifact) || block.artifact.includes(".."))
          throw new Error("Full CFD field block has an invalid artifact path");
        const blockUrl=new URL(block.artifact,source);
        if(blockUrl.origin!==source.origin || blockUrl.pathname.slice(0,blockUrl.pathname.lastIndexOf("/")+1)!==directory)
          throw new Error("Full CFD field block is outside its artifact directory");
        full=await loadFullFieldArtifact(blockUrl.href,null,true);
      } else full=await loadFullFieldArtifact(url,block,true);
      if (full?.kind!=="structured_cells" || !validFullField(full) ||
          full.iteration!==field.iteration || full.mesh_sha256!==field.mesh_sha256 ||
          full.cell_order!==field.cell_order || (full.id!=null && full.id!==block.id) ||
          (block.ni!=null && Number(block.ni)!==full.ni) || (block.nj!=null && Number(block.nj)!==full.nj))
        throw new Error("Full CFD field block belongs to a different grid or iteration");
      blocks.push({...full,id:block.id,sample_shape:[full.ni,full.nj]});
    }
    const columns=Object.keys(blocks[0].samples).sort().join(",");
    if(blocks.some(block=>Object.keys(block.samples).sort().join(",")!==columns))
      throw new Error("Full CFD field blocks have different quantity columns");
    return {...field,blocks};
  }
  if (field?.kind !== "structured_cells_manifest") return field;
  const ni = Number(field.ni), nj = Number(field.nj), count = ni * nj;
  const columns = field.columns, parts = field.parts;
  if (!Number.isSafeInteger(ni) || !Number.isSafeInteger(nj) || ni < 2 || nj < 2 ||
      !Number.isSafeInteger(count) || count > 0xffffffff ||
      field.cell_order !== "j-major, i-fastest" ||
      !Number.isSafeInteger(field.iteration) || field.iteration < 1 ||
      !/^[0-9a-f]{64}$/i.test(field.mesh_sha256 || "") ||
      !Array.isArray(columns) || columns.length < 2 || columns.length > 32 ||
      new Set(columns).size !== columns.length || !columns.includes("x") || !columns.includes("y") ||
      !columns.every(key => typeof key === "string" && /^[a-z][a-z0-9_]*$/.test(key)) ||
      !Array.isArray(parts) || !parts.length || parts.length > nj)
    throw new Error("Full CFD field manifest has invalid dimensions or provenance");
  const source = new URL(url, location.href), directory = source.pathname.slice(0, source.pathname.lastIndexOf("/") + 1);
  const samples = Object.fromEntries(columns.map(key => [key, new Array(count)]));
  let nextRow = 1;
  for (const part of parts) {
    const start = Number(part.j_start), end = Number(part.j_end);
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start !== nextRow || end < start || end > nj ||
        typeof part.path !== "string" || !/^[a-zA-Z0-9_.-]+\.msgpack$/.test(part.path) || part.path.includes(".."))
      throw new Error("Full CFD field chunks overlap, have gaps, or have an invalid path");
    const partUrl = new URL(part.path, source);
    if (partUrl.origin !== source.origin || partUrl.pathname.slice(0, partUrl.pathname.lastIndexOf("/") + 1) !== directory)
      throw new Error("Full CFD field chunk is outside its artifact directory");
    const chunk = await request(partUrl.href);
    const rows = end - start + 1, chunkCount = ni * rows;
    if (chunk?.kind !== "structured_cells_part" || Number(chunk.ni) !== ni || Number(chunk.nj) !== rows ||
        Number(chunk.full_nj) !== nj || Number(chunk.j_start) !== start || Number(chunk.j_end) !== end ||
        chunk.cell_order !== field.cell_order || chunk.iteration !== field.iteration ||
        chunk.mesh_sha256 !== field.mesh_sha256 || !chunk.samples ||
        Object.keys(chunk.samples).length !== columns.length)
      throw new Error("Full CFD field chunk belongs to a different grid, iteration, or row range");
    const offset = ni * (start - 1);
    for (const key of columns) {
      const values = chunk.samples[key];
      if (!Array.isArray(values) || values.length !== chunkCount || !values.every(finite))
        throw new Error(`Full CFD field chunk has invalid ${key} values`);
      for (let i = 0; i < chunkCount; i++) samples[key][offset + i] = values[i];
    }
    nextRow = end + 1;
  }
  if (nextRow !== nj + 1) throw new Error("Full CFD field chunks do not cover every row");
  return { kind: "structured_cells", id:field.id, ni, nj, samples, cell_order: field.cell_order,
    iteration: field.iteration, mesh_sha256: field.mesh_sha256 };
}
function fieldBlocks(fields) {
  if(fields?.kind!=="structured_blocks")return fields?.samples?[fields]:[];
  const blocks=fields.blocks;
  return Array.isArray(blocks)&&blocks.length>1&&blocks.length<=32&&
    new Set(blocks.map(block=>block?.id)).size===blocks.length&&
    blocks.every(block=>typeof block?.id==="string"&&/^[a-zA-Z0-9_-]+$/.test(block.id)&&
      (fields.iteration==null||block.iteration===fields.iteration)&&
      (fields.mesh_sha256==null||block.mesh_sha256===fields.mesh_sha256))?blocks:[];
}
function fieldCellCount(fields) {
  return fieldBlocks(fields).reduce((count,block)=>count+(block.samples?.x?.length||0),0);
}
function fieldQuantities(fields) {
  const blocks=fieldBlocks(fields);
  if(!blocks.length)return [];
  return Object.keys(blocks[0].samples||{}).filter(key=>!["x","y"].includes(key) &&
    blocks.every(block=>Array.isArray(block.samples?.[key]) && block.samples[key].length>0 &&
      block.samples[key].length===block.samples?.x?.length));
}
function inspectableFields(point) {
  const preview = point?.fields;
  const url = safeHref(preview?.full_artifact);
  if (!url) return { fields: preview, full: false };
  const loaded = state.fullFields.get(url);
  if (loaded) return { fields: loaded.kind==="structured_blocks" ? {...preview,...loaded,sample_stride:1} : { ...preview, samples: loaded.samples,
    sample_shape: [loaded.ni, loaded.nj], sample_stride: 1 }, full: true };
  if (!state.fieldLoading.has(url)) {
    state.fieldLoading.add(url);
    const done=beginOperation(`field:${url}`, "Loading full CFD field…");
    loadFullFieldArtifact(url).then(full => {
      if (!validFullField(full)) throw new Error("Full CFD field has invalid dimensions");
      if (preview.full_cell_count != null && Number(preview.full_cell_count) !== fieldCellCount(full))
        throw new Error("Full CFD field cell count differs from point metadata");
      if(preview.kind==="structured_blocks" && (full.kind!=="structured_blocks" ||
          fieldBlocks(preview).length!==fieldBlocks(full).length || fieldBlocks(preview).some(block=>
            !fieldBlocks(full).some(actual=>actual.id===block.id &&
              (block.full_ni==null || Number(block.full_ni)===actual.ni) &&
              (block.full_nj==null || Number(block.full_nj)===actual.nj)))))
        throw new Error("Full CFD field blocks differ from point metadata");
      if (preview.mesh_sha256 && full.mesh_sha256 !== preview.mesh_sha256)
        throw new Error("Full CFD field belongs to a different mesh");
      if (preview.full_iteration != null && Number(preview.full_iteration) !== full.iteration)
        throw new Error("Full CFD field belongs to a different final iteration");
      while (state.fullFields.size >= 3) state.fullFields.delete(state.fullFields.keys().next().value);
      state.fullFields.set(url, full);
      if (state.activeTab === "field" &&
          safeHref(currentResult()?.fields?.full_artifact) === url) renderField();
    }).catch(err => { el["field-legend"].title = `Full CFD field unavailable: ${err.message}`; })
      .finally(() => {state.fieldLoading.delete(url);done();});
  }
  return { fields: preview, full: false };
}
function renderSelection() {
  const results = state.result?.results || [];
  const backends = [...new Set(results.map(r => r.backend))];
  const priorBackend = el["result-backend"].value;
  el["result-backend"].replaceChildren(...backends.map(b => { const o = document.createElement("option"); o.value = b; o.textContent = labelBackend(b); return o; }));
  if (backends.includes(priorBackend)) el["result-backend"].value = priorBackend;
  el["result-backend"].disabled = !backends.length;
  renderAlphaChoices();
}
function renderAlphaChoices() {
  const backend = el["result-backend"].value;
  const results = (state.result?.results || []).filter(r => r.backend === backend);
  const prior = el["result-alpha"].value;
  el["result-alpha"].replaceChildren(...results.map(r => { const o = document.createElement("option"); o.value = String(r.alpha_deg); o.textContent = `${numberText(Number(r.alpha_deg), 2)}°`; return o; }));
  if (results.some(r => String(r.alpha_deg) === prior)) el["result-alpha"].value = prior;
  else if (results.length) {
    const target = Number(el.alpha.value);
    const closest = results.reduce((a, b) => Math.abs(Number(b.alpha_deg) - target) < Math.abs(Number(a.alpha_deg) - target) ? b : a);
    el["result-alpha"].value = String(closest.alpha_deg);
  }
  el["result-alpha"].disabled = !results.length;
  renderPoint();
}
function matchingReference(point) {
  if(point?.reference_compatible===false)return null;
  const reference = state.result?.reference;
  if (!el["show-reference"].checked || !reference || !Array.isArray(reference.points)) return null;
  const target = Number(point?.alpha_deg ?? state.requestedAlpha);
  const tolerance = Number(reference.conditions?.alpha_tolerance_deg ?? 0.005);
  const match = reference.points.find(p => Math.abs(Number(p.alpha_deg) - target) <= tolerance);
  return match ? { reference, match } : null;
}
function renderPoint() {
  const result = state.result;
  const point = currentResult();
  renderSelectedGeometry();
  const cp = point?.cp || {};
  const cpPoint = p => Array.isArray(p) ? { x: Number(p[0]), y: Number(p[1]) } : { x: Number(p.x), y: Number(p.cp) };
  const cpSeries = [
    { name: "Upper surface", color: colors.jxfoil, points: (cp.upper || []).map(cpPoint) },
    { name: "Lower surface", color: colors.transonic, points: (cp.lower || []).map(cpPoint) }
  ];
  if(Array.isArray(point?.base_cp)&&point.base_cp.length){
    const xte=Math.max(...[...(cp.upper||[]),...(cp.lower||[])].map(p=>cpPoint(p).x).filter(finite),1);
    cpSeries.push({ name: point?.baffle ? "Base (at the TE station)" : point?.geometry?.gurney ? "Base + Gurney plate back face (at the TE station)" : "Base (at the TE station)", color: "#c2185b", points: point.base_cp.map(p=>({x:xte,y:Number(p[1])})), connect: false });
  }
  if(point?.baffle_cp){      // the Gurney baffle: its front (upstream, foil-side) and back (downstream, wake-side) faces, at the TE station
    const xte=Math.max(...[...(cp.upper||[]),...(cp.lower||[])].map(p=>cpPoint(p).x).filter(finite),1);
    for(const [plate,sides] of Object.entries(point.baffle_cp))for(const [side,rows] of Object.entries(sides)){
      if(!Array.isArray(rows)||!rows.length)continue;
      cpSeries.push({ name: `Gurney plate ${plate}, ${side==="front"?"front (upstream) face":"back (downstream) face"}`, color: side==="front"?"#e0a000":"#7a3b00", points: rows.map(r=>({x:xte,y:Number(r[1])})).filter(q=>finite(q.y)), connect: false });
    }
  }
  const referenceMatch = matchingReference(point);
  if (referenceMatch) {
    const refCp = referenceMatch.match.cp || {};
    cpSeries.push(
      { name: "Experimental upper", color: "#7555ac", points: (refCp.upper || []).map(cpPoint), dash: "6 3" },
      { name: "Experimental lower", color: "#527f44", points: (refCp.lower || []).map(cpPoint), dash: "6 3" }
    );
  }
  putChart("cp-chart", polyline(cpSeries, { invertY: true, title: "Surface pressure coefficient" }), "Surface pressure is unavailable for this point.");
  const refNote = el["reference-note"];
  if (refNote) {
    refNote.classList.toggle("hidden", !referenceMatch);
    if (referenceMatch) {
      const {reference} = referenceMatch;
      refNote.replaceChildren(document.createTextNode(`${reference.name}. ${reference.note || ""} `));
      const href = safeHref(reference.source_url);
      if (href) {
        const link = document.createElement("a");
        link.href = href; link.target = "_blank"; link.rel = "noopener noreferrer";
        link.textContent = "Source"; refNote.append(link);
      }
      const forceHref = safeHref(reference.force_source_url);
      if (forceHref) {
        refNote.append(document.createTextNode(" / "));
        const forceLink = document.createElement("a");
        forceLink.href = forceHref; forceLink.target = "_blank"; forceLink.rel = "noopener noreferrer";
        forceLink.textContent = "Experimental forces"; refNote.append(forceLink);
      }
    } else refNote.replaceChildren();
  }
  renderIterationHistory(point);
  renderFieldChoices(point);refreshPointOverlays();
}
function fieldValues(field) {
  if (Array.isArray(field)) return field;
  return field?.values || field?.data || field?.grid || null;
}
function renderFieldChoices(point) {
  if (window.AirfoilLBM?.fieldChoices(point)) return;   // LBM 3D: mid-plane frames, video and lattice (lbm.js)
  const fields = point?.fields || {};
  const choices = [];
  if (currentMesh(point)) choices.push(["mesh:grid", "Mesh"]);
  if (["samples","structured_cells","structured_blocks"].includes(fields.kind)) {
    const quantities=fieldQuantities(fields);
    // A backend may declare per-quantity labels, constant columns to hide, and the column that already holds the speed
    // (the low-speed CFD's |U|/U∞ in the "mach" column); without them the choices are the established ones.
    const hidden=Array.isArray(fields.hidden_quantities)?fields.hidden_quantities:[],speedColumn=typeof fields.speed_column==="string"&&quantities.includes(fields.speed_column)?fields.speed_column:null;
    for (const key of quantities) if (availableFieldQuantity(point,key)&&!hidden.includes(key)&&key!==speedColumn) choices.push([`sample:${key}`, quantityLabel(point,key)]);
    if (speedColumn) choices.unshift([`sample:${speedColumn}`, quantityLabel(point,speedColumn)]);
    else if (quantities.includes("u") && quantities.includes("v") && !quantities.includes("speed")) choices.unshift(["sample:speed", "Speed"]);
  }
  if (fields.images) for (const key of Object.keys(fields.images)) if (availableFieldQuantity(point,key)&&safeHref(fields.images[key])) choices.push([`image:${key}`, fieldNames[key] || key.replaceAll("_", " ")]);
  if (point?.boundary_layer) for (const key of ["ue", "dstar", "theta", "cf", "h", "n_factor"]) {
    if (point.boundary_layer.upper?.some(p => finite(p[key])) || point.boundary_layer.lower?.some(p => finite(p[key]))) choices.push([`surface:${key}`, ({ue:"Edge speed",dstar:"Displacement thickness",theta:"Momentum thickness",cf:"Skin friction",h:"Shape factor",n_factor:"N factor"})[key]]);
  }
  if (!choices.length) for (const key of Object.keys(fields)) if (availableFieldQuantity(point,key)&&fieldValues(fields[key])) choices.push([`grid:${key}`, fieldNames[key] || key.replaceAll("_", " ")]);
  if (liveFieldApplies(point)) {
    // The running low-speed point's live field (speed first): replaced by the final field when the point completes.
    const lp=livePoint(),order=["mach","vorticity","cp","pressure","u","v","nut"];
    choices.unshift(...order.filter(key=>Array.isArray(lp.fields.samples?.[key])).map(key=>[`live:${key}`,`Live α ${numberText(lp.alpha_deg,2)}° · step ${state.liveField.iteration} · ${lowspeedFieldMeta.quantity_labels[key]}`]));
  }
  const keys = choices.map(c => c[0]);
  const prior = el["field-kind"].value;
  el["field-kind"].replaceChildren(...choices.map(([key, name]) => { const o = document.createElement("option"); o.value = key; o.textContent = name; return o; }));
  if (keys.includes(prior)) el["field-kind"].value = prior;
  el["field-kind"].disabled = !keys.length;
  renderField(point);
}
function validMesh(mesh) {
  if(!["structured","structured_blocks"].includes(mesh?.kind))return false;
  const blocks=meshBlocks(mesh);
  return blocks.length>0&&blocks.every(block=>{
    const ni=Number(block?.ni),nj=Number(block?.nj);
    return Number.isInteger(ni)&&Number.isInteger(nj)&&ni>1&&nj>1&&Array.isArray(block.x)&&Array.isArray(block.y)&&block.x.length===ni*nj&&block.y.length===ni*nj;
  });
}
function meshBlocks(mesh){return Array.isArray(mesh?.blocks)?mesh.blocks:mesh?[mesh]:[];}
function meshCellCount(mesh,full=false){
  return meshBlocks(mesh).reduce((sum,b)=>sum+(Number(full?(b.full_ni||b.ni):b.ni)-1)*(Number(full?(b.full_nj||b.nj):b.nj)-1),0);
}
function fullMeshMatchesPreview(full,preview){
  if(!validMesh(full)||!validMesh(preview))return false;
  const blocks=meshBlocks(full),prior=meshBlocks(preview);
  return blocks.length===prior.length&&prior.every((p,i)=>{
    const b=p.id?blocks.find(block=>block.id===p.id):blocks[i];
    return b&&Number(b.ni)===Number(p.full_ni||p.ni)&&Number(b.nj)===Number(p.full_nj||p.nj);
  });
}
function meshIsFull(mesh){return meshBlocks(mesh).every(b=>Number(b.ni)===Number(b.full_ni||b.ni)&&Number(b.nj)===Number(b.full_nj||b.nj));}
function meshDimensionsText(mesh){
  return meshBlocks(mesh).map(b=>`${b.id?`${b.id} `:""}${b.ni} × ${b.nj}${Number(b.full_ni||b.ni)!==Number(b.ni)||Number(b.full_nj||b.nj)!==Number(b.nj)?` (full ${b.full_ni||b.ni} × ${b.full_nj||b.nj})`:""}`).join(" + ");
}
function meshTopologyText(mesh){
  const blocks=meshBlocks(mesh),raw=mesh?.mesh_topology||mesh?.topology||mesh?.te_topology||mesh?.metadata?.topology||mesh?.geometry_metadata?.mesh_topology;
  const names={conformal_C_H_v1:"C+H conformal (finite TE)",sharp_C:"C (sharp TE)",aplus_v1:"legacy A-plus"};
  const topology=names[raw]||(raw?String(raw).replaceAll("_"," "):blocks.length>1?"C+H connected blocks":"");
  const admission=mesh?.admission,passed=typeof mesh?.quality_passed==="boolean"?mesh.quality_passed:typeof admission==="boolean"?admission:admission?.passed;
  const failures=Array.isArray(mesh?.quality_failures)?mesh.quality_failures.join("; "):admission?.reason;
  const supported=mesh?.solver_supported===false?` · solver blocked${mesh.solver_support_message?`: ${mesh.solver_support_message}`:""}`:"";
  const qualification=mesh?.geometry_metadata?.qualification;
  return `${topology?` · topology ${topology}`:""}${streamlineMeshText(mesh)}${passed===true?(Array.isArray(mesh?.quality?.pbcav_gates)?" · PBCav mesh-audit gates passed":" · air geometry checks passed"):passed===false?(Array.isArray(mesh?.quality?.pbcav_gates)?" · PBCav mesh-audit gates FAILED":" · air geometry checks FAILED"):""}${failures?` · ${failures}`:""}${supported}${qualification?` · ${qualification}`:""}`;
}
// The label of a streamline-aligned mesh (development): the JXfoil path, the admission rule and the gates that got worse.
function streamlineMeshText(mesh){
  const sa=mesh?.geometry_metadata?.streamline_alignment;if(!sa?.aligned)return "";
  const adm=sa.admission||{},worse=Array.isArray(adm.gates_worse)?adm.gates_worse:[];
  const src=String(sa.path_source||""),kind=src.startsWith("viscous")?"viscous":src==="inviscid"?"inviscid":"inviscid FALLBACK";
  return ` · STREAMLINE-ALIGNED with JXfoil's ${kind} paths at α ${Number(sa.alpha_deg)}° (${String(adm.rule||"").toUpperCase()} admission${adm.stage&&adm.stage!=="default"?`, ${adm.stage}`:""}; ${worse.length?`gates worse: ${worse.join(", ")}`:"no gate worse"}; development, not validated)`;
}
function spatialCoordinatesSI(point=null,mesh=currentMesh(point)){
  return mesh?.units==="m"||mesh?.coordinate_units==="m"||point?.fields?.coordinate_units==="m"||point?.fields?.units?.x==="m";
}
function coordinateBounds(grids){
  let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
  for(const grid of grids||[])for(let i=0;i<(grid?.x?.length||0);i++){
    const x=Number(grid.x[i]),y=Number(grid.y?.[i]);if(!finite(x)||!finite(y))continue;
    xmin=Math.min(xmin,x);xmax=Math.max(xmax,x);ymin=Math.min(ymin,y);ymax=Math.max(ymax,y);
  }
  if(![xmin,xmax,ymin,ymax].every(finite))return{xmin:-.65,xmax:1.65,ymin:-.58,ymax:.58};
  const pad=.04*Math.max(xmax-xmin,ymax-ymin,1e-6);
  return{xmin:xmin-pad,xmax:xmax+pad,ymin:ymin-pad,ymax:ymax+pad};
}
function spatialCoordinateText(x,y,si=false){return `${si?"x, m":"x/c"} ${numberText(x,3)} · ${si?"y, m":"y/c"} ${numberText(y,3)}`;}
function quantityUnits(point,key,mesh=currentMesh(point)){
  const declared=point?.fields?.quantity_units?.[key]||point?.fields?.units?.[key];if(typeof declared==="string")return declared;
  if(key==="speed"&&point?.fields?.quantity_units?.u===point?.fields?.quantity_units?.v&&typeof point?.fields?.quantity_units?.u==="string")return point.fields.quantity_units.u;
  if(["beta_water","beta_air","beta_vapor","alpha","vapor_fraction","cp","mach"].includes(key))return"dimensionless";
  if(spatialCoordinatesSI(point,mesh))return({pressure:"Pa",dynamic_pressure:"Pa",hydrostatic_pressure:"Pa",density:"kg/m³",speed:"m/s",velocity:"m/s",u:"m/s",ux:"m/s",v:"m/s",temperature:"K",vorticity:"1/s",nut:"m²/s"})[key]||"SI";
  return fieldUnits[key]||"solver units";
}
function meshQualityText(mesh) {
  const quality = mesh?.quality_report||mesh?.quality;
  const q = quality?.near_foil;
  if (!q) return "";
  const n = (v, digits) => finite(Number(v)) ? Number(v).toFixed(digits) : "?";
  const gate = quality.geometric_gate?.passed === true ? " · nonfolding check passed" : quality.geometric_gate?.passed === false ? " · nonfolding check FAILED" : "";
  const advisory = quality.air_reference_gates?.passed === false ? ` · air mesh checks FAILED (${quality.air_reference_gates.failed?.length || 0} limits exceeded)` : quality.air_reference_gates?.passed===true ? " · air mesh checks passed" : "";
  const wake = quality.wake_first_rows;
  const wakeText = wake ? ` · wake first-row AR p95 ${n(wake.first_row_aspect_ratio_p95,1)}` : "";
  const solverAngle = quality.global?.maximum_solver_center_nonorthogonality_deg;
  const solverText = finite(Number(solverAngle)) ? ` · max solver-center nonorthogonality ${n(solverAngle,1)}°` : "";
  return `${gate}${advisory} · near foil p95: AR ${n(q.aspect_ratio_p95,1)}, nonorthogonality ${n(q.face_nonorthogonality_p95_deg,1)}°, skew ${n(q.face_skew_p95,3)}${wakeText}${solverText} · min corner Jacobian ${n(q.minimum_corner_jacobian_ratio,3)}`;
}
function meshLineIndices(count, pixels, zoom, critical = []) {
  // Keep only screen-resolvable lines at overview. Zooming reveals every line
  // of the already loaded full mesh without another server request.
  const stride = Math.max(1, Math.ceil((count-1)*5 / (Math.max(1,pixels)*Math.max(0.15,zoom))));
  const indices = [];
  for (let k=0; k<count; k+=stride) indices.push(k);
  indices.push(count-1);
  for (const k of critical) if (Number.isInteger(k) && k>=0 && k<count) indices.push(k);
  return [...new Set(indices)].sort((a,b)=>a-b);
}
function meshWall(mesh){
  if(Array.isArray(mesh?.wall_x)&&mesh.wall_x.length>2&&mesh.wall_x.length===mesh.wall_y?.length)return{x:mesh.wall_x,y:mesh.wall_y};
  if(!validMesh(mesh))return null;
  // A multiblock wall must include the H-block base as well as C-block foil.
  // Guessing from a C-block index range would display a false sharp closure.
  if(meshBlocks(mesh).length>1)return null;
  const first=Array.isArray(mesh.i_indices)?mesh.i_indices.indexOf(Number(mesh.wall_i_start)):Number(mesh.wall_i_start)-1;
  const last=Array.isArray(mesh.i_indices)?mesh.i_indices.indexOf(Number(mesh.wall_i_end)):Number(mesh.wall_i_end)-1;
  return first>=0&&last>first?{x:mesh.x.slice(first,last+1),y:mesh.y.slice(first,last+1)}:null;
}
function drawMeshLines(ctx,plot,mesh){
  if(!validMesh(mesh))return;
  for(const block of meshBlocks(mesh))drawMeshBlockLines(ctx,plot,block);
}
function drawMeshBlockLines(ctx,plot,mesh){
  const ni=Number(mesh.ni),nj=Number(mesh.nj),coords=(i,j)=>[Number(mesh.x[j*ni+i]),Number(mesh.y[j*ni+i])];
  ctx.lineWidth=0.65;ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue("--mesh-line").trim()||"#4a8ba2";
  const first=Array.isArray(mesh.i_indices)?mesh.i_indices.indexOf(mesh.wall_i_start):Number(mesh.wall_i_start)-1;
  const last=Array.isArray(mesh.i_indices)?mesh.i_indices.indexOf(mesh.wall_i_end):Number(mesh.wall_i_end)-1;
  for(const j of meshLineIndices(nj,plot.height,plot.zoom,[0,1,2,3,nj-1])){ctx.beginPath();for(let i=0;i<ni;i++){const[x,y]=coords(i,j);if(i)ctx.lineTo(plot.x(x),plot.y(y));else ctx.moveTo(plot.x(x),plot.y(y));}ctx.stroke();}
  for(const i of meshLineIndices(ni,plot.width,plot.zoom,[first,last])){ctx.beginPath();for(let j=0;j<nj;j++){const[x,y]=coords(i,j);if(j)ctx.lineTo(plot.x(x),plot.y(y));else ctx.moveTo(plot.x(x),plot.y(y));}ctx.stroke();}
}
function renderMesh2D(mesh) {
  if (!validMesh(mesh)) return false;
  const geometry = state.result?.geometry || state.preview?.geometry;
  const wall=meshWall(mesh),gx=wall?.x||geometry?.x||[],gy=wall?.y||geometry?.y||[];
  const si=spatialCoordinatesSI(null,mesh),bounds=si?coordinateBounds(meshBlocks(mesh)):{xmin:-0.65,xmax:1.65,ymin:-0.58,ymax:0.58};
  new SpatialPlot("field", el["field-view"], bounds, (ctx, plot) => {
    drawMeshLines(ctx,plot,mesh);
    if (gx.length > 1 && gx.length === gy.length) drawFoil(ctx,plot,gx,gy,null,geometry?.name);
    drawStreamlines(ctx,plot,mesh?.streamlines?mesh:currentMesh());
    drawBaffles(ctx,plot,mesh?.baffles?mesh:currentMesh());
  }, (x,y) => `${meshIsFull(mesh) ? "Full mesh" : "Mesh preview"}${meshTopologyText(mesh)} · ${spatialCoordinateText(x,y,si)}`,{axisUnits:si?"m":"chord"});
  el["field-legend"].classList.remove("hidden");
  el["field-legend"].replaceChildren(document.createTextNode(`${meshIsFull(mesh) ? "Full mesh" : "Mesh preview"} ${meshDimensionsText(mesh)} nodes${meshTopologyText(mesh)}${si?" · coordinates in metres":""} · adaptive line spacing; zoom to inspect every line · available while CFD runs${meshQualityText(mesh)}`));
  const href = safeHref(mesh.artifact || mesh.preview_artifact);
  if (href) { const a=document.createElement("a"); a.href=href; a.textContent="Download full mesh"; a.target="_blank"; a.rel="noopener noreferrer"; el["field-legend"].append(document.createTextNode(" · "), a); }
  return true;
}
function fieldGrid(field) {
  const raw = fieldValues(field);
  if (!raw) return null;
  let nx = Number(field?.nx || field?.width || 0), ny = Number(field?.ny || field?.height || 0), values;
  if (Array.isArray(raw) && Array.isArray(raw[0])) { ny = raw.length; nx = raw[0].length; values = raw.flat(); }
  else values = Array.from(raw);
  if (!nx || !ny || values.length !== nx * ny) return null;
  return { nx, ny, values: values.map(Number), x: field?.x || null, y: field?.y || null };
}
function ramp(t) {
  if(window.LudwigPalettes)return window.LudwigPalettes.sample(t).map(v=>Math.round(v*255));
  const stops = [[23,62,114],[52,124,187],[91,192,201],[237,212,107],[233,139,77],[155,56,77]];
  t = Math.max(0, Math.min(1, t));
  const a = Math.min(stops.length - 2, Math.floor(t * (stops.length - 1)));
  const k = Math.max(0, Math.min(1, t * (stops.length - 1) - a));
  return stops[a].map((v, i) => Math.round(v * (1 - k) + stops[a+1][i] * k));
}
// Rasterize the sampled curvilinear cells with linear scalar interpolation.
// Each pixel comes from an actual neighboring sample triangle; no synthetic
// field detail is introduced between the CFD samples. Folded cells are blank.
function rasterizeStructuredSamples(pts, ni, nj, plot, vmin, vmax, pixelRatio = 1) {
  const width = Math.max(1, Math.round(plot.width * pixelRatio));
  const height = Math.max(1, Math.round(plot.height * pixelRatio));
  const pixels = new Uint8ClampedArray(width * height * 4);
  const values = new Float32Array(width * height);
  values.fill(NaN);
  const palette = Array.from({ length: 256 }, (_, i) => ramp(i / 255));
  const span = vmax - vmin || 1;
  const vertices = pts.map(p => ({
    x: plot.x(p.x) * pixelRatio, y: plot.y(p.y) * pixelRatio, v: p.v,
    wx: p.x, wy: p.y
  }));
  let drawnCells = 0, rejectedCells = 0;
  const signed = (a, b, c) => (b.wx-a.wx)*(c.wy-a.wy) - (b.wy-a.wy)*(c.wx-a.wx);
  function triangle(a, b, c) {
    const minx = Math.max(0, Math.floor(Math.min(a.x,b.x,c.x)));
    const maxx = Math.min(width-1, Math.ceil(Math.max(a.x,b.x,c.x)));
    const miny = Math.max(0, Math.floor(Math.min(a.y,b.y,c.y)));
    const maxy = Math.min(height-1, Math.ceil(Math.max(a.y,b.y,c.y)));
    if (maxx < minx || maxy < miny) return;
    const abx = b.x-a.x, aby = b.y-a.y, acx = c.x-a.x, acy = c.y-a.y;
    const area = abx*acy - aby*acx;
    if (!Number.isFinite(area) || Math.abs(area) < 1e-12) return;
    const inv = 1/area, dvb = b.v-a.v, dvc = c.v-a.v;
    for (let y=miny; y<=maxy; y++) {
      const dy = y+0.5-a.y;
      for (let x=minx; x<=maxx; x++) {
        const dx = x+0.5-a.x;
        const wb = (dx*acy-dy*acx)*inv;
        const wc = (abx*dy-aby*dx)*inv;
        if (wb < -1e-9 || wc < -1e-9 || wb+wc > 1+1e-9) continue;
        const value = a.v + wb*dvb + wc*dvc;
        const t = (value - vmin)/span;
        const color = palette[Math.max(0,Math.min(255,Math.round(t*255)))];
        const offset = (y*width+x)*4;
        pixels[offset] = color[0]; pixels[offset+1] = color[1];
        pixels[offset+2] = color[2]; pixels[offset+3] = 255;
        values[y*width+x] = value;
      }
    }
  }
  for (let j=0; j<nj-1; j++) for (let i=0; i<ni-1; i++) {
    const a=vertices[j*ni+i], b=vertices[j*ni+i+1];
    const d=vertices[(j+1)*ni+i], c=vertices[(j+1)*ni+i+1];
    const area1=signed(a,b,c), area2=signed(a,c,d);
    if (!Number.isFinite(area1) || !Number.isFinite(area2) ||
        Math.abs(area1)<1e-14 || Math.abs(area2)<1e-14 || area1*area2<=0) {
      rejectedCells++; continue;
    }
    drawnCells++;
    triangle(a,b,c); triangle(a,c,d);
  }
  // Cell centers on opposite sides of the C-grid wake cut leave occasional
  // one-pixel holes. Fill only fully enclosed pixels, so the far-field edge
  // and other truly uncovered regions remain transparent.
  const isolated = [];
  for (let y=1; y<height-1; y++) for (let x=1; x<width-1; x++) {
    const p=y*width+x;
    if (pixels[4*p+3]!==0) continue;
    const neighbors=[p-1,p+1,p-width,p+width];
    if (neighbors.every(k=>pixels[4*k+3]===255))
      isolated.push([p,neighbors]);
  }
  for (const [p,neighbors] of isolated) {
    const value=neighbors.reduce((sum,k)=>sum+values[k],0)/4;
    const rgb=palette[Math.max(0,Math.min(255,Math.round((value-vmin)/span*255)))];
    values[p]=value;
    pixels.set([...rgb,255],4*p);
  }
  return { width, height, pixels, values, drawnCells, rejectedCells };
}
function safeHref(url) {
  if (typeof url !== "string") return null;
  if (url.startsWith("/") && !url.startsWith("//")) return url;
  try { const parsed = new URL(url, location.href); return ["http:", "https:"].includes(parsed.protocol) ? parsed.href : null; }
  catch { return null; }
}
function fieldLegend(name, min, max, extra = "") {
  el["field-legend"].classList.remove("hidden");
  el["field-legend"].innerHTML = `<span>${escapeHtml(name)}${extra ? ` · ${escapeHtml(extra)}` : ""}</span><span>${numberText(min, 3)}</span><span class="legend-ramp"></span><span>${numberText(max, 3)}</span>`;
}
// Every structured block retains its own connectivity. Flattening C and H
// samples would invent triangles across their unrelated row indices.
function scalarFieldBlocks(fields, key) {
  const blocks=fieldBlocks(fields), result=[];
  for(const block of blocks){
    const samples=block.samples||{},xs=samples.x||[],ys=samples.y||[];
    const raw=key==="speed"&&samples.u&&samples.v ? samples.u.map((u,i)=>Math.hypot(Number(u),Number(samples.v[i]))) : samples[key];
    if(!Array.isArray(xs)||!Array.isArray(ys)||!Array.isArray(raw)||!xs.length||xs.length!==ys.length||xs.length!==raw.length)return null;
    const shape=block.sample_shape,ni=Number(block.ni||(Array.isArray(shape)?shape[0]:shape?.ni)),nj=Number(block.nj||(Array.isArray(shape)?shape[1]:shape?.nj));
    const structured=Number.isInteger(ni)&&Number.isInteger(nj)&&ni>1&&nj>1&&ni*nj===xs.length;
    if(fields.kind==="structured_blocks"&&!structured)return null;
    const pts=xs.map((x,i)=>({x:Number(x),y:Number(ys[i]),v:Number(raw[i])}));
    if(structured&&pts.some(p=>!finite(p.x)||!finite(p.y)||!finite(p.v)))return null;
    const good=structured?pts:pts.filter(p=>finite(p.x)&&finite(p.y)&&finite(p.v));
    if(!good.length)return null;
    result.push({id:block.id||"",ni,nj,structured,pts:good,x:xs,y:ys});
  }
  return result.length?result:null;
}
function renderSamples(point, key) {
  const selected=inspectableFields(point),blocks=scalarFieldBlocks(selected.fields,key);
  if(!blocks)return false;
  const count=blocks.reduce((sum,block)=>sum+block.pts.length,0),structured=blocks.every(block=>block.structured);
  const sorted=blocks.flatMap(block=>block.pts.map(p=>p.v)).sort((a,b)=>a-b);
  const percentile=q=>sorted[Math.min(sorted.length-1,Math.max(0,Math.round(q*(sorted.length-1))))];
  const vmin=percentile(0.02),vmax=percentile(0.98);
  const palette=Array.from({length:256},(_,i)=>`rgb(${ramp(i/255).join(",")})`);
  const color=v=>palette[Math.max(0,Math.min(255,Math.round((v-vmin)/(vmax-vmin||1)*255)))];
  const mesh=currentMesh(point),wall=meshWall(mesh),si=spatialCoordinatesSI(point,mesh),units=quantityUnits(point,key,mesh);
  const gx=wall?.x||point?.native_geometry?.x||state.result?.geometry?.x||[],gy=wall?.y||point?.native_geometry?.y||state.result?.geometry?.y||[];
  if(spatialViews.field?.image)spatialMemory.field=null;
  let probeRasters=[];
  new SpatialPlot("field",el["field-view"],si?coordinateBounds(blocks):{xmin:-0.65,xmax:1.65,ymin:-0.58,ymax:0.58},(ctx,plot)=>{
    probeRasters=[];
    for(const block of blocks){
      if(block.structured){
        const raster=rasterizeStructuredSamples(block.pts,block.ni,block.nj,plot,vmin,vmax,Math.min(plot.dpr||1,1.5));
        probeRasters.push(raster);
        const off=document.createElement("canvas");off.width=raster.width;off.height=raster.height;
        const offctx=off.getContext("2d"),image=offctx.createImageData(raster.width,raster.height);
        image.data.set(raster.pixels);offctx.putImageData(image,0,0);
        ctx.imageSmoothingEnabled=true;ctx.drawImage(off,0,0,plot.width,plot.height);
      }else{
        const radius=Math.max(1.8,Math.min(4,Math.sqrt(plot.width*plot.height/count)*0.25));
        for(const p of block.pts){const x=plot.x(p.x),y=plot.y(p.y);if(x<0||y<0||x>plot.width||y>plot.height)continue;ctx.fillStyle=color(p.v);ctx.beginPath();ctx.arc(x,y,radius,0,Math.PI*2);ctx.fill();}
      }
    }
    if(state.meshOverlay)drawMeshLines(ctx,plot,inspectableMesh(currentMesh(point)));
    drawStreamlines(ctx,plot,currentMesh(point));
    if(gx.length===gy.length&&gx.length>1)drawFoil(ctx,plot,gx,gy);
    drawBaffles(ctx,plot,currentMesh(point));
  },(x,y)=>{
    if(structured&&probeRasters.length&&spatialViews.field){
      const plot=spatialViews.field;
      for(const raster of probeRasters){
        const col=Math.floor(plot.x(x)*raster.width/plot.width),row=Math.floor(plot.y(y)*raster.height/plot.height);
        const value=col>=0&&row>=0&&col<raster.width&&row<raster.height?raster.values[row*raster.width+col]:NaN;
        if(finite(value))return `${spatialCoordinateText(x,y,si)} ? ${quantityLabel(point,key)} ${numberText(value,4)} ${units}`;
      }
      return spatialCoordinateText(x,y,si);
    }
    let best=null,d2=Infinity;
    for(const block of blocks)for(const p of block.pts){const d=(p.x-x)**2+(p.y-y)**2;if(d<d2){best=p;d2=d;}}
    return best&&Math.sqrt(d2)*spatialViews.field.scale<35?`${spatialCoordinateText(x,y,si)} ? ${quantityLabel(point,key)} ${numberText(best.v,4)} ${units}`:spatialCoordinateText(x,y,si);
  },{axisUnits:si?"m":"chord"});
  const clipped=sorted.length>10&&(vmin>sorted[0]||vmax<sorted[sorted.length-1]);
  const sampling=selected.full?"full cell field":`sampled preview, stride ${Number(point?.fields?.sample_stride)||"?"}`;
  const topology=blocks.length>1?` · ${blocks.map(block=>block.id||"block").join(" + ")} blocks`:"";
  fieldLegend(quantityLabel(point,key),vmin,vmax,`${units}${si?" · coordinates in metres":""} · ${count} ${selected.full?"cells":"samples"} · ${sampling}${topology}${structured?" · triangle interpolation":" · points"}${clipped?` · color scale 2–98% (raw ${numberText(sorted[0],3)} to ${numberText(sorted[sorted.length-1],3)})`:""}`);
  const artifact=safeHref(point?.fields?.artifact);
  if(artifact){const a=document.createElement("a");a.href=artifact;a.target="_blank";a.rel="noopener noreferrer";a.textContent="Download sampled field";el["field-legend"].append(a);}
  const fullArtifact=safeHref(point?.fields?.full_artifact);
  if(fullArtifact){const a=document.createElement("a");a.href=fullArtifact;a.target="_blank";a.rel="noopener noreferrer";a.textContent="Download full field";el["field-legend"].append(a);}
  return true;
}
function renderImage(point, key) {
  const fields = point?.fields || {}, src = safeHref(fields.images?.[key]);
  if (!src) return false;
  spatialViews.field?.destroy(); spatialViews.field = null;
  el["field-view"].classList.add("empty-chart"); el["field-view"].textContent = "Loading CFD image…";
  const img = new Image();
  img.onload = () => {
    if (el["field-kind"].value !== `image:${key}` || currentResult() !== point) return;
    spatialMemory.field = null;
    const width=img.naturalWidth||img.width, height=img.naturalHeight||img.height;
    const plot = new SpatialPlot("field",el["field-view"],{xmin:0,xmax:width,ymin:0,ymax:height},(ctx,view)=>{
      ctx.drawImage(img,view.x(0),view.y(height),width*view.scale,height*view.scale);
    },null,{hideAxes:true});
    plot.image = true;
  };
  img.onerror = () => { if (el["field-kind"].value === `image:${key}`) el["field-view"].textContent = "The CFD image could not be loaded."; };
  img.src = src;
  const links = [];
  for (const [label, url] of Object.entries(fields.videos || {})) if (safeHref(url)) links.push(`<a href="${escapeHtml(safeHref(url))}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)} video</a>`);
  if (safeHref(fields.artifact)) links.push(`<a href="${escapeHtml(safeHref(fields.artifact))}" target="_blank" rel="noopener noreferrer">CFD artifact</a>`);
  el["field-legend"].classList.remove("hidden");
  el["field-legend"].innerHTML = `<span>${escapeHtml(fieldNames[key] || key.replaceAll("_"," "))}</span><span class="field-links">${links.join("")}</span>`;
  return true;
}
function renderSurface(point, key) {
  const bl = point?.boundary_layer || {};
  const series = [
    { name: "Upper surface", color: colors.jxfoil, points: (bl.upper || []).map(p => ({x:Number(p.x),y:Number(p[key])})) },
    { name: "Lower surface", color: colors.transonic, points: (bl.lower || []).map(p => ({x:Number(p.x),y:Number(p[key])})) }
  ];
  const chart = polyline(series, { title: `${key} along the surface` });
  if (!chart) return false;
  spatialViews.field?.destroy(); spatialViews.field = null;
  putChart("field-view",chart,"");
  el["field-legend"].classList.add("hidden"); return true;
}
function renderField(point = currentResult()) {
  if (window.AirfoilLBM?.renderField(point)) return;
  const choice = el["field-kind"].value;
  const [kind, key] = choice.split(":", 2);
  const mesh = kind === "mesh" ? inspectableMesh(currentMesh(point)) : currentMesh(point);
  if(kind!=="surface")chartViews.get("field-view")?.destroy();
  const selectedField = kind === "sample" ? inspectableFields(point) : null;
  const displayPoint = selectedField?.full ? { ...point, fields: selectedField.fields } : point;
  const useBabylon = el["field-renderer"].value === "3d" && (kind === "mesh" || kind === "sample");
  el["field-view"].classList.toggle("hidden", useBabylon);
  el["babylon-view"].classList.toggle("hidden", !useBabylon);
  if (useBabylon) {
    spatialViews.field?.destroy(); spatialViews.field = null;
    const wall=meshWall(mesh),geometry=state.result?.geometry||state.preview?.geometry;
    if (state.activeTab === "field" && window.AirfoilBabylon?.show({ node: el["babylon-view"], mesh, point: displayPoint, quantity: kind === "mesh" ? "mesh" : key, geometry: wall?{...geometry,x:wall.x,y:wall.y}:geometry }) === false) {
      el["babylon-view"].textContent = "This field does not have a structured 3D view.";
    }
    if (kind === "mesh" && validMesh(mesh)) {
      el["field-legend"].classList.remove("hidden");
      el["field-legend"].textContent = `${meshIsFull(mesh) ? "Full mesh" : "Mesh preview"} ${meshDimensionsText(mesh)} nodes${meshTopologyText(mesh)}${meshQualityText(mesh)}`;
    } else if (kind === "sample") {
      el["field-legend"].classList.remove("hidden");
      const count = fieldCellCount(selectedField?.fields);
      const blocks=fieldBlocks(selectedField?.fields),range=window.AirfoilBabylon?.scalarColorRange?.();
      const extra=`${quantityUnits(displayPoint,key,mesh)}${spatialCoordinatesSI(displayPoint,mesh)?" · coordinates in metres":""} · ${count} ${selectedField?.full ? "full field cells" : "sampled preview points"}${blocks.length>1?` · ${blocks.map(block=>block.id).join(" + ")} blocks`:""} · vertex interpolation · drag to orbit · right drag to pan · scroll to zoom`;
      if(range)fieldLegend(quantityLabel(displayPoint,key),range.low,range.high,extra);
      else el["field-legend"].textContent=`${quantityLabel(displayPoint,key)} · ${extra}`;
    }
    return;
  }
  window.AirfoilBabylon?.hide();
  if (kind === "live") { const lp=livePoint(); if (lp && renderSamples(lp, key)) return; }
  if (kind === "mesh" && renderMesh2D(mesh)) return;
  if (kind === "sample" && renderSamples(point, key)) return;
  if (kind === "image" && renderImage(point, key)) return;
  if (kind === "surface" && renderSurface(point, key)) return;
  const field = point?.fields?.[key];
  const grid = fieldGrid(field);
  if (!grid) {
    spatialViews.field?.destroy(); spatialViews.field = null;
    el["field-view"].classList.add("empty-chart");
    el["field-view"].textContent = ["jxfoil","xfoil"].includes(point?.backend) ? `${labelBackend(point.backend)} returns surface values, not a spatial CFD flow field.` : "This result does not include a plottable spatial field.";
    el["field-legend"].classList.add("hidden"); return;
  }
  const good = grid.values.filter(Number.isFinite);
  if (!good.length) { el["field-view"].classList.add("empty-chart"); el["field-view"].textContent = "Field values are unavailable."; el["field-legend"].classList.add("hidden"); return; }
  let min = Infinity, max = -Infinity;
  for (const v of good) { if (v < min) min = v; if (v > max) max = v; }
  const span = max - min || 1;
  const off = document.createElement("canvas"); off.width = grid.nx; off.height = grid.ny;
  const ctx = off.getContext("2d"), image = ctx.createImageData(grid.nx, grid.ny);
  const flip = Array.isArray(grid.y) && grid.y.length === grid.ny && Number(grid.y[0]) < Number(grid.y[grid.ny - 1]);
  for (let row = 0; row < grid.ny; row++) for (let col = 0; col < grid.nx; col++) {
    const sourceRow = flip ? grid.ny - 1 - row : row;
    const v = grid.values[sourceRow * grid.nx + col];
    const i = (row * grid.nx + col) * 4;
    if (!Number.isFinite(v)) { image.data[i+3] = 0; continue; }
    const c = ramp((v - min) / span);
    image.data[i] = c[0]; image.data[i+1] = c[1]; image.data[i+2] = c[2]; image.data[i+3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  const xaxis=Array.isArray(grid.x) && grid.x.length===grid.nx ? grid.x.map(Number) : null;
  const yaxis=Array.isArray(grid.y) && grid.y.length===grid.ny ? grid.y.map(Number) : null;
  const xlo=xaxis?Math.min(...xaxis):0, xhi=xaxis?Math.max(...xaxis):1;
  const ylo=yaxis?Math.min(...yaxis):0, yhi=yaxis?Math.max(...yaxis):1;
  new SpatialPlot("field",el["field-view"],{xmin:xlo,xmax:xhi,ymin:ylo,ymax:yhi},(target,plot)=>{
    target.imageSmoothingEnabled=true;
    target.drawImage(off,plot.x(xlo),plot.y(yhi),(xhi-xlo)*plot.scale,(yhi-ylo)*plot.scale);
    const gx=state.result?.geometry?.x||[], gy=state.result?.geometry?.y||[];
    if(xaxis&&yaxis&&gx.length>1&&gx.length===gy.length) drawFoil(target,plot,gx,gy);
  });
  fieldLegend(fieldNames[key] || key, min, max, field?.units || "");
}
function validCfdPolarPoint(r) {
  if(!isCfdBackend(r.backend))return true;
  if(!["converged","completed","ok"].includes(String(r.status||"").toLowerCase()))return false;
  if(r.backend==="lowspeed")return lowspeedSteady(r)&&finite(r.cd)&&r.cd>=0;
  if(r.backend==="lbm")return finite(r.cl)&&finite(r.cd)&&r.cd>=0;   // a completed LBM run: the mean of its force history's last third
  if(isPhysicalPoint(r)){const stats=r.diagnostics?.time_statistics;return stats?.physical_mean_eligible===true&&["cl_mean","cd_mean","cm_mean"].every(key=>finite(stats[key]))&&stats.cd_mean>=0;}
  return r.diagnostics?.convergence_guard?.status === "early_stop_eligible" && (r.backend!=="transonic"||r.diagnostics?.physical_sanity === true) && finite(r.cd) && r.cd >= 0;
}
function polarSeries(rows, key) {
  const plottable = rows.filter(r => finite(Number(r.alpha_deg)) && finite(r[key]));
  const points = matches => matches.sort((a,b) => Number(a.alpha_deg)-Number(b.alpha_deg)).map(r => ({x:Number(r.alpha_deg),y:Number(r[key])}));
  return [...new Set(plottable.map(r => r.backend))].flatMap(backend => {
    const color=colors[backend] || colors.cavitation;
    const backendRows=plottable.filter(r => r.backend === backend);
    if (!isCfdBackend(backend)) return [{name:`${labelBackend(backend)}${backendRows.length&&backendRows.every(r=>r.geometry_pairing==="same_as_lowspeed_cfd")?" · same section as the low-speed CFD":backendRows.length&&backendRows.every(r=>r.comparison_role==="no-flap reference (XFOIL cannot model the plate)")?" · no-flap reference (XFOIL cannot model the plate)":""}`,color,points:points(backendRows)}];
    const stationary=points(backendRows.filter(r=>!isPhysicalPoint(r)&&validCfdPolarPoint(r)));
    const physicalMeans=points(backendRows.filter(r=>isPhysicalPoint(r)&&validCfdPolarPoint(r)).map(r=>({...r,[key]:r.diagnostics.time_statistics[key+"_mean"]})));
    const diagnostic=points(backendRows.filter(r => !validCfdPolarPoint(r)));
    return [
      {name:`${labelBackend(backend)} · ${backend==="lowspeed"?"solver steady stop (unvalidated)":backend==="lbm"?"mean of the last third of the run (unvalidated)":"continuity/force gated (unvalidated)"}`,color,points:stationary},
      {name:`${labelBackend(backend)} · physical mean (sampling qualified, stall unvalidated)`,color,points:physicalMeans},
      {name:`${labelBackend(backend)} · diagnostic only`,color,points:diagnostic,connect:false,marker:"hollow"}
    ].filter(s => s.points.length);
  });
}
function renderPolars() {
  const rawRows = state.result?.results || [];
  const diagnostic=rawRows.filter(r => isCfdBackend(r.backend) && !validCfdPolarPoint(r));
  const cfdRows = rawRows.filter(r => isCfdBackend(r.backend));
  const risingMass = cfdRows.filter(r => finite(r.diagnostics?.residual_ratio) && r.diagnostics.residual_ratio >= 1);
  const reference = el["show-reference"].checked ? state.result?.reference : null;
  // Wind-tunnel lift at NEARBY conditions (NACA 0012 with the low-speed CFD): own Re/Mach in the legend, never matched.
  const nearby = el["show-reference"].checked && Array.isArray(state.job?.nearby_experiments) ? state.job.nearby_experiments.filter(e=>Array.isArray(e?.points)&&e.points.length) : [];
  const alphas = rawRows.map(r=>Number(r.alpha_deg)).filter(finite);
  const nearbyRange = alphas.length ? [Math.min(...alphas)-1, Math.max(...alphas)+1] : null;
  el["polar-note"].classList.toggle("hidden", !cfdRows.length && !reference && !nearby.length);
  el["polar-note"].textContent = [
    nearby.length ? `Wind-tunnel lift at nearby conditions, not matched to this case: ${nearby.map(e=>`${e.name} (${e.conditions})`).join("; ")}; digitized by NASA TMR, approximate. The CFD is Re ${finite(Number(state.analysisCase?.flow?.reynolds))?(Number(state.analysisCase.flow.reynolds)/1e6).toPrecision(3)+"M":"as set"}, incompressible and fully turbulent.` : "",
    cfdRows.some(r=>r.backend==="lowspeed") ? "Connected low-speed CFD points reached the solver's own steady stop (momentum, k and omega residuals and the far-field circulation settled) on the coarse demo mesh; version 2 is not yet validated." : "",
    cfdRows.some(r=>r.backend==="lbm") ? "LBM 3D points are the means of the last third of each run's force history (periodic span); grid and span independence are not established. Friction (viscous) drag from the LBM surface-force routine is underestimated (measured 0.22-0.71 of exact in a laminar channel test); drag values are mostly pressure drag. Under investigation. The LBM curved-wall conditions place the effective wall about 1.1-1.3 cells inside the fluid (measured in laminar channel tests; independent of resolution), so bodies are effectively about that much thicker on each side and surface shear is misread. Root cause identified (a write-location defect in the linear Bouzidi family, a double relaxation in the grad condition, and the solid-cell marking); fixes are in development. LBM results are development-only until then." : "",
    cfdRows.some(r=>!["lowspeed","lbm"].includes(r.backend)) ? "Connected steady CFD points passed the continuity and force-history guard. Physical means are connected only when sampling, stationarity, block independence and timestep refinement qualify; their legend identifies the mean. Aerodynamic and stall parity remain unvalidated." : "",
    diagnostic.length ? `${diagnostic.length} unverified CFD point${diagnostic.length === 1 ? " is" : "s are"} shown as hollow, unconnected markers.` : "",
    risingMass.length ? `${risingMass.length} CFD point${risingMass.length === 1 ? " has" : "s have"} a continuity residual that did not fall after damping.` : "",
    reference ? (reference.points.some(p => finite(p.cl) || finite(p.cd) || finite(p.cm))
      ? `${reference.name} includes published force markers at the matched validation condition; these are isolated reference points, not a measured polar.`
      : `${reference.name} supplies experimental Cp only; no measured CL, CD, or CM is available for a polar overlay.`) : ""
  ].filter(Boolean).join(" ");
  for (const [key, id, digits] of [["cl", "polar-cl", 3], ["cd", "polar-cd", 4], ["cm", "polar-cm", 3]]) {
    const series = polarSeries(rawRows,key);
    if (reference?.points) {
      const measured = reference.points.filter(p => p[key] != null && finite(Number(p[key])))
        .map(p => ({x:Number(p.alpha_deg),y:Number(p[key])}));
      if (measured.length) series.push({name:`${reference.name} experiment`,color:"#9b5bc4",points:measured,connect:false});
    }
    if (key === "cl" && nearbyRange) nearby.forEach((e,i)=>{
      const pts=e.points.map(p=>({x:Number(p.alpha_deg),y:Number(p.cl)})).filter(p=>finite(p.x)&&finite(p.y)&&p.x>=nearbyRange[0]&&p.x<=nearbyRange[1]);
      if (pts.length) series.push({name:`${e.name}, ${e.conditions} · nearby condition, digitized`,color:["#8a5a1f","#4f6d8f"][i%2],points:pts,connect:false,dash:"2 3"});
    });
    const hasNegativeDrag = key === "cd" && series.some(s => s.points.some(p => p.y < 0));
    putChart(id, polyline(series, { short: true, title: `${key.toUpperCase()} versus angle of attack`, digits, singlePad: key === "cd" ? 0.005 : 0, nonnegative: key === "cd" && !hasNegativeDrag }), `No ${key.toUpperCase()} polar values were returned.`);
  }
}
function renderTable() {
  const rows = state.result?.results || [];
  if (!rows.length) { el["points-body"].innerHTML = '<tr><td colspan="6" class="table-empty">No computed points were returned.</td></tr>'; return; }
  el["points-body"].innerHTML = rows.map(r => {
    const checked = validCfdPolarPoint(r);
    const warning = !checked || (r.status && !["converged","completed","ok"].includes(String(r.status).toLowerCase()));
    const status = r.backend==="lbm" ? (window.AirfoilLBM?.statusLabel(r) || String(r.status || "Computed")) : r.backend==="lowspeed" ? (checked ? `Steady stop at step ${r.diagnostics?.steps ?? "?"} · unvalidated` : r.status==="not_converged" ? `${String(r.diagnostics?.solver_outcome||"not steady").toLowerCase()} · diagnostic only` : (r.status || "Computed")) : r.status==="experimental_physical_state"?experimentalStateLabel(r):isCfdBackend(r.backend) && checked ? isPhysicalPoint(r) ? "Sampling qualified · stall unvalidated" : "Continuity/force gated · unvalidated" : !checked && ["converged","completed","ok"].includes(String(r.status || "completed").toLowerCase()) ? "Unverified CFD" : (r.status || "Computed");
    return `<tr><td>${escapeHtml(labelBackend(r.backend))}</td><td>${numberText(Number(r.alpha_deg), 2)}</td><td>${numberText(r.cl)}</td><td>${numberText(r.cd, 5)}</td><td>${numberText(r.cm)}</td><td><span class="status-tag ${warning ? "warning" : ""}">${escapeHtml(status)}</span></td></tr>`;
  }).join("");
  const describe = value => Array.isArray(value) ? value.join("; ") : typeof value === "object" ? JSON.stringify(value) : String(value);
  const notes = [...new Set((state.result?.results || []).flatMap(r => {
    const prefix = `${labelBackend(r.backend)} at ${numberText(Number(r.alpha_deg), 2)}°`;
    return [["Error",r.error],["Message",r.message],["Notes",r.notes],["Convergence",r.convergence],["Validation",r.validation],["Geometry transform",r.geometry_transform || r.geometry],["Comparison role",r.comparison_role],["Analysis Mach",r.analysis_mach],["Requested Mach",r.requested_mach],["Coefficient method",r.coefficient_method],["Diagnostics",r.diagnostics],["Provenance",r.provenance]].filter(([,v]) => v != null && v !== "").map(([label,v]) => `${prefix} · ${label}: ${describe(v)}`);
  }))];
  el["result-notes"].innerHTML = notes.map(n => `<p>${escapeHtml(n)}</p>`).join("");
}
function renderResult() {
  const result = state.result;
  if (!result) { displayError("The job completed without a result payload."); return; }
  const experimental=(result.results||[]).filter(r=>r.status==="experimental_physical_state");
  const incomplete = (result.results || []).filter(r => !["completed", "converged", "ok", "experimental_physical_state"].includes(String(r.status || "completed").toLowerCase()));
  const stalled=incomplete.filter(r=>r.status==="stopped_without_convergence");
  if(experimental.length&&!incomplete.length&&["completed","partial","failed"].includes(state.job?.status)){
    el["job-title"].textContent="Experimental physical state saved";
    const details=experimental.slice(0,3).map(point=>{
      const d=point.diagnostics||{},elapsed=d.elapsed_seconds??d.elapsed_seconds_physical,requested=d.requested_duration_seconds??d.duration_seconds_requested;
      const steps=d.steps??point.steps,max=d.max_steps??point.max_steps,stop=d.stop_reason??point.stop_reason;
      return `${experimental.length>1?`${numberText(Number(point.alpha_deg),2)}°: `:""}${finite(elapsed)?`actual physical time ${Number(elapsed).toPrecision(5)}${finite(requested)?` / requested ${Number(requested).toPrecision(5)}`:""} s`:"physical time recorded in Diagnostics"}${finite(steps)?` · ${steps}${finite(max)?` / ${max}`:""} steps`:""}${stop?` · ${String(stop).replaceAll("_"," ")}`:""}`;
    });
    el["job-message"].textContent=`${details.join("; ")}${experimental.length>3?`; ${experimental.length-3} more saved states`:""}. Instantaneous inviscid fields and pressure loads are diagnostic; cavitation, drag and stall parity remain unqualified.`;
  } else if (stalled.length && stalled.length===incomplete.length && ["completed","partial","failed"].includes(state.job?.status)) {
    el["job-title"].textContent = "Analysis stopped before convergence";
    el["job-message"].textContent = `${stalled.length} operating point${stalled.length===1?"":"s"} reached a persistent residual plateau. Results remain diagnostic; inspect the stop reason and history.`;
  } else if (incomplete.length && ["completed","partial","failed"].includes(state.job?.status)) {
    el["job-title"].textContent = incomplete.length === result.results.length ? "Analysis failed" : "Analysis finished with partial results";
    el["job-message"].textContent = `${incomplete.length} operating point${incomplete.length === 1 ? "" : "s"} did not complete. See the point status and notes below.`;
  }
  el["download-result"].disabled = false;
  renderSelection(); renderPolars(); renderTable();syncRunStatusDetails();refreshPointOverlays();
}
async function downloadResult() {
  if (!state.result) return;
  const done=beginOperation("download", "Preparing result download…", [[el["download-result"], "Preparing…"]]);
  try {
    await afterPaint();
    const blob = new Blob([JSON.stringify({ job: state.job, result: state.result }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob), a = document.createElement("a");
    a.href = url; a.download = `airfoil-${state.job?.id || "analysis"}.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch(err) {displayError(err);} finally {done();}
}

function sourceChanged() {
  const source = el["source-kind"].value;
  if(source==="cavitation_reference"){updateSetup();return selectCavitationReference();}
  state.cavitationReferenceSelection++;
  if (source === "rae2822" || source === "oat15a") {
    // Selecting a validation section loads its published operating point;
    // every input remains editable for other operating conditions.
    el.backend.value = "transonic";
    el.mach.value = source === "rae2822" ? "0.734" : "0.73";
    el.reynolds.value = source === "rae2822" ? "6500000" : "3000000";
    el.alpha.value = source === "rae2822" ? "2.79" : "3.25";
    el["cfd-viscosity-model"].value = source === "rae2822" ? "constant_dynamic" : "sutherland";
    el["transition-mode"].value="forced";
    el["trip-upper"].value=el["trip-lower"].value=source==="rae2822" ? "0.03" : "0.07";
    if (source === "oat15a") el["cfd-temperature"].value = "271";
    state.machEdited = true;
  }
  updateSetup();
  scheduleGeometryPreview();
}
function setTheme(mode) {
  if (["light", "dark"].includes(mode)) document.documentElement.dataset.theme = mode;
  else document.documentElement.dataset.theme = window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  localStorage.setItem("airfoil-theme", mode);
  Object.values(spatialViews).forEach(view=>view?.schedule());
  window.AirfoilBabylon?.redrawTheme();
}
function switchTab(name) {
  const tab = document.querySelector(`[data-tab-target="${name}"]`);
  if (!tab) return;
  state.activeTab = name;
  window.AirfoilCampaign?.activate(name==="campaign");
  document.querySelectorAll("[data-tab-target]").forEach(node => {
    const active = node === tab;
    node.classList.toggle("is-active", active);
    node.setAttribute("aria-selected", String(active));
    node.tabIndex = active ? 0 : -1;
  });
  document.querySelectorAll("[data-tab-panel]").forEach(node => {
    const active = node.dataset.tabPanel === name;
    node.hidden = !active;
    node.classList.toggle("is-active", active);
  });
  if (name === "geometry") {spatialViews.geometry?.resize();spatialViews.curvature?.resize();}
  if (name === "field") {
    if (el["field-renderer"].value === "3d") renderField();
    else spatialViews.field?.resize();
  } else window.AirfoilBabylon?.hide();
}
// ---- Low-speed CFD live flow fields: the running point's committed state, written every few dozen steps. -------------
const lowspeedFieldMeta={speed_column:"mach",hidden_quantities:["density","temperature"],
  quantity_labels:{mach:"Speed |U|/U∞",cp:"Pressure coefficient Cp",pressure:"Pressure (p - p∞)/(ρU∞²)",u:"Velocity u/U∞",v:"Velocity v/U∞",vorticity:"Vorticity ωc/U∞",nut:"Eddy viscosity ratio μt/μ"},
  quantity_units:{mach:"|U|/U∞",cp:"dimensionless",pressure:"ρU∞²",u:"U∞",v:"U∞",vorticity:"U∞/c",nut:"μt/μ",density:"ρ∞",temperature:"T∞"}};
function liveFieldEntry() {
  const live=state.result?.live_fields;
  if(!live||typeof live!=="object")return null;
  const entries=Object.values(live).filter(e=>e&&typeof e.artifact==="string"&&Number.isInteger(Number(e.iteration)));
  return entries.find(e=>e.backend===state.activeCfdBackend&&Number(e.alpha_deg)===Number(state.activeCfdAlpha))||entries[entries.length-1]||null;
}
function refreshLiveField() {
  const entry=["queued","running"].includes(state.job?.status)?liveFieldEntry():null;
  if(!entry){
    if(state.liveField){state.liveField=null;renderFieldChoices(currentResult());}
    return;
  }
  const url=safeHref(entry.artifact);
  if(!url||state.liveField?.url===url||state.liveFieldLoading===url)return;
  const generation=state.pollGeneration;state.liveFieldLoading=url;
  request(url).then(field=>{
    if(generation!==state.pollGeneration||!validFullField(field))return;
    const first=!state.liveField;
    state.liveField={url,iteration:Number(entry.iteration),alpha:Number(entry.alpha_deg),backend:entry.backend,field};
    // The first live field of a run replaces the mesh view once (a later choice of the user is kept).
    if(first&&!state.liveAutoShown&&["","mesh:grid"].includes(el["field-kind"].value)){state.liveAutoShown=true;el["field-kind"].value="live:mach";}
    renderFieldChoices(currentResult());
  }).catch(()=>{}).finally(()=>{if(state.liveFieldLoading===url)state.liveFieldLoading=null;});
}
function livePoint() {
  const live=state.liveField;
  if(!live)return null;
  const f=live.field,labels=Object.fromEntries(Object.entries(lowspeedFieldMeta.quantity_labels).map(([k,v])=>[k,`Live step ${live.iteration} · ${v}`]));
  return {backend:live.backend,alpha_deg:live.alpha,status:"running",live:true,mesh:currentMesh({alpha_deg:live.alpha}),
    fields:{kind:"structured_cells",ni:f.ni,nj:f.nj,samples:f.samples,cell_order:f.cell_order,iteration:f.iteration,mesh_sha256:f.mesh_sha256,
      sample_stride:1,...lowspeedFieldMeta,quantity_labels:labels}};
}
function liveFieldApplies(point) {
  const live=state.liveField;
  if(!live)return false;
  if(point&&point.backend===live.backend&&Number(point.alpha_deg)===live.alpha)return false;   // its final field exists
  return !(point?.fields&&["samples","structured_cells","structured_blocks"].includes(point.fields.kind));
}

// ---- Recent results: jobs saved under runs/<id>/ reopen read-only (also after a server restart). --------------------
function recentLabel(item) {
  const foil=item?.airfoil?(item.airfoil.kind==="naca4"?`NACA ${item.airfoil.id}`:String(item.airfoil.id||item.airfoil.kind||"?")):"?";
  const short={lbm:"LBM 3D",lowspeed:"Low-speed CFD",xfoil:"XFOIL",jxfoil:"JXfoil",transonic:"Compressible CFD",cavitation:"Species CFD"};
  const backends=(Array.isArray(item?.backends)?item.backends:[]).map(b=>short[b]||b).join(" + ");
  const a=Array.isArray(item?.alphas)?item.alphas.map(Number).filter(finite):[];
  const alpha=a.length>1?`α ${a[0]}…${a[a.length-1]}° (${a.length})`:a.length?`α ${a[0]}°`:"";
  const re=finite(Number(item?.reynolds))?`Re ${Number(item.reynolds)>=1e6?`${(Number(item.reynolds)/1e6).toPrecision(3)}M`:Number(item.reynolds).toPrecision(3)}`:"";
  const date=new Date(item?.updated_at||item?.saved_at||item?.created_at||"");
  const when=Number.isNaN(date.getTime())?"":date.toLocaleString([],{month:"short",day:"numeric",hour:"2-digit",minute:"2-digit",hour12:false});
  return [foil,backends,alpha,re,when,item?.status].filter(Boolean).join(" · ");
}
async function loadRecentResults() {
  try {
    const data=await request("/api/recent",{format:el.transport.value||"msgpack"});
    const items=Array.isArray(data?.items)?data.items.filter(item=>typeof item?.id==="string"&&/^[0-9a-f-]{36}$/.test(item.id)):[];
    state.recentResults=items;
    const prior=el["recent-select"].value;
    const options=items.length?items.map(item=>{const o=document.createElement("option");o.value=item.id;o.textContent=recentLabel(item);o.title=item.message||"";return o;}):
      [Object.assign(document.createElement("option"),{value:"",textContent:"No saved results yet"})];
    el["recent-select"].replaceChildren(...options);
    if(items.some(item=>item.id===prior))el["recent-select"].value=prior;
    el["recent-select"].disabled=!items.length;el["recent-open"].disabled=!items.length;
    el["recent-status"].textContent=items.length?`${items.length} saved job${items.length===1?"":"s"}, newest first. Open one to show its fields, Cp, polar and history.`:"No saved results yet.";
    return items;
  } catch (err) {
    el["recent-status"].textContent=`Saved results unavailable: ${err.message}`;
    return [];
  }
}
// Put a reopened job's own inputs back in the form (NACA, RAE 2822 and OAT15A sources), so the sidebar describes the shown
// result and Run analysis repeats it. Other sources leave the form as it is. Returns whether the form was updated.
function applySavedRequestToForm(saved) {
  window.LudwigOutputControls?.restore(saved);
  const foil=saved?.airfoil||{},flow=saved?.flow||{},backends=Array.isArray(saved?.backends)?saved.backends:[];
  if(foil.kind==="naca4"&&/^\d{4}$/.test(String(foil.id))){el["source-kind"].value="naca4";el["naca-code"].value=String(foil.id);}
  else if(["rae2822","oat15a"].includes(foil.kind))el["source-kind"].value=foil.kind;
  else return false;
  const selection=["jxfoil","xfoil","transonic","cavitation","lowspeed","both","cfd_both","xfoil_cfd_both","xfoil_lowspeed","lbm"].find(value=>{
    const b=selectedBackends(value);return b.length===backends.length&&b.every((x,i)=>x===backends[i]);});
  if(selection)el.backend.value=selection;
  for(const [key,id] of [["mach","mach"],["reynolds","reynolds"],["alpha_deg","alpha"],["ncrit","ncrit"]])if(finite(Number(flow[key])))el[id].value=String(flow[key]);
  if(finite(Number(flow.lowspeed_tu)))el["lowspeed-tu"].value=String(Number((100*flow.lowspeed_tu).toPrecision(6)));
  if(finite(Number(flow.lowspeed_mut_ratio)))el["lowspeed-mut-ratio"].value=String(flow.lowspeed_mut_ratio);
  el["lowspeed-gpu"].checked=flow.lowspeed_gpu===true;
  const sweep=saved?.sweep||{};el["sweep-enabled"].checked=sweep.enabled===true;
  if(sweep.enabled===true)for(const key of ["start","stop","step"])if(finite(Number(sweep[key])))el[`alpha-${key}`].value=String(sweep[key]);
  if(backends.includes("xfoil")&&(Number(flow.trip_upper)>0||Number(flow.trip_lower)>0)){
    el["transition-mode"].value="forced";el["trip-upper"].value=String(flow.trip_upper??"");el["trip-lower"].value=String(flow.trip_lower??"");
  }
  el["trailing-edge-thickness"].value=finite(Number(flow.cfd_te_thickness))&&Number(flow.cfd_te_thickness)>0?String(flow.cfd_te_thickness):"";
  el["gurney-enabled"].checked=finite(Number(flow.gurney_height));
  if(el["gurney-enabled"].checked){
    el["gurney-height"].value=String(flow.gurney_height);el["gurney-mode"].value=["symmetric","lower","upper"].includes(flow.gurney_mode)?flow.gurney_mode:"lower";
    el["gurney-refine"].value=gurneyRules.refineLevels.includes(flow.gurney_refine)?flow.gurney_refine:"off";   // a case saved before the option had none
  }
  el["streamline-align"].checked=flow.cfd_streamline_align===true;
  // a saved native-base C+H control (an API option the form cannot set): its alignment note must not read as a refusal
  state.savedLowspeedTopology=flow.lowspeed_topology==="ch_native"?"ch_native":null;
  if(el["streamline-align"].checked){el["streamline-mode"].value=flow.cfd_streamline_mode==="inviscid"?"inviscid":"viscous";el["streamline-admission"].value=flow.cfd_streamline_admission==="strict"?"strict":"demo";el["streamline-fallback"].checked=flow.cfd_streamline_fallback===true;}
  window.AirfoilLBM?.applySaved(flow);
  state.machEdited=true;state.reynoldsEdited=true;
  updateSetup();
  return true;
}
async function openSavedJob(id=el["recent-select"].value) {
  if(!id||!/^[0-9a-f-]{36}$/.test(id))return;
  clearError();
  const done=beginOperation("open-saved","Opening saved result…",[[el["recent-open"],"Opening…"]]);
  try {
    const wire=el.transport.value||"msgpack";
    const job=await request(`/api/jobs/${encodeURIComponent(id)}`,{format:wire});
    if(!job||job.id!==id)throw new Error("The server returned a different job.");
    if(state.pollTimer)clearTimeout(state.pollTimer);
    clearTimeout(state.historyTimer);
    const generation=++state.pollGeneration;
    const saved=job.request||{},results=job.result?.results||[];
    state.job=null;state.result=null;state.jobWire=wire;
    state.analysisCase={airfoil:saved.airfoil||{},flow:saved.flow||{},sweep:saved.sweep||{enabled:false},
      backends:Array.isArray(saved.backends)&&saved.backends.length?saved.backends:[...new Set(results.map(r=>r.backend))]};
    const formUpdated=applySavedRequestToForm(saved);
    state.analysisGeometryKey=geometrySourceKey({kind:state.analysisCase.airfoil.kind,id:state.analysisCase.airfoil.id},state.analysisCase.flow||{});
    state.analysisMeshKey=formUpdated?meshSettingsKey():null;
    if(formUpdated)scheduleGeometryPreview();
    state.requestedAlpha=state.analysisCase.flow?.alpha_deg??null;state.meshKeys="";
    state.fullMeshes.clear();state.meshLoading.clear();state.fullFields.clear();state.fieldLoading.clear();state.liveField=null;state.liveFieldLoading=null;state.liveAutoShown=false;
    chartMemory.clear();spatialMemory.geometry=null;spatialMemory.field=null;
    el["convergence-panel"].classList.toggle("hidden",!state.analysisCase.backends.some(isCfdBackend));
    resetLog();window.AirfoilBabylon?.hide();el["download-result"].disabled=true;
    renderSelection();renderPolars();renderTable();
    state.finishAnalysis?.();state.finishAnalysis=null;
    if(["queued","running"].includes(job.status))state.finishAnalysis=beginOperation("analysis","Julia analysis running…",[[el["run-button"],"Analysis running…"]]);
    jobStatus(job);
    const savedAt=new Date(job.saved_at||job.updated_at||"");
    el["recent-status"].textContent=job.restored?`Reopened saved result${Number.isNaN(savedAt.getTime())?"":` from ${savedAt.toLocaleString([],{hour12:false})}`} (read-only; Run analysis computes it again).`:
      ["queued","running"].includes(job.status)?"Attached to the running job: live charts continue.":"Opened a job from this server session.";
    pollTelemetry(id,generation);pollJob(id,generation);
  } catch(err) {displayError(err);} finally {done();}
}
el["source-kind"].addEventListener("change", sourceChanged);
el["recent-open"].addEventListener("click",()=>openSavedJob());
el["recent-refresh"].addEventListener("click",()=>loadRecentResults());
el["reynolds"].addEventListener("input",()=>{state.reynoldsEdited=true;});
el["cav-reference-select"].addEventListener("change",selectCavitationReference);
el.backend.addEventListener("change", () => {
  if (!state.machEdited) el.mach.value = ["jxfoil","lowspeed","xfoil_lowspeed"].includes(el.backend.value) ? "0" : "0.1";
  // The low-speed demonstration case is Re 3 million; the untouched page default (300,000) is replaced, an edited value kept.
  if(["lowspeed","xfoil_lowspeed"].includes(el.backend.value)&&!state.reynoldsEdited&&el.reynolds.value==="300000")el.reynolds.value="3000000";
  if(el.backend.value==="xfoil_lowspeed") {
    // XFOIL with fixed transition near the leading edge is the like-for-like reference of a fully turbulent CFD.
    el["transition-mode"].value="forced";
    if(!el["trip-upper"].value)el["trip-upper"].value="0.05";
    if(!el["trip-lower"].value)el["trip-lower"].value="0.05";
  }
  if(el.backend.value==="xfoil_cfd_both") {
    el["cav-fluid"].value="air";el["cav-domain"].value="free_fluid";
    el["transition-mode"].value="forced";
    el["advanced-cfd"].open=true;
  }
  updateSetup(); invalidatePreviewMesh();
});
el["transition-mode"].addEventListener("change",()=>{updateSetup();invalidatePreviewMesh();});
["trip-upper","trip-lower"].forEach(id=>el[id].addEventListener("input",invalidatePreviewMesh));
el.mach.addEventListener("input", () => { state.machEdited = true; updateSetup(); invalidatePreviewMesh(); });
el["sweep-enabled"].addEventListener("change", updateSetup);
el["cfd-mesh"].addEventListener("change", () => { updateSetup(); invalidatePreviewMesh(); });
el["cfd-te-topology"].addEventListener("change",()=>{updateSetup();invalidatePreviewMesh();});
el["cfd-gpu"].addEventListener("change", updateSetup);
el["cfd-viscosity-model"].addEventListener("change", updateSetup);
["cav-fluid","cav-domain"].forEach(id=>el[id].addEventListener("change",()=>{updateSetup();invalidatePreviewMesh();}));
["cav-air-mode","cfd-time-mode"].forEach(id=>el[id].addEventListener("change",updateSetup));
["cav-channel","cav-depth","cav-chord","cav-speed","cav-water-depth","cav-air-height","cav-fs-vapor"].forEach(id=>el[id].addEventListener("input",invalidatePreviewMesh));
el["cav-fs-resolution"].addEventListener("change",invalidatePreviewMesh);
state.fsDurationEdited=false;
el["cav-fs-duration"].addEventListener("input",()=>{state.fsDurationEdited=true;});
el["cav-fs-formulation"].addEventListener("change",()=>{
  if(!state.fsDurationEdited)el["cav-fs-duration"].value=el["cav-fs-formulation"].value==="barotropic_explicit"?"0.002":"0.1";
  updateSetup();invalidatePreviewMesh();
});
["alpha-start", "alpha-stop", "alpha-step"].forEach(id => el[id].addEventListener("input", updateSetup));
el["airfoil-search"].addEventListener("input", renderAirfoils);
el["airfoil-select"].addEventListener("change", () => { updateAirfoilMeta(); scheduleGeometryPreview(); });
el["naca-code"].addEventListener("input", () => {updateSetup();scheduleGeometryPreview();});
el["trailing-edge-thickness"].addEventListener("input", () => {updateSetup();scheduleGeometryPreview();});
el["gurney-enabled"].addEventListener("change", () => {updateSetup();scheduleGeometryPreview();});
["gurney-height"].forEach(id=>el[id].addEventListener("input", () => {updateSetup();scheduleGeometryPreview();}));
el["gurney-mode"].addEventListener("change", () => {updateSetup();scheduleGeometryPreview();});
el["gurney-refine"].addEventListener("change", () => {updateSetup();});
["streamline-align","streamline-mode","streamline-admission","streamline-fallback"].forEach(id=>el[id].addEventListener("change", () => {updateSetup();invalidatePreviewMesh();}));
el["lowspeed-gpu"].addEventListener("change",()=>updateSetup());
["geometry-streamlines","field-streamlines"].forEach(id=>el[id].addEventListener("change", () => {state.showStreamlines=el[id].checked;el["geometry-streamlines"].checked=el["field-streamlines"].checked=state.showStreamlines;renderSelectedGeometry();if(state.activeTab==="field")renderField();}));
el["geometry-display"].addEventListener("change", renderSelectedGeometry);
el["curvature-scale"].addEventListener("input", renderSelectedGeometry);
state.curvatureMapping=localStorage.getItem("airfoil-curvature-mapping")==="log" ? "log" : "linear";
document.querySelectorAll("input[name='curvature-mapping']").forEach(radio=>{
  radio.checked=radio.value===state.curvatureMapping;
  radio.addEventListener("change",()=>{if(!radio.checked)return;state.curvatureMapping=radio.value==="log" ? "log" : "linear";localStorage.setItem("airfoil-curvature-mapping",state.curvatureMapping);renderSelectedGeometry();});
});
el["create-mesh"].addEventListener("click", () => loadGeometryPreview(true));
el["refine-mesh"].addEventListener("click", () => {
  const profiles=["standard","fine","very_fine"],i=profiles.indexOf(el["preview-resolution"].value);
  el["preview-resolution"].value=profiles[Math.min(profiles.length-1,i+1)];
  el["cfd-resolution"].value=el["preview-resolution"].value;
  loadGeometryPreview(true);
});
el["preview-resolution"].addEventListener("change", () => {el["cfd-resolution"].value=el["preview-resolution"].value;if(state.preview?.mesh)loadGeometryPreview(true);else invalidatePreviewMesh();});
el["cfd-resolution"].addEventListener("change", () => {el["preview-resolution"].value=el["cfd-resolution"].value;invalidatePreviewMesh();});
["reynolds","alpha","cfd-n-base"].forEach(id=>el[id].addEventListener("input",invalidatePreviewMesh));
el["run-button"].addEventListener("click", runAnalysis);
el["result-backend"].addEventListener("change", renderAlphaChoices);
el["result-alpha"].addEventListener("change", renderPoint);
el["field-kind"].addEventListener("change", () => renderField());
el["field-renderer"].addEventListener("change", () => renderField());
el["field-grid"].addEventListener("change", () => {state.gridVisible=el["field-grid"].checked;Object.values(spatialViews).forEach(view=>view?.schedule());for(const view of chartViews.values())view.updateSpec(view.spec);});
el["field-mesh-overlay"].addEventListener("change", () => {state.meshOverlay=el["field-mesh-overlay"].checked;renderField();});
document.querySelectorAll("input[name='airfoil-label']").forEach(radio=>radio.addEventListener("change",()=>{state.showAirfoilName=radio.value==="show";Object.values(spatialViews).forEach(view=>view?.schedule());}));
el["show-reference"].addEventListener("change", () => { renderPoint(); renderPolars(); });
el["theme-mode"].value = ["auto", "light", "dark"].includes(localStorage.getItem("airfoil-theme")) ? localStorage.getItem("airfoil-theme") : "dark";
el["theme-mode"].addEventListener("change", () => setTheme(el["theme-mode"].value));
document.querySelectorAll("[data-tab-target]").forEach(tab => {
  tab.addEventListener("click", () => switchTab(tab.dataset.tabTarget));
  tab.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    const tabs = [...document.querySelectorAll("[data-tab-target]")];
    const at = tabs.indexOf(tab);
    const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (at + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    event.preventDefault(); tabs[next].focus(); switchTab(tabs[next].dataset.tabTarget);
  });
});
el["download-result"].addEventListener("click", downloadResult);
el["log-view"].addEventListener("scroll", updateLogFollow);
el["log-follow"].addEventListener("click", () => { el["log-view"].scrollTop = el["log-view"].scrollHeight; updateLogFollow(); });
// Trailing- and leading-edge close-ups: the TE window holds the trailing edge, a Gurney plate and the near base; the LE
// window is centred on the JXfoil stagnation point when the mesh carries streamlines.
function closeUp(plot, which, view){
  const point=currentResult(),mesh=view==="field"?currentMesh(point):(state.preview?.mesh||currentMesh(point));
  const wall=meshWall(inspectableMesh(mesh));
  const xs=wall?.x?.map(Number)||[1],ys=wall?.y?.map(Number)||[0];
  if(which==="te"){const xte=Math.max(...xs.filter(finite)),near=ys.filter((y,i)=>xs[i]>xte-0.002),ylo=Math.min(-0.012,...near),yhi=Math.max(0.012,...near);
    plot.focus(xte-0.06,xte+0.09,ylo-0.025,yhi+0.02);return;}
  const sp=mesh?.streamlines?.stagnation_point,cx=Array.isArray(sp)?Number(sp[0]):Math.min(...xs.filter(finite)),cy=Array.isArray(sp)?Number(sp[1]):0;
  plot.focus(cx-0.09,cx+0.07,cy-0.07,cy+0.05);
}
document.querySelectorAll(".view-tools").forEach(tools => tools.addEventListener("click", event => {
  const action = event.target.closest("button")?.dataset.action;
  if (tools.dataset.view === "field" && el["field-renderer"].value === "3d") {
    if (action === "center") window.AirfoilBabylon?.center();
    else if (action === "zoom-in") window.AirfoilBabylon?.zoom(1.5);
    else if (action === "zoom-out") window.AirfoilBabylon?.zoom(1 / 1.5);
    return;
  }
  const data=tools.dataset.view==="field"?chartViews.get("field-view"):null;
  if(data&&(action==="te"||action==="le"))return;
  if(data&&action){if(action==="center")data.fit();else data.zoomAt(action==="zoom-in"?1.5:1/1.5,{x:data.width/2,y:data.height/2});return;}
  const plot = spatialViews[tools.dataset.view==="geometry"&&el["geometry-display"].value==="curvature" ? "curvature" : tools.dataset.view];
  if (!plot || !action) return;
  if (action === "te" || action === "le") {closeUp(plot, action, tools.dataset.view); return;}
  if (action === "center") plot.center();
  else if (action === "zoom-in") plot.zoomAt(1.5);
  else if (action === "zoom-out") plot.zoomAt(1/1.5);
}));
window.addEventListener("ludwig-palette-change",()=>{if(state.activeTab==="field")renderField();else spatialViews.field?.schedule();});
setTheme(el["theme-mode"].value); updateSetup(); loadHealth(); scheduleGeometryPreview();

// Saved results (a separate line: the Node viewer tests remove the line above exactly).
loadRecentResults();
