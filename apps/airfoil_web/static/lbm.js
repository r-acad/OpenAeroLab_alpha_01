"use strict";
/* LBM 3D (LUDWIG) page code (27 Oct 2026). Loaded after app.js and using its globals (el, state, SpatialPlot,
   spatialViews, spatialMemory, safeHref, escapeHtml, numberText, finite, selectedBackends, invalidatePreviewMesh,
   scheduleGeometryPreview, updateSetup, renderFieldChoices, currentResult). app.js calls it through window.AirfoilLBM
   hooks only; with this file absent every other backend behaves exactly as before.
   - the form: span/chord, cells per chord, run length, frames, domain, plate thickness, wall model, the spoiler;
   - Create mesh: the 3D geometry with the domain box, periodic faces and mid-span plane (Babylon), the mid-span lattice
     (cells, 8-cell blocks and the solver's solid cells), the section with its thick Gurney plate or spoiler;
   - results: the mid-plane fields with a frame slider and player, the MP4 videos with download links, the live frames of
     a running case. On the lbm-io-opts solver profile ("lbmio") they are the solver's NUMERIC mid-span plane (|u|/U,
     vorticity, Cp, u, w, v) drawn cell by cell on fixed, editable colour ranges with a value probe; on the first profile
     ("lbmapp") the solver's slice pictures with the section drawn over them. A domain check ("lbmio") compares the
     solver's own mid-span obstacle mask with the app's in the lattice view. */
(() => {
  const $ = id => document.getElementById(id);
  // = lbm_case.jl's estimate constants (the server's lattice and estimate after Create mesh are the authority)
  // measured on the integration cases (27 Oct 2026; lbm_case.jl); the service's /api/health constants win (onHealth)
  const EST = {bytesPerCell: 528, fixedGb: 0.7, mlups: 116, limitGb: 9.0, uLattice: 0.08, tauMin: 0.501, startupS: 600, exportPerNode: 3.7e-6};
  // = lbm_case.jl's periodic-AMR figures (record T8: NACA 0012, 3 levels, 384 cells/c finest; one measured case)
  const AMR = {bytes: 462, wallBytes: 700, rate: [116e6, 67.4e6, 28.5e6, 28.5e6], fixed: [0, 0.0452, 0.143, 0.286], N: 384, finestArea: 0.2253, nextArea: 2.085, perimeter: 2.04, area: 0.0822, shell: 2};
  let maxLevels = 1, noSlip = false, levelVisc = false, levelViscTip = "";
  const FIELD_LABELS = {velocity: "Velocity magnitude", vorticity: "Spanwise vorticity", pressure: "Pressure", density: "Density",
    mach: "Mach number", schlieren: "Schlieren |∇ρ|", velocity_highpass: "Velocity high-pass"};
  const FIELD_ORDER = ["velocity", "vorticity", "pressure", "density", "mach", "schlieren", "velocity_highpass"];
  const SPEC = [["lbm-span-ratio", "lbm_span_ratio", "span / chord ratio", 0.05, 2, false],
    ["lbm-cells", "lbm_cells_per_chord", "cells per chord", 16, 512, true],
    ["lbm-ctu", "lbm_ctu", "run length in convective times", 0.05, 50, false],
    ["lbm-frames", "lbm_frames", "number of mid-plane frames", 2, 400, true],
    ["lbm-upstream", "lbm_upstream", "upstream extent (chords)", 0.5, 10, false],
    ["lbm-downstream", "lbm_downstream", "downstream extent (chords)", 1, 20, false],
    ["lbm-height", "lbm_height", "extent above and below (chords)", 0.5, 10, false],
    ["lbm-plate-cells", "lbm_plate_cells", "plate thickness in lattice cells", 1.5, 8, false]];
  const SPOILER = [["lbm-spoiler-hinge", "lbm_spoiler_hinge", "spoiler hinge x/c", 0.2, 0.95],
    ["lbm-spoiler-length", "lbm_spoiler_length", "spoiler length l/c", 0.01, 0.3],
    ["lbm-spoiler-deflection", "lbm_spoiler_deflection", "spoiler deflection (deg)", 10, 90]];
  const ceil8 = v => 8 * Math.ceil(v / 8);
  const active = (backends = selectedBackends(el.backend.value)) => backends.includes("lbm");

  // ------------------------------------------------------------------------------------------------ form
  function flowKeys(flow = {}, strict = true) {
    const fail = message => { if (strict) throw new Error(message); return null; };
    const read = (id, label, lo, hi, integer) => {
      const raw = $(id)?.value, v = Number(raw);
      if (raw == null || String(raw).trim() === "" || !Number.isFinite(v) || v < lo || v > hi || (integer && !Number.isInteger(v)))
        return fail(`Enter a valid LBM ${label} (${lo} to ${hi}${integer ? ", a whole number" : ""}).`);
      return v;
    };
    const keys = {};
    for (const [id, key, label, lo, hi, integer] of SPEC) { const v = read(id, label, lo, hi, integer); if (v === null) return null; keys[key] = v; }
    keys.lbm_wall_model = !!$("lbm-wall-model")?.checked;
    const walls = $("lbm-z-walls")?.value || "slip";
    if (walls === "no_slip" && !noSlip) return fail("No-slip top and bottom walls need the solver update (lbm-zwalls); use slip walls.");
    keys.lbm_z_walls = walls;
    const levels = Number($("lbm-levels")?.value || 1);
    if ($("lbm-level-visc")?.checked && levels > 1) {
      if (!levelVisc) return fail("The physical viscosity on coarse levels needs the solver update (lbm-taufix); untick it.");
      keys.lbm_level_viscosity = "physical";
    }
    if (levels > 1) {
      if (levels > maxLevels) return fail(`This Julia service's LBM solver runs ${maxLevels === 1 ? "one level only" : `at most ${maxLevels} levels`}.`);
      keys.lbm_levels = levels;
    }
    if ($("lbm-spoiler")?.checked) {
      keys.lbm_spoiler = true;
      for (const [id, key, label, lo, hi] of SPOILER) { const v = read(id, label, lo, hi, false); if (v === null) return null; keys[key] = v; }
    }
    if (strict) {
      const re = Number(flow.reynolds);
      if (!(re >= 100 && re <= 1e6)) throw new Error("LBM 3D (LUDWIG) takes Reynolds numbers from 100 to 1 million.");
      if (Number(flow.cfd_te_thickness) > 0) throw new Error("The LBM 3D backend extrudes the section's own trailing edge: clear the CFD trailing-edge thickness override.");
      if (state.serviceRevision != null && state.serviceRevision < 15) throw new Error("Restart run.cmd and use the browser tab it opens: this Julia service predates the LBM 3D backend's numeric mid-plane results (API revision 15).");
      keys.mach = 0;   // not an input of the LBM (incompressible regime at lattice Mach 0.139)
    }
    return keys;
  }
  /* tau and viscosity relative to physical of each level (coarsest first) = lbm_case.jl lbm_level_taus */
  function levelTaus(keys, re) {
    const L = keys.lbm_levels || 1, N = keys.lbm_cells_per_chord, tauF = 0.5 + 3 * EST.uLattice * N / re, tf = Math.max(tauF, EST.tauMin), out = [];
    for (let l = 1; l <= L; l++) {
      const k = 2 ** (L - l), phys = (tauF - 0.5) / k;
      let t = keys.lbm_level_viscosity === "physical" ? 0.5 + (tf - 0.5) / k : 0.5 + (tf - 0.5) * k;
      const floored = t < EST.tauMin || (l === L && tauF < EST.tauMin);
      t = Math.max(t, EST.tauMin);
      out.push({level: l, tau: t, ratio: (t - 0.5) / phys, floored});
    }
    return out;
  }
  function estimate(keys, re) {
    const N = keys.lbm_cells_per_chord, L = keys.lbm_levels || 1, f = 2 ** (L - 1), q = 8 * f;
    const span = q * Math.max(1, Math.round(keys.lbm_span_ratio * N / q));
    const coarse = len => ceil8(Math.ceil(len * N / f));      // coarsest cells, whole 8-cell blocks
    const ncx = coarse(keys.lbm_upstream + keys.lbm_downstream + 1.0), ncz = coarse(0.13 + 2 * keys.lbm_height), ncy = span / f;
    const nx = ncx * f, nz = ncz * f;
    const levelCells = [ncx * ncy * ncz];
    if (L > 1) {   // = lbm_amr_level_cells (NACA 0012-like section; Create mesh uses the real one)
      const dom = nx * nz / N / N, A = [];
      const k = (AMR.finestArea - AMR.area) / (AMR.perimeter * AMR.shell * 8 / AMR.N);
      A[L] = Math.min(dom, AMR.area + k * AMR.perimeter * 2 * 8 / N);
      for (let l = L - 1; l >= 2; l--) A[l] = l === L - 1 ? Math.min(dom, A[L] + AMR.nextArea - AMR.finestArea) : Math.min(dom, A[l + 1] + 6.2 * 16 * 2 ** (L - l) / N);
      for (let l = 2; l <= L; l++) { const Nl = N / 2 ** (L - l); levelCells.push(Math.round(A[l] * Nl * Nl * span / 2 ** (L - l))); }
    }
    const cells = levelCells.reduce((a, b) => a + b, 0);
    const gb = L > 1 ? levelCells.reduce((a, c, i) => a + c * (i === L - 1 ? AMR.wallBytes : AMR.bytes), 0) / 1e9 + EST.fixedGb : cells * EST.bytesPerCell / 1e9 + EST.fixedGb;
    // one solver step is one COARSEST-level step (2^(L-1) finest steps; W7): c/U = N / (u_lattice 2^(L-1)) solver steps
    const stepsPerCtu = N / (EST.uLattice * f), steps0 = Math.max(keys.lbm_frames, Math.round(keys.lbm_ctu * stepsPerCtu));
    const steps = 8 * Math.ceil(Math.ceil(steps0 / keys.lbm_frames) / 8) * keys.lbm_frames;
    const updates = L > 1 ? levelCells.reduce((a, c, i) => a + c * 2 ** i, 0) : cells;   // cell updates per solver step
    const stepS = L > 1 ? AMR.fixed[L - 1] + updates / AMR.rate[L - 1] : cells / (EST.mlups * 1e6);
    const nodes = L > 1 ? Math.round(3.6 * N) * Math.round(1.73 * N) : nx * nz;   // the mid-plane (windowed with several levels)
    const march = stepS * steps, seconds = EST.startupS + march + (keys.lbm_frames + 1) * (0.05 + EST.exportPerNode * nodes) + 5;
    const tau = 0.5 + 3 * EST.uLattice * N / re;
    return {N, L, span, nx, nz, ncx, ncy, ncz, cells, levelCells, gb, steps, stepsPerCtu, seconds, tau, quantum: q, fits: gb <= EST.limitGb, reEff: tau < EST.tauMin ? EST.uLattice * N / ((EST.tauMin - 0.5) / 3) : re};
  }
  const millions = n => n >= 1e6 ? `${(n / 1e6).toFixed(2)} M` : n.toLocaleString();
  const minutes = s => s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`;
  function updateLbmSetup(backends = selectedBackends(el.backend.value)) {
    const on = active(backends);
    $("lbm-controls")?.classList.toggle("hidden", !on);
    $("lbm-spoiler-group")?.classList.toggle("hidden", !on);
    const spoiler = on && !!$("lbm-spoiler")?.checked;
    $("lbm-spoiler-controls")?.classList.toggle("hidden", !spoiler);
    $("lbm-spoiler-note")?.classList.toggle("hidden", !spoiler);
    if (!on) return;
    el.mach.disabled = true; el.mach.title = "Not an input of the LBM 3D backend: it runs the incompressible regime at lattice Mach 0.139.";
    el.ncrit.disabled = true; el["transition-mode"].value = "default"; el["transition-mode"].disabled = true;
    el["trip-controls"].classList.add("hidden");
    const keys = flowKeys({}, false), re = Number(el.reynolds.value);
    el["flow-hint"].textContent = "Reynolds (chord) and angle set the LBM 3D case. Mach, Ncrit and trips are not used: the lattice resolves the flow directly (no transition or turbulence model besides the solver's WALE subgrid viscosity).";
    el["backend-note"].textContent = "LBM 3D (LUDWIG): the workspace's lattice-Boltzmann solver, run as it is from a pinned snapshot, on the section extruded along the span with periodic span faces (one uniform level, or several refinement levels when the service's solver has periodic-Y AMR). The run waits for the shared GPU lock; the live charts show CL and CD; the solver's mid-span plane becomes a frame player of its numeric fields on fixed colour scales and MP4 videos. Development path, not validated for airfoils.";
    const g = gurneyEnabled() ? gurneyFlow(false) : null, plate = keys ? keys.lbm_plate_cells : NaN, N = keys ? keys.lbm_cells_per_chord : NaN;
    const parts = [];
    if (g) parts.push(`Gurney flap h/c ${g.gurney_height} (${g.gurney_mode}): a THICK plate of ${plate} cells (${finite(plate / N) ? (plate / N).toFixed(4) : "?"} c) at the trailing edge, ${finite(g.gurney_height * N) ? (g.gurney_height * N).toFixed(1) : "?"} cells high.`);
    if (spoiler && keys?.lbm_spoiler) parts.push(`Spoiler: hinge x/c ${keys.lbm_spoiler_hinge}, l/c ${keys.lbm_spoiler_length} (${(keys.lbm_spoiler_length * N).toFixed(1)} cells), ${keys.lbm_spoiler_deflection}° from the surface tangent, ${plate} cells thick.`);
    if (finite(plate) && plate < 2) parts.push("A plate thinner than 2 cells is at the edge of what the immersed boundary resolves.");
    if (keys && finite(re) && re > 0 && (keys.lbm_levels || 1) > 1) {
      const t = levelTaus(keys, re);
      parts.push(`Coarse-level viscosity (${keys.lbm_level_viscosity === "physical" ? "PHYSICAL option, experimental, its stability screen did not pass, not validated; peak coarse-level velocity was 9-20 % above the legacy twin's in that screen" : "the solver's legacy rule, 4^(L-l) x physical"}): ` +
        t.map(x => `level ${x.level} tau ${x.tau.toFixed(4)}${x.floored ? " (floored at 0.501)" : ""}, ${x.ratio.toPrecision(3)} x physical`).join("; ") + ".");
    }
    parts.push("Friction (viscous) drag from the LBM surface-force routine is underestimated (measured 0.22-0.71 of exact in a laminar channel test); drag values are mostly pressure drag. Under investigation.");
    parts.push("The LBM curved-wall conditions place the effective wall about 1.1-1.3 cells inside the fluid (measured in laminar channel tests; independent of resolution), so bodies are effectively about that much thicker on each side and surface shear is misread. Root cause identified (a write-location defect in the linear Bouzidi family, a double relaxation in the grad condition, and the solid-cell marking); fixes are in development. LBM results are development-only until then.");
    parts.push(`Top and bottom walls: ${keys?.lbm_z_walls === "no_slip" ? "no-slip (stationary; each wall grows a boundary layer from the inlet, which may or may not be wanted, depending on what the case represents)" : "slip (an exact specular reflection; the subgrid models keep a small non-mirrored ghost at the faces, 1 to 3.4e-4 U, so it is nearly a symmetry plane)"}. Either wall type makes the domain a tunnel, not free air: the lift depends on the domain height (blockage and streamline curvature), more so for small heights.`);
    if (el["sweep-enabled"].checked) parts.push("Each angle of a sweep is a separate LBM run (at most 12).");
    $("lbm-note") && ($("lbm-note").textContent = parts.join(" "));
    if (!keys || !finite(re) || re <= 0) { $("lbm-estimate") && ($("lbm-estimate").textContent = "Enter valid LBM settings to see the lattice estimate."); return; }
    const e = estimate(keys, re);
    const spanText = `span ${e.span} cells = ${(e.span / e.N).toFixed(4)} c${Math.abs(e.span / e.N - keys.lbm_span_ratio) > 1e-9 ? ` (snapped from ${keys.lbm_span_ratio} c to whole ${e.quantum}-cell blocks)` : ""}`;
    $("lbm-estimate").textContent = (e.L > 1 ?
      `${e.L} levels: about ${millions(e.cells)} cells (${e.levelCells.map(millions).join(" / ")}, coarsest first; coarsest grid ${e.ncx} × ${e.ncy} × ${e.ncz}), ${spanText}, about ${e.gb.toFixed(1)} GB of the 10 GB card (${AMR.bytes} / ${AMR.wallBytes} B/cell + ${EST.fixedGb} GB, measured on the integration cases; cells per level scaled from one measured case), ${e.steps.toLocaleString()} solver steps (one solver step = one coarsest step = ${2 ** (e.L - 1)} finest steps; ${Math.round(e.stepsPerCtu).toLocaleString()} per c/U), about ${minutes(e.seconds)} on the RTX 3080 (Create mesh refines it).` :
      `About ${millions(e.cells)} cells (${e.nx} × ${e.span} × ${e.nz}), ${spanText}, about ${e.gb.toFixed(1)} GB of the 10 GB card (${EST.bytesPerCell} B/cell + ${EST.fixedGb} GB, measured on the integration cases), ${e.steps.toLocaleString()} steps, about ${minutes(e.seconds)} on the RTX 3080 (estimate; Create mesh gives the exact lattice).`) +
      (e.fits ? "" : ` Above the ${EST.limitGb} GB limit: reduce the cells per chord, the span or the domain.`) +
      (e.tau < EST.tauMin ? ` tau ${e.tau.toFixed(5)} is below the solver's floor ${EST.tauMin}: the effective Reynolds number is about ${Math.round(e.reEff).toLocaleString()}; raise the cells per chord.` : ` tau ${e.tau.toFixed(5)}.`);
    $("lbm-estimate").classList.toggle("warning-text", !e.fits || e.tau < EST.tauMin);
  }
  function onHealth(caps) {
    maxLevels = Math.max(1, Number(caps?.levels) || 1);
    const e = caps?.estimate || {};   // the service's own constants (lbm_case.jl) win over this file's copies
    if (Number(e.bytes_per_cell) > 0) EST.bytesPerCell = Number(e.bytes_per_cell);
    if (Number(e.fixed_gb) > 0) EST.fixedGb = Number(e.fixed_gb);
    if (Number(e.mlups) > 0) EST.mlups = Number(e.mlups);
    if (Number(e.amr_bytes_per_cell) > 0) AMR.bytes = Number(e.amr_bytes_per_cell);
    if (Number(e.amr_bytes_per_wall_cell) > 0) AMR.wallBytes = Number(e.amr_bytes_per_wall_cell);
    const m = e.amr_step_model || {};
    if (Array.isArray(m.rate_per_s) && m.rate_per_s.length >= 3) AMR.rate = m.rate_per_s.map(Number);
    if (Array.isArray(m.fixed_s) && m.fixed_s.length >= 3) AMR.fixed = m.fixed_s.map(Number);
    if (Number(e.export_seconds_per_node) > 0) EST.exportPerNode = Number(e.export_seconds_per_node);
    noSlip = Array.isArray(caps?.z_walls) && caps.z_walls.includes("no_slip");
    levelVisc = caps?.level_viscosity === true;
    if (typeof caps?.level_viscosity_tooltip === "string") levelViscTip = caps.level_viscosity_tooltip;
    const zw = $("lbm-z-walls");
    if (zw) {
      for (const o of Array.from(zw.options || zw.children || [])) if (o.value === "no_slip") { o.disabled = !noSlip; o.textContent = noSlip ? "No-slip" : "No-slip (needs the solver update)"; o.title = noSlip ? "" : "needs the solver update (lbm-zwalls)"; }
      if (!noSlip && zw.value === "no_slip") zw.value = "slip";
    }
    const lv = $("lbm-level-visc");
    if (lv) { lv.disabled = !levelVisc; if (!levelVisc) lv.checked = false; }
    const lvl = $("lbm-level-visc-line");
    if (lvl) lvl.title = (levelVisc ? "" : "Needs the solver update (lbm-taufix). ") + (levelViscTip || lvl.title);
    const sel = $("lbm-levels");
    if (!sel) return;
    for (const o of Array.from(sel.options || sel.children || [])) o.disabled = Number(o.value) > maxLevels;
    if (Number(sel.value) > maxLevels) sel.value = "1";
    sel.disabled = maxLevels === 1;
    sel.title = maxLevels === 1 ? "One uniform level: this service's LBM solver snapshot applies the periodic span to single-level runs only (periodic-Y AMR is not pinned yet)." :
      `Grid levels with periodic-Y AMR (1 to ${maxLevels}): cells per chord is the finest level's; each coarser level halves it; the span snaps to whole coarsest blocks.`;
    updateSetup();
  }
  function applySaved(flow) {
    for (const [id, key] of SPEC) if (finite(Number(flow?.[key])) && $(id)) $(id).value = String(flow[key]);
    if ($("lbm-levels")) $("lbm-levels").value = String(Number(flow?.lbm_levels) > 1 ? flow.lbm_levels : 1);
    if ($("lbm-z-walls")) $("lbm-z-walls").value = flow?.lbm_z_walls === "no_slip" && noSlip ? "no_slip" : "slip";
    if ($("lbm-level-visc")) $("lbm-level-visc").checked = flow?.lbm_level_viscosity === "physical" && levelVisc;
    if ($("lbm-wall-model")) $("lbm-wall-model").checked = flow?.lbm_wall_model === true;
    if ($("lbm-spoiler")) $("lbm-spoiler").checked = flow?.lbm_spoiler === true;
    for (const [id, key] of SPOILER) if (finite(Number(flow?.[key])) && $(id)) $(id).value = String(flow[key]);
  }

  // ------------------------------------------------------------------------------------------------ geometry tab
  const added = {};
  function syncGeometryOptions(preview) {
    const select = el["geometry-display"], has = !!preview?.lbm;
    for (const [value, text] of [["lbm3d", "LBM 3D geometry + domain"], ["lbmlattice", "LBM mid-span lattice"]]) {
      if (has && !added[value]) { const o = document.createElement("option"); o.value = value; o.textContent = text; select.append(o); added[value] = o; }
      if (!has && added[value]) { added[value].remove(); delete added[value]; }
    }
  }
  function displayAfterMesh(preview) { syncGeometryOptions(preview); return preview?.lbm ? "lbm3d" : "outline"; }
  function sectionOf(preview) {
    if (!active()) return null;
    const s = preview?.lbm?.section || preview?.lbm_section;
    return s && Array.isArray(s.x) && Array.isArray(s.z) && s.x.length > 2 ? s : null;
  }
  function drawSection(ctx, plot, preview) {
    const s = sectionOf(preview);
    if (!s) return;
    const css = getComputedStyle(document.documentElement);
    ctx.save(); ctx.beginPath();
    s.x.forEach((x, i) => { const px = plot.x(Number(x)), py = plot.y(Number(s.z[i])); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
    ctx.closePath(); ctx.fillStyle = "rgba(122,79,181,0.18)"; ctx.fill();
    ctx.strokeStyle = css.getPropertyValue("--orange").trim() || "#df713e"; ctx.lineWidth = 1.5; ctx.stroke(); ctx.restore();
  }
  function renderGeometry(mode, preview) {
    syncGeometryOptions(preview);
    const lbm = preview?.lbm;
    if (!["lbm3d", "lbmlattice"].includes(mode)) { dispose3D("geometry"); return false; }
    if (!lbm) { el["geometry-display"].value = "outline"; dispose3D("geometry"); return false; }
    for (const id of ["curvature-scale-field", "curvature-mapping-field", "curvature-mapping-help", "curvature-chart"]) el[id]?.classList.add("hidden");
    el["geometry-chart"].classList.remove("hidden");
    el["geometry-title"].textContent = `${preview.geometry?.name || "Airfoil"} · ${mode === "lbm3d" ? "LBM 3D geometry and domain" : "LBM mid-span lattice"}`;
    el["geometry-status"].textContent = statusText(lbm);
    if (mode === "lbm3d") show3D(el["geometry-chart"], lbm, "geometry");
    else { dispose3D("geometry"); latticeView("geometry", el["geometry-chart"], lbm); }
    return true;
  }
  function statusText(lbm) {
    const d = lbm?.domain || {}, s = lbm?.settings || {}, e = lbm?.estimate || {}, stl = lbm?.stl || {};
    if (Number(d.levels) > 1) {
      const lc = (e.level_cells || []).map(Number);
      return `LBM ${d.levels} levels (periodic-Y AMR): coarsest grid ${(d.coarse_cells || []).join(" × ")} cells in blocks of ${8 * 2 ** (d.levels - 1)} finest cells · ${s.cells_per_chord} cells per chord at the finest level · about ${millions(Number(e.cells))} cells in all (${lc.map(millions).join(" / ")}, an estimate) · span ${d.ny} finest cells = ${numberText(Number(s.span), 4)} c (periodic) · ` +
        `${finite(Number(e.memory_gb)) ? `about ${Number(e.memory_gb).toFixed(1)} GB` : ""}${finite(Number(e.total_seconds)) ? `, about ${minutes(Number(e.total_seconds))}` : ""} · ${Number(s.steps).toLocaleString()} solver steps of the coarsest level (${numberText(Number(s.ctu), 3)} CTU) · STL ${Number(stl.triangles).toLocaleString()} triangles${stl.closed ? ", closed" : ""} · basic.alpha ${numberText(Number(lbm.basic_alpha_deg), 2)}° · ${s.z_walls === "no_slip" ? "no-slip" : "slip"} top and bottom walls`;
    }
    return `LBM lattice ${d.nx} × ${d.ny} × ${d.nz} = ${millions(Number(d.cells))} cells, ${Number(d.blocks).toLocaleString()} blocks of 8³ · ${s.cells_per_chord} cells per chord · span ${d.ny} cells = ${numberText(Number(s.span), 4)} c (periodic) · ` +
      `${finite(Number(e.memory_gb)) ? `about ${Number(e.memory_gb).toFixed(1)} GB` : ""}${finite(Number(e.total_seconds)) ? `, about ${minutes(Number(e.total_seconds))}` : ""} · ${Number(s.steps).toLocaleString()} steps (${numberText(Number(s.ctu), 3)} CTU) · STL ${Number(stl.triangles).toLocaleString()} triangles${stl.closed ? ", closed" : ""} · basic.alpha ${numberText(Number(lbm.basic_alpha_deg), 2)}°${s.z_walls ? ` · ${s.z_walls === "no_slip" ? "no-slip" : "slip"} top and bottom walls` : ""}`;
  }

  // ------------------------------------------------------------------------------------------------ mid-span lattice (2D)
  function latticeView(kind, node, lbm) {
    const d = lbm.domain, off = d.offset, dx = Number(d.dx), m = lbm.mask, sec = lbm.placed;
    const mc = lbm.mask_check, sm = mc?.compared && Array.isArray(mc.solver_cells) && mc.solver_cells.length === m.ni * m.nk ? mc.solver_cells : null;
    const ox = Number(off[0]), oz = Number(off[2]);
    const X = i => (i - 1) * dx - ox, Z = k => (k - 1) * dx - oz;            // cell i spans [X(i), X(i+1)]
    const xs = sec.x.map(Number), zs = sec.z.map(Number);
    const bounds = {xmin: Math.min(...xs) - ox - 0.12, xmax: Math.max(...xs) - ox + 0.12, ymin: Math.min(...zs) - oz - 0.12, ymax: Math.max(...zs) - oz + 0.12};
    const css = getComputedStyle(document.documentElement), ink = css.getPropertyValue("--ink").trim() || "#17354b";
    const painter = (ctx, plot) => {
      const cellPx = dx * plot.scale;
      const i0 = Math.max(1, Math.floor((plot.worldX(0) + ox) / dx)), i1 = Math.min(d.nx + 1, Math.ceil((plot.worldX(plot.width) + ox) / dx) + 1);
      const k0 = Math.max(1, Math.floor((plot.worldY(plot.height) + oz) / dx)), k1 = Math.min(d.nz + 1, Math.ceil((plot.worldY(0) + oz) / dx) + 1);
      // solid cells (the solver's rule)
      ctx.fillStyle = "rgba(70,80,95,0.55)";
      for (let k = 0; k < m.nk; k++) for (let i = 0; i < m.ni; i++) {
        if (!m.cells[k * m.ni + i]) continue;
        const gi = m.i0 + i, gk = m.k0 + k;
        ctx.fillRect(plot.x(X(gi)), plot.y(Z(gk + 1)), cellPx + 0.5, cellPx + 0.5);
      }
      // the solver's own mid-span mask where it differs from the app's (red: solid only for the solver; blue: only for the app)
      if (sm) for (let k = 0; k < m.nk; k++) for (let i = 0; i < m.ni; i++) {
        const q = k * m.ni + i, a = !!m.cells[q], b = !!sm[q];
        if (a === b) continue;
        ctx.fillStyle = b ? "rgba(214,48,49,0.85)" : "rgba(40,110,220,0.85)";
        ctx.fillRect(plot.x(X(m.i0 + i)), plot.y(Z(m.k0 + k + 1)), cellPx + 0.5, cellPx + 0.5);
      }
      // cell lines and block lines
      const line = (x0, y0, x1, y1) => { ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); };
      if (cellPx >= 4) {
        ctx.beginPath(); ctx.strokeStyle = "rgba(60,110,140,0.28)"; ctx.lineWidth = 0.6;
        for (let i = i0; i <= i1; i++) line(plot.x(X(i)), plot.y(Z(k0)), plot.x(X(i)), plot.y(Z(k1)));
        for (let k = k0; k <= k1; k++) line(plot.x(X(i0)), plot.y(Z(k)), plot.x(X(i1)), plot.y(Z(k)));
        ctx.stroke();
      }
      if (8 * cellPx >= 4) {
        ctx.beginPath(); ctx.strokeStyle = "rgba(122,79,181,0.75)"; ctx.lineWidth = 1.2;
        for (let i = i0; i <= i1; i++) if ((i - 1) % 8 === 0) line(plot.x(X(i)), plot.y(Z(k0)), plot.x(X(i)), plot.y(Z(k1)));
        for (let k = k0; k <= k1; k++) if ((k - 1) % 8 === 0) line(plot.x(X(i0)), plot.y(Z(k)), plot.x(X(i1)), plot.y(Z(k)));
        ctx.stroke();
      }
      // domain boundary
      ctx.strokeStyle = ink; ctx.lineWidth = 2; ctx.strokeRect(plot.x(X(1)), plot.y(Z(d.nz + 1)), d.nx * cellPx, d.nz * cellPx);
      // the section
      ctx.beginPath(); xs.forEach((x, i) => { const px = plot.x(x - ox), py = plot.y(zs[i] - oz); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.closePath(); ctx.strokeStyle = css.getPropertyValue("--orange").trim() || "#df713e"; ctx.lineWidth = 1.6; ctx.stroke();
      ctx.font = "11px Segoe UI, Arial, sans-serif";
      const note = `Mid-span plane (y = ${numberText(Number(d.mid_span_y), 4)} c) · ${Number(d.levels) > 1 ? `the finest of ${d.levels} levels near the body, ` : ""}cell dx = 1/${lbm.settings.cells_per_chord} c (thin lines from 4 px) · 8-cell blocks (purple) · solid cells by the solver's voxel rule (${m.solid} in view window)` +
        (sm ? (mc.matches ? ` · the solver's own mask (build-only run) is identical cell by cell` : ` · the solver's own mask differs in ${mc.differing_cells} cells (red: solver only, blue: app only)`) : "");
      const tw = Number(ctx.measureText?.(note)?.width) || 600;
      ctx.fillStyle = "rgba(255,255,255,0.82)"; ctx.fillRect(34, 6, tw + 12, 18);
      ctx.fillStyle = ink; ctx.fillText(note, 40, 19);
    };
    const probe = (x, y) => {
      const i = Math.floor((x + ox) / dx) + 1, k = Math.floor((y + oz) / dx) + 1;
      if (i < 1 || k < 1 || i > d.nx || k > d.nz) return "";
      const wi = i - m.i0, wk = k - m.k0, solid = wi >= 0 && wk >= 0 && wi < m.ni && wk < m.nk && m.cells[wk * m.ni + wi];
      const own = sm && wi >= 0 && wk >= 0 && wi < m.ni && wk < m.nk ? ` · solver: ${sm[wk * m.ni + wi] ? "solid" : "fluid"}` : "";
      return `cell i ${i}, k ${k} · block ${Math.ceil(i / 8)}, ${Math.ceil(k / 8)} · ${solid ? "solid" : "fluid"}${own}`;
    };
    spatialViews[kind]?.destroy();
    spatialMemory[kind] = null;
    new SpatialPlot(kind, node, bounds, painter, probe, {axisUnits: "chord"});
  }

  // ------------------------------------------------------------------------------------------------ 3D geometry (Babylon)
  const views3d = {};
  function dispose3D(kind) {
    const v = views3d[kind];
    if (!v) return;
    delete views3d[kind];
    try { v.observer?.disconnect(); v.listeners.forEach(([n, f]) => v.canvas.removeEventListener(n, f)); v.camera.detachControl(); v.scene.dispose(); v.engine.dispose(); } catch {}
  }
  function earclip(px, pz) {
    const n = px.length, idx = [...Array(n).keys()], out = [];
    const cross = (a, b, c) => (px[b] - px[a]) * (pz[c] - pz[b]) - (pz[b] - pz[a]) * (px[c] - px[b]);
    const inside = (a, b, c, p) => {
      const s1 = (px[b] - px[a]) * (pz[p] - pz[a]) - (pz[b] - pz[a]) * (px[p] - px[a]);
      const s2 = (px[c] - px[b]) * (pz[p] - pz[b]) - (pz[c] - pz[b]) * (px[p] - px[b]);
      const s3 = (px[a] - px[c]) * (pz[p] - pz[c]) - (pz[a] - pz[c]) * (px[p] - px[c]);
      return s1 >= 0 && s2 >= 0 && s3 >= 0;
    };
    let guard = 0;
    while (idx.length > 3 && guard++ < 10 * n) {
      let cut = false;
      for (let k = 0; k < idx.length; k++) {
        const a = idx[(k + idx.length - 1) % idx.length], b = idx[k], c = idx[(k + 1) % idx.length];
        if (cross(a, b, c) <= 0) continue;
        if (idx.some(p => p !== a && p !== b && p !== c && inside(a, b, c, p))) continue;
        out.push(a, b, c); idx.splice(k, 1); cut = true; break;
      }
      if (!cut) break;
    }
    if (idx.length === 3) out.push(...idx);
    return out;
  }
  function show3D(node, lbm, kind) {
    dispose3D(kind);
    spatialViews[kind]?.destroy(); spatialViews[kind] = null;
    if (typeof BABYLON === "undefined") { node.classList.add("empty-chart"); node.textContent = "The 3D view needs WebGL (Babylon.js is unavailable in this browser)."; return false; }
    const d = lbm.domain, Lx = Number(d.Lx), Ly = Number(d.Ly), Lz = Number(d.Lz);
    const px = lbm.placed.x.map(Number), pz = lbm.placed.z.map(Number), n = px.length;
    node.classList.remove("empty-chart");
    const canvas = document.createElement("canvas"), caption = document.createElement("div");
    canvas.setAttribute("aria-label", "LBM 3D geometry and domain");
    caption.className = "babylon-caption";
    caption.textContent = `Domain ${numberText(Lx, 3)} × ${numberText(Ly, 4)} × ${numberText(Lz, 3)} c (x streamwise, span y, z up)${Number(d.levels) > 1 ? ` · ${d.levels} levels, coarsest blocks drawn` : ""} · periodic span faces (purple) · inlet (blue) · outlet (orange) · mid-span slice plane (green) · drag to orbit, right drag to pan, wheel to zoom, double click: body / domain`;
    node.replaceChildren(canvas, caption);
    let engine, scene, camera;
    try {
      engine = new BABYLON.Engine(canvas, true, {preserveDrawingBuffer: true, stencil: true});
      scene = new BABYLON.Scene(engine);
      const soft = (getComputedStyle(document.documentElement).getPropertyValue("--soft").trim() || "#f5f8f9").replace("#", "");
      const rgb = soft.length === 6 ? [0, 2, 4].map(i => parseInt(soft.slice(i, i + 2), 16) / 255) : [0.96, 0.97, 0.98];
      scene.clearColor = new BABYLON.Color4(rgb[0], rgb[1], rgb[2], 1);
      const V = (x, y, z) => new BABYLON.Vector3(x, z, y);   // solver (x, y span, z up) -> Babylon (x, y up, z)
      const cx = (Math.min(...px) + Math.max(...px)) / 2, cz = (Math.min(...pz) + Math.max(...pz)) / 2;
      const body = {target: V(cx + 0.05, Ly / 2, cz), radius: 2.1, alpha: -Math.PI / 2 - 0.42, beta: 1.12};
      const whole = {target: V(Lx / 2, Ly / 2, Lz / 2), radius: 1.5 * Math.max(Lx, Lz), alpha: -Math.PI / 2 - 0.8, beta: 1.0};
      camera = new BABYLON.ArcRotateCamera("lbm", -Math.PI / 2 - 0.42, 1.12, body.radius, body.target, scene);
      camera.lowerRadiusLimit = 0.05; camera.upperRadiusLimit = 100; camera.wheelDeltaPercentage = 0.06; camera.panningSensibility = 600;
      camera.minZ = 0.005; camera.inertia = 0; camera.panningInertia = 0;
      camera.attachControl(canvas, true);
      const light = new BABYLON.HemisphericLight("sky", new BABYLON.Vector3(0.2, 1, -0.6), scene); light.intensity = 1.0;
      light.groundColor = new BABYLON.Color3(0.45, 0.45, 0.5);
      const light2 = new BABYLON.DirectionalLight("key", new BABYLON.Vector3(-0.4, -0.8, 0.6), scene); light2.intensity = 0.45;
      // the body between the periodic faces (the STL runs 6 cells past each face; the solver sees this part)
      const pos = [], ind = [];
      for (let i = 0; i < n; i++) { pos.push(px[i], pz[i], 0); }
      for (let i = 0; i < n; i++) { pos.push(px[i], pz[i], Ly); }
      for (let i = 0; i < n; i++) { const j = (i + 1) % n; ind.push(i, j, n + j, i, n + j, n + i); }
      const caps = earclip(px, pz);
      for (let t = 0; t < caps.length; t += 3) { ind.push(caps[t], caps[t + 2], caps[t + 1]); ind.push(n + caps[t], n + caps[t + 1], n + caps[t + 2]); }
      const mesh = new BABYLON.Mesh("section", scene), vd = new BABYLON.VertexData(), normals = [];
      vd.positions = pos; vd.indices = ind; BABYLON.VertexData.ComputeNormals(pos, ind, normals); vd.normals = normals; vd.applyToMesh(mesh);
      const mat = new BABYLON.StandardMaterial("body", scene);
      mat.diffuseColor = new BABYLON.Color3(0.86, 0.56, 0.33); mat.emissiveColor = new BABYLON.Color3(0.18, 0.11, 0.06); mat.specularColor = new BABYLON.Color3(0.2, 0.2, 0.2); mat.backFaceCulling = false;
      mesh.material = mat;
      // the domain box
      const corners = [[0, 0, 0], [Lx, 0, 0], [Lx, Ly, 0], [0, Ly, 0], [0, 0, Lz], [Lx, 0, Lz], [Lx, Ly, Lz], [0, Ly, Lz]].map(c => V(...c));
      const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]].map(([a, b]) => [corners[a], corners[b]]);
      const box = BABYLON.MeshBuilder.CreateLineSystem("domain", {lines: edges}, scene); box.color = new BABYLON.Color3(0.15, 0.25, 0.35); box.isPickable = false;
      const plane = (name, pts, color, alpha) => {
        const m = new BABYLON.Mesh(name, scene), v = new BABYLON.VertexData();
        v.positions = pts.flatMap(p => [p.x, p.y, p.z]); v.indices = [0, 1, 2, 0, 2, 3]; const nn = []; BABYLON.VertexData.ComputeNormals(v.positions, v.indices, nn); v.normals = nn; v.applyToMesh(m);
        const mm = new BABYLON.StandardMaterial(name + "m", scene); mm.diffuseColor = color; mm.emissiveColor = color.scale(0.6); mm.alpha = alpha; mm.backFaceCulling = false; m.material = mm; m.isPickable = false;
        return m;
      };
      const purple = new BABYLON.Color3(0.48, 0.31, 0.71);
      plane("periodic y=0", [V(0, 0, 0), V(Lx, 0, 0), V(Lx, 0, Lz), V(0, 0, Lz)], purple, 0.06);
      plane("periodic y=Ly", [V(0, Ly, 0), V(Lx, Ly, 0), V(Lx, Ly, Lz), V(0, Ly, Lz)], purple, 0.06);
      plane("inlet", [V(0, 0, 0), V(0, Ly, 0), V(0, Ly, Lz), V(0, 0, Lz)], new BABYLON.Color3(0.24, 0.45, 0.85), 0.30);
      plane("outlet", [V(Lx, 0, 0), V(Lx, Ly, 0), V(Lx, Ly, Lz), V(Lx, 0, Lz)], new BABYLON.Color3(0.92, 0.50, 0.20), 0.30);
      for (const yy of [0, Ly]) { const e = BABYLON.MeshBuilder.CreateLines("periodic edge", {points: [V(0, yy, 0), V(Lx, yy, 0), V(Lx, yy, Lz), V(0, yy, Lz), V(0, yy, 0)]}, scene); e.color = purple; e.isPickable = false; }
      const ym = Number(d.mid_span_y) || Ly / 2;
      const mid = BABYLON.MeshBuilder.CreateLines("mid-span", {points: [V(0, ym, 0), V(Lx, ym, 0), V(Lx, ym, Lz), V(0, ym, Lz), V(0, ym, 0)]}, scene);
      mid.color = new BABYLON.Color3(0.15, 0.6, 0.3); mid.isPickable = false;
      // the 8-cell block lattice on the periodic face y = 0, near the body
      const dx = Number(d.dx), bsz = Number(d.block_m) || 8 * dx, lines = [];   // the coarsest level's blocks
      const bx0 = Math.max(0, Math.floor((Math.min(...px) - 0.3) / bsz)), bx1 = Math.min(d.bx, Math.ceil((Math.max(...px) + 0.6) / bsz));
      const bz0 = Math.max(0, Math.floor((Math.min(...pz) - 0.3) / bsz)), bz1 = Math.min(d.bz, Math.ceil((Math.max(...pz) + 0.3) / bsz));
      for (let b = bx0; b <= bx1; b++) lines.push([V(b * bsz, 0, bz0 * bsz), V(b * bsz, 0, bz1 * bsz)]);
      for (let b = bz0; b <= bz1; b++) lines.push([V(bx0 * bsz, 0, b * bsz), V(bx1 * bsz, 0, b * bsz)]);
      for (let b = 0; b <= d.by; b++) lines.push([V(bx0 * bsz, b * bsz, bz0 * bsz), V(bx1 * bsz, b * bsz, bz0 * bsz)]);
      const grid = BABYLON.MeshBuilder.CreateLineSystem("blocks", {lines}, scene); grid.color = purple; grid.alpha = 0.35; grid.isPickable = false;
      const view = {engine, scene, camera, canvas, listeners: [], observer: null, frame: null, framing: "body"};
      const paint = () => { if (view.frame != null) return; view.frame = requestAnimationFrame(() => { view.frame = null; if (views3d[kind] !== view) return; engine.resize(); scene.render(); }); };
      for (const name of ["pointermove", "pointerdown", "pointerup", "wheel"]) { const f = () => paint(); canvas.addEventListener(name, f, {passive: true}); view.listeners.push([name, f]); }
      const flip = which => { const to = typeof which === "string" ? which : view.framing === "body" ? "domain" : "body", f = to === "domain" ? whole : body;
        view.framing = to; camera.setTarget(f.target); camera.radius = f.radius; camera.alpha = f.alpha; camera.beta = f.beta; paint(); };
      view.flip = flip;
      canvas.addEventListener("dblclick", flip); view.listeners.push(["dblclick", flip]);
      camera.onViewMatrixChangedObservable.add(paint);
      view.observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(paint); view.observer?.observe(node);
      views3d[kind] = view;
      paint(); scene.whenReadyAsync().then(paint).catch(() => {});
      return true;
    } catch (err) {
      console.warn("LBM 3D view unavailable:", err);
      try { scene?.dispose(); engine?.dispose(); } catch {}
      node.classList.add("empty-chart"); node.textContent = "The 3D view could not start (WebGL unavailable).";
      return false;
    }
  }

  // ------------------------------------------------------------------------------------------------ results: frames, video
  const imageCache = new Map(), rectCache = new Map();
  const viewer = {key: "", img: null, rect: null, index: -1, playing: false, timer: null};
  function lbmContext(point) {
    if (point?.backend === "lbm" && point.lbm) return {kind: "final", point, lbm: point.lbm};
    const running = ["queued", "running"].includes(state.job?.status), res = state.result || {};
    if (!running && !(point == null && (res.lbm_lattices || res.lbm_live))) return null;
    const keys = Object.keys(res.lbm_lattices || res.lbm_live || {});
    if (!keys.length || point?.backend && point.backend !== "lbm") return null;
    const key = keys[keys.length - 1];
    return {kind: "live", point: null, lbm: res.lbm_lattices?.[key] || null, live: res.lbm_live?.[key] || null, key};
  }
  function fieldChoices(point) {
    const ctx = lbmContext(point);
    if (!ctx) { stopPlayer(); stopNumeric(); hideMedia(); return false; }
    const choices = [];
    if (ctx.kind === "final") {
      const nm = ctx.lbm.numeric;
      for (const f of nm?.fields || []) if (nm.files?.[f.key]?.length) choices.push([`lbm:num_${f.key}`, `Mid-plane ${f.label} (${nm.files[f.key].length} frames)`]);
      for (const f of FIELD_ORDER) if (ctx.lbm.frames?.[f]?.files?.length) choices.push([`lbm:${f}`, `Mid-plane ${FIELD_LABELS[f]} (${ctx.lbm.frames[f].files.length} solver pictures)`]);
      if (ctx.lbm.startup_slices?.level) choices.push(["lbm:startup", "Solver's step-0 mid-span slice (obstacle cells)"]);
    } else if (ctx.live) {
      if (ctx.live.numeric) {
        for (const f of ctx.live.numeric.fields || []) if (ctx.live.numeric.files?.[f.key]) choices.push([`lbm:livenum_${f.key}`, `Live mid-plane ${f.label} · step ${ctx.live.iteration}`]);
      } else {
        choices.push(["lbm:live_velocity", `Live mid-plane velocity · step ${ctx.live.iteration}`], ["lbm:live_vorticity", `Live mid-plane vorticity · step ${ctx.live.iteration}`]);
      }
    }
    if (ctx.lbm?.domain) choices.push(["lbm:lattice", ctx.lbm.mask_check?.compared ? "LBM mid-span lattice (the solver's own mask compared)" : "LBM mid-span lattice"], ["lbm:geometry3d", "LBM 3D geometry and domain"]);
    if (!choices.length) return false;
    const prior = el["field-kind"].value, keys = choices.map(c => c[0]);
    el["field-kind"].replaceChildren(...choices.map(([v, t]) => { const o = document.createElement("option"); o.value = v; o.textContent = t; return o; }));
    el["field-kind"].value = keys.includes(prior) ? prior : keys[0];
    el["field-kind"].disabled = false;
    renderField(point);
    return true;
  }
  function hideMedia() { const m = $("lbm-media"); if (m) m.remove(); }
  function stopPlayer() { viewer.playing = false; clearTimeout(viewer.timer); viewer.timer = null; }
  function mediaNode() {
    let m = $("lbm-media");
    if (!m) { m = document.createElement("div"); m.id = "lbm-media"; m.className = "lbm-media"; el["field-legend"].parentNode?.insertBefore?.(m, el["field-legend"].nextSibling); if (!m.parentNode) el["field-legend"].parent?.append?.(m); }
    return m;
  }
  function maskText(lbm) {
    const mc = lbm?.mask_check;
    if (!mc) return "";
    if (!mc.compared) return ` · masks not compared: ${mc.reason}`;
    return mc.matches ? ` · the solver's own mid-span obstacle mask equals the app's cell by cell (${mc.solver_solid} solid cells)` :
      ` · the solver's own mid-span obstacle mask differs in ${mc.differing_cells} cells (${mc.solver_only_cells} solver only, ${mc.app_only_cells} app only)`;
  }
  function renderField(point) {
    const choice = String(el["field-kind"].value || "");
    if (!choice.startsWith("lbm:")) { dispose3D("field"); stopPlayer(); stopNumeric(); hideMedia(); return false; }
    const ctx = lbmContext(point);
    if (!ctx) { dispose3D("field"); hideMedia(); return false; }
    const what = choice.slice(4);
    window.AirfoilBabylon?.hide();
    el["babylon-view"].classList.add("hidden"); el["field-view"].classList.remove("hidden");
    el["field-legend"].classList.remove("hidden");
    if (what !== "geometry3d") dispose3D("field");
    if (!FIELD_ORDER.includes(what)) stopPlayer();
    if (!what.startsWith("num_")) stopNumeric();
    if (what.startsWith("num_") && ctx.kind === "final" && ctx.lbm?.numeric) { renderNumeric(ctx, what.slice(4)); return true; }
    if (what.startsWith("livenum_")) { hideMedia(); renderLiveNumeric(ctx, what.slice(8)); return true; }
    if (what === "geometry3d") { hideMedia(); show3D(el["field-view"], ctx.lbm, "field"); el["field-legend"].textContent = statusText(ctx.lbm); return true; }
    if (what === "lattice") { hideMedia(); latticeView("field", el["field-view"], ctx.lbm); el["field-legend"].textContent = statusText(ctx.lbm) + maskText(ctx.lbm); return true; }
    if (what === "startup") { hideMedia(); showImage(ctx.lbm, safeHref(ctx.lbm.startup_slices.level), "The solver's step-0 mid-span slice: solid cells are black (its voxelization of the STL), the colour is the single lattice level", true); return true; }
    if (what.startsWith("live_")) {
      hideMedia();
      const f = what.slice(5), url = safeHref(ctx.live?.[f]);
      if (!url) { el["field-view"].textContent = "Waiting for the first mid-plane frame."; return true; }
      showImage(ctx.lbm, url, `Live mid-plane ${FIELD_LABELS[f]} · step ${ctx.live.iteration} · the solver's own slice image (colour scale per frame)`, true);
      return true;
    }
    renderFrames(ctx, what);
    return true;
  }
  function loadImage(url) {
    if (imageCache.has(url)) return imageCache.get(url);
    const p = new Promise((resolve, reject) => { const img = new Image(); img.onload = () => resolve(img); img.onerror = () => reject(new Error("image failed")); img.src = url; });
    imageCache.set(url, p);
    if (imageCache.size > 120) imageCache.delete(imageCache.keys().next().value);
    return p;
  }
  /* The plot area of a solver slice PNG (a Plots heatmap of the cell centres 1..n with a colour bar on the right): the
     longest run of saturated pixels along rows and columns away from the body. Checked against the lattice aspect ratio;
     null when it cannot be found (then the image is shown without the section overlay). */
  function plotRect(img, nx, nz) {
    const key = `${img.src}|${nx}|${nz}`;
    if (rectCache.has(key)) return rectCache.get(key);
    let rect = null;
    try {
      const w = img.naturalWidth, h = img.naturalHeight, c = document.createElement("canvas"); c.width = w; c.height = h;
      const g = c.getContext("2d"); g.drawImage(img, 0, 0); const data = g.getImageData(0, 0, w, h).data;
      const sat = (x, y) => { const i = 4 * (y * w + x), r = data[i], gg = data[i + 1], b = data[i + 2]; return Math.max(r, gg, b) - Math.min(r, gg, b) > 40; };
      const run = (len, at) => { let best = [0, -1], s = -1; for (let t = 0; t <= len; t++) { const on = t < len && at(t); if (on && s < 0) s = t; if (!on && s >= 0) { if (t - 1 - s > best[1] - best[0]) best = [s, t - 1]; s = -1; } } return best; };
      const rows = [0.22, 0.28, 0.72, 0.78].map(f => run(w, x => sat(x, Math.round(f * h))));
      const med = a => a.slice().sort((p, q) => p - q)[Math.floor(a.length / 2)];
      const left = med(rows.map(r => r[0])), right = med(rows.map(r => r[1]));
      const cols = [0.12, 0.2, 0.8, 0.88].map(f => run(h, y => sat(Math.round(left + f * (right - left)), y)));
      const top = med(cols.map(r => r[0])), bottom = med(cols.map(r => r[1]));
      const aspect = (right - left) / Math.max(1, bottom - top), want = (nx - 1) / Math.max(1, nz - 1);
      if (right - left > 50 && bottom - top > 30 && Math.abs(aspect / want - 1) < 0.03) rect = {left, right, top, bottom};
    } catch { rect = null; }
    rectCache.set(key, rect);
    return rect;
  }
  function showImage(lbm, url, caption, overlay) {
    loadImage(url).then(img => {
      if (!String(el["field-kind"].value).startsWith("lbm:")) return;
      drawImage(lbm, img, overlay);
      if (caption != null) el["field-legend"].textContent = caption + (viewer.rect ? " · the app's section outline (orange) is drawn over the solver's solid cells" : overlay ? " · plot area not detected: shown without the section outline" : "");
    }).catch(() => { el["field-view"].textContent = "The slice image could not be loaded."; });
  }
  function drawImage(lbm, img, overlay) {
    const d = lbm?.domain;
    const rect = overlay && d ? plotRect(img, Number(d.nx), Number(d.nz)) : null;
    const key = `${rect ? "world" : "pixel"}|${d ? `${d.nx}x${d.nz}|${d.offset}` : ""}|${lbm?.alpha_deg}|${img.naturalWidth}x${img.naturalHeight}`;
    viewer.img = img; viewer.rect = rect;
    if (viewer.key === key && spatialViews.field && spatialViews.field.lbmViewer) { spatialViews.field.schedule(); return; }
    viewer.key = key;
    const W = img.naturalWidth || img.width, H = img.naturalHeight || img.height;
    if (!rect) {
      const plot = new SpatialPlot("field", el["field-view"], {xmin: 0, xmax: W, ymin: 0, ymax: H}, (ctx, view) => {
        ctx.drawImage(viewer.img, view.x(0), view.y(H), W * view.scale, H * view.scale);
      }, null, {hideAxes: true});
      plot.lbmViewer = true; plot.image = true;
      return;
    }
    const dx = Number(d.dx), ox = Number(d.offset[0]), oz = Number(d.offset[2]);
    const sx = (Number(d.nx) - 1) * dx / (rect.right - rect.left), sz = (Number(d.nz) - 1) * dx / (rect.bottom - rect.top);
    const x0 = 0.5 * dx - rect.left * sx - ox, ztop = (Number(d.nz) - 0.5) * dx + rect.top * sz - oz;   // world of the image's corners
    const sec = lbm.placed, xs = sec.x.map(Number), zs = sec.z.map(Number);
    const bounds = {xmin: -ox, xmax: Number(d.Lx) - ox, ymin: -oz, ymax: Number(d.Lz) - oz};
    const plot = new SpatialPlot("field", el["field-view"], bounds, (ctx, view) => {
      const i = viewer.img, r = viewer.rect || rect;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(i, view.x(x0), view.y(ztop), W * sx * view.scale, H * sz * view.scale);
      ctx.beginPath(); xs.forEach((x, k) => { const px = view.x(x - ox), py = view.y(zs[k] - oz); k ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.closePath(); ctx.strokeStyle = "#ff8a3d"; ctx.lineWidth = 1.4; ctx.stroke();
    }, (x, y) => `x ${numberText(x, 3)} c · z ${numberText(y, 3)} c (body frame)`, {axisUnits: "chord"});
    plot.lbmViewer = true;
  }
  function renderFrames(ctx, field) {
    const lbm = ctx.lbm, fr = lbm.frames[field], files = fr.files.map(safeHref).filter(Boolean), n = files.length;
    if (!n) { el["field-view"].textContent = "No frames for this field."; return; }
    if (viewer.field !== field || viewer.point !== ctx.point || viewer.index >= n || viewer.index < 0) { viewer.index = n - 1; viewer.field = field; viewer.point = ctx.point; stopPlayer(); }
    const legend = el["field-legend"]; legend.replaceChildren();
    const play = document.createElement("button"); play.type = "button"; play.className = "secondary-button"; play.textContent = viewer.playing ? "Pause" : "Play";
    const slider = document.createElement("input"); slider.type = "range"; slider.min = "0"; slider.max = String(n - 1); slider.value = String(viewer.index); slider.className = "lbm-frame-slider"; slider.setAttribute("aria-label", "Mid-plane frame");
    const label = document.createElement("span"); label.className = "legend-range";
    const links = document.createElement("span"); links.className = "field-links";
    const link = (text, url) => { if (!safeHref(url)) return; const a = document.createElement("a"); a.href = safeHref(url); a.textContent = text; a.target = "_blank"; a.rel = "noopener noreferrer"; a.download = ""; links.append(a, document.createTextNode(" ")); };
    for (const f of FIELD_ORDER) if (lbm.videos?.[f]?.file) link(`${FIELD_LABELS[f]} MP4`, lbm.videos[f].file);
    link("STL", lbm.stl?.file); link("config.yaml", lbm.config);
    legend.append(play, slider, label, links);
    const show = k => {
      viewer.index = k; slider.value = String(k);
      const steps = fr.steps || [], ctu = fr.ctu || [];
      label.textContent = `${FIELD_LABELS[field]} · frame ${k + 1}/${n} · step ${steps[k] ?? "?"}${finite(Number(ctu[k])) ? ` · t U/c = ${Number(ctu[k]).toFixed(2)}` : ""} · the solver's slice image (its colour scale per frame)`;
      loadImage(files[k]).then(img => { if (viewer.field === field && viewer.index === k && String(el["field-kind"].value) === `lbm:${field}`) drawImage(lbm, img, true); }).catch(() => {});
      if (k + 1 < n) loadImage(files[k + 1]).catch(() => {});
    };
    const tick = () => { if (!viewer.playing) return; show((viewer.index + 1) % n); viewer.timer = setTimeout(tick, 160); };
    play.addEventListener("click", () => { viewer.playing = !viewer.playing; play.textContent = viewer.playing ? "Pause" : "Play"; clearTimeout(viewer.timer); if (viewer.playing) tick(); });
    slider.addEventListener("input", () => { stopPlayer(); play.textContent = "Play"; show(Number(slider.value)); });
    show(viewer.index);
    // the video of this field (or of the velocity), with its download link
    const m = mediaNode(); m.replaceChildren();
    const vid = lbm.videos?.[field] || lbm.videos?.velocity;
    if (vid?.file && safeHref(vid.file)) {
      const video = document.createElement("video"); video.controls = true; video.preload = "metadata"; video.src = safeHref(vid.file); video.className = "lbm-video";
      video.setAttribute("aria-label", "Mid-plane video");
      const cap = document.createElement("p"); cap.className = "hint";
      cap.textContent = `Video: ${FIELD_LABELS[lbm.videos?.[field] ? field : "velocity"]} at the mid-span plane, ${vid.frames} frames at ${vid.fps} frames/s (${vid.codec}).`;
      m.append(video, cap);
    }
  }
  // ------------------------------------------------------------------------------------------------ results: numeric mid-plane
  /* "lbmio" profile: the solver's own numeric mid-span plane, converted by the worker into Float16 files (n1 x n2 values,
     i along x fastest, NaN outside the fluid) plus a uint8 mask (0 fluid, 1 solid, 2 no level). Drawn cell by cell on a
     FIXED colour range per field (editable; reset to the field's default), with the section outline and a value probe. */
  const DIV_STOPS = [[40, 70, 140], [110, 155, 210], [245, 245, 245], [225, 130, 100], [150, 35, 45]];   // = lbm_render.jl
  const SOLID_RGB = [128, 134, 140], NOLEVEL_RGB = [215, 220, 224];
  const numCache = new Map();
  let numCacheBytes = 0;
  const numRange = {};
  const num = {key: "", field: "", index: -1, playing: false, timer: null, canvas: null, values: null, grid: null, showLevels: true, levels: null};
  // the level that supplied each node (num/level.u8, next to the mask); null when absent (older results)
  const fetchLevels = (maskUrl, n) => fetchBinary(String(maskUrl).replace(/mask\.u8$/, "level.u8"), "mask", n).catch(() => null);
  const multiLevel = levels => { if (!levels) return false; let a = -1; for (let q = 0; q < levels.length; q++) { const v = levels[q]; if (!v) continue; if (a < 0) a = v; else if (v !== a) return true; } return false; };
  const HALF = (() => {   // Float16 -> Float32 for every bit pattern
    const t = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      t[h] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
    }
    return t;
  })();
  function stopsColor(stops, t) {
    t = Math.max(0, Math.min(1, t));
    const a = Math.min(stops.length - 2, Math.floor(t * (stops.length - 1))), k = Math.max(0, Math.min(1, t * (stops.length - 1) - a));
    return stops[a].map((v, i) => Math.round(v * (1 - k) + stops[a + 1][i] * k));
  }
  const SEQ_STOPS = [[23, 62, 114], [52, 124, 187], [91, 192, 201], [237, 212, 107], [233, 139, 77], [155, 56, 77]];   // = app.js ramp()
  function fieldColor(v, meta, lo, hi) {
    if (!Number.isFinite(v)) return NOLEVEL_RGB;
    const mid = meta.mid;
    if (meta.cmap === "div" && Number.isFinite(mid) && lo < mid && mid < hi)
      return stopsColor(DIV_STOPS, v < mid ? 0.5 * (v - lo) / (mid - lo) : 0.5 + 0.5 * (v - mid) / (hi - mid));
    return stopsColor(meta.cmap === "div" ? DIV_STOPS : SEQ_STOPS, (v - lo) / (hi - lo));
  }
  function gradientCss(meta, lo, hi) {
    const stops = [];
    for (let s = 0; s <= 12; s++) { const v = lo + (hi - lo) * s / 12, c = fieldColor(v, meta, lo, hi); stops.push(`rgb(${c.join(",")}) ${(100 * s / 12).toFixed(1)}%`); }
    return `linear-gradient(90deg,${stops.join(",")})`;
  }
  function fetchBinary(url, kind, bytesHint) {
    if (numCache.has(url)) { const v = numCache.get(url); numCache.delete(url); numCache.set(url, v); return v.promise; }
    const promise = (typeof fetch === "function" ? fetch(url) : Promise.reject(new Error("fetch unavailable"))).then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.arrayBuffer(); }).then(buf => {
      if (kind === "mask") return new Uint8Array(buf);
      const u = new Uint16Array(buf), out = new Float32Array(u.length);
      for (let i = 0; i < u.length; i++) out[i] = HALF[u[i]];
      return out;
    });
    const bytes = kind === "mask" ? bytesHint : 4 * bytesHint;
    numCache.set(url, {promise, bytes}); numCacheBytes += bytes;
    promise.catch(() => { if (numCache.get(url)?.promise === promise) { numCache.delete(url); numCacheBytes -= bytes; } });
    while (numCacheBytes > 160e6 && numCache.size > 2) { const [k, v] = numCache.entries().next().value; numCache.delete(k); numCacheBytes -= v.bytes; }
    return promise;
  }
  function numericGrid(ctx, live) {
    const g = live ? live : ctx.lbm?.numeric?.grid;
    if (!g) return null;
    const off = ctx.lbm?.domain?.offset || g.mesh_offset_m || [0, 0, 0];
    return {n1: Number(g.n1), n2: Number(g.n2), x0: Number(g.origin_m[0]), z0: Number(g.origin_m[1]), h: Number(g.spacing_m),
            ox: Number(off[0]), oz: Number(off[2]), y: Number(g.normal_coord_m)};
  }
  function paintCanvas(values, mask, g, meta, lo, hi, levels = null) {
    const c = document.createElement("canvas"); c.width = g.n1; c.height = g.n2;
    const cx = c.getContext("2d"), img = cx.createImageData(g.n1, g.n2), px = img.data;
    const at = (i, k) => 4 * ((g.n2 - 1 - k) * g.n1 + i);   // canvas rows top-down, z up
    for (let k = 0; k < g.n2; k++) for (let i = 0; i < g.n1; i++) {
      const q = k * g.n1 + i, m = mask[q], rgb = m === 1 ? SOLID_RGB : m === 2 ? NOLEVEL_RGB : fieldColor(values[q], meta, lo, hi);
      const p = at(i, k);
      px[p] = rgb[0]; px[p + 1] = rgb[1]; px[p + 2] = rgb[2]; px[p + 3] = 255;
    }
    // level boundaries (several levels): the nodes on both sides of a change of the supplying level, darkened
    if (levels && levels.length === g.n1 * g.n2 && multiLevel(levels)) {
      const dark = p => { px[p] = Math.round(px[p] * 0.35 + 12); px[p + 1] = Math.round(px[p + 1] * 0.35 + 12); px[p + 2] = Math.round(px[p + 2] * 0.35 + 12); };
      const edge = new Uint8Array(g.n1 * g.n2);
      for (let k = 0; k < g.n2; k++) for (let i = 0; i < g.n1; i++) {
        const q = k * g.n1 + i, v = levels[q];
        if (i + 1 < g.n1 && levels[q + 1] !== v) { edge[q] = 1; edge[q + 1] = 1; }
        if (k + 1 < g.n2 && levels[q + g.n1] !== v) { edge[q] = 1; edge[q + g.n1] = 1; }
      }
      for (let k = 0; k < g.n2; k++) for (let i = 0; i < g.n1; i++) if (edge[k * g.n1 + i] && mask[k * g.n1 + i] !== 1) dark(at(i, k));
    }
    cx.putImageData(img, 0, 0);
    return c;
  }
  function drawNumeric(ctx, g, meta, values, mask, lo, hi, levels = null) {
    num.canvas = paintCanvas(values, mask, g, meta, lo, hi, num.showLevels ? levels : null); num.values = values; num.grid = g; num.meta = meta; num.mask = mask; num.levels = levels;
    const key = `num|${g.n1}x${g.n2}|${g.x0},${g.z0},${g.h}|${g.ox},${g.oz}|${ctx.lbm?.alpha_deg}`;
    if (num.key === key && spatialViews.field?.lbmNumeric) { spatialViews.field.schedule(); return; }
    num.key = key; viewer.key = "";
    const sec = ctx.lbm?.placed, xs = sec ? sec.x.map(Number) : [], zs = sec ? sec.z.map(Number) : [];
    const X0 = g.x0 - g.ox, Z0 = g.z0 - g.oz, W = g.n1 * g.h, H = g.n2 * g.h;
    const body = xs.length ? {xmin: Math.min(...xs) - g.ox - 0.6, xmax: Math.max(...xs) - g.ox + 1.6, ymin: Math.min(...zs) - g.oz - 0.75, ymax: Math.max(...zs) - g.oz + 0.75} :
      {xmin: X0, xmax: X0 + W, ymin: Z0, ymax: Z0 + H};
    const plot = new SpatialPlot("field", el["field-view"], body, (c2, view) => {
      if (!num.canvas) return;
      c2.imageSmoothingEnabled = false;
      c2.drawImage(num.canvas, view.x(X0), view.y(Z0 + H), W * view.scale, H * view.scale);
      if (xs.length) {
        c2.beginPath(); xs.forEach((x, k) => { const px = view.x(x - g.ox), py = view.y(zs[k] - g.oz); k ? c2.lineTo(px, py) : c2.moveTo(px, py); });
        c2.closePath(); c2.strokeStyle = "#ff8a3d"; c2.lineWidth = 1.2; c2.stroke();
      }
    }, (x, y) => {
      const gg = num.grid; if (!gg) return "";
      const i = Math.floor((x + gg.ox - gg.x0) / gg.h), k = Math.floor((y + gg.oz - gg.z0) / gg.h);
      if (i < 0 || k < 0 || i >= gg.n1 || k >= gg.n2) return `x ${numberText(x, 3)} c · z ${numberText(y, 3)} c`;
      const q = k * gg.n1 + i, m = num.mask?.[q], lv = num.levels?.[q];
      const v = (m === 1 ? "solid" : m === 2 ? "no level" : `${num.meta.label} ${numberText(num.values[q], 4)}`) + (lv ? ` · level ${lv}` : "");
      return `x ${numberText(x, 3)} c · z ${numberText(y, 3)} c (body frame) · cell ${i + 1}, ${k + 1} · ${v}`;
    }, {axisUnits: "chord"});
    plot.lbmNumeric = true;
  }
  function rangeOf(meta) {
    const r = numRange[meta.key];
    return r ? r : [Number(meta.range[0]), Number(meta.range[1])];
  }
  function rangeControls(meta, redraw) {
    const [lo, hi] = rangeOf(meta);
    const wrap = document.createElement("span"); wrap.className = "lbm-range";
    const bar = document.createElement("span"); bar.className = "legend-ramp"; bar.style.background = gradientCss(meta, lo, hi);
    const inp = (v, label) => { const i = document.createElement("input"); i.type = "number"; i.step = "any"; i.value = String(v); i.className = "lbm-range-input"; i.setAttribute("aria-label", label); return i; };
    const a = inp(lo, `${meta.label} colour range minimum`), b = inp(hi, `${meta.label} colour range maximum`);
    const reset = document.createElement("button"); reset.type = "button"; reset.className = "secondary-button lbm-range-reset"; reset.textContent = "Default range";
    const apply = () => { const x = Number(a.value), y = Number(b.value); if (Number.isFinite(x) && Number.isFinite(y) && x < y) { numRange[meta.key] = [x, y]; bar.style.background = gradientCss(meta, x, y); redraw(); } };
    a.addEventListener("change", apply); b.addEventListener("change", apply);
    reset.addEventListener("click", () => { delete numRange[meta.key]; a.value = String(meta.range[0]); b.value = String(meta.range[1]); bar.style.background = gradientCss(meta, Number(meta.range[0]), Number(meta.range[1])); redraw(); });
    wrap.append(a, bar, b, reset);
    return wrap;
  }
  function numericMeta(fields, key) { return (fields || []).find(f => f.key === key) || null; }
  function renderNumeric(ctx, key) {
    const n = ctx.lbm.numeric, meta = numericMeta(n.fields, key), files = (n.files?.[key] || []).map(safeHref), g = numericGrid(ctx);
    const maskUrl = safeHref(n.mask), count = files.length;
    if (!meta || !count || !g || !maskUrl || files.some(f => !f)) { el["field-view"].textContent = "No numeric frames for this field."; return; }
    if (num.field !== key || num.point !== ctx.point || num.index >= count || num.index < 0) { num.index = count - 1; num.field = key; num.point = ctx.point; stopNumeric(); }
    const legend = el["field-legend"]; legend.replaceChildren();
    const play = document.createElement("button"); play.type = "button"; play.className = "secondary-button"; play.textContent = num.playing ? "Pause" : "Play";
    const slider = document.createElement("input"); slider.type = "range"; slider.min = "0"; slider.max = String(count - 1); slider.value = String(num.index); slider.className = "lbm-frame-slider"; slider.setAttribute("aria-label", "Mid-plane frame");
    const label = document.createElement("span"); label.className = "legend-range";
    const links = document.createElement("span"); links.className = "field-links";
    const link = (text, url) => { if (!safeHref(url)) return; const a = document.createElement("a"); a.href = safeHref(url); a.textContent = text; a.target = "_blank"; a.rel = "noopener noreferrer"; a.download = ""; links.append(a, document.createTextNode(" ")); };
    for (const [k, v] of Object.entries(ctx.lbm.videos || {})) if (v?.file && v.source) link(`${v.label || k} MP4`, v.file);
    link("STL", ctx.lbm.stl?.file); link("config.yaml", ctx.lbm.config);
    const show = k => {
      num.index = k; slider.value = String(k);
      const steps = n.steps || [], ctu = n.ctu || [], lo0 = n.min?.[key]?.[k], hi0 = n.max?.[key]?.[k];
      label.textContent = `${meta.label} · frame ${k + 1}/${count} · step ${steps[k] ?? "?"}${finite(Number(ctu[k])) ? ` · t U/c = ${Number(ctu[k]).toFixed(2)}` : ""}${finite(Number(lo0)) ? ` · this frame ${numberText(Number(lo0), 3)} to ${numberText(Number(hi0), 3)}` : ""}`;
      Promise.all([fetchBinary(files[k], "field", g.n1 * g.n2), fetchBinary(maskUrl, "mask", g.n1 * g.n2), fetchLevels(maskUrl, g.n1 * g.n2)]).then(([values, mask, levels]) => {
        if (num.field !== key || num.index !== k || String(el["field-kind"].value) !== `lbm:num_${key}`) return;
        const [lo, hi] = rangeOf(meta);
        drawNumeric(ctx, g, meta, values, mask, lo, hi, levels);
      }).catch(() => { el["field-view"].textContent = "The numeric frame could not be loaded."; });
      if (k + 1 < count) fetchBinary(files[k + 1], "field", g.n1 * g.n2).catch(() => {});
    };
    const redraw = () => show(num.index);
    const tick = () => { if (!num.playing) return; show((num.index + 1) % count); num.timer = setTimeout(tick, 160); };
    play.addEventListener("click", () => { num.playing = !num.playing; play.textContent = num.playing ? "Pause" : "Play"; clearTimeout(num.timer); if (num.playing) tick(); });
    slider.addEventListener("input", () => { stopNumeric(); play.textContent = "Play"; show(Number(slider.value)); });
    const used = n.grid?.levels_used || [];
    const units = document.createElement("span"); units.className = "hint lbm-units"; units.textContent = `${meta.units}; the solver's cell values on the mid-span plane (y = ${numberText(g.y, 4)} m), fixed colour scale, grey: solid cells${used.length > 1 ? `; levels ${used.join(", ")} (each node the finest level there), dark lines: level boundaries` : ""}`;
    legend.append(play, slider, label, rangeControls(meta, redraw));
    if (used.length > 1) {
      const lab = document.createElement("label"); lab.className = "check-line lbm-levels-toggle";
      const box = document.createElement("input"); box.type = "checkbox"; box.checked = num.showLevels; box.setAttribute("aria-label", "Show the level boundaries");
      box.addEventListener("change", () => { num.showLevels = box.checked; redraw(); });
      const t = document.createElement("span"); t.textContent = "Level boundaries"; lab.append(box, t); legend.append(lab);
    }
    legend.append(links, units);
    show(num.index);
    const m = mediaNode(); m.replaceChildren();
    const vid = ctx.lbm.videos?.[key] || ctx.lbm.videos?.umag;
    if (vid?.file && safeHref(vid.file)) {
      const video = document.createElement("video"); video.controls = true; video.preload = "metadata"; video.src = safeHref(vid.file); video.className = "lbm-video";
      video.setAttribute("aria-label", "Mid-plane video");
      const cap = document.createElement("p"); cap.className = "hint";
      cap.textContent = `Video: ${vid.label || key} at the mid-span plane, ${vid.frames} frames at ${vid.fps} frames/s (${vid.codec}), colour range ${Array.isArray(vid.colour_range) ? vid.colour_range.join(" to ") : "fixed"}.`;
      m.append(video, cap);
    }
  }
  function renderLiveNumeric(ctx, key) {
    const live = ctx.live?.numeric, meta = numericMeta(live?.fields, key), url = safeHref(live?.files?.[key]), maskUrl = safeHref(live?.mask), g = numericGrid(ctx, live);
    if (!meta || !url || !maskUrl || !g) { el["field-view"].textContent = "Waiting for the first mid-plane frame."; return; }
    const legend = el["field-legend"]; legend.replaceChildren();
    const label = document.createElement("span"); label.className = "legend-range";
    label.textContent = `Live ${meta.label} · step ${live.step}${finite(Number(live.ctu)) ? ` · t U/c = ${Number(live.ctu).toFixed(2)}` : ""} · the solver's numeric mid-plane`;
    const draw = () => Promise.all([fetchBinary(url, "field", g.n1 * g.n2), fetchBinary(maskUrl, "mask", g.n1 * g.n2), fetchLevels(maskUrl, g.n1 * g.n2)]).then(([values, mask, levels]) => {
      if (String(el["field-kind"].value) !== `lbm:livenum_${key}`) return;
      const [lo, hi] = rangeOf(meta);
      drawNumeric(ctx, g, meta, values, mask, lo, hi, levels);
    }).catch(() => { el["field-view"].textContent = "The live frame could not be loaded."; });
    legend.append(label, rangeControls(meta, draw));
    draw();
  }
  function stopNumeric() { num.playing = false; clearTimeout(num.timer); num.timer = null; }

  function onResult() {
    const res = state.result;
    if (!res?.lbm_live && !res?.lbm_lattices) return;
    const k = JSON.stringify([Object.keys(res.lbm_lattices || {}), Object.values(res.lbm_live || {}).map(v => v?.iteration)]);
    if (k === onResult.last) return;
    onResult.last = k;
    if (!(res.results || []).some(r => r.backend === "lbm")) renderFieldChoices(currentResult());
  }
  function statusLabel(r) {
    if (r.status === "domain_check") {
      const mc = r.lbm?.mask_check, lat = r.lbm?.lattice_matches_solver;
      if (!lat) return "Domain check: lattice MISMATCH";
      if (mc?.compared) return mc.matches ? "Domain check: the solver's lattice and mid-span mask match" : `Domain check: lattice matches, mask differs in ${mc.differing_cells} cells`;
      return "Domain check: the solver's lattice matches";
    }
    if (r.status === "completed") return `LBM run complete · loads ${r.lbm?.settled ? "settled" : "not settled (unsteady or still changing)"} · unvalidated`;
    if (r.status === "not_converged") return "Stopped by the wall cap · diagnostic only";
    return String(r.status || "Computed");
  }

  // ------------------------------------------------------------------------------------------------ wiring
  for (const [id] of SPEC) $(id)?.addEventListener("input", () => { updateSetup(); invalidatePreviewMesh(); });
  ["lbm-plate-cells", "lbm-cells"].forEach(id => $(id)?.addEventListener("change", () => scheduleGeometryPreview()));
  $("lbm-wall-model")?.addEventListener("change", () => { updateSetup(); invalidatePreviewMesh(); });
  $("lbm-levels")?.addEventListener("change", () => { updateSetup(); invalidatePreviewMesh(); });
  $("lbm-z-walls")?.addEventListener("change", () => { updateSetup(); invalidatePreviewMesh(); });
  $("lbm-level-visc")?.addEventListener("change", () => { updateSetup(); invalidatePreviewMesh(); });
  $("lbm-spoiler")?.addEventListener("change", () => { updateSetup(); scheduleGeometryPreview(); });
  for (const [id] of SPOILER) $(id)?.addEventListener("input", () => { updateSetup(); scheduleGeometryPreview(); });
  el.backend.addEventListener("change", () => {
    // the LBM's default case is a moderate Reynolds number; an edited value is kept
    if (el.backend.value === "lbm" && !state.reynoldsEdited && ["300000", "3000000"].includes(el.reynolds.value)) el.reynolds.value = "20000";
    updateSetup();
  });
  const frame3D = (kind, which) => views3d[kind]?.flip?.(which);
  window.AirfoilLBM = {frame3D, flowKeys, estimate, updateSetup: updateLbmSetup, applySaved, statusText, displayAfterMesh, renderGeometry, drawSection,
    fieldChoices, renderField, onResult, statusLabel, plotRect, latticeView, show3D, dispose3D, earclip,
    fieldColor, gradientCss, HALF, paintCanvas, numRange, onHealth, levelTaus};
  updateSetup();
})();
