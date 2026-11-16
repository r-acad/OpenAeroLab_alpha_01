/* The request Aero Lab sends for a mesh and for a run, built from control values alone.
   No DOM or scene access: the page passes a reader of its controls, tests pass a table of values.
   Each solver's request carries only the settings that solver has. */
(function (root) {
  'use strict';

  /* The solver choices of the page. `solver` and `lattice` are the two request keys a choice stands for. */
  const SOLVERS = {
    main_d3q27: {solver: 'main', lattice: 'd3q27', gpu: true, maxMach: 0.6, maxLevels: 7},
    main_d3q39: {solver: 'main', lattice: 'd3q39', gpu: true, maxMach: 1.6, maxLevels: 7},
    pond_thermal: {solver: 'pond_thermal', lattice: 'd3q27', gpu: true, thermal: true, minMach: 0.05, maxMach: 1.5, maxLevels: 7},
    q19_octree: {solver: 'q19_octree', lattice: 'd3q19', octree: true, maxMach: 0.3, maxLevels: 10}
  };
  const GAMMA = 1.4, GAS_CONSTANT = 287.05;   // ideal air, as the thermal solver assumes

  /* The choice that a saved case's solver and lattice stand for; null when Aero Lab no longer offers it. */
  function choiceOf(solver, lattice) {
    if (solver === 'main') return lattice === 'd3q39' ? 'main_d3q39' : 'main_d3q27';
    return SOLVERS[solver] ? solver : null;
  }

  function number(io, id, label, {min = -Infinity, max = Infinity, integer = false} = {}) {
    const raw = String(io.value(id) ?? '').trim(), n = Number(raw);
    if (!raw || !Number.isFinite(n) || n < min || n > max || (integer && !Number.isInteger(n)))
      throw new Error(`Enter a valid ${label}${Number.isFinite(min) ? ` (at least ${min})` : ''}.`);
    return n;
  }

  /* Thermal free stream: Mach number and temperature give the speed and the sound speed. */
  function thermalFlow(io) {
    const temperature_k = number(io, 'flow-temperature', 'freestream temperature', {min: 100, max: 2000});
    const mach = number(io, 'flow-mach', 'freestream Mach', {min: 0.05, max: 1.5});
    const speed_of_sound = Math.sqrt(GAMMA * GAS_CONSTANT * temperature_k);
    return {mach, temperature_k, speed_of_sound, velocity: mach * speed_of_sound,
      cfl: number(io, 'flow-cfl', 'CFL number', {min: 0.01, max: 0.4}), wall_model: io.value('flow-wall-model') === 'noslip' ? 'noslip' : 'slip'};
  }

  function transform(io) {
    const axes = ['x', 'y', 'z'];
    return {scale: axes.map(a => number(io, 'scale-' + a, 'scale ' + a, {min: 0.00001, max: 100000})),
      rotation_deg: axes.map(a => number(io, 'rotate-' + a, 'rotation ' + a, {min: -360, max: 360})),
      translation: axes.map(a => number(io, 'move-' + a, 'offset ' + a, {min: -10000, max: 10000}))};
  }

  function recordedPlanes(cuts) {
    return (cuts || []).map(cut => {
      const position = Number(cut.position);
      if (String(cut.position).trim() === '' || !Number.isFinite(position) || position < 0)
        throw new Error('Recorded plane positions must be nonnegative metres from the domain origin.');
      if (!['x', 'y', 'z'].includes(cut.axis)) throw new Error('A recorded plane needs an X, Y or Z axis.');
      return {axis: cut.axis, position};
    });
  }

  /* The mesh request: everything that steps 1 to 4 set, plus the run settings current at that moment. */
  function buildRequest(io, context = {}) {
    const choice = SOLVERS[io.value('solver')];
    if (!choice) throw new Error('Select a solver.');
    if (!context.modelId) throw new Error('Load a model first.');
    const geometry_handling = io.value('geometry-handling') === 'two_sided_open' ? 'two_sided_open' : 'closed_solid';
    const boundary = io.value('boundary');
    if (!['free_air', 'ground', 'symmetry'].includes(boundary)) throw new Error('Choose a boundary condition.');
    const sizing = io.value('domain-sizing') || 'minimum';
    if (!['minimum', 'fixed', 'proportions'].includes(sizing)) throw new Error('Choose the domain sizing.');
    const placed = transform(io);
    const domain = {
      upstream: number(io, 'upstream', 'upstream clearance', {min: 0.02, max: 10000}),
      downstream: number(io, 'downstream', 'downstream clearance', {min: 0.02, max: 10000}),
      lateral: number(io, 'lateral', 'lateral clearance', {min: 0.02, max: 10000}),
      height: number(io, 'height', 'vertical clearance', {min: 0.02, max: 10000}),
      boundary, sizing, x_offset_m: number(io, 'domain-x-offset', 'domain X offset', {min: -10000, max: 10000})};
    if (sizing === 'proportions')
      domain.proportions = ['x', 'y', 'z'].map(a => number(io, 'domain-ratio-' + a, 'domain proportion ' + a, {min: 0.000001, max: 1000000}));
    if (boundary === 'symmetry') {
      const placement = ['none', 'wall', 'center'].includes(io.value('symmetry-placement')) ? io.value('symmetry-placement') : 'none';
      if (placement !== 'none' && placed.translation[1] !== 0)
        throw new Error('Placing the model at the symmetry plane needs lateral offset Y = 0. Choose Leave the model where it is to place it by hand.');
      domain.symmetry_placement = placement;
      if (placement === 'none') domain.symmetry_offset_y_m = Number(io.value('symmetry-placement-offset-y')) || 0;
    }
    const flow = {
      alpha: number(io, 'flow-alpha', 'angle of attack', {min: -85, max: 85}),
      beta: boundary === 'symmetry' ? 0 : number(io, 'flow-beta', 'sideslip', {min: -85, max: 85}),
      density: number(io, 'density', 'density', {min: 0.01, max: 10000}),
      kinematic_viscosity: number(io, 'viscosity', 'kinematic viscosity', {min: 1e-10, max: 1})};
    if (choice.thermal) {
      const thermal = thermalFlow(io);
      Object.assign(flow, {mach: thermal.mach, temperature_k: thermal.temperature_k, cfl: thermal.cfl, wall_model: thermal.wall_model});
    } else {
      flow.velocity = number(io, 'velocity', 'flow speed', {min: 0.01, max: 1000});
      flow.speed_of_sound = number(io, 'sound-speed', 'sound speed', {min: 1, max: 100000});
      if (flow.velocity / flow.speed_of_sound > choice.maxMach) throw new Error(`The selected solver requires Mach ≤ ${choice.maxMach}.`);
    }
    const request = {
      model_id: context.modelId, solver: choice.solver, lattice: choice.lattice, geometry_handling,
      transform: placed, domain, flow,
      surface_dx: number(io, 'surface-dx', 'surface cell size', {min: 0.00001, max: 100}),
      levels: number(io, 'levels', 'mesh levels', {min: 1, max: choice.maxLevels, integer: true}),
      steps: number(io, 'steps', 'solver steps', {min: 8, max: 1000000, integer: true}),
      display_every: context.displayEvery || 100, outputs: context.outputs || {},
      cuts: recordedPlanes(context.cuts)};
    if (context.reference) request.reference = context.reference;
    if (choice.octree) {
      request.octree = {
        band_fine: number(io, 'octree-band-fine', 'finest band', {min: 3, max: 64, integer: true}),
        band: number(io, 'octree-band', 'coarser bands', {min: 3, max: 64, integer: true}),
        wake_length: number(io, 'octree-wake', 'wake length', {min: 0, max: 20}),
        wake_level_offset: number(io, 'octree-wake-offset', 'wake level offset', {min: 0, max: 4, integer: true}),
        seal_cavities: Boolean(io.checked('octree-seal-cavities')),
        wall_function: Boolean(io.checked('octree-wall-function')),
        threads: number(io, 'octree-threads', 'CPU threads', {min: 0, max: 256, integer: true}),
        u_lattice: number(io, 'octree-u-lattice', 'inlet lattice speed', {min: 0.00001, max: 0.12}),
        hrr_sigma: number(io, 'octree-hrr-sigma', 'hybrid collision number', {min: 0.5, max: 1}),
        checkpoint_keep: number(io, 'octree-checkpoint-keep', 'checkpoints kept', {min: 1, max: 10, integer: true}),
        max_plane_samples: number(io, 'octree-plane-samples', 'samples per saved plane', {min: 10000, max: 16000000, integer: true})};
    } else {
      request.mesh_sizing = io.value('mesh-sizing') === 'vram' ? 'vram' : 'manual';
      request.target_vram_percent = number(io, 'target-vram', 'GPU memory target', {min: 1, max: 100});
      request.surface_triangle_max_edge_m = number(io, 'surface-triangle-max-edge-mm', 'maximum surface triangle edge in millimetres', {min: 0.1, max: 100000}) / 1000;
    }
    return request;
  }

  /* The run request: the settings of step 5, applied to the built mesh without rebuilding it. */
  function runOptions(io, context = {}) {
    const choice = SOLVERS[io.value('solver')];
    if (!choice) throw new Error('Select a solver.');
    const options = {
      steps: number(io, 'steps', 'solver steps', {min: 8, max: 1000000, integer: true}),
      display_every: context.displayEvery || 100, outputs: context.outputs || {},
      cuts: recordedPlanes(context.cuts),
      geometry_handling: io.value('geometry-handling') === 'two_sided_open' ? 'two_sided_open' : 'closed_solid',
      study_name: context.studyName || 'Aero study'};
    if (context.reference) options.reference = context.reference;
    if (context.studySnapshot) options.study_snapshot = context.studySnapshot;
    if (choice.gpu) options.target_vram_percent = number(io, 'target-vram', 'GPU memory target', {min: 1, max: 100});
    return options;
  }

  const api = {SOLVERS, GAMMA, GAS_CONSTANT, choiceOf, thermalFlow, transform, buildRequest, runOptions};
  root.LudwigAeroRequest = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
