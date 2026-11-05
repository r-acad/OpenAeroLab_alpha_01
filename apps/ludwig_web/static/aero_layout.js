/* Menu navigation and dismissible information for the Aero Lab workbench. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const menus = [...document.querySelectorAll('.top-menu')];
  const panels = [...document.querySelectorAll('[data-parameter-panel]')];
  const choices = [...document.querySelectorAll('[data-parameter-target]')];
  const cards = ['viewer-info-panel', 'mesh-notice', 'viewer-message'];
  const openers = {'viewer-info-panel': 'viewer-info', 'mesh-notice': 'mesh-notice-open', 'viewer-message': 'viewer-message-open'};
  let lastMeshText = null;
  let messageState = {text: '', key: null, urgent: false, recovery: false, diagnosticUrl: null};
  let lastAutoMessageKey = null;
  const dismissedMessages = new Set();
  let buildJob = null;
  let buildLog = [];
  let buildState = 'idle';
  let buildTerminalKey = null;
  const dismissedBuilds = new Set();
  let resultsAvailable = false;
  let solverJob = null, solverLog = [], solverMetrics = {}, solverTerminalKey = null;
  const dismissedSolvers = new Set();

  function syncLiveResults() {
    if ($('live-results') && $('run-live-results')) $('run-live-results').checked = $('live-results').checked;
  }

  function syncDomainControls() {
    const sizing = $('domain-sizing')?.value || 'minimum';
    const proportions = sizing === 'proportions';
    const symmetry = $('boundary')?.value === 'symmetry';
    if ($('symmetry-alignment-actions')) $('symmetry-alignment-actions').hidden = !symmetry;
    if ($('domain-proportions-controls')) $('domain-proportions-controls').hidden = !proportions;
    for (const axis of ['x', 'y', 'z']) {
      const input = $(`domain-ratio-${axis}`);
      if (input) input.disabled = !proportions || (symmetry && axis === 'y');
    }
    if ($('domain-ratio-y-label')) $('domain-ratio-y-label').textContent = symmetry ? 'Y · automatic' : 'Y';
    if ($('domain-ratio-note')) $('domain-ratio-note').textContent = symmetry
      ? 'Symmetry uses Y = 0.5 × min(resolved X, resolved Z). The domain grows as needed to preserve minimum clearances and all AMR levels; the Y ratio is automatic.'
      : 'Target X:Y:Z proportions respect minimum clearances and requested AMR levels; native lattice snapping is reported.';
    if ($('domain-sizing-note')) $('domain-sizing-note').textContent = sizing === 'fixed'
      ? 'No automatic AMR growth; native lattice snapping is reported. Boundary conditions are selected separately.'
      : proportions ? 'Target domain proportions while respecting minimum clearances and AMR requirements. Native lattice snapping is reported.'
      : 'Minimum clearances may expand to retain AMR levels. Ground and symmetry planes stay anchored; native lattice snapping is reported.';
  }

  function syncModelControls() {
    const widget = $('model-widget');
    for (const mode of ['none', 'move', 'rotate']) {
      const radio = $(`model-widget-${mode}`);
      if (radio) { radio.checked = widget?.value === mode; radio.disabled = !!widget?.disabled; }
    }
    const symmetry = $('boundary')?.value === 'symmetry';
    const center = $('symmetry-auto-center-y');
    if ($('symmetry-placement-controls')) $('symmetry-placement-controls').hidden = !symmetry;
    for (const mode of ['wall', 'center']) {
      const radio = $(`symmetry-placement-${mode}`);
      if (radio) { radio.checked = !!center?.checked === (mode === 'center'); radio.disabled = !symmetry || !!center?.disabled; }
    }
    syncDomainControls();
    syncLiveResults();
  }

  function setResultsAvailable(available) {
    resultsAvailable = !!available;
    const menu = $('menu-results'), summary = $('results-menu-summary');
    if (!menu || !summary) return;
    menu.classList.toggle('results-available', resultsAvailable);
    menu.classList.toggle('results-disabled', !resultsAvailable);
    // Opening saved results must remain discoverable before a case is loaded.
    summary.setAttribute('aria-disabled', 'false');
    summary.setAttribute('tabindex', '0');
    summary.setAttribute('title', resultsAvailable ? 'Inspect available results' : 'Open a saved study or import native / VTK results');
  }

  function positionMenus() {
    const bar = menus[0]?.closest('.menu-bar');
    const rect = bar?.getBoundingClientRect?.();
    if (rect) document.documentElement?.style?.setProperty('--aero-menu-bottom', `${rect.bottom + 6}px`);
  }

  function closeMenus(except) {
    for (const menu of menus) if (menu !== except) menu.open = false;
  }

  function selectParameters(name, focusId) {
    const selected = panels.find(panel => panel.dataset.parameterPanel === name);
    if (!selected) return false;
    for (const panel of panels) panel.hidden = panel !== selected;
    const heading = selected.querySelector('.parameter-heading');
    if ($('parameter-title')) $('parameter-title').textContent = heading?.textContent || name;
    for (const choice of choices) choice.setAttribute('aria-pressed', String(choice.dataset.parameterTarget === name));
    for (const menu of menus) menu.classList.toggle('has-active-parameters', !!menu.querySelector(`[data-parameter-target="${name}"]`));
    closeMenus();
    if (focusId) {
      const control = $(focusId);
      control?.focus({preventScroll: true});
      control?.scrollIntoView({block: 'nearest'});
    }
    return true;
  }

  function setCard(id, visible, focusClose = false) {
    if (visible && $('result-pane') && $('result-pane').value !== 'scene') {
      $('result-pane').value = 'scene';
      $('result-pane').dispatchEvent(new Event('change', {bubbles: true}));
    }
    for (const other of cards) {
      const card = $(other);
      if (!card) continue;
      if (other === id || visible) {
        card.hidden = other !== id || !visible;
        $(openers[other])?.setAttribute('aria-expanded', String(!card.hidden));
      }
    }
    const modalVisible = !$('viewer-message')?.hidden;
    if ($('viewer-message-backdrop')) $('viewer-message-backdrop').hidden = !modalVisible;
    if (visible && focusClose) $(id)?.querySelector('.canvas-close')?.focus({preventScroll: true});
  }

  function closeMessage() {
    if (messageState.key) dismissedMessages.add(messageState.key);
    setCard('viewer-message', false);
  }

  function clearMessage(key) {
    if (messageState.key !== String(key)) return false;
    dismissedMessages.delete(messageState.key);
    if (lastAutoMessageKey === messageState.key) lastAutoMessageKey = null;
    showMessage('', {urgent: false});
    return true;
  }

  function toggleCard(id) {
    const card = $(id);
    if (!card) return;
    if (id === 'viewer-message' && !card.hidden) closeMessage();
    else {
      setCard(id, card.hidden);
      if (id === 'viewer-message' && !card.hidden) $('job-message')?.focus({preventScroll: true});
    }
  }

  function showMessage(value, options = {}) {
    const text = String(value || '').trim();
    const budget = /above\s+(?:gpu|vram)\s+budget|(?:gpu|vram)\s+budget\s+(?:exceeded|exceeds)/i.test(text);
    const urgent = options.urgent ?? (budget || /failed|cannot|could not|unable|blocked|error|too few|must be rebuilt/i.test(text));
    const scopedBudgetKey = `gpu-budget:${buildJob || 'draft'}`;
    const currentBudgetKey = messageState.key?.startsWith('gpu-budget:') && (!buildJob || messageState.key === scopedBudgetKey) ? messageState.key : scopedBudgetKey;
    const key = String(options.key || (budget ? currentBudgetKey : text));
    const recovery = options.recovery === undefined ? budget && !!messageState.recovery : !!options.recovery;
    const diagnosticCandidate = options.diagnosticUrl === undefined && key === messageState.key ? messageState.diagnosticUrl : options.diagnosticUrl;
    const diagnosticUrl = typeof diagnosticCandidate === 'string' && /^\/api\/aero\/diagnostics\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(diagnosticCandidate) ? diagnosticCandidate : null;
    if ($('viewer-message-diagnostic')) {
      $('viewer-message-diagnostic').hidden = !diagnosticUrl;
      if (diagnosticUrl) $('viewer-message-diagnostic').setAttribute('href', diagnosticUrl);
      else $('viewer-message-diagnostic').removeAttribute('href');
    }
    if (!recovery) {
      if ($('mesh-proposal-note')) $('mesh-proposal-note').hidden = true;
      if ($('mesh-proposal-apply')) $('mesh-proposal-apply').hidden = true;
    }
    if ($('job-message') && $('job-message').textContent !== text) $('job-message').textContent = text;
    if ($('viewer-message-title')) $('viewer-message-title').textContent = options.title || (budget ? 'Above GPU budget' : urgent ? 'Analysis needs attention' : 'Analysis message');
    if ($('viewer-message-open')) {
      $('viewer-message-open').hidden = !text;
      $('viewer-message-open').classList.toggle('notice-urgent', !!urgent);
    }
    messageState = {text, key, urgent, recovery, diagnosticUrl};
    if (!text) { setCard('viewer-message', false); return; }
    // Semantic keys let callers update measurements without reopening a warning
    // that the user has already dismissed during this job.
    if (options.force || (urgent && key !== lastAutoMessageKey && !dismissedMessages.has(key))) {
      setCard('viewer-message', true);
      $('job-message')?.focus({preventScroll: true});
    }
    if (urgent) lastAutoMessageKey = key;
  }

  function refreshNotices() {
    const messageText = $('job-message')?.textContent?.trim() || '';
    if (messageText !== messageState.text) {
      const newUrgent = /above\s+(?:gpu|vram)\s+budget|(?:gpu|vram)\s+budget\s+(?:exceeded|exceeds)|failed|cannot|could not|unable|blocked|error|too few|must be rebuilt/i.test(messageText);
      if (messageState.urgent && !$('viewer-message')?.hidden && !newUrgent) {
        // Routine cancellation/polling updates must not erase a warning before
        // it has been read. Explicit showMessage calls can replace it.
        $('job-message').textContent = messageState.text;
      } else showMessage(messageText);
    }
    const mesh = $('mesh-details');
    const meshText = !mesh?.hidden ? mesh?.textContent?.trim() || '' : '';
    if ($('mesh-notice-open')) $('mesh-notice-open').hidden = !meshText;
    if (!meshText) setCard('mesh-notice', false);
    else if (meshText !== lastMeshText) {
      lastMeshText = meshText;
      // Completed-build summaries are inspectable without covering the model.
      $('mesh-notice-open')?.classList.toggle('has-update', true);
    }
  }

  function closeMeshBuild() {
    if (buildTerminalKey) dismissedBuilds.add(buildTerminalKey);
    if ($('mesh-build-overlay')) $('mesh-build-overlay').hidden = true;
    if ($('mesh-build-open')) {
      $('mesh-build-open').hidden = !buildTerminalKey;
      $('mesh-build-open').setAttribute('aria-expanded', 'false');
    }
  }

  function setDisplayMode(mode) {
    if ($('view-mode')) { $('view-mode').value = mode; $('view-mode').dispatchEvent(new Event('change', {bubbles: true})); }
    if ($('result-pane')) { $('result-pane').value = 'scene'; $('result-pane').dispatchEvent(new Event('change', {bubbles: true})); }
    closeMenus();
  }

  function setMeshBuildProgress(update = {}) {
    const overlay = $('mesh-build-overlay');
    if (!overlay) return;
    const active = !!update.active;
    const job = update.jobId ?? buildJob;
    if (job !== buildJob) { buildJob = job; buildLog = []; $('mesh-build-log')?.replaceChildren(); }
    const status = String(update.status || update.state || update.phase || (active ? 'creating' : '')).toLowerCase();
    const terminal = /^(ready|mesh_ready|completed)$/.test(status) ? 'ready' : /^(error|failed|mesh_failed)$/.test(status) ? 'error' : null;
    buildState = terminal || (active ? 'creating' : 'idle');
    buildTerminalKey = terminal ? String(update.key || `${buildJob || 'draft'}:${terminal}`) : null;
    overlay.dataset.state = buildState;
    overlay.hidden = buildState === 'idle' || !!(buildTerminalKey && dismissedBuilds.has(buildTerminalKey));
    overlay.setAttribute('aria-busy', String(buildState === 'creating'));
    if ($('mesh-build-close')) $('mesh-build-close').hidden = !terminal;
    if ($('mesh-build-ready-actions')) $('mesh-build-ready-actions').hidden = terminal !== 'ready';
    if ($('mesh-build-open')) { $('mesh-build-open').hidden = !buildTerminalKey || !overlay.hidden; $('mesh-build-open').setAttribute('aria-expanded', String(!overlay.hidden)); }
    if (buildState === 'idle') return;
    const percent = terminal === 'ready' ? 100 : typeof update.percent === 'number' && Number.isFinite(update.percent) ? Math.min(100, Math.max(0, update.percent)) : null;
    if (percent === null) $('mesh-build-progress')?.removeAttribute('value');
    else if ($('mesh-build-progress')) $('mesh-build-progress').value = percent;
    if ($('mesh-build-percent')) $('mesh-build-percent').textContent = percent === null ? terminal === 'error' ? 'Error' : 'Working…' : `${Math.round(percent)}%`;
    if ($('mesh-build-title')) $('mesh-build-title').textContent = update.title || (terminal === 'ready' ? 'Mesh ready' : terminal === 'error' ? 'Mesh build failed' : 'Creating mesh');
    const stage = [update.stage, update.message].filter(Boolean).map(String);
    if ($('mesh-build-stage')) $('mesh-build-stage').textContent = [...new Set(stage)].join(' · ') || 'Building the lattice…';
    const normalize = entry => typeof entry === 'string' ? entry : String(entry?.message ?? entry?.text ?? entry?.line ?? '');
    let next;
    if (Array.isArray(update.log) && update.log.length) next = update.log.map(normalize).filter(Boolean).slice(-80);
    else {
      const line = String(update.message || update.stage || '');
      next = line && line !== buildLog.at(-1) ? [...buildLog, line].slice(-80) : buildLog;
    }
    if (next.length === buildLog.length && next.every((line, index) => line === buildLog[index])) return;
    const log = $('mesh-build-log');
    if (log) {
      const pinned = !log.scrollHeight || log.scrollHeight - log.scrollTop - log.clientHeight < 30;
      const prefix = next.length >= buildLog.length && buildLog.every((line, index) => line === next[index]);
      if (!prefix) log.replaceChildren();
      for (const line of next.slice(prefix ? buildLog.length : 0)) {
        const item = document.createElement('li'); item.textContent = line; log.appendChild(item);
      }
      if (pinned) log.scrollTop = log.scrollHeight;
    }
    buildLog = next;
  }

  function closeSolverProgress() {
    if (solverTerminalKey) dismissedSolvers.add(solverTerminalKey);
    if ($('solver-progress-overlay')) $('solver-progress-overlay').hidden = true;
    if ($('solver-progress-open')) { $('solver-progress-open').hidden = !solverTerminalKey; $('solver-progress-open').setAttribute('aria-expanded', 'false'); }
  }

  function setSolverProgress(update = {}) {
    const card = $('solver-progress-overlay'); if (!card) return;
    const job = update.jobId ?? solverJob;
    if (job !== solverJob) { solverJob = job; solverLog = []; solverMetrics = {}; $('solver-progress-log')?.replaceChildren(); }
    const status = String(update.status || (update.active ? 'running' : 'idle')).toLowerCase();
    const terminal = /^(completed|complete|ready)$/.test(status) ? 'completed' : /^(failed|error)$/.test(status) ? 'error' : /^(cancelled|canceled)$/.test(status) ? 'cancelled' : null;
    const active = !!update.active, visible = active || !!terminal;
    if (terminal && update.restored) dismissedSolvers.add(String(update.key || `${solverJob || 'draft'}:${terminal}`));
    solverTerminalKey = terminal ? String(update.key || `${solverJob || 'draft'}:${terminal}`) : null;
    card.dataset.state = terminal || (status === 'waiting_for_gpu' || status === 'queued' ? 'waiting' : 'running');
    card.hidden = !visible || !!(solverTerminalKey && dismissedSolvers.has(solverTerminalKey));
    card.setAttribute('aria-busy', String(active));
    if ($('solver-progress-close')) $('solver-progress-close').hidden = !terminal;
    if ($('solver-progress-open')) { $('solver-progress-open').hidden = !solverTerminalKey || !card.hidden; $('solver-progress-open').setAttribute('aria-expanded', String(!card.hidden)); }
    if (!visible) return;
    const finite = value => typeof value === 'number' && Number.isFinite(value);
    const percent = terminal === 'completed' ? 100 : finite(update.percent) ? Math.max(0, Math.min(100, update.percent)) : null;
    if (percent === null) $('solver-progress-bar')?.removeAttribute('value'); else if ($('solver-progress-bar')) $('solver-progress-bar').value = percent;
    if ($('solver-progress-title')) $('solver-progress-title').textContent = terminal === 'completed' ? 'Solver completed' : terminal === 'error' ? 'Solver failed' : terminal === 'cancelled' ? 'Solver cancelled' : 'Solver progress';
    if ($('solver-progress-percent')) $('solver-progress-percent').textContent = percent === null ? '—' : `${percent.toFixed(percent < 10 ? 1 : 0)}%`;
    if ($('solver-progress-phase')) $('solver-progress-phase').textContent = [...new Set([update.phase, update.message].filter(Boolean).map(String))].join(' · ') || status.replaceAll('_', ' ');
    if (update.metrics) solverMetrics = update.metrics;
    const metrics = solverMetrics, holder = $('solver-progress-metrics');
    const number = value => finite(value) ? value.toLocaleString(undefined, {maximumFractionDigits: 2}) : null;
    const memory = value => finite(value) && value >= 0 ? `${value.toFixed(2)} GiB` : 'Not reported';
    const duration = value => !finite(value) || value < 0 ? null : value >= 3600 ? `${Math.floor(value / 3600)}h ${Math.floor(value % 3600 / 60)}m` : value >= 60 ? `${Math.floor(value / 60)}m ${Math.round(value % 60)}s` : `${Math.round(value)}s`;
    const residual = metrics.residual == null ? metrics.residual_reason || 'Not exported' : finite(metrics.residual) ? metrics.residual.toExponential(2) : typeof metrics.residual === 'object' ? Object.entries(metrics.residual).map(([key, value]) => `${key}: ${finite(value) ? value.toExponential(2) : value}`).join(' · ') : String(metrics.residual);
    if (holder) {
      holder.replaceChildren();
      const add = (label, value, used, total, hint) => {
        const row = document.createElement('div'), name = document.createElement('dt'), text = document.createElement('dd');
        name.textContent = label; text.textContent = value; if (hint) row.title = hint;
        row.append(name, text);
        if (finite(used) && finite(total) && total > 0) { const gauge = document.createElement('progress'); gauge.max = total; gauge.value = Math.max(0, Math.min(total, used)); gauge.setAttribute('aria-label', label); row.appendChild(gauge); }
        holder.appendChild(row);
      };
      add('Iterations', number(metrics.iteration) ? `${number(metrics.iteration)}${number(metrics.total_iterations) ? ' / ' + number(metrics.total_iterations) : ''}` : 'Not reported');
      add('Residual', residual);
      add('Elapsed', duration(metrics.elapsed_s) || 'Not reported');
      add('ETA', metrics.eta_credible === false ? 'Not estimated' : duration(metrics.eta_s) || 'Not estimated');
      add('GPU VRAM', finite(metrics.gpu_used_gb) && finite(metrics.gpu_total_gb) ? `${memory(metrics.gpu_used_gb)} / ${memory(metrics.gpu_total_gb)}` : 'Not reported', metrics.gpu_used_gb, metrics.gpu_total_gb);
      add(metrics.gpu_process_source === 'recorded' ? 'Solver GPU · sample' : 'Solver GPU', memory(metrics.gpu_process_gb));
      add(metrics.process_ram_source === 'recorded' ? 'Process RAM · sample' : 'Process RAM', memory(metrics.process_ram_gb), undefined, undefined, metrics.pid ? `Solver process PID ${metrics.pid}` : undefined);
      add('System RAM', finite(metrics.system_ram_used_gb) && finite(metrics.system_ram_total_gb) ? `${memory(metrics.system_ram_used_gb)} / ${memory(metrics.system_ram_total_gb)}` : 'Not reported', metrics.system_ram_used_gb, metrics.system_ram_total_gb);
    }
    const normalize = entry => typeof entry === 'string' ? entry : String(entry?.message ?? entry?.text ?? entry?.line ?? '');
    const fallback = String(update.message || update.phase || '');
    const next = Array.isArray(update.log) && update.log.length ? update.log.map(normalize).filter(Boolean).slice(-80) : fallback && fallback !== solverLog.at(-1) ? [...solverLog, fallback].slice(-80) : solverLog;
    if (next.length !== solverLog.length || next.some((line, index) => line !== solverLog[index])) {
      const log = $('solver-progress-log'); if (log) {
        const pinned = !log.scrollHeight || log.scrollHeight - log.scrollTop - log.clientHeight < 30;
        const prefix = next.length >= solverLog.length && solverLog.every((line, index) => line === next[index]); if (!prefix) log.replaceChildren();
        for (const line of next.slice(prefix ? solverLog.length : 0)) { const item = document.createElement('li'); item.textContent = line; log.appendChild(item); }
        if (pinned) log.scrollTop = log.scrollHeight;
      }
      solverLog = next;
    }
  }

  function applyTooltips() {
    const tips = {
      'model-file':'Load STL, OBJ or GLB geometry. GLB materials are retained when the original file is available.',
      'model-appearance':'Choose the neutral surface or original GLB materials and textures. Computed fields use their own colors.',
      'model-projections':'Project model silhouettes onto domain walls to inspect placement and clearances.',
      'model-widget-none':'Turn off model transform widgets to orbit and inspect the scene.',
      'model-widget-move':'Drag the colored axes to translate the model. Numeric offsets update together.',
      'model-widget-rotate':'Drag the rotation rings to orient the model. Numeric angles update together.',
      'symmetry-placement-wall':'Place a prepared half model at the symmetry wall using its minimum Y edge.',
      'symmetry-placement-center':'Place the centre of a full model on the symmetry plane; simulate its positive-Y half.',
      'geometry-handling':'Closed solid validates topology. Intentional open mode accepts ambiguous or open surfaces using two-sided wall intersections.',
      'boundary':'Choose free-air, ground-wall or symmetry boundary conditions independently of domain sizing.',
      'surface-dx':'Requested finest cell size near the model surface, in metres.',
      'levels':'Requested AMR levels. The mesher retains this count or offers a different setup.',
      'mesh-sizing':'Choose an explicit surface spacing or let CPU mesh probes find a spacing within the VRAM target.',
      'target-vram':'Percentage of currently available VRAM allowed for this simulation, checked before allocation.',
      'build-mesh':'Build and inspect the AMR lattice using the selected backend before starting flow iterations.',
      'run-solver':'Run the selected solver. Choose a mandatory output folder under Outputs first.',
      'cancel-job':'Request cancellation of the active meshing or solver job; completed output files remain available.',
      'section-enabled':'Enable an inspection plane through the geometry, mesh and available field data.',
      'section-axis':'Choose the normal direction of the movable inspection plane.',
      'section-slider':'Move the inspection plane through the current domain; this does not create a recorded solver plane.',
      'section-position':'Position of the inspection plane in solver-domain metres.',
      'section-field':'Choose the quantity on the inspection plane. Arbitrary field positions require exported volume data.',
      'section-mesh':'Draw actual cell edges on the section, omitting coarse cells covered by finer levels.',
      'section-clip':'Cut away one side of the model and mesh at the inspection plane.',
      'section-record':'Add the current inspection plane to the planes the solver will record during the next run.',
      'section-show':'Open a zoomable 2D view of the current inspection plane; this does not record a new solver output.',
      'add-cut':'Add a plane to record during the next solver run. Its location must be set before running.',
      'cut-result-select':'Choose an already recorded solver plane and iteration to display.',
      'cut-field':'Choose a quantity from the selected recorded cut-plane output.',
      'surface-field':'Color the model using an exported surface field; unavailable quantities are reported explicitly.',
      'velocity-vectors':'Choose prescribed inlet arrows, vectors from the selected recorded cut, both, or none.',
      'output-folder':'Mandatory destination for solver output files. Each run creates its own subfolder.',
      'results-import':'Read saved native or VTK files into the viewer without starting a solver.',
      'study-save':'Save the complete setup, original geometry and result pointers using the current destination.',
      'study-save-menu':'Save the current study to its associated file; choose a destination on the first save.',
      'study-save-as':'Choose another study filename and destination using the system save dialog.'
    };
    for (const [id, title] of Object.entries(tips)) if ($(id)) $(id).setAttribute('title', title);
    for (const control of document.querySelectorAll('button, select, input')) {
      if (control.getAttribute('title') || control.type === 'hidden') continue;
      const text = control.getAttribute('aria-label') || control.closest('label')?.textContent || control.textContent;
      if (text?.trim()) control.setAttribute('title', text.trim().replace(/\s+/g, ' ').slice(0, 180));
    }
  }

  for (const menu of menus) {
    menu.querySelector('summary')?.addEventListener('click', event => {
      positionMenus(); if (!menu.open) closeMenus(menu);
    });
    menu.addEventListener('toggle', () => {
      if (menu.open) closeMenus(menu);
    });
  }
  for (const choice of choices) choice.addEventListener('click', () => selectParameters(choice.dataset.parameterTarget, choice.dataset.focusControl));
  for (const choice of document.querySelectorAll('[data-result-target]')) choice.addEventListener('click', () => {
    const picker = $('result-pane');
    if (picker) { picker.value = choice.dataset.resultTarget; picker.dispatchEvent(new Event('change', {bubbles: true})); }
    closeMenus();
  });
  for (const choice of document.querySelectorAll('[data-open-info]')) choice.addEventListener('click', () => {
    closeMenus(); setCard('viewer-info-panel', true, true);
  });
  for (const choice of document.querySelectorAll('[data-display-mode]')) choice.addEventListener('click', () => setDisplayMode(choice.dataset.displayMode));
  for (const mode of ['none', 'move', 'rotate']) $(`model-widget-${mode}`)?.addEventListener('change', event => {
    if (!event.target.checked || !$('model-widget')) return;
    $('model-widget').value = mode; $('model-widget').dispatchEvent(new Event('change', {bubbles: true})); syncModelControls();
  });
  for (const mode of ['wall', 'center']) $(`symmetry-placement-${mode}`)?.addEventListener('change', event => {
    if (!event.target.checked || !$('symmetry-auto-center-y')) return;
    $('symmetry-auto-center-y').checked = mode === 'center'; $('symmetry-auto-center-y').dispatchEvent(new Event('change', {bubbles: true})); syncModelControls();
  });
  for (const id of ['boundary', 'model-widget', 'symmetry-auto-center-y']) $(id)?.addEventListener('change', syncModelControls);
  $('domain-sizing')?.addEventListener('change', syncDomainControls);
  $('live-results')?.addEventListener('change', syncLiveResults);
  $('run-live-results')?.addEventListener('change', () => { if ($('live-results')) { $('live-results').checked = $('run-live-results').checked; $('live-results').dispatchEvent(new Event('change', {bubbles:true})); } });
  $('solver-progress-close')?.addEventListener('click', closeSolverProgress);
  $('solver-progress-open')?.addEventListener('click', () => { if ($('solver-progress-overlay')) $('solver-progress-overlay').hidden = false; $('solver-progress-open').hidden = true; $('solver-progress-open').setAttribute('aria-expanded','true'); });
  $('mesh-build-close')?.addEventListener('click', closeMeshBuild);
  $('mesh-build-open')?.addEventListener('click', () => {
    if (!$('mesh-build-overlay')) return;
    $('mesh-build-overlay').hidden = false; $('mesh-build-open').hidden = true; $('mesh-build-open').setAttribute('aria-expanded', 'true');
  });
  $('mesh-build-inspect')?.addEventListener('click', () => { setDisplayMode('mesh'); closeMeshBuild(); });
  $('mesh-build-details')?.addEventListener('click', () => { setCard('mesh-notice', true); closeMeshBuild(); });
  $('model-file')?.addEventListener('change', () => selectParameters('model'));
  $('study-file')?.addEventListener('change', () => selectParameters('model'));
  $('study-open')?.addEventListener('click', () => closeMenus());
  $('study-save-menu')?.addEventListener('click', () => closeMenus());
  for (const id of ['fit-model', 'fit-domain', 'viewer-fullscreen', 'build-mesh', 'run-solver', 'cancel-job', 'results-refresh']) $(id)?.addEventListener('click', () => closeMenus());
  for (const [id, opener] of Object.entries(openers)) $(opener)?.addEventListener('click', () => toggleCard(id));
  for (const [id, closer] of Object.entries({'viewer-info-panel': 'viewer-info-close', 'mesh-notice': 'mesh-notice-close', 'viewer-message': 'viewer-message-close'})) {
    $(closer)?.addEventListener('click', () => { if (id === 'viewer-message') closeMessage(); else setCard(id, false); $(openers[id])?.focus({preventScroll: true}); });
  }
  document.addEventListener('click', event => { if (!event.target.closest?.('.top-menu')) closeMenus(); });
  document.addEventListener('keydown', event => {
    const modalVisible = $('viewer-message') && !$('viewer-message').hidden;
    if (modalVisible && event.key === 'Tab') {
      const controls = [$('job-message'), $('mesh-proposal-apply'), $('viewer-message-diagnostic'), $('viewer-message-close')].filter(control => control && !control.hidden && !control.disabled);
      const index = controls.indexOf(document.activeElement);
      const next = index < 0 ? 0 : (index + (event.shiftKey ? -1 : 1) + controls.length) % controls.length;
      controls[next]?.focus({preventScroll: true}); event.preventDefault(); return;
    }
    if (event.key !== 'Escape') return;
    if (modalVisible) { closeMessage(); $('viewer-message-open')?.focus(); event.preventDefault(); return; }
    const openMenu = menus.find(menu => menu.open);
    if (openMenu) { closeMenus(); openMenu.querySelector('summary')?.focus(); event.preventDefault(); return; }
    const visible = cards.find(id => $(id) && !$(id).hidden);
    if (visible) { setCard(visible, false); $(openers[visible])?.focus(); event.preventDefault(); }
  });
  window.addEventListener?.('resize', positionMenus);
  if (typeof MutationObserver !== 'undefined') {
    const observer = new MutationObserver(refreshNotices);
    for (const id of ['mesh-details', 'job-message']) if ($(id)) observer.observe($(id), {childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ['hidden']});
  }
  selectParameters('model');
  syncModelControls();
  applyTooltips();
  setResultsAvailable(false);
  refreshNotices();
  window.LudwigAeroLayout = {selectParameters, refreshNotices, closeMenus, showMessage, closeMessage, clearMessage, setMeshBuildProgress, setSolverProgress, setResultsAvailable, syncModelControls, syncDomainControls, syncLiveResults, applyTooltips, showInfo: () => setCard('viewer-info-panel', true)};
})();
