"use strict";

// Published data has its own lifetime, separate from a running/saved CFD job.
window.AirfoilBenchmarks = (() => {
  const $ = id => document.getElementById(id);
  let items = [], loading = null, selection = 0, selectedId = null;
  async function catalog() {
    if (items.length) return items;
    if (!loading) loading = fetch('/api/airfoil/benchmarks').then(async response => {
      if (!response.ok) throw new Error('Could not load the bundled literature cases. Restart the updated Julia service.');
      const data = await response.json(); items = data.items; return items;
    }).finally(() => { loading = null; });
    return loading;
  }
  function current() { return items.find(item => item.id === selectedId); }
  function isMatch(item, request) {
    if (!item || !request || request.airfoil.kind !== item.kind) return false;
    const f = request.flow, expected = item.case.flow;
    if ('gurney_height' in f || Number(f.cfd_te_thickness)>0 || f.cfd_fully_turbulent || f.cfd_transition_model==='fully_turbulent') return false;
    if (request.sweep?.enabled) return false;
    return ['mach','reynolds','alpha_deg','trip_upper','trip_lower'].every(key =>
      Number.isFinite(f[key]) && Math.abs(f[key]-expected[key]) <= Math.max(1e-8,Math.abs(expected[key])*1e-8)) &&
      f.cfd_viscosity_model===expected.cfd_viscosity_model &&
      (expected.cfd_viscosity_model!=='sutherland' || Math.abs(f.cfd_freestream_temperature_k-expected.cfd_freestream_temperature_k)<1e-6) &&
      f.cfd_time_mode===expected.cfd_time_mode;
  }
  function refresh() {
    const kind = el['source-kind'].value, available = items.filter(item=>item.kind===kind);
    if (!available.some(item=>item.id===selectedId)) selectedId=available.find(item=>item.default)?.id||null;
    const item=current(), enabled=Boolean(item);
    $('benchmark-controls').classList.toggle('hidden',!enabled);
    $('literature-panel').classList.toggle('hidden',!enabled||!el['show-reference'].checked);
    $('literature-menu').classList.toggle('hidden',!enabled);
    window.dispatchEvent(new CustomEvent('airfoil-reference-updated'));
    state.previewReference=null;
    if (!item) return;
    for (const id of ['benchmark-select','literature-select']) {
      const select=$(id);
      if (select.dataset.kind!==kind) {
        select.replaceChildren(...available.map(entry=>new Option(entry.label,entry.id)));
        select.dataset.kind=kind;
      }
      select.value=selectedId;
    }
    let request=null;
    try { request=collectCase(); } catch (_) { /* Incomplete editable draft. */ }
    const matching=isMatch(item,request);
    if (matching) state.previewReference=item.reference;
    $('benchmark-note').textContent=item.note+' '+item.numerics_note;
    $('benchmark-match').textContent=matching ? 'Published operating conditions loaded.' :
      'Edited setup: these literature curves are shown for context. Apply the published case to restore its conditions.';
    $('benchmark-match').classList.toggle('study-warning',!matching);
    const point=item.reference.points[0], cp=point.cp;
    $('literature-title').textContent=item.label+' — experimental mean Cp';
    $('literature-note').textContent=item.reference.note+' '+(matching?'Current draft matches these operating conditions.':'Current draft differs from these conditions.');
    $('literature-source').href=item.reference.source_url;
    $('literature-conditions').href=item.conditions_url;
    putChart('literature-chart',polyline([
      {name:'Measured upper',color:'#9b6bdd',points:cp.upper.map(p=>({x:p.x,y:p.cp})),dash:'6 3'},
      {name:'Measured lower',color:'#37ad83',points:cp.lower.map(p=>({x:p.x,y:p.cp})),dash:'6 3'}
    ],{invertY:true,title:'Published pressure coefficient · x/c'}),'Reference data unavailable.');
    renderPoint();
  }
  function apply(item,{show=true}={}) {
    if (!item) return;
    selectedId=item.id;
    state.studyFlowExtras={}; state.embeddedAirfoil=null;
    const f=item.case.flow;
    // Select the supported topology before updateSetup can enforce C+H's
    // current fully-turbulent/constant-kinematic constraints.
    const values={'source-kind':item.kind,backend:'transonic',mach:f.mach,reynolds:f.reynolds,alpha:f.alpha_deg,ncrit:f.ncrit,
      'cfd-te-topology':'aplus','cfd-mesh':f.cfd_mesh,'cfd-resolution':'fine','preview-resolution':'fine',
      'transition-mode':'forced','trip-upper':f.trip_upper,'trip-lower':f.trip_lower,
      'cfd-viscosity-model':f.cfd_viscosity_model,'cfd-temperature':f.cfd_freestream_temperature_k,
      'cfd-time-mode':f.cfd_time_mode,'trailing-edge-thickness':'','cfd-sfd-storage':'float32','cfd-n-base':4,
      'cfd-steps':item.unsteady?2000000:500000,'cfd-duration-ctu':100,'cfd-burnin-ctu':20};
    for (const [id,value] of Object.entries(values)) if($(id)) $(id).value=String(value);
    for (const id of ['sweep-enabled','gurney-enabled','streamline-align','lbm-spoiler']) if($(id)) $(id).checked=false;
    $('cfd-gpu').checked=true; $('cfd-rk').checked=!item.unsteady; $('cfd-auto-stop').checked=!item.unsteady;
    el['show-reference'].checked=true;
    state.machEdited=true; state.reynoldsEdited=true;
    updateSetup(); scheduleGeometryPreview(); refresh();
    window.AirfoilStudies?.markDirty();
    if(show) switchTab('results');
  }
  async function selectSource(kind) {
    const token=++selection;
    if(!['rae2822','oat15a'].includes(kind)) { refresh(); return; }
    $('benchmark-controls').classList.remove('hidden');
    $('benchmark-match').textContent='Loading published operating conditions…';
    try {
      await catalog();
      if(token!==selection||el['source-kind'].value!==kind)return;
      apply(items.find(item=>item.kind===kind&&item.default));
    } catch(error) { displayError(error); }
  }
  async function restore(id) {
    ++selection;
    if(['rae2822','oat15a'].includes(el['source-kind'].value))await catalog();
    selectedId=id||null;refresh();
  }
  for(const id of ['benchmark-select','literature-select']) $(id).addEventListener('change',()=>apply(items.find(item=>item.id===$(id).value)));
  $('benchmark-apply').addEventListener('click',()=>apply(current()));
  $('literature-menu').addEventListener('click',()=>{switchTab('results');$('literature-panel').scrollIntoView({block:'nearest'});});
  document.addEventListener('change',event=>{
    if(event.target.matches('.setup-column input,.setup-column select,#cfd-te-topology,#show-reference')) refresh();
  });
  return {catalog,selectSource,refresh,restore,apply,isMatch,current,get selectedId(){return selectedId;}};
})();
