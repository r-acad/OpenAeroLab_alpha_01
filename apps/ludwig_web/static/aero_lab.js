"use strict";

/* Aero Lab's browser workbench. Source geometry stays local until it is
   converted to STL and uploaded; build-only solver artifacts are authoritative
   for the actual domain and lattice. */
(() => {
  const $ = id => document.getElementById(id);
  const finite = value => value!==null&&value!==undefined&&value!==''&&Number.isFinite(Number(value));
  const fmt = value => finite(value) ? Number(value).toLocaleString(undefined, {maximumFractionDigits: 2}) : "—";
  const colorStops = [[34,83,151],[41,148,190],[95,204,193],[238,215,119],[233,132,79],[166,57,68]];
  const activeStatuses = new Set(['meshing','sizing','mesh_sizing','waiting_for_gpu','running','queued','starting']);
  const jobStorageKey = 'ludwig:aero-lab:last-job';
  const jobIdPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  const state = {engine:null,scene:null,camera:null,source:null,sourceBounds:null,modelId:null,
    jobId:null,job:null,amr:null,amrUrl:null,surface:null,lastSurfaceAt:0,lastSurfaceKey:null,
    mode:"geometry",sceneObjects:[],pollTimer:null,cuts:[],capabilities:null,uploading:false,
    importing:false,renderQueued:false,selectedCut:"",cutRenderKey:null,draftDirty:false,activeConfig:null,
    resumePending:false,resumeDraftEdited:false,sourceModelId:null,labels:[],frame:null,
    vectorKey:null,vectorData:null,vectorLoading:false,vectorError:''};
  let inspector=null,modelTools=null,resultBrowser=null,referenceTools=null;
  state.meshVisible=true;state.liveResults=true;

  function message(text, error=false) {
    const node=$('model-state');node.textContent=text;node.classList.toggle('error',error);
    node.classList.toggle('success',!error&&Boolean(state.modelId));
  }
  function progress(text,status) {
    $('job-message').textContent=text||'';
    $('job-state').textContent=status||'No case yet';
    window.LudwigAeroLayout?.refreshNotices();
  }
  function jobLink(id) {
    const url=new URL(window.location.href);url.searchParams.set('job',id);return url;
  }
  function persistJob(id) {
    if(!jobIdPattern.test(id))return;
    const url=jobLink(id),link=$('case-link');link.href=url.href;link.hidden=false;
    try{window.localStorage.setItem(jobStorageKey,id);}catch{}
    try{window.history.replaceState(null,'',url.pathname+url.search+url.hash);}catch{}
  }
  function clearPersistedJob() {
    $('case-link').hidden=true;$('case-link').removeAttribute('href');resultBrowser?.render(null);window.LudwigAeroLayout?.setSolverProgress?.({active:false,status:'idle'});
    try{window.localStorage.removeItem(jobStorageKey);}catch{}
    try{const url=new URL(window.location.href);url.searchParams.delete('job');window.history.replaceState(null,'',url.pathname+url.search+url.hash);}catch{}
  }
  function storedJobId() {
    let query=null,saved=null;
    try{query=new URL(window.location.href).searchParams.get('job');}catch{}
    try{saved=window.localStorage.getItem(jobStorageKey);}catch{}
    return [query,saved].find(id=>id&&jobIdPattern.test(id))||null;
  }
  function apiUrl(value) { return typeof value==='string'&&value.startsWith('/api/aero/')?value:null; }
  async function api(path,options={}) {
    const response=await fetch(path,{cache:'no-store',...options});
    let body;
    try { body=await response.json(); } catch { throw new Error(`Service returned HTTP ${response.status} without JSON.`); }
    if(!response.ok||body?.ok===false){const error=new Error(body?.error||body?.message||`HTTP ${response.status}`);error.status=response.status;error.details=body;throw error;}
    return body;
  }
  function inputNumber(id,label,{min=-Infinity,max=Infinity,integer=false}={}) {
    const raw=$(id).value.trim(),n=Number(raw);
    if(!raw||!Number.isFinite(n)||n<min||n>max||(integer&&!Number.isInteger(n)))
      throw new Error(`Enter a valid ${label}${Number.isFinite(min)?` (at least ${min})`:''}.`);
    return n;
  }
  function transformSettings() {
    return {scale:['x','y','z'].map(a=>inputNumber('scale-'+a,'scale '+a,{min:0.00001,max:100000})),
      rotation_deg:['x','y','z'].map(a=>inputNumber('rotate-'+a,'rotation '+a,{min:-360,max:360})),
      flow_angles:[Number($('flow-alpha')?.value)||0,$('boundary').value==='symmetry'?0:Number($('flow-beta')?.value)||0],translation:['x','y','z'].map(a=>inputNumber('move-'+a,'offset '+a,{min:-10000,max:10000}))};
  }
  function symmetryPlacement(domain=null) {
    return domain ? (domain.symmetry_placement || (domain.symmetry_center_y ? 'center' : 'wall')) : ($('symmetry-placement')?.value || 'none');
  }
  function fixedSymmetryOffset(domain=null) {
    return Number(domain ? domain.symmetry_offset_y_m || 0 : $('symmetry-placement-offset-y')?.value || 0);
  }
  function caseSettings() {
    if(state.outdatedServer)throw new Error('This is an older workbench server. Relaunch Ludwig Workbench to use studies, automatic domain sizing and all seven AMR levels.');
    if(!state.modelId)throw new Error('Load a model first.');
    const solver=$('solver').value;
    if(!solver)throw new Error('Select an available solver.');
    const lattice=$('lattice').value;
    const geometryHandling=$('geometry-handling')?.value||'closed_solid';
    if(!['closed_solid','two_sided_open'].includes(geometryHandling))throw new Error('Select a solver geometry handling mode.');
    if(!['d3q27','d3q39','d3q19'].includes(lattice))throw new Error('Select D3Q27, D3Q39 or D3Q19.');
    if(lattice==='d3q39'&&solver!=='main')throw new Error('D3Q39 requires the main LUDWIG solver.');
    const octree=solver==='q19_octree';
    if(lattice==='d3q19'&&!octree)throw new Error('D3Q19 is the lattice of the CPU octree solver.');
    if(octree&&lattice!=='d3q19')throw new Error('The CPU octree solver uses the D3Q19 lattice.');
    if(octree&&$('mesh-sizing').value!=='manual')throw new Error('VRAM sizing is not available for the CPU octree solver; choose a manual surface cell size.');
    const boundary=$('boundary').value;
    if(!['free_air','ground','symmetry'].includes(boundary))throw new Error('Choose a boundary condition.');
    const sizing=$('domain-sizing')?.value||'minimum';
    if(!['minimum','fixed','proportions'].includes(sizing))throw new Error('Choose domain sizing.');
    const proportions=['x','y','z'].map(axis=>sizing==='proportions'?inputNumber('domain-ratio-'+axis,'domain proportion '+axis,{min:.000001,max:1000000}):(Number($('domain-ratio-'+axis)?.value)>0?Number($('domain-ratio-'+axis).value):1));
    const {flow_angles,...transform}=transformSettings();
    if(solver==='lbmtau'&&transform.translation.some(v=>v!==0))throw new Error('The pinned lbmtau solver cannot offset the model. Select the main solver or reset offsets.');
    if(boundary==='symmetry'&&symmetryPlacement()!=='none'&&transform.translation[1]!==0)throw new Error('Automatic symmetry alignment requires lateral offset Y = 0. Choose Leave model where it is for manual placement.');
    const thermal=thermalFlowSettings();
    const velocity=thermal?.velocity??inputNumber('velocity','flow speed',{min:0.01,max:1000}),soundSpeed=thermal?.speed_of_sound??inputNumber('sound-speed','sound speed',{min:1,max:100000});
    const maxMach=thermal?1.5:octree?0.3:lattice==='d3q39'?1.6:0.6;
    if(velocity/soundSpeed>maxMach)throw new Error(`Selected backend requires Mach ≤ ${maxMach}.`);
    const alpha=inputNumber('flow-alpha','angle of attack',{min:-85,max:85}),beta=boundary==='symmetry'?0:inputNumber('flow-beta','sideslip',{min:-85,max:85});
    return {reference:referenceTools?.read(),model_id:state.modelId,geometry_handling:geometryHandling,transform,domain:{
        upstream:inputNumber('upstream','upstream clearance',{min:0.02,max:10000}),
        downstream:inputNumber('downstream','downstream clearance',{min:0.02,max:10000}),
        lateral:inputNumber('lateral','lateral clearance',{min:0.02,max:10000}),
        height:inputNumber('height','vertical clearance',{min:0.02,max:10000}),boundary,
        x_offset_m:inputNumber('domain-x-offset','domain X offset',{min:-10000,max:10000}),
        symmetry_center_y:symmetryPlacement()==='center',symmetry_placement:symmetryPlacement(),
        ...(symmetryPlacement()==='none'?{symmetry_offset_y_m:fixedSymmetryOffset()}:{}),sizing,proportions},
      surface_dx:inputNumber('surface-dx','surface cell size',{min:0.00001,max:100}),
      surface_triangle_max_edge_m:$('surface-triangle-max-edge-mm')?inputNumber('surface-triangle-max-edge-mm','maximum surface triangle edge in millimetres',{min:.1,max:100000})/1000:.05,
      levels:inputNumber('levels','AMR levels',{min:1,max:octree?10:7,integer:true}),solver,lattice,
      octree:octree?{band_fine:inputNumber('octree-band-fine','finest band',{min:3,max:64,integer:true}),band:inputNumber('octree-band','coarser bands',{min:3,max:64,integer:true}),
        wake_length:inputNumber('octree-wake','wake length',{min:0,max:20}),wake_level_offset:inputNumber('octree-wake-offset','wake level offset',{min:0,max:4,integer:true}),
        threads:inputNumber('octree-threads','CPU threads',{min:0,max:256,integer:true}),seal_cavities:Boolean($('octree-seal-cavities')?.checked),
        u_lattice:inputNumber('octree-u-lattice','lattice speed',{min:.00001,max:.12}),
        checkpoint_keep:inputNumber('octree-checkpoint-keep','retained checkpoints',{min:1,max:10,integer:true}),
        ramp_steps:$('octree-ramp-steps').value.trim()?inputNumber('octree-ramp-steps','startup ramp steps',{min:0,max:Number($('steps').value),integer:true}):null,
        max_plane_samples:inputNumber('octree-plane-samples','samples per recorded plane',{min:10000,max:16000000,integer:true})}:undefined,
      mesh_sizing:$('mesh-sizing').value,
      target_vram_percent:inputNumber('target-vram','percentage of available VRAM',{min:1,max:100}),
      outputs:window.LudwigOutputControls?.read()||{},display_every:window.LudwigOutputControls?.displayEvery()||100,
      flow:{alpha,beta,velocity,density:inputNumber('density','density',{min:0.01,max:10000}),
        kinematic_viscosity:inputNumber('viscosity','kinematic viscosity',{min:1e-10,max:1}),
        speed_of_sound:soundSpeed,...thermal},
      steps:inputNumber('steps','solver steps',{min:8,max:1000000,integer:true}),
      frames:inputNumber('frames','output frames',{min:2,max:40,integer:true}),
      cuts:state.cuts.map(c=>{
        const p=Number(c.position);if(String(c.position).trim()===''||!Number.isFinite(p)||p<0)throw new Error('Cut plane positions must be nonnegative metres from the solver domain origin.');
        return {axis:c.axis,position:p};
      })};
  }
  function validSettings() { try {caseSettings();return true;}catch{return false;} }
  function selectedGpuBudget(job=state.job) {
    const gpu=job?.estimate?.gpu||state.capabilities?.gpu;if(gpu?.available===false)return null;
    const free=Number(job?.case?.target_vram_available_gb??gpu?.free_gb),percent=Number($('target-vram').value),safe=Number(gpu?.budget_gb);
    if(free>0&&percent>=1&&percent<=100)return Math.min(free*percent/100,safe>0?safe:Infinity);
    return finite(job?.estimate?.budget_gb)?Number(job.estimate.budget_gb):null;
  }
  function gpuFit(job) {
    if((job?.case?.solver||$('solver')?.value)==='q19_octree')return null;
    if(job?.estimate?.measured===false&&!finite(job?.actual?.memory_gb))return null;
    const actual=job?.actual?.memory_gb,budget=selectedGpuBudget(job);
    if(finite(actual)&&finite(budget))return Number(actual)<=Number(budget);
    if(job?.estimate?.fits===null)return null;
    const memory=job?.estimate?.memory_gb;
    return finite(memory)&&finite(budget)?Number(memory)<=Number(budget):job?.estimate?.fits;
  }
  function updateBudgetGuard(job=state.job,cancel=true) {
    const over=gpuFit(job)===false,input=$('surface-dx'),note=$('surface-budget-note'),budget=selectedGpuBudget(job);
    input.classList.toggle('over-budget',over);if(over)input.setAttribute('aria-invalid','true');else input.removeAttribute('aria-invalid');
    if(note){note.hidden=!over;note.textContent=over?`Above GPU budget${finite(budget)?` (${Number(budget).toFixed(2)} GB)`:''}. Increase surface cell size, reduce levels, or raise the VRAM target. Active work is cancelled.`:'';}
    if(note)note.hidden=true;
    if(over){
      const key=`gpu-budget:${state.jobId||'draft'}`;
      if(state.budgetNoticeKey!==key){state.budgetNoticeKey=key;selectResultPane('scene');window.LudwigAeroLayout?.showMessage(note?.textContent||'Above GPU budget. Increase surface cell size or revise the VRAM target.',{key,urgent:true,title:'Above GPU budget',recovery:Boolean(state.meshProposal)});}
    }else state.budgetNoticeKey=null;
    if(!over||!cancel||!state.jobId||!activeStatuses.has(job?.status))return;
    state.budgetCancelIds||=new Set();const id=state.jobId;if(state.budgetCancelIds.has(id))return;state.budgetCancelIds.add(id);
    progress('Cancelling active work because the mesh exceeds the selected GPU budget.','Above GPU budget');
    api(`/api/aero/jobs/${encodeURIComponent(id)}/cancel`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reason:'gpu_budget_exceeded'})})
      .then(result=>{if(state.jobId===id){renderJob(result);pollJob(id);}})
      .catch(error=>{state.budgetCancelIds.delete(id);if(state.jobId===id)progress(`Could not cancel oversized work: ${error.message}`,'Cancellation failed');});
  }
  function canRun() {
    const s=state.job?.status;
    const actual=state.job?.actual,openMode=$('geometry-handling')?.value==='two_sided_open';
    const qualityAllows=actual?.quality?.runnable!==false||(openMode&&actual?.run_blocked_by_geometry===true&&actual?.run_blocked_by_mesh_quality!==true);
    return Boolean(state.jobId&&!state.draftDirty&&!state.surfaceRefinementPending&&!modelTools?.isInteracting()&&validSettings()&&s==='mesh_ready'&&gpuFit(state.job)!==false&&
      actual?.run_blocked_by_mesh_quality!==true&&qualityAllows);
  }
  function updateResultsAvailability(){
    const job=state.job,artifacts=job?.artifacts||{};
    const available=Boolean(state.importedResults||state.surface||state.amr||artifacts.amr||artifacts.summary||artifacts.surface_latest||(Array.isArray(artifacts.cuts)&&artifacts.cuts.length)||(job?.exports?.files||[]).some(file=>/\/run\/|\.(?:csv|lbmp|lbmslice|vtu|vtp|vtk)(?:$|\?)/i.test(file.url||file.name||'')));
    window.LudwigAeroLayout?.setResultsAvailable?.(available);updateResultViewStatus();return available;
  }
  function updateResultViewStatus(){
    const node=$('results-view-status');if(!node)return;
    const surface=state.importedResults?state.importedSurface:state.surface,cuts=state.job?.artifacts?.cuts||[];
    const parts=[];
    if(surface)parts.push(`Surface loaded${finite(surface.step)?` · iteration ${fmt(surface.step)}`:''}${surface.representation==='wall_samples'?` · ${surface.count.toLocaleString()} native wall samples (${surface.missing.toLocaleString()} missing values)`:' · original facet averages'}`);
    else if(state.job?.artifacts?.surface_latest)parts.push('Surface available · click Load results or Surface');
    if(cuts.length)parts.push(`${cuts.length} recorded plane files`);
    if(state.importedResults)parts.push(`Source: ${state.importedResults.name||'imported results'}`);
    if(state.surfaceError)parts.push(`Refresh issue: ${state.surfaceError}. Last loaded data retained.`);
    if(state.draftDirty&&(surface||cuts.length))parts.push('Showing results from the saved run; edited setup needs a new mesh');
    if(state.surfaceRefinementPending)parts.push(surface?.representation==='wall_samples'?'Geometry refinement edits do not add wall-field resolution; native samples follow the saved fluid lattice':`${$('surface-triangle-max-edge-mm')?.value||50} mm surface sampling is pending a new mesh; current results retain their original triangles`);
    const pane=$('result-pane')?.value||'scene';
    if(surface)parts.push(pane==='scene'&&state.mode==='surface'?'Surface colours visible':`Surface retained · current view: ${pane==='scene'?state.mode:pane==='cuts'?'recorded planes':'files and images'}`);
    node.textContent=parts.length?parts.join(' · '):'Open a study with results, or use Read results to load native / VTK files. Creating a mesh alone does not produce flow fields.';
    node.title=node.textContent;
  }
  function updateActions() {
    const requestedEdge=Number($('surface-triangle-max-edge-mm')?.value)/1000,builtEdge=Number(state.activeConfig?.surface_triangle_max_edge_m)||0;
    state.surfaceRefinementPending=Boolean(state.jobId&&state.activeConfig?.solver!=='q19_octree'&&requestedEdge>0&&Math.abs(requestedEdge-builtEdge)>1e-10*Math.max(1,requestedEdge,builtEdge));
    if($('mesh-visibility')){$('mesh-visibility').setAttribute('aria-pressed',String(state.meshVisible!==false));$('mesh-visibility').textContent=state.meshVisible===false?'Show mesh':'Hide mesh';}
    $('model-file').disabled=state.importing||state.uploading;
    if($('symmetry-auto-center-y'))$('symmetry-auto-center-y').disabled=$('boundary').value!=='symmetry'||!state.source||state.importing||state.uploading||activeStatuses.has(state.job?.status);
    $('build-mesh').disabled=state.importing||state.uploading||!validSettings()||activeStatuses.has(state.job?.status);
    $('run-solver').disabled=!canRun()||state.uploading;
    $('cancel-job').disabled=!state.jobId||!activeStatuses.has(state.job?.status);
    const meshing=['meshing','sizing','mesh_sizing'].includes(state.job?.status);
    $('build-mesh').textContent=meshing?'Creating mesh…':'Create mesh';
    $('build-mesh').classList.toggle('is-working',meshing);
    if($('model-widget'))$('model-widget').disabled=state.importing||state.uploading||activeStatuses.has(state.job?.status)||!state.modelId||Boolean(state.importedResults);
    updateModelAppearance();
    window.LudwigAeroLayout?.syncModelControls?.();updateResultsAvailability();
    const jobActive=activeStatuses.has(state.job?.status);if(modelTools&&state.modelToolJobActive!==jobActive){state.modelToolJobActive=jobActive;queueRender();}
  }
  function clearMeshProposal(){state.meshProposal=null;if($('mesh-proposal-apply'))$('mesh-proposal-apply').hidden=true;if($('mesh-proposal-note'))$('mesh-proposal-note').hidden=true;}
  function invalidateMesh() {
    clearMeshProposal();
    if(state.resumePending){
      state.resumeDraftEdited=true;state.draftDirty=true;
      progress('Your draft changed while the saved case is loading. Its result will remain available.','Draft changed');
      updateActions();queueRender();return;
    }
    if(activeStatuses.has(state.job?.status)){
      state.draftDirty=true;progress('The current job continues. Your changed settings require another lattice build.','Draft changed');
      updateActions();queueRender();return;
    }
    const hasFields=Boolean(state.surface||state.importedResults||state.job?.artifacts?.surface_latest||state.job?.artifacts?.cuts?.length);
    if(state.jobId&&state.modelId&&state.modelId===state.activeConfig?.model_id&&hasFields){
      // Editing a draft must not destroy the only route back to its saved run.
      // Keep field data, authoritative placement and the case link until the
      // user explicitly creates another mesh or opens another study.
      state.draftDirty=true;state.mode='geometry';syncViewButtons();
      progress('Setup edited. The saved run and loaded fields remain available under Results; build a new mesh to apply these changes.','Draft changed');
      updateBudgetGuard(null,false);updateActions();queueRender();return;
    }
    if(state.jobId){state.jobId=null;state.job=null;state.amr=null;state.amrUrl=null;state.surface=null;state.lastSurfaceKey=null;state.cutRenderKey=null;state.activeConfig=null;state.draftDirty=false;
      if(state.inspector){state.inspector.data=null;state.inspector.key=null;state.inspector.loading=false;state.inspector.error='';}
      clearPersistedJob();clearTimeout(state.pollTimer);
      $('metric-cells').textContent='—';$('metric-memory').textContent='—';$('metric-fit').textContent='Build the lattice';
      $('mesh-details').hidden=true;progress('Settings changed. Build the lattice again before running.','Needs rebuild');
      $('job-progress').value=0;$('artifact-links').replaceChildren();$('cut-image').hidden=true;}
    updateBudgetGuard(state.job,false);updateActions();queueRender();
  }

  function boundsOf(triangles) {
    const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
    for(let i=0;i<triangles.length;i+=3)for(let a=0;a<3;a++){
      const v=triangles[i+a];if(!Number.isFinite(v))continue;
      lo[a]=Math.min(lo[a],v);hi[a]=Math.max(hi[a],v);
    }
    if(lo.some(v=>!Number.isFinite(v)))throw new Error('Model has no finite triangle vertices.');
    return {min:lo,max:hi,center:lo.map((v,a)=>(v+hi[a])/2)};
  }
  function transformedPoint(x,y,z,t,center) {
    x=(x-center[0])*t.s[0];y=(y-center[1])*t.s[1];z=(z-center[2])*t.s[2];
    let next=y*t.cx-z*t.sx;z=y*t.sx+z*t.cx;y=next;
    next=x*t.cy+z*t.sy;z=-x*t.sy+z*t.cy;x=next;
    next=x*t.cz-y*t.sz;y=x*t.sz+y*t.cz;x=next;
    if(t.flow){const p=window.LudwigAeroReference?.multiply(window.LudwigAeroReference.rotation(t.flow),[x,y,z]);if(p)[x,y,z]=p;}
    return [x+t.d[0],y+t.d[1],z+t.d[2]];
  }
  function transformMath() {
    return rotationMath(transformSettings());
  }
  function rotationMath(p,flow=null) {
    const r=(p.rotation_deg||[0,0,0]).map(v=>v*Math.PI/180);
    return {flow:flow||(p.flow_angles?{alpha:p.flow_angles[0],beta:p.flow_angles[1]}:null),s:p.scale,d:p.translation,cx:Math.cos(r[0]),sx:Math.sin(r[0]),cy:Math.cos(r[1]),sy:Math.sin(r[1]),cz:Math.cos(r[2]),sz:Math.sin(r[2])};
  }
  function previewFrame() {
    if(!state.source)return null;
    const raw=state.source,p=transformSettings(),t=rotationMath(p),center=state.sourceBounds.center;
    // Keep the domain fixed when the user places the body. The native mesher
    // centers/scales/rotates the shape first, then applies placement separately.
    t.d=[0,0,0];
    const n=raw.length/9,out=new Float32Array(raw.length);
    let cursor=0;
    for(let tri=0;tri<n;tri++)for(let v=0;v<3;v++){
      const k=tri*9+v*3,p=transformedPoint(raw[k],raw[k+1],raw[k+2],t,center);
      out.set(p,cursor);cursor+=3;
    }
    const shape=out.subarray(0,cursor),boundary=$('boundary').value,ground=boundary==='ground';
    if(ground)for(let i=0;i<shape.length;i+=3){const y=shape[i+1];shape[i+1]=shape[i+2];shape[i+2]=-y;}
    const body=boundsOf(shape),ext=body.max.map((v,i)=>v-body.min[i]);
    const clearance=id=>Math.max(.02,Number($(id).value)||.02);
    let up=clearance('upstream'),down=clearance('downstream'),lat=clearance('lateral'),height=clearance('height');
    const placement=symmetryPlacement(),centeredSymmetry=boundary==='symmetry'&&placement==='center',fixedSymmetry=boundary==='symmetry'&&placement==='none';
    const occupiedY=fixedSymmetry?Math.max(0,body.max[1]+fixedSymmetryOffset()+p.translation[1]):centeredSymmetry?ext[1]/2:ext[1];
    const dims=[ext[0]+up+down,occupiedY+(boundary==='free_air'?2*lat:ground?height:lat),ext[2]+2*(ground?lat:height)];
    const sizing=$('domain-sizing')?.value||'minimum';
    const domainX=Number($('domain-x-offset')?.value)||0,relativeX=p.translation[0]-domainX,required=dims.slice();
    if(sizing!=='fixed')required[0]+=2*Math.abs(relativeX);
    const coarse=Math.max(.00001,Number($('surface-dx').value)||.01)*2**(Math.max(1,Number($('levels').value)||1)-1);
    const octree=$('solver').value==='q19_octree',minimumCells=octree?2:32;
    let target=required.slice();
    if(sizing==='proportions'){
      const ratio=['x','y','z'].map(axis=>Math.max(.000001,Number($('domain-ratio-'+axis)?.value)||1));
      if(boundary==='symmetry')ratio[1]=.5*Math.min(ratio[0],ratio[2]);
      const factor=Math.max(...required.map((size,axis)=>Math.max(size,minimumCells*coarse)/ratio[axis]));target=ratio.map(value=>value*factor);
    }else if(sizing==='minimum'&&boundary==='symmetry'){
      const floor=minimumCells*coarse,minimumY=Math.max(required[1],floor);
      target=[Math.max(required[0],floor,2*minimumY),0,Math.max(required[2],floor,2*minimumY)];target[1]=.5*Math.min(target[0],target[2]);
    }else if(sizing==='minimum')target=required.map(size=>size<minimumCells*coarse*(octree?1:1+1e-10)?(octree?2:40)*coarse:size);
    if(sizing==='proportions'||sizing==='minimum'&&boundary==='symmetry'){
      const block=(octree?2:8)*coarse;
      const blockCeil=value=>{const nearest=Math.round(value),tolerance=2**(Math.floor(Math.log2(Math.max(1,Math.abs(nearest))))-22);return Math.ceil(Math.abs(value-nearest)<=tolerance?nearest:value);};
      if(boundary==='symmetry'){const nx=Math.max(octree?2:8,2*blockCeil(target[0]/(2*block))),nz=Math.max(octree?2:8,2*blockCeil(target[2]/(2*block)));target=[nx*block,.5*Math.min(nx,nz)*block,nz*block];}
      else target=target.map(size=>Math.max(octree?1:4,blockCeil(size/block))*block);
    }
    const extra=target.map((size,axis)=>Math.max(0,size-dims[axis]));
    up+=extra[0]/2;down+=extra[0]/2;if(ground)height+=extra[1];else lat+=extra[1]/(boundary==='free_air'?2:1);
    if(ground)lat+=extra[2]/2;else height+=extra[2]/2;
    for(let axis=0;axis<3;axis++)dims[axis]+=extra[axis];
    const offset=[up-body.min[0]-domainX,boundary==='free_air'?dims[1]/2-body.center[1]:fixedSymmetry?fixedSymmetryOffset():centeredSymmetry?-body.center[1]:-body.min[1],dims[2]/2-body.center[2]];
    const move=ground?[p.translation[0],p.translation[2],-p.translation[1]]:p.translation;
    for(let i=0;i<3;i++)offset[i]+=move[i];
    for(let i=0;i<shape.length;i++)shape[i]+=offset[i%3];
    return {triangles:shape,body:boundsOf(shape),domain:{min:[0,0,0],max:dims},bodyOrigin:offset,boundary,ground,config:{transform:p,flow:{alpha:p.flow_angles[0],beta:p.flow_angles[1]},domain:{boundary,symmetry_center_y:centeredSymmetry,symmetry_placement:placement,symmetry_offset_y_m:fixedSymmetryOffset(),x_offset_m:domainX,sizing:$('domain-sizing')?.value||'minimum'}}};
  }
  function previewTriangles() {return previewFrame()?.triangles||null;}
  function binaryStl(triangles) {
    const count=triangles.length/9,buffer=new ArrayBuffer(84+count*50),view=new DataView(buffer);
    const title=new TextEncoder().encode('Aero Lab source geometry');new Uint8Array(buffer,0,title.length).set(title);
    view.setUint32(80,count,true);
    for(let i=0;i<count;i++){
      const j=i*9,base=84+i*50,ax=triangles[j+3]-triangles[j],ay=triangles[j+4]-triangles[j+1],az=triangles[j+5]-triangles[j+2],
        bx=triangles[j+6]-triangles[j],by=triangles[j+7]-triangles[j+1],bz=triangles[j+8]-triangles[j+2],
        nx=ay*bz-az*by,ny=az*bx-ax*bz,nz=ax*by-ay*bx,l=Math.hypot(nx,ny,nz)||1;
      for(let a=0;a<3;a++)view.setFloat32(base+4*a,[nx/l,ny/l,nz/l][a],true);
      for(let k=0;k<9;k++)view.setFloat32(base+12+4*k,triangles[j+k],true);
    }
    return buffer;
  }
  function readBinaryStl(buffer) {
    const view=new DataView(buffer);if(view.byteLength<84)throw new Error('Stored model STL is incomplete.');
    const count=view.getUint32(80,true);
    if(!count||count>1_000_000||84+count*50>view.byteLength)throw new Error('Stored model STL has invalid triangle data.');
    const triangles=new Float32Array(count*9);
    for(let i=0;i<count;i++)for(let k=0;k<9;k++)triangles[i*9+k]=view.getFloat32(84+i*50+12+k*4,true);
    return triangles;
  }
  async function loadStoredModel(modelId) {
    if(state.source&&state.sourceModelId===modelId){await restoreOriginalSource(modelId);return;}
    state.sourceModelId=modelId;
    try{
      const response=await fetch(`/api/aero/models/${encodeURIComponent(modelId)}/file`,{cache:'no-store'});
      if(!response.ok)throw new Error(`HTTP ${response.status}`);
      const source=readBinaryStl(await response.arrayBuffer());
      if(state.modelId!==modelId)return;
      state.source=source;state.sourceBounds=boundsOf(source);
      await restoreOriginalSource(modelId);
      if(state.camera)state.camera._aeroFit=false;
      queueRender();
    }catch(error){console.warn('Stored model preview unavailable:',error);}
  }
  async function restoreOriginalSource(modelId) {
    if(state.modelId!==modelId)return;
    const original=state.originalFile;
    // Reopening the same study resets originalFile. A remembered model ID alone
    // must never suppress retrieval of its retained GLB asset.
    if(state.sourceAssetModelId===modelId&&original&&state.sourceAssetFile===original){
      await modelTools?.refreshSource(original,{retry:true});updateModelAppearance();return;
    }
    const pending=state.sourceAssetRequest;
    if(pending?.modelId===modelId&&pending.original===original)return pending.promise;
    const request={modelId,original,promise:null};state.sourceAssetRequest=request;
    request.promise=(async()=>{
      try{
        let file=original;
        if(!/\.glb$/i.test(file?.name||'')){
          const response=await fetch(`/api/aero/models/${encodeURIComponent(modelId)}/source`,{cache:'no-store'});
          if(response.status!==404){
            if(!response.ok)throw new Error(`HTTP ${response.status}`);
            const asset=await response.json();
            const binary=atob(asset.data),bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
            file=new File([bytes],asset.name,{type:asset.media_type});
          }
        }
        if(state.modelId!==modelId||state.sourceAssetRequest!==request)return;
        state.originalFile=file||null;state.sourceAssetModelId=modelId;state.sourceAssetFile=file||null;
        await modelTools?.refreshSource(file||null,{retry:true});
        if(state.modelId===modelId){updateModelAppearance();queueRender();}
      }catch(error){
        if(state.modelId===modelId){state.modelToolStatus=`Original model appearance could not be restored: ${error.message}. Reopen the study to retry.`;updateModelAppearance();updateLabels();}
        console.warn('Original model appearance unavailable:',error);
      }finally{if(state.sourceAssetRequest===request)state.sourceAssetRequest=null;}
    })();
    return request.promise;
  }
  function updateModelAppearance(info) {
    const option=Array.from($('model-appearance')?.options||[]).find(o=>o.value==='glb');
    if(option)option.disabled=!/\.glb$/i.test(state.originalFile?.name||'');
    if(info)state.modelToolStatus=info.loading?'Loading original GLB materials and textures…':info.ready?'Original GLB materials and textures are available. Selecting GLB appearance switches to geometry; Surface displays computed fields.':info.error||'Original GLB appearance requires the original GLB file. Load it from File → Open geometry if it was not retained.';
  }
  function hydrateCase(caseInfo) {
    const set=(id,value)=>{if(value!==undefined&&value!==null)$(id).value=String(value);};
    const transform=caseInfo.transform||{},resolved=caseInfo.domain||{},domain=(resolved.sizing||'minimum')!=='fixed'?(caseInfo.requested_domain||caseInfo.domain_adjustment?.requested_domain||resolved):resolved,flow=caseInfo.flow||{};
    for(let i=0;i<3;i++){
      const axis=['x','y','z'][i];set('scale-'+axis,transform.scale?.[i]);
      set('rotate-'+axis,transform.rotation_deg?.[i]);set('move-'+axis,transform.translation?.[i]);
    }
    for(const key of ['upstream','downstream','lateral','height'])set(key,domain[key]);
    set('domain-x-offset',domain.x_offset_m??resolved.x_offset_m??0);
    set('domain-sizing',domain.sizing||'minimum');for(let axis=0;axis<3;axis++)set('domain-ratio-'+['x','y','z'][axis],domain.proportions?.[axis]??1);set('boundary',domain.boundary);set('symmetry-placement',symmetryPlacement(domain));set('symmetry-placement-offset-y',fixedSymmetryOffset(domain));if($('symmetry-auto-center-y'))$('symmetry-auto-center-y').checked=symmetryPlacement(domain)==='center';set('surface-dx',caseInfo.surface_dx);set('levels',caseInfo.levels);
    set('geometry-handling',caseInfo.geometry_handling||'closed_solid');
    set('surface-triangle-max-edge-mm',Number(caseInfo.surface_triangle_max_edge_m)>0?Number(caseInfo.surface_triangle_max_edge_m)*1000:50);
    const sized=caseInfo.target_vram_percent!=null||caseInfo.target_vram_gb!=null;
    set('mesh-sizing',caseInfo.mesh_sizing??(sized?'vram':'manual'));
    const free=Number(state.capabilities?.gpu?.free_gb),legacy=free>0?Math.min(100,Math.max(1,100*Number(caseInfo.target_vram_gb)/free)):80;
    set('target-vram',caseInfo.target_vram_percent??(sized?Math.round(legacy):80));$('target-vram').disabled=false;
    const octree=caseInfo.octree||{};set('octree-band-fine',octree.band_fine??8);set('octree-band',octree.band??6);set('octree-wake',octree.wake_length??1);set('octree-wake-offset',octree.wake_level_offset??1);set('octree-threads',octree.threads??0);if($('octree-seal-cavities'))$('octree-seal-cavities').checked=octree.seal_cavities===true;
    set('octree-u-lattice',octree.u_lattice??.08);set('octree-checkpoint-keep',octree.checkpoint_keep??2);set('octree-plane-samples',octree.max_plane_samples??4000000);set('octree-ramp-steps',octree.ramp_steps??'');
    set('lattice',caseInfo.lattice||'d3q27');set('solver',caseInfo.solver);syncSolverLattice();
    set('flow-alpha',flow.alpha||0);set('flow-beta',flow.beta||0);referenceTools?.restore(caseInfo.reference||{});syncFlowAngles();
    set('velocity',flow.velocity);set('density',flow.density);
    set('viscosity',flow.kinematic_viscosity);set('sound-speed',flow.speed_of_sound);
    set('flow-input-mode',flow.input_mode||'speed');set('flow-mach',flow.mach??flow.velocity/flow.speed_of_sound);
    set('flow-temperature',flow.temperature_k??288.15);set('flow-cfl',flow.cfl??.25);set('flow-wall-model',flow.wall_model||'slip');syncThermalFlow();
    set('steps',caseInfo.steps);set('frames',caseInfo.frames);
    state.cuts=Array.isArray(caseInfo.cuts)?caseInfo.cuts.map(c=>({axis:c.axis,position:c.position})):[];
    window.LudwigOutputControls?.restore(caseInfo);updateVramTarget();showGeometryAudit(caseInfo.geometry_audit);
    boundaryNote();renderCutRows();window.LudwigAeroLayout?.syncModelControls?.();
  }
  async function importModel(file) {
    if(!file)return;
    clearImportedResults(false);
    const ext=file.name.split('.').pop().toLowerCase();
    if(!['stl','obj','glb'].includes(ext)){message('Choose an STL, OBJ, or GLB model.',true);return;}
    state.importing=true;state.modelId=null;state.source=null;state.originalFile=file;state.sourceAssetModelId=null;state.studyResults=null;
    if($('study-results')){$('study-results').replaceChildren();$('study-results').hidden=true;}
    invalidateMesh();updateActions();
    message(`Loading ${file.name}…`);
    let imported=null;
    try {
      if(!state.scene)throw new Error('WebGL is unavailable; the model cannot be prepared.');
      imported=await BABYLON.SceneLoader.ImportMeshAsync(null,'file:',file,state.scene);
      const triangles=window.LudwigAeroModelTools.extractTriangles(imported.meshes),nTri=triangles.length/9;
      state.source=triangles;state.sourceBounds=boundsOf(triangles);if(state.camera)state.camera._aeroFit=false;
      $('metric-triangles').textContent=fmt(nTri);queueRender();
      state.importing=false;state.uploading=true;message(`Uploading ${fmt(nTri)} triangles…`);updateActions();
      const response=await api('/api/aero/models',{method:'POST',headers:{'Content-Type':'application/octet-stream','X-Model-Name':encodeURIComponent(file.name)},body:binaryStl(triangles)});
      state.modelId=response.model_id;
      state.sourceModelId=state.modelId;state.sourceAssetModelId=state.modelId;state.sourceAssetFile=file;
      if(file.size<=50_000_084)try{await api(`/api/aero/models/${state.modelId}/source?name=${encodeURIComponent(file.name)}`,{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:await file.arrayBuffer()});}catch(error){console.warn('Original asset could not be retained:',error);}
      showGeometryAudit(response.geometry_audit);
      message(`${file.name} · ${fmt(response.triangles||nTri)} triangles · ready for lattice build.`);
    }catch(error){state.modelId=null;message(`Could not load model: ${error.message}`,true);}
    finally{
      for(const mesh of imported?.meshes?.slice().reverse()||[])mesh.dispose(false,true);
      for(const node of imported?.transformNodes?.slice().reverse()||[])node.dispose();
      for(const group of imported?.animationGroups||[])group.dispose();
      for(const skeleton of imported?.skeletons||[])skeleton.dispose();
      for(const system of imported?.particleSystems||[])system.dispose();
      state.importing=false;state.uploading=false;updateActions();queueRender();
    }
  }

  function clearSceneObjects(retained=null) {
    for(const item of state.sceneObjects)if(!retained?.has(item))item.dispose?.();state.sceneObjects=[];state.labels=[];
    $('viewer-labels')?.replaceChildren();
  }
  function keep(item) {state.sceneObjects.push(item);return item;}
  function resultBoundary() {return state.amr?(state.activeConfig?.domain?.boundary||$('boundary').value):$('boundary').value;}
  function groundSolverFrame() {return resultBoundary()==='ground';}
  function v3(x,y,z) {return groundSolverFrame()?new BABYLON.Vector3(x,y,z):new BABYLON.Vector3(x,z,y);}
  function addTriangleSurface(triangles,values=null,range=null,offset=[0,0,0]) {
    const count=Math.floor(triangles.length/9),positions=new Float32Array(count*9),indices=new Uint32Array(count*3),
      normals=values?null:new Float32Array(count*9),colors=values?new Float32Array(count*12):null,ground=groundSolverFrame();
    let index=0;
    for(let tri=0;tri<count;tri++){
      const rgb=values?scalarColor(values[tri],range):null;
      for(let v=0;v<3;v++){
        const k=tri*9+v*3,x=triangles[k]+offset[0],y=triangles[k+1]+offset[1],z=triangles[k+2]+offset[2];
        positions[index*3]=x;positions[index*3+1]=ground?y:z;positions[index*3+2]=ground?z:y;indices[index]=index;
        if(rgb){const c=index*4;colors[c]=rgb[0];colors[c+1]=rgb[1];colors[c+2]=rgb[2];colors[c+3]=1;}
        index++;
      }
    }
    if(!indices.length)return null;
    if(normals)BABYLON.VertexData.ComputeNormals(positions,indices,normals);
    const mesh=keep(new BABYLON.Mesh('model',state.scene)),vertex=new BABYLON.VertexData();
    vertex.positions=positions;vertex.indices=indices;if(normals)vertex.normals=normals;
    if(values){vertex.colors=colors;mesh.useVertexColors=true;}
    vertex.applyToMesh(mesh);
    const material=keep(new BABYLON.StandardMaterial('model-material',state.scene));
    material.backFaceCulling=false;material.twoSidedLighting=true;
    material.diffuseColor=values?BABYLON.Color3.White():new BABYLON.Color3(.78,.84,.82);
    material.specularColor=new BABYLON.Color3(.18,.22,.24);
    // Scalar colours must stay readable on both sides, independent of the
    // model normals and scene lighting. A dark emissive multiplier previously
    // reduced the colour map to about one fifth of its intended brightness.
    if(values){material.emissiveColor=BABYLON.Color3.White();material.specularColor=BABYLON.Color3.Black();material.disableLighting=true;}
    mesh.material=material;return mesh;
  }
  function scalarColor(value,range) {
    if(!Number.isFinite(value))return [.45,.48,.52];
    if(window.LudwigPalettes)return window.LudwigPalettes.valueColor(Number(value),range.lo,range.hi,$('results-palette')?.value||window.LudwigPalettes.get());
    const t=Math.max(0,Math.min(1,(Number(value)-range.lo)/(range.hi-range.lo||1)))*(colorStops.length-1),
      i=Math.min(colorStops.length-2,Math.floor(t)),f=t-i;
    return colorStops[i].map((n,k)=>(n*(1-f)+colorStops[i+1][k]*f)/255);
  }
  function scalarRange(values,mode='contrast') {
    const sorted=[],stride=Math.max(1,Math.ceil(values.length/50000));let actualLo=Infinity,actualHi=-Infinity;
    for(let i=0;i<values.length;i++)if(Number.isFinite(values[i])){actualLo=Math.min(actualLo,values[i]);actualHi=Math.max(actualHi,values[i]);if(i%stride===0)sorted.push(values[i]);}
    if(!sorted.length&&Number.isFinite(actualLo))sorted.push(actualLo,actualHi);
    sorted.sort((a,b)=>a-b);
    if(!sorted.length)return null;
    const lo=sorted[Math.floor(.02*(sorted.length-1))],hi=sorted[Math.ceil(.98*(sorted.length-1))];
    return {lo:mode==='full'||lo===hi?actualLo:lo,hi:mode==='full'||lo===hi?actualHi:hi,actualLo,actualHi,mode};
  }
  function domainBounds(body) {
    const preview=previewFrame();if(preview)return preview.domain;
    const up=Number($('upstream').value)||2,down=Number($('downstream').value)||3,
      lat=Number($('lateral').value)||2,height=Number($('height').value)||2;
    return {min:[0,0,0],max:[up+down+1,resultBoundary()==='free_air'?2*lat+1:resultBoundary()==='ground'?height+1:lat+1,
      2*(resultBoundary()==='ground'?lat:height)+1]};
  }
  function addLabel(text,point,kind='boundary',color=null) {
    const label={text:String(text),point,kind,color};state.labels.push(label);
    if(kind.includes('aero-axis')){const holder=$('viewer-labels');if(holder){holder.hidden=false;const node=document.createElement('span');node.className='scene-label '+kind;node.textContent=label.text;if(color)node.style.color=color;holder.appendChild(node);label.node=node;}}
  }
  function projectAxisLabels(){
    if(!state.camera||!state.engine)return;const canvas=$('viewer'),width=canvas.clientWidth,height=canvas.clientHeight;if(!width||!height)return;
    const viewport=state.camera.viewport.toGlobal(width,height),transform=state.scene.getTransformMatrix();
    for(const label of state.labels)if(label.node){const p=BABYLON.Vector3.Project(v3(...label.point),BABYLON.Matrix.IdentityReadOnly,transform,viewport);label.node.hidden=p.z<0||p.z>1||p.x<0||p.x>width||p.y<0||p.y>height;label.node.style.left=p.x+'px';label.node.style.top=p.y+'px';}

  }
  function updateLabels() {
    const node=$('scene-info');if(!node)return;
    const frame=state.frame||{},lines=['Boundary colours: inlet = translucent yellow; outlet = blue; lower face = green.'];
    if(state.viewNotice)lines.unshift(state.viewNotice);
    if(resultBoundary()==='free_air')lines.push((state.amr?state.activeConfig?.solver:$('solver').value)==='pond_thermal'?'Green lower face is a visual floor reference; its kinetic far-field boundary remains open.':'Green lower face is a visual floor reference; the free-air slip boundary is unchanged.');
    lines.push(`Solver geometry: ${$('geometry-handling')?.value==='two_sided_open'?'Two-sided / intentional open surface':'Closed solid · verify topology'}. Wall intersections use both triangle sides; coincident backfaces are not duplicated.`);
    if(state.modelToolStatus)lines.push(state.modelToolStatus);
    if(frame.domain)lines.push(`Domain minimum [m]: ${frame.domain.min.map(fmt).join(', ')}\nDomain maximum [m]: ${frame.domain.max.map(fmt).join(', ')}`);
    try{const t=transformSettings();lines.push(`Model scale: ${t.scale.map(fmt).join(', ')}\nRotation X/Y/Z [degrees]: ${t.rotation_deg.map(fmt).join(', ')}\nOffset X/Y/Z [m]: ${t.translation.map(fmt).join(', ')}`);}catch{}
    for(const label of state.labels)lines.push(label.text);
    for(const warning of state.geometryAudit?.warnings||[])lines.push(warning);
    const text=Array.from(new Set(lines)).join('\n\n');if(node.textContent!==text)node.textContent=text;
  }
  function aidsMode() {return $('display-aids')?.value||'all';}
  function vectorAdd(a,b,k=1) {return a.map((v,i)=>v+k*b[i]);}
  function arrowSegments(start,end) {
    const direction=end.map((v,i)=>v-start[i]),length=Math.hypot(...direction);if(!(length>0))return;
    const unit=direction.map(v=>v/length),side=Math.abs(unit[2])<.8?[-unit[1],unit[0],0]:[0,-unit[2],unit[1]],norm=Math.hypot(...side);
    const wing=side.map(v=>v/norm*length*.075),back=vectorAdd(end,unit,-length*.18);
    return [[start,end],[end,vectorAdd(back,wing)],[end,vectorAdd(back,wing,-1)]];
  }
  function addArrow(name,start,end,rgb,kind=null,label=null) {
    const pairs=arrowSegments(start,end);if(!pairs)return;
    const direction=end.map((v,i)=>v-start[i]),length=Math.hypot(...direction),unit=direction.map(v=>v/length);
    addLines(name,pairs,new BABYLON.Color3(...rgb));
    if(label)addLabel(label,vectorAdd(end,unit,length*.14),kind||'axis',`rgb(${rgb.map(v=>Math.round(v*255)).join(',')})`);
  }
  function addLines(name,pairs,color) {
    const lines=keep(BABYLON.MeshBuilder.CreateLineSystem(name,{lines:pairs.map(([a,b])=>[v3(...a),v3(...b)])},state.scene));
    lines.color=color;lines.isPickable=false;return lines;
  }
  function drawDomain(b) {
    const a=b.min,c=b.max,points=[[a[0],a[1],a[2]],[c[0],a[1],a[2]],[c[0],c[1],a[2]],[a[0],c[1],a[2]],
      [a[0],a[1],c[2]],[c[0],a[1],c[2]],[c[0],c[1],c[2]],[a[0],c[1],c[2]]],
      edges=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    addLines('domain',edges.map(([x,y])=>[points[x],points[y]]),new BABYLON.Color3(.45,.69,.77));
    const boundary=resultBoundary(),mode=aidsMode(),showBoundaries=mode==='all'||mode==='boundaries',mid=a.map((v,i)=>(v+c[i])/2),size=c.map((v,i)=>v-a[i]);
    if(showBoundaries){
      addQuad('inlet boundary',[points[0],points[3],points[7],points[4]],new BABYLON.Color3(1,.94,.59),.16);
      addQuad('outlet boundary',[points[1],points[2],points[6],points[5]],new BABYLON.Color3(.54,.73,1),.16);
      const floor=boundary==='ground'&&groundSolverFrame()?[points[0],points[1],points[5],points[4]]:[points[0],points[1],points[2],points[3]];
      addQuad('ground boundary',floor,new BABYLON.Color3(.24,.75,.39),boundary==='ground'?.21:.15);
      if(boundary==='symmetry')addQuad('symmetry boundary',[points[0],points[1],points[5],points[4]],new BABYLON.Color3(.20,.49,.95),.17);
    }
    {
      const velocity=Number((state.amr?state.activeConfig?.flow?.velocity:null)??$('velocity').value);
      const thermal=(state.amr?state.activeConfig?.solver:$('solver').value)==='pond_thermal';
      const wall=(state.amr?state.activeConfig?.flow?.wall_model:$('flow-wall-model')?.value)||'slip';
      if(thermal){
        const farfield=`Characteristic far field · freestream at ${fmt(velocity)} m/s\nIncoming characteristics use freestream; outgoing use interior state`;
        addLabel(`X− · inlet at x = ${fmt(a[0])} m\n${farfield}`,[a[0],mid[1],mid[2]],'boundary inlet');
        addLabel(`X+ · outlet at x = ${fmt(c[0])} m\n${farfield}`,[c[0],mid[1],mid[2]],'boundary outlet');
        addLabel(boundary==='ground'?`Y− · ground at y = ${fmt(a[1])} m\n${wall==='slip'?'Inviscid slip · zero normal velocity':'Stationary laminar wall · no slip'}`:boundary==='symmetry'?`Y− · symmetry at y = ${fmt(a[1])} m\nSlip / mirror · zero normal velocity`:`Y− · lateral boundary\n${farfield}`,[mid[0],a[1],mid[2]],'boundary '+(boundary==='ground'?'ground':boundary==='symmetry'?'symmetry':''));
        addLabel(`Y+ · outer boundary\n${farfield}`,[mid[0],c[1],mid[2]],'boundary');
        addLabel(`Z− · bottom boundary\n${farfield}`,[mid[0],mid[1],a[2]],'boundary');
        addLabel(`Z+ · top boundary\n${farfield}`,[mid[0],mid[1],c[2]],'boundary');
      }else{
      addLabel(`X− · inlet at x = ${fmt(a[0])} m\nPrescribed +X velocity ${fmt(velocity)} m/s · local density`,[a[0],mid[1],mid[2]],'boundary inlet');
      addLabel(`X+ · outlet at x = ${fmt(c[0])} m\nExtrapolated outflow · pressure is not fixed`,[c[0],mid[1],mid[2]],'boundary outlet');
      addLabel(boundary==='ground'?'Y− · ground at y = 0\nStationary wall · no slip (u = v = w = 0)':boundary==='symmetry'?'Y− · symmetry at y = 0\nSlip / mirror · zero normal velocity':'Y− · lateral far field\nFreestream velocity condition',
        [mid[0],a[1],mid[2]],'boundary '+(boundary==='ground'?'ground':boundary==='symmetry'?'symmetry':''));
      addLabel(boundary==='free_air'?'Y+ · lateral far field\nFreestream velocity condition':boundary==='ground'?'Y+ · upper boundary\nSlip / mirror · zero normal velocity':'Y+ · outer boundary\nSlip / mirror · zero normal velocity',
        [mid[0],c[1],mid[2]],'boundary');
      addLabel(boundary==='ground'?'Z− · lateral boundary\nSlip / mirror · zero normal velocity':'Z− · bottom boundary\nSlip / mirror · zero normal velocity',[mid[0],mid[1],a[2]],'boundary');
      addLabel(boundary==='ground'?'Z+ · lateral boundary\nSlip / mirror · zero normal velocity':'Z+ · top boundary\nSlip / mirror · zero normal velocity',[mid[0],mid[1],c[2]],'boundary');
      }
    }
    if(mode==='all'){
      const up=groundSolverFrame()?1:2,lateral=groundSolverFrame()?2:1,grid=[];
      for(let i=1;i<10;i++){
        const p=a.slice(),q=a.slice();p[0]=q[0]=a[0]+size[0]*i/10;q[lateral]=c[lateral];grid.push([p,q]);
        const r=a.slice(),s=a.slice();r[lateral]=s[lateral]=a[lateral]+size[lateral]*i/10;s[0]=c[0];grid.push([r,s]);
      }
      addLines('domain floor grid',grid,new BABYLON.Color3(.21,.37,.43));
      const names=['X','Y','Z'];
      for(let axis=0;axis<3;axis++){
        const end=a.slice();end[axis]=c[axis];
        addLabel(`L${names[axis].toLowerCase()} = ${Number(size[axis]).toLocaleString(undefined,{maximumSignificantDigits:4})} m`,
          a.map((v,i)=>i===axis?(v+c[i])/2:v),'dimension');
      }
    }
    for(const cut of state.amr?(state.activeConfig?.cuts||[]):state.cuts){const p=Number(cut.position);if(!Number.isFinite(p))continue;
      const pts=cut.axis==='x'?[[p,a[1],a[2]],[p,c[1],a[2]],[p,c[1],c[2]],[p,a[1],c[2]]]:
        cut.axis==='y'?[[a[0],p,a[2]],[c[0],p,a[2]],[c[0],p,c[2]],[a[0],p,c[2]]]:
        [[a[0],a[1],p],[c[0],a[1],p],[c[0],c[1],p],[a[0],c[1],p]];
      addQuad('cut '+cut.axis,pts,new BABYLON.Color3(.28,.82,.64),.13);
      if(showBoundaries)addLabel(`Cut ${cut.axis.toUpperCase()} = ${fmt(p)} m`,pts[2],'cut');
    }
    const sizePick=b.max.map((value,axis)=>value-b.min[axis]),midPick=b.min.map((value,axis)=>(value+b.max[axis])/2);
    const domainPick=keep(BABYLON.MeshBuilder.CreateBox('domain-pick-surface',{width:sizePick[0],height:sizePick[groundSolverFrame()?1:2],depth:sizePick[groundSolverFrame()?2:1]},state.scene));
    domainPick.position.copyFrom(v3(...midPick));domainPick.visibility=0;domainPick.isVisible=true;domainPick.isPickable=true;domainPick.metadata={domainSurface:true};
    return b;
  }
  function drawAxes(frame) {
    const mode=aidsMode();if(mode!=='all'&&mode!=='axes')return;
    const b=frame.domain,size=b.max.map((v,i)=>v-b.min[i]),short=Math.min(...size),length=short*.17;
    const origin=b.min.map(v=>v+short*.045),colors=[[.98,.39,.36],[.40,.86,.55],[.40,.67,1]],names=['X','Y','Z'];
    for(let i=0;i<3;i++){const end=origin.slice();end[i]+=length;addArrow('domain '+names[i],origin,end,colors[i],'axis domain-axis','Domain '+names[i]);}
    if(referenceTools||!frame.body)return;
    const t=rotationMath(frame.config?.transform||{scale:[1,1,1],rotation_deg:[0,0,0],translation:[0,0,0]});
    t.s=[1,1,1];t.d=[0,0,0];
    const bodyLength=Math.max(short*.065,Math.max(...frame.body.max.map((v,i)=>v-frame.body.min[i]))*.48),bodyOrigin=frame.bodyOrigin||frame.body.center;
    for(let i=0;i<3;i++){
      const unit=[0,0,0];unit[i]=1;let direction=transformedPoint(...unit,t,[0,0,0]);
      if(frame.ground)direction=[direction[0],direction[2],-direction[1]];
      addArrow('body '+names[i],bodyOrigin,vectorAdd(bodyOrigin,direction,bodyLength),colors[i],'axis body-axis','Body '+names[i].toLowerCase());
    }
  }
  function drawFlow(frame) {
    const mode=$('velocity-vectors')?.value||'inlet';if(mode!=='inlet'&&mode!=='both')return;
    const b=frame.domain,size=b.max.map((v,i)=>v-b.min[i]),flow=state.amr?state.activeConfig?.flow:null,
      velocity=Number(flow?.velocity??$('velocity').value),length=Math.min(size[0]*.20,Math.min(size[1],size[2])*.40),mid=b.min.map((v,i)=>(v+b.max[i])/2);
    for(const fraction of [.24,.50,.76]){
      const q=[b.min[0],b.min[1]+size[1]*fraction,mid[2]],p=vectorAdd(q,[1,0,0],-length);
      addArrow('prescribed inlet velocity',p,q,[.36,.90,.82]);
      if(fraction===.5)addLabel(`U∞ = ${fmt(velocity)} m/s · prescribed`,vectorAdd(p,[1,0,0],length*.25),'velocity');
    }
  }
  async function refreshRecordedVectors(job=state.job,retry=false) {
    const mode=$('velocity-vectors')?.value||'inlet';if(mode!=='recorded'&&mode!=='both')return;
    const cuts=Array.isArray(job?.artifacts?.cuts)?job.artifacts.cuts:[],cut=cuts[Number($('cut-result-select').value)],url=apiUrl(cut?.json_url);
    if(!url){state.vectorKey=null;state.vectorData=null;state.vectorLoading=false;state.vectorError='No recorded cut velocity is available yet. Choose a recorded plane after the solver exports data.';return;}
    const key=`${url}:${cut.step}`;if(state.vectorKey===key&&(!retry||state.vectorLoading||!state.vectorError))return;
    state.vectorKey=key;state.vectorData=null;state.vectorLoading=true;state.vectorError='';
    try{
      const separator=url.includes('?')?'&':'?',parts=await Promise.all(['u','v','w'].map(field=>api(`${url}${separator}field=${field}`)));
      const first=parts[0],n1=Number(first.n1),n2=Number(first.n2);
      if(!Number.isInteger(n1)||!Number.isInteger(n2)||n1<2||n2<2||n1*n2>500_000||
          parts.some(p=>p.n1!==first.n1||p.n2!==first.n2||p.normal_axis!==first.normal_axis||
            !Array.isArray(p.values)||p.values.length!==n1*n2||p.units!=='m/s'))throw new Error('The selected cut has no compatible physical velocity components.');
      const spacing=Number(first.spacing_m),origin=first.origin_m,normal=Number(first.normal_coord_m),axes=['x','y','z'];
      const axis1=axes.indexOf(first.axis1),axis2=axes.indexOf(first.axis2),normalAxis=axes.indexOf(first.normal_axis);
      if(!(spacing>0)||!Array.isArray(origin)||origin.length!==2||!origin.every(finite)||!Number.isFinite(normal)||
          new Set([axis1,axis2,normalAxis]).size!==3||Math.min(axis1,axis2,normalAxis)<0)throw new Error('The selected cut has invalid coordinates.');
      const stride1=Math.max(1,Math.ceil(n1/12)),stride2=Math.max(1,Math.ceil(n2/12)),vectors=[];
      for(let j=Math.floor(stride2/2);j<n2;j+=stride2)for(let i=Math.floor(stride1/2);i<n1;i+=stride1){
        const index=i+j*n1,velocity=parts.map(p=>p.values[index]);if(!velocity.every(Number.isFinite))continue;
        const speed=Math.hypot(...velocity);if(!(speed>1e-8))continue;
        const point=[0,0,0];point[normalAxis]=normal;point[axis1]=Number(origin[0])+(i+.5)*spacing;point[axis2]=Number(origin[1])+(j+.5)*spacing;
        vectors.push({point,velocity,speed});
      }
      if(!vectors.length)throw new Error('The selected cut has no finite nonzero velocity vectors.');
      if(state.vectorKey===key)state.vectorData={vectors,step:first.step,maxSpeed:Math.max(...vectors.map(v=>v.speed)),axis:first.normal_axis,position:normal};
    }catch(error){if(state.vectorKey===key)state.vectorError=error.message;}
    finally{if(state.vectorKey===key){state.vectorLoading=false;queueRender();}}
  }
  function drawRecordedVectors(frame) {
    const mode=$('velocity-vectors')?.value||'inlet',note=$('vector-note');
    if(mode==='none'){if(note)note.textContent='Velocity vectors hidden.';return;}
    if(mode==='inlet'){if(note)note.textContent='Teal arrows show the prescribed inlet velocity; they are not computed flow vectors.';return;}
    const data=state.vectorData;
    if(!data){if(note)note.textContent=state.vectorLoading?'Loading computed velocity on the selected cut…':state.vectorError||'Choose a recorded cut plane to display computed velocity vectors.';return;}
    const a=frame.domain.min,c=frame.domain.max,p=data.position;
    const plane=data.axis==='x'?[[p,a[1],a[2]],[p,c[1],a[2]],[p,c[1],c[2]],[p,a[1],c[2]]]:
      data.axis==='y'?[[a[0],p,a[2]],[c[0],p,a[2]],[c[0],p,c[2]],[a[0],p,c[2]]]:
      [[a[0],a[1],p],[c[0],a[1],p],[c[0],c[1],p],[a[0],c[1],p]];
    addQuad('recorded velocity plane',plane,new BABYLON.Color3(1,.75,.30),.045);
    addLines('recorded velocity plane border',plane.map((point,i)=>[point,plane[(i+1)%4]]),new BABYLON.Color3(.83,.63,.29));
    addLabel(`Velocity cut ${data.axis.toUpperCase()} = ${fmt(data.position)} m · step ${fmt(data.step)}`,plane[2],'cut');
    const size=frame.domain.max.map((v,i)=>v-frame.domain.min[i]),scale=Math.min(...size)*.095/data.maxSpeed,pairs=[];
    for(const vector of data.vectors){
      if(vector.point.some((v,i)=>v<frame.domain.min[i]||v>frame.domain.max[i]))continue;
      const segments=arrowSegments(vector.point,vectorAdd(vector.point,vector.velocity,scale));if(segments)pairs.push(...segments);
    }
    if(pairs.length)addLines('recorded cut velocity',pairs,new BABYLON.Color3(1,.75,.30));
    if(note)note.textContent=`Gold arrows: computed velocity on ${data.axis.toUpperCase()} = ${fmt(data.position)} m, step ${fmt(data.step)} · ${pairs.length/3} samples · sample max ${fmt(data.maxSpeed)} m/s. Glyph lengths are scaled for display.${mode==='both'?' Teal arrows: prescribed inlet.':''}`;
  }
  function addQuad(name,points,color,alpha) {
    const mesh=keep(new BABYLON.Mesh(name,state.scene)),vertex=new BABYLON.VertexData(),pos=points.flatMap(p=>{const v=v3(...p);return [v.x,v.y,v.z]});
    vertex.positions=pos;vertex.indices=[0,1,2,0,2,3];vertex.applyToMesh(mesh);
    const material=keep(new BABYLON.StandardMaterial(name+'-material',state.scene));
    material.diffuseColor=color;material.emissiveColor=color.scale(.4);material.alpha=alpha;material.backFaceCulling=false;
    material.disableDepthWrite=alpha<1;
    mesh.material=material;mesh.isPickable=false;return mesh;
  }
  function drawAmr() {
    if(inspector)return inspector.drawAmr();
    const levels=(state.importedResults?state.importedResults.amr:state.amr)?.levels;if(!Array.isArray(levels))return;
    const palette=[[.35,.71,.89],[.51,.79,.59],[.92,.72,.37],[.92,.43,.39]];
    for(let level=0;level<levels.length;level++){
      const lv=levels[level],blocks=lv.blocks||[],n=Math.min(Number(lv.n_blocks)||Math.floor(blocks.length/3),Math.floor(blocks.length/3));
      if(!n)continue;const size=Number(lv.block_phys_size);if(!(size>0))continue;
      const stride=Math.max(1,Math.ceil(n/4500)),count=Math.ceil(n/stride);
      const box=keep(BABYLON.MeshBuilder.CreateBox('AMR level '+(lv.level_id||level+1),{size:size*.985},state.scene));
      const m=keep(new BABYLON.StandardMaterial('AMR-'+level,state.scene)),rgb=palette[level%palette.length];
      m.diffuseColor=new BABYLON.Color3(...rgb);m.emissiveColor=m.diffuseColor.scale(.35);m.alpha=.24;m.wireframe=true;
      box.material=m;box.isPickable=false;
      const matrices=new Float32Array(count*16),matrix=BABYLON.Matrix.Identity();let j=0;
      for(let i=0;i<n;i+=stride){const p=v3((blocks[i*3]-.5)*size,(blocks[i*3+1]-.5)*size,(blocks[i*3+2]-.5)*size);
        BABYLON.Matrix.TranslationToRef(p.x,p.y,p.z,matrix);matrix.copyToArray(matrices,j++*16);}
      box.thinInstanceSetBuffer('matrix',matrices,16,true);
    }
  }
  function fit(bounds) {
    if(!bounds||!state.camera)return;
    const c=bounds.min.map((v,i)=>(v+bounds.max[i])/2),size=bounds.max.map((v,i)=>v-bounds.min[i]);
    state.camera.setTarget(v3(...c));state.camera.radius=Math.max(.02,Math.hypot(...size)*1.05);
    state.camera.minZ=Math.max(.00001,state.camera.radius/1000);state.camera.maxZ=state.camera.radius*1000;
  }
  function cameraPreset(name) {
    if(!state.camera)return;
    const presets={perspective:[-Math.PI/2-.55,1.10],top:[-Math.PI/2,.005],front:[Math.PI,Math.PI/2],side:[-Math.PI/2,Math.PI/2]};
    const angles=presets[name]||presets.perspective;state.camera.alpha=angles[0];state.camera.beta=angles[1];
  }
  function clearCameraMotion(camera=state.camera) {
    if(!camera)return;
    for(const name of ['inertialAlphaOffset','inertialBetaOffset','inertialRadiusOffset','inertialPanningX','inertialPanningY'])camera[name]=0;
    const movement=camera.movement;
    if(movement){
      for(const name of ['zoomAccumulatedPixels','zoomDeltaCurrentFrame','_zoomVelocity'])movement[name]=0;
      for(const name of ['panAccumulatedPixels','rotationAccumulatedPixels','panDeltaCurrentFrame','rotationDeltaCurrentFrame','_panVelocity','_rotationVelocity'])movement[name]?.setAll?.(0);
      movement.activeInput=false;
    }
  }
  function focusPickedPoint(point,camera=state.camera) {
    if(!point||!camera)return;
    clearCameraMotion(camera);const eye=camera.position.clone();
    camera.setTarget(point.clone(),false,true);camera.setPosition(eye);camera.getViewMatrix(true);
  }
  function rayOnViewPlane(x,y,plane,scene=state.scene,camera=state.camera) {
    if(!scene||!camera)return null;
    const ray=scene.createPickingRay(x,y,BABYLON.Matrix.Identity(),camera),distance=ray.intersectsPlane(plane);
    return distance===null||!Number.isFinite(distance)||distance<0?null:ray.origin.add(ray.direction.scale(distance));
  }
  function panPinnedPoint(x,y,plane,anchor,scene=state.scene,camera=state.camera) {
    const underCursor=rayOnViewPlane(x,y,plane,scene,camera);if(!underCursor)return false;
    const delta=anchor.subtract(underCursor),eye=camera.position.add(delta),target=camera.target.add(delta);
    clearCameraMotion(camera);camera.setTarget(target,false,true);camera.setPosition(eye);camera.getViewMatrix(true);
    return true;
  }
  function displayedDomainBounds(frame=currentFrame()) {
    const min=frame.domain.min.slice(),max=frame.domain.max.slice(),size=max.map((v,i)=>v-min[i]);
    if(frame.body)for(let i=0;i<3;i++){min[i]=Math.min(min[i],frame.body.min[i]);max[i]=Math.max(max[i],frame.body.max[i]);}
    const vectors=$('velocity-vectors')?.value||'inlet';
    if(vectors==='inlet'||vectors==='both')min[0]-=Math.min(size[0]*.20,Math.min(size[1],size[2])*.40)*1.15;
    for(let i=0;i<3;i++){min[i]-=size[i]*.04;max[i]+=size[i]*.04;}
    return {min,max};
  }
  function fitDisplayedDomain() {
    if(!state.camera)return;
    const camera=state.camera,alpha=camera.alpha,beta=camera.beta;clearCameraMotion(camera);
    fit(displayedDomainBounds());camera.alpha=alpha;camera.beta=beta;camera._aeroFit=true;camera.getViewMatrix(true);
  }
  function fitNativeReport(surface){
    const bounds=surface?.geometry?.bounds||surface?.bounds,camera=state.camera;if(!bounds||!camera)return;
    const size=bounds.max.map((value,axis)=>value-bounds.min[axis]),centre=bounds.min.map((value,axis)=>(value+bounds.max[axis])/2),span=Math.max(.01,...size),diagonal=Math.max(.01,Math.hypot(...size));
    clearCameraMotion(camera);state.pendingStudyCamera=null;state.domainCameraAnchor=undefined;
    camera.setTarget(v3(...centre));camera.alpha=.95;camera.beta=1.1;camera.radius=1.6*span;
    // This dedicated view contains the body only. A tight depth interval avoids
    // the six-order near/far ratio used while inspecting the full fluid domain.
    camera.minZ=Math.max(.001,camera.radius-.65*diagonal);camera.maxZ=camera.radius+.8*diagonal;
    camera._aeroFit=true;camera.getViewMatrix(true);
  }
  function installViewerGestures(canvas,scene,camera) {
    let left=null,pan=null;
    const coordinates=event=>{const rect=canvas.getBoundingClientRect();return {x:(event.clientX-rect.left)*canvas.clientWidth/rect.width,y:(event.clientY-rect.top)*canvas.clientHeight/rect.height};};
    const isModel=mesh=>(mesh.name==='model'||modelTools?.isModelMesh(mesh))&&mesh.isPickable&&mesh.isEnabled()&&mesh.isVisible;
    const pickDomain=point=>scene.pick(point.x,point.y,mesh=>mesh.isPickable&&mesh.metadata?.domainSurface,false,camera);
    const pickModel=point=>{
      const canonical=state.sceneObjects.find(mesh=>mesh.name==='model'&&mesh.isPickable&&!mesh.isDisposed?.());
      if(!canonical)return scene.pick(point.x,point.y,isModel,false,camera);
      canonical.computeWorldMatrix(true);const world=canonical.getWorldMatrix(),material=canonical.material;
      return scene.pick(point.x,point.y,mesh=>mesh===canonical,false,camera,(a,b,c,ray)=>{
        const hit=ray.intersectsTriangle(a,b,c);if(!hit)return false;
        const local=ray.origin.add(ray.direction.scale(hit.distance)),point=BABYLON.Vector3.TransformCoordinates(local,world);
        return [material?.clipPlane,material?.clipPlane2].every(plane=>!plane||plane.dotCoordinate(point)<=1e-7);
      });
    };
    const stop=event=>{event.preventDefault();event.stopImmediatePropagation();};
    const finishPan=event=>{
      if(!pan||event.pointerId!==pan.id)return;
      pan=null;clearCameraMotion(camera);try{canvas.releasePointerCapture(event.pointerId);}catch{}stop(event);
    };
    canvas.addEventListener('contextmenu',event=>event.preventDefault());
    canvas.addEventListener('pointerdown',event=>{
      const p=coordinates(event);
      if(modelTools?.ownsPointer(p.x,p.y)){left=null;return;}
      if(event.button===0){const hit=pickModel(p),tool=scene.pick(p.x,p.y,mesh=>mesh.isPickable&&mesh.metadata?.sectionTool,false,camera);
        if(tool?.hit&&(!hit?.hit||tool.distance<=hit.distance+1e-7)){left=null;return;}
        const chosen=hit?.hit?hit:pickDomain(p);left={id:event.pointerId,x:event.clientX,y:event.clientY,point:chosen?.hit?chosen.pickedPoint?.clone():null};return;}
      if(event.button!==2)return;
      stop(event);clearCameraMotion(camera);camera.getViewMatrix(true);
      const hit=pickModel(p),depthPoint=hit?.hit&&hit.pickedPoint?hit.pickedPoint:camera.target;
      const plane=BABYLON.Plane.FromPositionAndNormal(depthPoint,camera.getForwardRay().direction),anchor=rayOnViewPlane(p.x,p.y,plane,scene,camera);
      if(anchor){pan={id:event.pointerId,plane,anchor};try{canvas.setPointerCapture(event.pointerId);}catch{}}
    },true);
    canvas.addEventListener('pointermove',event=>{
      if(!pan||pan.id!==event.pointerId)return;
      stop(event);const p=coordinates(event);panPinnedPoint(p.x,p.y,pan.plane,pan.anchor,scene,camera);
    },true);
    canvas.addEventListener('pointerup',event=>{
      if(pan&&pan.id===event.pointerId){finishPan(event);return;}
      if(event.button===0&&left?.id===event.pointerId){
        if(left.point&&Math.hypot(event.clientX-left.x,event.clientY-left.y)<=4)focusPickedPoint(left.point,camera);
        left=null;
      }
    },true);
    canvas.addEventListener('pointercancel',event=>{left=null;finishPan(event);},true);
    canvas.addEventListener('lostpointercapture',event=>{if(pan?.id===event.pointerId){pan=null;clearCameraMotion(camera);}},true);
    canvas.addEventListener('dblclick',event=>{
      if(event.button!==0)return;
      const hit=pickModel(coordinates(event));if(!hit?.hit){stop(event);fitDisplayedDomain();}
    },true);
  }
  function symmetrySourceOffset(config){
    if(config?.domain?.boundary!=='symmetry'||!state.source||!state.sourceBounds)return 0;
    if(symmetryPlacement(config.domain)==='none')return fixedSymmetryOffset(config.domain)+Number(config.transform?.translation?.[1]||0);
    const key=JSON.stringify([config.transform?.scale,config.transform?.rotation_deg,config.flow?.alpha,config.flow?.beta,symmetryPlacement(config.domain)]);
    if(state.symmetrySourceCache?.source===state.source&&state.symmetrySourceCache.key===key)return state.symmetrySourceCache.offset;
    const t=rotationMath(config.transform||transformSettings(),config.flow);t.d=[0,0,0];let lo=Infinity,hi=-Infinity;
    for(let i=0;i<state.source.length;i+=3){const y=transformedPoint(state.source[i],state.source[i+1],state.source[i+2],t,state.sourceBounds.center)[1];lo=Math.min(lo,y);hi=Math.max(hi,y);}
    const offset=symmetryPlacement(config.domain)==='center'?-(lo+hi)/2:-lo;
    state.symmetrySourceCache={source:state.source,key,offset};return offset;
  }
  function currentFrame() {
    if(state.importedResults){
      const dataset=state.importedResults,surface=state.importedSurface,offset=surface?.surface_offset||surface?.offset||dataset.surface_offset||[0,0,0];
      let triangles=null;if(surface?.vertices?.length>=9){
        if(!surface.domainVertices)surface.domainVertices=Float32Array.from(surface.vertices,(v,i)=>Number(v)+Number(offset[i%3]||0));
        triangles=surface.domainVertices;
      }else triangles=dataset.amr?.geometry?.vertices||state.amr?.geometry?.vertices||previewFrame()?.triangles||null;
      const bounds=dataset.domain_bounds,pointBounds=surface?.representation==='wall_samples'?surface.bounds:null,body=triangles?boundsOf(triangles):pointBounds?{...pointBounds,center:pointBounds.min.map((v,i)=>(v+pointBounds.max[i])/2)}:null;
      const domain=bounds&&Array.isArray(bounds.min)&&Array.isArray(bounds.max)?bounds:body||{min:[0,0,0],max:[1,1,1]};
      return {triangles,body,domain,bodyOrigin:[0,0,0],boundary:resultBoundary(),ground:groundSolverFrame(),config:dataset.case||state.activeConfig,imported:true};
    }
    if(state.draftDirty&&state.mode==='geometry'&&state.source)return previewFrame();
    if(Array.isArray(state.amr?.geometry?.vertices)&&state.amr.geometry.vertices.length>=9){
      const triangles=state.amr.geometry.vertices;
      return {triangles,body:boundsOf(triangles),domain:{min:[0,0,0],max:state.amr.domain_size.map(Number)},
        bodyOrigin:state.amr.mesh_offset.map((value,axis)=>value+(axis===1?symmetrySourceOffset(state.activeConfig):0)),boundary:resultBoundary(),ground:groundSolverFrame(),config:state.activeConfig,solverFrame:true};
    }
    return previewFrame()||{triangles:null,body:null,domain:domainBounds(null),boundary:resultBoundary(),ground:groundSolverFrame(),config:null};
  }
  function renderScene() {
    state.renderQueued=false;if(!state.scene||inspector?.isDragging()||modelTools?.isInteracting())return;
    let frame;
    try{frame=currentFrame();state.frame=frame;}catch(error){state.viewNotice=error.message;progress(`View error: ${error.message}`,'View error');$('viewer-overlay').hidden=true;updateLabels();return;}
    if(state.domainCameraAnchor!==undefined&&state.camera&&frame.body){
      // Keep the aircraft stationary on screen while the domain moves relative
      // to it. Mesh/field coordinates remain the native solver coordinates.
      const dx=frame.body.center[0]-state.domainCameraAnchor,eye=state.camera.position.clone(),target=state.camera.target.clone();
      state.domainCameraAnchor=undefined;eye.x+=dx;target.x+=dx;clearCameraMotion();state.camera.setTarget(target,false,true);state.camera.setPosition(eye);state.camera._aeroFit=true;
    }
    const source=frame.triangles,bodyBounds=frame.body;
    const imported=state.importedResults,field=imported?$('results-field').value:$('surface-field').value,surface=imported?state.importedSurface:state.surface;
    const values=surface?.fields?.[field]??surface?.[field],offset=imported?(surface?.surface_offset||surface?.offset||imported.surface_offset||[0,0,0]):surface?.surface_offset||state.amr?.mesh_offset;
    const hasPlacement=Array.isArray(offset)&&offset.length===3&&offset.every(Number.isFinite),rangeMode=$('surface-range-mode')?.value||'contrast',palette=$('results-palette')?.value||'spectrum';let range=null;
    let drawnSurface=null;
    const wallSamples=surface?.representation==='wall_samples',geometryIdentity=wallSamples?surface.geometry:surface?.vertices;
    const cached=state.scalarMeshCache,wantsScalar=state.mode==='surface'&&values&&hasPlacement&&(wallSamples?surface.geometry?.count>0:surface?.vertices?.length>=9);
    if(wantsScalar)range=referenceTools?.range(scalarRange(values,rangeMode))||scalarRange(values,rangeMode);
    const reuse=Boolean(wantsScalar&&range&&cached&&cached.scene===state.scene&&(!cached.mesh||!cached.mesh.isDisposed())&&cached.vertices===geometryIdentity&&cached.values===values&&cached.rangeMode===rangeMode&&cached.field===field&&cached.palette===palette&&cached.range.lo===range.lo&&cached.range.hi===range.hi&&cached.ground===groundSolverFrame()&&cached.offset.every((value,axis)=>value===offset[axis]));
    // Domain helpers and section fields refresh independently of the surface.
    // Keep immutable scalar buffers on the GPU while only those overlays change.
    clearSceneObjects(reuse?new Set(cached.objects||[cached.mesh,cached.material]):null);
    if(reuse){for(const object of cached.objects||[cached.mesh,cached.material])if(object)keep(object);drawnSurface=cached.mesh;}
    else if(wantsScalar&&range&&wallSamples){
      const start=state.sceneObjects.length;
      // The native link patches are the entire field surface. Drawing the
      // original STL beneath them makes the display look facet-averaged and
      // can hide fine variation where the two surfaces intersect in depth.
      const samples=window.LudwigAeroWallSurface.render(surface,values,range,{B:BABYLON,scene:state.scene,keep,color:scalarColor,ground:groundSolverFrame(),opacity:Number($('model-opacity')?.value??100)/100});
      state.scalarMeshCache={scene:state.scene,mesh:drawnSurface,material:drawnSurface?.material,objects:state.sceneObjects.slice(start),samples,vertices:geometryIdentity,values,field,range,rangeMode,palette,offset:offset.slice(),ground:groundSolverFrame()};
    }
    else if(wantsScalar&&range){
      drawnSurface=addTriangleSurface(surface.vertices,values,range,offset);
      state.scalarMeshCache={scene:state.scene,mesh:drawnSurface,material:drawnSurface.material,vertices:surface.vertices,values,field,range,rangeMode,palette,offset:offset.slice(),ground:groundSolverFrame()};
    }
    else if(wantsScalar&&source)drawnSurface=addTriangleSurface(source);
    else if(source)drawnSurface=addTriangleSurface(source);
    if(!wantsScalar||!range)state.scalarMeshCache=null;
    if(state.scalarMeshCache?.samples){const samples=state.scalarMeshCache.samples,opacity=Number($('model-opacity')?.value??100)/100;samples.material.alpha=opacity;for(const mesh of samples.meshes)mesh.setEnabled(opacity>0);}
    modelTools?.sync(frame,{surface:drawnSurface,fieldMode:state.mode==='surface',appearance:$('model-appearance')?.value||'neutral',opacity:Number($('model-opacity')?.value??100)/100,widget:$('model-widget')?.value||'none',projections:$('model-projections')?.value||'none',enabled:!activeStatuses.has(state.job?.status)&&!state.importing&&!state.uploading&&!state.importedResults});
    if(!state.reportPresentation){drawDomain(frame.domain);drawAxes(frame);referenceTools?.draw(frame);drawFlow(frame);if(!imported){refreshRecordedVectors();drawRecordedVectors(frame);}}
    if(state.mode==='mesh'&&state.meshVisible!==false)drawAmr();
    if(!state.reportPresentation)inspector?.draw();
    const units=surface?.field_units?.[field]||surface?.units?.[field]||'';
    $('surface-range').textContent=range?`${imported?field+(units?` [${units}]`:'')+' · ':''}${fmt(range.lo)} to ${fmt(range.hi)}`:'';
    if(window.LudwigPalettes)document.documentElement.style.setProperty('--result-ramp',window.LudwigPalettes.gradient(palette,range?.lo,range?.hi));
    referenceTools?.syncRange(range);
    const legend=$('surface-colour-legend');if(legend){legend.hidden=!range;if(range){
      const quantity=field==='cp'?'Pressure coefficient Cp':field==='shear'?'Shear stress':field.charAt(0).toUpperCase()+field.slice(1),unit=units||(field==='pressure'||field==='shear'?'Pa':'');
      $('surface-legend-title').textContent=`${quantity}${unit?` [${unit}]`:''} · ${range.mode==='manual'?'manual limits':range.mode==='full'?'full data range':'high contrast (2–98%)'}`;
      $('surface-legend-low').textContent=Number(range.lo).toPrecision(4);$('surface-legend-high').textContent=Number(range.hi).toPrecision(4);
      legend.title=`Actual data minimum ${Number(range.actualLo).toPrecision(6)}, maximum ${Number(range.actualHi).toPrecision(6)}. ${range.mode==='manual'?'Manual colour limits; use Results > Field scale limits > Automatic to restore automatic limits.':range.mode==='full'?'All values use the full colour scale.':'Values outside the displayed limits use the end colours. Choose Full data range to include every value in the scale.'}`;
    }}
    const noSurface=state.mode==='surface'&&!range;
    state.viewNotice=wallSamples&&range?`${surface.count.toLocaleString()} native wall-cell/facet samples displayed as ${state.scalarMeshCache?.samples?.patchTriangles?.toLocaleString()||'fine'} LBM link patch triangles. No original STL facets are drawn in this field view; no pressure averaging between cells or facets. Patches overlap at the local cell scale and are not physical integration areas. Missing samples leave gaps. ${surface.geometry.ambiguous?surface.geometry.ambiguous.toLocaleString()+' grazing/ambiguous sides are shown two-sided.':''}`:imported?(noSurface?'Choose an available imported surface field or enable Section for volume fields.':''):!source?'Load a geometry file to place a model in this domain.':noSurface?(values&&!hasPlacement?'Surface field is waiting for the solver mesh placement.':'Surface pressure appears when the solver exports its first field.'):'';
    $('viewer-overlay').textContent='';$('viewer-overlay').hidden=true;
    $('viewer-note').textContent=state.amr?'Solver placement and snapped domain · drag to orbit · right drag to pan · wheel to zoom':
      'Domain preview · +X prescribed flow · build for the snapped lattice · drag to orbit';
    updateLabels();updateResultViewStatus();window.LudwigAeroLayout?.refreshNotices();
    if(state.pendingStudyCamera&&state.camera&&source){const saved=state.pendingStudyCamera;state.pendingStudyCamera=null;state.camera.setTarget(new BABYLON.Vector3(...saved.target));state.camera.alpha=saved.alpha;state.camera.beta=saved.beta;state.camera.radius=saved.radius;state.camera._aeroFit=true;}
    else if(!state.camera?.target||!state.camera._aeroFit){fitDisplayedDomain();state.camera._aeroFit=true;}
    const serial=state.viewRenderSerial||0;
    state.scene.onAfterRenderObservable.addOnce(()=>{
      if(serial!==(state.viewRenderSerial||0))return;
      state.renderIdentity={jobId:state.jobId||null,step:surface?.step??null,mode:state.mode,representation:wallSamples?'wall_samples':'facet_averages',sampleCount:wallSamples?surface.count:null,field,sourceUrl:surface?.sourceUrl||null,geometrySha256:surface?.geometrySha256||null,frameSha256:surface?.frameSha256||null,colourLimits:range?[range.lo,range.hi]:null,palette};
      window.dispatchEvent(new CustomEvent('ludwig-aero-view-ready',{detail:state.renderIdentity}));
    });
  }
  function queueRender() {state.renderIdentity=null;state.viewRenderSerial=(state.viewRenderSerial||0)+1;if(!state.renderQueued){state.renderQueued=true;requestAnimationFrame(renderScene);}}
  async function applyImportedResults(dataset,surface=null) {
    const initialJob=state.jobId,initialModel=state.modelId,active={...dataset,id:dataset.id||dataset.dataset_id};
    if(dataset.amr_url){try{active.amr=await api(dataset.amr_url);}catch(error){console.warn('Imported AMR metadata could not be displayed:',error);}}
    if(activeStatuses.has(state.job?.status)||state.resumePending||state.importing||state.uploading||initialJob!==state.jobId||initialModel!==state.modelId)throw new Error('The current case changed or became active. Read the results again after it finishes.');
    state.importedResults=active;state.importedSurface=surface;state.importedResultsReference=null;
    if(state.inspector){state.inspector.data=null;state.inspector.key=null;state.inspector.error='';if(!surface){state.inspector.enabled=true;state.inspector.field=['velocity','pressure','density','u','v','w','vorticity','mach','temperature'].includes(state.importedField)?state.importedField:'mesh';if(!dataset.volume&&dataset.cuts?.length){state.inspector.axis=dataset.cuts[0].axis;state.inspector.position=dataset.cuts[0].position;}}}
    state.mode=surface?(Object.keys(surface.fields||{}).length?'surface':'geometry'):'mesh';syncViewButtons();if(state.camera)state.camera._aeroFit=false;
    inspector?.levelControls();updateResultsAvailability();queueRender();
  }
  function clearImportedResults(render=true) {
    state.importedResults=null;state.importedSurface=null;state.importedResultsReference=null;
    if(state.inspector){state.inspector.data=null;state.inspector.key=null;state.inspector.error='';}
    if($('results-source'))$('results-source').value='simulation';inspector?.levelControls();updateResultsAvailability();if(render)queueRender();
  }
  function setImportedField(field) {
    state.importedField=field;if(!state.importedResults)return;
    if(state.importedSurface?.fields?.[field])state.mode='surface';
    else if((state.importedResults.volume||state.importedResults.cuts?.length)&&['velocity','pressure','density','u','v','w','vorticity','mach','temperature'].includes(field)){
      if(state.importedSurface)state.mode='geometry';
      if(state.inspector){Object.assign(state.inspector,{enabled:true,field,data:null,key:null,error:''});$('section-field').value=field;}
    }
    syncViewButtons();queueRender();
  }
  function initScene() {
    if(typeof BABYLON==='undefined'){$('viewer-overlay').textContent='Babylon.js did not load. Reload the page to use the 3D workbench.';return;}
    try{
      const canvas=$('viewer'),engine=new BABYLON.Engine(canvas,true,{preserveDrawingBuffer:false,stencil:true});
      const scene=new BABYLON.Scene(engine);scene.clearColor=new BABYLON.Color4(.055,.15,.2,1);
      const camera=new BABYLON.ArcRotateCamera('aero-camera',-Math.PI/2-.38,1.08,9,v3(0,0,0),scene);
      camera.lowerRadiusLimit=.005;camera.upperRadiusLimit=1e8;camera.wheelDeltaPercentage=.06;
      camera.inertia=0;camera.panningInertia=0;camera.panningSensibility=0;camera.useInputToRestoreState=false;
      camera.attachControl(canvas,true,false,-1);
      if(camera.inputs.attached.pointers)camera.inputs.attached.pointers.buttons=[0];
      if(camera.movement){camera.movement.panInertia=0;camera.movement.rotationInertia=0;camera.movement.zoomInertia=0;}
      installViewerGestures(canvas,scene,camera);
      const light=new BABYLON.HemisphericLight('ambient',new BABYLON.Vector3(.2,1,.3),scene);light.intensity=1.1;
      const key=new BABYLON.DirectionalLight('key',new BABYLON.Vector3(-.5,-.8,.4),scene);key.intensity=.55;
      state.engine=engine;state.scene=scene;state.camera=camera;
      initModelTools();
      engine.runRenderLoop(()=>{if(canvas.clientWidth&&canvas.clientHeight){scene.render();projectAxisLabels();}});window.addEventListener('resize',()=>engine.resize());
      new ResizeObserver(()=>engine.resize()).observe(canvas.parentElement);
    }catch(error){$('viewer-overlay').textContent=`WebGL could not start: ${error.message}`;}
  }
  function initModelTools() {
    if(!state.scene||!window.LudwigAeroModelTools)return;
    try{modelTools=window.LudwigAeroModelTools.create({BABYLON,scene:state.scene,
      getSourceFile:()=>state.originalFile||null,getSourceBounds:()=>state.sourceBounds,getTransformSettings:transformSettings,
      setTransformSettings(settings){for(const [key,prefix] of [['scale','scale'],['rotation_deg','rotate'],['translation','move']])for(let i=0;i<3;i++)$(prefix+'-'+['x','y','z'][i]).value=String(Number(settings[key][i].toPrecision(10)));},
      onTransformCommit:invalidateMesh,onRender:queueRender,
      onAvailability(info){updateModelAppearance(info);updateLabels();},
      onStatus(text,error){state.modelToolStatus=text;if(error)progress(`Appearance error: ${text}`,'Appearance unavailable');updateLabels();}});
    }catch(error){state.modelToolStatus=`Model tools are unavailable: ${error.message}`;console.warn(state.modelToolStatus);}
  }

  function readLBMP(buffer) {
    const v=new DataView(buffer);if(v.byteLength<16||String.fromCharCode(...new Uint8Array(buffer,0,4))!=='LBMP')throw new Error('Invalid surface field file.');
    const version=v.getUint32(4,true),n=v.getUint32(8,true),objects=v.getUint32(12,true);
    if(version!==1||!n||n>5_000_000||objects>10000||n*64+16+objects*4>v.byteLength)throw new Error('Unsupported or incomplete surface field version or size.');
    let pos=16;const names=[],decoder=new TextDecoder();
    for(let i=0;i<objects;i++){
      if(pos+4>v.byteLength)throw new Error('Truncated surface field.');
      const len=v.getUint32(pos,true);pos+=4;if(len>4096||pos+len>v.byteLength)throw new Error('Invalid surface object name.');
      names.push(decoder.decode(new Uint8Array(buffer,pos,len)));pos+=len;
    }
    const total=(n*9+n*3+n*4)*4;
    if(pos+total>v.byteLength)throw new Error('Truncated surface field values.');
    const readFloat=count=>{const out=new Float32Array(count);for(let i=0;i<count;i++)out[i]=v.getFloat32(pos+i*4,true);pos+=count*4;return out;};
    const readInt=count=>{const out=new Int32Array(count);for(let i=0;i<count;i++)out[i]=v.getInt32(pos+i*4,true);pos+=count*4;return out;};
    return {version,names,vertices:readFloat(n*9),normals:readFloat(n*3),pressure:readFloat(n),shear:readFloat(n),cp:readFloat(n),objectIds:readInt(n)};
  }
  async function maybeLoadAmr(job) {
    const url=apiUrl(job?.artifacts?.amr);
    if(!url||url===state.amrUrl)return;
    try{
      const response=await fetch(url,{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);
      const data=await response.json();
      if(state.jobId!==job.job_id)return;
      if(!Array.isArray(data?.domain_size)||!Array.isArray(data?.levels))throw new Error('Invalid AMR artifact.');
      state.amr=data;state.amrUrl=url;if(state.camera)state.camera._aeroFit=false;
      inspector?.levelControls();
      queueRender();
    }catch(error){progress(`Lattice built, but its 3D artifact could not be displayed: ${error.message}`,job.status);}
  }
  function bindNativeSurface(data,record={}){
    if(record.geometry_frame==='domain'){data.surface_offset=[0,0,0];data.geometry_frame='domain';return data;}
    if(record.geometry_frame==='solver_local'&&Array.isArray(record.mesh_offset_m)){data.surface_offset=record.mesh_offset_m.slice();data.geometry_frame='solver_local';return data;}
    const vertices=state.amr?.geometry?.vertices;
    if(!vertices||vertices.length!==data.vertices.length)throw new Error('The surface snapshot does not match the solver mesh face count. Load its matching mesh or VTK surface.');
    data.vertices=Float32Array.from(vertices);data.surface_offset=[0,0,0];data.geometry_frame='domain';
    return data;
  }
  function selectedSurfaceRecord(job){
    const artifacts=job?.artifacts||{},choice=$('surface-representation')?.value||'auto';
    const key=choice==='facet'?'surface_facet_latest':choice==='native'?'surface_native_latest':'surface_latest';
    const url=apiUrl(artifacts[key]||(choice==='facet'&&!artifacts.surface_native_latest?artifacts.surface_latest:null));
    if(!url)return null;
    return {...(artifacts[key+'_info']||artifacts.surface_latest_info||{}),...(artifacts.surfaces||[]).find(item=>item.url===url),url};
  }
  async function readSurfaceRecord(record){
    if(/\.lbmw(?:\?|$)/i.test(record.url))return window.LudwigAeroWallSurface.load(record);
    const response=await fetch(record.url,{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const data=bindNativeSurface(readLBMP(await response.arrayBuffer()),record);data.step=record.step??Number(/_(\d+)\.lbmp(?:\?|$)/i.exec(record.url)?.[1]);data.sourceUrl=record.url;return data;
  }
  function syncSurfaceQuantities(surface){
    const select=$('surface-field');for(const option of select.options)option.disabled=!(surface?.fields?.[option.value]??surface?.[option.value]);
    if(select.selectedOptions[0]?.disabled)select.value=surface?.cp?'cp':'pressure';
    const description=$('surface-sampling-note');if(description)description.textContent=surface?.representation==='wall_samples'?`Native LBM link patch mesh · ${surface.count.toLocaleString()} local samples, two display triangles each. Each colour is its own fluid-cell pressure. Original STL facets are hidden in this field view. Patches overlap within the local cell's circumradius; their areas are not physical integration weights. Missing samples leave gaps. Shear is available only in facet averages.`:'Facet averages use the original geometry faces. Choose native wall samples when the solver has exported them to see detail within each face.';
  }
  async function maybeLoadSurface(job,force=false) {
    if(!force&&(state.liveResults===false||state.surfaceFollowLive===false))return;
    const record=selectedSurfaceRecord(job),url=record?.url,filenameStep=/_(\d+)\.lbmw?(?:p)?(?:\?|$)/i.exec(url||'')?.[1];
    const step=finite(filenameStep)?Number(filenameStep):finite(record?.step)?Number(record.step):null,key=`${url}:${step??'unknown'}`;
    if(!url||(!force&&state.lastSurfaceKey===key)||(!force&&job.status!=='completed'&&Date.now()-state.lastSurfaceAt<8000))return;
    state.lastSurfaceAt=Date.now();
    try{
      state.renderIdentity=null;const data=await readSurfaceRecord({...record,step});if(state.jobId!==job.job_id)return;data.step=step;state.surface=data;syncSurfaceQuantities(data);state.surfaceError=null;state.lastSurfaceKey=key;if(state.autoSurfaceDisplay){state.mode='surface';syncViewButtons();state.autoSurfaceDisplay=false;}
      updateResultsAvailability();if(state.mode==='surface')queueRender();
    }catch(error){state.surfaceError=error.message;updateResultViewStatus();if($('results-availability-summary'))$('results-availability-summary').textContent=`Surface could not be refreshed: ${error.message}${state.surface?'. The previously loaded surface is retained.':''}`;console.warn('Surface field unavailable:',error);}
  }
  function addArtifactLink(parent,label,url) {
    const safe=apiUrl(url);if(!safe)return;
    const a=document.createElement('a');a.href=safe;a.target='_blank';a.rel='noopener noreferrer';a.textContent=label;parent.appendChild(a);
  }
  function renderArtifacts(job) {
    const holder=$('artifact-links');holder.replaceChildren();const a=job?.artifacts||{};
    addArtifactLink(holder,'Solver STL',a.model);addArtifactLink(holder,'AMR mesh',a.amr);
    addArtifactLink(holder,'Case summary',a.summary);addArtifactLink(holder,'Latest surface',a.surface_latest);
    const lastCut=Array.isArray(a.cuts)?a.cuts.at(-1):null;
    addArtifactLink(holder,'Latest cut data',lastCut?.url);
  }
  function cutOptions(job) {
    const cuts=Array.isArray(job?.artifacts?.cuts)?job.artifacts.cuts:[],select=$('cut-result-select'),prior=state.selectedCut;
    select.replaceChildren();
    for(let i=0;i<cuts.length;i++){
      const cut=cuts[i],option=document.createElement('option');option.value=String(i);
      option.textContent=`${String(cut.axis||'?').toUpperCase()} = ${fmt(cut.position)} m · step ${fmt(cut.step)}`;
      select.appendChild(option);
    }
    if(!cuts.length){const o=document.createElement('option');o.textContent='No recorded planes';o.value='';select.appendChild(o);}
    select.disabled=!cuts.length;$('cut-field').disabled=!cuts.length;
    const retained=state.selectedCutRecord&&cuts.findIndex(cut=>(cut.url||cut.json_url||`${cut.axis}:${cut.position}:${cut.step}`)===state.selectedCutRecord);
    let chosen=Number.isInteger(retained)&&retained>=0?retained:cuts.length&&prior&&Number(prior)<cuts.length?Number(prior):Math.max(0,cuts.length-1);
    if(state.liveResults!==false&&state.cutFollowLive!==false&&cuts.length){
      const plane=state.selectedCutPlane;let candidates=cuts.map((cut,index)=>({cut,index}));
      if(plane){const matching=candidates.filter(({cut})=>cut.axis===plane.axis&&Math.abs(Number(cut.position)-plane.position)<1e-6);if(matching.length)candidates=matching;}
      chosen=candidates.sort((a,b)=>Number(a.cut.step||0)-Number(b.cut.step||0)).at(-1).index;
    }
    select.value=String(chosen);if(cuts[chosen])state.selectedCutPlane={axis:cuts[chosen].axis,position:Number(cuts[chosen].position)};
    state.selectedCut=select.value;if(cuts[chosen])state.selectedCutRecord=cuts[chosen].url||cuts[chosen].json_url||`${cuts[chosen].axis}:${cuts[chosen].position}:${cuts[chosen].step}`;
    renderCut(job);if(state.liveResults!==false)refreshRecordedVectors(job,true);
  }
  function paintNumericCut(data) {
    state.lastNumericCut=data;
    data=window.LudwigMeshInspector.visibleField(data,state.levelVisibility,state.importedResults?.amr||state.amr);
    const n1=Number(data.n1),n2=Number(data.n2),values=data.values;
    if(!Number.isInteger(n1)||!Number.isInteger(n2)||n1<2||n2<2||n1*n2>2_000_000||!Array.isArray(values)||values.length!==n1*n2)
      throw new Error('Invalid numeric cut dimensions.');
    const raw=scalarRange(values),range=referenceTools?.range(raw,'recorded',data.field)||raw,canvas=$('cut-canvas');if(!range){canvas.width=n1;canvas.height=n2;canvas.getContext('2d').clearRect(0,0,n1,n2);canvas.hidden=false;$('cut-result-note').textContent='No visible fluid values. Enable the owning AMR levels in Display > AMR levels.';return;}
    referenceTools?.syncRange(range,'recorded',data.field);
    canvas.width=n1;canvas.height=n2;const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(n1,n2);
    for(let j=0;j<n2;j++)for(let i=0;i<n1;i++){
      const value=values[i+j*n1],dst=((n2-j-1)*n1+i)*4,rgb=typeof value==='number'&&Number.isFinite(value)?scalarColor(value,range).map(v=>Math.round(v*255)):[33,43,51];
      pixels.data[dst]=rgb[0];pixels.data[dst+1]=rgb[1];pixels.data[dst+2]=rgb[2];pixels.data[dst+3]=255;
    }
    ctx.putImageData(pixels,0,0);canvas.hidden=false;applyRecordedCutZoom();
    $('cut-result-note').textContent=`${String(data.normal_axis||'?').toUpperCase()} = ${fmt(data.normal_coord_m)} m · step ${fmt(data.step)} · ${data.field} (${data.units||'solver units'}) · ${String(data.axis1||'?').toUpperCase()} / ${String(data.axis2||'?').toUpperCase()} · colour range ${fmt(range.lo)} to ${fmt(range.hi)}.`;
  }
  async function renderCut(job=state.job,force=false) {
    if(!force&&state.liveResults===false)return;
    const cuts=Array.isArray(job?.artifacts?.cuts)?job.artifacts.cuts:[],cut=cuts[Number($('cut-result-select').value)],img=$('cut-image'),canvas=$('cut-canvas');
    const hasDisplay=Boolean(state.cutDisplayJobId&&state.cutDisplayJobId===job?.job_id);
    if(!cut){if(!hasDisplay){img.hidden=true;canvas.hidden=true;}$('cut-result-note').textContent=hasDisplay?'The last loaded recorded plane is retained while the file list refreshes.':'Recorded slices appear here during a run. Plane locations must be chosen before the run.';return;}
    const field=$('cut-field').value,json=apiUrl(cut.json_url),key=`${json}:${field}:${cut.step}`;
    if(json){
      if(state.cutRenderKey===key)return;
      state.cutRenderKey=key;
      try{
        const separator=json.includes('?')?'&':'?',data=await api(`${json}${separator}field=${encodeURIComponent(field)}`);
        if(state.cutRenderKey!==key)return;
        paintNumericCut(data);state.cutDisplayJobId=job.job_id;img.hidden=true;img.removeAttribute('src');return;
      }catch(error){console.warn('Numeric cut unavailable:',error);state.cutRenderKey=null;}
    }
    const png=cut.png_urls||{},url=apiUrl(png[field]||png.velocity||png.pressure);
    if(url){canvas.hidden=true;img.src=url;img.hidden=false;state.cutDisplayJobId=job.job_id;applyRecordedCutZoom();$('cut-result-note').textContent=`${String(cut.axis).toUpperCase()} = ${fmt(cut.position)} m · step ${fmt(cut.step)} · solver image.`;}
    else{if(!hasDisplay){img.hidden=true;canvas.hidden=true;}$('cut-result-note').textContent=`${String(cut.axis).toUpperCase()} = ${fmt(cut.position)} m · step ${fmt(cut.step)}. ${hasDisplay?'The last loaded plane is retained because the new field is unavailable.':'Download its native numeric slice above.'}`;}
  }
  function renderMeshDetails(job) {
    const actual=job?.actual,node=$('mesh-details'),dims=actual?.domain_size_m;
    if(!actual||!Array.isArray(dims)||dims.length!==3){node.hidden=true;return;}
    const dimension=value=>Number(value).toLocaleString(undefined,{maximumSignificantDigits:4});
    const lines=[`Built domain ${dims.map(dimension).join(' × ')} m.`],requested=job?.estimate?.domain_size_m;
    const adjustment=job?.case?.domain_adjustment||job?.estimate?.domain_adjustment;
    if(adjustment?.expanded){lines.push(adjustment.message||'Free-air domain expanded automatically.');if(Array.isArray(adjustment.requested_size_m))lines.push(`Original envelope ${adjustment.requested_size_m.map(dimension).join(' × ')} m.`);}
    if(finite(job?.case?.target_vram_gb))lines.push(`VRAM target ${Number(job.case.target_vram_gb).toFixed(2)} GB; selected finest cell size ${dimension(job.case.surface_dx)} m; estimated run allocation ${Number(actual.memory_gb).toFixed(2)} GB.`);
    let warning=false;
    if(Array.isArray(requested)&&requested.length===3&&dims.some((v,i)=>Number(v)>Number(requested[i])*1.02)){
      lines.push(`Requested envelope ${requested.map(dimension).join(' × ')} m; the solver expanded the lattice domain.`);
    }
    let qualityWarning=actual.quality_warning||actual.quality?.warning||'';
    const openOverride=$('geometry-handling')?.value==='two_sided_open'&&actual.run_blocked_by_geometry===true;
    if(openOverride){qualityWarning=actual.geometry_warning?qualityWarning.replace(actual.geometry_warning,'').trim():actual.run_blocked_by_mesh_quality?qualityWarning:'';lines.push('Two-sided / intentional open surface selected. Run will apply this policy to the existing mesh; topology audit notes remain available under information.');}
    const levels=Array.isArray(actual.levels)?actual.levels:[];
    const finest=levels.filter(l=>finite(l.dx_m)&&Number(l.dx_m)>0).sort((a,b)=>Number(a.dx_m)-Number(b.dx_m))[0];
    if(finest){
      const count=Number(actual.finest_wall_cells??finest.bouzidi_boundary_cells),numLevels=actual.num_levels??levels.length;
      lines.push(`${numLevels} level${Number(numLevels)===1?'':'s'}; finest Δx = ${dimension(finest.dx_m)} m; ${fmt(count)} finest-level Bouzidi boundary cells.`);
      if(finite(actual.wall_cells_per_surface_cell))lines.push(`${dimension(actual.wall_cells_per_surface_cell)} wall cells per target surface cell.`);
      if(!qualityWarning&&Number.isFinite(count)&&count<16){warning=true;
        lines.push(count===0?'No body boundary cells lie on the finest level. Reduce cell size or inspect refinement before running.':
          `Only ${fmt(count)} body boundary cells lie on the finest level. Inspect surface coverage before running.`);}
    }
    if(qualityWarning){warning=true;lines.push(String(qualityWarning));}
    else if(actual.run_blocked_by_mesh_quality===true||actual.quality?.runnable===false){warning=true;lines.push('Run disabled by the mesh quality check.');}
    if(actual.mesh==='octree'&&actual.octree){
      const o=actual.octree,r=o.reports||{};
      lines.push(`Octree: ${fmt(o.fluid_cells_total)} fluid cells in ${fmt(o.stored_slots)} stored slots; weighted work ${Number(o.work_W).toExponential(2)}; bands ${o.band_fine}/${o.band}${Number(o.wake_length)>0?`; wake ${o.wake_length} body lengths at level ${o.wake_level}`:''}. ${fmt(r.wall_cells)} wall cells, ${fmt(r.cut_links)} cut links; unreached surface ${(100*Number(r.unreached_triangle_area_fraction||0)).toFixed(1)} %. RAM ${Number(actual.memory_gb).toFixed(2)} GB; no GPU is used.`);
      if(Number(r.leak_cells)>0){warning=true;lines.push(`${fmt(r.leak_cells)} fluid cells lie inside the surface by the parity check: the STL is open or self-intersecting there. Inspect the model before trusting the mesh.`);}
    }
    node.textContent=lines.join(' ');node.classList.toggle('warning',warning);node.hidden=false;window.LudwigAeroLayout?.refreshNotices();
  }
  function renderJob(job) {
    state.job=job;const status=String(job.status||'unknown'),estimate=job.estimate||{},actual=job.actual||{},progressInfo=job.progress||{};
    if(job.case)state.activeConfig=job.case;
    if(status==='mesh_ready'&&!state.draftDirty&&state.resolvedJob!==job.job_id){
      const resolved=job.case?.domain||{},editable=(resolved.sizing||'minimum')!=='fixed'?(job.case?.requested_domain||job.case?.domain_adjustment?.requested_domain||resolved):resolved;
      for(const key of ['upstream','downstream','lateral','height'])if(finite(editable[key]))$(key).value=String(editable[key]);
      state.resolvedJob=job.job_id;
    }
    if(status==='mesh_ready'&&!state.draftDirty&&finite(job?.case?.target_vram_gb))$('surface-dx').value=String(job.case.surface_dx);
    $('metric-cells').textContent=fmt(actual.cells??estimate.cells);
    const memory=actual.memory_gb??estimate.memory_gb,measuring=estimate.measured===false&&!finite(actual.memory_gb);$('metric-memory').textContent=measuring?'Measuring':finite(memory)?`${Number(memory).toFixed(2)} GB`:'—';
    const octreeJob=job?.case?.solver==='q19_octree';
    const fits=gpuFit(job),fit=octreeJob?'CPU solver · RAM, no GPU':fits===false?'Above GPU budget':fits===true?'Within GPU budget':measuring?'Build measurement pending':'Estimate pending';
    $('metric-fit').textContent=fit;$('metric-fit').className=octreeJob?'good':fits===false?'bad':fits===true?'good':'';
    if($('metric-memory-label'))$('metric-memory-label').textContent=octreeJob?'RAM':'VRAM';
    const selectedBudget=selectedGpuBudget(job);$('metric-budget').textContent=octreeJob?'—':finite(selectedBudget)?`${Number(selectedBudget).toFixed(1)} GB`:'Unknown';
    const draftNote=state.draftDirty?' Draft changed; build again to apply your edits.':'';
    progress((job.mesh_proposal&&job.run_started_utc?status==='completed'?'Saved solver results available':status.replaceAll('_',' '):job.message||status.replaceAll('_',' '))+draftNote,status.replaceAll('_',' '));
    const meshActive=['meshing','sizing','mesh_sizing'].includes(status),meshProgress=job.mesh_progress||{};
    window.LudwigAeroLayout?.setMeshBuildProgress({active:meshActive,status:meshActive?'creating':status==='mesh_ready'?'ready':(status==='mesh_failed'||status==='cancelled'&&job.cancel_reason==='gpu_budget_exceeded')?'error':'idle',percent:status==='mesh_ready'?100:finite(meshProgress.percent)?Number(meshProgress.percent):null,stage:meshProgress.phase||'Creating mesh',message:meshProgress.detail||job.message||'',log:job.activity_log||[],jobId:job.job_id});
    if(job.mesh_proposal && ['mesh_failed','cancelled'].includes(status) && !job.run_started_utc){if(state.meshProposal!==job.mesh_proposal)showMeshProposal(job.mesh_proposal,job.message||'A different mesh size is required.');}
    const percent=Number(progressInfo.percent);
    if(meshActive){if(finite(meshProgress.percent))$('job-progress').value=Number(meshProgress.percent);else $('job-progress').removeAttribute('value');if($('job-progress-label'))$('job-progress-label').textContent='Creating mesh…';}
    else{$('job-progress').value=status==='completed'?100:Number.isFinite(percent)?Math.max(0,Math.min(100,percent)):0;if($('job-progress-label'))$('job-progress-label').textContent='Solver progress';}
    window.LudwigOutputControls?.render(job);renderSolverProgress(job);resultBrowser?.render(job);updateVramTarget(job.estimate?.gpu);
    renderMeshDetails(job);renderArtifacts(job);cutOptions(job);updateActions();updateBudgetGuard(job);
  }
  async function attachStudyResults(id,{compatible=true}={}){
    if(!jobIdPattern.test(id))return;const modelId=state.modelId;
    const job=await api(`/api/aero/jobs/${encodeURIComponent(id)}`);if(state.modelId!==modelId)return;
    state.jobId=id;state.job=job;state.activeConfig=job.case;state.draftDirty=compatible===false;persistJob(id);
    state.restoredJobId=id;clearMeshProposal();renderJob(job);await maybeLoadAmr(job);if(job.artifacts?.surface_latest){state.mode='surface';syncViewButtons();await maybeLoadSurface(job,true);}
    if(compatible===false)progress('Historical results are loaded from the saved run. The editable study differs; create a new mesh to apply its setup.','Historical results');
    updateActions();queueRender();if(activeStatuses.has(job.status))pollLater(id,2000);
  }
  async function loadLatestResults(force=false){
    if(!state.jobId)return;state.surfaceFollowLive=true;state.cutFollowLive=true;
    const id=state.jobId,job=await api(`/api/aero/jobs/${encodeURIComponent(id)}`);if(state.jobId!==id)return;
    renderJob(job);await maybeLoadAmr(job);if(state.jobId!==id)return;await maybeLoadSurface(job,true);if(state.jobId!==id)return;state.cutRenderKey=null;await renderCut(job,true);queueRender();
  }
  async function showSurfaceResults({refresh=false}={}){
    if(state.importedResults&&!state.importedSurface&&state.surface)clearImportedResults(false);
    if(!state.importedResults&&state.jobId&&(refresh||!state.surface))await loadLatestResults(true);
    const surface=state.importedResults?state.importedSurface:state.surface;
    if(!surface){
      updateResultViewStatus();
      window.LudwigAeroLayout?.showMessage(state.surfaceError?`The surface could not be loaded: ${state.surfaceError}`:'No surface flow field is available for this case yet. Open a completed study with results, or use Read native / VTK results. A mesh by itself contains no pressure field.',{title:'Surface results',urgent:true});
      window.LudwigAeroLayout?.selectParameters?.('results','results-path');return false;
    }
    const fieldSelect=$(state.importedResults?'results-field':'surface-field');
    if(!state.importedResults)syncSurfaceQuantities(surface);
    if(!(surface.fields?.[fieldSelect.value]??surface[fieldSelect.value])){
      const field=Array.from(fieldSelect.options||[]).find(option=>surface.fields?.[option.value]??surface[option.value]);if(field)fieldSelect.value=field.value;
    }
    // "Show surface" always restores the full coloured surface. A section may
    // have clipped it completely during an earlier inspection; data were still
    // loaded, but the view looked empty with no obvious recovery action.
    if(state.inspector){state.inspector.clip=false;if($('section-clip'))$('section-clip').checked=false;}
    state.mode='surface';syncViewButtons();selectResultPane('scene');updateResultsAvailability();queueRender();return true;
  }
  async function loadCurrentResults(){
    const button=$('results-load-current');if(button)button.disabled=true;
    try{
      if(state.importedSurface)return await showSurfaceResults();
      if(!state.jobId){window.LudwigAeroLayout?.selectParameters?.('results','results-path');window.LudwigAeroLayout?.showMessage('Open a saved study to restore its linked results, or choose native / VTK files in Read results. No solver run is started by loading results.',{title:'Load saved results'});return;}
      await loadLatestResults(true);
      if(state.surface)await showSurfaceResults();
      else if(state.job?.artifacts?.cuts?.length){selectResultPane('cuts');await renderCut(state.job,true);}
      else{selectResultPane('case');window.LudwigAeroLayout?.showMessage('This case has no completed surface or recorded-plane field yet. Files and images lists the available outputs and explains which output types were enabled.',{title:'Available results'});}
    }catch(error){window.LudwigAeroLayout?.showMessage(`Results could not be refreshed: ${error.message}. Previously loaded fields remain available.`,{title:'Load results',urgent:true});}
    finally{if(button)button.disabled=false;updateResultViewStatus();}
  }
  const recordedCutView={zoom:1,pan:[0,0]};
  function applyRecordedCutZoom(){
    for(const id of ['cut-canvas','cut-image']){const node=$(id);if(node?.style)node.style.transform=`translate(${recordedCutView.pan[0]}px, ${recordedCutView.pan[1]}px) scale(${recordedCutView.zoom})`;}
    const status=$('cut-zoom-status');if(status)status.textContent=`${Math.round(recordedCutView.zoom*100)}%`;
  }
  function zoomRecordedCut(factor,point=null){
    const viewport=$('recorded-cut-viewport'),rect=viewport?.getBoundingClientRect?.(),old=recordedCutView.zoom,next=Math.max(1,Math.min(128,old*factor));
    const anchor=point||(rect?[rect.width/2,rect.height/2]:[0,0]),center=rect?[rect.width/2,rect.height/2]:anchor,ratio=next/old;
    recordedCutView.pan=anchor.map((v,i)=>v-center[i]-(v-center[i]-recordedCutView.pan[i])*ratio);recordedCutView.zoom=next;
    if(next===1)recordedCutView.pan=[0,0];applyRecordedCutZoom();
  }
  function wireRecordedCutZoom(){
    const viewport=$('recorded-cut-viewport');if(!viewport)return;let pan=null;
    const point=event=>{const rect=viewport.getBoundingClientRect();return [event.clientX-rect.left,event.clientY-rect.top];};
    viewport.addEventListener('wheel',event=>{event.preventDefault();zoomRecordedCut(Math.exp(-event.deltaY*.0015),point(event));},{passive:false});
    viewport.addEventListener('pointerdown',event=>{if(![0,2].includes(event.button))return;event.preventDefault();pan={id:event.pointerId,start:point(event),offset:recordedCutView.pan.slice()};try{viewport.setPointerCapture(event.pointerId);}catch{}});
    viewport.addEventListener('pointermove',event=>{if(pan?.id!==event.pointerId)return;event.preventDefault();const p=point(event);recordedCutView.pan=p.map((v,i)=>pan.offset[i]+v-pan.start[i]);applyRecordedCutZoom();});
    const finish=event=>{if(pan?.id!==event.pointerId)return;pan=null;try{viewport.releasePointerCapture(event.pointerId);}catch{}};
    viewport.addEventListener('pointerup',finish);viewport.addEventListener('pointercancel',finish);viewport.addEventListener('lostpointercapture',()=>{pan=null;});
    viewport.addEventListener('contextmenu',event=>event.preventDefault());
    const reset=()=>{recordedCutView.zoom=1;recordedCutView.pan=[0,0];applyRecordedCutZoom();};viewport.addEventListener('dblclick',reset);
    $('cut-zoom-in')?.addEventListener('click',()=>zoomRecordedCut(1.5));$('cut-zoom-out')?.addEventListener('click',()=>zoomRecordedCut(1/1.5));$('cut-reset-zoom')?.addEventListener('click',reset);
    applyRecordedCutZoom();
  }
  function renderSolverProgress(job){
    const monitor=job.monitoring||{},gpu=monitor.gpu||{},ram=monitor.ram||{},process=ram.solver||{},residual=monitor.residuals||{};
    const metrics={iteration:Number(monitor.iteration??job.progress?.step??0),total_iterations:Number(monitor.total_iterations??job.case?.steps??0),
      residual:residual.available?residual.values:null,residual_reason:residual.reason,elapsed_s:monitor.wall_s??monitor.elapsed_s,eta_s:null,eta_credible:false,
      gpu_used_gb:gpu.used_gb,gpu_total_gb:gpu.total_gb,gpu_process_gb:gpu.process_used_gb??gpu.recorded_solver_used_gb,gpu_process_source:gpu.process_used_gb==null?'recorded':'live',
      process_ram_gb:process.working_set_gb,process_ram_source:process.source,pid:process.pid,system_ram_used_gb:ram.system_used_gb,system_ram_total_gb:ram.system_total_gb};
    const log=[...(job.activity_log||[])],diagnostics=monitor.diagnostics||{},coefficients=monitor.coefficients||{};
    const details=[...Object.entries(diagnostics).filter(([key,value])=>/rho|min|max|MLUPS/i.test(key)&&Number.isFinite(value)).slice(0,6),...Object.entries(coefficients).slice(0,5)].map(([key,value])=>`${key}=${Number(value).toPrecision(5)}`);
    if(details.length)log.push({message:details.join(' | ')});
    const active=['running','waiting_for_gpu','queued','starting'].includes(job.status),status=active?job.status:['completed','failed'].includes(job.status)?job.status:job.status==='cancelled'&&job.run_started_utc?'cancelled':'idle';
    window.LudwigAeroLayout?.setSolverProgress?.({active,status,percent:Number(monitor.progress_percent??job.progress?.percent??0),phase:state.restoredJobId===job.job_id?'Saved run':monitor.phase?.label,message:job.run_started_utc&&job.mesh_proposal?.message===job.message?'Saved solver run':monitor.phase?.detail||job.message,metrics,log,jobId:job.job_id,restored:state.restoredJobId===job.job_id});
  }
  async function viewSurfaceRecord(file){
    const jobId=state.jobId;
    const native=['lbmp','lbmw'].includes(file.format)||/\.lbmw(?:p)?(?:\?|$)/i.test(file.url)?file:(state.job?.artifacts?.surfaces||[]).find(item=>item.format==='lbmp'&&Number(item.step)===Number(file.step));
    if(!native){if(window.LudwigAeroResults?.readUrl)return window.LudwigAeroResults.readUrl(file);throw new Error('Use the matching native surface or load this VTK file after the run.');}
    state.renderIdentity=null;const data=await readSurfaceRecord(native);if(state.jobId!==jobId)return;
    state.surface=data;state.surfaceFollowLive=false;if($('surface-representation'))$('surface-representation').value=data.representation==='wall_samples'?'native':'facet';syncSurfaceQuantities(data);await showSurfaceResults();
  }
  async function prepareReportView({jobId=state.jobId,step=null,sourceUrl=null,geometrySha256=null,frameSha256=null,field='cp',fit=false}={}){
    if(!['cp','pressure'].includes(field))throw new Error('Native wall reports support Cp or pressure only.');
    if(jobId!==state.jobId)throw new Error('The reporting view is attached to another study.');
    if(sourceUrl){const record=(state.job?.artifacts?.surfaces||[]).find(item=>item.url===sourceUrl);if(!record||record.format!=='lbmw'||step!=null&&Number(record.step)!==Number(step)||geometrySha256&&record.geometry_sha256!==geometrySha256)throw new Error('The requested native wall snapshot is not in this study.');await viewSurfaceRecord(record);}
    else{$('surface-representation').value='native';await loadLatestResults(true);}
    if(state.surface?.representation!=='wall_samples')throw new Error('No native wall samples are available for this report.');
    if(step!=null&&state.surface.step!==Number(step))throw new Error('The requested iteration is not loaded.');
    if(frameSha256&&state.surface.frameSha256!==frameSha256)throw new Error('The pressure frame SHA-256 does not match the requested report.');
    $('surface-field').value=field;state.meshVisible=false;await showSurfaceResults();updateActions();
    if(fit){
      state.reportPresentation=true;
      for(const id of ['model-projections','velocity-vectors','display-aids','model-widget'])if($(id))$(id).value='none';
      $('model-opacity').value='100';$('model-opacity-value').value='100';
      fitNativeReport(state.surface);
    }
    return new Promise((resolve,reject)=>{const timer=setTimeout(()=>{window.removeEventListener('ludwig-aero-view-ready',ready);reject(new Error('Timed out waiting for the native surface renderer.'));},120000);function ready(event){const identity=event.detail;if(identity.jobId!==jobId||identity.representation!=='wall_samples'||identity.field!==field||step!=null&&identity.step!==Number(step))return;clearTimeout(timer);window.removeEventListener('ludwig-aero-view-ready',ready);resolve(identity);}window.addEventListener('ludwig-aero-view-ready',ready);queueRender();});
  }
  function alignInspectorToRecorded(cut){
    if(!cut||!inspector?.getView||!inspector?.restoreView)return;
    const view=inspector.getView();view.inspector={...(view.inspector||{}),enabled:true,axis:cut.axis,position:Number(cut.position),field:$('cut-field').value};inspector.restoreView(view);
  }
  async function viewCutRecord(file){
    const cuts=state.job?.artifacts?.cuts||[],index=cuts.findIndex(cut=>cut.url===file.url);
    if(index<0)throw new Error('Choose an available recorded plane in Results, or import this saved cut file.');
    $('cut-result-select').value=String(index);state.selectedCut=String(index);state.selectedCutRecord=cuts[index].url||cuts[index].json_url;state.cutFollowLive=false;state.cutRenderKey=null;alignInspectorToRecorded(cuts[index]);await renderCut(state.job,true);selectResultPane('cuts');queueRender();
  }
  async function viewVolumeRecord(file){
    if(activeStatuses.has(state.job?.status)){
      $('section-enabled').checked=true;$('section-enabled').dispatchEvent(new Event('change',{bubbles:true}));$('section-field').value='velocity';$('section-field').dispatchEvent(new Event('change',{bubbles:true}));selectResultPane('cuts');return;
    }
    if(window.LudwigAeroResults?.readUrl)return window.LudwigAeroResults.readUrl(file);
    throw new Error('Read this volume file from Results > Read native / VTK results.');
  }
  function finishResume(job) {
    clearImportedResults(false);
    const caseInfo=job?.case;
    if(!caseInfo?.model_id)throw new Error('Saved case has no model reference.');
    if(state.modelId!==caseInfo.model_id){state.originalFile=null;state.source=null;state.sourceBounds=null;state.sourceModelId=null;state.sourceAssetModelId=null;modelTools?.clear();}
    if(job.study_name&&$('study-name'))$('study-name').value=job.study_name;
    state.resumePending=false;state.activeConfig=caseInfo;state.modelId=caseInfo.model_id;
    if(!state.resumeDraftEdited){hydrateCase(caseInfo);state.draftDirty=false;}
    else state.draftDirty=true;
    $('metric-triangles').textContent=fmt(caseInfo.triangles);
    message(`Resumed case · ${fmt(caseInfo.triangles)} triangles · model ready.`);
    state.mode=job?.artifacts?.surface_latest?'surface':job?.artifacts?.amr?'mesh':'geometry';
    syncViewButtons();loadStoredModel(caseInfo.model_id);queueRender();
  }
  function resumeStoredJob() {
    const id=storedJobId();if(!id)return;
    state.jobId=id;state.job={status:'resuming'};state.resumePending=true;state.restoredJobId=id;state.resumeDraftEdited=false;
    persistJob(id);progress('Loading the saved case and its solver artifacts…','Resuming');updateActions();pollJob(id);
  }
  function pollLater(id,ms=2000) {clearTimeout(state.pollTimer);state.pollTimer=setTimeout(()=>pollJob(id),ms);}
  async function pollJob(id) {
    if(state.jobId!==id)return;
    try{
      const job=await api(`/api/aero/jobs/${encodeURIComponent(id)}`);
      if(state.jobId!==id)return;
      if(state.refreshFailures){window.LudwigAeroLayout?.clearMessage?.(`case-refresh:${id}`);state.refreshFailures=0;}
      if(state.resumePending)finishResume(job);
      renderJob(job);
      await maybeLoadAmr(job);await maybeLoadSurface(job);
      if(activeStatuses.has(job.status)||job.exports?.status==='generating')pollLater(id,2000);
    }catch(error){
      if(state.resumePending&&(error.status===404||(error.status===400&&/Unknown Aero Lab job/i.test(error.message)))){
        state.resumePending=false;state.jobId=null;state.job=null;clearPersistedJob();
        progress('The saved case is no longer available. Build a new lattice.','Case unavailable');updateActions();return;}
      const failures=(state.refreshFailures||0)+1;state.refreshFailures=failures;
      $('job-state').textContent=`Refreshing case - retry ${failures}`;
      if(failures>=3)window.LudwigAeroLayout?.showMessage(`The case could not be refreshed after ${failures} attempts. ${error.message}`,{key:`case-refresh:${id}`,urgent:true,title:'Case refresh needs attention',diagnosticUrl:error.details?.diagnostic_url});
      pollLater(id,Math.min(15000,2000*2**Math.min(failures-1,3)));
    }
  }
  async function buildMesh() {
    let request;
    try{request=caseSettings();}catch(error){progress(error.message,'Check setup');return;}
    clearImportedResults(false);state.refreshFailures=0;state.meshProposal=null;state.meshProposalRequest=JSON.stringify(request);
    if($('mesh-proposal-apply'))$('mesh-proposal-apply').hidden=true;if($('mesh-proposal-note'))$('mesh-proposal-note').hidden=true;
    selectResultPane('scene');window.LudwigAeroLayout?.closeMessage();
    window.LudwigAeroLayout?.setMeshBuildProgress({active:true,status:'creating',percent:0,stage:'Preparing mesh',message:'Submitting geometry and mesh settings',log:['Submitting geometry and mesh settings'],jobId:'submitting'});
    clearTimeout(state.pollTimer);clearPersistedJob();state.jobId=null;state.job={status:'meshing'};state.amr=null;state.amrUrl=null;state.surface=null;state.lastSurfaceKey=null;state.cutRenderKey=null;state.draftDirty=false;state.activeConfig=request;
    state.mode='mesh';syncViewButtons();queueRender();updateActions();progress('Submitting the model and lattice settings…','Meshing');
    $('job-progress').removeAttribute('value');if($('job-progress-label'))$('job-progress-label').textContent='Creating mesh…';
    try{
      const result=await api('/api/aero/mesh',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});
      if(!result.job_id)throw new Error('Service did not return a job ID.');
      state.jobId=String(result.job_id);persistJob(state.jobId);renderJob(result);pollJob(state.jobId);
    }catch(error){state.job={status:'failed'};$('job-progress').value=0;window.LudwigAeroLayout?.setMeshBuildProgress({active:false,status:'error',message:error.message,log:[error.message],jobId:state.jobId||'submitting'});progress(`Lattice build failed: ${error.message}`,'Build failed');if(error.details?.mesh_proposal)showMeshProposal(error.details.mesh_proposal,error.message);updateActions();}
  }
  function showMeshProposal(proposal,reason){
    state.meshProposal=proposal;
    const button=$('mesh-proposal-apply'),note=$('mesh-proposal-note');
    const dx=Number(proposal.suggested_surface_dx),levels=Number(proposal.suggested_levels);
    if(note){note.hidden=false;note.textContent=proposal.message||(dx>0?`Proposed finest cell size ${dx.toPrecision(4)} m, ${levels} AMR levels.`:'No feasible mesh found within the selected limits.');}
    if(button){button.hidden=proposal.available===false||!(dx>0&&Number.isInteger(levels));button.disabled=false;button.textContent=`Use ${dx.toPrecision(4)} m${Number.isInteger(levels)?`, ${levels} levels`:''} and create mesh`;}
    window.LudwigAeroLayout?.showMessage(reason||proposal.message,{key:`mesh-proposal:${state.jobId||'draft'}:${dx}:${levels}`,urgent:true,title:'Proposed mesh size',recovery:true});
  }
  function applyMeshProposal(){
    const proposal=state.meshProposal;if(!proposal)return;
    if(state.meshProposalRequest&&JSON.stringify(caseSettings())!==state.meshProposalRequest){
      state.meshProposal=null;if($('mesh-proposal-apply'))$('mesh-proposal-apply').hidden=true;
      window.LudwigAeroLayout?.showMessage('The setup changed since this proposal. Create the mesh again to calculate a current proposal.',{urgent:true,title:'Mesh proposal needs updating'});return;
    }
    $('surface-dx').value=String(proposal.suggested_surface_dx);if(Number.isInteger(Number(proposal.suggested_levels)))$('levels').value=String(proposal.suggested_levels);
    state.meshProposal=null;window.LudwigAeroLayout?.closeMessage();invalidateMesh();buildMesh();
  }
  async function runSolver() {
    if(!canRun())return;const folder=window.LudwigOutputControls?.read()?.folder?.trim();if(!folder){window.LudwigAeroLayout?.selectParameters('outputs','output-folder');window.LudwigAeroLayout?.showMessage('Choose an output folder before running. Each analysis will get a dated study subfolder containing its study and results.',{urgent:true,title:'Output folder required'});$('output-folder')?.focus();return;}const id=state.jobId;state.restoredJobId=null;$('run-solver').disabled=true;
    clearImportedResults(false);state.autoSurfaceDisplay=state.liveResults!==false;progress('Starting the selected LBM solver…','Starting');
    try{const options={reference:referenceTools?.read(),study_name:$('study-name')?.value||'Aero study',study_snapshot:{name:$('study-name')?.value||'Aero study',view:window.LudwigAeroStudiesContext?.getStudyView?.()||{},source_path:state.studySource?.path||null},cuts:state.cuts.map(c=>({axis:c.axis,position:Number(c.position)})),outputs:window.LudwigOutputControls?.read()||{},display_every:window.LudwigOutputControls?.displayEvery()||100,target_vram_percent:inputNumber('target-vram','percentage of available VRAM',{min:1,max:100}),geometry_handling:$('geometry-handling')?.value||'closed_solid'};
      const job=await api(`/api/aero/jobs/${encodeURIComponent(id)}/run`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(options)});renderJob(job);pollJob(id);}
    catch(error){progress(`Solver could not start: ${error.message}`,'Run refused');updateActions();}
  }
  async function cancelJob() {
    if(!state.jobId||$('cancel-job').disabled)return;
    const id=state.jobId;$('cancel-job').disabled=true;
    try{const job=await api(`/api/aero/jobs/${encodeURIComponent(id)}/cancel`,{method:'POST'});renderJob(job);pollJob(id);}
    catch(error){progress(`Cancellation failed: ${error.message}`,'Cancel failed');updateActions();}
  }
  function syncViewButtons() {
    for(const b of document.querySelectorAll('.view-mode'))b.classList.toggle('active',b.dataset.view===state.mode);
    if($('view-mode'))$('view-mode').value=state.mode;
    const option=Array.from($('view-mode')?.options||[]).find(item=>item.value==='surface');
    if(option){const available=Boolean(state.surface||state.importedSurface||state.job?.artifacts?.surface_latest);option.hidden=!available;option.disabled=!available;}
  }
  async function selectViewMode(mode){
    if(mode==='surface'){
      try{await showSurfaceResults();}catch(error){window.LudwigAeroLayout?.showMessage(error.message,{title:'Surface results',urgent:true});}
      syncViewButtons();return;
    }
    state.mode=['geometry','mesh'].includes(mode)?mode:'geometry';syncViewButtons();updateResultViewStatus();queueRender();
  }
  function selectResultPane(value) {
    const pane=['scene','cuts','case'].includes(value)?value:'scene';
    for(const key of ['scene','cuts','case'])if($('panel-'+key))$('panel-'+key).hidden=key!==pane;
    if($('result-pane'))$('result-pane').value=pane;
    updateResultViewStatus();if(pane==='scene')requestAnimationFrame(()=>state.engine?.resize());
  }
  function applyThemePreference() {
    let saved='dark';try{saved=window.localStorage.getItem('airfoil-theme')||'dark';}catch{}
    const select=$('theme-mode'),theme=['auto','light','dark'].includes(select?.value)?select.value:saved;
    if(select)select.value=theme;
    const dark=theme==='dark'||(theme==='auto'&&window.matchMedia?.('(prefers-color-scheme: dark)').matches);
    document.documentElement.dataset.theme=dark?'dark':'light';
    if(state.scene)state.scene.clearColor=dark?new BABYLON.Color4(14/255,19/255,26/255,1):new BABYLON.Color4(234/255,240/255,246/255,1);
  }
  function recordingChanged(){
    const seen=new Set();state.cuts=state.cuts.filter(cut=>{const key=`${cut.axis}:${Number(cut.position).toFixed(8)}`;if(seen.has(key))return false;seen.add(key);return true;});
    renderCutRows();updateActions();queueRender();
    if($('section-note')){$('section-note').hidden=false;$('section-note').textContent='Recorded planes are written during the next analysis. Existing fields remain available. To inspect a new position now, enable volume VTK for a future run or choose an existing recorded plane.';}
  }
  function renderCutRows() {
    const list=$('cut-list');list.replaceChildren();
    if(!state.cuts.length){const p=document.createElement('span');p.className='empty-note';p.textContent='Default X, Y and Z midplanes will be recorded.';list.appendChild(p);return;}
    state.cuts.forEach((cut,index)=>{
      const row=document.createElement('div');row.className='cut-row';
      const axis=document.createElement('select');axis.setAttribute('aria-label','Cut plane axis');
      for(const a of ['x','y','z']){const o=document.createElement('option');o.value=a;
        const ground=$('boundary').value==='ground';o.textContent=a.toUpperCase()+' plane · '+({x:'flow',y:ground?'vertical':'lateral',z:ground?'lateral':'vertical'}[a]);axis.appendChild(o);}axis.value=cut.axis;
      const position=document.createElement('input');position.type='number';position.step='any';position.value=String(cut.position);position.setAttribute('aria-label','Cut plane location in metres');position.title='Plane position in solver-domain metres. This recording plan applies to the next run.';
      const remove=document.createElement('button');remove.type='button';remove.setAttribute('aria-label','Remove cut plane');remove.textContent='×';
      axis.addEventListener('change',()=>{cut.axis=axis.value;recordingChanged();});
      position.addEventListener('input',()=>{cut.position=position.value;recordingChanged();});
      remove.addEventListener('click',()=>{state.cuts.splice(index,1);recordingChanged();});
      row.append(axis,position,remove);list.appendChild(row);
    });
  }
  function boundaryNote() {
    const b=$('boundary').value,center=$('symmetry-auto-center-y'),placement=symmetryPlacement(),automatic=b==='symmetry'&&placement!=='none';
    $('move-y').disabled=automatic;$('move-y').title=automatic?'Y placement follows the selected automatic symmetry alignment. Choose Leave model where it is to move it manually.':'Manual lateral offset from the retained model placement';
    if(center){center.disabled=b!=='symmetry';const row=center.closest?.('label');if(row)row.hidden=b!=='symmetry';}
    $('boundary-note').textContent=b==='free_air'?($('solver').value==='pond_thermal'?'Free air: X flow, Y lateral, Z up. All outer faces use characteristic freestream/outflow boundaries. Increase clearances to assess far-field sensitivity.':'Free air: X flow, Y lateral, Z up. The distant top and bottom faces use slip boundaries; increase vertical clearance if they affect the flow.'):
      b==='ground'?'Ground: Aero Lab maps source Z-up to solver Y-up. The lowest model point touches ground at zero vertical offset; positive Z offset lifts it. Cut positions use solver coordinates.':
      placement==='none'?'Symmetry: the current model position is retained, with no automatic Y alignment. Only Y ≥ 0 is solved. Use the Y offset to place the model manually.':placement==='center'?'Symmetry: the model Y centre is placed on Y = 0. The solver uses the positive half and mirrors the symmetry boundary.':'Symmetry: the minimum Y edge of the prepared half model is placed at Y = 0.';
  }
  function syncFlowAngles(){const symmetry=$('boundary').value==='symmetry';for(const id of ['flow-beta','reference-y'])if($(id)){$(id).disabled=symmetry;if(symmetry)$(id).value='0';}}
  function retainCurrentSymmetryPosition() {
    if(state.frame?.bodyOrigin&&$('symmetry-placement-offset-y')){
      // A ground frame maps source (X,Y,Z) to solver (X,Z,-Y).
      const lateral=state.frame.ground?-state.frame.bodyOrigin[2]:state.frame.bodyOrigin[1];
      $('symmetry-placement-offset-y').value=String(lateral-Number($('move-y').value||0));
    }
  }
  function applySymmetryPlacement() {
    if($('boundary').value!=='symmetry')return;
    if(symmetryPlacement()==='none')retainCurrentSymmetryPosition();else $('move-y').value='0';
    if($('symmetry-auto-center-y'))$('symmetry-auto-center-y').checked=symmetryPlacement()==='center';
    window.LudwigAeroLayout?.syncModelControls?.();boundaryNote();invalidateMesh();
  }
  function syncSolverLattice(origin='solver') {
    const sel=$('solver'),latticeSelect=$('lattice');
    const available=value=>Array.from(sel.options).some(option=>option.value===value&&option.dataset.available==='true');
    if(origin==='lattice'){
      if(latticeSelect.value==='d3q19'&&available('q19_octree'))sel.value='q19_octree';
      else if(sel.value==='q19_octree'||sel.value==='pond_thermal'||latticeSelect.value==='d3q39'){
        const next=Array.from(sel.options).find(option=>option.dataset.available==='true'&&!['q19_octree','pond_thermal'].includes(option.value)&&String(option.dataset.lattices||'d3q27').split(',').includes(latticeSelect.value));
        sel.value=next?.value||'';
      }
    }
    const thermal=sel.value==='pond_thermal',octree=sel.value==='q19_octree';
    if(thermal)$('lattice').value='d3q27';
    if(octree)$('lattice').value='d3q19';
    else if($('lattice').value==='d3q19')$('lattice').value='d3q27';   // D3Q19 belongs to the CPU octree solver only
    const lattice=$('lattice').value,forced=thermal||octree;
    $('lattice').disabled=false;
    $('lattice').querySelector('option[value="d3q19"]').disabled=!available('q19_octree');
    $('lattice-label').textContent=thermal?'Mesh layout':'LBM lattice / collision';
    $('lattice').querySelector('option[value="d3q27"]').textContent=thermal?'Cartesian AMR · thermal finite volume':'D3Q27 + cumulant';
    const levels=$('levels');
    for(const option of levels.options)option.disabled=!octree&&Number(option.value)>7;
    if(!octree&&Number(levels.value)>7)levels.value='7';
    levels.title=octree?'Choose 1 to 10 octree levels; inspect the built mesh and RAM estimate before running.':'Choose 1 to 7 AMR levels; inspect the built lattice and GPU memory estimate before running.';
    for(const option of sel.options){
      if(!option.value)continue;
      const own=['pond_thermal','q19_octree'].includes(option.value);
      option.disabled=option.dataset.available!=='true'||(!forced&&!own&&!String(option.dataset.lattices||'d3q27').split(',').includes(lattice));
    }
    if(sel.selectedOptions[0]?.disabled){const first=Array.from(sel.options).find(o=>o.value&&!o.disabled);sel.value=first?.value||'';}
    const octreeControls=$('octree-controls');if(octreeControls)octreeControls.hidden=!octree;
    const triangleEdge=$('surface-triangle-max-edge-mm');if(triangleEdge){triangleEdge.disabled=octree;const help=triangleEdge.parentElement?.querySelector('small');if(help)help.textContent=octree?'D3Q19 retains the original geometry. Native wall samples resolve pressure at the local fluid-cell size; change Surface cell size under Mesh to change resolution.':'Default 50 mm. Applied when creating a new mesh. Smaller triangles sample surface fields more closely; they do not add fluid cells.';}
    const sizing=$('mesh-sizing');if(octree){sizing.value='manual';}sizing.disabled=octree;$('target-vram').disabled=octree;
    if($('metric-memory-label'))$('metric-memory-label').textContent=octree?'RAM':'VRAM';
    window.LudwigOutputControls?.setCapabilities({aero:{slice_png:!octree,notices:octree?['CPU octree exports native field planes and VTK. Solver field PNG images are unavailable; use the recorded-plane viewer or Makie post-processing.']:[]}},['aero']);
    $('lattice-title').textContent=thermal?'Thermal kinetic FV · experimental':octree?'D3Q19 + regularised · CPU octree (development)':lattice==='d3q39'?'D3Q39 + entropic':'D3Q27 + cumulant';
    $('lattice-note').textContent=thermal?'Conservative mass, momentum and total-energy fluxes on the same Cartesian AMR cells. Mach 0.05–1.5; inviscid slip or experimental laminar no-slip walls. No turbulence model or established aircraft validation.':
      octree?'Athermal low-Mach flow (Mach at most 0.3) on a cell-level octree that follows the surface to within two cells, CPU only: no GPU is needed and any platform serves. Up to 10 levels. Mesh creation, CPU runs and saved result viewing are available. Development; not validated.':
      lattice==='d3q39'?
      'Compressible flow, Mach at most 1.6. Uses the main solver with up to 7 AMR levels, thermal energy, and a conservative GPU memory allowance. Development path; check mesh, AMR interfaces, and convergence.':
      'Incompressible regime, Mach at most 0.6. Both solver sources support this web workflow.';
    syncThermalFlow();boundaryNote();
  }
  function thermalFlowSettings() {
    if($('solver').value!=='pond_thermal')return null;
    const input_mode=$('flow-input-mode').value,temperature_k=inputNumber('flow-temperature','freestream temperature',{min:100,max:2000}),gamma=1.4,gas_constant=287.05;
    const speed_of_sound=Math.sqrt(gamma*gas_constant*temperature_k);
    const velocity=input_mode==='mach'?inputNumber('flow-mach','freestream Mach',{min:.05,max:1.5})*speed_of_sound:inputNumber('velocity','flow speed',{min:.01,max:10000});
    const mach=velocity/speed_of_sound;if(mach<.05||mach>1.5)throw new Error('Thermal kinetic FV requires freestream Mach from 0.05 to 1.5.');
    return {input_mode,temperature_k,gamma,gas_constant,speed_of_sound,velocity,mach,cfl:inputNumber('flow-cfl','CFL number',{min:.01,max:.4}),wall_model:$('flow-wall-model').value};
  }
  function syncThermalFlow() {
    const thermal=$('solver').value==='pond_thermal',machInput=$('flow-input-mode').value==='mach';
    $('thermal-flow-controls').hidden=!thermal;
    $('velocity').disabled=thermal&&machInput;$('velocity').max=thermal?'10000':'1000';$('flow-mach').disabled=!machInput;
    $('sound-speed').disabled=thermal;$('viscosity').disabled=thermal&&$('flow-wall-model').value==='slip';
    if(!thermal)return;
    try{const flow=thermalFlowSettings();$('sound-speed').value=flow.speed_of_sound;
      if(machInput)$('velocity').value=flow.velocity;else $('flow-mach').value=flow.mach;
      $('thermal-flow-summary').textContent=`Mach ${flow.mach.toFixed(4)} · U∞ ${flow.velocity.toFixed(3)} m/s · a∞ ${flow.speed_of_sound.toFixed(3)} m/s. ${flow.wall_model==='slip'?'Inviscid: molecular viscosity is not applied.':'Molecular viscosity only; no turbulence closure.'}`;
    }catch(error){$('thermal-flow-summary').textContent=error.message;}
  }
  function updateVramTarget(gpu=state.capabilities?.gpu) {
    const node=$('target-vram-resolved');if(!node)return;
    const percent=Number($('target-vram').value),free=Number(gpu?.free_gb),budget=Number(gpu?.budget_gb);
    node.textContent=free>0&&percent>0?`${percent}% of ${free.toFixed(2)} GB available = ${(free*percent/100).toFixed(2)} GB.${budget>0&&free*percent/100>budget?` Safe run allowance currently caps this at ${budget.toFixed(2)} GB.`:''} Rechecked before running.`:
      'Percentage of free VRAM measured at mesh sizing and checked again before running.';
  }
  function showGeometryAudit(audit) {
    const node=$('geometry-audit');if(!node)return;
    state.geometryAudit=audit;node.textContent='';node.hidden=true;updateLabels();
  }
  async function loadCapabilities() {
    try{
      const data=await api('/api/aero/capabilities');state.capabilities=data;
      state.outdatedServer=data.features?.studies!==true||Number(data.features?.max_amr_levels)<7;
      const sel=$('solver');sel.replaceChildren();
      for(const solver of data.solvers||[]){const o=document.createElement('option');o.value=solver.id;
        o.dataset.available=String(Boolean(solver.available));o.dataset.lattices=(solver.web_lattices||['d3q27']).join(',');
        o.textContent=solver.label+(solver.available?'':' · unavailable');o.disabled=!solver.available;
        if(solver.reason)o.title=solver.reason;sel.appendChild(o);}
      syncSolverLattice();
      const first=Array.from(sel.options).find(o=>!o.disabled);if(first)sel.value=first.value;
      else{const o=document.createElement('option');o.value='';o.textContent='No available solver';sel.appendChild(o);sel.value='';}
      const budget=data.gpu?.budget_gb??data.gpu?.free_gb;
      $('metric-budget').textContent=finite(budget)?`${Number(budget).toFixed(1)} GB`:'Unknown';
      $('connection').textContent=state.outdatedServer?'Older server · relaunch workbench':'Solver service online · AMR 1–7';$('connection').className=state.outdatedServer?'connection error':'connection online';
      if(state.outdatedServer)progress('This tab is connected to an older server. Run the Ludwig Workbench launcher again to open the current app; this server cannot use the new setup controls.','Update available');
      updateVramTarget();updateActions();
    }catch(error){$('connection').textContent='Solver service unavailable';$('connection').className='connection error';
      progress(`Aero Lab service is unavailable: ${error.message}`,'Offline');}
  }
  function wireEvents() {
    $('model-file').addEventListener('change',event=>importModel(event.target.files?.[0]));
    for(const id of ['scale-x','scale-y','scale-z','rotate-x','rotate-y','rotate-z','move-x','move-y','move-z',
      'upstream','downstream','lateral','height','domain-ratio-x','domain-ratio-y','domain-ratio-z','surface-dx','levels','velocity','density','viscosity','sound-speed','steps','frames'])
      $(id).addEventListener('input',invalidateMesh);
    $('surface-triangle-max-edge-mm')?.addEventListener('input',invalidateMesh);
    $('domain-x-offset')?.addEventListener('input',()=>{if(state.frame?.body&&state.domainCameraAnchor===undefined)state.domainCameraAnchor=state.frame.body.center[0];invalidateMesh();});
    $('symmetry-placement')?.addEventListener('change',applySymmetryPlacement);
    for(const id of ['octree-band-fine','octree-band','octree-wake','octree-wake-offset','octree-threads','octree-u-lattice','octree-checkpoint-keep','octree-ramp-steps','octree-plane-samples'])$(id)?.addEventListener('input',invalidateMesh);
    $('octree-seal-cavities')?.addEventListener('change',invalidateMesh);
    $('lattice').addEventListener('change',()=>{syncSolverLattice('lattice');invalidateMesh();});
    $('solver').addEventListener('change',()=>{syncSolverLattice();invalidateMesh();});
    for(const id of ['flow-input-mode','flow-mach','flow-temperature','flow-cfl','flow-wall-model'])$(id).addEventListener('input',()=>{syncThermalFlow();invalidateMesh();});
    $('velocity').addEventListener('input',syncThermalFlow);
    $('domain-sizing')?.addEventListener('change',()=>{boundaryNote();invalidateMesh();});
    $('mesh-sizing').addEventListener('change',()=>{updateVramTarget();invalidateMesh();});
    $('target-vram').addEventListener('input',()=>{updateVramTarget(state.job?.estimate?.gpu);
      if($('mesh-sizing').value==='manual'){if(state.job)renderJob(state.job);else{updateBudgetGuard();updateActions();}}
      else{updateBudgetGuard();invalidateMesh();}});
    $('output-controls')?.addEventListener('input',updateActions);
    $('output-controls')?.addEventListener('change',updateActions);
    $('boundary').addEventListener('change',()=>{if($('boundary').value==='symmetry'){if(symmetryPlacement()==='none')retainCurrentSymmetryPosition();else $('move-y').value='0';}boundaryNote();renderCutRows();invalidateMesh();});
    $('symmetry-auto-center-y')?.addEventListener('change',()=>{if($('symmetry-placement'))$('symmetry-placement').value=$('symmetry-auto-center-y').checked?'center':'wall';applySymmetryPlacement();});
    $('model-projections')?.addEventListener('change',queueRender);
    syncFlowAngles();$('boundary').addEventListener('change',syncFlowAngles);for(const id of ['flow-alpha','flow-beta'])$(id)?.addEventListener('change',invalidateMesh);
    $('mesh-visibility')?.addEventListener('click',()=>{state.meshVisible=state.meshVisible===false;inspector?.setMeshVisible?.(state.meshVisible);$('mesh-visibility').setAttribute('aria-pressed',String(state.meshVisible));$('mesh-visibility').textContent=state.meshVisible?'Hide mesh':'Show mesh';queueRender();});
    $('live-results')?.addEventListener('change',()=>{state.liveResults=$('live-results').checked;state.surfaceFollowLive=true;state.cutFollowLive=true;window.LudwigAeroLayout?.syncLiveResults?.();if(state.liveResults)loadLatestResults();});
    $('results-refresh')?.addEventListener('click',()=>loadLatestResults(true));
    $('results-load-current')?.addEventListener('click',loadCurrentResults);
    $('mesh-proposal-apply')?.addEventListener('click',()=>{try{applyMeshProposal();}catch(error){progress(error.message,'Check setup');}});
    $('results-browse')?.addEventListener('click',async()=>{const button=$('results-browse');button.disabled=true;try{const path=await window.LudwigOutputControls.chooseFolder({path:$('results-path').value,title:'Choose results folder'});if(path)$('results-path').value=path;}catch(error){window.LudwigAeroLayout?.showMessage(error.message,{urgent:true,title:'Folder selection'});}finally{button.disabled=false;}});
    $('reset-transform').addEventListener('click',()=>{for(const a of ['x','y','z']){$('scale-'+a).value='1';$('rotate-'+a).value='0';$('move-'+a).value='0';}invalidateMesh();});
    $('geometry-handling')?.addEventListener('change',()=>{updateActions();renderMeshDetails(state.job);updateLabels();});
    $('model-appearance')?.addEventListener('change',()=>{if($('model-appearance').value==='glb'){state.mode='geometry';syncViewButtons();selectResultPane('scene');}queueRender();});
    $('model-opacity')?.addEventListener('input',()=>{$('model-opacity-value').value=$('model-opacity').value;queueRender();});$('model-widget')?.addEventListener('change',queueRender);
    $('add-cut').addEventListener('click',()=>{if(state.cuts.length>=6)return;state.cuts.push({axis:'y',position:0.5});recordingChanged();});
    $('build-mesh').addEventListener('click',buildMesh);$('run-solver').addEventListener('click',runSolver);$('cancel-job').addEventListener('click',cancelJob);
    for(const button of document.querySelectorAll('.view-mode'))button.addEventListener('click',()=>selectViewMode(button.dataset.view));
    $('view-mode')?.addEventListener('change',()=>selectViewMode($('view-mode').value));
    $('display-aids')?.addEventListener('change',queueRender);
    $('velocity-vectors')?.addEventListener('change',()=>{if(state.vectorError)state.vectorKey=null;queueRender();});
    $('camera-view')?.addEventListener('change',()=>cameraPreset($('camera-view').value));
    $('result-pane')?.addEventListener('change',()=>selectResultPane($('result-pane').value));
    $('theme-mode')?.addEventListener('change',()=>{try{window.localStorage.setItem('airfoil-theme',$('theme-mode').value);}catch{}applyThemePreference();});
    const colorScheme=window.matchMedia?.('(prefers-color-scheme: dark)');colorScheme?.addEventListener?.('change',applyThemePreference);
    $('viewer-fullscreen')?.addEventListener('click',async()=>{
      const panel=$('viewer').parentElement;
      try{if(document.fullscreenElement)await document.exitFullscreen();else await panel.requestFullscreen();state.engine?.resize();}
      catch(error){console.warn('Fullscreen is unavailable:',error);}
    });
    document.addEventListener('fullscreenchange',()=>{
      state.engine?.resize();const button=$('viewer-fullscreen');if(button){const label=document.fullscreenElement?'Exit full screen':'Full screen';button.textContent=document.fullscreenElement?'↙':'⛶';button.setAttribute('aria-label',label);button.title=label;}
    });
    $('surface-field').addEventListener('change',()=>{showSurfaceResults().catch(error=>progress(error.message,'Surface results'));});
    $('surface-representation')?.addEventListener('change',async()=>{state.surfaceFollowLive=true;try{const selected=selectedSurfaceRecord(state.job);if(!selected)throw new Error('This study has no saved fields of that type. The last loaded surface is retained.');await maybeLoadSurface(state.job,true);await showSurfaceResults();}catch(error){window.LudwigAeroLayout?.showMessage(error.message,{title:'Surface sampling',urgent:true});}});
    $('surface-range-mode')?.addEventListener('change',queueRender);
    $('cut-result-select').addEventListener('change',()=>{state.selectedCut=$('cut-result-select').value;state.cutFollowLive=false;state.cutRenderKey=null;const cut=state.job?.artifacts?.cuts?.[Number(state.selectedCut)];state.selectedCutRecord=cut?.url||cut?.json_url;alignInspectorToRecorded(cut);renderCut(state.job,true);queueRender();});
    $('cut-field').addEventListener('change',()=>{state.cutRenderKey=null;renderCut(state.job,true);});
    $('fit-model').addEventListener('click',()=>{const frame=currentFrame();fit(frame.body||frame.domain);});
    $('fit-domain').addEventListener('click',fitDisplayedDomain);
    wireRecordedCutZoom();
  }
  if(window.LudwigAeroReference)referenceTools=window.LudwigAeroReference.create({state,$,queueRender,currentFrame,addArrow,addLabel,addLines,keep,v3});
  for(const id of ['scale-apply','scale-auto','scale-target','scale-min-slider','scale-max-slider'])$(id)?.addEventListener(id.endsWith('slider')?'input':id==='scale-target'?'change':'click',()=>{if(state.lastNumericCut&&!$('cut-canvas').hidden)paintNumericCut(state.lastNumericCut);});
  if(window.LudwigAeroInspector)inspector=window.LudwigAeroInspector.create({state,$,v3,keep,addQuad,addLines,addLabel,queueRender,currentFrame,scalarColor,scalarRange,api,apiUrl,renderCutRows,invalidateMesh:recordingChanged,selectResultPane,clearCameraMotion,referenceTools});
  window.LudwigAeroResultsContext={state,api,queueRender,applyImportedResults,clearImportedResults,setImportedField,showSurfaceResults,prepareReportView,getRenderIdentity:()=>state.renderIdentity?{...state.renderIdentity}:null,
    getPlacement:()=>({mesh_offset:state.amr?.mesh_offset||[0,0,0],case:state.activeConfig,job_id:state.jobId})};
  window.LudwigAeroStudiesContext={state,caseSettings,attachStudyResults,hydrateCase,loadStoredModel,clearPersistedJob,queueRender,updateActions,showGeometryAudit,progress,clearImportedResults,
    getStudyView(){const view={model_opacity:Number($('model-opacity')?.value??100),reference_display:referenceTools?.capture(),live_results:state.liveResults,mesh_visible:state.meshVisible,...(inspector?.getView()||{}),mode:state.mode,selections:{},imported_results:window.LudwigAeroResults?.getReference?.()||null};for(const id of ['display-aids','velocity-vectors','camera-view','surface-field','surface-representation','surface-range-mode','results-palette','model-appearance','model-widget','model-projections'])if($(id))view.selections[id]=$(id).value;if(state.camera)view.camera={alpha:state.camera.alpha,beta:state.camera.beta,radius:state.camera.radius,target:state.camera.target.asArray()};return view;},
    restoreStudyView(view){if($('surface-representation'))$('surface-representation').value='auto';const opacity=Number(view?.model_opacity??100);$('model-opacity').value=String(Number.isFinite(opacity)?Math.max(0,Math.min(100,opacity)):100);$('model-opacity-value').value=$('model-opacity').value;referenceTools?.restoreView(view?.reference_display);clearImportedResults(false);state.liveResults=view?.live_results!==false;if($('live-results'))$('live-results').checked=state.liveResults;window.LudwigAeroLayout?.syncLiveResults?.();window.LudwigAeroResults?.restoreReference?.(view?.imported_results);inspector?.restoreView(view);state.mode=['geometry','mesh','surface'].includes(view?.mode)?view.mode:'geometry';syncViewButtons();for(const [id,value] of Object.entries(view?.selections||{})){const select=$(id);if(select&&(id==='model-widget'?['none','move','rotate'].includes(value):Array.from(select.options||[]).some(o=>o.value===value)))select.value=value;}const c=view?.camera;state.pendingStudyCamera=null;if(c&&Array.isArray(c.target)&&c.target.length===3&&[c.alpha,c.beta,c.radius,...c.target].every(Number.isFinite)&&c.radius>0)state.pendingStudyCamera=c;window.LudwigAeroLayout?.syncModelControls?.();updateActions();updateResultsAvailability();queueRender();}};
  if(window.LudwigAeroResultBrowser)resultBrowser=window.LudwigAeroResultBrowser.create({$,isLive:()=>state.liveResults!==false,onOpenSurface:viewSurfaceRecord,onOpenCut:viewCutRecord,onOpenVolume:viewVolumeRecord,onStatus:(text,error)=>{if(error)window.LudwigAeroLayout?.showMessage(text,{urgent:true,title:'Result display'});else progress(text,'Results');}});
  window.addEventListener('ludwig-amr-visibility-change',()=>{if(state.lastNumericCut&&!$('cut-canvas').hidden)paintNumericCut(state.lastNumericCut);});
  window.addEventListener('ludwig-palette-change',()=>{state.cutRenderKey=null;if(state.lastNumericCut&&!$('cut-canvas').hidden)paintNumericCut(state.lastNumericCut);queueRender();});
  initScene();wireEvents();boundaryNote();syncSolverLattice();renderCutRows();
  try{if($('theme-mode'))$('theme-mode').value=window.localStorage.getItem('airfoil-theme')||'dark';}catch{}
  applyThemePreference();selectResultPane('scene');loadCapabilities().then(resumeStoredJob);updateActions();queueRender();
})();
