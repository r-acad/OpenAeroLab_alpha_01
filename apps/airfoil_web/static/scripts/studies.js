"use strict";

window.AirfoilStudies = (() => {
  const $ = id => document.getElementById(id);
  let path=null, etag=null, resultLink=null, dirty=false, busy=false, restoring=false, edits=0;
  const controls=()=>[...document.querySelectorAll('.setup-column input[id],.setup-column select[id],#cfd-te-topology,#preview-resolution')]
    .filter(node=>!['file','button','submit'].includes(node.type)&&!['recent-select','airfoil-search','benchmark-select'].includes(node.id));
  function captureControls() { return Object.fromEntries(controls().map(node=>[node.id,node.type==='checkbox'?node.checked:node.value])); }
  const defaults=captureControls();
  function restoreControls(saved) {
    for(const node of controls()) {
      const value=saved[node.id];
      if(node.type==='checkbox'&&typeof value==='boolean')node.checked=value;
      else if(typeof value==='string'&&(node.tagName!=='SELECT'||[...node.options].some(option=>option.value===value)))node.value=value;
    }
  }
  async function api(action,value) {
    const response=await fetch('/api/airfoil/studies/'+action,value===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});
    let data;try{data=await response.json();}catch{throw new Error('Study service unavailable. Restart the updated Julia service.');}
    if(!response.ok||data.ok===false)throw new Error(data.error||'Study action failed');
    return data;
  }
  function status(message='') {
    $('study-name').textContent=(path?path.split(/[\\/]/).pop():'Untitled study')+(dirty?' *':'');
    $('study-name').title=path||'Save study to choose a folder and filename';
    if(message)$('study-status').textContent=message;
    $('study-load-results').disabled=!resultLink;
    for(const id of ['study-new','study-open','study-save','study-save-as'])$(id).disabled=busy;
  }
  function markDirty() { if(restoring)return;dirty=true;edits++;status('Unsaved changes'); }
  function closeMenus(){document.querySelectorAll('.top-menu[open]').forEach(node=>node.open=false);}
  function detach() {
    clearTimeout(state.pollTimer);clearTimeout(state.historyTimer);state.pollGeneration++;
    state.previewAbort?.abort();clearTimeout(state.previewTimer);state.previewGeneration++;
    state.cavitationReferenceSelection++;
    state.finishAnalysis?.();state.finishAnalysis=null;
    Object.assign(state,{job:null,result:null,analysisCase:null,analysisGeometryKey:null,analysisMeshKey:null,preview:null,
      previewReference:null,requestedAlpha:null,liveField:null,liveFieldLoading:null,liveAutoShown:false,meshKeys:'',embeddedAirfoil:null,studyFlowExtras:{}});
    state.fullMeshes.clear();state.meshLoading.clear();state.fullFields.clear();state.fieldLoading.clear();
    chartMemory.clear();spatialMemory.geometry=null;spatialMemory.field=null;
    resetLog();window.AirfoilBabylon?.hide();clearError();
    el['job-title'].textContent='Ready';el['job-message'].textContent='Choose the setup, then run an analysis.';
    el['job-id'].textContent='';el['job-percent'].textContent='0%';el['progress-bar'].style.width='0%';
    el['download-result'].disabled=true;
    renderSelection();renderPolars();renderTable();
  }
  async function allowReplace() {
    if(!dirty&&!['queued','running'].includes(state.job?.status))return true;
    const dialog=$('study-confirm');
    $('study-confirm-text').textContent=(dirty?'The current study has unsaved changes. ':'')+
      (['queued','running'].includes(state.job?.status)?'The running job will continue and can be reopened from saved results. ':'')+
      'Save first, discard changes, or cancel.';
    dialog.returnValue='cancel';dialog.showModal();
    const choice=await new Promise(resolve=>dialog.addEventListener('close',()=>resolve(dialog.returnValue),{once:true}));
    if(choice==='save')return save(false);
    return choice==='discard';
  }
  async function newStudy() {
    if(busy||!await allowReplace())return;
    restoring=true;
    try {
        detach();restoreControls(defaults);path=null;etag=null;resultLink=null;
        window.WingLegends?.restore(null);
      state.savedLowspeedTopology=null;state.machEdited=false;state.reynoldsEdited=false;
      updateSetup();scheduleGeometryPreview();await window.AirfoilBenchmarks.restore(null);
      switchTab('geometry');dirty=false;status('New study. Save to choose its location.');closeMenus();
    } finally {restoring=false;}
  }
  async function openPath(file) {
    if(busy||!await allowReplace())return false;
    busy=true;status();
    try {
      const data=await api('open',{path:file}), study=data.study;
      if(study.request.airfoil.kind==='cavitation_reference')await loadCavitationReferences();
      restoring=true;detach();restoreControls(defaults);
      // Source selectors must contain the embedded UIUC/cavitation option before
      // applying the snapshot. No network fetch is needed for UIUC geometry.
      applySavedRequestToForm(study.request);
      restoreControls(study.controls);
      const foil=study.request.airfoil;
      state.embeddedAirfoil=foil.kind==='uiuc'?{kind:foil.kind,id:foil.id,coordinates:{x:study.geometry.x,y:study.geometry.y}}:null;
      updateSetup();
      // Retain API-only numerical settings without resurrecting optional GUI
      // controls (e.g. a flap or trip after its checkbox is switched off).
      const represented=collectCase().flow;
      const managed=new Set(window.AirfoilManagedFlowKeys||[]);
      state.studyFlowExtras=Object.fromEntries(Object.entries(study.request.flow).filter(([key])=>!(key in represented)&&!managed.has(key)));
      path=data.path;etag=data.etag;resultLink=study.results;
      window.WingLegends?.restore(study.view?.color_scales);
        if(study.view?.color_scales?.palettes?.flow)window.LudwigPalettes?.set(study.view.color_scales.palettes.flow);
      await window.AirfoilBenchmarks.restore(study.view?.benchmark_id);
      scheduleGeometryPreview();
      let resultsOpened=false;
      if(data.results_available&&resultLink) {
        try {await openSavedJob(resultLink.job_id,{restoreForm:false,rethrow:true});resultsOpened=true;}
        catch(error) {data.message+=' · Setup loaded, but linked results could not be read: '+error.message;}
      }
      switchTab(['geometry','results','field','polar','diagnostics','log'].includes(study.view?.tab)?study.view.tab:resultLink?'results':'geometry');
      dirty=false;status(data.message+(resultsOpened?' · Linked results opened.':''));
      $('study-open-dialog').close();closeMenus();return true;
    } catch(error) {displayError(error);status(error.message);return false;}
    finally {restoring=false;busy=false;status();}
  }
  async function browseOpen() {
    if(busy)return;
    try {
      const selected=await api('choose-open',{path:path||'',title:'Open Airfoil Lab study'});
      if(!selected.cancelled)await openPath(selected.path);
    } catch(error){displayError(error);}
  }
  async function showOpen() {
    closeMenus();const dialog=$('study-open-dialog');dialog.showModal();
    const list=$('study-recent-list');list.textContent='Loading saved studies…';
    try {
      const data=await api('recent');list.replaceChildren();
      for(const item of data.items||[]) {
        const button=document.createElement('button'),name=document.createElement('strong'),location=document.createElement('small');
        button.type='button';name.textContent=item.name;location.textContent=item.path+(item.available?'':' (unavailable)');
        button.append(name,location);button.disabled=!item.available;button.title='Open this study and its available linked results';
        button.addEventListener('click',()=>openPath(item.path));list.append(button);
      }
      if(!list.children.length)list.textContent='No recent Airfoil studies. Browse to open an existing .ludwig-study file.';
    } catch(error){list.textContent=error.message;}
  }
  async function save(asNew=false) {
    if(busy)return false;
    busy=true;status();
    try {
      const request=collectCase(),controls=captureControls(),revision=edits;
      let destination=path,expected=etag;
      if(asNew||!destination) {
        const selected=await api('choose-save',{path:path||'',title:'Save Airfoil Lab study as'});
        if(selected.cancelled)return false;
        destination=selected.path;expected=selected.etag;
      }
      const link=state.job?.id?{job_id:state.job.id}:resultLink;
      const data=await api('save',{path:destination,etag:expected,request,controls,results:link,
        view:{tab:state.activeTab,benchmark_id:window.AirfoilBenchmarks.selectedId,color_scales:window.WingLegends?.capture()}});
      path=data.path;etag=data.etag;resultLink=data.study.results;dirty=edits!==revision;
      status(data.message+(dirty?' Newer edits still need saving.':''));closeMenus();return true;
    } catch(error){displayError(error);status(error.message);return false;}
    finally {busy=false;status();}
  }
  async function loadResults() {
    if(!resultLink)return;
    try {await openSavedJob(resultLink.job_id,{restoreForm:false,rethrow:true});switchTab('results');status('Linked results opened. Study setup preserved.');}
    catch(error){displayError(error);status('Linked results unavailable: '+error.message);}
  }
  for(const [id,fn] of [['study-new',newStudy],['study-open',showOpen],['study-save',()=>save()],['study-save-as',()=>save(true)],['study-browse-open',browseOpen],['study-load-results',loadResults]])
    $(id).addEventListener('click',fn);
  document.addEventListener('input',event=>{if(controls().includes(event.target))markDirty();});
  document.addEventListener('change',event=>{if(controls().includes(event.target))markDirty();});
  window.addEventListener('airfoil-job-updated',event=>{
    if(restoring||!event.detail?.id||event.detail.id===resultLink?.job_id)return;
    resultLink={job_id:event.detail.id};markDirty();status('New analysis linked. Save study to retain its results link.');
  });
  window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});
  document.addEventListener('keydown',event=>{
    if(!(event.ctrlKey||event.metaKey)||event.altKey)return;
    if(event.key.toLowerCase()==='s'){event.preventDefault();save(event.shiftKey);}
    if(event.key.toLowerCase()==='o'){event.preventDefault();showOpen();}
  });
  status('New study. Project → Save study stores its setup and source geometry.');
  return {save,openPath,newStudy,showOpen,loadResults,markDirty,captureControls,get path(){return path;},get dirty(){return dirty;}};
})();
