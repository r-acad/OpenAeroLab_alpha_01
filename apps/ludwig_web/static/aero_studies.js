/* Portable simulation studies. Geometry travels with the setup; large results stay linked. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  const ctx=window.LudwigAeroStudiesContext;
  if(!ctx||!$('study-save'))return;
  const MAX_GEOMETRY_BYTES=50_000_084,MAX_STUDY_BYTES=136_000_000;
  const activeStatuses=new Set(['meshing','sizing','mesh_sizing','waiting_for_gpu','running','queued','starting']);
  let busy=false,sourceTarget=null,recentDialog=null,recentList=null,recentHelp=null;
  function status(text,error=false){$('study-status').textContent=text;$('study-status').classList.toggle('warning',error);}
  function setBusy(value){busy=value;for(const id of ['study-save','study-save-menu','study-save-as','study-open'])if($(id))$(id).disabled=value;}
  function associate(target){
    sourceTarget=target;ctx.state.studySource=target;window.LudwigAeroPathUI?.refresh();
    for(const id of ['study-folder','study-filename'])if($(id)){$(id).readOnly=Boolean(target);$(id).title=target?'Associated study destination. Use Save study as to choose another file.':'Choose a destination on Save study as.';}
    if($('study-browse')){$('study-browse').disabled=Boolean(target);$('study-browse').title=target?'Use Save study as to change the associated file.':'Choose a study folder.';}
  }
  function activeCase(){return ctx.state.importing||ctx.state.uploading||ctx.state.resumePending||activeStatuses.has(ctx.state.job?.status);}
  function openAllowed(){
    if(ctx.state.outdatedServer){status('Relaunch Ludwig Workbench before opening a study.',true);return false;}
    if(!activeCase())return true;
    status('Wait for the active mesh, run or model upload to finish, or cancel it before opening another study.',true);return false;
  }
  async function request(path,body){
    const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const result=await response.json();if(!response.ok||result.ok===false){const error=new Error(result.error||`HTTP ${response.status}`);error.status=response.status;error.details=result;throw error;}
    return result;
  }
  function bytesBase64(bytes){
    const chunks=[];for(let start=0;start<bytes.length;start+=24_576)chunks.push(btoa(String.fromCharCode(...bytes.subarray(start,start+24_576))));
    return chunks.join('');
  }
  async function originalGeometry(file){
    if(!file)return null;
    if(file.size>MAX_GEOMETRY_BYTES)throw new Error('The original geometry exceeds the 50 MB study limit.');
    return {name:file.name,media_type:file.type||'application/octet-stream',encoding:'base64',data:bytesBase64(new Uint8Array(await file.arrayBuffer()))};
  }
  function sourceFile(document){
    const source=document.source_geometry;if(!source||source.encoding!=='base64'||typeof source.data!=='string')return null;
    const decoded=atob(source.data),bytes=new Uint8Array(decoded.length);
    for(let i=0;i<decoded.length;i++)bytes[i]=decoded.charCodeAt(i);
    return new File([bytes],source.name||'source_geometry',{type:source.media_type||'application/octet-stream'});
  }
  function renderReferences(results,resumeJob,historical=false){
    const host=$('study-results');host.replaceChildren();host.hidden=!results;if(!results)return;
    const label=document.createElement('p');label.textContent=historical?'Historical fields are loaded from a different saved setup. They do not represent the editable study.':results.compatible?'Saved surface and recorded fields are available for this model and flow.':'Saved results use a different model or flow setup, or are unavailable here.';host.appendChild(label);if(results.note&&!historical){const note=document.createElement('p');note.textContent=results.note;host.appendChild(note);}
    for(const item of Array.isArray(results.locations)?results.locations:[]){
      if(typeof item?.path!=='string')continue;
      const p=document.createElement('p');p.textContent=`${item.kind||'Results'}: ${item.path}${item.available?'':' (not verified here)'}`;host.appendChild(p);
    }
    if(resumeJob&&/^[0-9a-f-]{36}$/.test(resumeJob)){
      const link=document.createElement('a');link.href=`/aero-lab?job=${encodeURIComponent(resumeJob)}`;link.target='_blank';link.rel='noopener';link.textContent=historical?'Open historical result case':'Open matching result case';host.appendChild(link);
    }
  }
  async function save(saveAs=false){
    if(busy)return;if(ctx.state.outdatedServer){status('Relaunch Ludwig Workbench to use study saving with this version.',true);return;}setBusy(true);status('Preparing the study and embedded geometry…');
    try{
      const setup=ctx.caseSettings(),view=ctx.getStudyView?.()||{};
      const body={name:$('study-name').value.trim()||'Aero study',case:setup,view,job_id:ctx.state.jobId||null,results:ctx.state.studyResults||null};
      let target=saveAs?null:sourceTarget;
      let folder=$('study-folder')?.value?.trim(),fileName=$('study-filename')?.value?.trim();
      if(saveAs||(!target&&!folder)){
        const initial=sourceTarget?.path||(folder?folder.replace(/[\\/]$/,'')+'/'+(fileName||'Aero study.ludwig-study'):(fileName||'Aero study.ludwig-study'));
        const chosen=await request('/api/aero/studies/choose-save',{path:initial,title:'Save study as'});
        if(chosen.cancelled){status('Save cancelled.');return;}
        target={path:chosen.path,sha256:chosen.existing_sha256};folder=chosen.folder;fileName=chosen.file_name;
      }
      if(target){const normalized=target.path.replace(/\\/g,'/'),split=normalized.lastIndexOf('/');folder=normalized.slice(0,split);fileName=normalized.slice(split+1);if(target.sha256){body.overwrite=true;body.overwrite_sha256=target.sha256;}}
      if(folder)body.folder=folder;if(fileName)body.file_name=fileName;
      if(ctx.state.originalFile)body.source_geometry=await originalGeometry(ctx.state.originalFile);
      let saved;
      try{saved=await request('/api/aero/studies',body);}catch(error){
        if(error.status!==409||!window.confirm(`Replace the existing study file?\n${error.details.path}`))throw error;
        body.overwrite=true;body.overwrite_sha256=error.details.existing_sha256;saved=await request('/api/aero/studies',body);
      }
      if(!saved.saved_path){const link=document.createElement('a');link.href=saved.url;link.download=saved.file_name;document.body.appendChild(link);link.click();link.remove();}
      if(saved.saved_path){associate({path:saved.saved_path,sha256:saved.saved_sha256});if($('study-folder'))$('study-folder').value=saved.folder||folder;if($('study-filename'))$('study-filename').value=saved.file_name;}
      status(saved.saved_path?`Saved study to ${saved.saved_path}.`:`Downloaded ${saved.file_name}. A retained copy is stored at ${saved.storage_path}.`);
    }catch(error){status(`Study could not be saved: ${error.message}`,true);}finally{setBusy(false);}
  }
  async function open(file=null,recentId=null){
    if(busy)return;
    if(!openAllowed()){$('study-file').value='';return;}
    const initialModelId=ctx.state.modelId,initialJobId=ctx.state.jobId;
    setBusy(true);status(file?`Opening ${file.name}...`:'Choose a study file...');
    try{
      let document=null,loaded;
      if(file){
        if(file.size>MAX_STUDY_BYTES)throw new Error('Study exceeds the 136 MB file limit.');
        try{document=JSON.parse(await file.text());}catch{throw new Error('Choose a valid .ludwig-study file.');}
        if(activeCase())throw new Error('Wait for the active case before opening a study.');
        loaded=await request('/api/aero/studies/import',document);
      }else if(recentId){
        loaded=await request('/api/aero/studies/open-recent',{id:recentId});
      }else{
        loaded=await request('/api/aero/studies/open-native',{path:sourceTarget?.path||$('study-folder')?.value||'',title:'Open Ludwig study'});
        if(loaded.cancelled){status('Open cancelled.');return;}
      }
      const state=ctx.state;
      if(activeCase()||state.modelId!==initialModelId||state.jobId!==initialJobId)
        throw new Error('The current case changed while opening the study. Its controls have been preserved; open the study again when ready.');
      clearTimeout(state.pollTimer);ctx.clearPersistedJob();state.jobId=null;state.job=null;state.resumePending=false;
      ctx.clearImportedResults?.(false);
      state.source=null;state.sourceBounds=null;state.sourceModelId=null;state.amr=null;state.amrUrl=null;state.surface=null;
      state.lastSurfaceKey=null;state.cutRenderKey=null;state.vectorKey=null;state.vectorData=null;state.selectedCut=null;
      state.pendingStudyCamera=null;
      state.modelId=loaded.case.model_id;state.activeConfig=loaded.case;state.draftDirty=false;
      state.studyResults=loaded.results||null;
      state.originalFile=document?sourceFile(document):null;state.mode='geometry';
      ctx.hydrateCase(loaded.case);ctx.showGeometryAudit(loaded.model.geometry_audit);
      $('study-name').value=loaded.name;$('model-state').textContent=`${loaded.name} · ${loaded.model.triangles.toLocaleString()} triangles · study ready.`;
      associate(loaded.source_path?{path:loaded.source_path,sha256:loaded.source_sha256}:null);
      if(sourceTarget){const normalized=sourceTarget.path.replace(/\\/g,'/'),split=normalized.lastIndexOf('/');if($('study-folder'))$('study-folder').value=normalized.slice(0,split);if($('study-filename'))$('study-filename').value=normalized.slice(split+1);}else{if($('study-folder'))$('study-folder').value='';if($('study-filename'))$('study-filename').value=file?.name||loaded.file_name;}
      $('metric-triangles').textContent=loaded.model.triangles.toLocaleString();
      $('metric-cells').textContent='—';$('metric-memory').textContent='—';$('metric-fit').textContent='Create a mesh';
      $('case-link').hidden=true;$('artifact-links').replaceChildren();$('output-results').replaceChildren();
      $('job-progress').value=0;ctx.progress('Study loaded. Inspect the setup, then create a mesh to run it.','Study ready');
      ctx.updateActions();await ctx.loadStoredModel(state.modelId);
      ctx.restoreStudyView?.(loaded.view||{});ctx.queueRender();
      const resultId=loaded.job_id||loaded.results?.source_job_id;
      const matching=Boolean(loaded.job_id&&loaded.results?.compatible);let attached=false,resultError=null;
      if(typeof resultId==='string'&&/^[0-9a-f-]{36}$/.test(resultId)&&ctx.attachStudyResults){
        try{
          // attachStudyResults first GETs an existing local job. A reference in
          // an imported document cannot invent a result or start a solver.
          await ctx.attachStudyResults(resultId,{compatible:matching});attached=true;
        }catch(error){resultError=error.message||'The referenced result case is unavailable.';}
      }
      renderReferences(loaded.results,attached?resultId:loaded.job_id,attached&&!matching);
      if(attached&&matching)status(`Opened ${loaded.name}. Matching results are loaded; use Results for surface fields, recorded planes and images.`);
      else if(attached)status(`Opened ${loaded.name}. Historical results from the saved run are loaded. They belong to a different setup; the editable study is preserved. Create a new mesh to run the edited setup.`,true);
      else if(resultError)status(`Opened ${loaded.name}. Geometry and setup are restored. Saved results could not be loaded: ${resultError} Use Results to read saved native / VTK files.`,true);
      else status(`Opened ${loaded.name}. Geometry and setup are restored; result references are listed below.`);
    }catch(error){status(`Study could not be opened: ${error.message}`,true);}finally{setBusy(false);$('study-file').value='';}
  }
  function pickerButton(text,handler,className=''){
    const button=document.createElement('button');button.type='button';button.textContent=text;button.className=className;button.addEventListener('click',handler);return button;
  }
  function ensureRecentDialog(){
    if(recentDialog)return recentDialog;
    const dialog=document.createElement('dialog');
    // Native browsing remains available in browsers without HTML dialogs.
    if(typeof dialog.showModal!=='function')return null;
    dialog.className='study-picker';dialog.setAttribute('aria-labelledby','study-picker-title');
    const header=document.createElement('div');header.className='study-picker-header';
    const title=document.createElement('h2');title.id='study-picker-title';title.textContent='Open study';
    const close=pickerButton('Close',()=>dialog.close());close.title='Close the study picker and keep the current study';
    header.appendChild(title);header.appendChild(close);dialog.appendChild(header);
    const actions=document.createElement('div');actions.className='study-picker-actions';
    const browse=pickerButton('Browse other study…',()=>{dialog.close();open();});browse.title='Navigate to another .ludwig-study file using the system file dialog';
    const refresh=pickerButton('Refresh list',()=>loadRecentList());refresh.title='Refresh available saved studies and run snapshots';
    actions.appendChild(browse);actions.appendChild(refresh);dialog.appendChild(actions);
    recentHelp=document.createElement('p');recentHelp.className='study-picker-help';dialog.appendChild(recentHelp);
    recentList=document.createElement('div');recentList.className='study-picker-list';recentList.setAttribute('aria-label','Previously saved studies');dialog.appendChild(recentList);
    document.body.appendChild(dialog);recentDialog=dialog;return dialog;
  }
  async function loadRecentList(){
    if(busy)return;
    setBusy(true);recentHelp.textContent='Finding saved studies…';
    try{
      const response=await fetch('/api/aero/studies/recent',{headers:{'Accept':'application/json'}}),result=await response.json();
      if(!response.ok||result.ok===false)throw new Error(result.error||`HTTP ${response.status}`);
      const entries=Array.isArray(result.studies)?result.studies:[];recentList.replaceChildren();
      recentHelp.textContent=entries.length?'Click a study to restore its setup, geometry and available results. Files stay in their saved folders.':'No previously saved studies were found. Use Browse other study to choose a file.';
      for(const entry of entries){
        if(typeof entry.id!=='string'||typeof entry.path!=='string')continue;
        const row=pickerButton('',()=>{if(busy||!openAllowed())return;recentDialog.close();open(null,entry.id);},'study-recent-row');
        row.title='Open this study and automatically restore matching saved results';
        const name=document.createElement('span');name.className='study-recent-name';name.textContent=entry.name||'Aero study';
        const meta=document.createElement('span');meta.className='study-recent-meta';
        const modified=entry.modified_utc?new Date(entry.modified_utc):null;
        meta.textContent=[entry.kind==='run snapshot'?'Run snapshot':'Study',modified&&!Number.isNaN(modified.getTime())?modified.toLocaleString():null,entry.results_hint||'Setup and geometry'].filter(Boolean).join(' · ');
        const path=document.createElement('span');path.className='study-recent-path';path.textContent=entry.path;
        row.appendChild(name);row.appendChild(meta);row.appendChild(path);recentList.appendChild(row);
      }
    }catch(error){recentHelp.textContent=`The recent list could not be read: ${error.message}. Use Browse other study to open a file.`;}
    finally{setBusy(false);}
  }
  async function chooseOpen(){
    if(busy||!openAllowed())return;
    const dialog=ensureRecentDialog();if(!dialog){await open();return;}
    if(!dialog.open)dialog.showModal();await loadRecentList();
  }
  $('study-save').addEventListener('click',()=>save(false));
  $('study-save-menu')?.addEventListener('click',()=>save(false));$('study-save-as')?.addEventListener('click',()=>save(true));
  $('study-open').addEventListener('click',chooseOpen);
  $('study-file').addEventListener('change',event=>open(event.target.files?.[0]));
  async function browseFolder(){
    const button=$('study-browse');button.disabled=true;
    try{
      const path=await window.LudwigOutputControls.chooseFolder({path:$('study-folder')?.value||'',title:'Choose study folder'});
      if(path){$('study-folder').value=path;status(`Study folder: ${path}`);}
    }catch(error){status(error.message,true);window.LudwigAeroLayout?.showMessage(error.message,{urgent:true,title:'Folder selection'});}
    finally{button.disabled=false;}
  }
  $('study-browse')?.addEventListener('click',browseFolder);
  window.LudwigAeroStudies={save,saveAs:()=>save(true),open:chooseOpen,getSource:()=>sourceTarget};
})();
