/* Saved native/VTK datasets are imported for inspection; this never starts a solver. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id),ctx=window.LudwigAeroResultsContext;
  if(!ctx||!$('results-import'))return;
  const datasets=new Map(),defaultSource=$('results-source').value||'simulation';let busy=false,offsetEdited=false;
  const active=new Set(['meshing','sizing','mesh_sizing','waiting_for_gpu','running','queued','starting']);
  function status(text,error=false){$('results-status').textContent=text;$('results-status').classList.toggle('warning',error);}
  function available(){if(active.has(ctx.state.job?.status)||ctx.state.importing||ctx.state.uploading||ctx.state.resumePending){status('Wait for the active case or model upload to finish or cancel it before changing the displayed results.',true);return false;}return true;}
  function setBusy(value){busy=value;$('results-import').disabled=value;$('results-file').disabled=value;}
  async function request(url,options){const response=await fetch(url,options),body=await response.json();if(!response.ok||body.ok===false)throw new Error(body.error||`HTTP ${response.status}`);return body;}
  function options(){
    const coordinates=$('results-coordinates').value,offset=['x','y','z'].map(axis=>Number($('results-offset-'+axis).value));
    if(offset.some(v=>!Number.isFinite(v)))throw new Error('Result offsets must be finite metres.');
    const placement=ctx.getPlacement?.()||{},result={coordinates,fallback_mesh_offset:placement.mesh_offset||offset};
    if(placement.job_id)result.job_id=placement.job_id;
    if(coordinates!=='auto'||offsetEdited){result.mesh_offset=offset;result.offset_explicit=true;}
    return result;
  }
  function fieldOptions(meta,surface){
    const supported=new Set(Object.keys(surface?.fields||{}));
    for(const field of meta.volume_fields||[]){const key=String(field).toLowerCase();if(key==='velocity'){for(const f of ['velocity','u','v','w'])supported.add(f);}else if(['pressure','density','vorticity','mach','temperature'].includes(key))supported.add(key);}
    for(const cut of meta.cuts||[])for(const field of cut.fields||['velocity','pressure','density','u','v','w','vorticity'])supported.add(field.startsWith('vorticity')?'vorticity':field);
    const select=$('results-field');for(const field of supported)if(!Array.from(select.options).some(option=>option.value===field)){const option=document.createElement('option');option.value=field;option.textContent=field.charAt(0).toUpperCase()+field.slice(1);select.appendChild(option);}
    for(const option of select.options)option.disabled=!supported.has(option.value);
    if(select.selectedOptions[0]?.disabled){const first=Array.from(select.options).find(o=>!o.disabled);if(first)select.value=first.value;}
    ctx.state.importedField=select.value;
  }
  async function display(meta){
    let surface=null;if(meta.surface_url)surface=await request(meta.surface_url);
    datasets.set(meta.dataset_id,meta);
    if(!Array.from($('results-source').options).some(o=>o.value===meta.dataset_id)){const option=document.createElement('option');option.value=meta.dataset_id;option.textContent=meta.name;$('results-source').appendChild(option);}
    if(!available()){status('Results were imported and retained. Select their dataset when the active case finishes.');return;}
    $('results-source').value=meta.dataset_id;fieldOptions(meta,surface);
    await ctx.applyImportedResults(meta,surface);ctx.setImportedField?.(ctx.state.importedField);ctx.queueRender();
    status(`Loaded ${meta.files.length} saved file${meta.files.length===1?'':'s'} · ${meta.surface?.triangles?.toLocaleString()||0} surface triangles. ${(meta.warnings||[]).join(' ')}`);
  }
  async function importPath(){
    if(busy||!available())return;setBusy(true);status('Reading saved results…');
    try{const meta=await request('/api/aero/results/import',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...options(),path:$('results-path').value.trim()})});await display(meta);}
    catch(error){status(`Results could not be loaded: ${error.message}`,true);}finally{setBusy(false);}
  }
  async function upload(files){
    if(!files.length||busy||!available())return;setBusy(true);status('Uploading saved results for inspection…');
    try{
      const placement=ctx.getPlacement?.(),chosen=Array.from(files).sort((a,b)=>(a.name.endsWith('.json')?0:a.name.endsWith('.toml')?1:2)-(b.name.endsWith('.json')?0:b.name.endsWith('.toml')?1:2));
      if(!offsetEdited&&placement?.mesh_offset)for(const [i,a] of ['x','y','z'].entries())$('results-offset-'+a).value=String(placement.mesh_offset[i]);
      let meta=null;const opts=options(),hasAmr=chosen.some(file=>file.name.toLowerCase()==='amr_data.json');
      for(const file of chosen){
        if(file.size>512_000_000)throw new Error('Each interactive result file must be at most 512 MB.');
        const query=new URLSearchParams({name:file.name,coordinates:opts.coordinates});
        if(opts.mesh_offset)query.set('mesh_offset',opts.mesh_offset.join(','));
        if(opts.fallback_mesh_offset)query.set('fallback_mesh_offset',opts.fallback_mesh_offset.join(','));
        if(opts.offset_explicit)query.set('offset_explicit','true');
        if(opts.job_id)query.set('job_id',opts.job_id);
        if(meta)query.set('dataset_id',meta.dataset_id);
        status(`Reading ${file.name}…`);meta=await request('/api/aero/results/upload?'+query,{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:await file.arrayBuffer()});
      }
      await display(meta);
    }catch(error){status(`Results could not be loaded: ${error.message}`,true);}finally{setBusy(false);$('results-file').value='';}
  }
  $('results-import').addEventListener('click',importPath);
  $('results-file').addEventListener('change',event=>upload(event.target.files||[]));
  $('results-source').addEventListener('change',async()=>{
    if(busy||!available())return;
    const id=$('results-source').value;
    if(id===defaultSource){ctx.clearImportedResults();ctx.queueRender();status('Showing the current simulation results.');return;}
    try{setBusy(true);await display(datasets.get(id)||await request('/api/aero/results/'+encodeURIComponent(id)));}
    catch(error){status(error.message,true);}finally{setBusy(false);}
  });
  $('results-field').addEventListener('change',()=>{ctx.state.importedField=$('results-field').value;ctx.setImportedField?.(ctx.state.importedField);ctx.queueRender();});
  for(const axis of ['x','y','z'])$('results-offset-'+axis).addEventListener('input',()=>{offsetEdited=true;});
  window.LudwigAeroResults={
    async readUrl(file){const response=await fetch(file.url,{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);await upload([new File([await response.arrayBuffer()],file.name.split(/[\\/]/).at(-1),{type:'application/octet-stream'})]);},
    getReference(){const meta=ctx.state.importedResults;return meta?{dataset_id:meta.dataset_id,source_paths:meta.source_paths,coordinates:meta.options?.coordinates,mesh_offset:meta.surface_offset}:ctx.state.importedResultsReference||null;},
    async restoreReference(reference){
      ctx.state.importedResultsReference=reference||null;if(!reference)return;
      if(!/^[0-9a-f-]{36}$/.test(reference.dataset_id||'')){status('The study refers to saved results. Choose their current location to load them.');return;}
      try{
        const meta=await request('/api/aero/results/'+reference.dataset_id);
        if(JSON.stringify(meta.source_paths)!==JSON.stringify(reference.source_paths))throw new Error('Result references differ');
        datasets.set(meta.dataset_id,meta);if(!Array.from($('results-source').options).some(o=>o.value===meta.dataset_id)){const option=document.createElement('option');option.value=meta.dataset_id;option.textContent=meta.name;$('results-source').appendChild(option);}
        status('The study has a matching imported dataset. Select it under Results source to visualize it.');
      }catch{status('The study result reference is unavailable here. Choose the native or VTK files to load them.');}
    }
  };
  window.LudwigImportedResults=window.LudwigAeroResults;
})();
