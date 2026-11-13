"use strict";

/* Shared output choices for the local Airfoil and Aero workbenches. */
(() => {
  const $ = id => document.getElementById(id);
  const kinds = ['surface_vtk','surface_lbmp','volume_vtk','numeric_slices','slice_png','csv_history'];
  const labels = {surface_vtk:'Surface VTK (ParaView)',surface_lbmp:'Native surface data (LBMP)',volume_vtk:'Volume / field VTK',
    numeric_slices:'Numeric cut-plane data',slice_png:'Solver field PNG images',csv_history:'CSV histories and coefficients'};
  let endpoint = '/api/select-folder', capabilities = null, backends = [];
  const defaults = {surface_vtk:true,surface_lbmp:document.body.dataset.workbench==='aero',volume_vtk:false,numeric_slices:true,slice_png:true,csv_history:true};
  function init() {
    const host = $('output-controls'); if (!host) return;
    endpoint = document.body.dataset.workbench === 'aero' ? '/api/aero/select-folder' : '/api/select-folder';
    host.innerHTML = `<label class="field"><span>Update history every, iterations</span><input id="display-every" type="number" min="1" max="1000000" step="1" value="100"><small>History update cadence. CFD field snapshots have a separate time interval below. LBM outputs use this iteration interval.</small></label>
      <label class="field" id="snapshot-interval-field"><span>Save field snapshots every, minutes</span><input id="snapshot-interval-minutes" type="number" min="0" max="1440" step="any" value="10" title="Elapsed real time after marching starts, not simulation time. Zero disables periodic snapshots. Final results are always saved."><small>CFD: default 10 minutes of real time. 0 = off. Saved snapshots remain with the study and can be selected in Results → Flow field.</small></label>
      <label class="field"><span>Export profile</span><select id="output-profile"><option value="standard">Standard files</option><option value="all">All available files</option><option value="minimal">Coefficients only</option><option value="custom">Custom selection</option></select></label>
      <div class="output-file-options">${kinds.map(key=>`<label class="output-check"><input id="output-${key}" type="checkbox" ${defaults[key]?'checked':''}><span>${labels[key]}</span></label>`).join('')}</div>
      <p id="output-capability-note" class="hint">Only formats supported by the selected solver are exported. Internal GUI data remains available.</p>
      <label class="field"><span>${document.body.dataset.workbench==='airfoil'?'Results parent folder (optional)':'Output folder'}</span><div class="output-folder-row"><input id="output-folder" type="text" placeholder="${document.body.dataset.workbench==='airfoil'?'Study folder (automatic)':'Default private exports folder'}" autocomplete="off" spellcheck="false"><button id="output-browse" type="button" class="secondary-button">Browse…</button></div><small>${document.body.dataset.workbench==='airfoil'?'Leave empty to save beside the study. Each run creates a unique date-and-time folder (UTC), including its native fields and reports.':'Each job creates its own folder. Existing files are preserved.'}</small></label>
      <details class="output-makie-options"><summary>Reports, charts and 3D renders</summary>
        <label class="field"><span>Report</span><select id="output-report"><option value="none">Do not generate</option><option value="html">HTML report</option></select></label>
        <label class="field"><span>Makie charts</span><select id="output-charts"><option value="none">Do not generate</option><option value="png">PNG charts</option><option value="svg">SVG charts</option><option value="pdf">PDF charts</option></select></label>
        <label class="field"><span>Makie 3D render</span><select id="output-render"><option value="none">Do not generate</option><option value="png">PNG render</option></select></label>
        <div class="output-two-fields"><label class="field"><span>Surface quantity</span><select id="output-quantity"><option value="cp">Pressure coefficient Cp</option><option value="pressure">Pressure</option><option value="shear">Wall shear</option></select></label><label class="field"><span>Camera</span><select id="output-camera"><option value="perspective">Perspective</option><option value="top">Top</option><option value="front">Front</option><option value="side">Side</option></select></label></div>
        <label class="field"><span>Image size</span><select id="output-image-size"><option value="1600x1000">1600 × 1000</option><option value="1920x1080">1920 × 1080</option><option value="2560x1440">2560 × 1440</option><option value="3840x2160">3840 × 2160</option></select></label>
        <p class="hint">Makie renders saved solver data after the run. Missing quantities are identified in the report.</p>
      </details>`;
    $('output-profile').addEventListener('change', () => {
      const profile=$('output-profile').value;
      for(const key of kinds) $('output-'+key).checked=profile==='all'||(profile==='standard'&&defaults[key])||(profile==='minimal'&&key==='csv_history');
    });
    for(const key of kinds) $('output-'+key).addEventListener('change',()=>{$('output-profile').value='custom';});
    for(const id of ['output-report','output-charts','output-render'])$(id).addEventListener('change',updatePresentation);
    $('output-browse').addEventListener('click',openBrowser);
    render(null);
  }
  function displayEvery() {
    const raw=$('display-every')?.value ?? '100', value=Number(raw);
    if(!String(raw).trim()||!Number.isInteger(value)||value<1||value>1000000) throw new Error('Display interval must be a whole number from 1 to 1,000,000 iterations.');
    return value;
  }
  function read() {
    const outputs={folder:($('output-folder')?.value||'').trim()};if(document.body.dataset.workbench==='aero')outputs.folder_selected=Boolean(outputs.folder);
    const raw=$('snapshot-interval-minutes')?.value??'10', interval=Number(raw);
    if(!String(raw).trim()||!Number.isFinite(interval)||interval<0||interval>1440)throw new Error('Snapshot interval must be between 0 and 1440 minutes; use 0 to disable periodic snapshots.');
    outputs.snapshot_interval_minutes=interval;
    for(const key of kinds) outputs[key]=Boolean($('output-'+key)?.checked&&!$('output-'+key)?.disabled);
    const report=$('output-report')?.value||'none', charts=$('output-charts')?.value||'none', render=$('output-render')?.value||'none';
    outputs.report=report!=='none'; outputs.report_format='html'; outputs.charts=charts!=='none'; outputs.chart_format=charts==='none'?'png':charts;
    outputs.renders_3d=render!=='none'&&!$('output-render')?.disabled; outputs.render_format='png';
    outputs.surface_quantity=$('output-quantity')?.value||'cp'; outputs.camera=$('output-camera')?.value||'perspective';
    [outputs.image_width,outputs.image_height]=($('output-image-size')?.value||'1600x1000').split('x').map(Number);
    return outputs;
  }
  function restore(saved) {
    if(!$('output-controls'))return;
    const options=saved?.outputs||saved?.output_settings?.outputs||{}, cadence=saved?.display_every??saved?.flow?.display_every;
    $('snapshot-interval-minutes').value=String(options.snapshot_interval_minutes??10);
    if(cadence!=null)$('display-every').value=String(cadence);
    for(const key of kinds)if(typeof options[key]==='boolean')$('output-'+key).checked=options[key];
    const folder=options.folder||saved?.output_directory||'';
    const legacyTemporaryDefault=document.body.dataset.workbench==='aero'&&options.folder_selected!==true&&/(?:^|[\\/])TEMP_DELETEABLE[\\/]exports[\\/]?$/i.test(folder);
    $('output-profile').value='custom';$('output-folder').value=legacyTemporaryDefault?'':folder;
    $('output-report').value=options.report?'html':'none'; $('output-charts').value=options.charts?(options.chart_format||'png'):'none';
    $('output-render').value=options.renders_3d?'png':'none'; $('output-quantity').value=options.surface_quantity||'cp'; $('output-camera').value=options.camera||'perspective';
    const size=`${options.image_width||1600}x${options.image_height||1000}`;
    if(Array.from($('output-image-size').options).some(option=>option.value===size))$('output-image-size').value=size;
    updatePresentation();
  }
  function setCapabilities(value, selected) {
    if(value)capabilities=value; if(selected)backends=selected;
    if(!capabilities||!backends.length)return;
    const active=backends.map(name=>capabilities[name]).filter(Boolean);if(!active.length)return;
    const timed=active.some(item=>item.timed_snapshots===true);
    $('snapshot-interval-field').hidden=!timed;
    $('snapshot-interval-minutes').disabled=!timed;
    for(const key of [...kinds,'renders_3d']){
      const available=active.some(item=>item[key]!==false),node=key==='renders_3d'?$('output-render'):$('output-'+key);
      if(node){node.disabled=!available;node.title=available?'':'Not available for the selected solver';if(document.body.dataset.workbench==='airfoil')node.closest('label').hidden=!available;}
    }
    const notices=active.flatMap(item=>[...(Array.isArray(item.notices)?item.notices:[]),...(item.display_note?[item.display_note]:[])]);
    if($('output-capability-note'))$('output-capability-note').textContent=notices.join(' ')||'Selected exports use the formats supported by this solver. Internal GUI data remains available.';
    updatePresentation();
  }
  function updatePresentation(){
    if(document.body.dataset.workbench!=='airfoil')return;
    const render=!$('output-render').disabled&&$('output-render').value!=='none',charts=$('output-charts').value!=='none';
    $('output-quantity').closest('.output-two-fields').hidden=!render;
    $('output-image-size').closest('label').hidden=!render&&!charts;
  }
  function render(job) {
    const host=$('output-results');if(!host)return;host.replaceChildren();
    const exports=job?.exports||job?.output_exports;
    host.dataset.available=String(Boolean(exports));
    const line=document.createElement('p');line.className='hint';
    line.textContent=exports?`Exports: ${exports.status||'pending'}`:'Selected exports and Makie reports appear here after a run.';host.appendChild(line);
    if(!exports)return;
    if(exports.folder){const folder=document.createElement('p');folder.className='output-path';folder.textContent=exports.folder;host.appendChild(folder);}
    const list=document.createElement('div');list.className='output-export-links';
    for(const file of exports.files||[]){const url=file.url||file.href;if(typeof url!=='string'||!url.startsWith('/api/'))continue;
      const link=document.createElement('a');link.href=url;link.textContent=file.name||file.label||url.split('/').at(-1);link.target='_blank';link.rel='noopener noreferrer';list.appendChild(link);}
    host.appendChild(list);
    for(const warning of exports.warnings||[]){const p=document.createElement('p');p.className='hint output-warning';p.textContent=String(warning);host.appendChild(p);}
    const requested=job.case?.display_every??job.request?.display_every??job.output_settings?.display_every_requested;
    const interval=job.display_interval||job.output_settings?.display_interval||(requested?{requested,effective:job.case?.effective_display_every??requested}:null);
    if(interval){const p=document.createElement('p');p.className='hint';p.textContent=typeof interval==='string'?interval:`Display interval: ${interval.effective??interval.requested} iterations.`;host.appendChild(p);}
  }
  async function chooseFolder({path='',title='Choose folder'}={}) {
    const response=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path,title})});
    const data=await response.json();
    if(!response.ok||data.ok===false)throw new Error(data.error||`Folder selection failed (HTTP ${response.status}).`);
    return data.cancelled?null:data.path;
  }
  async function openBrowser(){
    const button=$('output-browse'),label=button.textContent;button.disabled=true;button.textContent='Choose in Windows...';
    try{
      const path=await chooseFolder({path:$('output-folder').value,title:'Choose simulation output folder'});
      if(path){$('output-folder').value=path;$('output-folder').dispatchEvent(new Event('input',{bubbles:true}));}
    }catch(error){
      if($('output-capability-note'))$('output-capability-note').textContent=error.message;
      window.LudwigAeroLayout?.showMessage(error.message,{urgent:true,title:'Folder selection'});
    }finally{button.disabled=false;button.textContent=label;}
  }
  window.LudwigOutputControls={read,displayEvery,restore,render,setCapabilities,chooseFolder};
  init();
})();
