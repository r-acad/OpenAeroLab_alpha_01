/* Catalog solver files and retain image selection while a job is polled. */
(() => {
  'use strict';
  const groups = {surface:'Surface fields',volume:'Volume fields',cut:'Cut planes',image:'Images',history:'Histories',setup:'Mesh and setup',log:'Logs and reports',other:'Other files'};
  const safeUrl = value => {
    if (typeof value !== 'string' || !value.startsWith('/api/aero/') || /[\\\r\n]/.test(value)) return null;
    try { const path = decodeURIComponent(value.split('?')[0]); if (/[\x00-\x1f\\]/.test(path) || path.split('/').some(part => part === '.' || part === '..')) return null; } catch { return null; }
    return value;
  };
  const textSize = bytes => {
    if (!Number.isFinite(bytes) || bytes < 0) return '';
    const units = ['B','KiB','MiB','GiB']; let unit = 0;
    while (bytes >= 1024 && unit < units.length - 1) { bytes /= 1024; unit++; }
    return `${bytes.toFixed(unit ? 1 : 0)} ${units[unit]}`;
  };
  function normalize(raw, fallback = {}) {
    if (typeof raw === 'string') raw = {url:raw};
    if (!raw || typeof raw !== 'object') return null;
    const originalUrl = safeUrl(raw.url || raw.href), url = safeUrl(raw.source_url) || originalUrl; if (!url) return null;
    let basename = url.split('?')[0].split('/').at(-1) || 'Result file';
    try { basename = decodeURIComponent(basename); } catch {}
    const name = String(raw.name || raw.label || basename), format = String(raw.format || (name.includes('.') ? name : basename).split('.').at(-1) || '').toLowerCase();
    let kind = String(raw.kind || raw.type || fallback.kind || '').toLowerCase();
    if (['slice','numeric_cut','cut_plane'].includes(kind)) kind = 'cut';
    if (['amr','mesh','geometry','config','summary'].includes(kind)) kind = 'setup';
    if (['report','chart'].includes(kind)) kind = /^(png|jpe?g|webp|gif|svg)$/.test(format) ? 'image' : 'log';
    if (!groups[kind]) kind = /^(png|jpe?g|webp|gif|svg)$/.test(format) ? 'image' : format === 'lbmp' || format === 'vtp' ? 'surface' : format === 'lbmslice' ? 'cut' : format === 'vtu' ? (/surface|wall|body/i.test(name) ? 'surface' : 'volume') : format === 'vtk' ? (/volume/i.test(name) ? 'volume' : 'surface') : format === 'csv' ? 'history' : /^(stl|obj|glb|yaml|yml|toml|json)$/.test(format) ? 'setup' : /^(log|txt|html|pdf)$/.test(format) ? 'log' : 'other';
    const stepValue = raw.step ?? raw.iteration ?? fallback.step;
    const inferred = /(?:step|iter(?:ation)?)[_-]?(\d+)/i.exec(name);
    const step = stepValue != null && Number.isFinite(Number(stepValue)) ? Number(stepValue) : inferred ? Number(inferred[1]) : null;
    const phase = String(raw.phase || fallback.phase || (url.includes('/mesh/') ? 'mesh' : 'run'));
    return {...fallback,...raw,url,original_url:originalUrl,name,kind,format,phase,step,bytes:Number.isFinite(Number(raw.bytes)) && raw.bytes != null ? Number(raw.bytes) : null};
  }
  function create(adapter) {
    const $ = adapter.$ || (id => document.getElementById(id));
    let jobId = null, files = new Map(), selected = '', imageList = [], lastImageUrl = '', lastImageStamp = '', imageLoadFailed = false, lastJob = null, followLatest = true;
    const element = (tag, text, className) => { const node = document.createElement(tag); if (text != null) node.textContent = String(text); if (className) node.className = className; return node; };
    const status = (message, error = false) => adapter.onStatus?.(message,error);
    const live = () => typeof adapter.isLive === 'function' ? !!adapter.isLive() : ($('live-results')?.checked ?? true);
    const following = () => $('result-image-follow')?.checked ?? followLatest;
    if ($('artifact-links')) $('artifact-links').hidden = true;

    function chooseImage(url, force = false, explicit = true) {
      const file = files.get(url); if (!file || file.kind !== 'image') return;
      if (explicit) { followLatest = false; if ($('result-image-follow')) $('result-image-follow').checked = false; }
      selected = url; if ($('result-image-select')) $('result-image-select').value = url;
      const image = $('result-image');
      if (image) {
        image.alt = file.name; image.hidden = false;
        const stamp = `${file.step ?? ''}:${file.bytes ?? ''}:${file.updated_utc || file.modified_utc || ''}`;
        if (force || lastImageUrl !== url || lastImageStamp !== stamp) { imageLoadFailed = false; image.src = force ? `${url}${url.includes('?') ? '&' : '?'}reload=${Date.now()}` : lastImageUrl === url && lastImageStamp !== stamp ? `${url}${url.includes('?') ? '&' : '?'}version=${encodeURIComponent(stamp)}` : url; lastImageUrl = url; lastImageStamp = stamp; }
      }
      if ($('result-image-empty')) $('result-image-empty').hidden = !imageLoadFailed;
      if ($('result-image-caption')) $('result-image-caption').textContent = [file.name,file.phase,file.step == null ? '' : `iteration ${file.step}`,file.quantity,textSize(file.bytes)].filter(Boolean).join(' · ');
    }

    async function view(file) {
      if (file.kind === 'image') { chooseImage(file.url); return; }
      const callback = {surface:adapter.onOpenSurface,cut:adapter.onOpenCut,volume:adapter.onOpenVolume}[file.kind];
      if (!callback) return;
      try { await callback(file); } catch (error) { status(`Could not display ${file.name}: ${error.message || error}`,true); }
    }

    function catalog(job) {
      const a = job?.artifacts || {}, incoming = [];
      const add = (raw, fallback) => { const value = normalize(raw,fallback); if (value) incoming.push(value); };
      for (const key of ['files','surfaces','volumes','images']) for (const raw of a[key] || []) add(raw,{kind:{surfaces:'surface',volumes:'volume',images:'image'}[key]});
      for (const raw of a.cuts || []) { add(raw,{kind:'cut'}); for (const key of ['png_url','image_url','png']) if (raw[key]) add(raw[key],{kind:'image',step:raw.step,phase:'run'}); }
      for (const raw of job?.exports?.files || job?.output_exports?.files || []) add(raw);
      for (const [key,kind,name] of [['model','setup','Solver geometry.stl'],['amr','setup',null],['summary','setup',null],['surface_latest','surface',null]]) if (a[key]) add({url:a[key],...(name ? {name} : {})},{kind,phase:key === 'surface_latest' ? 'run' : 'mesh'});
      for (const file of incoming) {
        const prior = files.get(file.url), merged = {...prior,...file};
        if (prior && file.source_url && file.original_url !== file.url) continue;
        if (prior) for (const key of ['step','bytes','quantity']) if (file[key] == null) merged[key] = prior[key];
        files.set(file.url,merged);
      }
    }

    function renderAvailability(job) {
      const reports = [];
      for (const [key,label,kind] of [['surface','Surface fields','surface'],['volume','Volume fields','volume'],['cuts','Recorded cut planes','cut'],['images','Generated images','image']]) {
        const metadata = job?.artifacts?.availability?.[key], known = [...files.values()].filter(file => file.kind === kind);
        const available = typeof metadata === 'boolean' ? metadata : metadata?.available == null ? known.length > 0 : !!metadata.available;
        const reason = typeof metadata === 'object' ? metadata.reason : '';
        reports.push({label,available,text:available ? `${label}: available${known.length ? ` (${known.length})` : ''}.${reason ? ` ${reason}` : ''}` : `${label}: unavailable. ${reason || 'No matching output has been written.'}`});
      }
      for (const id of ['result-availability','results-availability-summary']) if ($(id)) {
        $(id).replaceChildren();
        for (const report of reports) $(id).appendChild(element('p',report.text,report.available ? 'result-available' : 'result-unavailable'));
      }
    }

    function renderFiles() {
      const holder = $('result-file-browser'); if (!holder) return;
      const openGroups = new Set([...holder.children].filter(group => group.open).map(group => group.dataset.kind));
      holder.replaceChildren();
      for (const [kind,label] of Object.entries(groups)) {
        const matching = [...files.values()].filter(file => file.kind === kind); if (!matching.length) continue;
        const group = element('details',null,'result-file-group'); group.dataset.kind = kind; group.open = openGroups.has(kind); group.appendChild(element('summary',`${label} · ${matching.length}`));
        const phases = new Map();
        for (const file of matching.sort((a,b) => (b.step ?? -1) - (a.step ?? -1) || a.name.localeCompare(b.name))) { const phase = `${file.phase}${file.step == null ? '' : ` · iteration ${file.step}`}`; if (!phases.has(phase)) phases.set(phase,[]); phases.get(phase).push(file); }
        for (const [phase,records] of phases) {
          const section = element('section',null,'result-phase-group'); section.appendChild(element('h4',phase));
          for (const file of records) {
            const row = element('div',null,'result-file-row'), description = element('div',null,'result-file-description'); description.append(element('strong',file.name),element('small',[file.format.toUpperCase(),file.quantity,textSize(file.bytes)].filter(Boolean).join(' · '))); row.appendChild(description);
            if (file.geometry_frame === 'legacy_unverified') { const note = element('small','Legacy native surface · uses matching AMR geometry'); note.title = 'Legacy native surface: viewer aligns values with the matching AMR mesh. Use VTK for standalone geometry export.'; description.appendChild(note); }
            const actions = element('div',null,'result-file-actions'), callback = {surface:adapter.onOpenSurface,cut:adapter.onOpenCut,volume:adapter.onOpenVolume}[kind];
            if (kind === 'image' || callback) { const button = element('button','View'); button.type = 'button'; button.title = `Display ${file.name} in the workbench`; button.addEventListener('click',() => view(file)); actions.appendChild(button); }
            else if (['html','pdf'].includes(file.format)) { const link = element('a','View'); link.href = file.url; link.target = '_blank'; link.rel = 'noopener noreferrer'; link.title = `Open ${file.name}`; actions.appendChild(link); }
            const download = element('a','Download'); download.href = file.url; download.setAttribute('download',file.name); download.title = `Download ${file.name}`; actions.appendChild(download); row.appendChild(actions); section.appendChild(row);
          }
          group.appendChild(section);
        }
        holder.appendChild(group);
      }
      if (!holder.children.length) holder.appendChild(element('p','No result files have been written yet.','hint'));
    }

    function renderImages() {
      imageList = [...files.values()].filter(file => file.kind === 'image').sort((a,b) => Number(b.phase === 'run') - Number(a.phase === 'run') || (b.step ?? -1) - (a.step ?? -1) || a.name.localeCompare(b.name));
      const select = $('result-image-select'); if (select) {
        select.replaceChildren();
        if (!imageList.length) { const option = element('option','No generated images'); option.value = ''; select.appendChild(option); }
        for (const file of imageList) { const option = element('option',`${file.step == null ? '' : `Iteration ${file.step} · `}${file.name}`); option.value = file.url; select.appendChild(option); }
        select.disabled = !imageList.length;
      }
      for (const id of ['result-image-prev','result-image-next','result-image-reload']) if ($(id)) $(id).disabled = !imageList.length;
      if (imageList.length && live() && following()) chooseImage(imageList[0].url,false,false);
      else if (selected && files.has(selected)) chooseImage(selected,false,false);
      else if (imageList.length && live()) chooseImage(imageList[0].url,false,false);
      else {
        if ($('result-image')) $('result-image').hidden = true;
        if ($('result-image-empty')) { $('result-image-empty').hidden = false; $('result-image-empty').textContent = imageList.length ? 'Automatic display is off. Choose an image to view it.' : 'No generated images are available yet.'; }
        if (select) select.value = '';
      }
    }

    function render(job) {
      const current = String(job?.job_id || job?.id || job?.dataset_id || '');
      if (current !== jobId) { jobId = current; files = new Map(); selected = ''; lastImageUrl = ''; lastImageStamp = ''; imageLoadFailed = false; $('result-file-browser')?.replaceChildren(); if ($('result-image-caption')) $('result-image-caption').textContent = ''; }
      lastJob = job || null; catalog(job); renderAvailability(job); renderFiles(); renderImages();
    }
    $('result-image-select')?.addEventListener('change',() => chooseImage($('result-image-select').value));
    for (const [id,direction] of [['result-image-prev',-1],['result-image-next',1]]) $(id)?.addEventListener('click',() => { if (!imageList.length) return; const index = imageList.findIndex(file => file.url === selected); chooseImage(imageList[(Math.max(0,index)+direction+imageList.length)%imageList.length].url); });
    $('result-image-reload')?.addEventListener('click',() => { if (selected) chooseImage(selected,true,false); });
    $('result-image-follow')?.addEventListener('change',() => { followLatest = $('result-image-follow').checked; if (following() && live() && imageList.length) chooseImage(imageList[0].url,false,false); });
    $('result-image')?.addEventListener('error',() => { imageLoadFailed = true; if ($('result-image-empty')) { $('result-image-empty').hidden = false; $('result-image-empty').textContent = 'This image is temporarily unavailable. Its selection is retained; use Reload to try again.'; } });
    $('result-image')?.addEventListener('load',() => { imageLoadFailed = false; if ($('result-image-empty')) $('result-image-empty').hidden = true; });
    return {render,selectImage:chooseImage,getSelection:() => selected,isFollowing:following,refresh:() => render(lastJob)};
  }
  window.LudwigAeroResultBrowser = {create};
})();
