/* Shared navigation keeps existing workbench controls and their event handlers. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id),kind=document.body.dataset.workbench;
  const menus=Array.from(document.querySelectorAll('.top-menu'));
  function closeMenus(except){for(const menu of menus)if(menu!==except)menu.open=false;}
  function positionMenu(menu){
    const popover=menu.querySelector('.menu-popover');if(!popover)return;
    const bar=menu.closest('.menu-bar')?.getBoundingClientRect();
    if(bar)document.documentElement.style.setProperty('--aero-menu-bottom',`${bar.bottom+3}px`);
    popover.style.left='';popover.style.right='';
    if(window.innerWidth>780&&popover.getBoundingClientRect().right>window.innerWidth-8){popover.style.left='auto';popover.style.right='0';}
  }
  function choices(menu){return Array.from(menu.querySelectorAll('button:not(:disabled),a[href],select,input:not([type=hidden])')).filter(node=>!node.hidden&&!node.closest('[hidden]'));}
  menus.forEach((menu,index)=>{
    const summary=menu.querySelector(':scope > summary');
    summary?.setAttribute('aria-haspopup','true');
    menu.addEventListener('toggle',()=>{summary?.setAttribute('aria-expanded',String(menu.open));if(menu.open){closeMenus(menu);positionMenu(menu);}});
    menu.addEventListener('keydown',event=>{
      if(['ArrowLeft','ArrowRight'].includes(event.key)&&document.activeElement===summary){event.preventDefault();const next=menus[(index+(event.key==='ArrowRight'?1:-1)+menus.length)%menus.length];const opened=menu.open;closeMenus();next.open=opened;next.querySelector('summary')?.focus();}
      else if(event.key==='ArrowDown'&&document.activeElement===summary){event.preventDefault();menu.open=true;choices(menu)[0]?.focus();}
      else if(event.key==='Escape'&&menu.open){event.preventDefault();closeMenus();summary?.focus();}
    });
  });
  document.addEventListener('click',event=>{if(!event.target.closest('.top-menu'))closeMenus();});
  function selectAirfoilParameters(name,focus){
    const groups=Array.from(document.querySelectorAll('[data-airfoil-panel]')),chosen=groups.find(node=>node.dataset.airfoilPanel===name);
    if(!chosen)return;
    for(const node of groups)node.hidden=node!==chosen;
    if(chosen.tagName==='DETAILS')chosen.open=true;
    const heading=$('airfoil-parameter-title');if(heading)heading.textContent=chosen.dataset.parameterTitle||name;
    for(const menu of menus)menu.classList.toggle('has-active-parameters',!!menu.querySelector(`[data-airfoil-parameters="${name}"]`));
    closeMenus();const control=$(focus);control?.focus({preventScroll:true});control?.scrollIntoView({block:'nearest'});
  }
  for(const button of document.querySelectorAll('[data-airfoil-parameters]'))button.addEventListener('click',()=>selectAirfoilParameters(button.dataset.airfoilParameters,button.dataset.focusControl));
  for(const button of document.querySelectorAll('[data-open-tab]'))button.addEventListener('click',()=>{$(`tab-button-${button.dataset.openTab}`)?.click();closeMenus();});
  for(const button of document.querySelectorAll('[data-command]')){
    const target=$(button.dataset.command);if(!target)continue;
    const sync=()=>{button.disabled=target.disabled;};sync();
    if(typeof MutationObserver!=='undefined')new MutationObserver(sync).observe(target,{attributes:true,attributeFilter:['disabled']});
    button.addEventListener('click',()=>{if(target.disabled)return;closeMenus();target.click();});
  }
  $('open-saved-airfoil')?.addEventListener('click',()=>{closeMenus();const section=$('recent-section');if(section){section.open=true;section.scrollIntoView({block:'nearest'});$('recent-select')?.focus();}});
  if(kind==='airfoil'&&$('download-result')){
    const syncResults=()=>{$('menu-results')?.classList.toggle('results-available',!$('download-result').disabled||Boolean(window.AirfoilBenchmarks?.current()));};
    window.addEventListener('airfoil-reference-updated',syncResults);
    syncResults();if(typeof MutationObserver!=='undefined')new MutationObserver(syncResults).observe($('download-result'),{attributes:true,attributeFilter:['disabled']});
  }
  function applyTheme(){
    let preference='dark';try{preference=localStorage.getItem('airfoil-theme')||'dark';}catch{}
    if(!['auto','light','dark'].includes(preference))preference='dark';
    const select=$('theme-mode');if(select)select.value=preference;
    document.documentElement.dataset.theme=preference==='auto'?(window.matchMedia?.('(prefers-color-scheme: dark)').matches?'dark':'light'):preference;
  }
  $('theme-mode')?.addEventListener('change',event=>{try{localStorage.setItem('airfoil-theme',event.target.value);}catch{}applyTheme();});
  window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener?.('change',applyTheme);
  applyTheme();
  const palette=$('results-palette');
  if(palette&&window.LudwigPalettes){
    for(const [value,label] of Object.entries(window.LudwigPalettes.labels)){const option=document.createElement('option');option.value=value;option.textContent=label;palette.appendChild(option);}
    palette.value=window.LudwigPalettes.get();
    palette.addEventListener('change',()=>window.LudwigPalettes.set(palette.value));
    document.documentElement.style.setProperty('--result-ramp',window.LudwigPalettes.gradient());
  }
  for(const button of document.querySelectorAll('[data-workflow-help]'))button.addEventListener('click',()=>{
    closeMenus();let dialog=$('workflow-help');
    if(!dialog){
      dialog=document.createElement('dialog');dialog.id='workflow-help';dialog.className='workflow-help';dialog.setAttribute('aria-labelledby','workflow-help-title');
      const close=document.createElement('button');close.className='workflow-help-close';close.textContent='Close';close.addEventListener('click',()=>dialog.close());dialog.appendChild(close);
      const title=document.createElement('h2');title.id='workflow-help-title';title.textContent=kind==='aero'?'Aero Lab · from model to results':'Airfoil Lab · from section to polar';dialog.appendChild(title);
      const list=document.createElement('ol');
      const steps=kind==='aero'?[
        'Project: open a saved study or load STL, OBJ or GLB geometry. Save study keeps the current filename; Save study as chooses another.',
        'Model and Domain: place the geometry, enter reference wing area, chord and moment origin, then choose boundary conditions and minimum clearances. The selected settings appear in the left pane.',
        'Mesh: select the solver, surface cell size and AMR levels, then create the mesh. Inspect the memory estimate and geometry checks.',
        'Analysis: set speed, alpha and beta, iterations and the output folder before Run solver. Symmetry fixes beta to zero. The progress window reports activity and memory use.',
        'Results: select Surface, Recorded planes or Images & files. Set field scale limits or inspect coefficients under Reference dimensions. Display changes retain the loaded fields.',
        'Display: Move a cutting plane offers an edge handle (Shift for fine movement), exact position and surface-section curves. Record required planes before running, or export volume VTK to inspect arbitrary sections later.',
        'Display: choose view, guides, AMR visibility and a contour palette. The palette recolours numeric fields; saved PNG images keep their original colours.'
      ]:[
        'Project: New, Open, Save and Save As manage complete studies. Open offers recent studies; Save reuses the opened filename. Save after an analysis to retain its results link.',
        'Model: choose the airfoil. RAE 2822 and ONERA OAT15A load published conditions and show measured Cp in Results immediately. Choose another measured case there or edit the setup.',
        'Mesh: choose topology and resolution in the geometry view; Create mesh previews the solver grid.',
        'Analysis: set the flow, solver and optional alpha sweep. Outputs controls saved files, destination and reporting.',
        'Run analysis: watch progress in the left pane and review the run log or diagnostics.',
        'Results: open saved jobs, coefficient plots, the flow field or a polar. Check convergence before comparing values.',
        'Display: change contour palette or theme. Imported PNGs retain their saved colours; native numeric fields are recoloured.'
      ];
      for(const step of steps){const li=document.createElement('li');li.textContent=step;list.appendChild(li);}dialog.appendChild(list);
      const help=document.createElement('p');help.textContent='Menus support Tab, Escape and left/right arrows. Down opens a menu. View controls and setup changes do not delete saved files.';dialog.appendChild(help);document.body.appendChild(dialog);
    }
    dialog.showModal();
  });
  if(kind==='airfoil')selectAirfoilParameters('model');
  window.LudwigWorkbench={closeMenus,selectAirfoilParameters};
})();
