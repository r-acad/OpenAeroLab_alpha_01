/* Visible workflow navigation. Existing controls are moved, never cloned. */
(() => {
  "use strict";
  const $=id=>document.getElementById(id);
  const panels=Object.fromEntries([...document.querySelectorAll("[data-airfoil-panel]")].map(node=>[node.dataset.airfoilPanel,node]));
  const buttons=[...document.querySelectorAll("[data-workflow-target]")];
  let selected="model",scheduled=false;
  const move=(node,parent)=>{if(node&&parent)parent.append(node);};
  const field=id=>$(id)?.closest(".field,.header-control");

  // Keep source, flow, geometry additions and mesh choices in one place each.
  move($("flow-heading")?.closest("section"),panels.flow);
  panels.sweep.hidden=false;panels.sweep.removeAttribute("data-airfoil-panel");move(panels.sweep,panels.flow);delete panels.sweep;
  const flap=$("gurney-group"),extras=$("streamline-align")?.closest("details");
  move(extras,panels.mesh);move(flap,panels.flap);move($("lbm-spoiler-group"),panels.flap);
  for(const id of ["lowspeed-resolution","lowspeed-topology","cfd-mesh","trailing-edge-thickness","preview-resolution"])
    move(field(id),panels.mesh);
  move($("cfd-te-topology-field"),panels.mesh);
  move($("recent-section"),panels.study);
  move(document.querySelector(".results-toolbar > .toolbar-selects"),panels["result-controls"]);
  move($("output-results")?.closest("details"),panels["result-controls"]);
  move($("field-grid")?.closest("label"),panels.display);
  move($("chart-interrogate")?.closest("label"),panels.display);
  for(const id of ["lowspeed-tu","lowspeed-mut-ratio"]){
    const node=field(id);if(node){node.dataset.workflowPhysics="lowspeed";move(node,$("flow-heading").closest("section").querySelector(".form-grid"));}
  }

  // The selected top-level tab is the only disclosure mechanism for setup.
  // Preserve live control nodes and IDs so study snapshots and listeners survive.
  for(const details of [...document.querySelectorAll(".setup-column details")].reverse()){
    const section=document.createElement("section");
    for(const attribute of [...details.attributes])if(attribute.name!=="open")section.setAttribute(attribute.name,attribute.value);
    section.classList.add("workflow-section");
    for(const child of [...details.childNodes]){
      if(child.nodeType===1&&child.tagName==="SUMMARY"){
        const heading=document.createElement("h3");heading.append(...child.childNodes);section.append(heading);
      }else section.append(child);
    }
    details.replaceWith(section);
    if(details.id&&typeof el!=="undefined"&&el[details.id]===details)el[details.id]=section;
    for(const [key,node] of Object.entries(panels))if(node===details)panels[key]=section;
  }
  if($("solver-heading"))$("solver-heading").textContent="Solver and stopping limits";
  if($("airfoil-heading"))$("airfoil-heading").textContent="Airfoil coordinates";
  if($("flow-heading"))$("flow-heading").textContent="Flow conditions";
  $("tab-button-results").textContent="Surface and forces";
  $("tab-button-campaign").hidden=true; // The historical development dashboard is not a case result.
  const sourceExtra=field("gurney-refine");
  const streamlineLabel=$("streamline-align")?.closest("label");

  function hasFields(value){
    return Boolean(value&&(value.full_artifact||value.artifact||
      Object.values(value.samples||{}).some(samples=>Array.isArray(samples)&&samples.length)||
      value.blocks?.some(hasFields)));
  }
  function availability(){
    const rows=Array.isArray(state.result?.results)?state.result.results:[];
    const point=currentResult(),geometry=state.preview?.geometry||state.result?.geometry||state.embeddedAirfoil?.coordinates;
    const reference=window.AirfoilBenchmarks?.current?.()||state.previewReference||state.result?.reference;
    const result=rows.length>0||Boolean(reference?.points?.length)||!$("literature-panel").classList.contains("hidden");
    const history=Object.values(state.liveHistory||{}).some(values=>Array.isArray(values)&&values.length);
    return {geometry:Boolean(geometry),results:result,exports:Boolean(state.job?.exports||state.job?.output_exports),
      field:hasFields(point?.fields)||hasFields(state.liveField?.field)||hasFields(state.liveField?.point?.fields)||Boolean(point?.boundary_layer?.upper?.length||point?.boundary_layer?.lower?.length),
      polar:new Set([...rows,...(reference?.points||[])].filter(row=>Number.isFinite(Number(row.cl))).map(row=>Number(row.alpha_deg))).size>1,
      diagnostics:history||Boolean(isCfdBackend(point?.backend)&&Object.keys(point?.diagnostics||{}).length),
      log:state.logLineCount>0};
  }
  function select(name,{focus=false}={}){
    const button=buttons.find(node=>node.dataset.workflowTarget===name),panel=panels[name];
    if(!button||button.hidden||!panel)return;
    selected=name;
    for(const [key,node] of Object.entries(panels))node.hidden=key!==name;
    for(const node of buttons){const active=node===button;node.classList.toggle("is-active",active);node.setAttribute("aria-selected",String(active));node.tabIndex=active?0:-1;}
    $("airfoil-parameter-title").textContent=panel.dataset.parameterTitle||name;
    if(name==="mesh"||name==="model")switchTab("geometry");
    if(name==="result-controls"&&($("tab-button-"+state.activeTab)?.hidden||state.activeTab==="geometry")){
      const visible=["results","field","polar","diagnostics","log"].find(tab=>!$("tab-button-"+tab).hidden);
      if(visible)switchTab(visible);
    }
    if(focus)panel.querySelector("input:not([hidden]),select,button,a[href]")?.focus({preventScroll:true});
  }
  function refresh(){
    scheduled=false;
    const active=selectedBackends(el.backend.value),low=active.includes("lowspeed"),gas=active.includes("transonic"),lbm=active.includes("lbm");
    const geometry=state.preview?.geometry||state.embeddedAirfoil?.coordinates;
    const gap=geometry?.x?.length>2?Math.hypot(geometry.x[0]-geometry.x.at(-1),geometry.y[0]-geometry.y.at(-1)):null;
    const lowCh=low&&el["lowspeed-topology"].value!=="sharp"&&(el["lowspeed-topology"].value==="ch_native"||gap===null||gap>1e-8);
    const data=availability(),flapAllowed=(low||lbm)&&!active.some(name=>["transonic","cavitation"].includes(name));
    const menuVisibility={mesh:data.geometry,flap:flapAllowed||$("gurney-enabled").checked,
      "result-controls":data.results||data.field||data.log||data.diagnostics||data.exports,display:data.geometry||data.field||data.results};
    for(const node of buttons)node.hidden=menuVisibility[node.dataset.workflowTarget]===false;
    for(const [name,available] of Object.entries({results:data.results,field:data.field,polar:data.polar,diagnostics:data.diagnostics,log:data.log}))$("tab-button-"+name).hidden=!available;
    $("tab-button-campaign").hidden=true;
    $("tab-button-field").textContent=["jxfoil","xfoil"].includes(currentResult()?.backend)?"Boundary layer":"Flow field";
    if($("tab-button-"+state.activeTab)?.hidden)switchTab("geometry");
    for(const id of ["lowspeed-resolution","lowspeed-topology"])if(field(id))field(id).hidden=!low;
    for(const id of ["lowspeed-start","lowspeed-mut-base"])if(field(id))field(id).hidden=!lowCh;
    if(field("cfd-mesh"))field("cfd-mesh").hidden=!gas&&!active.includes("cavitation");
    if(field("trailing-edge-thickness"))field("trailing-edge-thickness").hidden=!gas;
    if(field("preview-resolution"))field("preview-resolution").hidden=low||!active.some(isCfdBackend)&&!lbm;
    for(const node of document.querySelectorAll("[data-workflow-physics=lowspeed]"))node.hidden=!low;
    if(sourceExtra)sourceExtra.hidden=!low||!$("gurney-enabled").checked;
    if(streamlineLabel)streamlineLabel.hidden=!(low||gas)||active.includes("cavitation");
    if(streamlineLabel?.closest(".workflow-section"))streamlineLabel.closest(".workflow-section").hidden=!low&&!gas;
    $("streamline-controls").hidden=streamlineLabel?.hidden||!$("streamline-align").checked;
    $("streamline-note").hidden=streamlineLabel?.hidden||!$("streamline-align").checked;
    const profiles=el.backend.options;
    for(const option of profiles){
      const unsupported=$("gurney-enabled").checked&&selectedBackends(option.value).some(name=>["transonic","cavitation"].includes(name));
      option.hidden=unsupported&&option.value!==el.backend.value;
    }
    for(const id of ["geometry-streamlines","field-streamlines"]){
      const node=$(id),mesh=state.preview?.mesh||currentMesh();if(node)node.closest("label").hidden=!mesh?.streamlines;
    }
    $("field-snapshot").closest("label").hidden=!(state.result?.saved_snapshots?.length);
    $("result-backend").closest("label").hidden=$("result-backend").options.length<2;
    $("result-alpha").closest("label").hidden=$("result-alpha").options.length<2;
    $("study-load-results").hidden=$("study-load-results").disabled;
    $("open-saved-airfoil").hidden=true; // The saved-run list is already visible in Study.
    $("output-results").closest("section").hidden=$("output-results").dataset.available!=="true";
    const path=window.AirfoilStudies?.path||"",parent=path.replace(/[\\/][^\\/]*$/,"");
    const currentDirectory=state.job?.results_directory||"",currentLocation=$("current-run-location");
    currentLocation.hidden=!currentDirectory;currentLocation.textContent=currentDirectory?`This run: ${currentDirectory}`:"";
    $("study-result-location").textContent=path?`New runs: ${parent} / date_time_UTC_airfoil_…`:'Save the study to use its folder for new results. Unsaved studies use private results.';
    const folder=$("output-folder");if(folder){folder.placeholder=path?parent:"Private results folder (unsaved study)";folder.title="Leave empty to save beside the study. A chosen parent overrides this default for future runs; every run gets a unique date-and-time folder.";}
    if(buttons.find(node=>node.dataset.workflowTarget===selected)?.hidden)select("model");
  }
  function schedule(){if(!scheduled){scheduled=true;requestAnimationFrame(refresh);}}
  for(const button of buttons){
    const key=button.dataset.workflowTarget,panel=panels[key];
    if(panel){if(!button.id)button.id="workflow-tab-"+key;if(!panel.id)panel.id="workflow-panel-"+key;button.setAttribute("aria-controls",panel.id);panel.setAttribute("role","tabpanel");panel.setAttribute("aria-labelledby",button.id);}
    button.addEventListener("click",()=>select(button.dataset.workflowTarget));
    button.addEventListener("keydown",event=>{
      if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key))return;
      event.preventDefault();const visible=buttons.filter(node=>!node.hidden),index=visible.indexOf(button);
      const next=event.key==="Home"?visible[0]:event.key==="End"?visible.at(-1):visible[(index+(event.key==="ArrowRight"?1:-1)+visible.length)%visible.length];
      select(next.dataset.workflowTarget);next.focus();
    });
  }
  document.addEventListener("change",schedule);
  window.addEventListener("airfoil-job-updated",schedule);
  window.addEventListener("airfoil-reference-updated",schedule);
  if(typeof MutationObserver!=="undefined")for(const id of ["log-count","result-backend","result-alpha","field-kind","output-results","study-name"])
    new MutationObserver(schedule).observe($(id),{childList:true,subtree:id==="output-results"});
  window.AirfoilWorkflow={select,refresh:schedule,availability,get selected(){return selected;}};
  refresh();select("model");
})();
