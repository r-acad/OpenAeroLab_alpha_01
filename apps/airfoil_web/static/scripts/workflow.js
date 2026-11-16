/* Airfoil Lab workflow: the one step selector, what each step shows for the selected solver, and the Advanced groups.
 * Controls stand in their step in index.html; nothing is cloned. */
(() => {
  "use strict";
  const $=id=>document.getElementById(id);
  const panels=Object.fromEntries([...document.querySelectorAll("[data-airfoil-panel]")].map(node=>[node.dataset.airfoilPanel,node]));
  const buttons=[...document.querySelectorAll("[data-workflow-target]")];
  const usable=node=>!node.hidden&&!node.disabled;
  const renamed={flap:"model",sweep:"flow"};   // steps named in studies saved by earlier pages
  let selected="model",scheduled=false;
  const field=id=>$(id)?.closest("label");

  // The output controls are generated at load by output_controls.js (shared with Aero Lab). Their rarely needed part
  // joins the Advanced group of this step; the nodes are moved, so their listeners and saved values stay.
  const outputsAdvanced=$("advanced-outputs")?.querySelector(".advanced-body"),reports=$("output-controls")?.querySelector(".output-makie-options");
  if(outputsAdvanced){
    for(const node of [field("display-every"),$("snapshot-interval-field")])if(node)outputsAdvanced.append(node);
    if(reports){
      const summary=reports.querySelector("summary"),heading=document.createElement("h3");
      if(summary){heading.append(...summary.childNodes);summary.remove();}
      outputsAdvanced.append(heading,...reports.childNodes);reports.remove();
    }
  }

  // Advanced groups: one per step, closed by default. A control applies when nothing between it and its group hides it.
  const advanced=[...document.querySelectorAll(".setup-column details.advanced-group")];
  const valueOf=node=>node.type==="checkbox"?node.checked:node.value;
  const controlsOf=group=>[...group.querySelectorAll("input,select")];
  // The page default is what the markup says (not what a reloaded browser tab restored into the form).
  const pageDefault=node=>node.type==="checkbox"?node.hasAttribute("checked"):node.tagName==="SELECT"?
    (([...node.options].find(option=>option.hasAttribute("selected"))||node.options[0])?.value??""):node.getAttribute("value")??"";
  // A number field holds its default whatever its spelling (1e-6 and 0.000001).
  function atDefault(node){
    const value=valueOf(node),initial=pageDefault(node);
    return value===initial||node.type==="number"&&value!==""&&initial!==""&&Number(value)===Number(initial);
  }
  function applies(node,group){
    for(let parent=node;parent&&parent!==group;parent=parent.parentElement)if(parent.hidden||parent.classList.contains("hidden"))return false;
    return true;
  }
  // Open every group in which a control that applies holds a value other than the page default: a reopened study, a
  // saved job or a preset uses it. A disabled control holds a value the selected method fixes, not one the setup chose.
  // Called when a setup is loaded, not on every edit, so a group the user closed stays closed. collapse also closes the
  // groups that hold only defaults (a new or freshly opened setup).
  function revealAdvanced({collapse=false}={}){
    refresh();
    for(const group of advanced){
      const used=controlsOf(group).some(node=>applies(node,group)&&!node.disabled&&!atDefault(node));
      if(used)group.open=true;else if(collapse)group.open=false;
    }
  }

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
    name=renamed[name]||name;
    const button=buttons.find(node=>node.dataset.workflowTarget===name),panel=panels[name];
    if(!button||!usable(button)||!panel)return;
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
  // The numbered step after the selected one that the selected solver uses (a panel method has no Mesh step).
  function next(){
    const steps=buttons.filter(node=>node.classList.contains("workflow-step")),at=steps.findIndex(node=>node.dataset.workflowTarget===selected);
    const target=steps.slice(at+1).find(usable);
    if(target)select(target.dataset.workflowTarget,{focus:true});
  }
  function refresh(){
    scheduled=false;
    const active=selectedBackends(el.backend.value),low=active.includes("lowspeed"),gas=active.includes("transonic"),cav=active.includes("cavitation"),lbm=active.includes("lbm");
    const underwater=cav&&el["cav-domain"].value==="underwater",lowCh=low&&lowspeedUsesCh(),meshed=active.some(isCfdBackend);
    const data=availability(),flapAllowed=(low||lbm)&&!gas&&!cav;
    // Steps stay in the row in their order. Mesh is disabled for a panel method, Results until there is something to show.
    const stepState={mesh:[meshed,"The selected solver is a panel method: it needs no mesh."],
      "result-controls":[data.results||data.field||data.log||data.diagnostics||data.exports,"Results appear here after a run, or when a saved result is opened."]};
    for(const node of buttons){
      const key=node.dataset.workflowTarget;
      if(stepState[key]){node.disabled=!stepState[key][0];node.title=stepState[key][0]?"":stepState[key][1];}
      if(key==="display")node.hidden=!(data.geometry||data.field||data.results);
    }
    for(const [name,available] of Object.entries({results:data.results,field:data.field,polar:data.polar,diagnostics:data.diagnostics,log:data.log}))$("tab-button-"+name).hidden=!available;
    $("tab-button-field").textContent=["jxfoil","xfoil"].includes(currentResult()?.backend)?"Boundary layer":"Flow field";
    if($("tab-button-"+state.activeTab)?.hidden)switchTab("geometry");
    // Airfoil: the Gurney flap belongs to the low-speed CFD and to LBM 3D.
    $("gurney-group").hidden=!(flapAllowed||$("gurney-enabled").checked);
    for(const option of el.backend.options){
      const unsupported=$("gurney-enabled").checked&&selectedBackends(option.value).some(name=>["transonic","cavitation"].includes(name));
      option.hidden=unsupported&&option.value!==el.backend.value;
    }
    // Flow, Mesh and Solver settings: the fields of another solver are not shown.
    for(const node of document.querySelectorAll("[data-workflow-physics]"))node.hidden=!active.includes(node.dataset.workflowPhysics);
    for(const id of ["create-mesh","create-mesh-step"])$(id).hidden=!meshed;
    field("preview-resolution").hidden=!(gas||cav)||underwater;
    field("lowspeed-resolution").hidden=!low;
    // On a sharp trailing edge the low-speed Fine mesh is the Standard mesh (the same generator settings): it is offered
    // where it changes the mesh, on C+H.
    for(const option of el["lowspeed-resolution"].options)if(option.value==="fine"){
      option.hidden=!lowCh&&el["lowspeed-resolution"].value!=="fine";
      option.textContent=lowCh?"Fine (four times the cells, slower)":"Fine (the Standard mesh on a sharp trailing edge)";
    }
    for(const id of ["lowspeed-start","lowspeed-mut-base"])field(id).hidden=!lowCh;
    field("trailing-edge-thickness").hidden=!(gas||cav);
    field("gurney-refine").hidden=!low||!$("gurney-enabled").checked;
    const streamline=field("streamline-align");
    streamline.hidden=!(low||gas)||cav||newtonMethodSelected(active);   // the marched mesh of the Newton method is not aligned
    $("streamline-controls").hidden=streamline.hidden||!$("streamline-align").checked;
    $("streamline-note").hidden=streamline.hidden||!$("streamline-align").checked;
    // An Advanced group with nothing for the selected solver is not shown.
    for(const group of advanced)group.hidden=!controlsOf(group).some(node=>applies(node,group));
    // Results and displays.
    for(const id of ["geometry-streamlines","field-streamlines"]){
      const node=$(id),mesh=state.preview?.mesh||currentMesh();if(node)node.closest("label").hidden=!mesh?.streamlines;
    }
    $("field-snapshot").closest("label").hidden=!(state.result?.saved_snapshots?.length);
    $("result-backend").closest("label").hidden=$("result-backend").options.length<2;
    $("result-alpha").closest("label").hidden=$("result-alpha").options.length<2;
    $("study-load-results").hidden=$("study-load-results").disabled;
    $("exported-files").hidden=$("output-results").dataset.available!=="true";
    const path=window.AirfoilStudies?.path||"",parent=path.replace(/[\\/][^\\/]*$/,"");
    const currentDirectory=state.job?.results_directory||"",currentLocation=$("current-run-location");
    currentLocation.hidden=!currentDirectory;currentLocation.textContent=currentDirectory?`This run: ${currentDirectory}`:"";
    $("study-result-location").textContent=path?`New runs: ${parent} / date_time_UTC_airfoil_…`:'Save the study to use its folder for new results. Unsaved studies use private results.';
    const folder=$("output-folder");if(folder){folder.placeholder=path?parent:"Private results folder (unsaved study)";folder.title="Leave empty to save beside the study. A chosen parent overrides this default for future runs; every run gets a unique date-and-time folder.";}
    const current=buttons.find(node=>node.dataset.workflowTarget===selected);
    if(current&&!usable(current))select("model");
  }
  function schedule(){if(!scheduled){scheduled=true;requestAnimationFrame(refresh);}}
  // A step ends with its action. The button in the step presses the one existing button, so there is one handler and
  // one busy state: Create mesh in the geometry view, Run analysis below the steps.
  function mirror(copyId,sourceId){
    const copy=$(copyId),source=$(sourceId);if(!copy||!source)return;
    const sync=()=>{copy.disabled=source.disabled;copy.textContent=source.textContent;};
    copy.addEventListener("click",()=>{if(!source.disabled)source.click();});sync();
    if(typeof MutationObserver!=="undefined")new MutationObserver(sync).observe(source,{attributes:true,childList:true,characterData:true,subtree:true});
  }
  mirror("create-mesh-step","create-mesh");mirror("run-step-button","run-button");
  for(const button of document.querySelectorAll("[data-workflow-next]"))button.addEventListener("click",next);
  for(const button of buttons){
    const key=button.dataset.workflowTarget,panel=panels[key];
    if(panel){if(!button.id)button.id="workflow-tab-"+key;if(!panel.id)panel.id="workflow-panel-"+key;button.setAttribute("aria-controls",panel.id);panel.setAttribute("role","tabpanel");panel.setAttribute("aria-labelledby",button.id);}
    button.addEventListener("click",()=>select(button.dataset.workflowTarget));
    button.addEventListener("keydown",event=>{
      if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key))return;
      event.preventDefault();const visible=buttons.filter(usable),index=visible.indexOf(button);
      const target=event.key==="Home"?visible[0]:event.key==="End"?visible.at(-1):visible[(index+(event.key==="ArrowRight"?1:-1)+visible.length)%visible.length];
      select(target.dataset.workflowTarget);target.focus();
    });
  }
  document.addEventListener("change",schedule);
  window.addEventListener("airfoil-job-updated",schedule);
  window.addEventListener("airfoil-reference-updated",schedule);
  if(typeof MutationObserver!=="undefined")for(const id of ["log-count","result-backend","result-alpha","field-kind","output-results","study-name"])
    new MutationObserver(schedule).observe($(id),{childList:true,subtree:id==="output-results"});
  window.AirfoilWorkflow={select,refresh:schedule,availability,revealAdvanced,get selected(){return selected;}};
  refresh();select("model");
})();
