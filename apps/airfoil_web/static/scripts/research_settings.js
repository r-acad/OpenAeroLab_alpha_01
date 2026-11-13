/* Explicit saved research settings. Blank controls retain automatic defaults. */
window.AirfoilResearchSettings = (() => {
  "use strict";
  const $=id=>document.getElementById(id);
  const profiles=new Set(["aplus_uniform_quality","aplus_uniform_fixed"]);
  // Earlier production-admitted families only. Private MUSCL/precision
  // prototypes require engine workspaces that this page does not construct.
  const savedPreconditioners=new Set(["block_adi","cut_block_adi","swept_line","cut_swept_line","kinetic_block_adi","kinetic_fullvisc_block_adi","kinetic_weno_fullvisc_block_adi","kinetic_block_unfloored_adi","kinetic_weno_fullvisc_unfloored_adi"]);
  const numbers={
    cfd_implicit_linear_maxiter:["implicit-linear-maxiter",1,10000,true],
    cfd_implicit_max_cfl:["implicit-max-cfl",.01,1e5,false],
    cfd_implicit_physical_fraction:["implicit-physical-fraction",.01,.5,false],
    cfd_implicit_linear_tolerance:["implicit-linear-tolerance",1e-6,.9,false]
  };
  const keys=["cfd_implicit_globalization",...Object.keys(numbers)];
  function meshProfile(value=$("cfd-mesh").value){return profiles.has(value);}
  function restore(flow={}){
    if(meshProfile(flow.cfd_mesh||"")){
      $("cfd-mesh").value=flow.cfd_mesh;
      if(flow.cfd_n_base!=null)$("cfd-n-base").value=String(flow.cfd_n_base);
      if(flow.cfd_resolution!=null)for(const id of ["cfd-resolution","preview-resolution"])$(id).value=String(flow.cfd_resolution);
    }
    for(const [key,id] of Object.entries({rans_method:"rans-method",cfd_implicit_warmup_steps:"implicit-warmup",cfd_implicit_cfl:"implicit-cfl",cfd_implicit_basis:"implicit-basis",cfd_implicit_sst_substeps:"implicit-sst-substeps",cfd_implicit_nonmonotone_window:"implicit-nonmonotone",cfd_implicit_recovery_steps:"implicit-recovery"}))if(flow[key]!=null)$(id).value=String(flow[key]);
    $("implicit-globalization").value=flow.cfd_implicit_globalization??"";
    for(const [key,[id]] of Object.entries(numbers))$(id).value=flow[key]==null?"":String(flow[key]);
    // Additional admitted preconditioners need not clutter an ordinary case,
    // but a saved explicit selection must remain a real select option.
    const value=flow.cfd_implicit_preconditioner,select=$("implicit-preconditioner");
    for(const option of [...select.options])if(option.dataset.savedResearch)option.remove();
    if(typeof value==="string"&&![...select.options].some(option=>option.value===value)){
      const option=document.createElement("option");option.value=value;
      const available=savedPreconditioners.has(value);
      option.textContent=`${available?"Saved research preconditioner":"Unavailable saved preconditioner"} · ${value}`;
      option.dataset.savedResearch="true";option.disabled=!available;
      if(!available)option.dataset.unavailableResearch="true";
      select.append(option);
    }
    if(value!=null)select.value=String(value);
  }
  function collect(flow,{implicit,newton,allowUnavailable=false}){
    // API-only implicit options from older studies must not survive a switch
    // to a solver or time mode for which they have no meaning.
    if(!implicit){for(const key of Object.keys(flow))if(key.startsWith("cfd_implicit_"))delete flow[key];return;}
    const selected=[...$("implicit-preconditioner").options].find(option=>option.value===$("implicit-preconditioner").value);
    if(newton&&selected?.dataset.unavailableResearch&&!allowUnavailable)throw new Error("The saved research preconditioner is not available in this web app. Choose an available preconditioner before running.");
    if(!newton)delete flow.cfd_implicit_preconditioner;
    for(const key of keys)delete flow[key];
    const globalization=$("implicit-globalization").value;
    if(newton&&globalization)flow.cfd_implicit_globalization=globalization;
    for(const [key,[id,min,max,integer]] of Object.entries(numbers)){
      if(!newton&&key!=="cfd_implicit_max_cfl")continue;
      const text=$(id).value.trim();if(!text)continue;
      const value=Number(text),limit=key==="cfd_implicit_max_cfl"&&globalization!=="pseudo_transient"?10:max;
      if(!Number.isFinite(value)||value<min||value>limit||integer&&!Number.isInteger(value))throw new Error(`Enter a valid ${$(id).closest("label").querySelector("span").textContent}.`);
      flow[key]=value;
    }
    if(flow.cfd_implicit_max_cfl!=null&&flow.cfd_implicit_max_cfl<flow.cfd_implicit_cfl)throw new Error("Maximum pseudo CFL must be at least the initial pseudo CFL.");
  }
  function refresh({gas=false,species=false,implicit=false,newton=false}={}){
    if(species&&!gas&&meshProfile())$("cfd-mesh").value="aplus_nested_refined";
    const cflLimit=newton&&$("implicit-globalization").value==="pseudo_transient"?1e5:10;
    $("implicit-cfl").max=String(cflLimit);$("implicit-max-cfl").max=String(cflLimit);
    for(const id of ["implicit-globalization","implicit-linear-maxiter","implicit-physical-fraction","implicit-linear-tolerance"]){
      $(id).closest("label").hidden=!gas||!implicit||!newton;$(id).disabled=!gas||!implicit||!newton;
    }
    $("implicit-max-cfl").closest("label").hidden=!gas||!implicit;
    $("implicit-max-cfl").disabled=!gas||!implicit;
    for(const option of $("cfd-mesh").options)if(profiles.has(option.value))option.hidden=!gas;
    const explicit=meshProfile()||implicit&&keys.some(key=>key==="cfd_implicit_globalization"?newton&&$("implicit-globalization").value!=="":(newton||key==="cfd_implicit_max_cfl")&&$(numbers[key][0]).value.trim()!=="")||newton&&Boolean([...$("implicit-preconditioner").options].find(option=>option.value===$("implicit-preconditioner").value)?.dataset.savedResearch);
    const note=$("research-settings-note");note.hidden=!gas||!explicit;
    note.textContent=explicit?"Research settings are loaded or selected. They are retained when you save or run this steady compressible case. Reliable airfoil convergence and a speed advantage are not established. Blank optional fields use the solver defaults.":"";
    if(newton&&[...$("implicit-preconditioner").options].some(option=>option.value===$("implicit-preconditioner").value&&option.dataset.unavailableResearch))note.textContent+=" The saved preconditioner is unavailable here. Its name is retained when saving, but choose an available preconditioner before running.";
  }
  return {keys,meshProfile,restore,collect,refresh};
})();
