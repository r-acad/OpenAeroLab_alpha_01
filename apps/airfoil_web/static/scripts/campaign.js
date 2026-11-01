"use strict";
// The batch worker survives the browser. Poll only while its compact tab is open.
(() => {
  const $=id=>document.getElementById(id),escape=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  let active=false,timer=null,pending=false;
  const pretty=v=>Number.isFinite(v)?Number(v).toPrecision(5):"—";
  const count=v=>Number.isSafeInteger(v)&&v>=0?v.toLocaleString():"—";
  const evidenceAliases=new Set(["recovery_report.txt","recovery_comparison.json","stall_reference_report.txt","stall_reference.json","stall_reference.png","ch_pilot_report.txt","ch_pilot_receipt.toml","ch_pilot_summary.json"]);
  function evidenceLink(url,label){
    const prefix="/api/campaign/files/";
    return typeof url==="string"&&url.startsWith(prefix)&&evidenceAliases.has(url.slice(prefix.length))?`<a href="${escape(url)}" target="_blank" rel="noopener">${escape(label)}</a>`:"";
  }
  function renderEvidence(data){
    const r=data.recovery_pilot||{},s=data.stall_reference_extension||{},c=data.ch_coupled_pilot||{};
    const status=v=>escape(String(v||"pending").replaceAll("_"," "));
    $("campaign-evidence").innerHTML=`<article class="campaign-evidence-card"><h4>A-plus recovery · unconverged</h4><p>${r.available?`${status(r.status)} · ${count(r.actual_iterations)} / ${count(r.requested_iterations)} iterations` : "Awaiting current diagnostic receipt"}</p><p>CL ${pretty(r.cl)} · CD ${pretty(r.cd)} · guard ${escape(r.guard_status||"pending")}${Number.isFinite(r.mass_convective_linf)?` · continuity L∞ ${pretty(r.mass_convective_linf)}`:""}</p><p class="evidence-warning">Full CFD sweeps held · no stall qualification</p>${evidenceLink(r.report_url,"Report")}${evidenceLink(r.comparison_url,"Data")}</article>
      <article class="campaign-evidence-card"><h4>XFOIL stall references · ${count(s.airfoils)} airfoils</h4><p>${s.available?`${count(s.supplement_converged)} / ${count(s.supplement_requested)} supplemental points converged · ${count(s.new_actual_executable_runs)} new executable runs`:"Reference extension pending"}</p><p>Complete sampled brackets: + ${count(s.positive_complete_brackets)} · − ${count(s.negative_complete_brackets)}</p><p>Failed/missing angles retained · CFD stall parity unestablished</p>${evidenceLink(s.report_url,"Report")}${evidenceLink(s.polar_url,"Polars")}${evidenceLink(s.data_url,"Data")}</article>
      <article class="campaign-evidence-card"><h4>C+H gas / SST pilot · diagnostic</h4><p>${status(c.status)} · ${count(c.actual_iterations)} / ${count(c.requested_iterations)} iterations</p><p>CL ${pretty(c.cl)} · CD ${pretty(c.cd)}${Number.isFinite(c.raw_mass_linf)?` · raw continuity L∞ ${pretty(c.raw_mass_linf)}`:""}</p><p>Local pseudo-time · convergence / stall unqualified${c.failure?` · ${escape(c.failure)}`:""}</p>${evidenceLink(c.report_url,"Report")}${evidenceLink(c.receipt_url,"Receipt")}${evidenceLink(c.summary_url,"Data")}</article>`;
  }
  function renderDevelopment(rows){
    const labels={running:"Running",prepared:"Prepared",refused:"Refused",stopped:"Stopped",completed:"Completed · unqualified",unavailable:"Unavailable"};
    const count=v=>Number.isSafeInteger(v)&&v>=0?v.toLocaleString():"—";
    $("campaign-development-summary").textContent=rows.length?`${rows.filter(r=>r.status==="running").length} running · ${rows.length} experimental arms · outside the full campaign queue`:"No solver development tests have been recorded.";
    $("campaign-development-rows").innerHTML=rows.length?rows.map(r=>{
      const status=Object.hasOwn(labels,r.status)?r.status:"unavailable",total=r.requested_iterations;
      const fraction=Number.isFinite(total)&&total>0&&Number.isFinite(r.accepted_iterations)?Math.max(0,Math.min(1,r.accepted_iterations/total)):0;
      const snapshot=r.sample_iteration!==r.iteration&&Number.isSafeInteger(r.sample_iteration)?` · coefficients at n ${count(r.sample_iteration)}`:"";
      return `<tr><td><strong>${escape(r.kind==="sfd"?"Transonic SFD":"Species boundary")}</strong><span>${escape(r.name)} · ${escape(r.arm)}</span></td><td class="trial-${status}" title="${escape(r.detail)}">${labels[status]}<progress max="1" value="${fraction}" aria-label="Accepted experimental iterations"></progress></td><td class="mono">${count(r.accepted_iterations)} / ${count(total)} accepted<span>total n ${count(r.iteration)}${snapshot}</span></td><td class="mono">${pretty(r.cl)}</td><td class="mono">${pretty(r.cd)}</td><td class="mono" title="${escape(r.mass_residual_definition)}">${pretty(r.mass_residual)}</td><td class="mono">${pretty(r.mass_net_frac)}</td></tr>`;
    }).join(""):'<tr><td colspan="7">No development tests available.</td></tr>';
  }
  function render(data){
    renderEvidence(data);
    renderDevelopment(Array.isArray(data.development_tests)?data.development_tests:[]);
    if(!data.available){$("campaign-state").textContent=data.message||"No campaign is available.";return;}
    $("campaign-state").textContent=String(data.queue_status||"pending").replaceAll("_"," ");
    const rows=data.airfoils||[],admitted=rows.filter(r=>r.mesh_status==="passed").length;
    const finished=rows.reduce((s,r)=>s+(r.transonic?.finished||0)+(r.cavitation?.finished||0),0);
    $("campaign-summary").textContent=`${admitted} admitted airfoils · ${finished} CFD points finished · ${data.telegram_delivered_files||0} Telegram files delivered · ${data.full_campaign_admitted?"full sweep admitted":"pilot assessment in progress"}`;
    const current=data.active;$("campaign-active").classList.toggle("hidden",!current);
    if(current){
      const name=current.backend==="cavitation"?"Species air":"Transonic";
      $("campaign-current").textContent=`${name} · NACA ${current.naca} · α ${current.alpha_deg}° · ${current.phase||current.status}`;
      $("campaign-progress").value=current.total_iterations>0?current.iteration/current.total_iterations:0;
      $("campaign-values").textContent=`n ${current.iteration} / ${current.total_iterations} · CL ${pretty(current.cl)} · CD ${pretty(current.cd)} · mass residual ${pretty(current.residual_scaled)}`;
    }
    const ref=q=>`${q?.converged||0}/${q?.attempted||0} converged`,cfd=q=>`${q?.finished||0}/31 finished · ${q?.continuity_guard_passed||0} guard passed${q?.failed?` · ${q.failed} failed`:""}${q?.interrupted?` · ${q.interrupted} interrupted`:""}`;
    $("campaign-rows").innerHTML=rows.map(r=>`<tr><td>${escape(r.naca)}</td><td class="${r.mesh_status==="failed"?"refused":""}">${escape(r.mesh_status)}${r.mesh_cells?` · ${Number(r.mesh_cells).toLocaleString()}`:""}</td><td>${ref(r.xfoil_tripped)}</td><td>${ref(r.xfoil_natural)}</td><td>${cfd(r.transonic)}</td><td>${cfd(r.cavitation)}</td><td>${r.polar_url&&/^\/api\/campaign\/files\/naca\d{4}_polars\.png$/.test(r.polar_url)?`<a href="${escape(r.polar_url)}" target="_blank" rel="noopener">View</a>`:"—"}</td></tr>`).join("");
    $("campaign-qualification").textContent=data.qualification||"CFD parity remains unqualified.";
  }
  async function refresh(){
    if(!active||pending)return;
    pending=true;
    try{
      const format=$("transport").value==="bson"?"bson":"msgpack";
      const response=await fetch("/api/campaign",{headers:{Accept:format==="bson"?"application/bson":"application/msgpack"},cache:"no-store"});
      if(!response.ok)throw new Error(`Campaign status HTTP ${response.status}; restart run.cmd if the Julia service is older.`);
      const bytes=await response.arrayBuffer();render(format==="bson"?window.AirfoilWire.decodeBson(bytes):window.AirfoilWire.decodeMsgpack(bytes));
    }catch(error){$("campaign-state").textContent=error.message;}
    finally{pending=false;if(active)timer=setTimeout(refresh,5000);}
  }
  window.AirfoilCampaign={activate(on){active=on;clearTimeout(timer);if(on)refresh();},render};
})();
