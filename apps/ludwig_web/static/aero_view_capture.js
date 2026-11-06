/* Images from Aero Lab's actual Babylon scene. Full desktop capture is separate. */
(() => {
  'use strict';
  const root=globalThis;
  function sameFrame(a,b){return !!a&&!!b&&a.jobId===b.job_id&&a.step===b.step&&a.representation==='wall_samples'&&a.sampleCount===b.sample_count&&a.geometrySha256===b.geometry_sha256&&a.frameSha256===b.frame_sha256&&a.field===b.field&&a.sourceUrl===b.source_url;}
  function identityHeaders(identity){return {'Content-Type':'image/png','X-Aero-Step':String(identity.step),'X-Aero-Samples':String(identity.sampleCount),'X-Aero-Geometry':identity.geometrySha256,'X-Aero-Frame':identity.frameSha256,'X-Aero-Representation':identity.representation,'X-Aero-Field':identity.field};}
  root.LudwigAeroViewCapture={sameFrame,identityHeaders};
  if(typeof module==='object'&&module.exports)module.exports=root.LudwigAeroViewCapture;
  if(!root.document)return;
  const dedicated=new URLSearchParams(root.location.search).get('reporting')==='1';
  let busy=false,lastRequest='',allow=dedicated;
  const context=()=>root.LudwigAeroResultsContext;
  const menu=document.querySelector('#menu-results .menu-popover');
  const group=document.createElement('div');group.className='menu-section';
  const title=document.createElement('h2');title.textContent='Images from Aero Lab';group.appendChild(title);
  const save=document.createElement('button');save.type='button';save.id='save-aero-view-png';save.textContent='Save 3D view PNG';
  save.title='Export the current rendered surface view with its field scale, frame and sample count.';group.appendChild(save);
  const label=document.createElement('label');label.className='check-field';const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.checked=allow;
  checkbox.title='Keep this tab open. Requested report images use the exact native pressure frame, your camera and the displayed field palette.';
  label.append(checkbox,document.createTextNode('Provide this view for Telegram reports'));group.appendChild(label);
  const status=document.createElement('p');status.className='hint';status.id='aero-view-capture-status';status.setAttribute('role','status');group.appendChild(status);
  if(menu)menu.appendChild(group);
  checkbox.addEventListener('change',()=>{allow=checkbox.checked;status.textContent=allow?'This tab can provide Aero Lab report images. Keep it open.':'Automatic view capture is off.';});
  async function renderPNG(identity,width=1600,height=1000){
    const ctx=context(),state=ctx?.state;if(!state?.engine||!state?.camera)throw new Error('The 3D view is not ready.');
    if(!identity||identity.representation!=='wall_samples')throw new Error('Choose native wall-sample surface results first.');
    const capture=root.BABYLON?.CreateScreenshotUsingRenderTargetAsync;
    if(!capture)throw new Error('The Babylon view export component is unavailable.');
    const picture=await capture(state.engine,state.camera,{width,height},'image/png');
    const image=new Image();image.src=picture;await image.decode();
    const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;const c=canvas.getContext('2d');
    c.drawImage(image,0,0,width,height);
    c.fillStyle='rgba(15,29,38,.94)';c.fillRect(0,0,width,76);c.fillRect(0,height-80,width,80);
    c.fillStyle='#edf5fa';c.font='bold 24px sans-serif';c.fillText(`Aero Lab · ${identity.field==='cp'?'Surface Cp':'Surface pressure [Pa]'}`,24,32);
    c.font='16px sans-serif';c.fillText(`Native wall samples · step ${identity.step} · ${Number(identity.sampleCount).toLocaleString()} samples`,24,59);
    const limits=identity.colourLimits,palette=identity.palette;
    if(Array.isArray(limits)&&limits.length===2&&limits.every(Number.isFinite)&&root.LudwigPalettes){
      const x=24,y=height-56,w=Math.min(560,width-80);
      for(let i=0;i<w;i++){const rgb=root.LudwigPalettes.valueColor(limits[0]+i/(w-1)*(limits[1]-limits[0]),limits[0],limits[1],palette);c.fillStyle=`rgb(${rgb.map(v=>Math.round(v*255)).join(',')})`;c.fillRect(x+i,y,1,15);}
      c.fillStyle='#edf5fa';c.fillText(Number(limits[0]).toPrecision(5),x,y+37);c.textAlign='right';c.fillText(Number(limits[1]).toPrecision(5),x+w,y+37);c.textAlign='left';
    }
    c.fillStyle='#edf5fa';c.font='14px sans-serif';c.fillText('Aero Lab renderer view · recorded fluid samples · convergence is reported separately',Math.min(620,width*.48),height-44);
    const blob=await new Promise((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(new Error('PNG export failed.')),'image/png'));
    if(!sameFrame(context()?.getRenderIdentity?.(),{job_id:identity.jobId,step:identity.step,representation:identity.representation,sample_count:identity.sampleCount,geometry_sha256:identity.geometrySha256,frame_sha256:identity.frameSha256,field:identity.field,source_url:identity.sourceUrl}))throw new Error('The result frame changed during capture; retry with a stable view.');
    return blob;
  }
  save.addEventListener('click',async()=>{if(busy)return;busy=true;save.disabled=true;try{const id=context()?.getRenderIdentity?.();const blob=await renderPNG(id);const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download=`AeroLab_${id.field}_step_${id.step}.png`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);status.textContent='PNG exported from the Aero Lab renderer.';}catch(e){status.textContent=e.message;}finally{busy=false;save.disabled=false;}});
  async function poll(){
    if(!allow||busy||document.hidden)return;
    const ctx=context(),jobId=ctx?.state?.jobId;if(!jobId)return;
    busy=true;
    let activeRequest=null;
    try{
      const response=await fetch(`/api/aero/jobs/${encodeURIComponent(jobId)}/view-captures`,{cache:'no-store',headers:{'X-Aero-View-Client':'viewport'}});if(!response.ok)return;
      const request=(await response.json()).request;if(!request||request.request_id===lastRequest)return;
      activeRequest=request;
      status.textContent=`Preparing Aero Lab image for step ${request.step}…`;
      await ctx.prepareReportView({jobId,step:request.step,sourceUrl:request.source_url,geometrySha256:request.geometry_sha256,frameSha256:request.frame_sha256,field:request.field,fit:dedicated&&!lastRequest});
      const identity=ctx.getRenderIdentity();if(!sameFrame(identity,request))throw new Error('Aero Lab has not finished rendering the requested native frame.');
      const blob=await renderPNG(identity,request.width,request.height);
      const uploaded=await fetch(`/api/aero/jobs/${jobId}/view-captures/${request.request_id}`,{method:'POST',headers:identityHeaders(identity),body:blob});
      if(!uploaded.ok)throw new Error((await uploaded.json()).error||'The image could not be saved.');
      lastRequest=request.request_id;status.textContent=`Aero Lab image saved for step ${request.step}.`;
    }catch(e){status.textContent=e.message;if(activeRequest)await fetch(`/api/aero/jobs/${jobId}/view-captures/${activeRequest.request_id}/error`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({error:e.message})}).catch(()=>{});}finally{busy=false;}
  }
  setInterval(poll,3000);root.addEventListener('ludwig-aero-view-ready',poll);
})();
