/* Aerodynamic frames and field controls adapted from Boxer. See common/THIRD_PARTY_NOTICES.md. */
((root)=>{
  'use strict';
  const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const rad=x=>Number(x||0)*Math.PI/180;
  function rotation(flow={}){const a=rad(flow.alpha),b=rad(flow.beta),ca=Math.cos(a),sa=Math.sin(a),cb=Math.cos(b),sb=Math.sin(b);return [[cb*ca,-sb,cb*sa],[sb*ca,cb,sb*sa],[-sa,0,ca]];}
  const multiply=(R,v)=>R.map(row=>dot(row,v)),transpose=R=>R[0].map((_,i)=>R.map(row=>row[i]));
  function frames(flow={},a0=0){const a=rad(flow.alpha),b=rad(flow.beta),s=rad(a0),ca=Math.cos(a),sa=Math.sin(a),cb=Math.cos(b),sb=Math.sin(b);return {
    body:[[-1,0,0],[0,1,0],[0,0,-1]],
    wind:[[-ca*cb,sb,-sa*cb],[ca*sb,cb,sa*sb],[sa,0,-ca]],
    stability:[[-Math.cos(s),0,-Math.sin(s)],[0,1,0],[Math.sin(s),0,-Math.cos(s)]]};}
  function coefficients(force,moment,origin,target,reference,flow={},ground=false,symmetry=false){
    const area=Number(reference.area),chord=Number(reference.chord),span=Number(reference.span)||chord,q=.5*Number(flow.density)*Number(flow.velocity)**2;
    if(!(area>0&&chord>0&&q>0)||![...force,...moment,...origin,...target].every(Number.isFinite))return null;
    const shift=cross(origin.map((v,i)=>v-target[i]),force),m=moment.map((v,i)=>v+shift[i]);
    const unground=v=>ground?[v[0],-v[2],v[1]]:v,Rt=transpose(rotation(flow)),F=multiply(Rt,unground(force)),M=multiply(Rt,unground(m));
    // Solver histories contain mirrored FULL-model loads. Never halve the area twice.
    const qS=q*area,out={q,area,effectiveArea:symmetry?area/2:area,force,origin:target,frames:{}};
    for(const [name,T] of Object.entries(frames(flow,reference.stability_alpha))){out.frames[name]={force:multiply(T,F).map(v=>v/qS),moment:multiply(T,M).map((v,i)=>v/(qS*(i===1?chord:span)))};}
    const w=out.frames.wind,b=out.frames.body;Object.assign(out,{CD:-w.force[0],CY:w.force[1],CL:-w.force[2],Cl:b.moment[0],Cm:b.moment[1],Cn:b.moment[2],LD:Math.abs(w.force[0])>1e-12?w.force[2]/w.force[0]:null});return out;
  }
  function surfaceSection(surface,values,axis,position,offset=[0,0,0],epsilon=1e-8){
    if(surface?.representation==='wall_samples')return root.LudwigAeroWallSurface?.section(surface,values,axis,position)||[];
    const vertices=surface?.vertices;if(!vertices||!values)return [];
    const out=[],n='xyz'.indexOf(axis);
    for(let face=0;face<values.length;face++){
      const value=values[face];if(!Number.isFinite(value))continue;
      const base=face*9,ds=[0,1,2].map(i=>vertices[base+i*3+n]+offset[n]-position);
      if(ds.every(d=>d>epsilon)||ds.every(d=>d< -epsilon)||ds.every(d=>Math.abs(d)<=epsilon))continue;
      const points=[];
      for(let i=0;i<3;i++){const j=(i+1)%3,a=ds[i],b=ds[j];if(a*b>0&&Math.abs(a)>epsilon&&Math.abs(b)>epsilon)continue;
        const t=Math.abs(a)<=epsilon?0:Math.abs(b)<=epsilon?1:a/(a-b);if(!Number.isFinite(t)||t<0||t>1)continue;
        const p=[0,1,2].map(k=>vertices[base+i*3+k]*(1-t)+vertices[base+j*3+k]*t+offset[k]);p[n]=position;
        if(!points.some(q=>Math.hypot(...p.map((v,k)=>v-q[k]))<=epsilon))points.push(p);
      }
      if(points.length===2)out.push({points,value});
    }return out;
  }
  function create({state,$,queueRender,currentFrame,addArrow,addLabel,addLines,keep,v3}){
    let limits={},lastRange=null,sectionCache=null;const ranges={};
    const quantity=()=>state.importedResults?$('results-field')?.value:$('surface-field')?.value;
    const target=()=>$('scale-target')?.value||'surface';
    const fieldFor=kind=>kind==='section'?$('section-field')?.value:kind==='recorded'?$('cut-field')?.value:quantity();
    const key=(kind=target(),field=fieldFor(kind))=>`${state.importedResults?.id||'native'}:${kind}:${field||'cp'}`;
    const read=()=>({area:Number($('reference-area')?.value)||null,chord:Number($('reference-chord')?.value)||null,span:Number($('reference-span')?.value)||null,
      point:['x','y','z'].map(a=>Number($('reference-'+a)?.value)||0),stability_alpha:Number($('stability-alpha')?.value)||0});
    function restore(ref={}){for(const [id,value] of [['reference-area',ref.area],['reference-chord',ref.chord],['reference-span',ref.span],['stability-alpha',ref.stability_alpha||0],...['x','y','z'].map((a,i)=>['reference-'+a,ref.point?.[i]||0])])if($(id))$(id).value=value??'';}
    function range(raw,kind='surface',field=fieldFor(kind)){const manual=limits[key(kind,field)];return raw&&manual?{...raw,lo:manual[0],hi:manual[1],mode:'manual'}:raw;}
    function syncRange(current,kind='surface',field=fieldFor(kind)){if(!current)return;ranges[key(kind,field)]=current;if(kind!==target())return;lastRange=current;for(const [id,value] of [['scale-min',current.lo],['scale-max',current.hi]])if($(id)&&document.activeElement!==$(id))$(id).value=Number(value.toPrecision(8));
      const lo=Math.min(current.actualLo,current.lo),hi=Math.max(current.actualHi,current.hi),size=hi-lo||1;for(const [id,value] of [['scale-min-slider',current.lo],['scale-max-slider',current.hi]])if($(id))$(id).value=String(Math.round(1000*(value-lo)/size));
      if($('scale-status'))$('scale-status').textContent=`${current.mode==='manual'?'Manual':'Automatic'} limits. Data: ${current.actualLo.toPrecision(5)} to ${current.actualHi.toPrecision(5)}. End colours clip out-of-range values.`;}
    function apply(lo,hi){if(!Number.isFinite(lo)||!Number.isFinite(hi)||!(lo<hi)||!Number.isFinite(hi-lo)){if($('scale-status'))$('scale-status').textContent='Enter finite limits with minimum less than maximum.';return false;}limits[key()]=[lo,hi];state.scalarMeshCache=null;queueRender();return true;}
    $('scale-apply')?.addEventListener('click',()=>{if(!$('scale-min').value.trim()||!$('scale-max').value.trim())return;apply(Number($('scale-min').value),Number($('scale-max').value));});
    $('scale-auto')?.addEventListener('click',()=>{delete limits[key()];state.scalarMeshCache=null;queueRender();});
    $('scale-target')?.addEventListener('change',()=>{lastRange=null;const saved=ranges[key()];if(saved)syncRange(saved,target());else if($('scale-status'))$('scale-status').textContent='Display this numeric field to set its scale. Saved PNG images retain their original colours.';queueRender();});
    for(const id of ['scale-min-slider','scale-max-slider'])$(id)?.addEventListener('input',()=>{if(!lastRange)return;const lo=Math.min(lastRange.actualLo,lastRange.lo),hi=Math.max(lastRange.actualHi,lastRange.hi),v=lo+(hi-lo)*Number($(id).value)/1000;apply(id.includes('min')?v:lastRange.lo,id.includes('max')?v:lastRange.hi);});
    const worldPoint=(frame,ref,flow)=>{let p=multiply(rotation(flow),ref.point);if(frame.ground)p=[p[0],p[2],-p[1]];p=p.map((v,i)=>v+(frame.bodyOrigin||frame.body?.center||[0,0,0])[i]);if(frame.boundary==='symmetry')p[1]=0;return p;};
    function update(frame=currentFrame()){
      const ref=read(),job=state.job,flow=state.activeConfig?.flow||frame.config?.flow||{},f=job?.monitoring?.dimensional_loads,holder=$('aero-coefficients');if(!holder)return;
      if(!(ref.area>0&&ref.chord>0)){holder.textContent='Enter the total wing area and reference chord to calculate coefficients. Saved dimensional loads are unchanged.';return;}
      if(!f?.force_N||!f?.moment_Nm||!frame.body||!state.activeConfig?.bounds){holder.textContent='No dimensional force history is loaded. Open a study with solver results to evaluate coefficients.';return;}
      const built=state.activeConfig,bc=built?.domain?.boundary,symmetric=bc==='symmetry',old=built?.reference;
      // Legacy runs used c = body X extent and a quarter-chord point at the staged mesh centre.
      const c=old?.chord||Math.max(built.bounds.max[0]-built.bounds.min[0],built.surface_dx||0),o=old?.point?worldPoint(frame,old,flow):[frame.body.min[0]+.25*c,symmetric?0:frame.body.center[1],frame.body.center[2]];
      const result=coefficients(f.force_N,f.moment_Nm,o,worldPoint(frame,ref,flow),ref,flow,frame.ground,symmetric);if(!result)return;
      holder.replaceChildren();const title=document.createElement('strong');title.textContent=`Saved loads · iteration ${f.step??'—'}`;holder.appendChild(title);
      const list=document.createElement('dl');for(const name of ['CL','CD','CY','Cm','Cl','Cn','LD']){const row=document.createElement('div'),label=document.createElement('dt'),value=document.createElement('dd');label.textContent=name==='LD'?'L/D':name;value.textContent=Number.isFinite(result[name])?result[name].toPrecision(6):'—';row.append(label,value);list.appendChild(row);}holder.appendChild(list);
      const note=document.createElement('p');note.className='hint';note.textContent=`q = ${result.q.toPrecision(5)} Pa · full S = ${ref.area} m² · effective solved S = ${result.effectiveArea} m². ${symmetric?'Mirrored full-model loads / full area; equivalent to half loads / half area. ':''}Cm is body pitch; Cl/Cn use ${ref.span?'span':'chord (no span entered)'}. Completion alone does not establish convergence.`;holder.appendChild(note);state.aeroCoefficients=result;
    }
    function draw(frame){if(!frame.body)return;const ref=read(),flow=frame.config?.flow||{alpha:Number($('flow-alpha')?.value),beta:Number($('flow-beta')?.value)},R=rotation(flow),origin=worldPoint(frame,ref,flow),size=Math.max(...frame.body.max.map((v,i)=>v-frame.body.min[i])),r=Math.max(.02,Math.min(size*.13,ref.chord?ref.chord*.7:size*.13));
      const world=v=>{const p=multiply(R,v);return frame.ground?[p[0],p[2],-p[1]]:p;},end=(v,l)=>world(v).map((x,i)=>origin[i]+x*l),colors=[[1,.55,.5],[.51,.88,.62],[.52,.75,1]],suffix={body:'B',wind:'W′',stability:'S'};
      for(const [name,T] of Object.entries(frames(flow,ref.stability_alpha)))if($('show-'+name+'-axes')?.checked)T.forEach((axis,i)=>addArrow(`${name} ${i}`,origin,end(axis,r*({body:1,wind:1.36,stability:1.55}[name])),colors[i],'axis aero-axis',`${'XYZ'[i]}_${suffix[name]}`));
      if($('show-moment-origin')?.checked){const sphere=keep(BABYLON.MeshBuilder.CreateSphere('moment origin',{diameter:r*.11,segments:12},state.scene));sphere.position=v3(...origin);sphere.isPickable=false;const material=keep(new BABYLON.StandardMaterial('moment origin material',state.scene));material.diffuseColor=new BABYLON.Color3(.3,.9,.6);material.emissiveColor=material.diffuseColor;sphere.material=material;addLabel('Moment origin',origin,'axis aero-axis');}
      if($('show-flow-angles')?.checked){const a=rad(flow.alpha),b=rad(flow.beta),projection=[Math.cos(a),0,Math.sin(a)],v=[projection[0]*Math.cos(b),-Math.sin(b),projection[2]*Math.cos(b)],glyph=end(v,-r*3);
        addArrow('relative wind',glyph,origin,[.8,.94,1],'axis aero-axis','−V');
        for(const [angle,basis,side,label,color,k] of [[a,[1,0,0],[0,0,1],'α',[1,.81,.47],1],[b,projection,[0,-1,0],'β',[.89,.67,1],1.35]]){const pts=Array.from({length:25},(_,i)=>world(basis.map((x,j)=>(x*Math.cos(angle*i/24)+side[j]*Math.sin(angle*i/24))*r*k)).map((x,j)=>x+glyph[j]));addLines(label+' arc',pts.slice(1).map((p,i)=>[pts[i],p]),new BABYLON.Color3(...color));addLabel(`${label} = ${(angle*180/Math.PI).toFixed(2)}°`,pts[12],'axis aero-axis');}
      }update(frame);
    }
    function section(axis,position){const surface=state.importedResults?state.importedSurface:state.surface,values=surface?.fields?.[quantity()]??surface?.[quantity()],offset=surface?.surface_offset||[0,0,0];if(!surface||!values)return [];
      if(sectionCache?.surface!==surface||sectionCache?.values!==values||sectionCache.axis!==axis||sectionCache.position!==position)sectionCache={surface,values,axis,position,segments:surfaceSection(surface,values,axis,position,offset)};return sectionCache.segments;}
    function paintCurves(ctx,segments,axes,px,py,canvas){if(!$('section-surface-curves')?.checked||!segments.length)return;
      let lo=Infinity,hi=-Infinity;for(const s of segments){lo=Math.min(lo,s.value);hi=Math.max(hi,s.value);}const span=hi-lo||1,top=92,right=canvas.width-105;
      ctx.save();ctx.fillStyle='rgba(245,250,255,.91)';ctx.fillRect(0,0,canvas.width,top);ctx.fillRect(right,top,105,canvas.height-top);ctx.strokeStyle='#354f66';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(35,top);ctx.lineTo(right,top);ctx.lineTo(right,canvas.height-35);ctx.stroke();
      for(const [direction,color] of [[0,'#bc3f37'],[1,'#265bb1']]){ctx.strokeStyle=color;ctx.beginPath();for(const s of segments){const f=(s.value-lo)/span;const p=s.points.map(point=>direction===0?[px(point[axes[0]]),top-12-f*55]:[right+12+f*70,py(point[axes[1]])]);ctx.moveTo(...p[0]);ctx.lineTo(...p[1]);}ctx.stroke();}
      ctx.fillStyle='#18364b';ctx.font='11px sans-serif';ctx.fillText(`${quantity()} section curves · ${lo.toPrecision(4)} … ${hi.toPrecision(4)}`,40,17);ctx.fillText(`vs ${'XYZ'[axes[0]]} (top) / ${'XYZ'[axes[1]]} (right)`,40,32);ctx.restore();
    }
    for(const id of ['reference-area','reference-chord','reference-span','reference-x','reference-y','reference-z','stability-alpha','show-moment-origin','show-body-axes','show-wind-axes','show-stability-axes','show-flow-angles'])$(id)?.addEventListener('change',()=>{if($('boundary')?.value==='symmetry'&&$('reference-y'))$('reference-y').value='0';update();queueRender();});
    return {read,restore,range,syncRange,draw,update,section,paintCurves,capture:()=>({limits,flags:Object.fromEntries(['show-moment-origin','show-body-axes','show-wind-axes','show-stability-axes','show-flow-angles'].map(id=>[id,!!$(id)?.checked]))}),restoreView(view){limits={};for(const [k,v] of Object.entries(view?.limits||{}))if(Array.isArray(v)&&v.length===2&&v.every(Number.isFinite)&&v[0]<v[1])limits[k]=v.slice();for(const [id,value] of Object.entries(view?.flags||{}))if($(id)&&$(id).type==='checkbox')$(id).checked=value===true;}};
  }
  root.LudwigAeroReference={rotation,multiply,transpose,frames,coefficients,surfaceSection,create};
  if(typeof module==='object'&&module.exports)module.exports=root.LudwigAeroReference;
})(globalThis);
