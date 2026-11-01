"use strict";

/* Interactive mesh inspection and saved-data sections. Moving a plane never
   substitutes a neighbouring field for the requested physical position. */
(() => {
  function create(context){
    const {state,$,v3,keep,addQuad,addLines,addLabel,queueRender,currentFrame,scalarColor,scalarRange,
      api,apiUrl,renderCutRows,invalidateMesh,selectResultPane,clearCameraMotion,referenceTools}=context;
    const mesh=window.LudwigMeshInspector,axes=mesh.axes;
    const defaultInspector=()=>({enabled:false,axis:'z',position:null,field:'mesh',clip:true,mesh:true,data:null,key:null,error:'',loading:false});
    state.levelVisibility={};state.inspector=defaultInspector();state.meshVisible=state.meshVisible!==false;
    let requestTimer=null,dragging=false,sectionKey=null,sectionData=null,geometryKey=null,geometryData=null,geometryTriangles=null;
    let shiftPressed=false;window.addEventListener('keydown',event=>{if(event.key==='Shift')shiftPressed=true;});window.addEventListener('keyup',event=>{if(event.key==='Shift')shiftPressed=false;});window.addEventListener('blur',()=>{shiftPressed=false;});
    let activeDrag=null;const meshHiddenByUser=new Map();
    let sectionView={zoom:1,pan:[0,0]},fieldBitmap=null,fieldBitmapData=null,fieldBitmapPalette=null;
    const options=()=>state.inspector;
    const fieldRange=data=>{const raw=scalarRange(data.values),range=referenceTools?.range(raw,'section',data.field||options().field)||raw;referenceTools?.syncRange(range,'section',data.field||options().field);return range;};
    const amr=()=>state.importedResults?(state.importedResults.amr||null):state.amr;
    const inspectionFrame=()=>dragging&&state.frame?state.frame:currentFrame();
    const isAmr=item=>Boolean(item?.metadata?.amrMesh||item?.name?.startsWith('AMR '));
    const amrMeshes=()=>Array.from(state.scene?.meshes||state.sceneObjects||[]).filter(isAmr);
    function setMeshVisible(value){
      state.meshVisible=Boolean(value);
      for(const item of amrMeshes()){
        if(!state.meshVisible&&!meshHiddenByUser.has(item))meshHiddenByUser.set(item,activeDrag?.visibility.get(item)??item.isEnabled());
        if(dragging||!state.meshVisible)item.setEnabled(false);
        else if(meshHiddenByUser.has(item)){item.setEnabled(meshHiddenByUser.get(item));meshHiddenByUser.delete(item);}
      }
      if(state.meshVisible&&!dragging)meshHiddenByUser.clear();
      if(!dragging)queueRender();return state.meshVisible;
    }
    function finishDrag(release=false){
      if(!activeDrag)return;
      const saved=activeDrag;activeDrag=null;dragging=false;
      if(release&&saved.behavior.dragging)saved.behavior.releaseDrag();
      for(const [item,enabled]of saved.visibility)if(!item.isDisposed?.()){
        item.setEnabled(isAmr(item)?state.meshVisible===false?false:meshHiddenByUser.get(item)??enabled:enabled);
      }
      if(state.meshVisible!==false)meshHiddenByUser.clear();
      clearCameraMotion();queueRender();scheduleField();
    }
    const cancelDrag=()=>finishDrag(true),viewer=$('viewer')||state.scene?.getEngine?.().getRenderingCanvas?.();
    for(const event of ['pointercancel','lostpointercapture'])viewer?.addEventListener?.(event,cancelDrag);
    window.addEventListener?.('blur',cancelDrag);window.addEventListener?.('pointerup',cancelDrag);
    window.addEventListener?.('keydown',event=>{if(event.key==='Escape')cancelDrag();});
    function sync(){
      const cut=options(),frame=inspectionFrame(),a=axes.indexOf(cut.axis),min=frame.domain.min[a],max=frame.domain.max[a];
      if(!Number.isFinite(cut.position))cut.position=(min+max)/2;
      cut.position=Math.max(min,Math.min(max,cut.position));
      $('section-enabled').checked=cut.enabled;$('section-axis').value=cut.axis;$('section-field').value=cut.field;
      $('section-clip').checked=cut.clip;$('section-mesh').checked=cut.mesh;
      $('section-slider').min=String(min);$('section-slider').max=String(max);$('section-slider').step=String(Math.max(1e-8,(max-min)/1500));
      $('section-slider').value=String(cut.position);$('section-position').min=String(min);$('section-position').max=String(max);$('section-position').value=String(Number(cut.position.toPrecision(7)));
      $('section-note').hidden=!cut.enabled;$('section-canvas').hidden=!cut.enabled;$('section-2d-note').hidden=!cut.enabled;
    }
    function levelControls(){
      const holder=$('amr-level-list');holder.replaceChildren();
      const levels=mesh.levelInfo(amr());
      if(!levels.length){holder.textContent='Create a mesh to inspect its levels.';return;}
      for(const info of levels){
        const label=document.createElement('label');label.className='level-option';
        const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.checked=state.levelVisibility[info.id]!==false;
        checkbox.addEventListener('change',()=>{state.levelVisibility[info.id]=checkbox.checked;sectionKey=null;queueRender();});
        const swatch=document.createElement('span');swatch.className='level-swatch';swatch.style.background=`rgb(${mesh.palette[(info.id-1)%mesh.palette.length].map(c=>Math.round(c*255)).join(',')})`;
        const text=document.createElement('span');text.textContent=`Level ${info.id} · Δx ${info.dx.toPrecision(3)} m · ${info.count.toLocaleString()} blocks`;
        label.append(checkbox,swatch,text);holder.appendChild(label);
      }
    }
    function clipPlane(){
      const cut=options();if(!cut.enabled||!cut.clip)return null;
      const normal=[0,0,0];normal[axes.indexOf(cut.axis)]=1;const n=v3(...normal);return new BABYLON.Plane(n.x,n.y,n.z,-cut.position);
    }
    function instanceMatrices(points,size){
      const matrices=new Float32Array(points.length*16),matrix=BABYLON.Matrix.Identity();
      points.forEach((point,i)=>{const p=v3(...point);BABYLON.Matrix.ComposeToRef(new BABYLON.Vector3(size,size,size),BABYLON.Quaternion.Identity(),p,matrix);matrix.copyToArray(matrices,i*16);});
      return matrices;
    }
    function edgeTemplate(name){
      const vertices=[[-.5,-.5,-.5],[.5,-.5,-.5],[.5,.5,-.5],[-.5,.5,-.5],[-.5,-.5,.5],[.5,-.5,.5],[.5,.5,.5],[-.5,.5,.5]],
        edges=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
      const lines=keep(BABYLON.MeshBuilder.CreateLineSystem(name,{lines:edges.map(pair=>pair.map(i=>new BABYLON.Vector3(...vertices[i]))),useVertexAlpha:false},state.scene));
      lines.alpha=1;lines.isPickable=false;lines.metadata={amrMesh:true};lines.material.disableDepthWrite=false;lines.material.clipPlane=clipPlane();return lines;
    }
    function supportsOit(){
      const engine=state.scene.getEngine(),caps=engine.getCaps();
      return Boolean((engine.isWebGPU||engine.webGLVersion>=2)&&caps.drawBuffersExtension&&caps.textureFloatRender&&caps.depthTextureExtension);
    }
    function renderFaces(faces,translucent,opacity){
      if(!faces.length)return false;
      let oit=false;
      if(translucent&&supportsOit())try{state.scene.useOrderIndependentTransparency=true;state.scene.depthPeelingRenderer.passCount=8;oit=true;}catch{state.scene.useOrderIndependentTransparency=false;}
      const plane=keep(BABYLON.MeshBuilder.CreatePlane('AMR voxel faces',{size:1},state.scene));plane.isPickable=false;
      const material=keep(new BABYLON.StandardMaterial('AMR voxel material',state.scene));material.diffuseColor=BABYLON.Color3.White();material.emissiveColor=new BABYLON.Color3(.22,.22,.22);
      material.specularColor=BABYLON.Color3.Black();material.alpha=translucent?opacity:1;material.transparencyMode=translucent?BABYLON.Material.MATERIAL_ALPHABLEND:BABYLON.Material.MATERIAL_OPAQUE;
      material.backFaceCulling=true;material.disableDepthWrite=translucent;material.clipPlane=clipPlane();plane.material=material;
      // Per-face GPU instances share one quad and one material. The instance
      // colour is independent of level, so fallback sorting spans all levels.
      plane.setVerticesData(BABYLON.VertexBuffer.ColorKind,new Float32Array(16).fill(1));plane.useVertexColors=true;
      const records=faces.map(face=>{const n=[0,0,0];n[face.normal]=face.sign;const rotation=BABYLON.Quaternion.Identity();BABYLON.Quaternion.FromUnitVectorsToRef(new BABYLON.Vector3(0,0,-1),v3(...n),rotation);
        return {face,position:v3(...face.center),rotation};});
      const matrices=new Float32Array(records.length*16),colors=new Float32Array(records.length*4),matrix=BABYLON.Matrix.Identity();
      function update(){
        if(translucent&&!oit&&state.camera){const eye=state.camera.position,direction=state.camera.getForwardRay().direction;for(const record of records)record.depth=(record.position.x-eye.x)*direction.x+(record.position.y-eye.y)*direction.y+(record.position.z-eye.z)*direction.z;records.sort((a,b)=>b.depth-a.depth);}
        records.forEach((record,i)=>{const face=record.face;BABYLON.Matrix.ComposeToRef(new BABYLON.Vector3(face.size,face.size,face.size),record.rotation,record.position,matrix);matrix.copyToArray(matrices,i*16);colors.set([...mesh.palette[(face.level-1)%mesh.palette.length],1],i*4);});
      }
      update();plane.thinInstanceSetBuffer('matrix',matrices,16,false);plane.thinInstanceSetBuffer('color',colors,4,false);
      if(translucent&&!oit){let previous='',last=0;const observer=state.scene.onBeforeRenderObservable.add(()=>{if(dragging||state.meshVisible===false||!plane.isEnabled()||!state.camera)return;const now=Date.now(),key=state.camera.position.asArray().concat(state.camera.target.asArray()).join(',');if(key===previous||now-last<80)return;previous=key;last=now;update();plane.thinInstanceBufferUpdated('matrix');plane.thinInstanceBufferUpdated('color');});keep({dispose(){state.scene.onBeforeRenderObservable.remove(observer);}});}
      plane.metadata={amrMesh:true,voxelFaces:faces.length,depthMode:translucent?(oit?'depth-peeling':'sorted-blending'):'opaque-depth',gpuInstanced:true};return oit;
    }
    function drawAmr(){
      if(dragging||state.meshVisible===false){$('amr-render-note').textContent=dragging?'Mesh hidden while moving the section.':'Mesh hidden. Use Show mesh to display it.';return;}
      const mode=$('amr-render').value,opacity=Number($('voxel-opacity').value)/100,faces=[];
      let displayed=0,total=0;
      for(const info of mesh.levelInfo(amr())){
        if(state.levelVisibility[info.id]===false||!info.count||!(info.size>0))continue;
        let points,size;
        if(mode==='blocks'){size=info.size;points=Array.from({length:info.count},(_,i)=>info.blocks.slice(i*3,i*3+3).map(v=>(v-.5)*size));total+=info.count;}
        else{const cells=mesh.volumeCells(amr(),info);points=cells.points;size=info.dx;total+=cells.total;if(mode!=='wire_voxels')for(const face of cells.faces)faces.push(face);}
        displayed+=points.length;if(!points.length||!['blocks','wire_voxels'].includes(mode))continue;
        const lines=edgeTemplate(`AMR level ${info.id}`);lines.color=new BABYLON.Color3(...mesh.palette[(info.id-1)%mesh.palette.length]);lines.thinInstanceSetBuffer('matrix',instanceMatrices(points,size),16,true);
      }
      const oit=faces.length?renderFaces(faces,mode==='translucent_voxels',opacity):false;
      $('amr-render-note').textContent=total?(mode==='blocks'?`${displayed.toLocaleString()} block outlines · GPU instances.`:`${displayed.toLocaleString()} boundary cells from ${total.toLocaleString()} allocated cells · exposed-cell envelope, no cell decimation · GPU instances.${mode==='translucent_voxels'?` ${oit?'Order-independent depth peeling.':'Depth-sorted transparent faces.'}`:''} Interior cell edges appear in the section.`):'No visible AMR levels.';
    }
    function planePoints(frame,axis,position){
      const n=axes.indexOf(axis),a=[0,1,2].filter(i=>i!==n),lo=frame.domain.min,hi=frame.domain.max;
      return [[lo[a[0]],lo[a[1]]],[hi[a[0]],lo[a[1]]],[hi[a[0]],hi[a[1]]],[lo[a[0]],hi[a[1]]]].map(pair=>{const p=[0,0,0];p[n]=position;p[a[0]]=pair[0];p[a[1]]=pair[1];return p;});
    }
    function getSection(){
      const cut=options(),key=JSON.stringify([state.importedResults?.id||state.amrUrl,cut.axis,cut.position,state.levelVisibility]);
      if(sectionKey!==key){sectionData=mesh.section(amr(),cut.axis,cut.position,state.levelVisibility);sectionKey=key;}
      return sectionData;
    }
    function getGeometrySection(){const cut=options(),frame=state.frame||currentFrame(),triangles=frame.triangles,key=[state.importedResults?.id||state.amrUrl||state.modelId,cut.axis,cut.position,triangles?.length].join(':');
      if(geometryKey!==key||geometryTriangles!==triangles){const extent=Math.max(...frame.domain.max.map((v,i)=>v-frame.domain.min[i]));geometryData=mesh.geometrySection(triangles,cut.axis,cut.position,Math.max(extent*1e-10,1e-9));geometryKey=key;geometryTriangles=triangles;}return geometryData;}
    function drawFieldPlane(data){
      if(!data?.values?.length)return null;
        const range=fieldRange(data);if(!range)return null;
      const normal=axes.indexOf(data.normal_axis),a=axes.indexOf(data.axis1),b=axes.indexOf(data.axis2),origin=data.origin_m,spacing=Number(data.spacing_m);
      const p=(i,j)=>{const v=origin.slice();v[normal]=Number(data.normal_coord_m);v[a]+=i*spacing;v[b]+=j*spacing;return v;};
      const plane=addQuad('section field',[p(0,0),p(data.n1,0),p(data.n1,data.n2),p(0,data.n2)],BABYLON.Color3.White(),1);
      const texture=keep(new BABYLON.DynamicTexture('section values',{width:data.n1,height:data.n2},state.scene,false,BABYLON.Texture.NEAREST_SAMPLINGMODE));
      const ctx=texture.getContext(),image=ctx.createImageData(data.n1,data.n2);
      for(let j=0;j<data.n2;j++)for(let i=0;i<data.n1;i++){
        const value=data.values[i+j*data.n1],offset=((data.n2-j-1)*data.n1+i)*4;
        if(typeof value!=='number'||!Number.isFinite(value))continue;
        const rgb=scalarColor(value,range);for(let k=0;k<3;k++)image.data[offset+k]=Math.round(rgb[k]*255);image.data[offset+3]=235;
      }
      ctx.putImageData(image,0,0);texture.hasAlpha=true;texture.update(false);
      plane.setVerticesData(BABYLON.VertexBuffer.UVKind,[0,1,1,1,1,0,0,0]);
      plane.material.diffuseTexture=texture;plane.material.emissiveTexture=texture;plane.material.useAlphaFromDiffuseTexture=true;plane.material.disableLighting=true;
      return plane;
    }
    function statusText(){
      const cut=options(),section=getSection(),base=`${cut.axis.toUpperCase()} = ${cut.position.toPrecision(5)} m · ${section.cells.length.toLocaleString()} section cells${section.stride>1?' (display sampled)':''}`;
      if(cut.field==='mesh')return base+` · ${getGeometrySection().segments.length.toLocaleString()} geometry intersection segments · drag the plane border along its normal, or use the slider.`;
      if(cut.loading)return base+' · loading saved field…';
      if(cut.data){const range=fieldRange(cut.data);if(!range)return base+' · No finite fluid values in the saved field at this position.';return base+` · ${cut.field} [${cut.data.units||'solver units'}] · step ${cut.data.step} · ${cut.data.source==='saved_volume'?'saved volume':'recorded plane'} · colour range ${range.lo.toPrecision(4)} to ${range.hi.toPrecision(4)}`;}
      return base+' · '+(cut.error||'No saved field at this position. Enable volume VTK snapshots or record this plane before running.');
    }
    function paint2d(){
      const canvas=$('section-canvas'),cut=options();canvas.hidden=!cut.enabled;
      if(!cut.enabled)return;
      const frame=currentFrame(),section=getSection(),a=section.axes,lo=frame.domain.min,hi=frame.domain.max;
      const width=hi[a[0]]-lo[a[0]],height=hi[a[1]]-lo[a[1]],baseScale=760/Math.max(width,height),pad=35,scale=baseScale*sectionView.zoom;
      canvas.width=Math.ceil(width*baseScale+pad*2);canvas.height=Math.ceil(height*baseScale+pad*2);
      const ctx=canvas.getContext('2d');if(!ctx)return;
      ctx.fillStyle='#f6fafb';ctx.fillRect(0,0,canvas.width,canvas.height);
      const px=x=>pad+(x-lo[a[0]])*scale+sectionView.pan[0],py=y=>canvas.height-pad-(y-lo[a[1]])*scale+sectionView.pan[1];
      const data=cut.data,range=data&&fieldRange(data);
      if(data&&range){
        const spacing=Number(data.spacing_m),i1=axes.indexOf(data.axis1),i2=axes.indexOf(data.axis2);
        const palette=[$('results-palette')?.value||'spectrum',range.lo,range.hi].join(':');
        if(fieldBitmapData!==data||fieldBitmapPalette!==palette){fieldBitmapPalette=palette;const bitmap=document.createElement('canvas'),imageContext=bitmap.getContext?.('2d');fieldBitmap=null;
          if(imageContext){bitmap.width=data.n1;bitmap.height=data.n2;const image=imageContext.createImageData(data.n1,data.n2);for(let j=0;j<data.n2;j++)for(let i=0;i<data.n1;i++){
            const value=data.values[i+j*data.n1];if(typeof value!=='number'||!Number.isFinite(value))continue;const rgb=scalarColor(value,range),offset=((data.n2-j-1)*data.n1+i)*4;for(let k=0;k<3;k++)image.data[offset+k]=Math.round(rgb[k]*255);image.data[offset+3]=255;}
            imageContext.putImageData(image,0,0);fieldBitmap=bitmap;}fieldBitmapData=data;}
        if(fieldBitmap&&ctx.drawImage){ctx.imageSmoothingEnabled=false;ctx.drawImage(fieldBitmap,px(Number(data.origin_m[i1])),py(Number(data.origin_m[i2])+data.n2*spacing),data.n1*spacing*scale,data.n2*spacing*scale);}
        else for(let j=0;j<data.n2;j++)for(let i=0;i<data.n1;i++){const value=data.values[i+j*data.n1];if(typeof value!=='number'||!Number.isFinite(value))continue;
          const x=Number(data.origin_m[i1])+i*spacing,y=Number(data.origin_m[i2])+j*spacing;ctx.fillStyle=`rgb(${scalarColor(value,range).map(v=>Math.round(v*255)).join(',')})`;ctx.fillRect(px(x),py(y+spacing),spacing*scale+.2,spacing*scale+.2);}
      }
      for(const cell of section.cells){
        const rgb=mesh.palette[(cell.level-1)%mesh.palette.length],x=px(cell.min[0]),y=py(cell.max[1]),w=cell.dx*scale;
        if(x>canvas.width||y>canvas.height||x+w<0||y+w<0||w<.65)continue;
        if(cut.field==='mesh'){ctx.fillStyle=`rgba(${rgb.map(v=>Math.round(v*255)).join(',')},.28)`;ctx.fillRect(x,y,w,w);}
        if(cut.mesh){ctx.strokeStyle=`rgba(${rgb.map(v=>Math.round(v*180)).join(',')},.8)`;ctx.lineWidth=.5;ctx.strokeRect(x,y,w,w);}
      }
      const geometry=getGeometrySection();
      if(geometry.loops.length&&ctx.beginPath){ctx.beginPath();for(const loop of geometry.loops){ctx.moveTo(px(loop[0][a[0]]),py(loop[0][a[1]]));for(const point of loop.slice(1))ctx.lineTo(px(point[a[0]]),py(point[a[1]]));ctx.closePath();}ctx.fillStyle='rgba(29,49,62,.42)';ctx.fill('evenodd');}
      if(geometry.segments.length&&ctx.beginPath){ctx.beginPath();for(const pair of geometry.segments){ctx.moveTo(px(pair[0][a[0]]),py(pair[0][a[1]]));ctx.lineTo(px(pair[1][a[0]]),py(pair[1][a[1]]));}ctx.lineWidth=1.6;ctx.strokeStyle='#162c3b';ctx.stroke();}
      ctx.fillStyle='#264354';ctx.font='12px sans-serif';ctx.fillText(`${axes[a[0]].toUpperCase()} [m]`,canvas.width/2,canvas.height-8);ctx.fillText(`${axes[a[1]].toUpperCase()} [m]`,3,17);
      const left=lo[a[0]]-sectionView.pan[0]/scale,right=lo[a[0]]+(canvas.width-2*pad-sectionView.pan[0])/scale;
      ctx.fillText(left.toPrecision(3),pad,canvas.height-12);ctx.fillText(right.toPrecision(3),canvas.width-pad-35,canvas.height-12);
      referenceTools?.paintCurves(ctx,referenceTools.section(cut.axis,cut.position),a,px,py,canvas);
      $('section-2d-note').textContent=statusText()+` · zoom ${sectionView.zoom.toPrecision(3)}× · wheel to zoom, drag to pan, double-click to fit.`;
    }
    function scheduleField(){clearTimeout(requestTimer);requestTimer=setTimeout(loadField,260);}
    async function loadField(){
      const cut=options();if(!cut.enabled||cut.field==='mesh'||(!state.jobId&&!state.importedResults))return;
      const key=JSON.stringify([state.importedResults?.id||state.jobId,cut.axis,cut.position,cut.field,state.job?.progress?.step]);if(cut.key===key)return;
      cut.key=key;cut.data=null;cut.loading=true;cut.error='';
      $('section-note').textContent=statusText();$('section-2d-note').textContent=statusText();
      try{
        let data;
        try{const url=state.importedResults?(state.importedResults.cut_url||`/api/aero/results/${state.importedResults.id}/cut`):`/api/aero/jobs/${state.jobId}/volume-cut`;data=await api(`${url}?axis=${cut.axis}&position=${cut.position}&field=${cut.field}&resolution=384`);}
        catch(volumeError){
          if(state.importedResults)throw volumeError;
          const recorded=(state.job?.artifacts?.cuts||[]).filter(c=>c.axis===cut.axis&&Math.abs(Number(c.position)-cut.position)<1e-7*Math.max(1,Math.abs(cut.position))).sort((a,b)=>b.step-a.step)[0];
          if(!recorded)throw volumeError;
          const url=apiUrl(recorded.json_url);if(!url)throw volumeError;data=await api(`${url}?field=${encodeURIComponent(cut.field)}`);
        }
        if(options().key!==key)return;cut.data=data;
      }catch(error){if(options().key===key)cut.error=`No field available here. ${error.message}`;}
      finally{if(options().key===key){cut.loading=false;queueRender();}}
    }
    function move(position,render=true){
      const cut=options(),frame=inspectionFrame(),a=axes.indexOf(cut.axis);cut.position=Math.max(frame.domain.min[a],Math.min(frame.domain.max[a],Number(position)));
      cut.key=null;cut.data=null;sectionKey=null;sync();if(render){queueRender();scheduleField();}
    }
    function draw(){
      sync();const cut=options();if(!cut.enabled)return;
      const frame=currentFrame(),points=planePoints(frame,cut.axis,cut.position),section=getSection(),n=axes.indexOf(cut.axis),a=section.axes;
      const backing=addQuad('section plane',points,new BABYLON.Color3(1,.64,.17),.075),border=addLines('section border',points.map((p,i)=>[p,points[(i+1)%4]]),new BABYLON.Color3(1,.64,.17));
      const details=[];if(cut.data)details.push(drawFieldPlane(cut.data));
      if(cut.mesh){
        const byLevel=new Map();for(const cell of section.cells){const rect=[[cell.min[0],cell.min[1]],[cell.max[0],cell.min[1]],[cell.max[0],cell.max[1]],[cell.min[0],cell.max[1]]].map(pair=>{const p=[0,0,0];p[n]=cut.position;p[a[0]]=pair[0];p[a[1]]=pair[1];return p;});
          const list=byLevel.get(cell.level)||[];for(let k=0;k<4;k++)list.push([rect[k],rect[(k+1)%4]]);byLevel.set(cell.level,list);}
        for(const [level,pairs] of byLevel)details.push(addLines(`section level ${level}`,pairs,new BABYLON.Color3(...mesh.palette[(level-1)%mesh.palette.length])));
      }
      const geometry=getGeometrySection(),extent=Math.hypot(...frame.domain.max.map((v,i)=>v-frame.domain.min[i]));
      if(geometry.segments.length){const epsilon=Math.max(1e-9,extent*1e-7),pairs=geometry.segments.map(pair=>pair.map(p=>{const point=p.slice();point[n]+=epsilon;return point;}));details.push(addLines('geometry section',pairs,new BABYLON.Color3(1,.94,.75)));}
      backing.metadata={...(backing.metadata||{}),sectionTool:true};
      border.isPickable=true;border.metadata={...(border.metadata||{}),sectionTool:true};border.intersectionThreshold=Math.max(extent*.002,1e-5);
      const handles=[];
      if($('section-move-handle')?.checked){
        const radius=Math.min(extent*.012,Math.max(.0001,(state.camera?.radius||extent)*.015));
        const handle=keep(BABYLON.MeshBuilder.CreateSphere('section edge handle',{diameter:radius,segments:16},state.scene));
        handle.position=v3(...points[2]);handle.isPickable=true;handle.metadata={sectionTool:true};
        const material=keep(new BABYLON.StandardMaterial('section handle material',state.scene));material.diffuseColor=new BABYLON.Color3(1,.65,.18);material.emissiveColor=material.diffuseColor.scale(.5);handle.material=material;handles.push(handle);
      }
      const normal=[0,0,0];normal[n]=1;
      for(const target of [border,...handles]){
        const behavior=new BABYLON.PointerDragBehavior({dragAxis:v3(...normal)});behavior.moveAttached=false;behavior.useObjectOrientationForDragging=false;target.addBehavior(behavior);
        let start=cut.position,travel=0;const starts=handles.map(handle=>handle.position.clone());
        behavior.onDragStartObservable.add(()=>{finishDrag(true);dragging=true;start=cut.position;travel=0;const visibility=new Map();for(const item of [...amrMeshes(),...details])if(item?.setEnabled){visibility.set(item,item.isEnabled());item.setEnabled(false);}activeDrag={behavior,visibility};clearCameraMotion();});
        behavior.onDragObservable.add(event=>{travel+=(Number(event.dragDistance)||0)*(shiftPressed ? .1 : 1);move(start+travel,false);const delta=options().position-start,offset=v3(...normal).scale(delta);backing.position=offset;border.position=offset;handles.forEach((handle,i)=>handle.position=starts[i].add(offset));applyClip();$('section-note').textContent='Moving section · Shift for 10× precision · release to refresh fields.';});
        behavior.onDragEndObservable.add(()=>finishDrag());target.onDisposeObservable?.add(()=>{if(activeDrag?.behavior===behavior)cancelDrag();});
      }
      if($('section-surface-curves')?.checked&&referenceTools){
        const segments=referenceTools.section(cut.axis,cut.position);
        if(segments.length){let lo=Infinity,hi=-Infinity;const bmin=[Infinity,Infinity],bmax=[-Infinity,-Infinity];for(const s of segments){lo=Math.min(lo,s.value);hi=Math.max(hi,s.value);for(const p of s.points)for(let j=0;j<2;j++){bmin[j]=Math.min(bmin[j],p[a[j]]);bmax[j]=Math.max(bmax[j],p[a[j]]);}}
          const span=hi-lo||1,height=Math.max(bmax[0]-bmin[0],bmax[1]-bmin[1])*.16;
          for(let direction=0;direction<2;direction++){const other=1-direction,pairs=segments.map(s=>s.points.map(p=>{const out=p.slice();out[a[other]]=bmax[other]+height*.25+height*(s.value-lo)/span;return out;}));details.push(addLines(`surface ${axes[a[direction]]} distribution`,pairs,new BABYLON.Color3(...(direction?[.35,.65,1]:[1,.4,.3]))));}
          addLabel(`Surface section field: ${lo.toPrecision(4)} … ${hi.toPrecision(4)}`,segments[0].points[0],'cut');
        }
      }
      addLabel(`${cut.axis.toUpperCase()} = ${cut.position.toPrecision(4)} m · drag edge handle or border`,points[2],'cut');
      applyClip();
      $('section-note').textContent=statusText();paint2d();scheduleField();
    }
    function applyClip(){const plane=clipPlane(),seen=new Set();const apply=material=>{if(!material||seen.has(material))return;seen.add(material);material.clipPlane=plane;for(const child of material.subMaterials||[])apply(child);};for(const item of state.scene?.meshes||state.sceneObjects)if(item?.material&&(item.name==='model'||item.name.startsWith('AMR ')||item.metadata?.modelSurface))apply(item.material);}
    function refresh(){sync();sectionKey=null;queueRender();scheduleField();}
    $('section-enabled').addEventListener('change',()=>{options().enabled=$('section-enabled').checked;refresh();});
    $('section-axis').addEventListener('change',()=>{options().axis=$('section-axis').value;options().position=null;options().data=null;options().key=null;refresh();});
    for(const id of ['section-position','section-slider'])$(id).addEventListener('input',()=>{options().enabled=true;move($(id).value);});
    $('section-field').addEventListener('change',()=>{options().field=$('section-field').value;options().key=null;options().data=null;refresh();});
    $('section-move-handle')?.addEventListener('change',queueRender);$('section-surface-curves')?.addEventListener('change',queueRender);
    $('section-mesh').addEventListener('change',()=>{options().mesh=$('section-mesh').checked;queueRender();});
    $('section-clip').addEventListener('change',()=>{options().clip=$('section-clip').checked;queueRender();});
    $('section-show').addEventListener('click',()=>{options().enabled=true;if($('interactive-section-panel'))$('interactive-section-panel').open=true;refresh();selectResultPane('cuts');});
    $('section-record').addEventListener('click',()=>{if(state.cuts.length>=6){$('section-note').textContent='Six recorded planes are already configured.';return;}state.cuts.push({axis:options().axis,position:options().position});renderCutRows();invalidateMesh();});
    $('amr-render').addEventListener('change',()=>{state.mode='mesh';$('view-mode').value='mesh';queueRender();});$('voxel-opacity').addEventListener('input',queueRender);
    const canvas=$('section-canvas');let pan=null;
    function canvasPoint(event){const rect=canvas.getBoundingClientRect();return [(event.clientX-rect.left)*canvas.width/rect.width,(event.clientY-rect.top)*canvas.height/rect.height];}
    function resetZoom(){sectionView={zoom:1,pan:[0,0]};paint2d();}
    canvas.addEventListener('wheel',event=>{if(!options().enabled)return;event.preventDefault();const p=canvasPoint(event),old=sectionView.zoom,next=Math.max(.25,Math.min(4096,old*Math.exp(-event.deltaY*.0015))),ratio=next/old,origin=[35,canvas.height-35];sectionView.pan=p.map((value,i)=>value-origin[i]-(value-origin[i]-sectionView.pan[i])*ratio);sectionView.zoom=next;paint2d();},{passive:false});
    canvas.addEventListener('contextmenu',event=>event.preventDefault());
    canvas.addEventListener('pointerdown',event=>{if(!options().enabled||![0,2].includes(event.button))return;event.preventDefault();pan={id:event.pointerId,point:canvasPoint(event),offset:sectionView.pan.slice()};try{canvas.setPointerCapture(event.pointerId);}catch{}});
    canvas.addEventListener('pointermove',event=>{if(pan?.id!==event.pointerId)return;event.preventDefault();const p=canvasPoint(event);sectionView.pan=p.map((v,i)=>pan.offset[i]+v-pan.point[i]);paint2d();});
    const finishPan=event=>{if(pan?.id!==event.pointerId)return;pan=null;try{canvas.releasePointerCapture(event.pointerId);}catch{}};canvas.addEventListener('pointerup',finishPan);canvas.addEventListener('pointercancel',finishPan);
    canvas.addEventListener('dblclick',event=>{event.preventDefault();resetZoom();});$('section-reset-zoom')?.addEventListener('click',resetZoom);
    return {draw,drawAmr,sync,levelControls,refresh,setMeshVisible,cancelDrag,isDragging:()=>dragging,
      getView:()=>({surface_section_curves:$('section-surface-curves')?.checked!==false,inspector:{...options(),data:undefined,key:undefined,loading:undefined,error:undefined},section_view:{zoom:sectionView.zoom,pan:sectionView.pan.slice()},mesh_visible:state.meshVisible!==false,levelVisibility:{...state.levelVisibility},amr_render:$('amr-render').value,voxel_opacity:Number($('voxel-opacity').value)}),
      restoreView(view){if($('section-surface-curves'))$('section-surface-curves').checked=view?.surface_section_curves!==false;cancelDrag();clearTimeout(requestTimer);setMeshVisible(view?.mesh_visible!==false);state.inspector=defaultInspector();if(view?.inspector){const c=view.inspector;Object.assign(options(),{enabled:c.enabled===true,axis:axes.includes(c.axis)?c.axis:'z',position:Number.isFinite(c.position)?c.position:null,field:['mesh','velocity','pressure','density','vorticity','u','v','w','mach','temperature'].includes(c.field)?c.field:'mesh',mesh:c.mesh!==false,clip:c.clip!==false});}const saved=view?.section_view;sectionView={zoom:Math.max(.25,Math.min(4096,Number(saved?.zoom)||1)),pan:Array.isArray(saved?.pan)&&saved.pan.length===2&&saved.pan.every(Number.isFinite)?saved.pan.slice():[0,0]};state.levelVisibility=view?.levelVisibility||{};$('amr-render').value=['blocks','wire_voxels','solid_voxels','translucent_voxels'].includes(view?.amr_render)?view.amr_render:'blocks';$('voxel-opacity').value=String(Math.max(1,Math.min(65,Number(view?.voxel_opacity)||12)));sectionKey=null;geometryKey=null;levelControls();refresh();}};
  }
  window.LudwigAeroInspector={create};
})();
