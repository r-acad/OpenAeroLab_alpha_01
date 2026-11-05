/* Original GLB appearance and persistent placement widgets. Canonical solver triangles stay unchanged. */
(() => {
  'use strict';
  const copySettings=p=>({flow_angles:[...(p?.flow_angles||[0,0])],scale:[...(p?.scale||[1,1,1])],rotation_deg:[...(p?.rotation_deg||[0,0,0])],translation:[...(p?.translation||[0,0,0])]});
  const radians=Math.PI/180;
  const materialAppearance=new WeakMap();
  // Display only: keep GLB alpha/textures and the canonical solver vertices intact.
  function applyOpacity(material,opacity,B=window.BABYLON) {
    if(!material)return;
    if(material.subMaterials){for(const part of material.subMaterials)applyOpacity(part,opacity,B);return;}
    let original=materialAppearance.get(material);
    if(!original){original={alpha:material.alpha,transparencyMode:material.transparencyMode,disableDepthWrite:material.disableDepthWrite};materialAppearance.set(material,original);}
    const factor=Math.max(0,Math.min(1,Number.isFinite(Number(opacity))?Number(opacity):1));
    material.alpha=(original.alpha??1)*factor;
    material.transparencyMode=factor<1?B.Material.MATERIAL_ALPHABLEND:original.transparencyMode;
    material.disableDepthWrite=factor<1?true:original.disableDepthWrite;
  }
  function sourceRotation(p,point) {
    const [rx,ry,rz]=p.rotation_deg.map(v=>v*radians),[cx,sx,cy,sy,cz,sz]=[Math.cos(rx),Math.sin(rx),Math.cos(ry),Math.sin(ry),Math.cos(rz),Math.sin(rz)];
    let [x,y,z]=point,n=y*cx-z*sx;z=y*sx+z*cx;y=n;
    n=x*cy+z*sy;z=-x*sy+z*cy;x=n;n=x*cz-y*sz;y=x*sz+y*cz;return window.LudwigAeroReference?window.LudwigAeroReference.multiply(window.LudwigAeroReference.rotation({alpha:p.flow_angles?.[0],beta:p.flow_angles?.[1]}),[n,y,z]):[n,y,z];
  }
  // Source -> solver -> Babylon. Free air displays Z up; the ground backend uses Y up.
  const toViewer=(p,ground)=>ground?[p[0],p[2],-p[1]]:[p[0],p[2],p[1]];
  const fromViewer=(p,ground)=>ground?[p[0],-p[2],p[1]]:[p[0],p[2],p[1]];
  const solverToViewer=(p,ground)=>ground?[...p]:[p[0],p[2],p[1]];
  function affineMatrix(B,linear,translation) {
    return B.Matrix.FromValues(...linear[0],0,...linear[1],0,...linear[2],0,...translation,1);
  }
  function rotationMatrix(B,p,ground) {
    return affineMatrix(B,[[1,0,0],[0,1,0],[0,0,1]].map(axis=>toViewer(sourceRotation(p,fromViewer(axis,ground)),ground)),[0,0,0]);
  }
  function geometryMatrix(B,p,center,origin,ground) {
    const basis=[[1,0,0],[0,1,0],[0,0,1]].map((axis,i)=>toViewer(sourceRotation(p,axis.map(v=>v*p.scale[i])),ground));
    const offset=solverToViewer(origin,ground).map((v,a)=>v-basis.reduce((sum,b,i)=>sum+b[a]*center[i],0));
    return affineMatrix(B,basis,offset);
  }
  function rotationDegrees(B,quaternion,ground,flowAngles=[0,0]) {
    const matrix=new B.Matrix();quaternion.toRotationMatrix(matrix);
    const axes=[[1,0,0],[0,1,0],[0,0,1]].map(axis=>{
      const v=toViewer(axis,ground),r=B.Vector3.TransformNormal(B.Vector3.FromArray(v),matrix);
      const point=fromViewer([r.x,r.y,r.z],ground);return window.LudwigAeroReference?window.LudwigAeroReference.multiply(window.LudwigAeroReference.transpose(window.LudwigAeroReference.rotation({alpha:flowAngles[0],beta:flowAngles[1]})),point):point;
    });
    // Rz * Ry * Rx, matching the mesher's numeric X, Y, Z controls.
    const y=Math.asin(Math.max(-1,Math.min(1,-axes[0][2]))),cy=Math.cos(y);
    const x=Math.abs(cy)>1e-7?Math.atan2(axes[1][2],axes[2][2]):Math.atan2(-axes[2][1],axes[1][1]);
    const z=Math.abs(cy)>1e-7?Math.atan2(axes[0][1],axes[0][0]):0;
    return [x,y,z].map(v=>v/radians);
  }
  function extractTriangles(meshes,max=1_000_000) {
    if(!Number.isSafeInteger(max)||max<1)throw new Error('The triangle limit must be a positive integer.');
    const parts=[];let total=0;
    for(const mesh of meshes||[]){
      if(typeof mesh.getVerticesData!=='function')continue;
      const positions=mesh.getVerticesData('position'),rawIndices=mesh.getIndices?.();
      if(!positions||positions.length<9)continue;
      const indices=rawIndices?.length>=3?rawIndices:null,count=Math.floor(indices?indices.length/3:positions.length/9);
      if(!count)continue;
      const thin=Boolean(mesh.hasThinInstances),copies=thin?Number(mesh.thinInstanceCount):1;
      if(!Number.isSafeInteger(copies)||copies<1)throw new Error('The model has invalid thin-instance geometry.');
      total+=count*copies;
      if(total>max)throw new Error(`Model exceeds the solver's ${max.toLocaleString('en-US')} triangle upload limit.`);
      parts.push({mesh,positions,indices,count,thin,copies});
    }
    if(!total)throw new Error('No triangle surface was found in this file.');
    const triangles=new Float32Array(total*9);let cursor=0;
    for(const part of parts){
      part.mesh.computeWorldMatrix(true);const world=part.mesh.getWorldMatrix();
      // Babylon's thin-instance matrices are mesh-local despite the method name.
      // A base mesh with thin instances represents those instances only, not an extra copy.
      const matrices=part.thin?part.mesh.thinInstanceGetWorldMatrices():[null];
      if(matrices.length!==part.copies)throw new Error('The model thin-instance matrix count changed during import.');
      for(const local of matrices){
        const matrix=local?local.multiply(world):world,m=matrix.m;
        for(let i=0;i<part.count*3;i++){
          const index=part.indices?part.indices[i]:i,k=index*3;
          if(!Number.isInteger(index)||index<0||k+2>=part.positions.length)throw new Error('The model contains an invalid triangle vertex index.');
          const x=part.positions[k],y=part.positions[k+1],z=part.positions[k+2],w=x*m[3]+y*m[7]+z*m[11]+m[15];
          const out=[(x*m[0]+y*m[4]+z*m[8]+m[12])/w,(x*m[1]+y*m[5]+z*m[9]+m[13])/w,(x*m[2]+y*m[6]+z*m[10]+m[14])/w];
          if(!out.every(Number.isFinite))throw new Error('The model contains a non-finite triangle vertex.');
          triangles.set(out,cursor);cursor+=3;
        }
      }
    }
    return triangles;
  }
  function create(adapter) {
    const B=adapter.BABYLON||window.BABYLON,scene=adapter.scene;
    if(!B||!scene)throw new Error('Model tools require a Babylon scene.');
    const proxy=new B.Mesh('aero-model-widget-target',scene);proxy.isVisible=false;proxy.isPickable=false;proxy.rotationQuaternion=B.Quaternion.Identity();
    const manager=new B.GizmoManager(scene);manager.usePointerToAttachGizmos=false;manager.clearGizmoOnEmptyPointerEvent=false;
    const visualRoot=new B.TransformNode('aero-original-glb-placement',scene);visualRoot.rotationQuaternion=B.Quaternion.Identity();visualRoot.setEnabled(false);
    let file=null,loadPromise=Promise.resolve(false),container=null,loadToken=0,disposed=false,loading=false,error='',textured=false,frame=null,surface=null,options={},drag=null,projections=[];
    const availability=()=>({supported:Boolean(file&&/\.glb$/i.test(file.name||'')),ready:Boolean(container),loading,error});
    const notify=()=>adapter.onAvailability?.(availability());
    function disposeContainer(value) {if(!value)return;value.dispose();}
    function clear() {return refreshSource(null);}
    async function refreshSource(nextFile,{retry=false}={}) {
      if(disposed)return false;
      if(nextFile===file&&!(retry&&error))return loadPromise;
      file=nextFile||null;const token=++loadToken;loading=false;error='';textured=false;loadPromise=Promise.resolve(false);visualRoot.setEnabled(false);
      disposeContainer(container);container=null;if(surface&&!surface.isDisposed?.())surface.setEnabled((options.opacity??1)>0);notify();
      if(!availability().supported){adapter.onRender?.();return false;}
      loading=true;notify();
      loadPromise=(async()=>{
        let loaded=null;
        try {
          loaded=await B.SceneLoader.LoadAssetContainerAsync('file:',file,scene);
          if(disposed||token!==loadToken){disposeContainer(loaded);return false;}
          // Keep GLB hierarchy, UVs, textures, materials and handedness conversion together.
          loaded.addAllToScene();
          for(const group of loaded.animationGroups||[])group.stop();
          const nodes=[...(loaded.meshes||[]),...(loaded.transformNodes||[])],members=new Set(nodes);
          for(const node of nodes)if(!node.parent||!members.has(node.parent))node.parent=visualRoot;
          for(const mesh of loaded.meshes||[]){
            if(typeof mesh.getTotalVertices==='function'&&mesh.getTotalVertices()>0){
              mesh.metadata={...mesh.metadata,modelSurface:true};mesh.isPickable=true;
              if(!mesh.material){const fallback=scene.defaultMaterial.clone('original-glb-default');mesh.material=fallback;loaded.materials.push(fallback);}
            }
          }
          for(const material of loaded.materials||[]){material.backFaceCulling=false;if('twoSidedLighting' in material)material.twoSidedLighting=true;}
          container=loaded;loading=false;notify();adapter.onRender?.();return true;
        }catch(cause){
          if(loaded&&loaded!==container)disposeContainer(loaded);
          if(disposed||token!==loadToken)return false;
          loading=false;error=`Original GLB appearance could not be loaded: ${cause.message||cause}`;notify();adapter.onStatus?.(error,true);adapter.onRender?.();return false;
        }
      })();
      return loadPromise;
    }
    function applyMatrix(node,matrix) {matrix.decompose(node.scaling,node.rotationQuaternion,node.position);node.computeWorldMatrix(true);}
    function applyAppearance(p) {
      const bounds=adapter.getSourceBounds?.();
      if(container&&bounds?.center&&frame?.bodyOrigin)applyMatrix(visualRoot,geometryMatrix(B,p,bounds.center,frame.bodyOrigin,Boolean(frame.ground)));
    }
    function applySymmetryClip() {
      // A setup preview shows the complete body across the symmetry plane.
      // Built solver frames contain only Y >= 0; retained GLB textures must match.
      const clip=frame?.solverFrame&&frame.boundary==='symmetry'?new B.Plane(0,0,-1,0):null,seen=new Set();
      const apply=material=>{if(!material||seen.has(material))return;seen.add(material);material.clipPlane2=clip;for(const child of material.subMaterials||[])apply(child);};
      apply(surface?.material);for(const material of container?.materials||[])apply(material);
      for(const mesh of container?.meshes||[])apply(mesh.material);
    }
    function clearProjections() {
      for(const item of projections){item.mesh.dispose();item.material.dispose();}projections=[];
    }
    function updateProjections() {
      if(!surface||surface.isDisposed?.())return;
      surface.computeWorldMatrix(true);
      for(const item of projections){item.mesh.setPreTransformMatrix(surface.getWorldMatrix().multiply(item.flatten));item.mesh.computeWorldMatrix(true);}
    }
    function drawProjections() {
      const selection=options.projections===true?'all':options.projections||'none',domain=frame?.domain;
      if(selection==='none'||!surface?.geometry||!domain?.min||!domain?.max)return;
      const ground=Boolean(frame.ground),vertical=ground?1:2,lateral=ground?2:1;
      const size=domain.max.map((v,i)=>Number(v)-Number(domain.min[i]));
      if(size.some(v=>!Number.isFinite(v)||v<=0))return;
      const epsilon=Math.max(1e-7,Math.min(...size)*1e-5);
      const specs=[{name:'ground',axis:vertical,max:false,color:[.24,.72,.39,.23],bit:1},
        {name:'inlet',axis:0,max:false,color:[.97,.80,.25,.24],bit:2},
        {name:'lateral',axis:lateral,max:true,color:[.28,.65,.92,.23],bit:4}];
      for(const spec of specs){
        if(selection!=='all'&&selection!==spec.name)continue;
        const axis=ground?spec.axis:spec.axis===1?2:spec.axis===2?1:0;
        const coordinate=spec.max?Number(domain.max[spec.axis])-epsilon:Number(domain.min[spec.axis])+epsilon;
        const flatten=B.Matrix.Identity();flatten.m[axis*5]=0;const shift=B.Vector3.Zero();shift[['x','y','z'][axis]]=coordinate;flatten.setTranslation(shift);
        const mesh=new B.Mesh(`Model projection ${spec.name}`,scene);surface.geometry.applyToMesh(mesh);
        mesh.isPickable=false;mesh.metadata={placementProjection:spec.name};mesh.alwaysSelectAsActiveMesh=true;
        const material=new B.ShaderMaterial(`model-projection-${spec.name}`,scene,{
          vertexSource:'precision highp float; attribute vec3 position; uniform mat4 worldViewProjection; uniform mat4 world; uniform mat4 modelWorld; varying vec3 modelPosition; varying vec3 projectionPosition; void main(){ modelPosition=(modelWorld*vec4(position,1.0)).xyz; projectionPosition=(world*vec4(position,1.0)).xyz; gl_Position=worldViewProjection*vec4(position,1.0); }',
          fragmentSource:'precision highp float; uniform vec4 projectionColor; uniform vec4 modelClipPlane; uniform float modelClipEnabled; uniform vec4 modelClipPlane2; uniform float modelClipEnabled2; uniform vec3 projectionMin; uniform vec3 projectionMax; varying vec3 modelPosition; varying vec3 projectionPosition; void main(){ if(any(lessThan(projectionPosition,projectionMin)) || any(greaterThan(projectionPosition,projectionMax))) discard; if(modelClipEnabled>0.5 && dot(vec4(modelPosition,1.0),modelClipPlane)>0.0) discard; if(modelClipEnabled2>0.5 && dot(vec4(modelPosition,1.0),modelClipPlane2)>0.0) discard; gl_FragColor=projectionColor; }'
        },{attributes:['position'],uniforms:['worldViewProjection','world','modelWorld','projectionColor','modelClipPlane','modelClipEnabled','modelClipPlane2','modelClipEnabled2','projectionMin','projectionMax'],needAlphaBlending:true});
        material.backFaceCulling=false;material.disableDepthWrite=true;material.setColor4('projectionColor',new B.Color4(...spec.color));
        material.setVector3('projectionMin',B.Vector3.FromArray(solverToViewer(domain.min,ground)));material.setVector3('projectionMax',B.Vector3.FromArray(solverToViewer(domain.max,ground)));
        // A planar projection can contain thousands of overlapping triangles.
        // Shade each pixel once per wall, retaining a uniform translucent silhouette.
        const stencil=material.stencil;stencil.enabled=true;stencil.func=B.Constants.NOTEQUAL;stencil.backFunc=B.Constants.NOTEQUAL;
        stencil.funcRef=spec.bit;stencil.funcMask=spec.bit;stencil.mask=spec.bit;
        stencil.opStencilFail=B.Constants.KEEP;stencil.opDepthFail=B.Constants.KEEP;stencil.opStencilDepthPass=B.Constants.REPLACE;
        stencil.backOpStencilFail=B.Constants.KEEP;stencil.backOpDepthFail=B.Constants.KEEP;stencil.backOpStencilDepthPass=B.Constants.REPLACE;
        const original=surface;
        mesh.onBeforeRenderObservable.add(()=>{
          if(original.isDisposed?.())return;
          material.setMatrix('modelWorld',original.getWorldMatrix());const clip=original.material?.clipPlane||scene.clipPlane;
          material.setFloat('modelClipEnabled',clip?1:0);material.setVector4('modelClipPlane',clip?new B.Vector4(clip.normal.x,clip.normal.y,clip.normal.z,clip.d):B.Vector4.Zero());
          const clip2=original.material?.clipPlane2||scene.clipPlane2;material.setFloat('modelClipEnabled2',clip2?1:0);material.setVector4('modelClipPlane2',clip2?new B.Vector4(clip2.normal.x,clip2.normal.y,clip2.normal.z,clip2.d):B.Vector4.Zero());
        });
        mesh.material=material;projections.push({mesh,material,flatten});
      }
      updateProjections();
    }
    function beginDrag() {
      if(!frame||drag)return;
      const params=copySettings({...frame.config?.transform,...(frame.config?.flow?{flow_angles:[frame.config.flow.alpha||0,frame.config.flow.beta||0]}:adapter.getTransformSettings())});proxy.computeWorldMatrix(true);
      drag={params,origin:proxy.position.clone(),inverse:proxy.getWorldMatrix().clone().invert(),ground:Boolean(frame.ground)};
    }
    function updateDrag() {
      if(!drag)return;
      // Only automatic symmetry placement locks the lateral move handle.
      const aligned=frame.boundary==='symmetry'&&frame.config?.domain?.symmetry_placement!=='none';
      if(aligned)proxy.position.z=drag.origin.z;
      const p=copySettings(drag.params),delta=proxy.position.subtract(drag.origin);
      const translation=fromViewer([delta.x,delta.y,delta.z],drag.ground);
      p.translation=p.translation.map((v,i)=>v+translation[i]);
      if(aligned)p.translation[1]=0;
      p.rotation_deg=rotationDegrees(B,proxy.rotationQuaternion||B.Quaternion.FromEulerVector(proxy.rotation),drag.ground,drag.params.flow_angles);
      adapter.setTransformSettings(p);
      proxy.computeWorldMatrix(true);
      // The neutral mesh is transient; deform it only until the commit rebuilds the preview.
      if(surface&&!surface.isDisposed?.())surface.setPreTransformMatrix(drag.inverse.multiply(proxy.getWorldMatrix()));
      if(container){
        // frame.bodyOrigin already includes the initial translation; only its delta moves here.
        const solverDelta=drag.ground?[delta.x,delta.y,delta.z]:[delta.x,delta.z,delta.y];
        const origin=frame.bodyOrigin.map((v,i)=>v+solverDelta[i]);
        applyMatrix(visualRoot,geometryMatrix(B,p,adapter.getSourceBounds().center,origin,drag.ground));
      }
      updateProjections();
    }
    function endDrag() {if(!drag)return;updateDrag();drag=null;adapter.onTransformCommit?.();adapter.onRender?.();}
    function wireGizmo(gizmo) {
      gizmo.updateGizmoRotationToMatchAttachedMesh=false;
      gizmo.onDragStartObservable.add(beginDrag);gizmo.onDragObservable.add(updateDrag);gizmo.onDragEndObservable.add(endDrag);
    }
    // Create once. Utility-layer gizmos survive the app's per-frame model redraws.
    manager.positionGizmoEnabled=true;wireGizmo(manager.gizmos.positionGizmo);manager.positionGizmoEnabled=false;
    manager.rotationGizmoEnabled=true;wireGizmo(manager.gizmos.rotationGizmo);manager.rotationGizmoEnabled=false;
    function sync(nextFrame,nextOptions={}) {
      if(disposed)return false;
      if(adapter.getSourceFile)refreshSource(adapter.getSourceFile()||null);
      if(drag)return textured;
      clearProjections();
      frame=nextFrame;surface=nextOptions.surface||null;options=nextOptions;
      const p=copySettings({...frame?.config?.transform,...(frame?.config?.flow?{flow_angles:[frame.config.flow.alpha||0,frame.config.flow.beta||0]}:adapter.getTransformSettings())}),bounds=adapter.getSourceBounds?.();
      const placed=Boolean(frame?.bodyOrigin?.length===3&&bounds?.center&&!frame?.imported);
      textured=Boolean(placed&&container&&options.appearance==='glb'&&!options.fieldMode);
      const opacity=Math.max(0,Math.min(1,Number.isFinite(Number(options.opacity))?Number(options.opacity):1));
      visualRoot.setEnabled(textured&&opacity>0);if(surface)surface.setEnabled(!textured&&opacity>0);
      applyOpacity(surface?.material,opacity,B);
      for(const item of container?.meshes||[])applyOpacity(item.material,opacity,B);
      applySymmetryClip();
      if(placed){
        const pose=rotationMatrix(B,p,Boolean(frame.ground));pose.setTranslation(B.Vector3.FromArray(solverToViewer(frame.bodyOrigin,Boolean(frame.ground))));
        applyMatrix(proxy,pose);applyAppearance(p);
      }
      const widget=placed&&options.enabled!==false?options.widget:'none';
      manager.positionGizmoEnabled=widget==='move';manager.rotationGizmoEnabled=widget==='rotate';
      manager.gizmos.positionGizmo.zGizmo.isEnabled=frame?.boundary!=='symmetry'||frame.config?.domain?.symmetry_placement==='none';
      manager.attachToMesh(widget==='move'||widget==='rotate'?proxy:null);
      drawProjections();
      return textured;
    }
    function isModelMesh(mesh) {for(let node=mesh;node;node=node.parent)if(node.metadata?.modelSurface)return true;return false;}
    function ownsPointer(x,y) {
      if(drag)return true;
      if(!manager.positionGizmoEnabled&&!manager.rotationGizmoEnabled)return false;
      const utility=manager.utilityLayer?.utilityLayerScene;
      return Boolean(utility?.activeCamera&&utility.pick(x,y)?.hit);
    }
    function dispose() {
      if(disposed)return;disposed=true;++loadToken;disposeContainer(container);container=null;
      clearProjections();manager.dispose();proxy.dispose();visualRoot.dispose();
    }
    return {sync,refreshSource,clear,availability,isModelMesh,isInteracting:()=>Boolean(drag),isTexturedVisible:()=>textured&&visualRoot.isEnabled(),ownsPointer,dispose};
  }
  window.LudwigAeroModelTools={create,extractTriangles,applyOpacity};
})();
