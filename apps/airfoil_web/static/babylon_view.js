"use strict";

/* Optional interactive view for the structured CFD mesh and sampled fields.
   The 2D canvas remains the fallback if WebGL is unavailable. */
(() => {
  const palette = [[23,62,114],[52,124,187],[91,192,201],[237,212,107],[233,139,77],[155,56,77]];
  let active = null;
  const finite = v => typeof v === "number" && Number.isFinite(v);
  const arrayLike = v => Array.isArray(v) || ArrayBuffer.isView(v);
  const clamp = (v,lo,hi) => Math.max(lo,Math.min(hi,v));

  function colorAt(t) {
    const u = clamp(t,0,1) * (palette.length - 1), i = Math.min(palette.length-2,Math.floor(u));
    const f = u-i;
    return palette[i].map((v,k) => (v*(1-f)+palette[i+1][k]*f)/255);
  }
  function cssColor(name, fallback) {
    const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    const m = /^#([0-9a-f]{6})$/i.exec(raw);
    if (!m) return fallback;
    return [0,2,4].map(i => parseInt(m[1].slice(i,i+2),16)/255);
  }
  function sceneTheme(scene) {
    const [r,g,b] = cssColor("--soft",[0.96,0.98,0.99]);
    scene.clearColor = new BABYLON.Color4(r,g,b,1);
  }
  function validGrid(data) {
    const ni = Number(data?.ni), nj = Number(data?.nj);
    return Number.isInteger(ni) && Number.isInteger(nj) && ni>1 && nj>1 &&
      arrayLike(data.x) && arrayLike(data.y) &&
      data.x.length===ni*nj && data.y.length===ni*nj;
  }
  function gridBlocks(mesh) {
    const blocks=Array.isArray(mesh?.blocks)?mesh.blocks:mesh?[mesh]:[];
    return blocks.length&&blocks.every(validGrid)?blocks:null;
  }
  function physicalHome(grids){
    let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
    for(const grid of grids)for(let k=0;k<grid.x.length;k++){
      const x=Number(grid.x[k]),y=Number(grid.y[k]);if(!finite(x)||!finite(y))continue;
      xmin=Math.min(xmin,x);xmax=Math.max(xmax,x);ymin=Math.min(ymin,y);ymax=Math.max(ymax,y);
    }
    return [xmin,xmax,ymin,ymax].every(finite)?{x:(xmin+xmax)/2,y:(ymin+ymax)/2,radius:Math.max(xmax-xmin,ymax-ymin,1e-6)*1.05}:{x:.5,y:0,radius:2.45};
  }
  function sampleData(field, quantity) {
    const samples = field?.samples;
    const shape = field?.sample_shape;
    const ni = Number(field?.ni || (Array.isArray(shape) ? shape[0] : shape?.ni));
    const nj = Number(field?.nj || (Array.isArray(shape) ? shape[1] : shape?.nj));
    let values = samples?.[quantity];
    if (quantity === "speed" && arrayLike(samples?.u) && arrayLike(samples?.v) && samples.u.length===samples.v.length)
      values = Array.from(samples.u,(v,i) => Math.hypot(Number(v),Number(samples.v[i])));
    const data = {id:field?.id||"",ni,nj,x:samples?.x,y:samples?.y,values};
    return validGrid(data) && arrayLike(values) && values.length===ni*nj &&
      Array.from(values).every(finite) && Array.from(data.x).every(finite) && Array.from(data.y).every(finite) ? data : null;
  }
  function sampleBlocks(point, quantity) {
    const fields=point?.fields,blocks=fields?.kind==="structured_blocks"?fields.blocks:[fields];
    if(!Array.isArray(blocks)||!blocks.length)return null;
    if(fields?.kind==="structured_blocks"&&(blocks.length<2||blocks.length>32||
        new Set(blocks.map(block=>block?.id)).size!==blocks.length||blocks.some(block=>
          typeof block?.id!=="string"||!/^[a-zA-Z0-9_-]+$/.test(block.id)||
          (fields.iteration!=null&&block.iteration!==fields.iteration)||
          (fields.mesh_sha256!=null&&block.mesh_sha256!==fields.mesh_sha256))))return null;
    const data=blocks.map(block=>sampleData(block,quantity));
    return data.every(Boolean)?data:null;
  }
  function scalarRange(blocks) {
    const good=blocks.flatMap(block=>Array.from(block.values,Number)).filter(finite).sort((a,b)=>a-b);
    if(good.length<4)return null;
    const percentile=q=>good[clamp(Math.round(q*(good.length-1)),0,good.length-1)];
    return {low:percentile(.02),high:percentile(.98)};
  }
  function vector(x,y,height=0) { return new BABYLON.Vector3(Number(x),height,Number(y)); }

  function meshLineSteps(grid, radius) {
    const detail=clamp(2.45/radius,0.25,8);
    return {
      i: Math.max(1,Math.ceil(grid.ni/(360*detail))),
      j: Math.max(1,Math.ceil(grid.nj/(120*detail)))
    };
  }
  function addMeshLines(scene, grid, steps) {
    const {ni,nj,x,y} = grid;
    const lines = [];
    const iStep = steps.i, jStep = steps.j;
    const ii = [...new Set([...Array(Math.ceil((ni-1)/iStep)+1)].map((_,k)=>Math.min(ni-1,k*iStep)))];
    const jj = [...new Set([...Array(Math.ceil((nj-1)/jStep)+1)].map((_,k)=>Math.min(nj-1,k*jStep)))];
    for (const j of jj) {
      const row = [];
      for (let i=0;i<ni;i++) {
        const k=j*ni+i, xv=Number(x[k]), yv=Number(y[k]);
        if (finite(xv)&&finite(yv)) row.push(vector(xv,yv));
      }
      if (row.length>1) lines.push(row);
    }
    for (const i of ii) {
      const column = [];
      for (let j=0;j<nj;j++) {
        const k=j*ni+i, xv=Number(x[k]), yv=Number(y[k]);
        if (finite(xv)&&finite(yv)) column.push(vector(xv,yv));
      }
      if (column.length>1) lines.push(column);
    }
    if (!lines.length) return null;
    const gridLines = BABYLON.MeshBuilder.CreateLineSystem(`structured mesh ${grid.id||""}`,{lines},scene);
    const [r,g,b] = cssColor("--mesh-line",[0.3,0.55,0.65]);
    gridLines.color = new BABYLON.Color3(r,g,b);
    gridLines.isPickable = false;
    return gridLines;
  }
  function addFoil(scene, geometry) {
    const xs=geometry?.x, ys=geometry?.y;
    if (!arrayLike(xs)||!arrayLike(ys)||xs.length<2||xs.length!==ys.length) return null;
    const points=[];
    for(let i=0;i<xs.length;i++) {
      const x=Number(xs[i]), y=Number(ys[i]);
      if (finite(x)&&finite(y)) points.push(vector(x,y,0.003));
    }
    if(points.length<2) return null;
    const first=points[0],last=points[points.length-1];
    if(first.subtract(last).length()>1e-8) points.push(first.clone());
    const foil=BABYLON.MeshBuilder.CreateLines("airfoil",{points},scene);
    foil.color=new BABYLON.Color3(0.95,0.45,0.22);
    foil.isPickable=false;
    return foil;
  }
  function addScalarSurface(scene,data,range) {
    const {ni,nj,x,y,values}=data;
    const numeric=Array.from(values,Number),{low,high}=range,span=high-low;
    const positions=[],colors=[],normals=[],indices=[],valid=[];
    for(let k=0;k<ni*nj;k++) {
      const xv=Number(x[k]),yv=Number(y[k]),v=numeric[k];
      const okay=finite(xv)&&finite(yv)&&finite(v);
      valid.push(okay);
      const t=okay && span>0 ? clamp((v-low)/span,0,1) : 0.5;
      const [r,g,b]=colorAt(t);
      // Height is an intentionally normalized display quantity, not geometry.
      positions.push(okay?xv:0,(t-0.5)*0.24,okay?yv:0);
      colors.push(r,g,b,1);
      normals.push(0,1,0);
    }
    const signed=(a,b,c)=>{
      const ax=Number(x[a]),ay=Number(y[a]),bx=Number(x[b]),by=Number(y[b]),cx=Number(x[c]),cy=Number(y[c]);
      return (bx-ax)*(cy-ay)-(by-ay)*(cx-ax);
    };
    for(let j=0;j<nj-1;j++) for(let i=0;i<ni-1;i++) {
      const a=j*ni+i,b=a+1,d=a+ni,c=d+1;
      if(!valid[a]||!valid[b]||!valid[c]||!valid[d]) continue;
      const area1=signed(a,b,c),area2=signed(a,c,d);
      // Do not draw folded or degenerate cells across a wake cut.
      if(Math.abs(area1)<1e-13||Math.abs(area2)<1e-13||area1*area2<=0) continue;
      indices.push(a,b,c,a,c,d);
    }
    if(!indices.length) return null;
    const surface=new BABYLON.Mesh(`scalar surface ${data.id}`,scene);
    const vertexData=new BABYLON.VertexData();
    vertexData.positions=positions;vertexData.indices=indices;
    vertexData.colors=colors;vertexData.normals=normals;
    vertexData.applyToMesh(surface);
    surface.useVertexColors=true;
    const material=new BABYLON.StandardMaterial("scalar colors",scene);
    material.backFaceCulling=false;
    material.disableLighting=true;
    material.diffuseColor=BABYLON.Color3.White();
    material.emissiveColor=BABYLON.Color3.White();
    surface.material=material;
    return {surface,low,high,drawnCells:indices.length/6};
  }
  function dispose() {
    if(!active) return;
    const old=active;active=null;
    if(old.frame!==null) cancelAnimationFrame(old.frame);
    old.observer?.disconnect();
    if(old.cameraObserver) old.camera.onViewMatrixChangedObservable.remove(old.cameraObserver);
    for(const [name,fn] of old.listeners) old.canvas.removeEventListener(name,fn);
    old.camera.detachControl(old.canvas);
    old.scene.dispose();
    old.engine.dispose();
    old.node.replaceChildren();
  }
  function requestPaint() {
    if(!active||active.frame!==null) return;
    const item=active;
    item.frame=requestAnimationFrame(()=>{
      item.frame=null;
      if(active!==item) return;
      item.engine.resize();
      // Babylon's fixed near plane otherwise clips the surface once the
      // unrestricted camera radius becomes smaller than its old zoom floor.
      item.camera.minZ=Math.max(Number.EPSILON,item.camera.radius*1e-5);
      item.camera.maxZ=Math.min(Number.MAX_VALUE,Math.max(100,item.camera.radius*1e4));
      if(item.mode==="mesh") {
        const steps=item.meshGrids.map(grid=>meshLineSteps(grid,item.camera.radius));
        if(steps.some((step,i)=>step.i!==item.meshSteps[i].i||step.j!==item.meshSteps[i].j)) {
          item.gridLines.forEach(line=>line?.dispose());
          item.gridLines=item.meshGrids.map((grid,i)=>addMeshLines(item.scene,grid,steps[i]));
          item.meshSteps=steps;
        }
      }
      item.scene.render();
    });
  }
  function show({node,mesh=null,point=null,quantity="mesh",geometry=null}={}) {
    dispose();
    if(!node||typeof BABYLON==="undefined") return false;
    const meshMode=quantity==="mesh"||quantity==="mesh:grid";
    const data=meshMode?null:sampleBlocks(point,quantity);
    const grids=meshMode?gridBlocks(mesh):null;
    if(meshMode?!grids:!data) return false;
    const range=meshMode?null:scalarRange(data);
    if(!meshMode&&!range)return false;
    const canvas=document.createElement("canvas");
    canvas.setAttribute("aria-label",meshMode?"Interactive CFD mesh":"Interactive three dimensional CFD scalar field");
    const caption=document.createElement("div");
    caption.className="babylon-caption";
    const fullGrid=meshMode&&grids.every(grid=>Number(grid.ni)===Number(grid.full_ni||grid.ni)&&Number(grid.nj)===Number(grid.full_nj||grid.nj));
    const si=mesh?.units==="m"||point?.fields?.coordinate_units==="m"||point?.fields?.units?.x==="m";
    caption.textContent=meshMode?`${fullGrid?"Full mesh":"Mesh preview"}${grids.length>1?` · ${grids.map(grid=>grid.id||"block").join(" + ")} connected blocks`:""}${si?" · coordinates in metres":""} · adaptive lines · drag to orbit · right drag to pan · scroll to zoom`:
      `${quantity}${data.length>1?` · ${data.map(block=>block.id||"block").join(" + ")} blocks`:""} · common color range ${range.low.toPrecision(4)}–${range.high.toPrecision(4)} · normalized height · drag to orbit · right drag to pan · scroll to zoom`;
    node.replaceChildren(canvas,caption);
    let engine,scene,camera;
    try {
      engine=new BABYLON.Engine(canvas,true,{preserveDrawingBuffer:false,stencil:false});
      scene=new BABYLON.Scene(engine);
      sceneTheme(scene);
      const home=si?physicalHome(meshMode?grids:data):{x:.5,y:0,radius:2.45};
      camera=new BABYLON.ArcRotateCamera("orbit",-Math.PI/2,0.78,home.radius,new BABYLON.Vector3(home.x,0,home.y),scene);
      camera.lowerBetaLimit=0.08;
      camera.upperBetaLimit=Math.PI-0.08;
      camera.lowerRadiusLimit=Number.EPSILON;
      camera.upperRadiusLimit=Number.MAX_VALUE;
      camera.panningSensibility=450;
      camera.wheelDeltaPercentage=0.075;
      camera.inertia=0;
      camera.panningInertia=0;
      camera.attachControl(canvas,true);
      const scalars=meshMode?[]:data.map(block=>addScalarSurface(scene,block,range));
      if(!meshMode&&scalars.some(scalar=>!scalar)) throw new Error("No drawable CFD scalar cells in one or more blocks");
      const meshSteps=meshMode?grids.map(grid=>meshLineSteps(grid,camera.radius)):null;
      const gridLines=meshMode?grids.map((grid,i)=>addMeshLines(scene,grid,meshSteps[i])):[];
      const wall=arrayLike(mesh?.wall_x)&&mesh.wall_x.length===mesh.wall_y?.length?{...geometry,x:mesh.wall_x,y:mesh.wall_y}:geometry;
      const foil=addFoil(scene,wall);
      const listeners=[];
      for(const name of ["pointermove","pointerdown","pointerup","wheel","touchmove"]) {
        const fn=()=>requestPaint();
        canvas.addEventListener(name,fn,{passive:true});
        listeners.push([name,fn]);
      }
      const fit=()=>center();canvas.addEventListener("dblclick",fit);listeners.push(["dblclick",fit]);
      const observer=typeof ResizeObserver==="undefined"?null:new ResizeObserver(requestPaint);
      observer?.observe(node);
      active={node,canvas,engine,scene,camera,gridLines,foil,scalars,range,observer,listeners,home,frame:null,
        mode:meshMode?"mesh":"scalar",meshGrids:grids,meshSteps};
      active.cameraObserver=camera.onViewMatrixChangedObservable.add(requestPaint);
      requestPaint();
      // Babylon may compile shaders after the first requested frame. A view
      // rendered only on demand needs one more paint when the scene is ready.
      scene.whenReadyAsync().then(() => {
        if(active?.scene===scene) requestPaint();
      }).catch(err => { if(active?.scene===scene) console.warn("Babylon scene failed to become ready:",err); });
      return true;
    } catch(err) {
      console.warn("Babylon view unavailable:",err);
      try{camera?.detachControl(canvas);scene?.dispose();engine?.dispose();}catch{}
      node.replaceChildren();
      return false;
    }
  }
  function zoom(factor) {
    if(!active||!finite(factor)||factor<=0) return;
    active.camera.radius=clamp(active.camera.radius/factor,Number.EPSILON,Number.MAX_VALUE);
    requestPaint();
  }
  function center() {
    if(!active) return;
    active.camera.setTarget(new BABYLON.Vector3(active.home.x,0,active.home.y));
    active.camera.alpha=-Math.PI/2;
    active.camera.beta=0.78;
    active.camera.radius=active.home.radius;
    requestPaint();
  }
  function redrawTheme() {
    if(!active) return;
    sceneTheme(active.scene);
    if(active.gridLines?.length) {
      const [r,g,b]=cssColor("--mesh-line",[0.3,0.55,0.65]);
      active.gridLines.forEach(line=>{if(line)line.color=new BABYLON.Color3(r,g,b);});
    }
    requestPaint();
  }
  function scalarColorRange(){return active?.range||null;}
  window.AirfoilBabylon={show,hide:dispose,zoom,center,redrawTheme,scalarColorRange};
})();
