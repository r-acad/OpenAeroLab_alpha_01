"use strict";

/* AMR inspection uses the solver's exported block coordinates. Rendering may
   sample large volumes; a section always uses the actual cell size. */
(() => {
  const axes=['x','y','z'];
  const palette=[[.25,.64,.91],[.31,.77,.50],[.97,.72,.25],[.94,.36,.34],[.69,.45,.91],[.20,.80,.81],[.94,.48,.72]];
  function levelId(level,index){return Number(level.level_id)||index+1;}
  function levelInfo(amr){return (amr?.levels||[]).map((level,index)=>({id:levelId(level,index),dx:Number(level.dx),
    size:Number(level.block_phys_size),blocks:level.blocks||[],count:Math.floor((level.blocks||[]).length/3),level,index}));}
  function indexLevels(amr){
    return levelInfo(amr).map(info=>({...info,keys:new Set(Array.from({length:info.count},(_,i)=>info.blocks.slice(i*3,i*3+3).join(',')))}));
  }
  function covered(point,info,index){
    for(const fine of index)if(fine.dx<info.dx&&fine.size>0){
      const key=point.map(v=>Math.floor((v+fine.dx*1e-7)/fine.size)+1).join(',');if(fine.keys.has(key))return true;
    }
    return false;
  }
  function section(amr,axis,position,visible=null,limit=70000){
    const normal=axes.indexOf(axis),index=indexLevels(amr),inPlane=[0,1,2].filter(i=>i!==normal),bs=Number(amr?.block_size)||8;
    if(normal<0||!Number.isFinite(position))return {cells:[],axes:inPlane,total:0,stride:1};
    const candidates=[],maximum=Number(amr?.domain_size?.[normal]),minimumDx=Math.min(...index.map(info=>info.dx).filter(dx=>dx>0));
    // A plane on the outer maximum face cuts the last interior cell layer.
    // Keep the displayed plane at its requested coordinate, but use a point
    // just inside that face when deciding which AMR block owns the cut.
    const probe=Number.isFinite(maximum)&&Math.abs(position-maximum)<=Math.max(1e-9,minimumDx*1e-6)?maximum-minimumDx*1e-6:position;
    for(const info of index){
      if(visible&&visible[info.id]===false)continue;
      for(let b=0;b<info.count;b++){
        const min=info.blocks.slice(b*3,b*3+3).map(v=>(v-1)*info.size);
        if(probe>=min[normal]-info.dx*1e-8&&probe<min[normal]+info.size-info.dx*1e-8)candidates.push({info,min});
      }
    }
    const total=candidates.length*bs*bs,stride=Math.max(1,Math.ceil(total/limit)),cells=[];let ordinal=0;
    for(const {info,min} of candidates)for(let j=0;j<bs;j++)for(let i=0;i<bs;i++,ordinal++){
      if(ordinal%stride)continue;
      const center=min.slice();center[normal]=probe;center[inPlane[0]]+=(i+.5)*info.dx;center[inPlane[1]]+=(j+.5)*info.dx;
      if(covered(center,info,index))continue;
      cells.push({level:info.id,dx:info.dx,min:[center[inPlane[0]]-info.dx/2,center[inPlane[1]]-info.dx/2],
        max:[center[inPlane[0]]+info.dx/2,center[inPlane[1]]+info.dx/2]});
    }
    return {cells,axes:inPlane,total,stride};
  }
  const envelopeCache=new WeakMap();
  // Render actual exposed cell faces, never every Nth cell. Arbitrary ordinal
  // decimation made striped holes and transparent interior faces overwhelmed
  // the depth buffer. Interior grid cells remain visible in the section view.
  function volumeCells(amr,info){
    let cached=envelopeCache.get(amr);if(!cached){cached=new Map();envelopeCache.set(amr,cached);}
    if(cached.has(info.id))return cached.get(info.id);
    const bs=Number(amr?.block_size)||8,index=indexLevels(amr),own=index.find(item=>item.id===info.id),points=[],faces=[],seen=new Set(),coveredSeen=new Set();let coveredSamples=0;
    if(!own)return {points,faces,total:0,stride:1,coveredSamples};
    for(let block=0;block<info.count;block++){
      const coordinate=info.blocks.slice(block*3,block*3+3),origin=coordinate.map(v=>(v-1)*info.size);
      for(let normal=0;normal<3;normal++)for(const sign of [-1,1]){
        const neighbour=coordinate.slice();neighbour[normal]+=sign;
        if(own.keys.has(neighbour.join(',')))continue;
        const lateral=[0,1,2].filter(axis=>axis!==normal);
        for(let j=0;j<bs;j++)for(let i=0;i<bs;i++){
          const local=[0,0,0];local[normal]=sign<0?0:bs-1;local[lateral[0]]=i;local[lateral[1]]=j;
          const point=origin.map((value,axis)=>value+(local[axis]+.5)*info.dx);
          const cellKey=coordinate.map((v,axis)=>(v-1)*bs+local[axis]).join(',');
          if(covered(point,info,index)){coveredSeen.add(cellKey);continue;}
          if(!seen.has(cellKey)){seen.add(cellKey);points.push(point);}
          const center=point.slice();center[normal]+=sign*info.dx/2;
          faces.push({center,normal,sign,size:info.dx,level:info.id});
        }
      }
    }
    coveredSamples=coveredSeen.size;const result={points,faces,total:info.count*bs**3,stride:1,coveredSamples,envelope:true};cached.set(info.id,result);return result;
  }
  // Exact triangle/plane intersections in solver coordinates. Coplanar shared
  // triangle edges cancel, leaving only the boundary of a coplanar patch.
  function geometrySection(triangles,axis,position,tolerance=1e-8){
    const normal=axes.indexOf(axis),segments=new Map(),coplanarTriangles=new Set(),key=p=>p.map(v=>Math.round(v/tolerance)).join(',');
    if(normal<0||!triangles?.length)return {segments:[],loops:[]};
    const add=(a,b,coplanar=false)=>{if(a.every((v,i)=>Math.abs(v-b[i])<=tolerance))return;const ka=key(a),kb=key(b),name=ka<kb?ka+'|'+kb:kb+'|'+ka,prior=segments.get(name);
      if(prior){prior.coplanar+=coplanar?1:0;prior.crossing=prior.crossing||!coplanar;}
      else segments.set(name,{a,b,coplanar:coplanar?1:0,crossing:!coplanar});};
    for(let offset=0;offset+8<triangles.length;offset+=9){
      const p=[0,3,6].map(start=>Array.from(triangles.slice(offset+start,offset+start+3))),d=p.map(v=>v[normal]-position);
      if(d.every(v=>v>tolerance)||d.every(v=>v< -tolerance))continue;
      if(d.every(v=>Math.abs(v)<=tolerance)){const triangleKey=p.map(key).sort().join('|');if(coplanarTriangles.has(triangleKey))continue;coplanarTriangles.add(triangleKey);for(let i=0;i<3;i++)add(p[i],p[(i+1)%3],true);continue;}
      const hits=[];const hit=point=>{point[normal]=position;if(!hits.some(other=>point.every((v,i)=>Math.abs(v-other[i])<=tolerance)))hits.push(point);};
      for(let i=0;i<3;i++){const j=(i+1)%3;if(Math.abs(d[i])<=tolerance)hit(p[i].slice());if(d[i]*d[j]<0){const t=d[i]/(d[i]-d[j]);hit(p[i].map((v,k)=>v+t*(p[j][k]-v)));}}
      if(hits.length===2)add(hits[0],hits[1]);
    }
    const output=Array.from(segments.values()).filter(item=>item.crossing||item.coplanar===1).map(item=>[item.a,item.b]);
    const adjacency=new Map(),coordinates=new Map();for(const [a,b] of output){const ka=key(a),kb=key(b);coordinates.set(ka,a);coordinates.set(kb,b);for(const [from,to] of [[ka,kb],[kb,ka]]){const list=adjacency.get(from)||[];list.push(to);adjacency.set(from,list);}}
    const visited=new Set(),loops=[];
    for(const start of adjacency.keys()){
      if(visited.has(start)||adjacency.get(start).length!==2)continue;
      let previous=null,current=start;const loop=[];
      while(!visited.has(current)&&adjacency.get(current)?.length===2){visited.add(current);loop.push(coordinates.get(current));const next=adjacency.get(current).find(item=>item!==previous);previous=current;current=next;if(current===start){if(loop.length>=3)loops.push(loop);break;}}
    }
    return {segments:output,loops};
  }
  function sampleField(data,a,b){
    if(!data)return null;
    const a1=axes.indexOf(data.axis1),a2=axes.indexOf(data.axis2),spacing=Number(data.spacing_m);
    if(!(spacing>0)||a1<0||a2<0)return null;
    const i=Math.floor((a-Number(data.origin_m[a1]))/spacing),j=Math.floor((b-Number(data.origin_m[a2]))/spacing);
    if(i<0||j<0||i>=data.n1||j>=data.n2)return null;
    const value=data.values[i+j*data.n1];return typeof value==='number'&&Number.isFinite(value)?value:null;
  }
  window.LudwigMeshInspector={palette,levelInfo,section,volumeCells,geometrySection,sampleField,axes};
})();
