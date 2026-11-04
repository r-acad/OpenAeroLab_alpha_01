"use strict";

/* AMR inspection uses the solver's exported block coordinates. Rendering may
   sample large volumes; a section always uses the actual cell size.
   Two mesh formats: the block mesh (format 3: 8^3 blocks per level) and the
   CPU octree (format 4: per level, disjoint fluid-leaf boxes as inclusive
   0-based cell ranges [i0,i1,j0,j1,k0,k1]). */
(() => {
  const axes=['x','y','z'];
  const palette=[[.25,.64,.91],[.31,.77,.50],[.97,.72,.25],[.94,.36,.34],[.69,.45,.91],[.20,.80,.81],[.94,.48,.72],[.55,.62,.30],[.82,.40,.60],[.40,.55,.80]];
  function isOctree(amr){return amr?.mesh==='octree'||Number(amr?.format_version)>=4&&Array.isArray(amr?.levels)&&amr.levels.some(level=>Array.isArray(level.boxes));}
  function levelId(level,index){return Number(level.level_id)||index+1;}
  function levelInfo(amr){
    if(isOctree(amr))return (amr?.levels||[]).map((level,index)=>({id:levelId(level,index),dx:Number(level.dx),size:Number(level.dx),boxes:level.boxes||[],
      count:Math.floor((level.boxes||[]).length/6),octree:true,level,index}));
    return (amr?.levels||[]).map((level,index)=>({id:levelId(level,index),dx:Number(level.dx),
      size:Number(level.block_phys_size),blocks:level.blocks||[],count:Math.floor((level.blocks||[]).length/3),level,index}));
  }
  function indexLevels(amr){
    return levelInfo(amr).map(info=>({...info,keys:new Set(Array.from({length:info.count},(_,i)=>info.blocks.slice(i*3,i*3+3).join(',')))}));
  }
  function covered(point,info,index){
    for(const fine of index)if(fine.dx<info.dx&&fine.size>0){
      const key=point.map(v=>Math.floor((v+fine.dx*1e-7)/fine.size)+1).join(',');if(fine.keys.has(key))return true;
    }
    return false;
  }
  // Octree box lookup: boxes bucketed on a coarse grid of the level's cells so
  // that "is cell (i,j,k) a fluid leaf of this level" is a short scan.
  const boxIndexCache=new WeakMap();
  function boxIndex(amr,info){
    let cached=boxIndexCache.get(amr);if(!cached){cached=new Map();boxIndexCache.set(amr,cached);}
    if(cached.has(info.id))return cached.get(info.id);
    const bucket=32,buckets=new Map(),boxes=info.boxes;
    for(let b=0;b<info.count;b++){
      const o=b*6,i0=boxes[o],i1=boxes[o+1],j0=boxes[o+2],j1=boxes[o+3],k0=boxes[o+4],k1=boxes[o+5];
      for(let bk=Math.floor(k0/bucket);bk<=Math.floor(k1/bucket);bk++)for(let bj=Math.floor(j0/bucket);bj<=Math.floor(j1/bucket);bj++)for(let bi=Math.floor(i0/bucket);bi<=Math.floor(i1/bucket);bi++){
        const key=bi+','+bj+','+bk;const list=buckets.get(key);if(list)list.push(b);else buckets.set(key,[b]);
      }
    }
    const result={bucket,buckets,contains(i,j,k){
      const list=buckets.get(Math.floor(i/bucket)+','+Math.floor(j/bucket)+','+Math.floor(k/bucket));if(!list)return false;
      for(const b of list){const o=b*6;if(i>=boxes[o]&&i<=boxes[o+1]&&j>=boxes[o+2]&&j<=boxes[o+3]&&k>=boxes[o+4]&&k<=boxes[o+5])return true;}
      return false;}};
    cached.set(info.id,result);return result;
  }
  function octreeSection(amr,axis,position,visible,limit){
    const normal=axes.indexOf(axis),inPlane=[0,1,2].filter(i=>i!==normal),maximum=Number(amr?.domain_size?.[normal]);
    const infos=levelInfo(amr).filter(info=>!(visible&&visible[info.id]===false)&&info.count);
    const minimumDx=Math.min(...infos.map(info=>info.dx).filter(dx=>dx>0));
    const probe=Number.isFinite(maximum)&&Math.abs(position-maximum)<=Math.max(1e-9,minimumDx*1e-6)?maximum-minimumDx*1e-6:position;
    const hits=[];let total=0;
    for(const info of infos){
      const n=Math.floor(probe/info.dx);
      for(let b=0;b<info.count;b++){
        const o=b*6,lo=[info.boxes[o],info.boxes[o+2],info.boxes[o+4]],hi=[info.boxes[o+1],info.boxes[o+3],info.boxes[o+5]];
        if(n<lo[normal]||n>hi[normal])continue;
        const count=(hi[inPlane[0]]-lo[inPlane[0]]+1)*(hi[inPlane[1]]-lo[inPlane[1]]+1);total+=count;hits.push({info,lo,hi,count});
      }
    }
    const stride=Math.max(1,Math.ceil(total/limit)),cells=[];let ordinal=0;
    for(const {info,lo,hi} of hits)for(let j=lo[inPlane[1]];j<=hi[inPlane[1]];j++)for(let i=lo[inPlane[0]];i<=hi[inPlane[0]];i++,ordinal++){
      if(ordinal%stride)continue;
      cells.push({level:info.id,dx:info.dx,min:[i*info.dx,j*info.dx],max:[(i+1)*info.dx,(j+1)*info.dx]});
    }
    return {cells,axes:inPlane,total,stride};
  }
  function section(amr,axis,position,visible=null,limit=70000){
    if(isOctree(amr))return octreeSection(amr,axis,position,visible,limit);
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
  // Octree: the exposed faces of a level are the box-face cells whose outward
  // neighbour is not a fluid leaf of the same level (the other levels show
  // their own envelopes). Large coarse boxes are shown by their box faces.
  function octreeVolumeCells(amr,info){
    let cached=envelopeCache.get(amr);if(!cached){cached=new Map();envelopeCache.set(amr,cached);}
    if(cached.has(info.id))return cached.get(info.id);
    const index=boxIndex(amr,info),points=[],faces=[],seen=new Set();let total=0;
    const limit=400000;
    for(let b=0;b<info.count&&faces.length<limit;b++){
      const o=b*6,lo=[info.boxes[o],info.boxes[o+2],info.boxes[o+4]],hi=[info.boxes[o+1],info.boxes[o+3],info.boxes[o+5]];
      total+=(hi[0]-lo[0]+1)*(hi[1]-lo[1]+1)*(hi[2]-lo[2]+1);
      for(let normal=0;normal<3;normal++)for(const sign of [-1,1]){
        const lateral=[0,1,2].filter(axis=>axis!==normal),fixed=sign<0?lo[normal]:hi[normal];
        for(let j=lo[lateral[1]];j<=hi[lateral[1]];j++)for(let i=lo[lateral[0]];i<=hi[lateral[0]];i++){
          const cell=[0,0,0];cell[normal]=fixed;cell[lateral[0]]=i;cell[lateral[1]]=j;
          const outside=cell.slice();outside[normal]+=sign;
          if(index.contains(outside[0],outside[1],outside[2]))continue;
          const key=cell.join(',');
          if(!seen.has(key)){seen.add(key);points.push(cell.map(v=>(v+.5)*info.dx));}
          const center=cell.map(v=>(v+.5)*info.dx);center[normal]+=sign*info.dx/2;
          faces.push({center,normal,sign,size:info.dx,level:info.id});
          if(faces.length>=limit)break;
        }
      }
    }
    const result={points,faces,total,stride:1,coveredSamples:0,envelope:true,truncated:faces.length>=limit};cached.set(info.id,result);return result;
  }
  // Render actual exposed cell faces, never every Nth cell. Arbitrary ordinal
  // decimation made striped holes and transparent interior faces overwhelmed
  // the depth buffer. Interior grid cells remain visible in the section view.
  function volumeCells(amr,info){
    if(isOctree(amr))return octreeVolumeCells(amr,info);
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
  // Octree box outlines: centre and extent (metres) of every fluid-leaf box.
  function octreeBoxes(info){
    const out=[];for(let b=0;b<info.count;b++){const o=b*6,bx=info.boxes;
      out.push({center:[(bx[o]+bx[o+1]+1)/2*info.dx,(bx[o+2]+bx[o+3]+1)/2*info.dx,(bx[o+4]+bx[o+5]+1)/2*info.dx],
        extent:[(bx[o+1]-bx[o]+1)*info.dx,(bx[o+3]-bx[o+2]+1)*info.dx,(bx[o+5]-bx[o+4]+1)*info.dx]});}
    return out;
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
  const fieldOwnerCache=new WeakMap(),visibleFieldCache=new WeakMap();
  function fieldOrigin(data){
    if(data.origin_m?.length===3)return data.origin_m;
    const origin=[0,0,0];origin[axes.indexOf(data.axis1)]=Number(data.origin_m?.[0]);origin[axes.indexOf(data.axis2)]=Number(data.origin_m?.[1]);origin[axes.indexOf(data.normal_axis)]=Number(data.normal_coord_m);return origin;
  }
  function fieldOwners(data,amr){
    if(data.owner_levels?.length===data.values?.length)return data.owner_levels;
    if(!amr||!data?.values)return [];
    let prior=fieldOwnerCache.get(data);if(prior?.amr===amr)return prior.owners;
    const origin=fieldOrigin(data),normal=axes.indexOf(data.normal_axis),a=axes.indexOf(data.axis1),b=axes.indexOf(data.axis2),spacing=Number(data.spacing_m),octree=isOctree(amr);
    const infos=(octree?levelInfo(amr):indexLevels(amr)).sort((x,y)=>x.dx-y.dx),owners=new Uint8Array(data.values.length);
    for(let j=0;j<data.n2;j++)for(let i=0;i<data.n1;i++){
      const point=origin.slice();point[normal]=Number(data.normal_coord_m);point[a]+=(i+.5)*spacing;point[b]+=(j+.5)*spacing;
      for(const info of infos){const probe=point.map((v,k)=>Number.isFinite(amr.domain_size?.[k])&&Math.abs(v-amr.domain_size[k])<info.dx*1e-6?v-info.dx*1e-6:v);
        const owned=octree?boxIndex(amr,info).contains(...probe.map(v=>Math.floor(v/info.dx))):info.keys.has(probe.map(v=>Math.floor(v/info.size)+1).join(','));
        if(owned){owners[i+j*data.n1]=info.id;break;}
      }
    }
    fieldOwnerCache.set(data,{amr,owners});return owners;
  }
  function visibleField(data,visibility,amr){
    if(!data?.values)return data;
    const key=JSON.stringify(Object.entries(visibility||{}).filter(([,v])=>v===false).sort()),prior=visibleFieldCache.get(data);
    if(prior?.key===key&&prior.amr===amr)return prior.result;
    const owners=fieldOwners(data,amr),result={...data,origin_m:fieldOrigin(data),owner_levels:owners,
      values:data.values.map((value,index)=>visibility?.[owners[index]]===false?null:value)};
    visibleFieldCache.set(data,{key,amr,result});return result;
  }
  function sampleField(data,a,b){
    if(!data)return null;
    const a1=axes.indexOf(data.axis1),a2=axes.indexOf(data.axis2),spacing=Number(data.spacing_m);
    if(!(spacing>0)||a1<0||a2<0)return null;
    const origin=fieldOrigin(data),i=Math.floor((a-Number(origin[a1]))/spacing),j=Math.floor((b-Number(origin[a2]))/spacing);
    if(i<0||j<0||i>=data.n1||j>=data.n2)return null;
    const value=data.values[i+j*data.n1];return typeof value==='number'&&Number.isFinite(value)?value:null;
  }
  window.LudwigMeshInspector={palette,levelInfo,section,volumeCells,geometrySection,sampleField,fieldOrigin,fieldOwners,visibleField,axes,isOctree,octreeBoxes};
})();
