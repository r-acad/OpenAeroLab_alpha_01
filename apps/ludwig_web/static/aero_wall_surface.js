/* Native wall-link samples. Geometry and values retain their native identities;
   drawing creates bounded GPU batches, never a subdivided solver STL. */
((root)=>{
  'use strict';
  const magic=(view,word)=>view.byteLength>=word.length&&Array.from(word,(c,i)=>view.getUint8(i)===c.charCodeAt(0)).every(Boolean);
  const count64=(view,offset)=>{const n=Number(view.getBigUint64(offset,true));if(!Number.isSafeInteger(n)||n<=0)throw new Error('Invalid native wall sample count.');return n;};
  const finite=Number.isFinite;
  // A cut through a cubic fluid cell can extend to its three-dimensional
  // circumradius. Each native link is displayed as a two-triangle tangent
  // patch at its measured intersection. Patches are display support, not area weights.
  const footprintRadius=dx=>Math.sqrt(3)*.5*dx;
  function readFrame(buffer){
    const v=new DataView(buffer);if(v.byteLength<128||!magic(v,'LBMWFLD\0')||v.getUint32(8,true)!==1||v.getUint32(12,true)!==1)throw new Error('Unsupported native wall field.');
    const count=count64(v,16);if(128+4*count!==v.byteLength)throw new Error('Truncated native wall field.');
    const geometrySha256=new TextDecoder().decode(new Uint8Array(buffer,24,64)),step=Number(v.getBigInt64(88,true)),time=v.getFloat64(96,true),rho=v.getFloat64(104,true),speed=v.getFloat64(112,true),q=v.getFloat64(120,true);
    if(!/^[0-9a-f]{64}$/.test(geometrySha256)||!Number.isSafeInteger(step)||step<0||!finite(time)||time<0||![rho,speed,q].every(x=>finite(x)&&x>0)||Math.abs(q-.5*rho*speed*speed)>q*1e-8)throw new Error('Invalid native wall field references.');
    const pressure=new Float32Array(buffer,128,count),cp=new Float32Array(count);let missing=0;
    for(let i=0;i<count;i++){cp[i]=finite(pressure[i])?pressure[i]/q:NaN;if(!finite(pressure[i]))missing++;}
    return {representation:'wall_samples',count,sampleCount:count,geometrySha256,step,time,pressure,cp,missing,field_units:{pressure:'Pa',cp:'1'},surface_offset:[0,0,0],geometry_frame:'domain'};
  }
  function readGeometry(buffer){
    const view=new DataView(buffer);if(view.byteLength<24||!magic(view,'LBMWGEO\0')||view.getUint32(8,true)!==1||view.getUint32(12,true)!==36)throw new Error('Unsupported native wall geometry.');
    const count=count64(view,16);if(24+36*count!==view.byteLength)throw new Error('Truncated native wall geometry.');
    const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];let ambiguous=0,invalid=0,minimumDx=Infinity,maximumDx=0;
    for(let i=0;i<count;i++){
      const k=24+36*i,flags=view.getUint16(k+34,true),dx=view.getFloat32(k+24,true),level=view.getUint8(k+32),links=view.getUint8(k+33),parent=view.getUint32(k+28,true);
      let norm=0;
      for(let a=0;a<3;a++){const x=view.getFloat32(k+a*4,true),n=view.getFloat32(k+12+a*4,true);if(!finite(x)||!finite(n))throw new Error('Nonfinite native wall geometry.');min[a]=Math.min(min[a],x);max[a]=Math.max(max[a],x);norm+=n*n;}
      if(!finite(dx)||dx<=0||Math.abs(norm-1)>.005||!parent||!level||level>255||!links||(flags&~3))throw new Error('Invalid native wall normal, size or sample identity.');
      if(!(flags&1))invalid++;if(flags&2)ambiguous++;minimumDx=Math.min(minimumDx,dx);maximumDx=Math.max(maximumDx,dx);
    }
    return {buffer,view,count,bounds:{min,max},ambiguous,invalid,minimumDx,maximumDx};
  }
  async function verifyGeometry(buffer,expected,subtle=root.crypto?.subtle){
    if(!subtle)throw new Error('Native surface identity verification requires a secure browser connection (localhost or HTTPS).');
    const digest=Array.from(new Uint8Array(await subtle.digest('SHA-256',buffer)),x=>x.toString(16).padStart(2,'0')).join('');
    if(digest!==expected)throw new Error('Native wall geometry SHA-256 does not match this field.');return digest;
  }
  let cachedGeometry=null;
  async function load(record,fetcher=root.fetch.bind(root)){
    const response=await fetcher(record.url,{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const buffer=await response.arrayBuffer(),surface=readFrame(buffer);
    if(!root.crypto?.subtle)throw new Error('Native surface verification requires localhost or HTTPS.');
    surface.frameSha256=Array.from(new Uint8Array(await root.crypto.subtle.digest('SHA-256',buffer)),x=>x.toString(16).padStart(2,'0')).join('');
    if(record.frame_sha256&&record.frame_sha256!==surface.frameSha256)throw new Error('Native pressure frame SHA-256 does not match the selected result.');
    if(record.step!=null&&Number(record.step)!==surface.step||record.geometry_sha256&&record.geometry_sha256!==surface.geometrySha256)throw new Error('Native wall snapshot differs from the selected result.');
    const geometryUrl=record.geometry_url||record.url.replace(/[^/?]+(?:\?.*)?$/,`surface_wall_geometry_${surface.geometrySha256}.lbmwg`);
    if(!cachedGeometry||cachedGeometry.sha!==surface.geometrySha256){
      const entry={sha:surface.geometrySha256,promise:null};
      entry.promise=(async()=>{const r=await fetcher(geometryUrl,{cache:'no-store'});if(!r.ok)throw new Error(`Wall geometry HTTP ${r.status}`);const buffer=await r.arrayBuffer();await verifyGeometry(buffer,surface.geometrySha256);return readGeometry(buffer);})();cachedGeometry=entry;
      entry.promise.catch(()=>{if(cachedGeometry===entry)cachedGeometry=null;});
    }
    surface.geometry=await cachedGeometry.promise;
    if(surface.geometry.count!==surface.count)throw new Error('Native wall field/geometry sample counts differ.');
    surface.bounds=surface.geometry.bounds;surface.sourceUrl=record.url;surface.geometryUrl=geometryUrl;return surface;
  }
  const vertexSource=`precision highp float;
    attribute vec3 position; attribute vec4 world0; attribute vec4 world1; attribute vec4 world2; attribute vec4 world3; attribute vec4 wallColor;
    uniform mat4 viewProjection; varying vec4 colour; varying vec3 pointWorld;
    void main(){mat4 transform=mat4(world0,world1,world2,world3);vec4 point=transform*vec4(position,1.0);pointWorld=point.xyz;gl_Position=viewProjection*point;colour=wallColor;}`;
  const fragmentSource=`precision highp float; varying vec4 colour; varying vec3 pointWorld;
    uniform float opacity; uniform float clipEnabled; uniform vec4 sampleClipPlane; uniform float clipEnabled2; uniform vec4 sampleClipPlane2;
    void main(){if(clipEnabled>0.5&&dot(vec4(pointWorld,1.0),sampleClipPlane)>0.0)discard;if(clipEnabled2>0.5&&dot(vec4(pointWorld,1.0),sampleClipPlane2)>0.0)discard;gl_FragColor=vec4(colour.rgb,opacity);}`;
  function render(surface,values,range,{B,scene,keep,color,ground=false,opacity=1,batchSize=125000}){
    const geometry=surface.geometry,view=geometry.view,objects=[],meshes=[];
    const retain=o=>{objects.push(o);keep(o);return o;};
    const material=retain(new B.ShaderMaterial('Native wall samples material',scene,{vertexSource,fragmentSource},{attributes:['position','world0','world1','world2','world3','wallColor'],uniforms:['viewProjection','opacity','clipEnabled','sampleClipPlane','clipEnabled2','sampleClipPlane2']}));
    material.backFaceCulling=false;material.alpha=opacity;material.needAlphaBlending=()=>material.alpha<1;material.setFloat('opacity',opacity);
    material.onBindObservable.add(()=>{material.setFloat('opacity',material.alpha);material.disableDepthWrite=material.alpha<1;for(const suffix of ['', '2']){const p=material['clipPlane'+suffix]||scene['clipPlane'+suffix];material.setFloat('clipEnabled'+suffix,p?1:0);material.setVector4('sampleClipPlane'+suffix,p?new B.Vector4(p.normal.x,p.normal.y,p.normal.z,p.d):B.Vector4.Zero());}});
    const map=(x,y,z)=>ground?[x,y,z]:[x,z,y];let drawn=0;
    for(let start=0;start<surface.count;start+=batchSize){
      const end=Math.min(surface.count,start+batchSize),matrices=new Float32Array((end-start)*16),colors=new Float32Array((end-start)*4);let used=0;
      for(let i=start;i<end;i++){
        const k=24+36*i;if(!(view.getUint16(k+34,true)&1)||!finite(values[i]))continue;
        const dx=view.getFloat32(k+24,true),n=map(view.getFloat32(k+12,true),view.getFloat32(k+16,true),view.getFloat32(k+20,true)),p=map(view.getFloat32(k,true),view.getFloat32(k+4,true),view.getFloat32(k+8,true));
        const tangent=Math.abs(n[2])<.9?[-n[1],n[0],0]:[0,-n[2],n[1]],length=Math.hypot(...tangent);for(let a=0;a<3;a++)tangent[a]/=length;
        const bitangent=[n[1]*tangent[2]-n[2]*tangent[1],n[2]*tangent[0]-n[0]*tangent[2],n[0]*tangent[1]-n[1]*tangent[0]],base=used*16,radius=footprintRadius(dx);
        // Move only the visualization footprint a small fraction of its own
        // cell spacing toward the fluid; retain ordinary depth occlusion.
        for(let a=0;a<3;a++){matrices[base+a]=radius*tangent[a];matrices[base+4+a]=radius*bitangent[a];matrices[base+8+a]=n[a];matrices[base+12+a]=p[a]+n[a]*dx*.02;}matrices[base+15]=1;
        const rgb=color(values[i],range);colors.set([rgb[0],rgb[1],rgb[2],1],used*4);used++;
      }
      if(!used)continue;
      const mesh=retain(new B.Mesh(`Native wall samples ${start}`,scene)),data=new B.VertexData();data.positions=[-1,-1,0,1,-1,0,1,1,0,-1,1,0];data.indices=[0,1,2,0,2,3];data.applyToMesh(mesh);
      mesh.material=material;mesh.isPickable=false;mesh.metadata={modelSurface:true,wallSamples:true,linkPatchMesh:true};
      mesh.thinInstanceSetBuffer('matrix',used===end-start?matrices:matrices.subarray(0,used*16),16,true);mesh.thinInstanceSetBuffer('wallColor',used===end-start?colors:colors.subarray(0,used*4),4,true);
      mesh.thinInstanceCount=used;mesh.thinInstanceRefreshBoundingInfo();meshes.push(mesh);drawn+=used;
    }
    return {objects,meshes,material,drawn,total:surface.count,patchTriangles:2*drawn};
  }
  function section(surface,values,axis,position){
    const geometry=surface.geometry;if(!geometry)return [];const view=geometry.view,n='xyz'.indexOf(axis),out=[];
    for(let i=0;i<surface.count;i++){
      const k=24+i*36;if(!finite(values[i])||!(view.getUint16(k+34,true)&1))continue;
      const radius=footprintRadius(view.getFloat32(k+24,true)),distance=position-view.getFloat32(k+n*4,true);if(Math.abs(distance)>radius)continue;
      const p=[0,1,2].map(a=>view.getFloat32(k+a*4,true)),normal=[0,1,2].map(a=>view.getFloat32(k+12+a*4,true)),denom=1-normal[n]*normal[n];
      if(denom<1e-8||distance*distance>radius*radius*denom)continue;
      const centre=p.map((x,a)=>x+distance*((a===n?1:0)-normal[n]*normal[a])/denom),direction=[0,0,0];direction[(n+1)%3]=-normal[(n+2)%3];direction[(n+2)%3]=normal[(n+1)%3];const norm=Math.sqrt(denom),half=Math.sqrt(Math.max(0,radius*radius-distance*distance/denom));
      out.push({value:values[i],points:[centre.map((x,a)=>x-half*direction[a]/norm),centre.map((x,a)=>x+half*direction[a]/norm)]});
    }return out;
  }
  root.LudwigAeroWallSurface={readFrame,readGeometry,verifyGeometry,load,render,section};
  if(typeof module==='object'&&module.exports)module.exports=root.LudwigAeroWallSurface;
})(globalThis);
