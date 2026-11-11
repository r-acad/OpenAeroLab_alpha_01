// Browser module tests; no solver or case is started.
const assert=require('node:assert/strict'),path=require('node:path'),crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..'),W=require(path.join(root,'apps/ludwig_web/static/aero_wall_surface.js')),B=require(path.join(root,'apps/airfoil_web/static/vendor/babylon.js'));
const checks=[];function check(value,message){assert.ok(value,message);checks.push(message);}function equal(a,b,message){assert.deepEqual(a,b,message);checks.push(message);}
function geometry(){const bytes=new ArrayBuffer(24+36*3),v=new DataView(bytes);new Uint8Array(bytes,0,8).set(Buffer.from('LBMWGEO\0'));v.setUint32(8,1,true);v.setUint32(12,36,true);v.setBigUint64(16,3n,true);for(let i=0;i<3;i++){const k=24+36*i;v.setFloat32(k,.1+i*.1,true);v.setFloat32(k+20,1,true);v.setFloat32(k+24,.1,true);v.setUint32(k+28,1,true);v.setUint8(k+32,3);v.setUint8(k+33,2);v.setUint16(k+34,i===1?3:1,true);}return bytes;}
function frame(hash,n=3){const bytes=new ArrayBuffer(128+4*n),v=new DataView(bytes);new Uint8Array(bytes,0,8).set(Buffer.from('LBMWFLD\0'));v.setUint32(8,1,true);v.setUint32(12,1,true);v.setBigUint64(16,BigInt(n),true);new Uint8Array(bytes,24,64).set(Buffer.from(hash));v.setBigInt64(88,12n,true);[.25,1,2,2].forEach((x,i)=>v.setFloat64(96+8*i,x,true));const p=new Float32Array(bytes,128,n);p.set([2,-3,NaN].slice(0,n));return bytes;}
(async()=>{
  const bytes=geometry(),sha=crypto.createHash('sha256').update(new Uint8Array(bytes)).digest('hex');
  const g=W.readGeometry(bytes),s=W.readFrame(frame(sha));s.geometry=g;
  equal(g.count,3,'All native samples are retained');equal(g.ambiguous,1,'Ambiguous/grazing sides are counted explicitly');
  equal(Array.from(s.pressure),[2,-3,NaN],'Pressure is not averaged to original facets');equal(Array.from(s.cp),[1,-1.5,NaN],'Cp uses the stored physical dynamic pressure');equal(s.missing,1,'Missing pressure stays missing');
  equal(g.view.getUint32(24+28,true),g.view.getUint32(24+36+28,true),'Different pressures intentionally belong to the same parent facet');
  equal(await W.verifyGeometry(bytes,sha,crypto.webcrypto.subtle),sha,'Exact static geometry SHA is accepted');
  await assert.rejects(W.verifyGeometry(bytes,'0'.repeat(64),crypto.webcrypto.subtle),/SHA-256/);checks.push('Mismatched geometry is rejected');
  assert.throws(()=>W.readFrame(frame(sha).slice(0,-1)),/Truncated/);checks.push('Incomplete field is rejected');
  assert.throws(()=>W.readGeometry(bytes.slice(0,-1)),/Truncated/);checks.push('Incomplete geometry is rejected');
  const bad=bytes.slice(0);new DataView(bad).setFloat32(24+20,0,true);assert.throws(()=>W.readGeometry(bad),/normal/);checks.push('Invalid normal is rejected');
  const large=W.readFrame(frame(sha,5_000_001));equal(large.count,5_000_001,'The old five-million-triangle limit does not truncate native samples');
  const engine=new B.NullEngine(),scene=new B.Scene(engine);new B.ArcRotateCamera('test',0,1,3,B.Vector3.Zero(),scene);
  const held=[],bundle=W.render(s,s.cp,{lo:-2,hi:2},{B,scene,keep:o=>held.push(o),color:value=>value>0?[1,0,0]:[0,0,1],batchSize:1});
  equal(bundle.drawn,2,'Missing pressures create no colored patch');equal(bundle.total,3,'Rendering reports the complete native count');equal(bundle.meshes.length,2,'GPU instances are partitioned into bounded batches');
  equal(bundle.meshes.map(m=>m.thinInstanceCount),[1,1],'All valid samples get their own GPU instance');
  const first=bundle.meshes[0]._thinInstanceDataStorage.matrixData,second=bundle.meshes[1]._thinInstanceDataStorage.matrixData;
  check(Math.abs(first[12]-.1)<1e-6&&Math.abs(second[12]-.2)<1e-6,'Samples retain their individual intersection positions');
  const radius=Math.hypot(first[0],first[1],first[2]);
  check(Math.abs(radius-Math.sqrt(3)*.05)<1e-6,'Local cube circumradius sets the display footprint');
  check(radius>Math.hypot(.05,.05),'Adjacent wet cells cover a regular grid corner instead of leaving inscribed-circle holes');
  check(Math.abs(first[13]-.002)<1e-7&&first[14]===0,'Display footprint has a two-percent-cell offset toward the fluid, with no lateral drift');
  check(bundle.material.disableDepthWrite===false,'Opaque native samples retain depth occlusion against opposite surfaces');
  const c1=bundle.meshes[0]._userThinInstanceBuffersStorage.data.wallColor,c2=bundle.meshes[1]._userThinInstanceBuffersStorage.data.wallColor;
  equal(Array.from(c1),[1,0,0,1],'First value reaches its own GPU color buffer');equal(Array.from(c2),[0,0,1,1],'Second value on the same facet retains a different GPU color');
  check(bundle.material.backFaceCulling===false,'Samples remain readable on either triangle side');equal(bundle.material.getClassName(),'ShaderMaterial','Native sample GPU shader material was created');
  const ray={origin:new B.Vector3(.1,1,0),direction:new B.Vector3(0,-1,0)};
  const picked=W.pick(s,s.cp,ray);
  equal(picked.sample,0,'A visible native patch is pickable without an original STL');
  check(Math.abs(picked.point[0]-.1)<1e-6&&Math.abs(picked.point[1]-.002)<1e-6,'Picking returns the actual displayed patch intersection');
  equal(W.pick(s,s.cp,ray,{visible:()=>false}),null,'A clipping plane can exclude an otherwise picked patch');
  equal(W.pick(s,s.cp,{origin:new B.Vector3(.3,1,0),direction:new B.Vector3(0,-1,0)}),null,'Missing-valued patches are not pickable');
  const groundRay={origin:new B.Vector3(.1,0,1),direction:new B.Vector3(0,0,-1)},groundPick=W.pick(s,s.cp,groundRay,{ground:true});
  check(groundPick&&Math.abs(groundPick.point[2]-.002)<1e-6,'Picking uses the ground coordinate convention');
  bundle.material.alpha=.4;check(bundle.material.needAlphaBlending(),'Native samples support model opacity');
  const sections=W.section(s,s.cp,'x',.1);equal(sections.length,1,'A cut intersects only the local sample footprint');equal(sections[0].value,1,'Section curves retain local pressure rather than a facet mean');
  equal(W.section(s,s.cp,'x',1).length,0,'No curve is invented across an unsampled gap');
  scene.render();scene.dispose();engine.dispose();
  const loaded=await W.load({url:'/surface_000012.lbmw',geometry_url:'/g.lbmwg',step:12,geometry_sha256:sha},async url=>({ok:true,arrayBuffer:async()=>url.endsWith('.lbmw')?frame(sha):bytes}));
  equal(loaded.frameSha256,crypto.createHash('sha256').update(new Uint8Array(frame(sha))).digest('hex'),'Frame identity covers the actual loaded pressure payload');equal(loaded.count,3,'Binary frame and geometry bind without an AMR triangle array');
  await assert.rejects(W.load({url:'/surface_000012.lbmw',geometry_url:'/g.lbmwg',frame_sha256:'f'.repeat(64)},async()=>({ok:true,arrayBuffer:async()=>frame(sha)})),/pressure frame SHA-256/);checks.push('Wrong requested pressure frame is rejected');
  console.log(JSON.stringify({passed:checks.length,checks},null,2));
})().catch(error=>{console.error(error);process.exitCode=1;});
