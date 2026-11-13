/* Dependency-free DOM fixture for the actual browser workflow component.
 * This checks control movement and behavior, not rendering or browser layout. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const root=process.argv[2];let checks=0;
function ok(value,message){checks++;assert.ok(value,message);}
const all=node=>node.childNodes.flatMap(child=>child.nodeType===1?[child,...all(child)]:[]);
const list=selector=>selector.split(',').map(value=>value.trim());
function matches(node,selector,scope){
  const direct=selector.lastIndexOf('>');
  if(direct>=0){const left=selector.slice(0,direct).trim(),right=selector.slice(direct+1).trim();return matches(node,right,scope)&&!!node.parentElement&&(left===':scope'?node.parentElement===scope:matches(node.parentElement,left,scope));}
  const space=selector.lastIndexOf(' ');
  if(space>=0){const left=selector.slice(0,space).trim(),right=selector.slice(space+1);if(!matches(node,right,scope))return false;for(let p=node.parentElement;p;p=p.parentElement)if(matches(p,left,scope))return true;return false;}
  if(selector===':scope')return node===scope;
  const negations=[...selector.matchAll(/:not\(([^)]+)\)/g)];
  if(negations.some(match=>matches(node,match[1],scope)))return false;
  selector=selector.replace(/:not\([^)]+\)/g,'');
  for(const match of selector.matchAll(/\[([^\]=]+)(?:=(?:"([^"]*)"|'([^']*)'|([^\]]+)))?\]/g)){
    if(!node.hasAttribute(match[1]))return false;
    const value=match[2]??match[3]??match[4];if(value!==undefined&&node.getAttribute(match[1])!==value)return false;
  }
  selector=selector.replace(/\[[^\]]+\]/g,'');
  const tag=selector.match(/^[a-z][\w-]*/i);if(tag&&node.tagName!==tag[0].toUpperCase())return false;
  const id=selector.match(/#([\w-]+)/);if(id&&node.id!==id[1])return false;
  for(const match of selector.matchAll(/\.([\w-]+)/g))if(!node.classList.contains(match[1]))return false;
  return true;
}
class Text{constructor(text){this.nodeType=3;this.textContent=text;this.parentElement=null;}}
class Element{
  constructor(name,attributes={}){this.nodeType=1;this.tagName=name.toUpperCase();this.attrs={...attributes};this.childNodes=[];this.parentElement=null;this.events={};this.style={};
    this.dataset=new Proxy({},{get:(_,key)=>this.attrs['data-'+String(key).replace(/[A-Z]/g,c=>'-'+c.toLowerCase())],set:(_,key,value)=>{this.attrs['data-'+String(key).replace(/[A-Z]/g,c=>'-'+c.toLowerCase())]=String(value);return true;}});
    this.classList={contains:value=>(this.attrs.class||'').split(/\s+/).includes(value),add:(...values)=>values.forEach(value=>this.classList.toggle(value,true)),remove:(...values)=>values.forEach(value=>this.classList.toggle(value,false)),toggle:(value,force)=>{const values=new Set((this.attrs.class||'').split(/\s+/).filter(Boolean)),on=force===undefined?!values.has(value):force;on?values.add(value):values.delete(value);this.attrs.class=[...values].join(' ');return on;}};
  }
  get attributes(){return Object.entries(this.attrs).map(([name,value])=>({name,value}));}
  setAttribute(name,value){this.attrs[name]=String(value);}getAttribute(name){return this.attrs[name]??null;}hasAttribute(name){return name in this.attrs;}removeAttribute(name){delete this.attrs[name];}
  get id(){return this.attrs.id||'';}set id(value){this.attrs.id=value;}
  get className(){return this.attrs.class||'';}set className(value){this.attrs.class=value;}
  get children(){return this.childNodes.filter(child=>child.nodeType===1);}
  get hidden(){return this.hasAttribute('hidden');}set hidden(value){value?this.setAttribute('hidden',''):this.removeAttribute('hidden');}
  get disabled(){return this.hasAttribute('disabled');}set disabled(value){value?this.setAttribute('disabled',''):this.removeAttribute('disabled');}
  get checked(){return this._checked??this.hasAttribute('checked');}set checked(value){this._checked=!!value;}
  get options(){return all(this).filter(node=>node.tagName==='OPTION');}
  get type(){return this.attrs.type||(this.tagName==='INPUT'?'text':'');}
  get value(){if(this.tagName==='SELECT'&&this._value!==undefined)return this.options.some(node=>node.value===this._value)?this._value:'';return this._value??(this.tagName==='SELECT'?(this.options.find(node=>node.hasAttribute('selected'))||this.options[0])?.getAttribute('value')||'':this.getAttribute('value')||'');}set value(value){this._value=String(value);if(this.tagName==='SELECT'&&!this.options.some(node=>node.value===this._value))this._value='';}
  get textContent(){return this.childNodes.map(node=>node.textContent).join('');}set textContent(value){this.replaceChildren(new Text(String(value)));}
  get innerHTML(){throw new Error('Fixture must not inspect serialized HTML');}set innerHTML(value){this.replaceChildren();parse(value,this);}
  append(...nodes){for(let node of nodes){if(typeof node==='string')node=new Text(node);if(node.parentElement){const i=node.parentElement.childNodes.indexOf(node);node.parentElement.childNodes.splice(i,1);}this.childNodes.push(node);node.parentElement=this;}}
  appendChild(node){this.append(node);return node;}
  add(node){this.append(node);}remove(){if(!this.parentElement)return;const parent=this.parentElement;parent.childNodes.splice(parent.childNodes.indexOf(this),1);this.parentElement=null;}
  replaceChildren(...nodes){for(const child of this.childNodes)child.parentElement=null;this.childNodes=[];this.append(...nodes);}
  replaceWith(node){const parent=this.parentElement;if(!parent)return;const i=parent.childNodes.indexOf(this);parent.childNodes[i]=node;node.parentElement=parent;this.parentElement=null;}
  querySelectorAll(selector){return all(this).filter(node=>list(selector).some(value=>matches(node,value,this)));}querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
  closest(selector){for(let p=this;p;p=p.parentElement)if(list(selector).some(value=>matches(p,value,this)))return p;return null;}
  addEventListener(event,handler){(this.events[event]??=[]).push(handler);}dispatchEvent(event){for(const handler of this.events[event.type]||[])handler({...event,target:this,preventDefault(){},stopPropagation(){}});return true;}
  focus(){document.activeElement=this;}
  close(){this.open=false;}showModal(){this.open=true;}
}
function parse(html,container){
  const voids=new Set('area base br col embed hr img input link meta param source track wbr'.split(' ')),stack=[container];
  for(const token of html.replace(/<!--[\s\S]*?-->/g,'').matchAll(/<\/?[A-Za-z][^>]*(?:>|$)|[^<]+/g)){
    const text=token[0];if(!text.startsWith('<')){stack.at(-1).append(new Text(text));continue;}
    const close=text.match(/^<\/([\w:-]+)/);if(close){if(stack.length>1)stack.pop();continue;}
    const tag=text.match(/^<([\w:-]+)/);if(!tag)continue;
    const attrs={};for(const attribute of text.slice(tag[0].length).replace(/\/?\s*>$/,'').matchAll(/([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g))attrs[attribute[1]]=attribute[2]??attribute[3]??attribute[4]??'';
    const node=new Element(tag[1],attrs);stack.at(-1).append(node);if(!voids.has(tag[1].toLowerCase())&&!text.endsWith('/>'))stack.push(node);
  }
}
const document=new Element('document');document.createElement=name=>new Element(name);document.getElementById=id=>all(document).find(node=>node.id===id)||null;
parse(fs.readFileSync(path.join(root,'apps/airfoil_web/static/index.html'),'utf8'),document);document.body=document.querySelector('body');
const before=new Map(all(document).filter(node=>node.id&&['INPUT','SELECT','BUTTON'].includes(node.tagName)).map(node=>[node.id,node]));
const callbacks={},frames=[],state={result:null,preview:null,embeddedAirfoil:null,liveField:null,liveHistory:{},logLineCount:0,activeTab:'geometry',job:null};
const window={AirfoilStudies:{path:null},AirfoilBenchmarks:{current:()=>null},addEventListener:(event,handler)=>(callbacks[event]??=[]).push(handler)};
const el=Object.fromEntries(all(document).filter(node=>node.id).map(node=>[node.id,node]));
const context={window,document,state,el,console,requestAnimationFrame:fn=>frames.push(fn),MutationObserver:class{observe(){}},Event:class{constructor(type){this.type=type;}},fetch:()=>{throw Error('Unexpected network access');},setTimeout(){},clearTimeout(){},
  currentResult:()=>state.result?.results?.[0],currentMesh:()=>state.preview?.mesh,
  selectedBackends:value=>({both:['jxfoil','transonic'],cfd_both:['transonic','cavitation'],xfoil_cfd_both:['xfoil','transonic','cavitation'],xfoil_lowspeed:['xfoil','lowspeed']})[value]||[value],
  isCfdBackend:name=>['transonic','cavitation','lowspeed','lbm'].includes(name),switchTab:name=>{state.activeTab=name;}};
vm.createContext(context);
for(const file of ['research_settings.js','output_controls.js','workflow.js'])vm.runInContext(fs.readFileSync(path.join(root,'apps/airfoil_web/static/scripts',file),'utf8'),context,{filename:file});
const $=id=>document.getElementById(id),workflow=window.AirfoilWorkflow;
function refresh(){workflow.refresh();let n=0;while(frames.length){if(n++>20)throw Error('Refresh loop');frames.shift()();}}
function tab(name){return document.querySelector(`[data-workflow-target=${name}]`);}
ok(!document.querySelector('.top-menu'),'No dropdown navigation remains');
ok(!document.querySelector('.setup-column details'),'Setup does not hide options in disclosure widgets');
for(const [id,node] of before)ok($(id)===node,'Existing control/listener node retained: '+id);
for(const [id,node] of Object.entries(el))ok(node===$(id),'Registered control or container refers to the live node: '+id);
ok(document.querySelectorAll('[id]').length===new Set(document.querySelectorAll('[id]').map(node=>node.id)).size,'Runtime IDs remain unique');
ok(!tab('flow').hidden&&tab('mesh').hidden&&tab('result-controls').hidden,'Empty case keeps setup, hides no-data views');
tab('flow').dispatchEvent({type:'click'});ok(!document.querySelector('[data-airfoil-panel=flow]').hidden,'Flow tab opens directly');
ok(!$('sweep-enabled').closest('section').hidden,'Multiple angles is visible inside Flow');
ok($('lowspeed-tu').closest('[data-airfoil-panel]').dataset.airfoilPanel==='flow','Freestream turbulence grouped with flow');
ok($('lowspeed-resolution').closest('[data-airfoil-panel]').dataset.airfoilPanel==='mesh','Only one mesh resolution control, in Mesh');
state.preview={geometry:{x:[1,0,1],y:[-.001,0,.001]},mesh:{}};refresh();
ok(!tab('mesh').hidden&&!tab('display').hidden,'Geometry makes Mesh and Display applicable');
tab('mesh').dispatchEvent({type:'click'});ok(state.activeTab==='geometry','Mesh tab opens its geometry inspection');
el.backend.value='jxfoil';refresh();ok(tab('flap').hidden,'Panel-only route hides unsupported flap');
ok($('lowspeed-tu').closest('[data-airfoil-panel]').dataset.airfoilPanel==='flow'&&$('lowspeed-resolution').closest('[data-airfoil-panel]').dataset.airfoilPanel==='mesh','Low-speed values stay in single locations');
ok($('lowspeed-tu').closest('label').hidden,'Low-speed physics hidden for panel solver');
ok($('trailing-edge-thickness').closest('label').hidden,'Compressible TE override hidden for panel solver');
el.backend.value='transonic';refresh();ok(!tab('solver').hidden&&tab('flap').hidden,'Compressible solver controls available, unsupported flap hidden');
el.backend.value='lowspeed';$('gurney-enabled').checked=true;refresh();
ok(!$('gurney-refine').closest('label').hidden,'Flap mesh refinement appears only for active low-speed flap');
ok([...el.backend.options].find(node=>node.value==='transonic').hidden,'Unsupported solver choice hidden while flap is active');
$('gurney-enabled').checked=false;refresh();ok($('gurney-refine').closest('label').hidden,'Disabled flap hides its refinement');
state.result={results:[{backend:'lowspeed',alpha_deg:4,cl:1,fields:{full_artifact:'/api/jobs/example/fields'},diagnostics:{solver_outcome:'INCOMPLETE'}}]};refresh();
ok(!tab('result-controls').hidden&&!$('tab-button-results').hidden&&!$('tab-button-field').hidden,'Available fields/forces shown');
ok($('tab-button-polar').hidden,'One angle does not offer an empty polar');
ok(!$('tab-button-diagnostics').hidden,'Actual solver diagnostics available');
state.result.results.push({backend:'lowspeed',alpha_deg:8,cl:1.5});refresh();ok(!$('tab-button-polar').hidden,'Two angles enable polar');
state.logLineCount=3;refresh();ok(!$('tab-button-log').hidden,'Actual log enables log view');
state.result={results:[{backend:'jxfoil',alpha_deg:4,cl:1,boundary_layer:{upper:[{}]}}]};refresh();
ok(!$('tab-button-field').hidden&&$('tab-button-field').textContent==='Boundary layer','Panel boundary-layer data remain accessible');
state.result=null;state.liveField={field:{samples:{x:[1],u:[.5]}}};refresh();ok(!$('tab-button-field').hidden,'Live field appears before a final point');
window.AirfoilStudies.path='C:\\studies\\NACA2412.ludwig-study';refresh();
ok($('output-folder').placeholder==='C:\\studies','Saved study folder shown as automatic destination');
ok($('study-result-location').textContent.includes('C:\\studies'),'Study shows result location');
state.job={results_directory:'C:\\studies\\20261112_run'};refresh();ok(!$('current-run-location').hidden&&$('current-run-location').textContent.includes('20261112_run'),'Current actual run folder shown');
window.LudwigOutputControls.setCapabilities({lowspeed:{surface_vtk:true,surface_lbmp:false,volume_vtk:true,numeric_slices:true,slice_png:false,csv_history:true,renders_3d:false,timed_snapshots:true}},['lowspeed']);
ok($('output-surface_lbmp').disabled&&$('output-surface_lbmp').closest('label').hidden,'Unsupported output hidden, not only disabled');
ok($('output-render').closest('label').hidden,'Unsupported 3D render hidden');
ok($('output-camera').closest('.output-two-fields').hidden,'Unused camera hidden');
ok(!$('snapshot-interval-field').hidden,'Supported timed snapshots retained');
state.result=null;state.liveField=null;state.liveHistory={};state.logLineCount=0;state.job=null;
window.LudwigOutputControls.render(null);refresh();
ok($('output-results').closest('section').hidden,'Pre-run export placeholder stays hidden');
ok(tab('result-controls').hidden,'No results tab before any real result or export status');
for(const status of ['failed','partial']){
  const warning='Requested export could not be written';
  state.job={exports:{status,files:[],warnings:[warning]}};
  window.LudwigOutputControls.render(state.job);refresh();
  ok(!$('output-results').closest('section').hidden,status+' export status stays visible with zero files');
  ok(!tab('result-controls').hidden,status+' export status makes Results accessible without solver data');
  ok($('output-results').textContent.includes(warning),status+' export warning remains readable');
  ok($('output-results').textContent.includes(status),status+' export status remains readable');
}
state.job={exports:{status:'completed',folder:'C:\\studies\\run\\exports',files:[{name:'field.vtu',url:'/api/jobs/example/files/field.vtu'}]}};
window.LudwigOutputControls.render(state.job);refresh();
ok(!$('output-results').closest('section').hidden,'Actual export files shown');
ok($('tab-button-campaign').hidden,'Historical development dashboard is not a primary empty case view');
// Exercise the actual study open/save implementation and request builder.
// Drawing, native file dialogs and solver execution are intentionally stubbed;
// no implementation of numerical setting serialization is duplicated here.
const appSource=fs.readFileSync(path.join(root,'apps/airfoil_web/static/scripts/app.js'),'utf8');
function appFunction(name){
  const start=appSource.indexOf('function '+name+'(');assert.ok(start>=0,name+' exists in app.js');
  const lineEnd=appSource.indexOf('\n',start),firstLine=appSource.slice(start,lineEnd).trim();
  const end=firstLine.endsWith('}')?lineEnd:appSource.indexOf('\n}',lineEnd)+2;
  assert.ok(end>start,name+' has a complete function body');return appSource.slice(start,end);
}
function effectivelyVisible(node){for(let p=node;p;p=p.parentElement)if(p.hidden||p.classList?.contains('hidden'))return false;return true;}
Object.assign(context,{finite:value=>typeof value==='number'&&Number.isFinite(value),labelBackend:value=>value,
  geometrySourceKey:()=>'',gurneyEnabled:()=>$('gurney-enabled').checked,gurneyFlow:()=>({}),streamlineFlow:()=>({}),
  gurneyBackendRefusal:()=>'',streamlineRefusal:()=>'',updateLowspeedSetup(){},updateGurneySetup(){},updateStreamlineSetup(){},
  lowSpeedOverride(){},lowspeedTopologyFlow:()=>({}),lowspeedGpuRefusal:()=>'',
  chartMemory:new Map(),spatialMemory:{geometry:null,field:null},resetLog(){},clearError(){},renderSelection(){},renderPolars(){},renderTable(){},
  displayError:error=>{throw error;},scheduleGeometryPreview(){},loadAirfoils:()=>{throw Error('Unexpected catalog fetch');},
  gurneyRules:{refineLevels:['auto','off','standard','fine']}});
Object.assign(state,{fullMeshes:new Map(),meshLoading:new Map(),fullFields:new Map(),fieldLoading:new Map(),fieldErrors:new Map(),pollGeneration:0,previewGeneration:0,cavitationReferenceSelection:0,serviceRevision:27});
window.AirfoilBenchmarks.restore=async()=>{};
for(const name of ['selectedBackends','selectedCavitationReference','selectedSourceId','portableAirfoil','inputNumber','applyReferenceTripControls','usesCoupledCh','coupledChForControls','updateSetup','updateLowspeedSetup','renderSavedFlowOverrides','restoreSavedFlowExtras','collectCase','previewCase','applySavedRequestToForm'])vm.runInContext(appFunction(name),context,{filename:'app.js:'+name});
for(const line of appSource.split('\n').filter(line=>line.startsWith('window.AirfoilManagedFlowKeys')))vm.runInContext(line,context,{filename:'app.js:managed-keys'});
vm.runInContext(fs.readFileSync(path.join(root,'apps/airfoil_web/static/scripts/studies.js'),'utf8'),context,{filename:'studies.js'});

(async()=>{
  const studies=window.AirfoilStudies,research=window.AirfoilResearchSettings;
  const base={airfoil:{kind:'naca4',id:'2412'},backends:['transonic'],flow:{mach:.3,reynolds:3e6,alpha_deg:4,cfd_mesh:'aplus_uniform_quality',cfd_resolution:'fine',cfd_n_base:6,cfd_steps:5000,cfd_time_mode:'steady',rans_method:'jfnk_native',cfd_implicit_warmup_steps:0,cfd_implicit_cfl:100,cfd_implicit_basis:60,cfd_implicit_globalization:'pseudo_transient',cfd_implicit_linear_maxiter:120,cfd_implicit_max_cfl:1000,cfd_implicit_physical_fraction:.15,cfd_implicit_linear_tolerance:.02,cfd_implicit_preconditioner:'kinetic_block_adi'}};
  let opened,captured;
  context.fetch=async(url,options)=>{
    const value=JSON.parse(options.body);
    if(url.endsWith('/open'))return {ok:true,json:async()=>({path:value.path,etag:'open-tag',study:opened,message:'Opened',results_available:false})};
    if(url.endsWith('/save')){captured=value;return {ok:true,json:async()=>({path:value.path,etag:'saved-tag',study:{results:null},message:'Saved'})};}
    throw Error('Unexpected network request '+url);
  };
  state.job=null;state.result=null;state.preview=null;state.liveField=null;state.embeddedAirfoil=null;
  const controlKeys=['cfd_mesh','cfd_resolution','cfd_n_base','rans_method','cfd_implicit_warmup_steps','cfd_implicit_cfl','cfd_implicit_basis','cfd_implicit_globalization','cfd_implicit_linear_maxiter','cfd_implicit_max_cfl','cfd_implicit_physical_fraction','cfd_implicit_linear_tolerance','cfd_implicit_preconditioner'];
  for(const profile of ['aplus_uniform_quality','aplus_uniform_fixed']){
    base.flow.cfd_mesh=profile;
    // Older controls could not represent the request's research settings.
    opened={request:structuredClone(base),controls:{'cfd-mesh':'aplus_nested_refined','implicit-basis':'16','implicit-globalization':'','implicit-linear-maxiter':''},geometry:{x:[1,0,1],y:[-.001,0,.001]},results:null,view:{}};
    ok(await studies.openPath('C:\\studies\\'+profile+'.ludwig-study'),'Actual study open accepts '+profile);
    ok($('cfd-mesh').value===profile,'Real select retains the explicit profile '+profile);
    ok($('implicit-basis').value==='60','Authoritative saved basis wins over obsolete control snapshot');
    ok($('implicit-preconditioner').value==='kinetic_block_adi','Saved advanced preconditioner has a real option');
    ok(!$('research-settings-note').hidden&&$('research-settings-note').textContent.includes('Research'),'Loaded research settings are visibly labelled');
    workflow.select('solver');
    ok(effectivelyVisible($('implicit-linear-maxiter'))&&effectivelyVisible($('implicit-basis')),'Loaded research controls are visible through every real ancestor');
    ok(effectivelyVisible($('research-settings-note'))&&effectivelyVisible($('cfd-steps')),'Research label and run settings use live visible containers');
    ok($('cfd-te-topology-field').classList.contains('hidden'),'Research profile does not need a duplicate topology choice');
    ok(!$('rans-method').options.find(option=>option.value==='jfnk_native').disabled,'Research profile is recognized as single A+');
    const run=vm.runInContext('collectCase()',context),preview=vm.runInContext('previewCase(true)',context);
    for(const key of controlKeys)ok(run.flow[key]===base.flow[key],'Run preserves '+key+' for '+profile);
    ok(run.flow.cfd_te_topology==='aplus'&&preview.flow.cfd_te_topology==='aplus','Run and preview derive A+ from the profile');
    ok(preview.flow.cfd_resolution===run.flow.cfd_resolution&&preview.flow.cfd_n_base===run.flow.cfd_n_base,'Preview retains the saved profile resolution and base faces');
    ok(!Object.keys(state.studyFlowExtras).some(key=>key.startsWith('cfd_implicit_')),'Represented implicit settings do not become hidden overrides');
    ok(await studies.save(),'Actual study save succeeds for '+profile);
    for(const key of controlKeys)ok(captured.request.flow[key]===base.flow[key],'Save preserves '+key+' for '+profile);
    opened={request:captured.request,controls:captured.controls,geometry:opened.geometry,results:null,view:{}};
    ok(await studies.openPath(captured.path),'Saved request and controls reopen together');
    const repeat=vm.runInContext('collectCase()',context);
    for(const key of controlKeys)ok(repeat.flow[key]===base.flow[key],'Reopened run preserves '+key+' for '+profile);
  }
  $('backend').value='lowspeed';vm.runInContext('updateSetup()',context);
  workflow.select('solver');
  ok(effectivelyVisible($('lowspeed-steady-tol')),'Switching to low-speed reveals its real solver controls');
  ok(!effectivelyVisible($('implicit-linear-maxiter'))&&!effectivelyVisible($('cfd-steps')),'Low-speed hides implicit and compressible run containers through actual ancestors');
  ok($('implicit-globalization').closest('label').hidden,'Low-speed hides compressible Newton options');
  ok($('cfd-mesh').options.filter(option=>research.meshProfile(option.value)).every(option=>option.hidden),'Low-speed hides research compressible mesh options');
  state.studyFlowExtras={cfd_implicit_future_option:3};
  const low=vm.runInContext('collectCase()',context);
  ok(!Object.keys(low.flow).some(key=>key.startsWith('cfd_implicit_')),'Low-speed request drops all inapplicable implicit overrides');
  $('backend').value='transonic';$('cfd-time-mode').value='physical';vm.runInContext('updateSetup()',context);
  ok($('implicit-linear-maxiter').closest('label').hidden,'Physical-time mode hides implicit solver controls');
  const physical=vm.runInContext('collectCase()',context);
  ok(!Object.keys(physical.flow).some(key=>key.startsWith('cfd_implicit_')),'Physical-time request drops implicit overrides');
  $('cfd-time-mode').value='steady';$('rans-method').value='standard';vm.runInContext('updateSetup()',context);
  ok(!effectivelyVisible($('implicit-basis'))&&effectivelyVisible($('cfd-steps')),'Standard compressible mode hides implicit settings and reveals its live run container');
  ok(!Object.keys(vm.runInContext('collectCase()',context).flow).some(key=>key.startsWith('cfd_implicit_')),'Standard method drops implicit overrides');
  $('rans-method').value='line_implicit';vm.runInContext('updateSetup()',context);
  ok($('implicit-physical-fraction').closest('label').hidden&&$('implicit-basis').closest('label').hidden,'Line implicit hides Newton-only controls');
  $('implicit-cfl').value='1';$('implicit-max-cfl').value='';
  const line=vm.runInContext('collectCase()',context);
  ok(line.flow.cfd_implicit_globalization===undefined&&line.flow.cfd_implicit_linear_maxiter===undefined,'Line implicit omits Newton-only research options');
  $('rans-method').value='jfnk_native';research.restore();$('implicit-cfl').value='1';state.studyFlowExtras={};vm.runInContext('updateSetup()',context);
  ok(effectivelyVisible($('implicit-basis'))&&effectivelyVisible($('implicit-linear-maxiter')),'Switching back to Newton updates the real implicit container');
  const automatic=vm.runInContext('collectCase()',context);
  ok(research.keys.every(key=>automatic.flow[key]===undefined),'Blank optional controls retain automatic solver defaults');
  $('implicit-globalization').value='pseudo_transient';$('implicit-max-cfl').value='0.5';
  assert.throws(()=>vm.runInContext('collectCase()',context),/Maximum pseudo CFL/);checks++;
  $('implicit-max-cfl').value='1000';$('implicit-globalization').value='steady';
  assert.throws(()=>vm.runInContext('collectCase()',context),/Maximum pseudo CFL/);checks++;
  opened={request:structuredClone(base),controls:{},geometry:{x:[1,0,1],y:[-.001,0,.001]},results:null,view:{}};
  opened.request.flow.cfd_implicit_preconditioner='private_muscl_prototype';
  opened.request.flow.future_research_option={method:'private'};
  ok(await studies.openPath('C:\\studies\\unavailable.ludwig-study'),'Unknown saved prototype can be inspected');
  ok($('implicit-preconditioner').value==='private_muscl_prototype'&&$('implicit-preconditioner').options.find(option=>option.value==='private_muscl_prototype').disabled,'Unavailable prototype is preserved but not selectable');
  ok($('research-settings-note').textContent.includes('unavailable'),'Unavailable prototype is explained');
  assert.throws(()=>vm.runInContext('collectCase()',context),/not available/);checks++;
  ok(await studies.save()&&captured.request.flow.cfd_implicit_preconditioner==='private_muscl_prototype','Saving preserves the unavailable name without executing it');
  ok(captured.request.flow.future_research_option.method==='private','Unknown saved flow extras survive open/save');
  $('implicit-preconditioner').value='line';
  ok(vm.runInContext('collectCase()',context).flow.cfd_implicit_preconditioner==='line','Choosing a supported preconditioner clears the run block explicitly');
  $('backend').value='cavitation';$('cav-fluid').value='air';$('cav-domain').value='free_fluid';vm.runInContext('updateSetup()',context);
  const species=vm.runInContext('collectCase()',context);
  ok(species.flow.cfd_mesh==='aplus_nested_refined','Switching to Species does not activate a compressible-only research profile');
  ok(!Object.keys(species.flow).some(key=>key.startsWith('cfd_implicit_')),'Species omits compressible implicit options');
  Object.assign(context,{numberText:value=>Number(value).toPrecision(4),refreshPointOverlays(){},syncRunStatusDetails(){},putChart(){},polyline:()=>null,
    colors:{blue:'#00f',transonic:'#f00',jxfoil:'#0f0',cavitation:'#09f'}});
  for(const name of ['implicitFlowQualification','validCfdPolarPoint','lowspeedSteady','isPhysicalPoint','escapeHtml','normalizeIterationHistory','mergeHistoryRows','orderedHistory','renderTable','renderIterationHistory','renderResult','polarSeries'])vm.runInContext(appFunction(name),context,{filename:'app.js:'+name});
  const rawNames=['turb_k_equation_l2','turb_omega_equation_l2','turb_k_equation_linf','turb_omega_equation_linf','turb_omega_wall_constraint_relative_linf'];
  const oldPoint={backend:'transonic',status:'completed',converged:true,alpha_deg:4,cl:.7,cd:.02,cm:-.1,fields:{full_artifact:'/api/jobs/example/files/field.msgpack'},
    iteration_history:[{iteration:100,residual_scaled:1e-8,mass_net_frac:1e-12,cl:.7,cd:.02}],diagnostics:{rans_method:'jfnk_native',time_mode:'steady',physical_sanity:true,
      convergence_guard:{status:'early_stop_eligible',reason:'Original engine guard passed'},steady_assessment:{reason:'All old monitored checks passed'},
      implicit_raw_sst_residuals:{available:true,qualified_stopping_limits:false,checks_passed:false,final_values:Object.fromEntries(rawNames.map(name=>[name,1e-9])),stopping_limits:Object.fromEntries(rawNames.map(name=>[name,1e-7]))}}};
  context.qualificationPoint=oldPoint;state.job={status:'completed'};state.result={results:[oldPoint]};state.liveHistory={};
  ok(!vm.runInContext('validCfdPolarPoint(qualificationPoint)',context),'Older completed guard is not a qualified polar point when raw SST limits are unqualified');
  vm.runInContext('renderTable();renderIterationHistory(qualificationPoint);renderResult()',context);
  ok($('points-body').textContent.includes('Solver stop reached; full flow checks unverified'),'Actual table distinguishes the solver stop from full-flow qualification');
  ok($('points-body').querySelector('.status-tag').classList.contains('warning'),'Unverified equation checks receive a warning badge');
  ok($('convergence-state').textContent.includes('full flow checks unverified'),'Actual history status does not promote an older guard');
  ok(!$('convergence-state').textContent.includes('All old monitored checks passed'),'Old assessment is not presented as a full-flow status');
  ok($('job-title').textContent.includes('full flow checks unverified')&&$('job-message').textContent.includes('Solver stop reached'),'Actual result heading labels an older completed job as unverified, not crashed');
  ok(oldPoint.fields.full_artifact==='/api/jobs/example/files/field.msgpack'&&oldPoint.diagnostics.convergence_guard.status==='early_stop_eligible','Original fields and engine guard remain available');
  const series=vm.runInContext('polarSeries([qualificationPoint],"cl")',context);
  ok(series.length===1&&series[0].name.includes('diagnostic only')&&series[0].connect===false,'Unverified forces use only diagnostic polar markers');
  for(const failure of ['limits_only','explicit_fail','over_limit','missing_constraint','not_finite']){
    const candidate=structuredClone(oldPoint),raw=candidate.diagnostics.implicit_raw_sst_residuals;
    raw.qualified_stopping_limits=true;raw.checks_passed=true;
    if(failure==='limits_only'){delete raw.final_values;delete raw.stopping_limits;}
    if(failure==='explicit_fail')raw.checks_passed=false;
    if(failure==='over_limit')raw.final_values.turb_k_equation_l2=1;
    if(failure==='missing_constraint')delete raw.stopping_limits.turb_omega_wall_constraint_relative_linf;
    if(failure==='not_finite')raw.final_values.turb_omega_equation_l2=NaN;
    context.qualificationPoint=candidate;ok(!vm.runInContext('validCfdPolarPoint(qualificationPoint)',context),'Raw qualification refuses '+failure);
  }
  const complete=structuredClone(oldPoint);complete.diagnostics.implicit_raw_sst_residuals.qualified_stopping_limits=true;complete.diagnostics.implicit_raw_sst_residuals.checks_passed=true;
  context.qualificationPoint=complete;state.result={results:[complete]};
  ok(vm.runInContext('validCfdPolarPoint(qualificationPoint)',context),'Explicit finite raw checks under qualified limits and the old guard can qualify a future point');
  vm.runInContext('renderTable();renderIterationHistory(qualificationPoint)',context);
  ok($('points-body').textContent.includes('Flow checks passed; physics unvalidated'),'Actual table distinguishes equation qualification from physical validation');
  ok(!$('points-body').querySelector('.status-tag').classList.contains('warning'),'Fully qualified synthetic point receives the checked badge');
  ok($('convergence-state').textContent.includes('Flow checks passed; physics unvalidated'),'Actual history reflects the fully qualified synthetic case');
  for(const outcome of ['STEADY','INCOMPLETE']){
    context.qualificationPoint={backend:'lowspeed',status:'completed',cd:.02,diagnostics:{solver_outcome:outcome}};
    ok(vm.runInContext('validCfdPolarPoint(qualificationPoint)',context)===(outcome==='STEADY'),'Low-speed retains its strict native '+outcome+' rule');
  }
  context.Option=function(text,value){const option=document.createElement('option');option.textContent=text;option.value=value;return option;};
  context.CustomEvent=class{constructor(type){this.type=type;}};
  window.dispatchEvent=event=>{for(const callback of callbacks[event.type]||[])callback(event);};
  context.renderPoint=()=>{};
  const presetErrors=[];context.displayError=error=>presetErrors.push(error);
  const presets=[['rae2822-case9','rae2822',.734,6.5e6,2.79,'transonic'],['oat15a-2.5','oat15a',.73,3e6,2.5,'transonic'],['naca0012-stall','naca4',.15,6e6,10,'lowspeed']].map(([id,kind,mach,reynolds,alpha,backend])=>({
    id,kind,label:id,default:true,note:'Published fixture conditions',numerics_note:'Component fixture',conditions_url:'/conditions',
    case:{backends:[backend],flow:{mach,reynolds,alpha_deg:alpha,ncrit:9,trip_upper:.03,trip_lower:.03,cfd_viscosity_model:'constant_kinematic',cfd_freestream_temperature_k:288.15,cfd_time_mode:'steady'}},
    reference:{points:[{cp:{upper:[],lower:[]}}],note:'Fixture reference',source_url:'/source'}}));
  context.fetch=async url=>{ok(url==='/api/airfoil/benchmarks','Preset click uses the bundled benchmark catalog');return {ok:true,json:async()=>({items:presets})};};
  vm.runInContext(fs.readFileSync(path.join(root,'apps/airfoil_web/static/scripts/benchmarks.js'),'utf8'),context,{filename:'benchmarks.js'});
  for(const preset of presets){
    const button=document.querySelector(`[data-validation="${preset.id}"]`);
    ok(button.closest('details')===null,'Validation preset is a visible top-level section action');
    for(const handler of button.events.click||[])await handler({type:'click',target:button,preventDefault(){},stopPropagation(){}});
    ok(presetErrors.length===0,'Actual preset click succeeds without a legacy dropdown close error');
    ok($('source-kind').value===preset.kind&&Number($('reynolds').value)===preset.case.flow.reynolds,'Actual preset click applies its source and Reynolds number');
    ok(Number($('alpha').value)===preset.case.flow.alpha_deg&&window.AirfoilBenchmarks.selectedId===preset.id,'Actual preset click applies its angle and selected reference');
    if(preset.id==='naca0012-stall')ok($('naca-code').value==='0012'&&$('sweep-enabled').checked,'Stall preset still selects NACA0012 and its sweep');
  }
  console.log('PASS '+checks+' browser-component assertions (DOM fixture; actual study open/save/run, not rendered UI)');
})().catch(error=>{console.error(error);process.exitCode=1;});
