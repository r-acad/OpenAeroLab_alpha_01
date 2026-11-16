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
  get open(){return this.hasAttribute('open');}set open(value){value?this.setAttribute('open',''):this.removeAttribute('open');}
  get options(){return all(this).filter(node=>node.tagName==='OPTION');}
  get type(){return this.attrs.type||(this.tagName==='INPUT'?'text':'');}set type(value){this.attrs.type=String(value);}
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
  focus(){document.activeElement=this;}click(){if(!this.disabled)this.dispatchEvent({type:'click'});}
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
// The actual app.js functions are loaded by name; drawing, network and solver execution stay stubbed.
const appSource=fs.readFileSync(path.join(root,'apps/airfoil_web/static/scripts/app.js'),'utf8');
function appFunction(name){
  const start=appSource.indexOf('function '+name+'(');assert.ok(start>=0,name+' exists in app.js');
  const lineEnd=appSource.indexOf('\n',start),firstLine=appSource.slice(start,lineEnd).trim();
  const end=firstLine.endsWith('}')?lineEnd:appSource.indexOf('\n}',lineEnd)+2;
  assert.ok(end>start,name+' has a complete function body');return appSource.slice(start,end);
}
// workflow.js asks app.js which low-speed path applies, so that function exists before the workflow script runs.
context.gurneyEnabled=()=>document.getElementById('gurney-enabled').checked;
for(const name of ['lowspeedUsesCh','newtonMethodSelected'])vm.runInContext(appFunction(name),context,{filename:'app.js:'+name});
// The buttons that end the Mesh and Run steps press the existing buttons: count those presses.
const pressed={};
for(const id of ['create-mesh','run-button'])document.getElementById(id).addEventListener('click',()=>{pressed[id]=(pressed[id]||0)+1;});
for(const file of ['output_controls.js','workflow_help.js','workflow.js'])vm.runInContext(fs.readFileSync(path.join(root,'apps/airfoil_web/static/scripts',file),'utf8'),context,{filename:file});
const $=id=>document.getElementById(id),workflow=window.AirfoilWorkflow;
function refresh(){workflow.refresh();let n=0;while(frames.length){if(n++>20)throw Error('Refresh loop');frames.shift()();}}
function tab(name){return document.querySelector(`[data-workflow-target=${name}]`);}
function tabs(){return document.querySelectorAll('[data-workflow-target]');}
ok(!document.querySelector('.top-menu'),'No dropdown navigation remains');
// The solver is chosen first: its selector stands above the steps, outside every step panel.
ok($('backend').closest('.setup-column')&&!$('backend').closest('[data-airfoil-panel]'),'Solver selector belongs to no step: it stays above all of them');
for(const name of ['model','flow','mesh','solver','outputs','study','validation','help']){
  tab(name).dispatchEvent({type:'click'});
  ok(workflow.selected===name&&effectivelyVisible($('backend')),'Solver selector visible in step '+name);
}
tab('model').dispatchEvent({type:'click'});
ok(tabs().filter(node=>node.classList.contains('workflow-step')).map(node=>node.dataset.workflowTarget).join()==='model,flow,mesh,solver,outputs,result-controls','Numbered steps follow the working order');
ok(tabs().filter(node=>node.classList.contains('workflow-step')).every((node,index)=>node.textContent.trim().startsWith(String(index+1))),'Primary steps show their numbers');
// Rarely needed settings: one Advanced group per step, closed until a loaded setup uses one.
const advancedGroups=document.querySelectorAll('.setup-column details');
ok(advancedGroups.length===4&&advancedGroups.every(node=>node.classList.contains('advanced-group')),'The only disclosure widgets of the setup are the Advanced groups');
ok(advancedGroups.map(node=>node.closest('[data-airfoil-panel]').dataset.airfoilPanel).join()==='flow,mesh,solver,outputs','One Advanced group in each step that has rarely needed settings');
ok(advancedGroups.every(node=>!node.open),'Advanced groups are closed by default');
ok($('display-every').closest('details')===$('advanced-outputs')&&$('output-report').closest('details')===$('advanced-outputs')&&!document.querySelector('.output-makie-options'),'History cadence and reports join the Advanced group of Run and outputs');
ok($('output-profile').closest('details')===null&&$('output-folder').closest('details')===null,'Everyday output choices stay in view');
// Removed controls and options are absent from the page.
for(const id of ['cfd-resolution','refine-mesh','open-saved-airfoil','transport','lbm-wall-model','tab-button-campaign','tab-campaign','setup-method','cfd-mesh','cfd-te-topology','cfd-n-base','lowspeed-topology','implicit-method-options','implicit-basis','saved-flow-overrides','clear-saved-flow-overrides','research-settings-note'])ok(!$(id),'Removed element is absent: '+id);
const values=id=>$(id).options.map(node=>node.value).join();
ok(values('backend')==='lowspeed,transonic,cavitation,lbm,jxfoil,xfoil,both,cfd_both,xfoil_cfd_both,xfoil_lowspeed'&&$('backend').options.every(node=>!node.disabled),'Solver selector offers the runnable solvers only');
ok(values('rans-method')==='standard,irsj,newton','The iteration methods are the standard one, residual smoothing and Implicit Newton');
ok(values('lowspeed-resolution')==='auto,fine'&&values('lowspeed-start')==='auto,projection'&&values('lowspeed-mut-base')==='auto,extrapolated','Low-speed options that equal the automatic choice or are withdrawn are not offered');
ok(values('gurney-refine')==='auto,standard,fine','One flap-tip default');
ok(all(document).filter(node=>node.tagName==='OPTION'||node.tagName==='LABEL').every(node=>!/legacy/i.test(node.textContent)),'No control label or option says legacy');
for(const [id,node] of before)ok($(id)===node,'Existing control/listener node retained: '+id);
for(const [id,node] of Object.entries(el))ok(node===$(id),'Registered control or container refers to the live node: '+id);
ok(document.querySelectorAll('[id]').length===new Set(document.querySelectorAll('[id]').map(node=>node.id)).size,'Runtime IDs remain unique');
ok(tabs().filter(node=>node.classList.contains('workflow-step')).every(node=>!node.hidden),'Every numbered step stays in the row');
ok(!tab('mesh').disabled&&tab('result-controls').disabled&&tab('display').hidden,'Empty case: Mesh is ready, Results waits for data, Display has nothing to show');
tab('result-controls').dispatchEvent({type:'click'});ok(workflow.selected==='model','A step without content cannot be opened');
tab('flow').dispatchEvent({type:'click'});ok(!document.querySelector('[data-airfoil-panel=flow]').hidden,'Flow tab opens directly');
ok(!$('sweep-enabled').closest('section').hidden,'Multiple angles is visible inside Flow');
ok($('lowspeed-tu').closest('[data-airfoil-panel]').dataset.airfoilPanel==='flow','Freestream turbulence grouped with flow');
ok($('lowspeed-resolution').closest('[data-airfoil-panel]').dataset.airfoilPanel==='mesh'&&$('preview-resolution').closest('[data-airfoil-panel]').dataset.airfoilPanel==='mesh','Mesh resolution lives in Mesh only');
ok(!$('lowspeed-resolution').closest('label').hidden&&$('preview-resolution').closest('label').hidden,'Low-speed shows its own resolution, not the compressible one');
ok($('gurney-enabled').closest('[data-airfoil-panel]').dataset.airfoilPanel==='model'&&!$('gurney-group').hidden,'The Gurney flap is part of the Airfoil step for a solver that has it');
ok(['trailing-edge-thickness','gurney-refine','streamline-align'].every(id=>$(id).closest('details')===$('advanced-mesh')),'The Advanced group of Mesh holds the trailing-edge thickness, the flap-tip refinement and the streamline alignment');
// Each primary step ends with its action; the extra buttons press the existing ones.
$('create-mesh-step').click();$('run-step-button').click();
ok(pressed['create-mesh']===1&&pressed['run-button']===1,'Create mesh in Mesh and Run in Run and outputs use the existing handlers');
ok($('create-mesh-step').closest('[data-airfoil-panel]').dataset.airfoilPanel==='mesh'&&$('run-step-button').closest('[data-airfoil-panel]').dataset.airfoilPanel==='outputs','The step actions stand in their steps');
state.preview={geometry:{x:[1,0,1],y:[-.001,0,.001]},mesh:{}};refresh();
ok(!tab('mesh').disabled&&!tab('display').hidden,'Geometry makes Display applicable');
tab('mesh').dispatchEvent({type:'click'});ok(state.activeTab==='geometry','Mesh tab opens its geometry inspection');
el.backend.value='jxfoil';refresh();ok($('gurney-group').hidden,'Panel-only route hides unsupported flap');
ok(tab('mesh').disabled&&$('create-mesh').hidden&&$('create-mesh-step').hidden&&workflow.selected==='model','A panel method has no Mesh step and no Create mesh');
tab('flow').dispatchEvent({type:'click'});document.querySelector('[data-airfoil-panel=flow]').querySelector('[data-workflow-next]').click();
ok(workflow.selected==='solver','Next step skips the step the selected solver does not use');
ok($('lowspeed-tu').closest('[data-airfoil-panel]').dataset.airfoilPanel==='flow'&&$('lowspeed-resolution').closest('[data-airfoil-panel]').dataset.airfoilPanel==='mesh','Low-speed values stay in single locations');
ok($('lowspeed-tu').closest('label').hidden,'Low-speed physics hidden for panel solver');
ok($('trailing-edge-thickness').closest('label').hidden,'Compressible TE override hidden for panel solver');
ok($('advanced-flow').hidden&&$('advanced-mesh').hidden,'An Advanced group with nothing for the selected solver is not shown');
el.backend.value='transonic';refresh();ok(!tab('solver').hidden&&!tab('mesh').disabled&&$('gurney-group').hidden,'Compressible solver controls available, unsupported flap hidden');
ok(!$('preview-resolution').closest('label').hidden&&$('lowspeed-resolution').closest('label').hidden,'Compressible shows the one mesh resolution control');
tab('model').dispatchEvent({type:'click'});document.querySelector('[data-airfoil-panel=model]').querySelector('[data-workflow-next]').click();
document.querySelector('[data-airfoil-panel=flow]').querySelector('[data-workflow-next]').click();
ok(workflow.selected==='mesh','Next step walks the numbered steps in order');
el.backend.value='lowspeed';$('gurney-enabled').checked=true;refresh();
ok(!$('gurney-refine').closest('label').hidden,'Flap mesh refinement appears only for active low-speed flap');
ok([...el.backend.options].find(node=>node.value==='transonic').hidden,'Unsupported solver choice hidden while flap is active');
$('gurney-enabled').checked=false;refresh();ok($('gurney-refine').closest('label').hidden,'Disabled flap hides its refinement');
state.result={results:[{backend:'lowspeed',alpha_deg:4,cl:1,fields:{full_artifact:'/api/jobs/example/fields'},diagnostics:{solver_outcome:'INCOMPLETE'}}]};refresh();
ok(!tab('result-controls').disabled&&!$('tab-button-results').hidden&&!$('tab-button-field').hidden,'Available fields/forces shown');
tab('result-controls').dispatchEvent({type:'click'});ok(workflow.selected==='result-controls'&&$('result-backend').closest('[data-airfoil-panel]').dataset.airfoilPanel==='result-controls','Results opens once there is a result; its selectors stand in that step');
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
ok(tab('result-controls').disabled&&workflow.selected==='model','Results is not available before any real result or export status');
for(const status of ['failed','partial']){
  const warning='Requested export could not be written';
  state.job={exports:{status,files:[],warnings:[warning]}};
  window.LudwigOutputControls.render(state.job);refresh();
  ok(!$('output-results').closest('section').hidden,status+' export status stays visible with zero files');
  ok(!tab('result-controls').disabled,status+' export status makes Results accessible without solver data');
  ok($('output-results').textContent.includes(warning),status+' export warning remains readable');
  ok($('output-results').textContent.includes(status),status+' export status remains readable');
}
state.job={exports:{status:'completed',folder:'C:\\studies\\run\\exports',files:[{name:'field.vtu',url:'/api/jobs/example/files/field.vtu'}]}};
window.LudwigOutputControls.render(state.job);refresh();
ok(!$('output-results').closest('section').hidden,'Actual export files shown');
ok(!document.querySelector('[data-tab-target=campaign]')&&!document.querySelector('.campaign-panel'),'The development dashboard that was never shown is gone');
// The workflow help describes the same steps and the Implicit Newton method.
document.querySelector('[data-airfoil-workflow-help]').dispatchEvent({type:'click'});
const helpText=$('field-help-dialog').textContent;
ok($('field-help-dialog').open&&$('field-help-dialog').querySelectorAll('li').length===6,'Workflow help lists the six numbered steps');
ok(['Airfoil','Flow','Mesh','Solver settings','Run and outputs','Results'].every((name,index)=>$('field-help-dialog').querySelectorAll('li')[index].textContent.startsWith(name+':')),'Workflow help follows the step order');
ok(helpText.includes('First choose the solver')&&helpText.includes('Implicit Newton (CPU)')&&helpText.includes('Marched regular mesh')&&helpText.includes('not validated'),'Workflow help covers the solver choice and the Implicit Newton method');
ok(document.querySelector('[data-airfoil-panel=help]').querySelectorAll('li').length===6&&document.querySelector('[data-airfoil-panel=help]').textContent.includes('Implicit Newton (CPU)'),'Help step shows the short form of the same steps');
// Exercise the actual study open/save implementation and request builder.
// Drawing, native file dialogs and solver execution are intentionally stubbed;
// no implementation of numerical setting serialization is duplicated here.
// Visible: no ancestor hides the node, and it does not sit inside a closed <details> (an Advanced group).
function effectivelyVisible(node){for(let p=node;p;p=p.parentElement)if(p.hidden||p.classList?.contains('hidden')||p!==node&&p.tagName==='DETAILS'&&!p.open)return false;return true;}
Object.assign(context,{finite:value=>typeof value==='number'&&Number.isFinite(value),labelBackend:value=>value,
  geometrySourceKey:()=>'',gurneyEnabled:()=>$('gurney-enabled').checked,gurneyFlow:()=>({}),streamlineFlow:()=>({}),streamlineEnabled:()=>$('streamline-align').checked,
  gurneyBackendRefusal:()=>'',streamlineRefusal:()=>'',updateLowspeedSetup(){},updateGurneySetup(){},updateStreamlineSetup(){},
  lowSpeedOverride(){},
  chartMemory:new Map(),spatialMemory:{geometry:null,field:null},resetLog(){},clearError(){},renderSelection(){},renderPolars(){},renderTable(){},
  displayError:error=>{throw error;},scheduleGeometryPreview(){},loadAirfoils:()=>{throw Error('Unexpected catalog fetch');},
  gurneyRules:{refineLevels:['auto','standard','fine']}});
Object.assign(state,{fullMeshes:new Map(),meshLoading:new Map(),fullFields:new Map(),fieldLoading:new Map(),fieldErrors:new Map(),pollGeneration:0,previewGeneration:0,cavitationReferenceSelection:0,serviceRevision:29,presetFlow:{}});
window.AirfoilBenchmarks.restore=async()=>{};
for(const name of ['selectedBackends','selectedCavitationReference','selectedSourceId','portableAirfoil','inputNumber','applyReferenceTripControls','finiteTrailingEdge','usesCoupledCh','coupledChForControls','lowspeedChFlow','updateSetup','updateLowspeedSetup','collectCase','previewCase','applySavedRequestToForm'])vm.runInContext(appFunction(name),context,{filename:'app.js:'+name});
vm.runInContext(appSource.split('\n').find(line=>line.startsWith('const speciesFiniteEdgeRefusal=')).replace('const ','var '),context,{filename:'app.js:speciesFiniteEdgeRefusal'});
vm.runInContext(fs.readFileSync(path.join(root,'apps/airfoil_web/static/scripts/studies.js'),'utf8'),context,{filename:'studies.js'});

(async()=>{
  const studies=window.AirfoilStudies;
  const removedKeys=flow=>Object.keys(flow).filter(key=>['cfd_mesh','cfd_te_topology','cfd_n_base','cfd_irsj','cfd_sfd_enabled','lowspeed_topology'].includes(key)||key.startsWith('cfd_implicit_'));
  let opened,captured;
  const studyService=async(url,options)=>{
    const value=JSON.parse(options.body);
    if(url.endsWith('/open'))return {ok:true,json:async()=>({path:value.path,etag:'open-tag',study:opened,message:'Opened',results_available:false})};
    if(url.endsWith('/save')){captured=value;return {ok:true,json:async()=>({path:value.path,etag:'saved-tag',study:{results:null},message:'Saved'})};}
    throw Error('Unexpected network request '+url);
  };
  context.fetch=studyService;
  state.job=null;state.result=null;state.preview=null;state.liveField=null;state.embeddedAirfoil=null;
  const closed={x:[1,.5,0,.5,1],y:[0,.05,0,-.05,0]},blunt={x:[1,.5,0,.5,1],y:[.002,.05,0,-.05,-.002]};
  // A study with current settings opens, repeats its request and saves it again.
  const current={airfoil:{kind:'naca4',id:'2412'},backends:['transonic'],flow:{mach:.3,reynolds:3e6,alpha_deg:4,cfd_resolution:'fine',cfd_steps:5000,cfd_time_mode:'steady',rans_method:'irsj',cfd_gpu:true}};
  opened={request:structuredClone(current),controls:{},geometry:blunt,results:null,view:{}};
  ok(await studies.openPath('C:\\studies\\current.ludwig-study'),'A study with current settings opens');
  ok($('rans-method').value==='irsj'&&$('preview-resolution').value==='fine'&&$('cfd-steps').value==='5000','The saved request restores the method, the mesh resolution and the step limit');
  let run=vm.runInContext('collectCase()',context),preview=vm.runInContext('previewCase(true)',context);
  for(const key of ['mach','reynolds','alpha_deg','cfd_resolution','cfd_steps','cfd_time_mode','rans_method','cfd_gpu'])ok(run.flow[key]===current.flow[key],'Run repeats the saved '+key);
  ok(removedKeys(run.flow).length===0&&removedKeys(preview.flow).length===0,'Run and Create mesh send no removed option');
  ok(run.flow.cfd_transition_model==='fully_turbulent'&&run.flow.cfd_viscosity_model==='constant_kinematic','A NACA four-digit section has a finite trailing edge: it runs on C+H, without a topology choice');
  ok(await studies.save()&&captured.request.flow.rans_method==='irsj'&&removedKeys(captured.request.flow).length===0,'Saving stores the same current request');
  // A study saved with options that were removed: the page applies what it still has and sends none of the rest.
  // (The server refuses such a study when it is opened; this is the page's behaviour if one reaches it.)
  opened={request:{airfoil:{kind:'naca4',id:'2412'},backends:['transonic'],flow:{mach:.3,reynolds:3e6,alpha_deg:4,cfd_mesh:'aplus_uniform_quality',cfd_te_topology:'aplus',cfd_n_base:6,cfd_resolution:'fine',
      cfd_steps:5000,cfd_time_mode:'steady',rans_method:'jfnk_native',cfd_implicit_basis:60,cfd_implicit_preconditioner:'kinetic_block_adi',cfd_gpu:true}},
    controls:{'cfd-mesh':'aplus_uniform_quality','cfd-te-topology':'aplus','cfd-n-base':'6','implicit-basis':'60','rans-method':'jfnk_native'},geometry:blunt,results:null,view:{}};
  ok(await studies.openPath('C:\\studies\\removed_options.ludwig-study'),'A study saved with removed options does not break the page');
  ok($('rans-method').value==='standard'&&values('rans-method')==='standard,irsj,newton','A removed iteration method leaves the selector at its default and adds no option');
  ok(!all(document).some(node=>node.tagName==='OPTION'&&(node.dataset.savedSetting||/^Saved setting/.test(node.textContent))),'No option is created for a saved value that was removed');
  run=vm.runInContext('collectCase()',context);preview=vm.runInContext('previewCase(true)',context);
  ok(removedKeys(run.flow).length===0&&removedKeys(preview.flow).length===0&&run.flow.rans_method==='standard','The rebuilt request carries no removed option');
  ok(run.flow.cfd_resolution==='fine'&&run.flow.cfd_steps===5000,'Settings that still exist are restored');
  ok(await studies.save()&&removedKeys(captured.request.flow).length===0,'Saving the study again stores a current request');
  workflow.select('solver');workflow.revealAdvanced({collapse:true});
  // Low-speed: its solver settings, and an Advanced group that opens for a non-default value.
  $('backend').value='lowspeed';vm.runInContext('updateSetup()',context);
  workflow.select('solver');workflow.revealAdvanced({collapse:true});
  ok(effectivelyVisible($('lowspeed-gpu'))&&!effectivelyVisible($('cfd-steps')),'Switching to low-speed shows its solver controls and hides the compressible ones');
  ok(!$('advanced-solver').hidden&&!$('advanced-solver').open&&!effectivelyVisible($('lowspeed-steady-tol'))&&!$('lowspeed-steady-tol').closest('label').hidden,'Steady limits wait in the closed Advanced group');
  $('lowspeed-steady-tol').value='1e-5';workflow.revealAdvanced();
  ok($('advanced-solver').open&&effectivelyVisible($('lowspeed-steady-tol')),'A non-default value in an Advanced group opens it');
  $('lowspeed-steady-tol').value='1e-6';
  ok(removedKeys(vm.runInContext('collectCase()',context).flow).length===0,'A low-speed request names no topology and no removed option');
  // Compressible: physical time has no steady iteration method.
  $('backend').value='transonic';$('source-kind').value='rae2822';$('cfd-time-mode').value='physical';vm.runInContext('updateSetup()',context);
  ok($('rans-method-line').classList.contains('hidden')&&vm.runInContext('collectCase()',context).flow.rans_method===undefined,'Physical time shows and sends no steady iteration method');
  $('cfd-time-mode').value='steady';$('rans-method').value='standard';vm.runInContext('updateSetup()',context);
  ok(!$('rans-method-line').classList.contains('hidden')&&effectivelyVisible($('cfd-steps')),'Steady compressible mode shows the method and its run settings');
  // Species on a finite trailing edge: the free-fluid domain runs on the C+H pair in physical time. Steady air and air in
  // the tunnel have one C block and are refused before anything is sent, each with its own message.
  $('source-kind').value='naca4';$('naca-code').value='0012';$('backend').value='cavitation';$('cav-fluid').value='air';$('cav-domain').value='free_fluid';$('cav-air-mode').value='steady';$('mach').value='0.2';vm.runInContext('updateSetup()',context);
  assert.throws(()=>vm.runInContext('collectCase()',context),/finite trailing edge \(C\+H mesh\) runs in physical time/);checks++;
  $('cav-air-mode').value='physical';vm.runInContext('updateSetup()',context);
  const finiteAir=vm.runInContext('collectCase()',context);
  ok(finiteAir.flow.cav_fluid==='air'&&finiteAir.flow.cav_domain==='free_fluid'&&finiteAir.flow.cav_air_mode==='physical'&&removedKeys(finiteAir.flow).length===0,'Species air in a free stream on a finite trailing edge is sent in physical time (C+H)');
  $('cav-domain').value='tunnel';vm.runInContext('updateSetup()',context);
  assert.throws(()=>vm.runInContext('collectCase()',context),/Species air in the tunnel runs on one C block/);checks++;
  $('cav-fluid').value='water';$('cav-domain').value='free_fluid';vm.runInContext('updateSetup()',context);
  const finiteWater=vm.runInContext('collectCase()',context);
  ok(finiteWater.flow.cav_fluid==='water'&&finiteWater.flow.cav_domain==='free_fluid'&&finiteWater.flow.sigma>0&&removedKeys(finiteWater.flow).length===0,'Free-fluid water on a finite trailing edge is sent (C+H)');
  $('cav-domain').value='tunnel';vm.runInContext('updateSetup()',context);
  ok(removedKeys(vm.runInContext('collectCase()',context).flow).length===0,'Water in the tunnel keeps a finite trailing edge (C+H) and sends no removed option');
  $('source-kind').value='rae2822';$('cav-fluid').value='air';$('cav-domain').value='free_fluid';$('cav-air-mode').value='steady';vm.runInContext('updateSetup()',context);
  const species=vm.runInContext('collectCase()',context);
  ok(species.flow.cav_fluid==='air'&&removedKeys(species.flow).length===0,'Species air on a sharp-edged section is sent, without a mesh profile');
  // Result rows: the solver's own status decides whether a point is a polar point.
  Object.assign(context,{numberText:value=>Number(value).toPrecision(4),refreshPointOverlays(){},syncRunStatusDetails(){},putChart(){},polyline:()=>null,
    colors:{blue:'#00f',transonic:'#f00',jxfoil:'#0f0',cavitation:'#09f'}});
  for(const name of ['validCfdPolarPoint','lowspeedSteady','isPhysicalPoint'])vm.runInContext(appFunction(name),context,{filename:'app.js:'+name});
  for(const outcome of ['STEADY','INCOMPLETE']){
    context.polarPoint={backend:'lowspeed',status:'completed',cd:.02,diagnostics:{solver_outcome:outcome}};
    ok(vm.runInContext('validCfdPolarPoint(polarPoint)',context)===(outcome==='STEADY'),'Low-speed keeps its strict native '+outcome+' rule');
  }
  const newtonPoint={backend:'transonic',status:'completed',alpha_deg:2,cl:.3,cd:.01,diagnostics:{rans_method:'newton',time_mode:'steady',physical_sanity:true,convergence_guard:{status:'early_stop_eligible'}}};
  context.polarPoint=newtonPoint;
  ok(vm.runInContext('validCfdPolarPoint(polarPoint)',context),'A settled Implicit Newton point is a polar point');
  newtonPoint.status='not_converged';newtonPoint.diagnostics.convergence_guard.status='loads_not_settled';
  ok(!vm.runInContext('validCfdPolarPoint(polarPoint)',context),'An unsettled Implicit Newton point is not a polar point');
  // Validation presets apply their case: RAE 2822 on one C block, ONERA OAT15A on C+H with its trips, viscosity law and
  // (the unsteady angles) physical time, NACA 0012 in the low-speed solver.
  context.Option=function(text,value){const option=document.createElement('option');option.textContent=text;option.value=value;return option;};
  context.CustomEvent=class{constructor(type){this.type=type;}};
  window.dispatchEvent=event=>{for(const callback of callbacks[event.type]||[])callback(event);};
  context.renderPoint=()=>{};
  const presetErrors=[];context.displayError=error=>presetErrors.push(error);
  const presets=[['rae2822-case9','rae2822',.734,6.5e6,2.79,'transonic'],['oat15a-2.5','oat15a',.73,3e6,2.5,'transonic'],['oat15a-3.5','oat15a',.73,3e6,3.5,'transonic'],['naca0012-stall','naca4',.15,6e6,10,'lowspeed']].map(([id,kind,mach,reynolds,alpha,backend])=>{
    const oat=kind==='oat15a',unsteady=oat&&alpha>=3.25,trip=oat?.07:.03;
    return {id,kind,label:id,default:id!=='oat15a-3.5',unsteady,note:'Published fixture conditions',numerics_note:'Component fixture',conditions_url:'/conditions',
      case:{backends:[backend],flow:{mach,reynolds,alpha_deg:alpha,ncrit:9,trip_upper:trip,trip_lower:trip,cfd_viscosity_model:oat?'sutherland':'constant_kinematic',cfd_freestream_temperature_k:oat?271:288.15,
        cfd_time_mode:unsteady?'physical':'steady',...(unsteady?{cfd_duration_ctu:100,cfd_burnin_ctu:20}:{}),lowspeed_max_steps:12000,lowspeed_wall_cap_s:7200}},
      reference:{points:[{cp:{upper:[],lower:[]}}],note:'Fixture reference',source_url:'/source'}};});
  context.fetch=async url=>{ok(url==='/api/airfoil/benchmarks','Preset click uses the bundled benchmark catalog');return {ok:true,json:async()=>({items:presets})};};
  vm.runInContext(fs.readFileSync(path.join(root,'apps/airfoil_web/static/scripts/benchmarks.js'),'utf8'),context,{filename:'benchmarks.js'});
  ok(document.querySelectorAll('[data-validation]').map(node=>node.dataset.validation).join()==='rae2822-case9,oat15a-2.5,naca0012-stall','Validation offers the RAE 2822, ONERA OAT15A and NACA 0012 presets');
  for(const preset of presets){
    // The unsteady OAT15A angles are chosen in the published-case list, not by a Validation button.
    const button=document.querySelector(`[data-validation="${preset.id}"]`);
    if(button){
      ok(button.closest('details')===null,'Validation preset is a visible top-level section action');
      for(const handler of button.events.click||[])await handler({type:'click',target:button,preventDefault(){},stopPropagation(){}});
    }else{await window.AirfoilBenchmarks.catalog();window.AirfoilBenchmarks.apply(preset);}
    ok(presetErrors.length===0,'Preset click succeeds');
    ok($('source-kind').value===preset.kind&&Number($('reynolds').value)===preset.case.flow.reynolds,'Preset click applies its source and Reynolds number');
    ok(Number($('alpha').value)===preset.case.flow.alpha_deg&&window.AirfoilBenchmarks.selectedId===preset.id,'Preset click applies its angle and selected reference');
    const applied=vm.runInContext('collectCase()',context);
    ok(removedKeys(applied.flow).length===0,'The applied preset sends no removed option');
    if(preset.id==='rae2822-case9')ok(applied.flow.trip_upper===.03&&applied.flow.cfd_transition_model===undefined&&applied.flow.rans_method==='standard','RAE 2822 has a closed trailing edge: one C block with the prescribed trips');
    if(preset.kind==='oat15a'){
      ok(applied.flow.trip_upper===.07&&applied.flow.trip_lower===.07&&applied.flow.cfd_transition_model===undefined,'OAT15A has a finite trailing edge: C+H with the prescribed trips, not the fully turbulent model');
      ok(applied.flow.cfd_viscosity_model==='sutherland'&&applied.flow.cfd_freestream_temperature_k===271,'OAT15A on C+H keeps its Sutherland law and temperature');
      ok(!$('cfd-viscosity-model').disabled&&!$('cfd-time-mode').disabled&&!$('transition-mode').disabled,'C+H leaves the viscosity, time and transition controls usable');
      ok(applied.flow.cfd_time_mode===(preset.unsteady?'physical':'steady')&&window.AirfoilBenchmarks.isMatch(preset,applied),'The applied OAT15A request matches its published case, time mode included');
      if(preset.unsteady)ok(applied.flow.cfd_duration_ctu===100&&applied.flow.cfd_burnin_ctu===20&&applied.flow.cfd_auto_stop===false&&applied.flow.cfd_steps===2000000&&applied.flow.rans_method===undefined,
        'An unsteady OAT15A angle runs in physical time: its duration and burn-in, no smart stop, no steady iteration method');
    }
    if(preset.id==='naca0012-stall')ok($('naca-code').value==='0012'&&$('sweep-enabled').checked&&applied.flow.lowspeed_max_steps===12000&&applied.flow.lowspeed_wall_cap_s===7200,'Stall preset selects NACA 0012, its sweep and its step and time limits');
  }
  // Implicit Newton (CPU): the request names the method; the controls that do not apply are hidden and not sent.
  // NACA 0012 has a finite trailing edge: without the method this section uses the coupled C+H driver.
  $('backend').value='transonic';$('cfd-time-mode').value='steady';
  $('mach').value='0.5';$('cfd-steps').value='300';$('cfd-gpu').checked=false;state.presetFlow={};
  $('rans-method').value='newton';vm.runInContext('updateSetup()',context);$('cfd-viscosity-model').value='sutherland';vm.runInContext('updateSetup()',context);refresh();
  ok($('cfd-gpu').closest('label').classList.contains('hidden')&&$('cfd-sfd-field').classList.contains('hidden')&&$('cfd-rk-field').classList.contains('hidden'),'Implicit Newton hides the GPU switch, SFD precision and RK weights');
  ok($('streamline-align').closest('label').hidden,'Streamline alignment is not offered with Implicit Newton');
  ok(!$('cfd-viscosity-model').disabled&&!$('cfd-time-mode').disabled&&$('cfd-viscosity-model').value==='sutherland','Implicit Newton is not handled as the coupled C+H driver');
  $('streamline-align').checked=true;
  const newtonRun=vm.runInContext('collectCase()',context),newtonPreview=vm.runInContext('previewCase(true)',context);
  ok(newtonRun.flow.rans_method==='newton'&&newtonRun.flow.cfd_gpu===false&&removedKeys(newtonRun.flow).length===0,'Implicit Newton run names the method, on the CPU, without a mesh profile');
  ok(newtonRun.flow.cfd_viscosity_model==='sutherland'&&newtonRun.flow.cfd_steps===300&&newtonRun.flow.cfd_sfd_storage===undefined,'The CPU viscosity check does not block Implicit Newton; inapplicable keys are not sent');
  ok(newtonRun.flow.cfd_streamline_align===undefined&&newtonPreview.flow.cfd_streamline_align===undefined,'A ticked alignment is not sent with Implicit Newton, which refuses it');
  ok(newtonPreview.flow.rans_method==='newton'&&removedKeys(newtonPreview.flow).length===0,'Create mesh requests the mesh of the Implicit Newton method by the method name');
  $('streamline-align').checked=false;
  $('rans-method').value='standard';vm.runInContext('updateSetup()',context);refresh();
  ok(!$('cfd-gpu').closest('label').classList.contains('hidden')&&!$('streamline-align').closest('label').hidden,'Another method shows the GPU switch and the alignment again');
  // A saved Implicit Newton request restores its method and step limit; its cfd_gpu=false is not the user's switch.
  $('cfd-gpu').checked=true;$('cfd-steps').value='500000';
  context.savedNewton={airfoil:{kind:'naca4',id:'0012'},backends:['transonic'],flow:{mach:.5,reynolds:3e6,alpha_deg:2,rans_method:'newton',cfd_gpu:false,cfd_steps:300,cfd_time_mode:'steady',cfd_resolution:'standard',cfd_viscosity_model:'sutherland',cfd_freestream_temperature_k:288.15}};
  ok(vm.runInContext('applySavedRequestToForm(savedNewton)',context),'A saved Implicit Newton request is applied to the form');
  ok($('rans-method').value==='newton'&&$('cfd-steps').value==='300','The saved request restores the method and the step limit');
  ok($('cfd-gpu').checked===true,'The fixed CPU setting of the method does not switch the GPU control off');
  const reopenedNewton=vm.runInContext('collectCase()',context);
  ok(reopenedNewton.flow.rans_method==='newton'&&reopenedNewton.flow.cfd_gpu===false&&reopenedNewton.flow.cfd_steps===300,'The restored form requests the same method again');
  // ---- The actual Gurney, low-speed and override code builds the requests below.
  vm.runInContext(appSource.split('\n').find(line=>line.startsWith('const gurneyRules=')).replace('const ','var '),context,{filename:'app.js:gurneyRules'});
  for(const name of ['gurneyFlow','lowSpeedOverride'])vm.runInContext(appFunction(name),context,{filename:'app.js:'+name});
  const build=()=>{vm.runInContext('updateSetup()',context);refresh();return {run:vm.runInContext('collectCase()',context),preview:vm.runInContext('previewCase(true)',context)};};
  // The preset clicks above marked the study as changed: save it, so that opening another study asks nothing.
  context.fetch=studyService;context.displayError=error=>{throw error;};
  ok(await studies.save()&&!studies.dirty,'The edited study saves before another one is opened');
  // A low-speed study saved with the coarse mesh and an explicit topology, both removed: the offered values are used.
  opened={request:{airfoil:{kind:'naca4',id:'0012'},backends:['lowspeed'],flow:{reynolds:3e6,alpha_deg:4,lowspeed_resolution:'demo',lowspeed_topology:'ch_native',lowspeed_start:'projection',lowspeed_turb_mut_base:'extrapolated',gurney_height:.02,gurney_mode:'lower',gurney_refine:'fine'}},
    controls:{'lowspeed-resolution':'demo','lowspeed-topology':'ch_native'},geometry:blunt,results:null,view:{workflow_panel:'flap'}};
  ok(await studies.openPath('C:\\studies\\removed_lowspeed.ludwig-study'),'A low-speed study saved with removed options opens on the page');
  while(frames.length)frames.shift()();
  ok(workflow.selected==='model','A study saved on the former Flap tab opens on Airfoil, where the flap now is');
  ok($('lowspeed-resolution').value==='auto'&&values('lowspeed-resolution')==='auto,fine','The removed coarse mesh leaves the resolution at its default and adds no option');
  ok($('lowspeed-start').value==='projection'&&$('lowspeed-mut-base').value==='extrapolated'&&$('gurney-refine').value==='fine'&&$('gurney-enabled').checked,'Saved values that are still offered are restored');
  ok($('advanced-mesh').open&&$('advanced-solver').open,'The restored non-default settings open their Advanced groups');
  let rebuilt=build();
  ok(rebuilt.run.flow.lowspeed_resolution===undefined&&rebuilt.run.flow.lowspeed_topology===undefined&&rebuilt.preview.flow.lowspeed_topology===undefined,'The rebuilt low-speed request names neither the removed mesh nor a topology');
  ok(rebuilt.run.flow.gurney_height===.02&&rebuilt.run.flow.gurney_refine==='fine'&&rebuilt.run.flow.lowspeed_start==='projection'&&rebuilt.run.flow.lowspeed_turb_mut_base==='extrapolated','Flap and C+H settings of the reopened study are sent again');
  ok(await studies.newStudy()===undefined&&$('lowspeed-resolution').value==='auto'&&$('rans-method').value==='standard'&&advancedGroups.every(node=>!node.open),'A new study returns to the defaults with every Advanced group closed');
  // Low-speed: the C+H start and base settings are not sent for a sharp trailing edge (the sharp path refuses them).
  $('source-kind').value='naca4';$('naca-code').value='0012';$('backend').value='lowspeed';$('lowspeed-start').value='projection';$('lowspeed-mut-base').value='extrapolated';
  state.preview={geometry:closed};rebuilt=build();
  ok(rebuilt.run.flow.lowspeed_start===undefined&&rebuilt.run.flow.lowspeed_turb_mut_base===undefined&&rebuilt.preview.flow.lowspeed_start===undefined&&rebuilt.preview.flow.lowspeed_turb_mut_base===undefined,'Stale C+H keys are not sent for a sharp section');
  ok($('lowspeed-start').closest('label').hidden&&$('lowspeed-mut-base').closest('label').hidden&&$('lowspeed-resolution').options.find(option=>option.value==='fine').hidden,'A sharp section shows neither the C+H settings nor the Fine mesh');
  state.preview={geometry:blunt};rebuilt=build();
  ok(rebuilt.run.flow.lowspeed_start==='projection'&&rebuilt.run.flow.lowspeed_turb_mut_base==='extrapolated'&&rebuilt.preview.flow.lowspeed_start==='projection','The same settings are sent for a blunt section, which runs on C+H');
  ok(!$('lowspeed-start').closest('label').hidden&&!$('lowspeed-resolution').options.find(option=>option.value==='fine').hidden,'A blunt section shows the C+H settings and the Fine mesh');
  ok(rebuilt.run.flow.ncrit===undefined&&rebuilt.run.flow.display_every===undefined&&rebuilt.run.display_every===100,'Low-speed sends no Ncrit; the history cadence travels once, at the top level');
  $('lowspeed-start').value='auto';$('lowspeed-mut-base').value='auto';state.preview=null;
  // One mesh resolution control, used by Create mesh and by Run.
  $('backend').value='transonic';$('mach').value='0.5';$('rans-method').value='standard';$('preview-resolution').value='very_fine';rebuilt=build();
  ok(rebuilt.run.flow.cfd_resolution==='very_fine'&&rebuilt.preview.flow.cfd_resolution==='very_fine','Compressible Run and Create mesh use the one resolution control');
  $('transition-mode').value='forced';$('trip-upper').value='0.1';$('trip-lower').value='0.2';rebuilt=build();
  ok(rebuilt.run.flow.cfd_transition_model===undefined&&rebuilt.run.flow.trip_upper===.1&&rebuilt.run.flow.trip_lower===.2,'C+H with prescribed trips sends them, per side, without the fully turbulent key');
  $('transition-mode').value='default';rebuilt=build();
  ok(rebuilt.run.flow.ncrit===undefined&&rebuilt.run.flow.cfd_transition_model==='fully_turbulent'&&rebuilt.run.flow.trip_upper===undefined&&removedKeys(rebuilt.run.flow).length===0&&removedKeys(rebuilt.preview.flow).length===0,'C+H: fully turbulent by default, no Ncrit, no removed option');
  $('backend').value='both';rebuilt=build();
  ok(rebuilt.run.flow.ncrit===9,'Ncrit is sent when a panel method runs');
  $('backend').value='cavitation';$('cav-fluid').value='water';$('cav-domain').value='tunnel';$('mach').value='0.1';rebuilt=build();
  ok(rebuilt.run.flow.cfd_resolution==='very_fine'&&rebuilt.preview.flow.cfd_resolution==='very_fine'&&removedKeys(rebuilt.preview.flow).length===0,'The water tunnel runs the resolution its Create mesh shows');
  $('preview-resolution').value='standard';
  // Implicit Newton: the scaled mass ceiling is hidden and the explicit-march settings are not sent.
  $('backend').value='transonic';$('mach').value='0.5';$('rans-method').value='newton';$('cfd-steps').value='300';rebuilt=build();
  ok($('cfd-mass-ceiling-field').classList.contains('hidden')&&rebuilt.run.flow.cfd_mass_ceiling===undefined&&rebuilt.run.flow.cfd_auto_stop===undefined&&rebuilt.run.flow.cfd_rk_weight_closure===undefined,'Implicit Newton hides the scaled mass ceiling and sends no explicit-march setting');
  $('rans-method').value='standard';$('cfd-steps').value='500000';rebuilt=build();
  ok(!$('cfd-mass-ceiling-field').classList.contains('hidden')&&rebuilt.run.flow.cfd_mass_ceiling===.1&&rebuilt.run.flow.cfd_auto_stop===true,'The standard method shows and sends them again');
  // The marched mesh report states what was measured: the geometric check and each comparative limit it lies outside.
  for(const line of appSource.split('\n').filter(line=>line.startsWith('const comparativeLimitNames=')))vm.runInContext(appSource.slice(appSource.indexOf(line),appSource.indexOf('};',appSource.indexOf(line))+2).replace('const ','var '),context,{filename:'app.js:comparativeLimitNames'});
  for(const name of ['meshBlocks','streamlineMeshText','marchedMeshReport','marchedMeshText','meshTopologyText','meshQualityText'])vm.runInContext(appFunction(name),context,{filename:'app.js:'+name});
  context.marched={kind:'structured_blocks',mesh_topology:'marched_C_H',quality_passed:false,quality_failures:['G2a theta_max=71.25 > 65.0','G2i minang=17.5 < 20.0'],solver_supported:true,blocks:[{},{}],
    quality:{geometric_gate:{passed:true,criteria:'positive cell areas in both blocks, checked by the mesher'},air_reference_gates:{passed:false,failed:['G2a theta_max=71.25 > 65.0','G2i minang=17.5 < 20.0'],applicability:'comparative CMeshQualityV1 air thresholds of the app\'s C+H mesh; the marched mesh is built by another rule'}}};
  const marchedText=vm.runInContext('meshTopologyText(marched)',context);
  ok(marchedText.includes('geometric check passed (positive cell areas in both blocks, checked by the mesher)')&&!marchedText.includes('FAILED'),'Marched mesh: the passed geometric check is stated, nothing is called failed');
  ok(marchedText.includes('outside 2 of the comparative limits'),'Marched mesh: each comparative limit it lies outside is counted');
  context.marchedSingle={kind:'structured',ni:3,nj:3,quality:{near_foil:{},geometric_gate:{passed:true,criteria:'strictly positive corner Jacobians, convex cells, finite metrics and positive area'},air_reference_gates:{passed:true,failed:[],applicability:'comparative CMeshQualityV1 air thresholds, as reported for the app\'s nested-refinement profile; the marched mesh is built by another rule'}}};
  ok(vm.runInContext('meshTopologyText(marchedSingle)',context).includes('convex cells')&&vm.runInContext('meshTopologyText(marchedSingle)',context).includes('inside all comparative limits'),'Marched single block: the geometric check and the comparative limits are both reported');
  context.ordinary={kind:'structured_blocks',mesh_topology:'conformal_C_H_v1',quality_passed:false,quality_failures:['G2a theta_max=71.25 > 65.0'],blocks:[{},{}]};
  ok(vm.runInContext('meshTopologyText(ordinary)',context).includes('air geometry checks FAILED'),'The limits of the app\'s own C+H mesh keep their verdict: nothing is relaxed');
  console.log('PASS '+checks+' browser-component assertions (DOM fixture; actual study open/save/run, not rendered UI)');
})().catch(error=>{console.error(error);process.exitCode=1;});
