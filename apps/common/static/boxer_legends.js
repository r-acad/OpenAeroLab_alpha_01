/* Adapted from JFEM-Boxer (AGPL-3.0); see THIRD_PARTY_NOTICES.md. */
(function(root,factory){const api=factory(root);if(typeof module==="object"&&module.exports)module.exports=api;else root.WingLegends=api;})(typeof globalThis!=="undefined"?globalThis:this,function(root){
  "use strict";
  const escape=(value)=>String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const number=value=>root.WingNumbers?.format(value,5)??(root.eng?root.eng(value,5):Number.isFinite(value)?String(Number(value.toPrecision(5))):String(value));
  const hosts=new Map(),collapsed=new Set(),limits=new Map(),palettes=new Map();
  const defaultPalettes=[{value:"spectrum",label:"Blue–green–red"},{value:"viridis",label:"Viridis"},
    {value:"inferno",label:"Inferno"},{value:"coolwarm",label:"Blue–white–red"},{value:"grayscale",label:"Grayscale"},{value:"signed",label:"Signed red (-) / blue (+)"}];
  const paletteIcon='<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 2a8 8 0 1 0 0 16h1.2a1.9 1.9 0 0 0 1.3-3.3 1.2 1.2 0 0 1 .8-2.1H15a3 3 0 0 0 3-3C18 5.5 14.4 2 10 2Z"/><circle cx="5.5" cy="9" r=".8"/><circle cx="7.5" cy="5.8" r=".8"/><circle cx="11.1" cy="5.2" r=".8"/><circle cx="14.2" cy="7.3" r=".8"/></svg>';
  const restoreIcon='<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M4 5h12M4 10h12M4 15h12"/><path d="M5 3v4M10 8v4M15 13v4"/></svg>';
  let config={getPalette:scope=>scope==="sensitivity"?"coolwarm":"spectrum",setPalette:null,onChange:null,onLimitsChange:null,palettes:defaultPalettes},openKey=null,openLimits=null,hostSerial=0;
  const stateKey=(host,kind)=>host+":"+String(kind);
  const keyPattern=/^[a-zA-Z0-9_-]{1,80}:[a-zA-Z0-9_-]{1,40}$/;
  function validateState(value){
    if(value==null)return{collapsed:{}};
    if(typeof value!=="object"||Array.isArray(value)||Object.keys(value).some(key=>!["collapsed","limits","palettes"].includes(key))||
      !value.collapsed||typeof value.collapsed!=="object"||Array.isArray(value.collapsed)||Object.keys(value.collapsed).length>64)
      throw Error("Invalid color-scale visibility settings.");
    const clean={};
    for(const [key,hidden] of Object.entries(value.collapsed)){
      if(!keyPattern.test(key)||typeof hidden!=="boolean")throw Error("Invalid color-scale visibility setting: "+key);
      if(hidden)clean[key]=true;
    }
    const ranges={};
    if(value.limits!==undefined){
      if(!value.limits||typeof value.limits!=="object"||Array.isArray(value.limits)||Object.keys(value.limits).length>256)throw Error("Invalid color-scale limits.");
      for(const [key,range]of Object.entries(value.limits)){
        if(!validLimitKey(key)||!Array.isArray(range)||range.length!==2||!range.every(Number.isFinite)||!(range[0]<range[1])||!Number.isFinite(range[1]-range[0]))throw Error("Invalid color-scale limits: "+key);
        Object.defineProperty(ranges,key,{value:range.slice(),enumerable:true});
      }
    }
    const colors={};
    if(value.palettes!==undefined){
      if(!value.palettes||typeof value.palettes!=="object"||Array.isArray(value.palettes)||Object.keys(value.palettes).length>32)throw Error("Invalid result color palettes.");
      for(const [scope,palette]of Object.entries(value.palettes)){
        if(!validPaletteScope(scope)||!defaultPalettes.some(p=>p.value===palette))throw Error("Invalid result color palette: "+scope);
        Object.defineProperty(colors,scope,{value:palette,enumerable:true});
      }
    }
    return{collapsed:clean,...(Object.keys(ranges).length?{limits:ranges}:{}),...(Object.keys(colors).length?{palettes:colors}:{})};
  }
  function validPaletteScope(scope){return typeof scope==="string"&&/^[a-z][a-z0-9_-]{0,39}$/.test(scope)&&!["__proto__","constructor","prototype"].includes(scope);}
  function getPalette(scope="fe"){return palettes.get(scope)||config.getPalette?.(scope)||"spectrum";}
  function setPalette(scope,value){
    if(!validPaletteScope(scope)||!defaultPalettes.some(p=>p.value===value))throw Error("Invalid result color palette.");
    if(getPalette(scope)===value)return false;
    if(!palettes.has(scope)&&palettes.size>=32)throw Error("Too many result color palettes.");
    palettes.set(scope,value);config.setPalette?.(value,scope);refresh();config.onChange?.(capture());return true;
  }
  function validLimitKey(key){return typeof key==="string"&&key.length>0&&key.length<=512&&!/[\x00-\x1f]/.test(key)&&!["__proto__","constructor","prototype"].includes(key);}
  function getLimits(key){return limits.has(key)?limits.get(key).slice():null;}
  function resolveLimits(key,min,max){const range=getLimits(key);return range?{min:range[0],max:range[1],manual:true}:{min,max,manual:false};}
  function setLimits(key,min,max,{keepOpen=false}={}){
    if(!validLimitKey(key))throw Error("Invalid color-scale quantity.");
    if(min===null&&max===null)limits.delete(key);
    else{if(!Number.isFinite(min)||!Number.isFinite(max)||!(min<max)||!Number.isFinite(max-min))throw Error("Enter finite limits with minimum less than maximum and a finite range.");
      if(!limits.has(key)&&limits.size>=256)throw Error("Too many saved color-scale ranges; reset unused ranges to Automatic.");limits.set(key,[min,max]);}
    if(!keepOpen)openLimits=null;config.onLimitsChange?.(key);refresh();config.onChange?.(capture());
  }
  function capture(){return{collapsed:Object.fromEntries([...collapsed].sort().map(key=>[key,true])),...(limits.size?{limits:Object.fromEntries([...limits].sort().map(([key,range])=>[key,range.slice()]))}:{}),...(palettes.size?{palettes:Object.fromEntries([...palettes].sort())}:{})};}
  function restore(value,{legacyPalette}={}){
    const clean=validateState(value);collapsed.clear();for(const key of Object.keys(clean.collapsed))collapsed.add(key);
    limits.clear();for(const [key,range]of Object.entries(clean.limits||{}))limits.set(key,range.slice());
    palettes.clear();for(const [scope,palette]of Object.entries(clean.palettes||{}))palettes.set(scope,palette);
    // Older Studies had one shared control; preserve their FE and VLM appearance.
    if(value?.palettes===undefined&&defaultPalettes.some(p=>p.value===legacyPalette)&&legacyPalette!=="spectrum"){
      palettes.set("fe",legacyPalette);palettes.set("vlm",legacyPalette);
    }
    openKey=null;openLimits=null;refresh();return capture();
  }
  function configure(options={}){
    config={...config,...options};
    config.palettes=(Array.isArray(config.palettes)&&config.palettes.length?config.palettes:defaultPalettes).map(p=>({...p}));
    refresh();
  }
  function entries(host){return(hosts.get(host)?.entries||[]).map(entry=>({...entry}));}
  function refresh(){for(const [host,record]of hosts)draw(host,record);}
  function focusAction(host,key,action){
    for(const button of host.querySelectorAll("[data-legend-action]"))
      if(button.dataset.legendKey===key&&button.dataset.legendAction===action){button.focus({preventScroll:true});break;}
  }
  function closePalette(returnFocus=false){
    if(!openKey)return;
    const key=openKey;openKey=null;refresh();
    if(returnFocus)for(const host of hosts.keys())focusAction(host,key,"palette");
  }
  function activate(host,event){
    const button=event.target.closest?.("button[data-legend-action]");if(!button||!host.contains(button))return;
    const {legendAction:action,legendKey:key,paletteValue:value}=button.dataset;
    event.preventDefault();event.stopPropagation();
    if(action==="collapse"||action==="restore"){
      action==="collapse"?collapsed.add(key):collapsed.delete(key);openKey=null;openLimits=null;refresh();
      focusAction(host,key,action==="collapse"?"restore":"collapse");config.onChange?.(capture());
    }else if(action==="palette"){
      openLimits=null;openKey=openKey===key?null:key;refresh();
      if(openKey){host.querySelector('[data-palette-menu][data-legend-key="'+key+'"] [aria-checked="true"]')?.focus({preventScroll:true});}
      else focusAction(host,key,"palette");
    }else if(action==="choose"&&config.palettes.some(p=>p.value===value)){
      const record=hosts.get(host),entry=record?.entries.find(entry=>stateKey(record.scope,entry.kind)===key);
      if(!entry)return;
      openKey=null;setPalette(entry.paletteKey||entry.kind,value);refresh();focusAction(host,key,"palette");
    }else if(action==="limits"){
      openKey=null;openLimits=openLimits===key?null:key;refresh();
      host.querySelector('[data-limit-form] input')?.focus({preventScroll:true});
    }else if(action==="limits-apply"||action==="limits-auto"){
      const form=button.closest('[data-limit-form]'),entry=hosts.get(host)?.entries.find(entry=>stateKey(hosts.get(host).scope,entry.kind)===key);
      if(!form||!entry?.limitKey)return;
      try{const lo=form.querySelector('[name="minimum"]').value.trim(),hi=form.querySelector('[name="maximum"]').value.trim();
        if(action==="limits-apply"&&(!lo||!hi))throw Error("Enter both minimum and maximum.");
        const factor=entry.displayFactor||1;
        setLimits(entry.limitKey,action==="limits-auto"?null:Number(lo)/factor,action==="limits-auto"?null:Number(hi)/factor);focusAction(host,key,"limits");
      }catch(error){form.querySelector('[role="alert"]').textContent=error.message;}
    }
  }
  function slide(host,event){
    const slider=event.target;if(!slider.matches?.('[data-limit-slider]'))return;
    const form=slider.closest('[data-limit-form]'),record=hosts.get(host),entry=record.entries.find(entry=>entry.limitKey===form.dataset.limitKey);
    if(!entry)return;
    const lo=form.querySelector('[data-limit-slider="minimum"]'),hi=form.querySelector('[data-limit-slider="maximum"]');
    if(slider===lo)lo.value=String(Math.min(Number(lo.value),Number(hi.value)-1));
    else hi.value=String(Math.max(Number(hi.value),Number(lo.value)+1));
    const a=Number(form.dataset.sliderMin),b=Number(form.dataset.sliderMax),value=t=>a*(1-t/1000)+b*(t/1000),min=value(Number(lo.value)),max=value(Number(hi.value)),factor=entry.displayFactor||1;
    for(const [name,v]of [['minimum',min],['maximum',max]]){form.querySelector('[name="'+name+'"]').value=String(Number((v*factor).toPrecision(12)));form.querySelector('[data-limit-slider="'+name+'"]').setAttribute('aria-valuetext',number(v*factor)+' '+(entry.displayUnit||entry.unit||''));}
    try{setLimits(entry.limitKey,min,max,{keepOpen:true});form.querySelector('[role="alert"]').textContent='';}catch(error){form.querySelector('[role="alert"]').textContent=error.message;}
  }
  function keydown(host,event){
    if(event.key==="Escape"&&openLimits){const key=openLimits;openLimits=null;event.preventDefault();event.stopPropagation();refresh();focusAction(host,key,"limits");return;}
    if(event.key==="Enter"&&event.target.closest?.('[data-limit-form]')){event.preventDefault();event.target.closest('[data-limit-form]').querySelector('[data-legend-action="limits-apply"]').click();return;}
    if(event.key==="Escape"&&openKey){event.preventDefault();event.stopPropagation();closePalette(true);return;}
    const menu=event.target.closest?.("[data-palette-menu]");if(!menu)return;
    const choices=[...menu.querySelectorAll('[role="menuitemradio"]')],index=choices.indexOf(event.target);
    let next;
    if(["ArrowDown","ArrowRight"].includes(event.key))next=(index+1)%choices.length;
    else if(["ArrowUp","ArrowLeft"].includes(event.key))next=(index+choices.length-1)%choices.length;
    else if(event.key==="Home")next=0;else if(event.key==="End")next=choices.length-1;
    if(next!==undefined){event.preventDefault();event.stopPropagation();choices[next].focus();}
  }
  function render(host,values){
    if(!host)return;
    let record=hosts.get(host);
    if(!record){
      record={scope:host.id||"legend-"+(++hostSerial),entries:[]};hosts.set(host,record);
      host.addEventListener("click",event=>activate(host,event));host.addEventListener("keydown",event=>keydown(host,event));
      host.addEventListener("input",event=>slide(host,event));
      // Interactions with overlays must not start viewport orbit/pan gestures.
      for(const type of["pointerdown","dblclick","wheel"])host.addEventListener(type,event=>event.stopPropagation());
      host.ownerDocument.addEventListener("pointerdown",event=>{
        if(openKey&&openKey.startsWith(record.scope+":")&&!host.contains(event.target))closePalette();
        if(openLimits&&openLimits.startsWith(record.scope+":")&&!host.contains(event.target)){openLimits=null;refresh();}
      });
    }
    record.entries=(Array.isArray(values)?values:[]).filter(entry=>Number.isFinite(entry.min)&&Number.isFinite(entry.max)).map(entry=>({...entry}));
    if(openKey?.startsWith(record.scope+":")&&!record.entries.some(entry=>stateKey(record.scope,entry.kind)===openKey))openKey=null;
    if(openLimits?.startsWith(record.scope+":")&&!record.entries.some(entry=>stateKey(record.scope,entry.kind)===openLimits))openLimits=null;
    draw(host,record);
  }
  function actionButton(action,key,label,icon,extra=""){
    return'<button type="button" class="legend-action legend-'+action+'" data-legend-action="'+action+'" data-legend-key="'+escape(key)+'" title="'+escape(label)+'" aria-label="'+escape(label)+'" '+extra+'>'+icon+'</button>';
  }
  function draw(host,record){
    host.classList.add("wing-legends");host.hidden=!record.entries.length;
    const oldForm=host.querySelector('[data-limit-form]'),oldSection=oldForm?.closest('.viewport-scale');
    const html=record.entries.map(entry=>{
      const key=stateKey(record.scope,entry.kind),title=entry.title||"Quantity",hidden=collapsed.has(key),opened=openKey===key;
      const format=typeof entry.formatNumber==="function"?entry.formatNumber:number,factor=entry.displayFactor||1,unit=entry.displayUnit||entry.unit,formatNumber=value=>format(value*factor);
      const palette=getPalette(entry.paletteKey||entry.kind),chosen=config.palettes.find(p=>p.value===palette),paletteName=chosen?.label||palette;
      if(hidden)return'<section class="viewport-scale is-collapsed" data-scale="'+escape(entry.kind)+'" aria-label="'+escape(title)+' color scale">'+
        actionButton("restore",key,"Show "+title+" color scale",restoreIcon)+"</section>";
      const paletteButton=config.setPalette&&entry.paletteEditable!==false?actionButton("palette",key,"Choose "+title+" color scale ("+paletteName+")",paletteIcon,
        'aria-haspopup="menu" aria-expanded="'+opened+'"'):"";
      const range=getLimits(entry.limitKey),limitsButton=entry.limitKey&&!entry.categories?actionButton("limits",key,"Set "+title+" color-scale limits",restoreIcon,'aria-expanded="'+(openLimits===key)+'"'):"";
      const low=range?.[0]??entry.min,high=range?.[1]??entry.max,padding=high>low?0:Math.max(Math.abs(low)*.1,1e-12),sliderMin=low-padding,sliderMax=high+padding;
      const limitsForm=openLimits===key?'<div class="legend-limits" data-limit-form data-limit-key="'+escape(entry.limitKey)+'" data-slider-min="'+sliderMin+'" data-slider-max="'+sliderMax+'"><div>Range in '+escape(unit||"dimensionless")+'</div>'+[['minimum',low],['maximum',high]].map(([name,value])=>'<label>'+name[0].toUpperCase()+name.slice(1)+'<input name="'+name+'" type="number" step="any" value="'+escape(Number((value*factor).toPrecision(12)))+'"></label><input type="range" data-limit-slider="'+name+'" min="0" max="1000" step="1" value="'+(sliderMax>sliderMin?Math.round(1000*(value-sliderMin)/(sliderMax-sliderMin)):500)+'" aria-label="'+name[0].toUpperCase()+name.slice(1)+' color-scale limit" aria-valuetext="'+escape(formatNumber(value)+' '+(unit||''))+'">').join('')+'<div class="legend-limits-actions">'+actionButton("limits-apply",key,"Apply color-scale limits","Apply")+actionButton("limits-auto",key,"Use automatic color-scale limits","Automatic")+'</div><div role="alert"></div><small>Sliders update colors immediately. Enter numbers and Apply to extend the range. Values outside the limits use the end colors. Data and exports remain unchanged.</small></div>':"";
      const menu=opened?'<div class="legend-palette-menu" role="menu" aria-label="'+escape(title)+' color scale" data-palette-menu data-legend-key="'+escape(key)+'">'+
        '<div class="legend-palette-heading">'+escape(title)+'</div>'+config.palettes.map(p=>
          '<button type="button" role="menuitemradio" aria-checked="'+(p.value===palette)+'" data-legend-action="choose" data-legend-key="'+escape(key)+'" data-palette-value="'+escape(p.value)+'">'+
          (p.gradient?'<span class="legend-palette-swatch" style="background:'+escape(typeof p.gradient==='function'?p.gradient(entry):p.gradient)+'"></span>':"")+escape(p.label)+"</button>").join("")+"</div>":"";
      return'<section class="viewport-scale" data-scale="'+escape(entry.kind)+'" aria-label="'+escape(title)+' color scale">'+
        '<div class="viewport-scale-header"><div class="viewport-scale-title">'+escape(title)+' <span>'+escape(unit||"dimensionless")+'</span></div>'+
        '<div class="legend-actions">'+limitsButton+paletteButton+actionButton("collapse",key,"Hide "+title+" color scale (keep contours visible)","−")+'</div></div>'+
        menu+limitsForm+'<div class="viewport-scale-case">'+escape(entry.caseLabel||"")+'</div>'+
        (entry.categories?'<div class="legend-categories">'+entry.categories.map(category=>'<div class="legend-category"><span class="legend-category-swatch" style="background:'+escape(category.color)+'"></span><span>'+escape(category.label)+'</span></div>').join('')+'</div>':
        '<div class="viewport-scale-ramp" style="background:'+escape(entry.gradient)+'"></div>'+
        '<div class="viewport-scale-ticks"><span>'+escape(formatNumber(entry.min))+'</span><span>'+escape(formatNumber(entry.min/2+entry.max/2))+'</span><span>'+escape(formatNumber(entry.max))+'</span></div>')+
        (range?'<div class="viewport-scale-note">Manual limits · end colors clip out-of-range values</div>':"")+(entry.note?'<div class="viewport-scale-note">'+escape(entry.note)+'</div>':"")+"</section>";
    }).join("");
    // Keep the live range input attached: replacing it during input breaks
    // native pointer capture and keyboard focus while dragging or animating.
    if(oldForm&&openLimits&&record.entries.some(entry=>stateKey(record.scope,entry.kind)===openLimits&&entry.limitKey===oldForm.dataset.limitKey)){
      const template=host.ownerDocument.createElement('div');template.innerHTML=html;
      const fresh=[...template.children],replacement=fresh.find(section=>section.dataset.scale===oldSection.dataset.scale),newForm=replacement?.querySelector('[data-limit-form]');
      if(newForm){
        for(const child of [...oldSection.children])if(child!==oldForm)child.remove();
        let after=false;for(const child of [...replacement.children]){if(child===newForm){after=true;continue;}after?oldSection.append(child):oldSection.insertBefore(child,oldForm);}
        for(const child of [...host.children])if(child!==oldSection)child.remove();
        after=false;for(const section of fresh){if(section===replacement){after=true;continue;}after?host.append(section):host.insertBefore(section,oldSection);}
        return;
      }
    }
    host.innerHTML=html;
  }
  return{render,number,configure,entries,capture,restore,validateState,getLimits,setLimits,resolveLimits,getPalette,setPalette};
});
