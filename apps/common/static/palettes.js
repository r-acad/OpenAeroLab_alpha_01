/* Colour definitions adapted from JFEM-Boxer (AGPL-3.0); see THIRD_PARTY_NOTICES.md. */
((root) => {
  'use strict';
  const scales = {
    spectrum: [[.14,.24,.50],[.16,.56,.74],[.28,.72,.44],[.94,.79,.26],[.86,.24,.22]],
    viridis: [[.267,.005,.329],[.230,.322,.546],[.128,.567,.551],[.369,.789,.383],[.993,.906,.144]],
    inferno: [[.001,.000,.014],[.342,.062,.429],[.735,.216,.330],[.978,.558,.035],[.988,.998,.645]],
    coolwarm: [[.230,.299,.754],[.554,.690,.996],[.865,.865,.865],[.957,.598,.477],[.706,.016,.150]],
    grayscale: [[.10,.10,.10],[1,1,1]],
    signed: [[195/255,52/255,66/255],[242/255,243/255,239/255],[48/255,112/255,196/255]]
  };
  const labels = {spectrum:'Boxer spectrum',viridis:'Viridis',inferno:'Inferno',coolwarm:'Cool–warm',grayscale:'Grayscale',signed:'Signed red (−) / blue (+)'};
  let selected='spectrum';
  try { const value=root.localStorage?.getItem('openaerolab-palette');if(scales[value])selected=value; } catch {}
  function sample(t,name=selected) {
    const stops=scales[name]||scales.spectrum,x=Math.max(0,Math.min(1,Number.isFinite(t)?t:.5))*(stops.length-1),i=Math.min(stops.length-2,Math.floor(x)),w=x-i;
    return stops[i].map((value,k)=>value*(1-w)+stops[i+1][k]*w);
  }
  function valueColor(value,min,max,name=selected) {
    if(![value,min,max].every(Number.isFinite)||min>max)return [.45,.48,.52];
    if(name!=='signed')return sample(min===max ? .5 : (value-min)/(max-min),name);
    const [red,zero,blue]=scales.signed,mix=(a,b,t)=>a.map((v,i)=>v+(b[i]-v)*Math.max(0,Math.min(1,t)));
    if(value===0)return zero.slice();if(min===max)return mix(zero,value<0?red:blue,.55);
    if(max<=0&&value<0)return mix(red,max===0?zero:[251/255,225/255,228/255],(value-min)/(max-min));
    if(min>=0&&value>0)return mix(min===0?zero:[220/255,233/255,250/255],blue,(value-min)/(max-min));
    return value<0?mix(zero,red,min<0?value/min:1):mix(zero,blue,max>0?value/max:1);
  }
  function gradient(name=selected,min=0,max=1) {
    if(name==='signed'){const positions=min<0&&max>0?[0,-min/(max-min),1]:[0,.5,1];return `linear-gradient(90deg,${positions.map(t=>`rgb(${valueColor(min+(max-min)*t,min,max,name).map(v=>Math.round(255*v)).join(',')}) ${t*100}%`).join(',')})`;}

    const stops=scales[name]||scales.spectrum;
    return `linear-gradient(90deg,${stops.map((rgb,i)=>`rgb(${rgb.map(v=>Math.round(255*v)).join(',')}) ${100*i/(stops.length-1)}%`).join(',')})`;
  }
  function set(name) {
    if(!Object.hasOwn(scales,name))return false;
    selected=name;try{root.localStorage?.setItem('openaerolab-palette',name);}catch{}
    root.document?.documentElement.style.setProperty('--result-ramp',gradient());
    root.dispatchEvent?.(new Event('ludwig-palette-change'));return true;
  }
  root.LudwigPalettes={sample,valueColor,gradient,set,get:()=>selected,labels};
  if(typeof module==='object'&&module.exports)module.exports=root.LudwigPalettes;
})(globalThis);
