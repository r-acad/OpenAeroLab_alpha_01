/* Colour definitions adapted from JFEM-Boxer (AGPL-3.0); see THIRD_PARTY_NOTICES.md. */
((root) => {
  'use strict';
  const scales = {
    spectrum: [[.14,.24,.50],[.16,.56,.74],[.28,.72,.44],[.94,.79,.26],[.86,.24,.22]],
    viridis: [[.267,.005,.329],[.230,.322,.546],[.128,.567,.551],[.369,.789,.383],[.993,.906,.144]],
    inferno: [[.001,.000,.014],[.342,.062,.429],[.735,.216,.330],[.978,.558,.035],[.988,.998,.645]],
    coolwarm: [[.230,.299,.754],[.554,.690,.996],[.865,.865,.865],[.957,.598,.477],[.706,.016,.150]],
    grayscale: [[.10,.10,.10],[1,1,1]]
  };
  const labels = {spectrum:'Boxer spectrum',viridis:'Viridis',inferno:'Inferno',coolwarm:'Cool–warm',grayscale:'Grayscale'};
  let selected='spectrum';
  try { const value=root.localStorage?.getItem('openaerolab-palette');if(scales[value])selected=value; } catch {}
  function sample(t,name=selected) {
    const stops=scales[name]||scales.spectrum,x=Math.max(0,Math.min(1,Number.isFinite(t)?t:.5))*(stops.length-1),i=Math.min(stops.length-2,Math.floor(x)),w=x-i;
    return stops[i].map((value,k)=>value*(1-w)+stops[i+1][k]*w);
  }
  function gradient(name=selected) {
    const stops=scales[name]||scales.spectrum;
    return `linear-gradient(90deg,${stops.map((rgb,i)=>`rgb(${rgb.map(v=>Math.round(255*v)).join(',')}) ${100*i/(stops.length-1)}%`).join(',')})`;
  }
  function set(name) {
    if(!Object.hasOwn(scales,name))return false;
    selected=name;try{root.localStorage?.setItem('openaerolab-palette',name);}catch{}
    root.document?.documentElement.style.setProperty('--result-ramp',gradient());
    root.dispatchEvent?.(new Event('ludwig-palette-change'));return true;
  }
  root.LudwigPalettes={sample,gradient,set,get:()=>selected,labels};
  if(typeof module==='object'&&module.exports)module.exports=root.LudwigPalettes;
})(globalThis);
