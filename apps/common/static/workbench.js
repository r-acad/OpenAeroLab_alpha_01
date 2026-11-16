/* Shared by the workbench pages: the colour theme and the result palette selector. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  function applyTheme(){
    let preference='dark';try{preference=localStorage.getItem('airfoil-theme')||'dark';}catch{}
    if(!['auto','light','dark'].includes(preference))preference='dark';
    const select=$('theme-mode');if(select)select.value=preference;
    document.documentElement.dataset.theme=preference==='auto'?(window.matchMedia?.('(prefers-color-scheme: dark)').matches?'dark':'light'):preference;
  }
  $('theme-mode')?.addEventListener('change',event=>{try{localStorage.setItem('airfoil-theme',event.target.value);}catch{}applyTheme();});
  window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener?.('change',applyTheme);
  applyTheme();
  const palette=$('results-palette');
  if(palette&&window.LudwigPalettes){
    for(const [value,label] of Object.entries(window.LudwigPalettes.labels)){const option=document.createElement('option');option.value=value;option.textContent=label;palette.appendChild(option);}
    palette.value=window.LudwigPalettes.get();
    palette.addEventListener('change',()=>window.LudwigPalettes.set(palette.value));
    document.documentElement.style.setProperty('--result-ramp',window.LudwigPalettes.gradient());
  }
})();
