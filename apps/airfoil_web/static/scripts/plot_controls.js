"use strict";

// Shared chart interactions; exported SVGs contain the visible data and legend.
window.AirfoilPlotTools = (() => {
  function download(text, name, type='image/svg+xml') {
    const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');
    a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function exportChart(view) {
    const source=view.node.querySelector('svg');if(!source)return;
    const svg=source.cloneNode(true),style=getComputedStyle(view.node),ns='http://www.w3.org/2000/svg';
    const series=view.spec.series.filter(s=>s.points?.length),legendRows=[];
    const context=view.node.querySelector('.point-overlay')?.textContent;
    if(context)series.push({name:context});
    for(const s of series){
      const words=String(s.name).split(/\s+/),max=Math.max(24,Math.floor((view.width-60)/6.2));let line='';
      for(const word of words){if(line&&line.length+word.length+1>max){legendRows.push({text:line,color:s.color,dash:s.dash});line='';}line+=(line?' ':'')+word;}
      if(line)legendRows.push({text:line,color:s.color,dash:s.dash});
    }
    const height=view.height+20+18*legendRows.length;
    svg.setAttribute('xmlns',ns);svg.setAttribute('width',view.width);svg.setAttribute('height',height);
    svg.setAttribute('viewBox',`0 0 ${view.width} ${height}`);svg.setAttribute('font-family',style.fontFamily||'sans-serif');
    const paper=style.getPropertyValue('--paper').trim()||'#fff',ink=style.getPropertyValue('--ink').trim()||'#243e4c';
    const bg=document.createElementNS(ns,'rect');bg.setAttribute('width','100%');bg.setAttribute('height','100%');bg.setAttribute('fill',paper);svg.prepend(bg);
    for(const node of svg.querySelectorAll('text'))node.setAttribute('fill',ink);
    for(const node of svg.querySelectorAll('line'))node.setAttribute('stroke',style.getPropertyValue('--grid').trim()||'#cbd5dc');
    for(const node of svg.querySelectorAll('[fill="var(--paper)"]'))node.setAttribute('fill',paper);
    legendRows.forEach((row,index)=>{
      const y=view.height+18+18*index;
      if(row.color){const line=document.createElementNS(ns,'path');line.setAttribute('d',`M12 ${y-4}h22`);line.setAttribute('stroke',row.color);line.setAttribute('stroke-width','2.5');if(row.dash)line.setAttribute('stroke-dasharray',row.dash);svg.append(line);}
      const label=document.createElementNS(ns,'text');label.setAttribute('x',row.color?'42':'12');label.setAttribute('y',y);label.setAttribute('fill',ink);label.setAttribute('font-size','11');label.textContent=row.text;svg.append(label);
    });
    const title=document.createElementNS(ns,'title');title.textContent=view.spec.options.title||view.id;svg.prepend(title);
    download(new XMLSerializer().serializeToString(svg),`${view.id}.svg`);
  }
  function toolbar(view) {
    const bar=document.createElement('div');bar.className='chart-tools';
    const button=document.createElement('button');button.type='button';button.textContent='Export SVG';button.title='Export the current chart, zoom and legend as a vector SVG file';
    button.addEventListener('pointerdown',e=>e.stopPropagation());button.onclick=()=>exportChart(view);bar.append(button);view.node.append(bar);
  }
  function probe(view,e) {
    let box=view.node.querySelector('.chart-probe');
    if(!document.getElementById('chart-interrogate')?.checked){box?.remove();return;}
    const p=view.position(e);if(!p)return;
    const b=view.currentBounds,m=view.margin;let best=null,distance=Infinity;
    for(const series of view.spec.series)for(const q of series.points||[]){
      if(!Number.isFinite(q.x)||!Number.isFinite(q.y)||q.x<b.xmin||q.x>b.xmax||q.y<b.ymin||q.y>b.ymax)continue;
      const x=m.left+(q.x-b.xmin)/(b.xmax-b.xmin)*(view.width-m.left-m.right);
      const y=m.top+(view.spec.options.invertY?(q.y-b.ymin):(b.ymax-q.y))/(b.ymax-b.ymin)*(view.height-m.top-m.bottom);
      const d=(p.x-x)**2+(p.y-y)**2;if(d<distance){distance=d;best={...q,name:series.name};}
    }
    if(!best){box?.remove();return;}
    if(!box){box=document.createElement('output');box.className='chart-probe';view.node.append(box);}
    const value=view.spec.options.logY?10**best.y:best.y;
    const name=view.spec.options.logY?best.name.replace(/^log₁₀\s*/,''):best.name;
    box.textContent=`${name}: ${view.spec.options.xLabel||'x'} = ${best.x.toPrecision(7)}, ${view.spec.options.yLabel||(view.spec.options.logY?'residual':'value')} = ${value.toPrecision(7)}`;
  }
  // Marching triangles avoids ambiguous saddle connections and uses each block's connectivity.
  function isolines(blocks,level=1) {
    const segments=[];
    for(const {pts,ni,nj,structured} of blocks||[]){if(!structured)continue;
      const triangle=(ids)=>{const out=[];for(let k=0;k<3;k++){
        const a=pts[ids[k]],b=pts[ids[(k+1)%3]];if((a.v<level)===(b.v<level))continue;
        const t=(level-a.v)/(b.v-a.v);out.push([a.x+t*(b.x-a.x),a.y+t*(b.y-a.y)]);
      }if(out.length===2)segments.push(out);};
      for(let j=0;j<nj-1;j++)for(let i=0;i<ni-1;i++){const a=j*ni+i,b=a+1,c=a+ni,d=c+1;triangle([a,b,d]);triangle([a,d,c]);}
    }return segments;
  }
  return {toolbar,probe,isolines,exportChart};
})();
