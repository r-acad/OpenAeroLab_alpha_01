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
  // Bilinear contours use the original quadrilateral samples without imposing
  // a diagonal. Trace the actual bilinear level set, including its saddle case.
  function quadIsoline(a,b,c,d,level) {
    const p=[a,b,c,d],f=p.map(q=>q.v-level);
    if(f.some(v=>!Number.isFinite(v))||Math.min(...f)>0||Math.max(...f)<0)return [];
    const turns=p.map((q,k)=>{const r=p[(k+1)%4],s=p[(k+2)%4];return (r.x-q.x)*(s.y-r.y)-(r.y-q.y)*(s.x-r.x);});
    if(turns.some(v=>!Number.isFinite(v)||Math.abs(v)<1e-14)||!turns.every(v=>v*turns[0]>0))return [];
    const uv=[[0,0],[1,0],[1,1],[0,1]],cross=[];
    for(let k=0;k<4;k++){
      const n=(k+1)%4;if((f[k]<0)===(f[n]<0))continue;
      const t=f[k]/(f[k]-f[n]);cross.push({edge:k,q:uv[k].map((v,h)=>v+t*(uv[n][h]-v))});
    }
    if(cross.length!==2&&cross.length!==4)return [];
    const A=f[0],B=f[1]-f[0],C=f[3]-f[0],D=f[0]-f[1]+f[2]-f[3];
    const world=([u,v])=>[(1-u)*(1-v)*a.x+u*(1-v)*b.x+u*v*c.x+(1-u)*v*d.x,(1-u)*(1-v)*a.y+u*(1-v)*b.y+u*v*c.y+(1-u)*v*d.y];
    let pairs;
    if(cross.length===2)pairs=[[cross[0].q,cross[1].q]];
    else {
      const saddle=(f[0]*f[2]-f[1]*f[3])/D;
      if(Math.abs(saddle)<1e-13 && Math.abs(D)>1e-13){const center=[-C/D,-B/D];return cross.map(e=>[world(e.q),world(center)]);}
      pairs=(f[0]<0)===(saddle<0)?[[cross[0].q,cross[1].q],[cross[2].q,cross[3].q]]:[[cross[0].q,cross[3].q],[cross[1].q,cross[2].q]];
    }
    const out=[];
    for(const [s,t] of pairs){
      const alongU=Math.abs(t[0]-s[0])>=Math.abs(t[1]-s[1]);let previous=world(s);
      for(let k=1;k<=8;k++){
        let q=t;
        if(k<8){
          if(alongU){const u=s[0]+(t[0]-s[0])*k/8;q=[u,-(A+B*u)/(C+D*u)];}
          else{const v=s[1]+(t[1]-s[1])*k/8;q=[-(A+C*v)/(B+D*v),v];}
          if(q.some(v=>!Number.isFinite(v)||v< -1e-9||v>1+1e-9))continue;
        }
        const next=world(q);out.push([previous,next]);previous=next;
      }
    }
    return out;
  }
  function isolines(blocks,level=1) {
    const segments=[];
    for(const {pts,ni,nj,structured} of blocks||[]){if(!structured)continue;
      for(let j=0;j<nj-1;j++)for(let i=0;i<ni-1;i++){
        const a=j*ni+i,b=a+1,d=a+ni,c=d+1;
        const lo=Math.min(pts[a].v,pts[b].v,pts[c].v,pts[d].v),hi=Math.max(pts[a].v,pts[b].v,pts[c].v,pts[d].v);
        if(lo<=level&&hi>=level)segments.push(...quadIsoline(pts[a],pts[b],pts[c],pts[d],level));
      }
    }return segments;
  }
  return {toolbar,probe,isolines,quadIsoline,exportChart};
})();
