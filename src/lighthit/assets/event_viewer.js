"use strict";
(()=>{
const $=id=>document.getElementById(id),sum=a=>a.reduce((x,y)=>x+y,0),fmt=(x,n=4)=>Number(x).toLocaleString("en-US",{maximumSignificantDigits:n});
function maxAbs(rows,floor=0){let out=floor;for(const row of rows){if(Array.isArray(row)){for(const value of row)out=Math.max(out,Math.abs(value))}else out=Math.max(out,Math.abs(row))}return out}
const labels=["Unscattered","One scattering","Two or more"],colors=["#f5c875","#65d3a5","#bb9aee"];
const config={responsive:true,displaylogo:false,scrollZoom:true,modeBarButtonsToRemove:["sendDataToCloud"]};
const layout={paper_bgcolor:"#101f30",plot_bgcolor:"#101f30",font:{color:"#c6d5e5",family:"system-ui",size:11},margin:{l:62,r:24,t:28,b:50}};
let data,event,visible=[],selected=0,playing=false,last=0,clock=0,signalTrace=0,smoothCache=new WeakMap();
const part=()=>$("part").value;
function pick(v){return part()==="all"?sum(v):part()==="prompt"?v[0]+v[1]:v[Number(part())]}
function signedLog(x,scale){return Math.sign(x)*Math.log10(1+Math.abs(x)/Math.max(scale,1e-300))}
function option(node,value,text){const o=document.createElement("option");o.value=value;o.textContent=text;node.append(o)}
function moduleName(i){const d=data.detector;return `OM ${i} · C${d.cluster_id[i]+1}/S${d.string_id[i]+1}/${d.module_id[i]+1}`}
function validate(x){if(!x||x.schema!=="lighthit/event-viewer/1"||!x.detector||!Array.isArray(x.events)||!x.events.length)throw Error("Unsupported LightHit viewer format");return x}
function groupedEdges(){const raw=data.readout.relative_time_edges_ns,g=Number($("rebin").value),e=[raw[0]];for(let i=g;i<raw.length;i+=g)e.push(raw[i]);if(e.at(-1)!==raw.at(-1))e.push(raw.at(-1));return e}
function centers(){const e=groupedEdges();return e.slice(1).map((x,i)=>(x+e[i])/2)}
function widths(){const e=groupedEdges();return e.slice(1).map((x,i)=>x-e[i])}
function displayVariant(){return $("display").value==="raw"?"raw":"smooth"}
function componentRows(kind){if(kind==="raw")return event.components;let rows=smoothCache.get(event);if(!rows){rows=LightHitViewerSmoothing.smoothComponents(event.components,data.readout.relative_time_edges_ns,3);smoothCache.set(event,rows)}return rows}
function bins(i,kind=displayVariant()){const raw=componentRows(kind)[i],g=Number($("rebin").value),out=[];for(let b=0;b<raw.length;b+=g){const value=[0,0,0];for(let j=b;j<Math.min(b+g,raw.length);j++)for(let p=0;p<3;p++)value[p]+=raw[j][p];out.push(value)}return out}
function windowCharge(i,kind=displayVariant()){const row=bins(i,kind);return [0,1,2].map(p=>sum(row.map(v=>v[p])))}
function refreshVisible(){visible=data.detector.positions_m.map((_,i)=>i).filter(i=>$("cluster").value==="all"||data.detector.cluster_id[i]===Number($("cluster").value));$("module").replaceChildren();visible.forEach(i=>option($("module"),i,`${moduleName(i)} · Q=${fmt(pick(event.charge_components[i]))}`));if(!visible.includes(selected))selected=visible.reduce((a,b)=>Math.abs(pick(event.charge_components[a]))>Math.abs(pick(event.charge_components[b]))?a:b,visible[0]);$("module").value=selected}
function sourceTraces(){const traces=[];for(const s of event.sources||[]){if(s.type==="point"){traces.push({type:"scatter3d",mode:"markers",x:[s.position_m[0]],y:[s.position_m[1]],z:[s.position_m[2]],marker:{size:9,color:"#f5c875",symbol:"diamond"},name:s.label||"Flash"})}else{const half=s.extent_m||40,a=s.position_m.map((x,j)=>x-half*s.direction[j]),b=s.position_m.map((x,j)=>x+half*s.direction[j]),shape=s.shape||"line",points=shape==="spindle"?Array.from({length:9},(_,i)=>a.map((x,j)=>x+(b[j]-x)*i/8)):[a,b],sizes=shape==="spindle"?[2,3,5,8,11,8,5,3,2]:points.map(()=>4),length=2*half;traces.push({type:"scatter3d",mode:"lines+markers",x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),marker:{size:sizes,color:"#f5c875"},line:{color:"#f5c875",width:shape==="spindle"?4:7},text:points.map(()=>`${s.label||"Source"}<br>length ≈ ${fmt(length)} m`),hovertemplate:"%{text}<extra></extra>",name:s.label||"Event axis"});traces.push({type:"cone",x:[b[0]],y:[b[1]],z:[b[2]],u:[s.direction[0]],v:[s.direction[1]],w:[s.direction[2]],sizemode:"absolute",sizeref:Math.max(2,Math.min(6,length*.03)),colorscale:[[0,"#f5c875"],[1,"#f5c875"]],showscale:false,hoverinfo:"skip",name:"Direction"})}}return traces}
function frameValues(){const index=Number($("time").value),mode=$("time-mode").value;return visible.map(i=>{if(mode==="integrated")return pick(event.charge_components[i]);const row=bins(i);if(mode==="frame")return pick(row[index]);let a=[0,0,0];for(let b=0;b<=index;b++)for(let p=0;p<3;p++)a[p]+=row[b][p];return pick(a)})}
function marker(){const value=frameValues(),peak=maxAbs(value,1e-300),sizes=value.map(v=>v?Number($("gain").value)*(3+22*Math.sqrt(Math.abs(v)/peak)):0),scale=Math.max(peak*1e-5,1e-300),colour=value.map(v=>signedLog(v,scale)),extent=maxAbs(colour,1);return{sizes,colour,extent}}
async function geometry(){const d=data.detector,pos=visible.map(i=>d.positions_m[i]),xyz={x:pos.map(p=>p[0]),y:pos.map(p=>p[1]),z:pos.map(p=>p[2])},traces=[{type:"scatter3d",mode:"markers",...xyz,marker:{size:2,color:"#40566c"},customdata:visible,text:visible.map(moduleName),hovertemplate:"%{text}<extra></extra>",showlegend:false}];const strings=new Map();visible.forEach(i=>{const k=`${d.cluster_id[i]}-${d.string_id[i]}`;if(!strings.has(k))strings.set(k,[]);strings.get(k).push(d.positions_m[i])});const line={x:[],y:[],z:[]};strings.forEach(v=>{v.sort((a,b)=>a[2]-b[2]);[v[0],v.at(-1),null].forEach(p=>{line.x.push(p?p[0]:null);line.y.push(p?p[1]:null);line.z.push(p?p[2]:null)})});traces.push({type:"scatter3d",mode:"lines",...line,line:{color:"#2b4258",width:2},hoverinfo:"skip",showlegend:false},...sourceTraces());signalTrace=traces.length;const m=marker();traces.push({type:"scatter3d",mode:"markers",...xyz,customdata:visible,text:visible.map(i=>`${moduleName(i)}<br>Q=${fmt(pick(event.charge_components[i]))}`),hovertemplate:"%{text}<extra></extra>",marker:{size:m.sizes,color:m.colour,cmin:-m.extent,cmax:m.extent,colorscale:[[0,"#a33d66"],[.5,"#dce8ef"],[1,"#2689b8"]],colorbar:{title:"sign·log",thickness:11}},showlegend:false});const axis=t=>({title:t,backgroundcolor:"#101f30",gridcolor:"#24384f",color:"#9fb5ce"});await Plotly.react("geometry",traces,{...layout,margin:{l:0,r:45,t:0,b:0},uirevision:`camera-${event.event_id}`,scene:{xaxis:axis("x, m"),yaxis:axis("y, m"),zaxis:axis("z, m"),aspectmode:"data"}},config);$("geometry").removeAllListeners("plotly_click");$("geometry").on("plotly_click",e=>{const i=e.points[0].customdata;if(Number.isInteger(i))choose(i)})}
async function waveform(){
  const t=centers(),w=widths(),shift=$("absolute").checked?event.time_origin_ns[selected]:0;
  const x=t.map(v=>v+shift),mode=$("display").value,kinds=mode==="both"?["raw","smooth"]:[mode],tr=[];
  for(const kind of kinds){
    const row=bins(selected,kind),suffix=kind==="raw"?"raw":"3 ns";
    const dash=kind==="raw"?"solid":"dash";
    for(let p=0;p<3;p++)tr.push({type:"scatter",mode:"lines",x,
      y:row.map((v,i)=>v[p]/w[i]),name:`${labels[p]} · ${suffix}`,
      line:{color:colors[p],width:1.7,dash}});
    tr.push({type:"scatter",mode:"lines",x,
      y:row.map((v,i)=>(v[0]+v[1])/w[i]),name:`0+1 · ${suffix}`,
      line:{color:"#63b7ff",width:2,dash}});
    tr.push({type:"scatter",mode:"lines",x,y:row.map((v,i)=>sum(v)/w[i]),
      name:`Total · ${suffix}`,line:{color:"#eef5fb",width:2.4,dash}});
  }
  await Plotly.react("waveform",tr,{...layout,legend:{orientation:"h",y:1.18},
    xaxis:{title:$("absolute").checked?"Absolute time, ns":"Relative time, ns",gridcolor:"#25384c"},
    yaxis:{title:`${data.units.signal} / ns`,gridcolor:"#25384c"}},config);
  const q=event.charge_components[selected],active=!event.active||event.active[selected];
  const details=[["Module",moduleName(selected)],
    ["Integrated charge",`${fmt(sum(q))} ${data.units.signal}`],
    ["0 / 1 / ≥2",q.map(value=>fmt(value)).join(" / ")]];
  for(const kind of kinds)details.push([kind==="raw"?"Raw window charge":"3 ns window charge",
    `${fmt(sum(windowCharge(selected,kind)))} ${data.units.signal}`]);
  details.push(["Time origin",`${fmt(event.time_origin_ns[selected])} ns`],
    ["Full time spectrum",active?"computed":"skipped: Q below threshold"]);
  $("details").replaceChildren();
  details.forEach(([a,b])=>{const dt=document.createElement("dt"),dd=document.createElement("dd");
    dt.textContent=a;dd.textContent=b;$("details").append(dt,dd)});
}
async function heatmap(){const t=centers(),raw=visible.map(i=>bins(i).map(pick)),peak=maxAbs(raw,1e-300),scale=peak*1e-6,z=raw.map(row=>row.map(v=>signedLog(v,scale))),extent=maxAbs(z,1);$("heatmap-mode").textContent=(displayVariant()==="raw"?"Raw bins":"3 ns Gaussian display")+" · click a row";await Plotly.react("heatmap",[{type:"heatmap",x:t,y:visible,z,zmin:-extent,zmax:extent,colorscale:[[0,"#a33d66"],[.5,"#f0f3f5"],[1,"#176b93"]],colorbar:{title:"sign·log"},hovertemplate:"OM %{y}<br>%{x:.1f} ns<br>%{z:.3f}<extra></extra>"}],{...layout,xaxis:{title:"Time relative to OM origin, ns"},yaxis:{title:"OM number",autorange:"reversed"}},config);$("heatmap").removeAllListeners("plotly_click");$("heatmap").on("plotly_click",e=>choose(Number(e.points[0].y)))}
function updateFrame(){const e=groupedEdges(),i=Number($("time").value),mode=$("time-mode").value;$("time-label").textContent=mode==="integrated"?"full signal":`${e[i]}…${e[i+1]} ns`;const m=marker();Plotly.restyle("geometry",{"marker.size":[m.sizes],"marker.color":[m.colour],"marker.cmin":[-m.extent],"marker.cmax":[m.extent]},[signalTrace])}
function windowTotal(kind){let total=0;for(const module of componentRows(kind))for(const bin of module)total+=pick(bin);return total}
function updateWindowMetric(){const mode=$("display").value,units=data.units.signal;if(mode==="both"){$("window-charge-title").textContent="Raw / 3 ns window charge";$("window-charge").textContent=`${fmt(windowTotal("raw"))} / ${fmt(windowTotal("smooth"))} ${units}`}else{$("window-charge-title").textContent=mode==="raw"?"Raw window charge":"3 ns window charge";$("window-charge").textContent=`${fmt(windowTotal(mode))} ${units}`}}
function stop(){playing=false;$("play").textContent="▶ Play"}
function tick(now){if(!playing)return;if(!last)last=now;if(now-last>90){last=now;let i=Number($("time").value)+1;if(i>Number($("time").max)){stop();return}$("time").value=i;updateFrame()}requestAnimationFrame(tick)}
function choose(i){selected=i;$("module").value=i;waveform()}
async function selectEvent(){stop();event=data.events[Number($("event").value)];selected=event.charge_components.map(pick).map(Math.abs).reduce((best,v,i,a)=>v>a[best]?i:best,0);$("time").max=groupedEdges().length-2;$("time").value=0;$("time-mode").value="frame";const active=event.active?event.active.filter(Boolean).length:data.detector.positions_m.length;$("modules").textContent=`${data.detector.positions_m.length} / ${active}`;const q=event.charge_components.map(pick);$("charge").textContent=`${fmt(sum(q))} ${data.units.signal}`;updateWindowMetric();$("elapsed").textContent=`${fmt(event.diagnostics?.elapsed_seconds||0)} s`;refreshVisible();const screen=event.diagnostics?.screening;$("diagnostics").textContent=(event.diagnostics?.note||"Signed bins are preserved without clipping.")+(screen?` Full spectra: ${screen.full_spectra_computed}; skipped: ${screen.full_spectra_skipped}; threshold ${screen.threshold_pe} p.e.; integrated charge of skipped OMs ${fmt(screen.skipped_integrated_charge_pe)} p.e. (${fmt(100*screen.skipped_fraction_of_array_charge)}%).`:"");await Promise.all([geometry(),waveform(),heatmap()]);updateFrame()}
async function load(x){data=validate(x);smoothCache=new WeakMap();$("display").value="raw";$("event").replaceChildren();data.events.forEach((e,i)=>option($("event"),i,e.label||e.event_id));$("cluster").replaceChildren();option($("cluster"),"all","All clusters");[...new Set(data.detector.cluster_id)].sort((a,b)=>a-b).forEach(c=>option($("cluster"),c,`Cluster ${c+1}`));const medium=data.medium?.provenance;$("description").textContent=`${data.detector.label}${medium?` · ${medium}`:""} · ${data.events.length} events`;await selectEvent();const active=event.active?event.active.filter(Boolean).length:data.detector.positions_m.length;$("status").textContent=active?"Ready · standalone local viewer":"Ready · no active time spectra; check source position and threshold";$("status").classList.remove("error")}
async function changeDisplay(){stop();updateWindowMetric();await Promise.all([geometry(),waveform(),heatmap()]);updateFrame()}
$("event").onchange=selectEvent;$("cluster").onchange=async()=>{refreshVisible();await Promise.all([geometry(),waveform(),heatmap()])};$("part").onchange=selectEvent;$("display").onchange=changeDisplay;$("rebin").onchange=selectEvent;$("module").onchange=()=>choose(Number($("module").value));$("absolute").onchange=waveform;$("gain").oninput=updateFrame;$("time").oninput=()=>{stop();if($("time-mode").value==="integrated")$("time-mode").value="frame";updateFrame()};$("time-mode").onchange=()=>{stop();updateFrame()};$("all-time").onclick=()=>{$("time-mode").value="integrated";stop();updateFrame()};$("play").onclick=()=>{if(playing){stop();return}if($("time-mode").value==="integrated")$("time-mode").value="frame";if(Number($("time").value)>=Number($("time").max))$("time").value=0;playing=true;last=0;$("play").textContent="Ⅱ Pause";requestAnimationFrame(tick)};document.addEventListener("visibilitychange",()=>{if(document.hidden)stop()});$("file").onchange=async e=>{try{await load(JSON.parse(await e.target.files[0].text()))}catch(err){$("status").textContent=err.message;$("status").classList.add("error")}};
load(JSON.parse($("result").textContent)).catch(err=>{$("status").textContent=err.message;$("status").classList.add("error")})
})();
