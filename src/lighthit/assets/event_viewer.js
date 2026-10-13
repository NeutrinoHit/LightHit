"use strict";
// Capture the rendered event, not the DOM, and never rewrite its physics arrays.
// Kept in this asset so existing write_event_viewer builds remain self-contained.
globalThis.LightHitViewerRecording=(()=>{
  const mimeTypes=["video/mp4;codecs=avc1.42E01E","video/mp4","video/webm;codecs=vp9","video/webm;codecs=vp8","video/webm"];
  function supportedTypes(env=globalThis){
    if(!env.MediaRecorder||typeof env.MediaRecorder.isTypeSupported!=="function"||!env.HTMLCanvasElement?.prototype.captureStream)return[];
    return mimeTypes.filter(type=>env.MediaRecorder.isTypeSupported(type));
  }
  function extension(type){return type.startsWith("video/mp4")?"mp4":"webm"}
  function filename(id,type){return `${String(id||"event").replace(/[^a-zA-Z0-9_-]+/g,"-").slice(0,80)||"event"}.${extension(type)}`}
  function detectorBounds(positions,padding=0.1){
    if(!positions.length||positions.some(p=>p.length!==3||p.some(v=>!Number.isFinite(v))))throw Error("Detector bounds require finite 3D positions.");
    if(!Number.isFinite(padding)||padding<0||padding>0.5)throw Error("Choose scene padding between 0 and 50 percent.");
    const ranges=[0,1,2].map(i=>{
      let lo=Infinity,hi=-Infinity;for(const p of positions){lo=Math.min(lo,p[i]);hi=Math.max(hi,p[i])}return[lo,hi];
    });
    const extent=Math.max(...ranges.map(([lo,hi])=>hi-lo),1);
    return ranges.map(([lo,hi])=>{
      // A planar/single-string array still needs a nonzero viewing range.
      const margin=hi===lo?Math.max(extent*0.05,1)/2:(hi-lo)*padding;
      return[lo-margin,hi+margin];
    });
  }
  function localSaveEnabled(location=globalThis.location){
    try{
      const page=new URL(location.href);
      return["http:","https:"].includes(page.protocol)&&["localhost","127.0.0.1","[::1]"].includes(page.hostname)&&page.searchParams.get("local_save")==="1";
    }catch(_){return false}
  }
  async function saveLocal(blob,id,{location=globalThis.location,fetch=globalThis.fetch}={}){
    if(!localSaveEnabled(location))throw Error("Local saving requires a localhost viewer opened with ?local_save=1.");
    if(!blob?.size||!blob.type?.startsWith("video/"))throw Error("No completed video is available to save.");
    const name=filename(id,blob.type),page=new URL(location.href);
    const response=await fetch(new URL(`/recordings/${encodeURIComponent(name)}`,page.origin).href,{
      method:"POST",body:blob,headers:{"Content-Type":blob.type},mode:"same-origin",credentials:"omit",redirect:"error"
    });
    if(!response.ok)throw Error(`Local server could not save the video (HTTP ${response.status}). The normal download remains available.`);
    const result=await response.json();
    if(result?.saved_as!==name)throw Error("Local server returned an unexpected filename. The normal download remains available.");
    return{saved_as:name};
  }
  class Recording{
    constructor({canvas,duration,fps,renderFrame,onProgress=()=>{},onFinished=()=>{},env=globalThis}){
      if(!Number.isFinite(duration)||duration<1||duration>600)throw Error("Choose a duration between 1 and 600 seconds.");
      if(!Number.isFinite(fps)||fps<1||fps>60)throw Error("Choose a frame rate between 1 and 60 FPS.");
      Object.assign(this,{canvas,duration,fps,renderFrame,onProgress,onFinished,env});
      this.chunks=[];this.closed=false;this.busy=false;this.raf=null;this.timer=null;this.error=null;
    }
    async start(){
      try{
        const types=supportedTypes(this.env);
        if(!types.length)throw Error("Video recording is unavailable in this browser. Try a browser with canvas capture and MediaRecorder support.");
        // Draw the first stored state before capture starts: no empty first frame.
        await this.renderFrame(0);
        if(this.closed)return;
        this.stream=this.canvas.captureStream(this.fps);
        for(const mimeType of types){
          try{this.recorder=new this.env.MediaRecorder(this.stream,{mimeType});break}catch(_){/* Try the next advertised codec. */}
        }
        if(!this.recorder)throw Error("The browser could not start any supported video encoder.");
        this.recorder.ondataavailable=e=>{if(e.data?.size)this.chunks.push(e.data)};
        this.recorder.onerror=e=>this.stop(e.error||Error("Video encoder failed."));
        this.recorder.onstop=()=>this.finish();
        this.recorder.start(1000);
        this.started=this.env.performance.now();this.lastFrame=-Infinity;
        // A timeout is independent of requestAnimationFrame, which is throttled
        // in a background tab. The UI also stops safely when the tab is hidden.
        this.timer=this.env.setTimeout(()=>this.stop(),this.duration*1000);
        this.raf=this.env.requestAnimationFrame(now=>this.tick(now));
      }catch(error){this.stop(error)}
    }
    async tick(now){
      if(this.closed)return;
      if(now-this.started>=this.duration*1000){this.stop();return}
      if(now-this.lastFrame>=1000/this.fps&&!this.busy){
        this.lastFrame=now;this.busy=true;
        try{
          const progress=Math.min(1,(now-this.started)/Math.max(1,(this.duration-1/this.fps)*1000));
          await this.renderFrame(progress);
          if(!this.closed)this.onProgress(progress);
        }catch(error){this.stop(error)}finally{this.busy=false}
      }
      if(!this.closed)this.raf=this.env.requestAnimationFrame(next=>this.tick(next));
    }
    stop(error=null){
      if(this.closed)return;
      this.closed=true;this.error=error;
      if(this.raf!==null)this.env.cancelAnimationFrame(this.raf);
      if(this.timer!==null)this.env.clearTimeout(this.timer);
      if(this.recorder&&this.recorder.state!=="inactive"){
        try{this.recorder.stop()}catch(stopError){this.error=this.error||stopError;this.finish()}
      }else this.finish();
    }
    finish(){
      if(this.finished)return;
      this.finished=true;this.closed=true;
      if(this.raf!==null)this.env.cancelAnimationFrame(this.raf);
      if(this.timer!==null)this.env.clearTimeout(this.timer);
      this.stream?.getTracks().forEach(track=>track.stop());
      const mimeType=this.recorder?.mimeType||"video/webm";
      const blob=this.error?null:new this.env.Blob(this.chunks,{type:mimeType});
      this.chunks=[];
      this.onFinished({blob:blob?.size?blob:null,mimeType,error:this.error||(!blob?.size?Error("No video frames were captured."):null)});
    }
  }
  return{supportedTypes,extension,filename,detectorBounds,localSaveEnabled,saveLocal,Recording};
})();
(()=>{
const $=id=>document.getElementById(id),sum=a=>a.reduce((x,y)=>x+y,0),fmt=(x,n=4)=>Number(x).toLocaleString("en-US",{maximumSignificantDigits:n});
function maxAbs(rows,floor=0){let out=floor;for(const row of rows){if(Array.isArray(row)){for(const value of row)out=Math.max(out,Math.abs(value))}else out=Math.max(out,Math.abs(row))}return out}
const labels=["Unscattered","One scattering","Two or more"],colors=["#f5c875","#65d3a5","#bb9aee"];
const config={responsive:true,displaylogo:false,scrollZoom:true,modeBarButtonsToRemove:["sendDataToCloud"]};
const layout={paper_bgcolor:"#101f30",plot_bgcolor:"#101f30",font:{color:"#c6d5e5",family:"system-ui",size:11},margin:{l:62,r:24,t:28,b:50}};
let data,event,visible=[],selected=0,playing=false,last=0,clock=0,signalTrace=0,smoothCache=new WeakMap(),liveTrace=-1,livePrep=null,userClock="detector",clockT0=0,clockStep=5,seriesKey="",seriesCache=new Map();
let recording=null,recordUrl=null,recordLocked=[],recordBlob=null,recordEventId=null;
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
const timed=()=>(event.sources||[]).some(s=>s.timing);
const clockMode=()=>timed()&&$("clock").value==="detector"?"detector":"om";
const currentTime=()=>clockT0+Number($("time").value)*clockStep;
// picked (part-selected) bins of one OM and their running sums, cached per view
function series(i){const key=`${$("event").value}|${part()}|${$("display").value}|${$("rebin").value}`;if(key!==seriesKey){seriesKey=key;seriesCache=new Map()}let s=seriesCache.get(i);if(!s){const v=bins(i).map(pick),c=new Float64Array(v.length+1);v.forEach((x,j)=>{c[j+1]=c[j]+x});s={v,c};seriesCache.set(i,s)}return s}
function sourceRange(){const p=(event.sources||[]).map(LightHitViewerSource.prepare).filter(Boolean);return p.length?[Math.min(...p.map(x=>x.t0)),Math.max(...p.map(x=>x.t1))]:null}
// Time slider: detector-clock steps when the source carries timing, else OM-relative bins
function setupClock(){$("clock").disabled=!timed();$("clock").value=timed()?userClock:"om";if(clockMode()==="detector"){const rows=data.detector.positions_m.map((_,i)=>series(i).v),range=LightHitViewerSource.clockRange(event.time_origin_ns,rows,centers(),sourceRange());clockT0=range.t0;clockStep=Math.min(5,...widths());const count=Math.ceil((range.t1-range.t0)/clockStep);if(count>800)clockStep=(range.t1-range.t0)/800;$("time").max=Math.min(count,800)}else $("time").max=groupedEdges().length-2;$("time").value=0}
function cursorShapes(){if(!event||$("time-mode").value==="integrated")return[];const det=clockMode()==="detector",origin=event.time_origin_ns[selected];let x=det?currentTime()-origin:groupedEdges()[Number($("time").value)];if($("absolute").checked)x+=origin;return[{type:"line",xref:"x",yref:"paper",x0:x,x1:x,y0:0,y1:1,line:{color:"#f5c875",width:1.2,dash:"dot"}}]}
function updateCursor(){if($("waveform").data)return Plotly.relayout("waveform",{shapes:cursorShapes()})}
function sourceTraces(){const traces=[];
  for(const s of event.sources||[]){
    if(s.type==="point"){traces.push({type:"scatter3d",mode:"markers",x:[s.position_m[0]],y:[s.position_m[1]],z:[s.position_m[2]],marker:{size:9,color:"#f5c875",symbol:"diamond"},name:s.label||"Flash"});continue}
    // with timing on the detector clock the full axis is only a dim guide; the moving part is drawn by updateFrame
    const prep=LightHitViewerSource.prepare(s),guide=clockMode()==="detector"&&prep,dir=prep&&prep.reversed?s.direction.map(x=>-x):s.direction,
      half=s.extent_m||40,a=s.position_m.map((x,j)=>x-half*dir[j]),b=s.position_m.map((x,j)=>x+half*dir[j]),shape=s.shape||"line",
      points=shape==="spindle"?Array.from({length:9},(_,i)=>a.map((x,j)=>x+(b[j]-x)*i/8)):[a,b],
      sizes=shape==="spindle"?[2,3,5,8,11,8,5,3,2]:points.map(()=>4),length=2*half,colour=guide?"#8a7443":"#f5c875";
    traces.push({type:"scatter3d",mode:guide?"lines":"lines+markers",x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),marker:{size:sizes,color:colour},line:{color:colour,width:guide?2:shape==="spindle"?4:7},text:points.map(()=>`${s.label||"Source"}<br>length ≈ ${fmt(length)} m`),hovertemplate:"%{text}<extra></extra>",name:s.label||"Event axis"});
    traces.push({type:"cone",x:[b[0]],y:[b[1]],z:[b[2]],u:[dir[0]],v:[dir[1]],w:[dir[2]],sizemode:"absolute",sizeref:Math.max(2,Math.min(6,length*.03)),colorscale:[[0,colour],[1,colour]],showscale:false,hoverinfo:"skip",name:"Direction"});
  }
  return traces}
function frameValues(){const mode=$("time-mode").value;if(mode==="integrated")return visible.map(i=>pick(event.charge_components[i]));
  if(clockMode()==="detector"){const T=currentTime(),e=groupedEdges();return visible.map(i=>{const rel=T-event.time_origin_ns[i],{v,c}=series(i),j=LightHitViewerSource.binAt(e,rel);if(j<0)return 0;if(j>=v.length)return mode==="cumulative"?c[v.length]:0;return mode==="frame"?v[j]:c[j]+v[j]*(rel-e[j])/(e[j+1]-e[j])})}
  const index=Number($("time").value);return visible.map(i=>{const row=bins(i);if(mode==="frame")return pick(row[index]);let a=[0,0,0];for(let b=0;b<=index;b++)for(let p=0;p<3;p++)a[p]+=row[b][p];return pick(a)})}
function marker(){const value=frameValues(),peak=maxAbs(value,1e-300),sizes=value.map(v=>v?Number($("gain").value)*(3+22*Math.sqrt(Math.abs(v)/peak)):0),scale=Math.max(peak*1e-5,1e-300),colour=value.map(v=>signedLog(v,scale)),extent=maxAbs(colour,1);return{sizes,colour,extent}}
async function geometry(){const d=data.detector,pos=visible.map(i=>d.positions_m[i]),xyz={x:pos.map(p=>p[0]),y:pos.map(p=>p[1]),z:pos.map(p=>p[2])},traces=[{type:"scatter3d",mode:"markers",...xyz,marker:{size:2,color:"#40566c"},customdata:visible,text:visible.map(moduleName),hovertemplate:"%{text}<extra></extra>",showlegend:false}];const strings=new Map();visible.forEach(i=>{const k=`${d.cluster_id[i]}-${d.string_id[i]}`;if(!strings.has(k))strings.set(k,[]);strings.get(k).push(d.positions_m[i])});const line={x:[],y:[],z:[]};strings.forEach(v=>{v.sort((a,b)=>a[2]-b[2]);[v[0],v.at(-1),null].forEach(p=>{line.x.push(p?p[0]:null);line.y.push(p?p[1]:null);line.z.push(p?p[2]:null)})});traces.push({type:"scatter3d",mode:"lines",...line,line:{color:"#2b4258",width:2},hoverinfo:"skip",showlegend:false},...sourceTraces());signalTrace=traces.length;const m=marker();traces.push({type:"scatter3d",mode:"markers",...xyz,customdata:visible,text:visible.map(i=>`${moduleName(i)}<br>Q=${fmt(pick(event.charge_components[i]))}`),hovertemplate:"%{text}<extra></extra>",marker:{size:m.sizes,...($("presentation").checked?{showscale:false}:{}),color:m.colour,cmin:-m.extent,cmax:m.extent,colorscale:[[0,"#a33d66"],[.5,"#dce8ef"],[1,"#2689b8"]],colorbar:{title:"sign·log",thickness:11}},showlegend:false});liveTrace=-1;livePrep=clockMode()==="detector"?(event.sources||[]).map(LightHitViewerSource.prepare).find(Boolean)||null:null;if(livePrep){liveTrace=traces.length;traces.push({type:"scatter3d",mode:"markers",x:[],y:[],z:[],marker:{size:[],color:[]},hoverinfo:"skip",showlegend:false})}const bounds=$("scene-bounds").value==="detector"?LightHitViewerRecording.detectorBounds(pos,Number($("scene-padding").value)/100):null;const axis=(t,i)=>({title:t,backgroundcolor:"#101f30",gridcolor:"#24384f",color:"#9fb5ce",...(bounds?{range:bounds[i],autorange:false}:{autorange:true})});await Plotly.react("geometry",traces,{...layout,margin:$("presentation").checked?{l:0,r:0,t:0,b:0}:{l:0,r:45,t:0,b:0},...($("presentation").checked?{showlegend:false}:{}),uirevision:`camera-${event.event_id}`,scene:{...($("presentation").checked?{domain:{x:[0,1],y:[0,1]}}:{}),xaxis:axis("x, m",0),yaxis:axis("y, m",1),zaxis:axis("z, m",2),aspectmode:"data"}},config);$("geometry").removeAllListeners("plotly_click");$("geometry").on("plotly_click",e=>{const i=e.points[0].customdata;if(Number.isInteger(i))choose(i)})}
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
  await Plotly.react("waveform",tr,{...layout,shapes:cursorShapes(),legend:{orientation:"h",y:1.18},
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
function updateFrame(){const mode=$("time-mode").value,det=clockMode()==="detector";
  if(det)$("time-label").textContent=mode==="integrated"?"full signal":`t = ${currentTime().toFixed(0)} ns · detector clock`;
  else{const e=groupedEdges(),i=Number($("time").value);$("time-label").textContent=mode==="integrated"?"full signal":`${e[i]}…${e[i+1]} ns`}
  const m=marker(),updates=[Plotly.restyle("geometry",{"marker.size":[m.sizes],"marker.color":[m.colour],"marker.cmin":[-m.extent],"marker.cmax":[m.extent]},[signalTrace])];
  if(liveTrace>=0&&livePrep){const f=LightHitViewerSource.frame(livePrep,mode==="integrated"?-Infinity:currentTime());updates.push(Plotly.restyle("geometry",{x:[f.x],y:[f.y],z:[f.z],"marker.size":[f.size],"marker.color":[f.color]},[liveTrace]))}
  updates.push(updateCursor());return Promise.all(updates)}
function windowTotal(kind){let total=0;for(const module of componentRows(kind))for(const bin of module)total+=pick(bin);return total}
function updateWindowMetric(){const mode=$("display").value,units=data.units.signal;if(mode==="both"){$("window-charge-title").textContent="Raw / 3 ns window charge";$("window-charge").textContent=`${fmt(windowTotal("raw"))} / ${fmt(windowTotal("smooth"))} ${units}`}else{$("window-charge-title").textContent=mode==="raw"?"Raw window charge":"3 ns window charge";$("window-charge").textContent=`${fmt(windowTotal(mode))} ${units}`}}
function stop(){playing=false;$("play").textContent="▶ Play"}
function tick(now){if(!playing)return;if(!last)last=now;if(now-last>(clockMode()==="detector"?45:90)){last=now;let i=Number($("time").value)+1;if(i>Number($("time").max)){stop();return}$("time").value=i;updateFrame()}requestAnimationFrame(tick)}
function choose(i){selected=i;$("module").value=i;waveform()}
function applyComputedOrders(){const orders=event.computed_orders||[0,1,2],missing=!orders.includes(2);for(const o of $("part").options){if(o.value==="all"||o.value==="2"){o.disabled=missing;o.textContent=o.value==="all"?(missing?"Total (order ≥2 not computed)":"Total"):(missing?"Two or more (not computed)":"Two or more")}}if(missing&&($("part").value==="all"||$("part").value==="2"))$("part").value="prompt"}
async function selectEvent(){stop();event=data.events[Number($("event").value)];applyComputedOrders();selected=event.charge_components.map(pick).map(Math.abs).reduce((best,v,i,a)=>v>a[best]?i:best,0);$("time-mode").value="frame";setupClock();const active=event.active?event.active.filter(Boolean).length:data.detector.positions_m.length;$("modules").textContent=`${data.detector.positions_m.length} / ${active}`;const q=event.charge_components.map(pick);$("charge").textContent=`${fmt(sum(q))} ${data.units.signal}`;updateWindowMetric();$("elapsed").textContent=`${fmt(event.diagnostics?.elapsed_seconds||0)} s`;refreshVisible();const screen=event.diagnostics?.screening;$("diagnostics").textContent=((event.computed_orders&&!event.computed_orders.includes(2))?"Prompt transport: orders 0 and 1 only; order ≥2 was not computed (it is not zero). ":"")+(event.diagnostics?.note||"Signed bins are preserved without clipping.")+(screen?` Full spectra: ${screen.full_spectra_computed}; skipped: ${screen.full_spectra_skipped}; threshold ${screen.threshold_pe} p.e.; integrated charge of skipped OMs ${fmt(screen.skipped_integrated_charge_pe)} p.e. (${fmt(100*screen.skipped_fraction_of_array_charge)}%).`:"");await Promise.all([geometry(),waveform(),heatmap()]);updateFrame()}
async function load(x){data=validate(x);smoothCache=new WeakMap();seriesKey="";$("display").value="raw";$("event").replaceChildren();data.events.forEach((e,i)=>option($("event"),i,e.label||e.event_id));$("cluster").replaceChildren();option($("cluster"),"all","All clusters");[...new Set(data.detector.cluster_id)].sort((a,b)=>a-b).forEach(c=>option($("cluster"),c,`Cluster ${c+1}`));const medium=data.medium?.provenance;$("description").textContent=`${data.detector.label}${medium?` · ${medium}`:""} · ${data.events.length} events`;await selectEvent();const active=event.active?event.active.filter(Boolean).length:data.detector.positions_m.length;$("status").textContent=active?"Ready · standalone local viewer":"Ready · no active time spectra; check source position and threshold";$("status").classList.remove("error");setupRecording()}
async function changeDisplay(){stop();updateWindowMetric();await Promise.all([geometry(),waveform(),heatmap()]);updateFrame()}
function setupRecording(){
  const types=LightHitViewerRecording.supportedTypes();
  $("record-video").disabled=!types.length;
  $("record-status").textContent=types.length?"Fixed camera · replay from the first stored time bin · silent video":"Video recording is unavailable here. Try a browser with canvas capture and MediaRecorder support.";
  $("record-local-save").hidden=!LightHitViewerRecording.localSaveEnabled();
}
function captureCanvas(){
  // Plotly can create hidden picking canvases. Capture only a visible scene.
  return[...$("geometry").querySelectorAll("canvas")].filter(c=>{
    const r=c.getBoundingClientRect();return r.width>0&&r.height>0&&getComputedStyle(c).visibility!=="hidden";
  }).sort((a,b)=>b.width*b.height-a.width*a.height)[0];
}
function lockRecordingControls(){
  recordLocked=[...document.querySelectorAll("input,select,button")].filter(n=>n.id!=="record-video").map(n=>[n,n.disabled]);
  recordLocked.forEach(([n])=>{n.disabled=true});document.body.classList.add("recording");
}
function unlockRecordingControls(){
  recordLocked.forEach(([n,disabled])=>{n.disabled=disabled});recordLocked=[];document.body.classList.remove("recording");
}
async function startRecording(){
  if(recording){recording.stop();return}
  const canvas=captureCanvas();
  if(!canvas){$("record-status").textContent="The 3D canvas is not ready. Wait for the event to load.";return}
  stop();const initialState={time:$("time").value,mode:$("time-mode").value};
  const eventId=event.event_id,maxIndex=Number($("time").max);
  try{
    recording=new LightHitViewerRecording.Recording({canvas,duration:Number($("record-duration").value),fps:Number($("record-fps").value),
      renderFrame:async progress=>{
        $("time-mode").value=initialState.mode==="integrated"?"frame":initialState.mode;
        // Temporal subsampling only: display existing stored bins, never new hits.
        $("time").value=Math.round(progress*maxIndex);await updateFrame();
      },
      onProgress:progress=>{$("record-status").textContent=`Recording · ${Math.round(progress*100)}% · keep this tab visible`},
      onFinished:({blob,mimeType,error})=>{
        recording=null;unlockRecordingControls();$("record-video").textContent="Record video";
        $("time").value=initialState.time;$("time-mode").value=initialState.mode;updateFrame();
        if(error){$("record-status").textContent=error.message;return}
        if(recordUrl)URL.revokeObjectURL(recordUrl);recordUrl=URL.createObjectURL(blob);
        const download=$("record-download");download.href=recordUrl;download.download=LightHitViewerRecording.filename(eventId,mimeType);download.hidden=false;
        recordBlob=blob;recordEventId=eventId;$("record-local-save").disabled=false;
        $("record-status").textContent=`Ready · ${(blob.size/1024/1024).toFixed(1)} MB · ${mimeType.split(";")[0]} · Save video`;
      }
    });
    lockRecordingControls();$("record-video").textContent="Stop recording";$("record-status").textContent="Starting video encoder…";
    await recording.start();
  }catch(error){recording=null;unlockRecordingControls();$("record-status").textContent=error.message}
}
$("record-video").onclick=startRecording;
async function changeSceneBounds(){
  stop();
  try{await geometry();await updateFrame()}catch(error){$("record-status").textContent=error.message}
}
$("scene-bounds").onchange=changeSceneBounds;$("scene-padding").onchange=changeSceneBounds;
$("record-local-save").onclick=async()=>{
  const button=$("record-local-save");button.disabled=true;$("record-status").textContent="Saving video to the local server…";
  try{
    const result=await LightHitViewerRecording.saveLocal(recordBlob,recordEventId);
    $("record-status").textContent=`Saved locally · ${result.saved_as}`;
  }catch(error){$("record-status").textContent=error.message}
  finally{button.disabled=!!recording||!recordBlob}
};
$("presentation").onchange=async()=>{
  stop();document.body.classList.toggle("presentation-view",$("presentation").checked);
  await Plotly.Plots.resize($("geometry"));await geometry();await updateFrame();
};
document.addEventListener("visibilitychange",()=>{if(document.hidden)recording?.stop()});
window.addEventListener("resize",()=>recording?.stop());
window.addEventListener("beforeunload",()=>{recording?.stop();if(recordUrl)URL.revokeObjectURL(recordUrl)});
$("event").onchange=selectEvent;$("cluster").onchange=async()=>{refreshVisible();await Promise.all([geometry(),waveform(),heatmap()])};$("part").onchange=selectEvent;$("display").onchange=changeDisplay;$("rebin").onchange=selectEvent;$("module").onchange=()=>choose(Number($("module").value));$("absolute").onchange=waveform;$("gain").oninput=updateFrame;$("time").oninput=()=>{stop();if($("time-mode").value==="integrated")$("time-mode").value="frame";updateFrame()};$("time-mode").onchange=()=>{stop();updateFrame()};$("clock").onchange=async()=>{stop();userClock=$("clock").value;setupClock();await geometry();updateFrame()};$("all-time").onclick=()=>{$("time-mode").value="integrated";stop();updateFrame()};$("play").onclick=()=>{if(playing){stop();return}if($("time-mode").value==="integrated")$("time-mode").value="frame";if(Number($("time").value)>=Number($("time").max))$("time").value=0;playing=true;last=0;$("play").textContent="Ⅱ Pause";requestAnimationFrame(tick)};document.addEventListener("visibilitychange",()=>{if(document.hidden)stop()});$("file").onchange=async e=>{try{await load(JSON.parse(await e.target.files[0].text()))}catch(err){$("status").textContent=err.message;$("status").classList.add("error")}};
load(JSON.parse($("result").textContent)).catch(err=>{$("status").textContent=err.message;$("status").classList.add("error")})
})();
