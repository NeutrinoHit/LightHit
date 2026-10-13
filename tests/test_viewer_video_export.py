"""Recording regressions with mocked browser APIs; no transport recomputation."""

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from lighthit.viewer import SCHEMA, write_event_viewer


ASSETS = Path(__file__).resolve().parents[1] / "src/lighthit/assets"


def small_payload():
    return {
        "schema": SCHEMA,
        "package_version": "test",
        "units": {"position": "m", "time": "ns", "signal": "m^-2"},
        "detector": {"label": "test", "positions_m": [[0.0, 0.0, 0.0]],
                     "cluster_id": [0], "string_id": [0], "module_id": [0]},
        "readout": {"relative_time_edges_ns": [0.0, 1.0, 2.0]},
        "events": [{"event_id": "signed", "label": "Signed source",
                    "time_origin_ns": [0.0], "active": [True],
                    "components": [[[0.1, -0.2, 0.3], [0.2, 0.3, -0.4]]],
                    "charge_components": [[0.3, 0.1, -0.1]], "sources": []}],
    }


def test_generated_viewer_embeds_recording_without_changing_data(tmp_path):
    value = small_payload()
    before = copy.deepcopy(value)
    output = write_event_viewer(value, tmp_path / "recordable.html")
    text = output.read_text()
    assert 'id="record-video"' in text
    assert 'id="record-duration"' in text
    assert 'id="record-fps"' in text
    assert 'id="presentation"' in text
    assert 'id="record-local-save"' in text
    assert 'id="scene-bounds"' in text
    assert 'id="scene-padding"' in text
    assert "captureStream(this.fps)" in text
    assert "LightHitViewerRecording" in text
    assert "__VIEWER_JS__" not in text
    assert value == before
    assert json.loads(output.with_suffix(".json").read_text()) == before


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js unavailable")
def test_recording_codec_lifecycle_cancellation_and_failure():
    # Execute only the DOM-independent recording helper from the same asset.
    # Real WebGL capture is a separate browser smoke test, not claimed here.
    source = (ASSETS / "event_viewer.js").read_text().split("\n(()=>{\nconst $=", 1)[0]
    test = r"""
const assert = require('node:assert/strict');
const {Recording, supportedTypes, filename, detectorBounds, localSaveEnabled, saveLocal} = LightHitViewerRecording;
function fixture({supported=['video/webm;codecs=vp8'], failFirst=false}={}) {
  let raf=0, timer=0, stopped=0, streamFps=null, result=null, frames=[];
  const rafs=new Map(), timers=new Map();
  class Recorder {
    static isTypeSupported(type) { return supported.includes(type); }
    constructor(stream, {mimeType}) {
      if(failFirst && mimeType===supported[0]) throw Error('Encoder busy');
      this.mimeType=mimeType; this.state='inactive';
    }
    start() {this.state='recording';}
    stop() {
      this.state='inactive';
      this.ondataavailable({data:new Blob(['frame'], {type:this.mimeType})});
      this.onstop();
    }
  }
  const stream={getTracks:()=>[{stop:()=>{stopped++;}}]};
  const env={MediaRecorder:Recorder, HTMLCanvasElement:{prototype:{captureStream(){}}},
    Blob, performance:{now:()=>0}, requestAnimationFrame:f=>{rafs.set(++raf,f); return raf;},
    cancelAnimationFrame:i=>rafs.delete(i), setTimeout:f=>{timers.set(++timer,f); return timer;},
    clearTimeout:i=>timers.delete(i)};
  const canvas={captureStream:fps=>{streamFps=fps; return stream;}};
  const recording=new Recording({canvas,duration:2,fps:30,env,
    renderFrame:async fraction=>{frames.push(fraction);},onFinished:r=>{result=r;}});
  return {recording,env,frames,rafs,timers,
    step:async now=>{const [id,fn]=rafs.entries().next().value;rafs.delete(id);await fn(now);},
    get result(){return result;},
    get stopped(){return stopped;},get fps(){return streamFps;}};
}
(async()=>{
  assert.deepEqual(supportedTypes({}),[]);
  const positions=[[-10,20,-100],[10,40,100],[0,30,0]],original=JSON.stringify(positions);
  assert.deepEqual(detectorBounds(positions,0.1),[[-12,12],[18,42],[-120,120]]);
  assert.deepEqual(detectorBounds(positions,0),[[-10,10],[20,40],[-100,100]]);
  for(const p of positions)detectorBounds(positions).forEach(([lo,hi],i)=>assert.ok(p[i]>=lo&&p[i]<=hi));
  assert.equal(JSON.stringify(positions),original);
  const flat=detectorBounds([[0,0,0],[0,0,100]]);
  assert.deepEqual(flat,[[-2.5,2.5],[-2.5,2.5],[-10,110]]);
  assert.deepEqual(detectorBounds([[1,2,3]],0),[[0.5,1.5],[1.5,2.5],[2.5,3.5]]);
  assert.throws(()=>detectorBounds([]),/finite 3D/);
  assert.throws(()=>detectorBounds([[0,NaN,1]]),/finite 3D/);
  assert.throws(()=>detectorBounds(positions,0.6),/padding/);
  assert.equal(filename('../../event <script>', 'video/mp4;codecs=avc1'), '-event-script-.mp4');
  assert.equal(filename('event', 'video/webm;codecs=vp9'), 'event.webm');
  const f=fixture(); await f.recording.start();
  assert.equal(f.fps,30); assert.equal(f.frames[0],0);
  await f.step(1000); assert.ok(f.frames.at(-1)>0.5);
  await f.step(1970); assert.equal(f.frames.at(-1),1);
  f.recording.stop();
  assert.ok(f.result.blob.size>0); assert.equal(f.result.mimeType,'video/webm;codecs=vp8');
  assert.equal(f.result.error,null); assert.equal(f.stopped,1);
  assert.equal(f.timers.size,0); assert.equal(f.rafs.size,0);
  f.recording.stop(); assert.equal(f.stopped,1);
  const fallback=fixture({supported:['video/mp4','video/webm'],failFirst:true});
  await fallback.recording.start(); fallback.recording.stop();
  assert.equal(fallback.result.mimeType,'video/webm');
  const timed=fixture(); await timed.recording.start(); [...timed.timers.values()][0]();
  assert.ok(timed.result.blob); assert.equal(timed.stopped,1);
  const unsupported=fixture({supported:[]}); await unsupported.recording.start();
  assert.match(unsupported.result.error.message,/unavailable/); assert.equal(unsupported.result.blob,null);
  const error=fixture(); await error.recording.start();
  error.recording.recorder.onerror({error:Error('Encoder failed')});
  assert.equal(error.result.blob,null); assert.match(error.result.error.message,/Encoder failed/);
  assert.equal(error.stopped,1);
  const cancelled=fixture(); let release;
  cancelled.recording.renderFrame=()=>new Promise(resolve=>{release=resolve;});
  const pending=cancelled.recording.start(); cancelled.recording.stop(); release(); await pending;
  assert.equal(cancelled.fps,null); assert.equal(cancelled.result.blob,null);
  assert.throws(()=>new Recording({canvas:{},duration:0,fps:30}),/duration/);
  assert.throws(()=>new Recording({canvas:{},duration:2,fps:Infinity}),/frame rate/);
  const local={href:'http://127.0.0.1:8784/viewer.html?local_save=1'};
  assert.equal(localSaveEnabled(local),true);
  assert.equal(localSaveEnabled({href:'https://localhost/viewer?local_save=1'}),true);
  assert.equal(localSaveEnabled({href:'http://[::1]:8784/?local_save=1'}),true);
  for(const href of ['https://neutrinohit.github.io/?local_save=1',
    'http://localhost.example.org/?local_save=1','http://192.168.1.2/?local_save=1',
    'file:///tmp/viewer.html?local_save=1','http://localhost/viewer.html',
    'http://127.0.0.1/?local_save=0']) assert.equal(localSaveEnabled({href}),false);
  const blob=new Blob(['original video'],{type:'video/mp4'});
  let requests=[];
  const fetch=async(url,options)=>{
    requests.push([url,options]);return{ok:true,json:async()=>({saved_as:'event.mp4'})};
  };
  assert.deepEqual(await saveLocal(blob,'event',{location:local,fetch}),{saved_as:'event.mp4'});
  assert.equal(requests[0][0],'http://127.0.0.1:8784/recordings/event.mp4');
  assert.equal(requests[0][1].body,blob);assert.equal(requests[0][1].mode,'same-origin');
  assert.equal(requests[0][1].redirect,'error');assert.equal(requests[0][1].credentials,'omit');
  assert.equal(requests[0][1].headers['Content-Type'],'video/mp4');
  const before=await blob.text();
  await assert.rejects(()=>saveLocal(blob,'event',{location:{href:'https://example.org/?local_save=1'},fetch}),/localhost/);
  assert.equal(requests.length,1);assert.equal(await blob.text(),before);
  await assert.rejects(()=>saveLocal(blob,'event',{location:local,fetch:async()=>({ok:false,status:404})}),/HTTP 404/);
  await assert.rejects(()=>saveLocal(blob,'event',{location:local,fetch:async()=>({ok:true,json:async()=>({saved_as:'../wrong.mp4'})})}),/unexpected filename/);
  await assert.rejects(()=>saveLocal(blob,'event',{location:local,fetch:async()=>{throw Error('Endpoint offline');}}),/Endpoint offline/);
  assert.equal(await blob.text(),before);
  console.log('codec, lifecycle, duration, cancellation and error checks passed');
})().catch(error=>{console.error(error);process.exit(1);});
"""
    result = subprocess.run([shutil.which("node"), "-e", source + "\n" + test],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
