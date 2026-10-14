"""Frame-by-frame movie export (studio_movie.js): MP4 muxer and WebCodecs driver.

The muxer is checked structurally and by a real ffmpeg decode; the encoder
driver runs in Node against a mocked WebCodecs environment.
"""

import base64
import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest


MOVIE = Path(__file__).resolve().parents[1] / "src/lighthit/assets/studio_movie.js"
NODE = shutil.which("node")
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is unavailable")
needs_ffmpeg = pytest.mark.skipif(NODE is None or FFMPEG is None or FFPROBE is None,
                                  reason="Node.js or ffmpeg/ffprobe is unavailable")

CONTAINERS = {"moov", "trak", "mdia", "minf", "dinf", "stbl"}
# Bytes between the box header and the first child of these sample-entry boxes.
SKIP = {"stsd": 8, "dref": 8, "avc1": 78}

MUX_SCRIPT = """
const M = require(%s);
const input = JSON.parse(require("fs").readFileSync(0, "utf8"));
const b = s => new Uint8Array(Buffer.from(s, "base64"));
const bytes = M.muxMp4({width: input.width, height: input.height, fps: input.fps,
                        avcC: b(input.avcC), samples: input.samples.map(s => ({data: b(s.data), key: s.key}))});
require("fs").writeFileSync(process.argv[1], bytes);
""" % json.dumps(str(MOVIE))


def node(script, *args, stdin=""):
    result = subprocess.run([NODE, "-e", script, *args], input=stdin, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout


def mux(path, width, height, fps, avcc, samples):
    payload = {"width": width, "height": height, "fps": fps,
               "avcC": base64.b64encode(avcc).decode(),
               "samples": [{"data": base64.b64encode(d).decode(), "key": k} for d, k in samples]}
    node(MUX_SCRIPT, str(path), stdin=json.dumps(payload))
    return path.read_bytes()


def boxes(data, start=0, end=None, prefix=""):
    """Map 'moov/trak/...' paths to (box_start, body_start, body_end), in file order."""
    out = {}
    end = len(data) if end is None else end
    at = start
    while at < end:
        size, kind = struct.unpack(">I4s", data[at:at + 8])
        kind = kind.decode("latin-1")
        header = 8
        if size == 1:
            size = struct.unpack(">Q", data[at + 8:at + 16])[0]
            header = 16
        assert size >= header and at + size <= end, (kind, size)
        path = prefix + kind
        out[path] = (at, at + header, at + size)
        if kind in CONTAINERS or kind in SKIP:
            out.update(boxes(data, at + header + SKIP.get(kind, 0), at + size, path + "/"))
        at += size
    assert at == end
    return out


def body(data, tree, path):
    _, lo, hi = tree[path]
    return data[lo:hi]


STBL = "moov/trak/mdia/minf/stbl/"


def summary(data):
    tree = boxes(data)
    top = [p for p in tree if "/" not in p]
    mvhd = body(data, tree, "moov/mvhd")
    tkhd = body(data, tree, "moov/trak/tkhd")
    mdhd = body(data, tree, "moov/trak/mdia/mdhd")
    stsz = body(data, tree, STBL + "stsz")
    count = struct.unpack(">I", stsz[8:12])[0]
    stss = body(data, tree, STBL + "stss")
    stco = body(data, tree, STBL + "stco")
    stts = body(data, tree, STBL + "stts")
    return {
        "top": top, "tree": tree,
        "mvhd": struct.unpack(">II", mvhd[12:20]),
        "tkhd_flags": int.from_bytes(tkhd[1:4], "big"),
        "tkhd_duration": struct.unpack(">I", tkhd[20:24])[0],
        "tkhd_size": struct.unpack(">II", tkhd[76:84]),
        "mdhd": struct.unpack(">II", mdhd[12:20]),
        "sizes": list(struct.unpack(f">{count}I", stsz[12:12 + 4 * count])),
        "keys": list(struct.unpack(f">{struct.unpack('>I', stss[4:8])[0]}I", stss[8:])),
        "stts": struct.unpack(">III", stts[4:16]),
        "offset": struct.unpack(">I", stco[8:12])[0],
        "avcC": body(data, tree, STBL + "stsd/avc1/avcC"),
        "ftyp": body(data, tree, "ftyp"),
    }


@needs_node
@pytest.mark.parametrize("fps,timescale,delta", [(30, 90000, 3000), (60, 90000, 1500),
                                                 (24, 90000, 3750), (25, 90000, 3600),
                                                 (29.97, 29970, 1000), (7, 7000, 1000)])
def test_mux_box_tree_tables_and_offsets(tmp_path, fps, timescale, delta):
    avcc = bytes([1, 0x64, 0, 0x28, 0xFF, 0xE1, 0, 3, 0x67, 1, 2, 1, 0, 2, 0x68, 3])
    samples = [(bytes([i]) * (20 + 7 * i), i % 4 == 0) for i in range(10)]
    data = mux(tmp_path / "s.mp4", 64, 48, fps, avcc, samples)
    s = summary(data)
    assert s["top"] == ["ftyp", "moov", "mdat"]
    assert s["ftyp"] == b"isom" + struct.pack(">I", 0x200) + b"isomiso2avc1mp41"
    assert s["sizes"] == [len(d) for d, _ in samples]
    assert s["keys"] == [1, 5, 9]
    assert s["stts"] == (1, 10, delta)
    assert s["avcC"] == avcc
    assert s["tkhd_flags"] == 3 and s["tkhd_size"] == (64 << 16, 48 << 16)
    # Durations agree everywhere and equal n / fps.
    assert s["mvhd"] == s["mdhd"] == (timescale, 10 * delta)
    assert s["tkhd_duration"] == 10 * delta
    assert s["mvhd"][1] / s["mvhd"][0] == pytest.approx(10 / fps)
    # The single chunk starts at the mdat payload and holds all samples in order.
    _, mdat_body, mdat_end = s["tree"]["mdat"]
    assert s["offset"] == mdat_body and mdat_end == len(data)
    assert data[mdat_body:] == b"".join(d for d, _ in samples)
    assert STBL + "stsd/avc1/pasp" in s["tree"]
    assert "moov/trak/mdia/minf/dinf/dref/url " in s["tree"]


@needs_node
def test_mux_rejects_bad_input():
    script = """
const M = require(%s);
const ok = {width: 64, height: 48, fps: 30, avcC: new Uint8Array([1, 66, 0, 30]),
            samples: [{data: new Uint8Array(4), key: true}]};
const errors = [{...ok, samples: []}, {...ok, avcC: null}, {...ok, width: 63},
                {...ok, samples: [{data: new Uint8Array(4), key: false}]}].map(o => {
  try {M.muxMp4(o); return null} catch (e) {return e.message}
});
console.log(JSON.stringify({errors, n: M.muxMp4(ok).length}));
""" % json.dumps(str(MOVIE))
    out = json.loads(node(script))
    assert all(out["errors"]) and out["n"] > 0
    assert "even" in out["errors"][2]


def annexb_access_units(stream):
    """Split an Annex-B stream (encoded with AUDs) into NAL-unit lists per frame."""
    parts = stream.split(b"\x00\x00\x01")
    nals = [p.rstrip(b"\x00") for p in parts[1:]]
    units, current = [], []
    for nal in nals:
        if nal[0] & 0x1F == 9:  # access unit delimiter starts a new frame
            if current:
                units.append(current)
            current = []
        else:
            current.append(nal)
    units.append(current)
    return units


def ffmpeg_samples(tmp_path, fps, frames=64):
    h264 = tmp_path / "pattern.h264"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc2=size=320x180:rate={fps}", "-frames:v", str(frames),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0", "-g", "16",
                    "-x264-params", "aud=1", "-f", "h264", str(h264)], check=True)
    units = annexb_access_units(h264.read_bytes())
    assert len(units) == frames
    sps = next(n for u in units for n in u if n[0] & 0x1F == 7)
    pps = next(n for u in units for n in u if n[0] & 0x1F == 8)
    # AVCDecoderConfigurationRecord for 4:2:0, 8-bit High profile.
    avcc = (bytes([1, sps[1], sps[2], sps[3], 0xFF, 0xE1]) + struct.pack(">H", len(sps)) + sps
            + bytes([1]) + struct.pack(">H", len(pps)) + pps + bytes([0xFD, 0xF8, 0xF8, 0]))
    samples = []
    for unit in units:
        vcl = [n for n in unit if n[0] & 0x1F not in (7, 8)]
        data = b"".join(struct.pack(">I", len(n)) + n for n in vcl)
        samples.append((data, any(n[0] & 0x1F == 5 for n in unit)))
    return avcc, samples


@needs_ffmpeg
@pytest.mark.parametrize("fps", [30, 60])
def test_mux_real_h264_decodes_with_ffmpeg(tmp_path, fps):
    avcc, samples = ffmpeg_samples(tmp_path, fps)
    assert [i for i, (_, k) in enumerate(samples) if k] == [0, 16, 32, 48]
    out = tmp_path / "out.mp4"
    mux(out, 320, 180, fps, avcc, samples)
    probe = json.loads(subprocess.run(
        [FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,nb_read_frames,duration,r_frame_rate",
         "-show_entries", "format=duration", "-of", "json", str(out)],
        capture_output=True, text=True, check=True).stdout)
    stream = probe["streams"][0]
    assert stream["codec_name"] == "h264"
    assert (stream["width"], stream["height"]) == (320, 180)
    assert int(stream["nb_read_frames"]) == 64
    assert stream["r_frame_rate"] == f"{fps}/1"
    assert float(stream["duration"]) == pytest.approx(64 / fps, abs=1e-3)
    assert float(probe["format"]["duration"]) == pytest.approx(64 / fps, abs=1e-3)
    decode = subprocess.run([FFMPEG, "-v", "error", "-i", str(out), "-f", "null", "-"],
                            capture_output=True, text=True)
    assert decode.returncode == 0 and decode.stderr.strip() == ""


# Mock WebCodecs: an encoder that drains one frame per millisecond, so a tight
# addFrame loop would overrun it without backpressure.
MOCK_ENV = r"""
const M = require(%s);
function makeEnv({supportedCodecs = null, softwareOnly = false, failAt = -1} = {}) {
  const log = {probes: [], encoded: [], frames: [], maxQueue: 0, closed: 0};
  class VideoFrame {
    constructor(source, init) {Object.assign(this, init, {source}); this.isClosed = false; log.frames.push(this)}
    close() {this.isClosed = true}
  }
  class VideoEncoder {
    static async isConfigSupported(config) {
      log.probes.push(config.codec + (config.hardwareAcceleration ? "/sw" : ""));
      const okCodec = !supportedCodecs || supportedCodecs.includes(config.codec);
      const okAccel = !softwareOnly || config.hardwareAcceleration === "prefer-software";
      return {supported: okCodec && okAccel, config};
    }
    constructor({output, error}) {
      Object.assign(this, {output, error, state: "unconfigured", encodeQueueSize: 0, queue: [], listeners: []});
      log.encoder = this;
    }
    configure(config) {this.config = config; this.state = "configured"}
    addEventListener(type, fn) {if (type === "dequeue") this.listeners.push(fn)}
    removeEventListener(type, fn) {this.listeners = this.listeners.filter(f => f !== fn)}
    encode(frame, {keyFrame}) {
      if (this.state !== "configured") throw Error("encode on " + this.state + " encoder");
      log.encoded.push({timestamp: frame.timestamp, duration: frame.duration, keyFrame});
      this.queue.push({timestamp: frame.timestamp, keyFrame});
      this.encodeQueueSize++;
      log.maxQueue = Math.max(log.maxQueue, this.encodeQueueSize);
      if (!this.timer) this.timer = setTimeout(() => this.drain(), 1);
    }
    drain() {
      this.timer = null;
      const item = this.queue.shift();
      if (!item || this.state === "closed") return;
      this.encodeQueueSize--;
      const index = log.encoded.length - this.queue.length - 1;
      if (index === failAt) {this.state = "closed"; this.error(Error("mock encoder exploded")); return}
      const bytes = new Uint8Array([0, 0, 0, 2, item.keyFrame ? 0x65 : 0x41, index & 255]);
      const chunk = {type: item.keyFrame ? "key" : "delta", timestamp: item.timestamp,
                     byteLength: bytes.length, copyTo: dst => dst.set(bytes)};
      const meta = index === 0 ? {decoderConfig: {description: new Uint8Array([1, 0x64, 0, 0x28, 0xff, 0xe0, 0]).buffer}} : undefined;
      this.output(chunk, meta);
      for (const fn of [...this.listeners]) fn();
      if (this.queue.length) this.timer = setTimeout(() => this.drain(), 1);
    }
    async flush() {
      while (this.queue.length) await new Promise(r => setTimeout(r, 1));
      if (this.state === "closed") throw Error("flush on closed encoder");
    }
    close() {this.state = "closed"; log.closed++}
  }
  class Blob {
    constructor(parts, {type}) {
      this.type = type; this.size = parts.reduce((n, p) => n + p.length, 0);
      this.bytes = Buffer.concat(parts.map(p => Buffer.from(p)));
    }
  }
  return {env: {VideoEncoder, VideoFrame, Blob}, log};
}
const canvas = (width, height) => ({width, height});
const message = async f => {try {await f(); return null} catch (e) {return e.message}};
""" % json.dumps(str(MOVIE))


def run_mock(body):
    return json.loads(node(MOCK_ENV + "(async () => {" + body + "})().catch(e => {console.error(e); process.exit(1)});"))


@needs_node
def test_encoder_keyframes_timestamps_and_backpressure():
    out = run_mock("""
const {env, log} = makeEnv();
const movie = new M.MovieEncoder({width: 64, height: 48, fps: 30, env});
await movie.open();
for (let i = 0; i < 40; i++) await movie.addFrame(canvas(64, 48));
const frames = movie.frames, codec = movie.config.codec;
const blob = await movie.finish();
console.log(JSON.stringify({frames, codec, encoded: log.encoded, maxQueue: log.maxQueue,
  allClosed: log.frames.every(f => f.isClosed), state: log.encoder.state, type: blob.type,
  mp4: blob.bytes.toString("base64"), after: await message(() => movie.addFrame(canvas(64, 48)))}));
""")
    assert out["frames"] == 40 and out["codec"].startswith("avc1.64")
    assert [e["keyFrame"] for e in out["encoded"]] == [i % 60 == 0 for i in range(40)]
    stamps = [e["timestamp"] for e in out["encoded"]]
    assert stamps == [round(i * 1e6 / 30) for i in range(40)]
    assert all(e["duration"] == round(1e6 / 30) for e in out["encoded"])
    assert out["maxQueue"] == 5  # queue reached the limit and was held there
    assert out["allClosed"] and out["state"] == "closed" and out["type"] == "video/mp4"
    assert out["after"]
    s = summary(base64.b64decode(out["mp4"]))
    assert len(s["sizes"]) == 40 and s["keys"] == [1]
    assert s["avcC"] == bytes([1, 0x64, 0, 0x28, 0xFF, 0xE0, 0])


@needs_node
def test_encoder_key_interval_errors_and_cancel():
    out = run_mock("""
const a = makeEnv();
const short = new M.MovieEncoder({width: 64, height: 48, fps: 30, keyInterval: 7, env: a.env});
await short.open();
for (let i = 0; i < 15; i++) await short.addFrame(canvas(64, 48));
const keys = a.log.encoded.map(e => e.keyFrame);
const wrongSize = await message(() => short.addFrame(canvas(64, 50)));
const mp4 = (await short.finish()).bytes.toString("base64");

const b = makeEnv({failAt: 3});
const failing = new M.MovieEncoder({width: 64, height: 48, fps: 30, env: b.env});
await failing.open();
let failure = null;
for (let i = 0; i < 20 && !failure; i++) {
  failure = await message(() => failing.addFrame(canvas(64, 48)));
  await new Promise(r => setTimeout(r, 3));
}
const finishFailure = await message(() => failing.finish());

const c = makeEnv();
const cancelled = new M.MovieEncoder({width: 64, height: 48, fps: 30, env: c.env});
await cancelled.open();
await cancelled.addFrame(canvas(64, 48));
cancelled.cancel();
const afterCancel = [await message(() => cancelled.addFrame(canvas(64, 48))),
                     await message(() => cancelled.finish())];

const none = makeEnv({supportedCodecs: []});
const unsupported = await message(() => new M.MovieEncoder({width: 64, height: 48, fps: 30, env: none.env}).open());
const missing = await message(() => new M.MovieEncoder({width: 64, height: 48, fps: 30, env: {}}).open());
const odd = await message(() => new M.MovieEncoder({width: 65, height: 48, fps: 30, env: a.env}));
console.log(JSON.stringify({keys, wrongSize, mp4, failure, finishFailure, afterCancel,
  cancelClosed: c.log.encoder.state, unsupported, missing, odd}));
""")
    assert out["keys"] == [i % 7 == 0 for i in range(15)]
    assert "64x48" in out["wrongSize"]
    assert summary(base64.b64decode(out["mp4"]))["keys"] == [1, 8, 15]
    assert "mock encoder exploded" in out["failure"]
    assert "mock encoder exploded" in out["finishFailure"]
    assert all("cancelled" in m for m in out["afterCancel"])
    assert out["cancelClosed"] == "closed"
    assert "cannot encode H.264" in out["unsupported"]
    assert "WebCodecs" in out["missing"]
    assert "even" in out["odd"]


@needs_node
def test_choose_config_fallback_order():
    out = run_mock("""
const hw = makeEnv({supportedCodecs: ["avc1.4d0028"]});
const main = await M.chooseConfig({width: 1920, height: 1080, fps: 30, bitrate: 12e6}, hw.env);
const sw = makeEnv({softwareOnly: true, supportedCodecs: ["avc1.42e01f"]});
const soft = await M.chooseConfig({width: 1280, height: 720, fps: 30, bitrate: 8e6}, sw.env);
const uhd = M.codecCandidates(3840, 2160, 60, M.defaultBitrate(3840, 2160, 60));
const odd = await message(() => M.chooseConfig({width: 1921, height: 1080, fps: 30}, hw.env));
console.log(JSON.stringify({main, mainProbes: hw.log.probes, soft, softProbes: sw.log.probes, uhd, odd,
  rates: [M.defaultBitrate(320, 180, 30), M.defaultBitrate(1920, 1080, 60), M.defaultBitrate(7680, 4320, 60)],
  supported: [M.supported({}), M.supported(makeEnv().env)]}));
""")
    assert out["main"]["codec"] == "avc1.4d0028"
    assert "hardwareAcceleration" not in out["main"]
    assert out["main"] | {"codec": None} == {
        "codec": None, "width": 1920, "height": 1080, "framerate": 30, "bitrate": 12e6,
        "bitrateMode": "variable", "latencyMode": "quality", "avc": {"format": "avc"}}
    # High profile, smallest fitting level first, before Main.
    assert out["mainProbes"][:3] == ["avc1.640028", "avc1.640029", "avc1.64002a"]
    assert out["mainProbes"][-1] == "avc1.4d0028"
    assert out["soft"]["codec"] == "avc1.42e01f"
    assert out["soft"]["hardwareAcceleration"] == "prefer-software"
    hardware = [p for p in out["softProbes"] if not p.endswith("/sw")]
    assert out["softProbes"][:len(hardware)] == hardware  # every hardware try comes first
    assert out["uhd"][0] == "avc1.640034" and "avc1.640033" not in out["uhd"]
    assert "even" in out["odd"]
    assert out["rates"] == [4e6, round(0.2 * 1920 * 1080 * 60), 120e6]
    assert out["supported"] == [False, True]


@needs_node
def test_filename_local_save_and_download():
    out = run_mock("""
const loc = href => ({href});
const enabled = ["http://localhost:8000/v.html?local_save=1", "http://127.0.0.1/v.html?local_save=1",
                 "http://[::1]:9/v.html?local_save=1", "http://localhost/v.html",
                 "https://example.org/v.html?local_save=1", "file:///v.html?local_save=1"]
  .map(h => M.localSaveEnabled(loc(h)));
const names = [M.filename("ν event #12/a"), M.filename(""), M.filename("x".repeat(100), "png")];
const calls = [];
const fetch = async (url, init) => {calls.push({url, init: {...init, body: init.body.type}});
  return {ok: true, json: async () => ({saved_as: decodeURIComponent(url.split("/").pop())})}};
const location = loc("http://localhost:8000/v.html?local_save=1");
const saved = await M.saveLocal({size: 10, type: "video/mp4"}, "event-1.mp4", {location, fetch});
const png = await M.saveLocal({size: 10, type: "image/png"}, "event-1.png", {location, fetch});
const refusals = [
  await message(() => M.saveLocal({size: 10, type: "video/mp4"}, "a.mp4", {location: loc("http://localhost/"), fetch})),
  await message(() => M.saveLocal({size: 0, type: "video/mp4"}, "a.mp4", {location, fetch})),
  await message(() => M.saveLocal({size: 5, type: "text/html"}, "a.html", {location, fetch})),
  await message(() => M.saveLocal({size: 5, type: "video/mp4"}, "../a.mp4", {location, fetch})),
  await message(() => M.saveLocal({size: 5, type: "video/mp4"}, "a.mp4", {location,
    fetch: async () => ({ok: true, json: async () => ({saved_as: "b.mp4"})})})),
  await message(() => M.saveLocal({size: 5, type: "video/mp4"}, "a.mp4", {location,
    fetch: async () => ({ok: false, status: 500})})),
];
const clicked = [];
const doc = {body: {appendChild: a => clicked.push(["append", a.download])},
  defaultView: {URL: {createObjectURL: b => "blob:fake/" + b.type}},
  createElement: tag => ({tag, style: {}, click() {clicked.push(["click", this.href])}, remove() {clicked.push(["remove"])}})};
const url = M.downloadBlob({type: "video/mp4"}, "movie.mp4", doc);
console.log(JSON.stringify({enabled, names, saved, png, calls, refusals, url, clicked}));
""")
    assert out["enabled"] == [True, True, True, False, False, False]
    assert out["names"] == ["-event-12-a.mp4", "event.mp4", "x" * 80 + ".png"]
    assert out["saved"] == {"saved_as": "event-1.mp4"} and out["png"] == {"saved_as": "event-1.png"}
    first = out["calls"][0]
    assert first["url"] == "http://localhost:8000/recordings/event-1.mp4"
    assert first["init"] == {"method": "POST", "body": "video/mp4", "headers": {"Content-Type": "video/mp4"},
                             "mode": "same-origin", "credentials": "omit", "redirect": "error"}
    assert out["calls"][1]["init"]["headers"] == {"Content-Type": "image/png"}
    assert all(out["refusals"])
    assert "HTTP 500" in out["refusals"][-1] and "unexpected filename" in out["refusals"][-2]
    assert out["url"] == "blob:fake/video/mp4"
    assert out["clicked"] == [["append", "movie.mp4"], ["click", "blob:fake/video/mp4"], ["remove"]]
