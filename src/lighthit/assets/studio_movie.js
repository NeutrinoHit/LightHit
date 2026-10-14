"use strict";

// Deterministic movie export: the app renders frame i exactly, hands the
// canvas to MovieEncoder, and receives a finished H.264 MP4 Blob.  Nothing is
// captured in real time, so no frame can be dropped.  WebCodecs encodes; the
// small ISO BMFF muxer below writes the file.  Plain script, no dependencies.
const LightHitMovie = (() => {
  const U32 = 0x100000000;

  function supported(env = globalThis) {
    return !!env && typeof env.VideoEncoder === "function" && typeof env.VideoFrame === "function";
  }

  function checkSize(width, height) {
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 2 || height < 2)
      throw Error(`Movie size must be positive integers, got ${width}x${height}.`);
    if (width % 2 || height % 2)
      throw Error(`H.264 needs even movie dimensions, got ${width}x${height}.`);
    if (width > 65535 || height > 65535) throw Error(`Movie size ${width}x${height} is too large.`);
  }

  function checkFps(fps) {
    if (!Number.isFinite(fps) || fps <= 0 || fps > 240) throw Error(`Choose a frame rate between 0 and 240 FPS, got ${fps}.`);
  }

  // ~0.2 bits per pixel per frame: generous enough for sharp points and text.
  function defaultBitrate(width, height, fps) {
    return Math.round(Math.min(120e6, Math.max(4e6, 0.2 * width * height * fps)));
  }

  // H.264 levels (Table A-1): [hex idc, max macroblocks/frame, max MB/s,
  // max bitrate in kbit/s for Baseline/Main; High allows 1.25x].
  const LEVELS = [
    ["1e", 1620, 40500, 10000], ["1f", 3600, 108000, 14000], ["20", 5120, 216000, 20000],
    ["28", 8192, 245760, 20000], ["29", 8192, 245760, 50000], ["2a", 8704, 522240, 50000],
    ["32", 22080, 589824, 135000], ["33", 36864, 983040, 240000], ["34", 36864, 2073600, 240000],
    ["3c", 139264, 4177920, 240000], ["3d", 139264, 8355840, 480000], ["3e", 139264, 16711680, 800000],
  ];
  // High, Main, Constrained Baseline.
  const PROFILES = [["6400", 1.25], ["4d00", 1], ["42e0", 1]];

  // Codec strings from high to low profile.  Within a profile the smallest
  // level that fits size, rate and bitrate comes first (most widely decodable),
  // then larger ones; if nothing fits we still try the largest level.
  function codecCandidates(width, height, fps, bitrate) {
    const mbs = Math.ceil(width / 16) * Math.ceil(height / 16), side = Math.max(width, height) / 16;
    const out = [];
    for (const [profile, brFactor] of PROFILES) {
      const fits = LEVELS.filter(([, fs, mbps, br]) =>
        mbs <= fs && side <= Math.sqrt(8 * fs) && mbs * fps <= mbps && bitrate <= br * 1000 * brFactor);
      for (const [idc] of fits.length ? fits : LEVELS.slice(-1)) out.push(`avc1.${profile}${idc}`);
    }
    return out;
  }

  async function chooseConfig({width, height, fps, bitrate}, env = globalThis) {
    checkSize(width, height);
    checkFps(fps);
    bitrate = bitrate || defaultBitrate(width, height, fps);
    if (!supported(env) || typeof env.VideoEncoder.isConfigSupported !== "function") return null;
    const codecs = codecCandidates(width, height, fps, bitrate);
    // First let the browser pick (usually hardware), then insist on software.
    for (const extra of [{}, {hardwareAcceleration: "prefer-software"}]) {
      for (const codec of codecs) {
        const config = {codec, width, height, framerate: fps, bitrate, bitrateMode: "variable",
                        latencyMode: "quality", avc: {format: "avc"}, ...extra};
        try {
          const result = await env.VideoEncoder.isConfigSupported(config);
          if (result && result.supported) return config;
        } catch (_) {/* Invalid for this browser: try the next one. */}
      }
    }
    return null;
  }

  // ---- ISO BMFF muxer ------------------------------------------------------

  const ascii = s => Array.from(s, c => c.charCodeAt(0));
  const u16 = v => [(v >>> 8) & 255, v & 255];
  const u32 = v => [(v >>> 24) & 255, (v >>> 16) & 255, (v >>> 8) & 255, v & 255];
  const u64 = v => [...u32(Math.floor(v / U32)), ...u32(v % U32)];

  // A box from its type and body parts (byte arrays or Uint8Arrays).
  function box(type, ...parts) {
    const size = 8 + parts.reduce((n, p) => n + p.length, 0);
    const out = new Uint8Array(size);
    out.set([...u32(size), ...ascii(type)]);
    let at = 8;
    for (const p of parts) {out.set(p, at); at += p.length}
    return out;
  }
  const fullBox = (type, version, flags, ...parts) =>
    box(type, [version, (flags >>> 16) & 255, (flags >>> 8) & 255, flags & 255], ...parts);

  const MATRIX = [0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000].flatMap(u32);
  const zeros = n => new Array(n).fill(0);

  function timing(fps, count) {
    // Exact integer ticks per frame: 90 kHz when it divides, else fps*1000.
    const timescale = 90000 % fps === 0 ? 90000 : Math.round(fps * 1000);
    const delta = 90000 % fps === 0 ? 90000 / fps : Math.round(timescale / fps);
    const duration = delta * count;
    if (duration >= U32) throw Error("Movie is too long for 32-bit durations.");
    return {timescale, delta, duration};
  }

  function moovBox({width, height, avcC, samples, timescale, delta, duration}, offset) {
    const n = samples.length;
    const mvhd = fullBox("mvhd", 0, 0, u32(0), u32(0), u32(timescale), u32(duration),
                         u32(0x10000), u16(0x100), zeros(10), MATRIX, zeros(24), u32(2));
    const tkhd = fullBox("tkhd", 0, 3, u32(0), u32(0), u32(1), u32(0), u32(duration), zeros(8),
                         u16(0), u16(0), u16(0), u16(0), MATRIX, u32(width * 65536), u32(height * 65536));
    const mdhd = fullBox("mdhd", 0, 0, u32(0), u32(0), u32(timescale), u32(duration), u16(0x55c4), u16(0));
    const hdlr = fullBox("hdlr", 0, 0, u32(0), ascii("vide"), zeros(12), ascii("VideoHandler\0"));
    const vmhd = fullBox("vmhd", 0, 1, zeros(8));
    const dinf = box("dinf", fullBox("dref", 0, 0, u32(1), fullBox("url ", 0, 1)));
    const avc1 = box("avc1", zeros(6), u16(1), zeros(16), u16(width), u16(height),
                     u32(0x480000), u32(0x480000), u32(0), u16(1), zeros(32), u16(0x18), u16(0xffff),
                     box("avcC", avcC), box("pasp", u32(1), u32(1)));
    const stsd = fullBox("stsd", 0, 0, u32(1), avc1);
    const stts = fullBox("stts", 0, 0, u32(1), u32(n), u32(delta));
    const keys = [];
    samples.forEach((s, i) => {if (s.key) keys.push(...u32(i + 1))});
    const stss = fullBox("stss", 0, 0, u32(keys.length / 4), keys);
    const stsc = fullBox("stsc", 0, 0, u32(1), u32(1), u32(n), u32(1));
    const sizes = new Uint8Array(4 * n), view = new DataView(sizes.buffer);
    samples.forEach((s, i) => view.setUint32(4 * i, s.data.length));
    const stsz = fullBox("stsz", 0, 0, u32(0), u32(n), sizes);
    // All samples sit in one chunk at `offset`.
    const stco = offset >= U32 ? fullBox("co64", 0, 0, u32(1), u64(offset))
                               : fullBox("stco", 0, 0, u32(1), u32(offset));
    const stbl = box("stbl", stsd, stts, stss, stsc, stsz, stco);
    const mdia = box("mdia", mdhd, hdlr, box("minf", vmhd, dinf, stbl));
    return box("moov", mvhd, box("trak", tkhd, mdia));
  }

  // The file as a list of parts (header, then the samples themselves), so
  // that a large movie is not copied once more before it becomes a Blob.
  function muxParts({width, height, fps, avcC, samples}) {
    checkSize(width, height);
    checkFps(fps);
    if (!avcC || !avcC.length) throw Error("Missing avcC (H.264 decoder configuration).");
    avcC = new Uint8Array(avcC.buffer ? avcC.buffer.slice(avcC.byteOffset, avcC.byteOffset + avcC.byteLength) : avcC);
    if (avcC[0] !== 1) throw Error("avcC must start with configurationVersion 1.");
    if (!samples || !samples.length) throw Error("No encoded frames to write.");
    if (!samples[0].key) throw Error("The first frame must be a key frame.");
    const info = {width, height, avcC, samples, ...timing(fps, samples.length)};
    const ftyp = box("ftyp", ascii("isom"), u32(0x200), ascii("isomiso2avc1mp41"));
    const payload = samples.reduce((n, s) => n + s.data.length, 0);
    const large = payload + 8 >= U32;
    const mdatHeader = large ? new Uint8Array([...u32(1), ...ascii("mdat"), ...u64(payload + 16)])
                             : new Uint8Array([...u32(payload + 8), ...ascii("mdat")]);
    // Fast start: moov precedes mdat, so its size fixes the chunk offset.
    let moov = moovBox(info, 0);
    let offset = ftyp.length + moov.length + mdatHeader.length;
    if (offset >= U32) {moov = moovBox(info, U32); offset = ftyp.length + moov.length + mdatHeader.length}
    moov = moovBox(info, offset);
    return [ftyp, moov, mdatHeader, ...samples.map(s => s.data)];
  }

  // Samples are AVCC (length-prefixed) access units in decode order.  No
  // B-frames are assumed: decode order equals presentation order, so there is
  // no ctts box and every sample lasts exactly 1/fps.
  function muxMp4(options) {
    const parts = muxParts(options);
    const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const p of parts) {out.set(p, at); at += p.length}
    return out;
  }

  // ---- Encoder -------------------------------------------------------------

  const QUEUE_LIMIT = 4;

  class MovieEncoder {
    constructor({width, height, fps, bitrate, keyInterval = 2 * fps, env = globalThis}) {
      checkSize(width, height);
      checkFps(fps);
      Object.assign(this, {width, height, fps, env,
                           bitrate: bitrate || defaultBitrate(width, height, fps),
                           keyInterval: Math.max(1, Math.round(keyInterval))});
      this.frames = 0;
      this.config = null;
      this.encoder = null;
      this.samples = [];
      this.avcC = null;
      this.lastTimestamp = -Infinity;
      this.error = null;
      this.state = "new";  // new -> open -> finished | cancelled
    }

    async open() {
      this.check("new");
      if (!supported(this.env)) throw Error("Movie export needs WebCodecs (VideoEncoder), which this browser lacks.");
      const config = await chooseConfig(this, this.env);
      this.check("new");
      if (!config) throw Error(`This browser cannot encode H.264 at ${this.width}x${this.height}, ${this.fps} FPS.`);
      this.config = config;
      this.encoder = new this.env.VideoEncoder({
        output: (chunk, metadata) => this.collect(chunk, metadata),
        error: e => {this.error = this.error || e},
      });
      this.encoder.configure(config);
      this.state = "open";
    }

    collect(chunk, metadata) {
      if (this.state === "cancelled") return;
      const description = metadata && metadata.decoderConfig && metadata.decoderConfig.description;
      if (description && !this.avcC) {
        const bytes = ArrayBuffer.isView(description)
          ? new Uint8Array(description.buffer, description.byteOffset, description.byteLength)
          : new Uint8Array(description);
        this.avcC = bytes.slice();
      }
      // Output must arrive in presentation order; anything else means B-frames.
      if (chunk.timestamp <= this.lastTimestamp)
        this.error = this.error || Error("Encoder reordered frames (B-frames are not supported).");
      this.lastTimestamp = chunk.timestamp;
      const data = new Uint8Array(chunk.byteLength);
      chunk.copyTo(data);
      this.samples.push({data, key: chunk.type === "key"});
    }

    check(...states) {
      if (this.state === "cancelled") throw Error("Movie export was cancelled.");
      if (this.error) {
        this.release();
        throw Error(`Video encoder failed: ${this.error.message || this.error}`);
      }
      if (!states.includes(this.state)) throw Error(`Movie encoder is ${this.state}, expected ${states.join(" or ")}.`);
    }

    // Resolves on the next dequeue event, or after a few ms as a fallback.
    waitDequeue() {
      return new Promise(resolve => {
        const encoder = this.encoder;
        let timer = null;
        const done = () => {
          clearTimeout(timer);
          if (encoder.removeEventListener) encoder.removeEventListener("dequeue", done);
          resolve();
        };
        if (encoder.addEventListener) encoder.addEventListener("dequeue", done);
        timer = setTimeout(done, 5);
      });
    }

    async addFrame(source) {
      this.check("open");
      if (!source || source.width !== this.width || source.height !== this.height)
        throw Error(`Frame must be ${this.width}x${this.height}, got ${source && source.width}x${source && source.height}.`);
      const i = this.frames;
      const frame = new this.env.VideoFrame(source, {
        timestamp: Math.round(i * 1e6 / this.fps), duration: Math.round(1e6 / this.fps)});
      try {
        this.encoder.encode(frame, {keyFrame: i % this.keyInterval === 0});
      } catch (e) {
        this.error = this.error || e;
      } finally {
        frame.close();
      }
      this.check("open");
      this.frames = i + 1;
      // Backpressure: never let the app render far ahead of the encoder.
      while (this.encoder.encodeQueueSize > QUEUE_LIMIT) {
        await this.waitDequeue();
        this.check("open");
      }
    }

    async finish() {
      this.check("open");
      try {await this.encoder.flush()} catch (e) {this.error = this.error || e}
      this.check("open");
      this.encoder.close();
      this.state = "finished";
      if (!this.avcC) throw Error("Encoder produced no H.264 decoder configuration (avcC).");
      const parts = muxParts({width: this.width, height: this.height, fps: this.fps,
                              avcC: this.avcC, samples: this.samples});
      this.samples = [];
      return new this.env.Blob(parts, {type: "video/mp4"});
    }

    release() {
      if (this.encoder && this.encoder.state !== "closed") {
        try {this.encoder.close()} catch (_) {/* already closed */}
      }
      this.samples = [];
    }

    cancel() {
      this.release();
      this.state = "cancelled";
    }
  }

  // ---- Saving --------------------------------------------------------------

  function filename(id, ext = "mp4") {
    return `${String(id || "event").replace(/[^a-zA-Z0-9_-]+/g, "-").slice(0, 80) || "event"}.${ext}`;
  }

  function localSaveEnabled(location = globalThis.location) {
    try {
      const page = new URL(location.href);
      return ["http:", "https:"].includes(page.protocol)
        && ["localhost", "127.0.0.1", "[::1]"].includes(page.hostname)
        && page.searchParams.get("local_save") === "1";
    } catch (_) {return false}
  }

  // POST to the talk pipeline's local receiver; it must echo the same name.
  async function saveLocal(blob, name, {location = globalThis.location, fetch = globalThis.fetch} = {}) {
    if (!localSaveEnabled(location)) throw Error("Local saving requires a localhost viewer opened with ?local_save=1.");
    const type = (blob && blob.type) || "";
    if (!blob || !blob.size || !(type.startsWith("video/") || type === "image/png"))
      throw Error("No completed video or PNG is available to save.");
    if (!/^[a-zA-Z0-9_-][a-zA-Z0-9_.-]*$/.test(String(name || ""))) throw Error(`Unsafe file name: ${name}`);
    const page = new URL(location.href);
    const response = await fetch(new URL(`/recordings/${encodeURIComponent(name)}`, page.origin).href, {
      method: "POST", body: blob, headers: {"Content-Type": type},
      mode: "same-origin", credentials: "omit", redirect: "error"});
    if (!response.ok) throw Error(`Local server could not save the file (HTTP ${response.status}). The normal download remains available.`);
    const result = await response.json();
    if (!result || result.saved_as !== name) throw Error("Local server returned an unexpected filename. The normal download remains available.");
    return {saved_as: name};
  }

  function downloadBlob(blob, name, doc = globalThis.document) {
    const URLs = (doc.defaultView && doc.defaultView.URL) || globalThis.URL;
    const url = URLs.createObjectURL(blob);
    const a = doc.createElement("a");
    a.href = url;
    a.download = name;
    a.style.display = "none";
    doc.body.appendChild(a);
    a.click();
    a.remove();
    return url;
  }

  return {supported, chooseConfig, defaultBitrate, codecCandidates, MovieEncoder, muxMp4,
          filename, localSaveEnabled, saveLocal, downloadBlob};
})();

if (typeof module !== "undefined" && module.exports) module.exports = LightHitMovie;
