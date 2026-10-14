"use strict";

// WebGL2 renderer of the event scene.  The scene is drawn in linear HDR into
// a multisampled float target, a bloom pyramid is added and a filmic tone map
// writes the canvas.  Without float render targets it falls back to a direct
// LDR pass.  Pure presentation: it never sees physics arrays, only geometry,
// colours and intensities prepared by the app.
const LightHitStudioGL = (() => {
  // ---------------------------------------------------------------- matrices
  const M = {
    perspective(fovy, aspect, near, far) {
      const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far), m = new Float32Array(16);
      m[0] = f / aspect; m[5] = f; m[10] = (far + near) * nf; m[11] = -1; m[14] = 2 * far * near * nf;
      return m;
    },
    lookAt(eye, target, up) {
      let zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
      let l = Math.hypot(zx, zy, zz) || 1; zx /= l; zy /= l; zz /= l;
      let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
      l = Math.hypot(xx, xy, xz);
      if (l < 1e-9) {xx = 1; xy = 0; xz = 0; l = 1}
      xx /= l; xy /= l; xz /= l;
      const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
      return new Float32Array([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
        -(xx * eye[0] + xy * eye[1] + xz * eye[2]), -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
        -(zx * eye[0] + zy * eye[1] + zz * eye[2]), 1]);
    },
    multiply(a, b) {
      const o = new Float32Array(16);
      for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
        o[c * 4 + r] = s;
      }
      return o;
    },
  };

  function cameraMatrices(camera, width, height) {
    const aspect = width / Math.max(1, height);
    const view = M.lookAt(camera.eye, camera.target, camera.up || [0, 0, 1]);
    const proj = M.perspective(camera.fov, aspect, camera.near, camera.far);
    return {view, proj, viewProj: M.multiply(proj, view), width, height};
  }

  // Screen position (CSS pixels, origin top-left) and view depth of a point.
  function project(mats, p) {
    const m = mats.viewProj;
    const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
    const y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
    const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    if (w <= 1e-6) return null;
    return {x: (x / w * 0.5 + 0.5) * mats.width, y: (1 - (y / w * 0.5 + 0.5)) * mats.height, depth: w};
  }

  // ----------------------------------------------------------------- shaders
  const COMMON = `#version 300 es
precision highp float;
precision highp int;
uniform mat4 uView, uProj;
uniform vec2 uViewport;
uniform float uFogDistance;
`;

  const BACKGROUND_VS = `#version 300 es
out vec2 vUv;
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);vUv=p;gl_Position=vec4(p*2.0-1.0,0.0,1.0);}`;
  const BACKGROUND_FS = `#version 300 es
precision highp float;
precision highp int;
in vec2 vUv;
uniform vec3 uTop, uBottom, uGlow;
uniform vec2 uGlowAt;
out vec4 outColor;
void main(){
  vec3 c=mix(uBottom,uTop,smoothstep(0.0,1.0,vUv.y));
  float g=exp(-6.0*dot(vUv-uGlowAt,vUv-uGlowAt));
  outColor=vec4(c+uGlow*g,1.0);
}`;

  // Billboards.  Instance: position.xyz, radius (m) | rgb, intensity.
  const SPRITE_VS = COMMON + `
layout(location=0) in vec4 aPosRadius;
layout(location=1) in vec4 aColor;
uniform float uHalo, uMinPixels, uMode;
out vec2 vUv;
out vec4 vColor;
out float vFog, vCore;
const vec2 corners[6]=vec2[6](vec2(-1,-1),vec2(1,-1),vec2(1,1),vec2(-1,-1),vec2(1,1),vec2(-1,1));
void main(){
  vec2 c=corners[gl_VertexID];
  vec4 view=uView*vec4(aPosRadius.xyz,1.0);
  float extent=aPosRadius.w*uHalo;
  float depth=max(-view.z,1e-3);
  float pixels=extent*uProj[1][1]*0.5*uViewport.y/depth;
  float grow=max(1.0,uMinPixels/max(pixels,1e-6));
  vCore=1.0/uHalo/grow;
  view.xy+=c*extent*grow;
  gl_Position=uProj*view;
  vUv=c;
  // Energy of an enlarged tiny glow is conserved; beads are not dimmed.
  float energy=uMode<0.5?1.0/(grow*grow):1.0;
  vColor=vec4(aColor.rgb,aColor.a*energy);
  vFog=exp(-depth/uFogDistance);
}`;
  const SPRITE_FS = `#version 300 es
precision highp float;
precision highp int;
in vec2 vUv;
in vec4 vColor;
in float vFog, vCore;
uniform float uMode, uFogMix;
uniform vec3 uFogColor;
out vec4 outColor;
void main(){
  float r=length(vUv);
  if(r>1.0)discard;
  if(uMode<0.5){
    // glow: hot core plus a soft halo, additive in linear light
    float core=1.0-smoothstep(vCore*0.55,vCore,r);
    float halo=exp(-r*r*9.0)*0.55+exp(-r*r*2.6)*0.18;
    halo*=1.0-smoothstep(0.85,1.0,r);
    vec3 col=vColor.rgb*(halo+core*1.4)+vec3(core*core*0.9);
    float fog=mix(1.0,vFog,uFogMix);
    outColor=vec4(col*vColor.a*fog,0.0);
  }else{
    // bead: small shaded glass sphere
    float rr=r*r;
    vec3 n=vec3(vUv,sqrt(max(0.0,1.0-rr)));
    float light=0.45+0.55*max(0.0,dot(n,normalize(vec3(-0.4,0.6,0.7))));
    float spec=pow(max(0.0,dot(n,normalize(vec3(-0.25,0.45,0.85)))),24.0)*0.6;
    vec3 col=vColor.rgb*light+spec*vColor.rgb*1.5;
    col=mix(uFogColor,col,mix(1.0,vFog,uFogMix));
    float edge=1.0-smoothstep(0.82,1.0,r);
    outColor=vec4(col*edge*vColor.a,edge*vColor.a);
  }
}`;

  // Screen-space segments.  Instance: p0, p1, colour0 (rgba), colour1, width.
  const LINE_VS = COMMON + `
layout(location=0) in vec3 aP0;
layout(location=1) in vec3 aP1;
layout(location=2) in vec4 aC0;
layout(location=3) in vec4 aC1;
layout(location=4) in float aWidth;
uniform float uFogMix;
out vec4 vColor;
out float vSide, vHalf, vExtent;
void main(){
  int id=gl_VertexID;
  float t=(id==1||id==2||id==4)?1.0:0.0;
  float side=(id==2||id==4||id==5)?1.0:-1.0;
  vec4 v0=uView*vec4(aP0,1.0), v1=uView*vec4(aP1,1.0);
  float near=-0.5;
  if(v0.z>near&&v1.z>near){gl_Position=vec4(2.0,2.0,2.0,1.0);vColor=vec4(0);vSide=0.0;vHalf=0.0;vExtent=1.0;return;}
  if(v0.z>near)v0=mix(v0,v1,(v0.z-near)/(v0.z-v1.z));
  if(v1.z>near)v1=mix(v1,v0,(v1.z-near)/(v1.z-v0.z));
  vec4 c0=uProj*v0, c1=uProj*v1;
  vec2 s0=c0.xy/c0.w*uViewport*0.5, s1=c1.xy/c1.w*uViewport*0.5;
  vec2 d=s1-s0;
  float l=length(d);
  vec2 n=l>1e-4?vec2(-d.y,d.x)/l:vec2(0.0,1.0);
  // Sub-pixel lines are drawn one pixel wide with their coverage as opacity.
  float width=max(aWidth,1.0);
  vHalf=width*0.5;
  vExtent=vHalf+1.0;
  vec4 c=t>0.5?c1:c0;
  c.xy+=n*side*vExtent/(uViewport*0.5)*c.w;
  gl_Position=c;
  vec4 col=mix(aC0,aC1,t);
  float depth=-mix(v0.z,v1.z,t);
  col.a*=mix(1.0,exp(-depth/uFogDistance),uFogMix)*min(1.0,aWidth);
  vColor=col;
  vSide=side;
}`;
const LINE_FS = `#version 300 es
precision highp float;
precision highp int;
in vec4 vColor;
in float vSide, vHalf, vExtent;
uniform float uAdditive;
out vec4 outColor;
void main(){
  float a=vColor.a*clamp(vHalf+0.5-abs(vSide)*vExtent,0.0,1.0);
  outColor=uAdditive>0.5?vec4(vColor.rgb*a,0.0):vec4(vColor.rgb*a,a);
}`;

  // Light fronts: a cone behind a moving particle or an expanding sphere.
  const SHELL_VS = COMMON + `
layout(location=0) in vec2 aUv;
uniform int uKind;
uniform vec3 uOrigin, uAxis, uE1, uE2;
uniform float uA0, uA1, uTanBeta, uRadius, uK1, uNg;
out float vLight, vTheta, vFacing, vFog;
void main(){
  float phi=aUv.y*6.28318530718;
  vec3 radial=cos(phi)*uE1+sin(phi)*uE2;
  vec3 p, n;
  if(uKind==0){
    float a=mix(uA0,uA1,aUv.x*aUv.x);
    p=uOrigin+uAxis*a+radial*a*uTanBeta;
    n=normalize(radial-uAxis*uTanBeta);
    vLight=a/uK1/uNg;
    vTheta=0.0;
  }else{
    float th=aUv.x*3.14159265359;
    n=cos(th)*uAxis+sin(th)*radial;
    p=uOrigin+uRadius*n;
    vLight=uRadius;
    vTheta=th;
  }
  vec4 view=uView*vec4(p,1.0);
  vec3 nv=normalize(mat3(uView)*n);
  vFacing=abs(dot(nv,normalize(-view.xyz)));
  vFog=exp(-max(-view.z,0.0)/uFogDistance);
  gl_Position=uProj*view;
}`;
  const SHELL_FS = `#version 300 es
precision highp float;
precision highp int;
in float vLight, vTheta, vFacing, vFog;
uniform int uKind;
uniform vec3 uColor;
uniform float uIntensity, uLambda, uPeak, uWidth, uFloor, uFogMix;
out vec4 outColor;
void main(){
  float fade=exp(-vLight/uLambda);
  float z=(vTheta-uPeak)/uWidth;
  float angular=uKind==1?mix(uFloor,1.0,exp(-z*z)):1.0;
  float rim=0.18+0.82*pow(clamp(1.0-vFacing,0.0,1.0),2.5);
  float a=uIntensity*fade*angular*rim*mix(1.0,vFog,uFogMix);
  outColor=vec4(uColor*a,0.0);
}`;

  const DOWN_FS = `#version 300 es
precision highp float;
precision highp int;
in vec2 vUv;
uniform sampler2D uSource;
uniform vec2 uTexel;
uniform float uThreshold;
out vec4 outColor;
void main(){
  vec3 c=texture(uSource,vUv).rgb*4.0;
  c+=texture(uSource,vUv+vec2(-1,-1)*uTexel).rgb;
  c+=texture(uSource,vUv+vec2(1,-1)*uTexel).rgb;
  c+=texture(uSource,vUv+vec2(-1,1)*uTexel).rgb;
  c+=texture(uSource,vUv+vec2(1,1)*uTexel).rgb;
  c/=8.0;
  if(uThreshold>0.0){float l=max(c.r,max(c.g,c.b));c*=max(0.0,l-uThreshold)/max(l,1e-4);}
  outColor=vec4(c,1.0);
}`;
  const UP_FS = `#version 300 es
precision highp float;
precision highp int;
in vec2 vUv;
uniform sampler2D uSource;
uniform vec2 uTexel;
out vec4 outColor;
void main(){
  vec3 c=vec3(0);
  c+=texture(uSource,vUv+vec2(-2,0)*uTexel).rgb;
  c+=texture(uSource,vUv+vec2(2,0)*uTexel).rgb;
  c+=texture(uSource,vUv+vec2(0,-2)*uTexel).rgb;
  c+=texture(uSource,vUv+vec2(0,2)*uTexel).rgb;
  c+=texture(uSource,vUv+vec2(-1,-1)*uTexel).rgb*2.0;
  c+=texture(uSource,vUv+vec2(1,-1)*uTexel).rgb*2.0;
  c+=texture(uSource,vUv+vec2(-1,1)*uTexel).rgb*2.0;
  c+=texture(uSource,vUv+vec2(1,1)*uTexel).rgb*2.0;
  outColor=vec4(c/12.0,1.0);
}`;
  const COMPOSITE_FS = `#version 300 es
precision highp float;
precision highp int;
in vec2 vUv;
uniform sampler2D uScene, uBloom;
uniform float uBloomStrength, uExposure, uVignette;
uniform vec2 uViewport;
out vec4 outColor;
vec3 aces(vec3 x){return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14),0.0,1.0);}
float hash(vec2 p){return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453);}
void main(){
  vec3 c=texture(uScene,vUv).rgb+texture(uBloom,vUv).rgb*uBloomStrength;
  c=aces(c*uExposure);
  vec2 q=vUv-0.5;
  c*=1.0-uVignette*dot(q,q)*1.6;
  c=pow(c,vec3(1.0/2.2));
  c+=(hash(gl_FragCoord.xy)-0.5)/255.0;
  outColor=vec4(c,1.0);
}`;
  const DIRECT_FS = `#version 300 es
precision highp float;
precision highp int;
in vec2 vUv;
uniform sampler2D uScene;
out vec4 outColor;
void main(){outColor=vec4(pow(texture(uScene,vUv).rgb,vec3(1.0/2.2)),1.0);}`;

  // ---------------------------------------------------------------- helpers
  function compile(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw Error(`Shader compilation failed: ${log}`);
    }
    return shader;
  }

  function program(gl, vs, fs) {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw Error(`Shader link failed: ${gl.getProgramInfoLog(p)}`);
    const uniforms = {}, count = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < count; i++) {
      const name = gl.getActiveUniform(p, i).name;
      uniforms[name] = gl.getUniformLocation(p, name);
    }
    return {p, u: uniforms};
  }

  function instanced(gl, layout) {
    const vao = gl.createVertexArray(), buffer = gl.createBuffer();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    const stride = layout.reduce((s, n) => s + n, 0) * 4;
    let offset = 0;
    layout.forEach((size, location) => {
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, stride, offset);
      gl.vertexAttribDivisor(location, 1);
      offset += size * 4;
    });
    gl.bindVertexArray(null);
    return {vao, buffer, floats: stride / 4, capacity: 0};
  }

  function upload(gl, batch, data) {
    gl.bindBuffer(gl.ARRAY_BUFFER, batch.buffer);
    if (data.byteLength > batch.capacity) {
      batch.capacity = Math.max(data.byteLength, 2 * batch.capacity);
      gl.bufferData(gl.ARRAY_BUFFER, batch.capacity, gl.DYNAMIC_DRAW);
    }
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
  }

  function shellMesh(gl, rings, segments) {
    const uv = [], index = [];
    for (let i = 0; i <= rings; i++) for (let j = 0; j <= segments; j++) uv.push(i / rings, j / segments);
    for (let i = 0; i < rings; i++) for (let j = 0; j < segments; j++) {
      const a = i * (segments + 1) + j, b = a + segments + 1;
      index.push(a, b, a + 1, a + 1, b, b + 1);
    }
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const vb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(uv), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    const ib = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(index), gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    return {vao, count: index.length};
  }

  function basis(axis) {
    const a = Math.abs(axis[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
    let e1 = [axis[1] * a[2] - axis[2] * a[1], axis[2] * a[0] - axis[0] * a[2], axis[0] * a[1] - axis[1] * a[0]];
    const l = Math.hypot(...e1); e1 = e1.map(x => x / l);
    const e2 = [axis[1] * e1[2] - axis[2] * e1[1], axis[2] * e1[0] - axis[0] * e1[2], axis[0] * e1[1] - axis[1] * e1[0]];
    return [e1, e2];
  }

  // ---------------------------------------------------------------- renderer
  function create(canvas) {
    const gl = canvas.getContext("webgl2", {antialias: false, alpha: false, depth: true,
      preserveDrawingBuffer: true, powerPreference: "high-performance"});
    if (!gl) throw Error("WebGL2 is unavailable in this browser.");
    const hdr = !!gl.getExtension("EXT_color_buffer_float");
    gl.getExtension("OES_texture_float_linear");
    const programs = {
      background: program(gl, BACKGROUND_VS, BACKGROUND_FS),
      sprite: program(gl, SPRITE_VS, SPRITE_FS),
      line: program(gl, LINE_VS, LINE_FS),
      shell: program(gl, SHELL_VS, SHELL_FS),
      down: program(gl, BACKGROUND_VS, DOWN_FS),
      up: program(gl, BACKGROUND_VS, UP_FS),
      composite: program(gl, BACKGROUND_VS, COMPOSITE_FS),
      direct: program(gl, BACKGROUND_VS, DIRECT_FS),
    };
    const sprites = instanced(gl, [4, 4]), lines = instanced(gl, [3, 3, 4, 4, 1]);
    const mesh = shellMesh(gl, 96, 128), empty = gl.createVertexArray();
    let targets = null;

    function texture(w, h, format, type) {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, format, w, h, 0, gl.RGBA, type, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      return {t, fb, w, h};
    }

    function release() {
      if (!targets) return;
      gl.deleteFramebuffer(targets.msaa);
      gl.deleteRenderbuffer(targets.colour);
      gl.deleteRenderbuffer(targets.depth);
      for (const x of [targets.scene, ...targets.levels]) {gl.deleteTexture(x.t); gl.deleteFramebuffer(x.fb)}
      targets = null;
    }

    function ensureTargets(w, h) {
      if (targets && targets.w === w && targets.h === h) return;
      release();
      const internal = hdr ? gl.RGBA16F : gl.RGBA8, type = hdr ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;
      const samples = Math.min(4, gl.getParameter(gl.MAX_SAMPLES) || 1);
      const msaa = gl.createFramebuffer(), colour = gl.createRenderbuffer(), depth = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, colour);
      gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, internal, w, h);
      gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
      gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.DEPTH_COMPONENT24, w, h);
      gl.bindFramebuffer(gl.FRAMEBUFFER, msaa);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, colour);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
      const scene = texture(w, h, internal, type), levels = [];
      let lw = w, lh = h;
      for (let i = 0; i < 6 && lw > 8 && lh > 8; i++) {
        lw = Math.max(1, lw >> 1); lh = Math.max(1, lh >> 1);
        levels.push(texture(lw, lh, internal, type));
      }
      targets = {w, h, msaa, colour, depth, scene, levels};
    }

    function setCommon(prog, mats, scene) {
      gl.useProgram(prog.p);
      if (prog.u.uView) gl.uniformMatrix4fv(prog.u.uView, false, mats.view);
      if (prog.u.uProj) gl.uniformMatrix4fv(prog.u.uProj, false, mats.proj);
      if (prog.u.uViewport) gl.uniform2f(prog.u.uViewport, mats.width, mats.height);
      if (prog.u.uFogDistance) gl.uniform1f(prog.u.uFogDistance, scene.fog.distance);
      if (prog.u.uFogMix) gl.uniform1f(prog.u.uFogMix, scene.fog.amount);
      if (prog.u.uFogColor) gl.uniform3fv(prog.u.uFogColor, scene.fog.color);
    }

    function drawSprites(mats, scene, data, mode, halo, minPixels) {
      if (!data || !data.length) return;
      const prog = programs.sprite;
      setCommon(prog, mats, scene);
      gl.uniform1f(prog.u.uMode, mode);
      gl.uniform1f(prog.u.uHalo, halo);
      gl.uniform1f(prog.u.uMinPixels, minPixels);
      upload(gl, sprites, data);
      gl.bindVertexArray(sprites.vao);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, data.length / 8);
    }

    function drawLines(mats, scene, data, additive) {
      if (!data || !data.length) return;
      const prog = programs.line;
      setCommon(prog, mats, scene);
      gl.uniform1f(prog.u.uAdditive, additive ? 1 : 0);
      upload(gl, lines, data);
      gl.bindVertexArray(lines.vao);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, data.length / 15);
    }

    function drawShell(mats, scene, shell) {
      const prog = programs.shell, [e1, e2] = basis(shell.axis);
      setCommon(prog, mats, scene);
      gl.uniform1i(prog.u.uKind, shell.kind === "cone" ? 0 : 1);
      gl.uniform3fv(prog.u.uOrigin, shell.origin);
      gl.uniform3fv(prog.u.uAxis, shell.axis);
      gl.uniform3fv(prog.u.uE1, e1);
      gl.uniform3fv(prog.u.uE2, e2);
      gl.uniform1f(prog.u.uA0, shell.a0 || 0);
      gl.uniform1f(prog.u.uA1, shell.a1 || 0);
      gl.uniform1f(prog.u.uTanBeta, shell.tanBeta || 0);
      gl.uniform1f(prog.u.uRadius, shell.radius || 0);
      gl.uniform1f(prog.u.uK1, shell.k1 || 1);
      gl.uniform1f(prog.u.uNg, shell.ng || 1);
      gl.uniform3fv(prog.u.uColor, shell.color);
      gl.uniform1f(prog.u.uIntensity, shell.intensity);
      gl.uniform1f(prog.u.uLambda, shell.lambda);
      gl.uniform1f(prog.u.uPeak, shell.peak || 0);
      gl.uniform1f(prog.u.uWidth, shell.width || 1);
      gl.uniform1f(prog.u.uFloor, shell.floor ?? 1);
      gl.bindVertexArray(mesh.vao);
      gl.drawElements(gl.TRIANGLES, mesh.count, gl.UNSIGNED_INT, 0);
    }

    function fullscreen(prog, target) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fb : null);
      gl.viewport(0, 0, target ? target.w : canvas.width, target ? target.h : canvas.height);
      gl.useProgram(prog.p);
      gl.bindVertexArray(empty);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    function bind(prog, name, unit, tex) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(prog.u[name], unit);
    }

    // scene: {background, fog, beads, lines, glows, additiveLines, shells, post}
    function render(scene, camera) {
      const w = canvas.width, h = canvas.height;
      ensureTargets(w, h);
      const mats = cameraMatrices(camera, w, h);
      gl.bindFramebuffer(gl.FRAMEBUFFER, targets.msaa);
      gl.viewport(0, 0, w, h);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
      const bg = programs.background;
      gl.useProgram(bg.p);
      gl.uniform3fv(bg.u.uTop, scene.background.top);
      gl.uniform3fv(bg.u.uBottom, scene.background.bottom);
      gl.uniform3fv(bg.u.uGlow, scene.background.glow || [0, 0, 0]);
      gl.uniform2fv(bg.u.uGlowAt, scene.background.glowAt || [0.5, 1.1]);
      gl.bindVertexArray(empty);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.clear(gl.DEPTH_BUFFER_BIT);

      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      drawLines(mats, scene, scene.lines, false);
      gl.depthMask(true);
      drawSprites(mats, scene, scene.beads, 1, 1, 1.6);
      gl.depthMask(false);
      gl.blendFunc(gl.ONE, gl.ONE);
      for (const shell of scene.shells || []) drawShell(mats, scene, shell);
      drawLines(mats, scene, scene.additiveLines, true);
      drawSprites(mats, scene, scene.glows, 0, scene.halo || 5, 2.2);
      gl.depthMask(true);
      gl.disable(gl.BLEND);
      gl.disable(gl.DEPTH_TEST);

      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, targets.msaa);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, targets.scene.fb);
      gl.blitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.COLOR_BUFFER_BIT, gl.NEAREST);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);

      const post = scene.post || {};
      if (!hdr) {
        bind(programs.direct, "uScene", 0, targets.scene.t);
        fullscreen(programs.direct, null);
        return mats;
      }
      // Bloom pyramid: thresholded downsample chain, then additive tent upsample.
      let source = targets.scene;
      targets.levels.forEach((level, i) => {
        const prog = programs.down;
        gl.useProgram(prog.p);
        bind(prog, "uSource", 0, source.t);
        gl.uniform2f(prog.u.uTexel, 1 / source.w, 1 / source.h);
        gl.uniform1f(prog.u.uThreshold, i === 0 ? (post.threshold ?? 0.6) : 0);
        fullscreen(prog, level);
        source = level;
      });
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      for (let i = targets.levels.length - 1; i > 0; i--) {
        const prog = programs.up, from = targets.levels[i];
        gl.useProgram(prog.p);
        bind(prog, "uSource", 0, from.t);
        gl.uniform2f(prog.u.uTexel, 0.5 / from.w, 0.5 / from.h);
        fullscreen(prog, targets.levels[i - 1]);
      }
      gl.disable(gl.BLEND);
      const comp = programs.composite;
      gl.useProgram(comp.p);
      bind(comp, "uScene", 0, targets.scene.t);
      bind(comp, "uBloom", 1, (targets.levels[0] || targets.scene).t);
      gl.uniform1f(comp.u.uBloomStrength, post.bloom ?? 0.8);
      gl.uniform1f(comp.u.uExposure, post.exposure ?? 1.2);
      gl.uniform1f(comp.u.uVignette, post.vignette ?? 0.35);
      gl.uniform2f(comp.u.uViewport, w, h);
      fullscreen(comp, null);
      return mats;
    }

    return {gl, hdr, render, release, matrices: (camera, w, h) => cameraMatrices(camera, w, h)};
  }

  return {create, project, cameraMatrices, M};
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = LightHitStudioGL;
}
