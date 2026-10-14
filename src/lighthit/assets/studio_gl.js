"use strict";

// three.js renderer of the event scene.  The scene is drawn in linear HDR into
// a multisampled half-float target; an EffectComposer chain adds a bloom
// pyramid and a grading pass applies a filmic tone map, vignette and dither.
// Without float render targets it falls back to 8 bits without bloom.  Pure
// presentation: it never sees physics arrays, only geometry, colours and
// intensities prepared by the app.
const LightHitStudioGL = (() => {
  // Screen position (pixels, origin top-left) and view depth of a point.
  // ``mats.viewProj`` is a column-major 4x4 matrix.
  function project(mats, p) {
    const m = mats.viewProj;
    const x = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
    const y = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
    const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    if (w <= 1e-6) return null;
    return {x: (x / w * 0.5 + 0.5) * mats.width, y: (1 - (y / w * 0.5 + 0.5)) * mats.height, depth: w};
  }

  // ----------------------------------------------------------------- shaders
  // Raw GLSL 3 programs; three.js supplies viewMatrix and projectionMatrix.
  const COMMON = `
precision highp float;
precision highp int;
uniform mat4 viewMatrix, projectionMatrix;
uniform vec2 uViewport;
uniform float uFogDistance;
`;

  const BACKGROUND_VS = `
precision highp float;
precision highp int;
in vec3 position;
out vec2 vUv;
void main(){vUv=position.xy*0.5+0.5;gl_Position=vec4(position.xy,1.0,1.0);}`;
  const BACKGROUND_FS = `
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
  // Base vertex: position.xy is the quad corner in [-1, 1]^2.
  const SPRITE_VS = COMMON + `
in vec3 position;
in vec4 aPosRadius;
in vec4 aColor;
uniform float uHalo, uMinPixels, uMode;
out vec2 vUv;
out vec4 vColor;
out float vFog, vCore;
void main(){
  vec2 c=position.xy;
  vec4 view=viewMatrix*vec4(aPosRadius.xyz,1.0);
  float extent=aPosRadius.w*uHalo;
  float depth=max(-view.z,1e-3);
  float pixels=extent*projectionMatrix[1][1]*0.5*uViewport.y/depth;
  float grow=max(1.0,uMinPixels/max(pixels,1e-6));
  vCore=1.0/uHalo/grow;
  view.xy+=c*extent*grow;
  gl_Position=projectionMatrix*view;
  vUv=c;
  // Energy of an enlarged tiny glow is conserved; beads are not dimmed.
  float energy=uMode<0.5?1.0/(grow*grow):1.0;
  vColor=vec4(aColor.rgb,aColor.a*energy);
  vFog=exp(-depth/uFogDistance);
}`;
  const SPRITE_FS = `
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
  // Base vertex: position.x is the end (0 or 1), position.y the side (-1 or 1).
  const LINE_VS = COMMON + `
in vec3 position;
in vec3 aP0;
in vec3 aP1;
in vec4 aC0;
in vec4 aC1;
in float aWidth;
uniform float uFogMix;
out vec4 vColor;
out float vSide, vHalf, vExtent;
void main(){
  float t=position.x;
  float side=position.y;
  vec4 v0=viewMatrix*vec4(aP0,1.0), v1=viewMatrix*vec4(aP1,1.0);
  float near=-0.5;
  if(v0.z>near&&v1.z>near){gl_Position=vec4(2.0,2.0,2.0,1.0);vColor=vec4(0);vSide=0.0;vHalf=0.0;vExtent=1.0;return;}
  if(v0.z>near)v0=mix(v0,v1,(v0.z-near)/(v0.z-v1.z));
  if(v1.z>near)v1=mix(v1,v0,(v1.z-near)/(v1.z-v0.z));
  vec4 c0=projectionMatrix*v0, c1=projectionMatrix*v1;
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
  const LINE_FS = `
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
  // Base vertex: position.xy is the (ring, segment) coordinate in [0, 1]^2.
  const SHELL_VS = COMMON + `
in vec3 position;
uniform int uKind;
uniform vec3 uOrigin, uAxis, uE1, uE2;
uniform float uA0, uA1, uTanBeta, uRadius, uK1, uNg;
out float vLight, vTheta, vFacing, vFog;
void main(){
  vec2 uv=position.xy;
  float phi=uv.y*6.28318530718;
  vec3 radial=cos(phi)*uE1+sin(phi)*uE2;
  vec3 p, n;
  if(uKind==0){
    float a=mix(uA0,uA1,uv.x*uv.x);
    p=uOrigin+uAxis*a+radial*a*uTanBeta;
    n=normalize(radial-uAxis*uTanBeta);
    vLight=a/uK1/uNg;
    vTheta=0.0;
  }else{
    float th=uv.x*3.14159265359;
    n=cos(th)*uAxis+sin(th)*radial;
    p=uOrigin+uRadius*n;
    vLight=uRadius;
    vTheta=th;
  }
  vec4 view=viewMatrix*vec4(p,1.0);
  vec3 nv=normalize(mat3(viewMatrix)*n);
  vFacing=abs(dot(nv,normalize(-view.xyz)));
  vFog=exp(-max(-view.z,0.0)/uFogDistance);
  gl_Position=projectionMatrix*view;
}`;
  const SHELL_FS = `
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

  // Post-processing programs for three.js full-screen passes.
  const FULLSCREEN_VS = `
varying vec2 vUv;
void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}`;
  // Bloom pyramid: a thresholded 5-tap downsample chain, then additive tent
  // upsampling.  Unlike three's UnrealBloomPass, whose Gaussian is cut at one
  // sigma, this keeps halos round around very bright HDR sources.
  const DOWN_FS = `
uniform sampler2D tSource;
uniform vec2 uTexel;
uniform float uThreshold;
varying vec2 vUv;
void main(){
  vec3 c=texture2D(tSource,vUv).rgb*4.0;
  c+=texture2D(tSource,vUv+vec2(-1,-1)*uTexel).rgb;
  c+=texture2D(tSource,vUv+vec2(1,-1)*uTexel).rgb;
  c+=texture2D(tSource,vUv+vec2(-1,1)*uTexel).rgb;
  c+=texture2D(tSource,vUv+vec2(1,1)*uTexel).rgb;
  c/=8.0;
  if(uThreshold>0.0){float l=max(c.r,max(c.g,c.b));c*=max(0.0,l-uThreshold)/max(l,1e-4);}
  gl_FragColor=vec4(c,1.0);
}`;
  const UP_FS = `
uniform sampler2D tSource;
uniform vec2 uTexel;
varying vec2 vUv;
void main(){
  vec3 c=vec3(0);
  c+=texture2D(tSource,vUv+vec2(-2,0)*uTexel).rgb;
  c+=texture2D(tSource,vUv+vec2(2,0)*uTexel).rgb;
  c+=texture2D(tSource,vUv+vec2(0,-2)*uTexel).rgb;
  c+=texture2D(tSource,vUv+vec2(0,2)*uTexel).rgb;
  c+=texture2D(tSource,vUv+vec2(-1,-1)*uTexel).rgb*2.0;
  c+=texture2D(tSource,vUv+vec2(1,-1)*uTexel).rgb*2.0;
  c+=texture2D(tSource,vUv+vec2(-1,1)*uTexel).rgb*2.0;
  c+=texture2D(tSource,vUv+vec2(1,1)*uTexel).rgb*2.0;
  gl_FragColor=vec4(c/12.0,1.0);
}`;

  // Final grading of the HDR image plus bloom: filmic tone map, vignette,
  // display gamma and a one-level dither against banding.
  const GRADE = {
    uniforms: {tDiffuse: {value: null}, tBloom: {value: null}, uBloomStrength: {value: 0},
               uExposure: {value: 1.2}, uVignette: {value: 0.35}},
    vertexShader: FULLSCREEN_VS,
    fragmentShader: `
uniform sampler2D tDiffuse, tBloom;
uniform float uBloomStrength, uExposure, uVignette;
varying vec2 vUv;
vec3 aces(vec3 x){return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14),0.0,1.0);}
float hash(vec2 p){return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453);}
void main(){
  vec3 c=texture2D(tDiffuse,vUv).rgb;
  if(uBloomStrength>0.0)c+=texture2D(tBloom,vUv).rgb*uBloomStrength;
  c=aces(max(c,0.0)*uExposure);
  vec2 q=vUv-0.5;
  c*=1.0-uVignette*dot(q,q)*1.6;
  c=pow(c,vec3(1.0/2.2));
  c+=(hash(gl_FragCoord.xy)-0.5)/255.0;
  gl_FragColor=vec4(c,1.0);
}`,
  };

  // ---------------------------------------------------------------- helpers
  function basis(axis) {
    const a = Math.abs(axis[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
    let e1 = [axis[1] * a[2] - axis[2] * a[1], axis[2] * a[0] - axis[0] * a[2], axis[0] * a[1] - axis[1] * a[0]];
    const l = Math.hypot(...e1); e1 = e1.map(x => x / l);
    const e2 = [axis[1] * e1[2] - axis[2] * e1[1], axis[2] * e1[0] - axis[0] * e1[2], axis[0] * e1[1] - axis[1] * e1[0]];
    return [e1, e2];
  }

  // Premultiplied "over" or pure additive blending in linear light.
  function rawMaterial(vertexShader, fragmentShader, uniforms, {additive = false, depthWrite = false} = {}) {
    return new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader, fragmentShader, uniforms,
      depthWrite, depthTest: true, transparent: true, side: THREE.DoubleSide,
      blending: THREE.CustomBlending, blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor, blendDst: additive ? THREE.OneFactor : THREE.OneMinusSrcAlphaFactor,
      blendSrcAlpha: THREE.OneFactor, blendDstAlpha: additive ? THREE.OneFactor : THREE.OneMinusSrcAlphaFactor,
    });
  }

  const corners = points => new THREE.BufferAttribute(new Float32Array(points), 3);
  const SPRITE_CORNERS = [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0];
  const LINE_CORNERS = [0, -1, 0, 1, -1, 0, 1, 1, 0, 0, -1, 0, 1, 1, 0, 0, 1, 0];

  // An instanced batch of six-vertex quads whose per-instance data is a packed
  // Float32Array that may change length between frames; storage doubles.
  function batch(points, layout, material, order) {
    const floats = layout.reduce((s, [, n]) => s + n, 0);
    const mesh = new THREE.Mesh(new THREE.InstancedBufferGeometry(), material);
    mesh.frustumCulled = false;
    mesh.renderOrder = order;
    mesh.visible = false;
    let buffer = null, source = null;
    function allocate(count) {
      mesh.geometry.dispose();
      const g = new THREE.InstancedBufferGeometry();
      g.setAttribute("position", corners(points));
      buffer = new THREE.InstancedInterleavedBuffer(new Float32Array(count * floats), floats);
      buffer.setUsage(THREE.DynamicDrawUsage);
      let offset = 0;
      for (const [name, size] of layout) {
        g.setAttribute(name, new THREE.InterleavedBufferAttribute(buffer, size, offset));
        offset += size;
      }
      mesh.geometry = g;
    }
    return {
      mesh,
      set(data) {
        const count = data && data.length ? data.length / floats : 0;
        mesh.visible = count > 0;
        if (!count || data === source) return;  // nothing, or unchanged static geometry
        source = data;
        if (!buffer || buffer.count < count) allocate(Math.max(count, 2 * (buffer ? buffer.count : 0), 64));
        buffer.array.set(data);
        buffer.needsUpdate = true;
        mesh.geometry.instanceCount = count;
      },
      dispose() {mesh.geometry.dispose(); material.dispose()},
    };
  }

  function shellGeometry(rings, segments) {
    const uv = [], index = [];
    for (let i = 0; i <= rings; i++) for (let j = 0; j <= segments; j++) uv.push(i / rings, j / segments, 0);
    for (let i = 0; i < rings; i++) for (let j = 0; j < segments; j++) {
      const a = i * (segments + 1) + j, b = a + segments + 1;
      index.push(a, b, a + 1, a + 1, b, b + 1);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", corners(uv));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(index), 1));
    return g;
  }

  // An EffectComposer pass that builds the bloom pyramid from the scene and
  // hands its finest level to the grading pass; the scene buffer is unchanged.
  function bloomPass(grade) {
    const material = (fragmentShader, additive) => new THREE.ShaderMaterial({
      vertexShader: FULLSCREEN_VS, fragmentShader, depthTest: false, depthWrite: false,
      uniforms: {tSource: {value: null}, uTexel: {value: new THREE.Vector2()}, uThreshold: {value: 0}},
      blending: additive ? THREE.CustomBlending : THREE.NoBlending,
      blendEquation: THREE.AddEquation, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
    });
    const down = material(DOWN_FS, false), up = material(UP_FS, true), quad = new THREE.FullScreenQuad(down);
    const pass = new THREE.Pass();
    pass.needsSwap = false;
    pass.strength = 0.8;
    pass.threshold = 0.6;
    let levels = [];
    pass.setSize = (width, height) => {
      for (const level of levels) level.dispose();
      levels = [];
      let w = width, h = height;
      for (let i = 0; i < 6 && w > 8 && h > 8; i++) {
        w = Math.max(1, w >> 1); h = Math.max(1, h >> 1);
        levels.push(new THREE.WebGLRenderTarget(w, h, {type: THREE.HalfFloatType, depthBuffer: false}));
      }
    };
    pass.render = (renderer, writeBuffer, readBuffer) => {
      grade.uniforms.uBloomStrength.value = levels.length ? pass.strength : 0;
      grade.uniforms.tBloom.value = levels.length ? levels[0].texture : null;
      if (!levels.length) return;
      const autoClear = renderer.autoClear;
      renderer.autoClear = false;
      let source = readBuffer;
      quad.material = down;
      levels.forEach((level, i) => {
        down.uniforms.tSource.value = source.texture;
        down.uniforms.uTexel.value.set(1 / source.width, 1 / source.height);
        down.uniforms.uThreshold.value = i === 0 ? pass.threshold : 0;
        renderer.setRenderTarget(level);
        quad.render(renderer);
        source = level;
      });
      quad.material = up;
      for (let i = levels.length - 1; i > 0; i--) {
        up.uniforms.tSource.value = levels[i].texture;
        up.uniforms.uTexel.value.set(0.5 / levels[i].width, 0.5 / levels[i].height);
        renderer.setRenderTarget(levels[i - 1]);
        quad.render(renderer);
      }
      renderer.autoClear = autoClear;
    };
    pass.dispose = () => {
      for (const level of levels) level.dispose();
      down.dispose(); up.dispose(); quad.dispose();
    };
    return pass;
  }

  // ---------------------------------------------------------------- renderer
  function create(canvas) {
    if (typeof THREE === "undefined") throw Error("The three.js library is missing from this page.");
    const gl = canvas.getContext("webgl2", {antialias: false, alpha: false, depth: true, stencil: false,
      preserveDrawingBuffer: true, powerPreference: "high-performance"});
    if (!gl) throw Error("WebGL2 is unavailable in this browser.");
    const renderer = new THREE.WebGLRenderer({canvas, context: gl});
    renderer.setPixelRatio(1);
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace;  // the grading pass encodes for the display
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.setClearColor(0x000000, 1);
    const hdr = renderer.extensions.has("EXT_color_buffer_float");

    // Uniforms shared by every scene material.
    const shared = {
      uViewport: {value: new THREE.Vector2(1, 1)},
      uFogDistance: {value: 900},
      uFogMix: {value: 0.85},
      uFogColor: {value: new THREE.Vector3()},
    };
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(34, 1, 1, 10000);

    const backgroundUniforms = {
      uTop: {value: new THREE.Vector3()}, uBottom: {value: new THREE.Vector3()},
      uGlow: {value: new THREE.Vector3()}, uGlowAt: {value: new THREE.Vector2(0.5, 1.1)},
    };
    const backgroundGeometry = new THREE.BufferGeometry();
    backgroundGeometry.setAttribute("position", corners([-1, -1, 0, 3, -1, 0, -1, 3, 0]));
    const background = new THREE.Mesh(backgroundGeometry, new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: BACKGROUND_VS, fragmentShader: BACKGROUND_FS,
      uniforms: backgroundUniforms, depthTest: false, depthWrite: false}));
    background.frustumCulled = false;
    scene.add(background);

    const spriteMaterial = (mode, halo, minPixels, options) => rawMaterial(SPRITE_VS, SPRITE_FS,
      {...shared, uMode: {value: mode}, uHalo: {value: halo}, uMinPixels: {value: minPixels}}, options);
    const lineMaterial = additive => rawMaterial(LINE_VS, LINE_FS, {...shared, uAdditive: {value: additive ? 1 : 0}}, {additive});
    const SPRITE_LAYOUT = [["aPosRadius", 4], ["aColor", 4]];
    const LINE_LAYOUT = [["aP0", 3], ["aP1", 3], ["aC0", 4], ["aC1", 4], ["aWidth", 1]];
    // Draw order: guides, glass beads (depth-writing), fronts, source, light.
    const batches = {
      lines: batch(LINE_CORNERS, LINE_LAYOUT, lineMaterial(false), 1),
      beads: batch(SPRITE_CORNERS, SPRITE_LAYOUT, spriteMaterial(1, 1, 1.6, {depthWrite: true}), 2),
      additiveLines: batch(LINE_CORNERS, LINE_LAYOUT, lineMaterial(true), 4),
      glows: batch(SPRITE_CORNERS, SPRITE_LAYOUT, spriteMaterial(0, 5, 2.2, {additive: true}), 5),
    };
    for (const b of Object.values(batches)) scene.add(b.mesh);

    const shellGeom = shellGeometry(96, 128), shells = [];
    function shellMesh() {
      const uniforms = {...shared, uKind: {value: 0}};
      for (const name of ["uOrigin", "uAxis", "uE1", "uE2", "uColor"]) uniforms[name] = {value: new THREE.Vector3()};
      for (const name of ["uA0", "uA1", "uTanBeta", "uRadius", "uK1", "uNg", "uIntensity", "uLambda", "uPeak", "uWidth", "uFloor"]) {
        uniforms[name] = {value: 0};
      }
      const mesh = new THREE.Mesh(shellGeom, rawMaterial(SHELL_VS, SHELL_FS, uniforms, {additive: true}));
      mesh.frustumCulled = false;
      mesh.renderOrder = 3;
      scene.add(mesh);
      return mesh;
    }
    function setShell(mesh, shell) {
      const u = mesh.material.uniforms, [e1, e2] = basis(shell.axis);
      u.uKind.value = shell.kind === "cone" ? 0 : 1;
      u.uOrigin.value.fromArray(shell.origin);
      u.uAxis.value.fromArray(shell.axis);
      u.uE1.value.fromArray(e1);
      u.uE2.value.fromArray(e2);
      u.uColor.value.fromArray(shell.color);
      u.uA0.value = shell.a0 || 0;
      u.uA1.value = shell.a1 || 0;
      u.uTanBeta.value = shell.tanBeta || 0;
      u.uRadius.value = shell.radius || 0;
      u.uK1.value = shell.k1 || 1;
      u.uNg.value = shell.ng || 1;
      u.uIntensity.value = shell.intensity;
      u.uLambda.value = shell.lambda;
      u.uPeak.value = shell.peak || 0;
      u.uWidth.value = shell.width || 1;
      u.uFloor.value = shell.floor ?? 1;
      mesh.visible = true;
    }

    // Post-processing: multisampled HDR scene -> bloom -> grading to the canvas.
    const samples = Math.min(4, renderer.capabilities.maxSamples || 0);
    const target = new THREE.WebGLRenderTarget(2, 2, {type: hdr ? THREE.HalfFloatType : THREE.UnsignedByteType, samples});
    const composer = new THREE.EffectComposer(renderer, target);
    composer.setPixelRatio(1);
    composer.addPass(new THREE.RenderPass(scene, camera));
    const grade = new THREE.ShaderPass(GRADE);
    const bloom = hdr ? bloomPass(grade) : null;
    if (bloom) composer.addPass(bloom);
    composer.addPass(grade);
    let size = [0, 0];

    function setCamera(cam, width, height) {
      camera.fov = cam.fov * 180 / Math.PI;
      camera.aspect = width / Math.max(1, height);
      camera.near = cam.near;
      camera.far = cam.far;
      camera.up.fromArray(cam.up || [0, 0, 1]);
      camera.position.fromArray(cam.eye);
      camera.lookAt(cam.target[0], cam.target[1], cam.target[2]);
      camera.updateProjectionMatrix();
      camera.updateMatrixWorld(true);
      const viewProj = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
      return {view: Float32Array.from(camera.matrixWorldInverse.elements), proj: Float32Array.from(camera.projectionMatrix.elements),
              viewProj: Float32Array.from(viewProj.elements), width, height};
    }

    // scene: {background, fog, beads, lines, glows, additiveLines, shells, halo, post}
    // Each call draws one complete, deterministic frame into the canvas.
    function render(desc, cam) {
      const w = canvas.width, h = canvas.height;
      if (size[0] !== w || size[1] !== h) {
        renderer.setSize(w, h, false);
        composer.setSize(w, h);
        size = [w, h];
      }
      const mats = setCamera(cam, w, h);
      shared.uViewport.value.set(w, h);
      shared.uFogDistance.value = desc.fog.distance;
      shared.uFogMix.value = desc.fog.amount;
      shared.uFogColor.value.fromArray(desc.fog.color);
      const bg = desc.background;
      backgroundUniforms.uTop.value.fromArray(bg.top);
      backgroundUniforms.uBottom.value.fromArray(bg.bottom);
      backgroundUniforms.uGlow.value.fromArray(bg.glow || [0, 0, 0]);
      backgroundUniforms.uGlowAt.value.fromArray(bg.glowAt || [0.5, 1.1]);
      batches.lines.set(desc.lines);
      batches.beads.set(desc.beads);
      batches.additiveLines.set(desc.additiveLines);
      batches.glows.set(desc.glows);
      batches.glows.mesh.material.uniforms.uHalo.value = desc.halo || 5;
      const list = desc.shells || [];
      while (shells.length < list.length) shells.push(shellMesh());
      shells.forEach((mesh, k) => {if (k < list.length) setShell(mesh, list[k]); else mesh.visible = false});
      const post = desc.post || {};
      grade.uniforms.uBloomStrength.value = 0;
      if (bloom) {
        bloom.enabled = (post.bloom ?? 0.8) > 0;
        bloom.strength = post.bloom ?? 0.8;
        bloom.threshold = post.threshold ?? 0.6;
      }
      grade.uniforms.uExposure.value = post.exposure ?? 1.2;
      grade.uniforms.uVignette.value = post.vignette ?? 0.35;
      composer.render(0);
      return mats;
    }

    function release() {
      for (const b of Object.values(batches)) b.dispose();
      for (const mesh of shells) mesh.material.dispose();
      shellGeom.dispose();
      backgroundGeometry.dispose();
      background.material.dispose();
      if (bloom) bloom.dispose();
      composer.dispose();
      renderer.dispose();
    }

    return {renderer, hdr, render, release};
  }

  return {create, project};
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = LightHitStudioGL;
}
