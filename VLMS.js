// VLMS.js — VLMS 2.0 Pro (single-file, production-ready baseline)
// Exposes window.VLMS = { Engine, Camera, Sphere, Material, Light }
// Author: asuali-cpu (Sourav)
// NOTE: Keep this file in repo root and serve via jsDelivr or GitHub Pages.

(function(global){
  'use strict';

  // ---------- Utilities ----------
  function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }
  function nowMs(){ return performance.now(); }

  // WebGL helpers
  function compileShader(gl, type, src){
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if(!gl.getShaderParameter(s, gl.COMPILE_STATUS)){
      const log = gl.getShaderInfoLog(s);
      console.error('Shader compile error', log, src);
      throw new Error(log);
    }
    return s;
  }
  function linkProgram(gl, vsSrc, fsSrc){
    const vs = compileShader(gl, gl.VERTEX_SHADER, vsSrc);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSrc);
    const p = gl.createProgram();
    gl.attachShader(p, vs); gl.attachShader(p, fs);
    gl.bindAttribLocation(p, 0, 'a_pos');
    gl.linkProgram(p);
    if(!gl.getProgramParameter(p, gl.LINK_STATUS)){
      const log = gl.getProgramInfoLog(p);
      console.error('Program link error', log);
      throw new Error(log);
    }
    return p;
  }
  function makeFullQuad(gl){
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
    return buf;
  }
  function makeTexture(gl,w,h,internal,format,type,linear=true){
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, linear?gl.LINEAR:gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, linear?gl.LINEAR:gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, null);
    return t;
  }
  function makeFBO(gl, texs){
    const f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    const attachments = [];
    for(let i=0;i<texs.length;i++){
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, texs[i], 0);
      attachments.push(gl.COLOR_ATTACHMENT0 + i);
    }
    gl.drawBuffers(attachments);
    const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if(st !== gl.FRAMEBUFFER_COMPLETE) console.warn('FBO incomplete', st);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return f;
  }

  // ---------- GLSL sources ----------
  const VS_FULL = `#version 300 es
  layout(location=0) in vec2 a_pos;
  out vec2 v_uv;
  void main(){ v_uv = a_pos*0.5 + 0.5; gl_Position = vec4(a_pos,0.0,1.0); }`;

  // Path tracer / primary hit shader
  // Outputs: outColor (rgba32f), outND (normal.xyz*0.5+0.5, depth/100), outMotion (prevUV - currUV)
  const FS_PATH = `#version 300 es
  precision highp float;
  in vec2 v_uv;
  layout(location=0) out vec4 outColor;
  layout(location=1) out vec4 outND;
  layout(location=2) out vec2 outMotion;

  uniform vec2 u_lowRes;
  uniform vec3 u_camPos;
  uniform vec3 u_camLook;
  uniform vec3 u_camUp;
  uniform vec3 u_prevCamPos;
  uniform vec3 u_prevCamLook;
  uniform vec3 u_prevCamUp;
  uniform float u_fov;
  uniform int u_frame;
  uniform int u_spp;
  uniform int u_bounces;
  uniform int u_sphereCount;
  uniform vec3 u_spherePos[16];
  uniform vec3 u_spherePrevPos[16];
  uniform float u_sphereR[16];
  uniform vec3 u_sphereColor[16];
  uniform float u_sphereRefl[16];

  uint wang_hash(uint seed){
    seed = (seed ^ 61u) ^ (seed >> 16);
    seed *= 9u;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15);
    return seed;
  }
  float rnd(inout uint seed){
    seed = wang_hash(seed);
    return float(seed) / 4294967296.0;
  }

  struct Ray { vec3 o; vec3 d; };
  struct Hit { float t; vec3 n; int idx; vec3 color; float refl; vec3 pos; };

  vec3 reflectv(vec3 I, vec3 N){ return I - 2.0 * dot(I,N) * N; }

  bool intersectSphere(vec3 ro, vec3 rd, vec3 c, float r, out float t, out vec3 n){
    vec3 oc = ro - c;
    float b = dot(oc, rd);
    float c2 = dot(oc,oc) - r*r;
    float disc = b*b - c2;
    if(disc < 0.0) return false;
    float sq = sqrt(disc);
    float t0 = -b - sq;
    float t1 = -b + sq;
    t = (t0 > 0.001) ? t0 : ((t1 > 0.001) ? t1 : -1.0);
    if(t < 0.0) return false;
    vec3 p = ro + rd * t;
    n = normalize(p - c);
    return true;
  }

  bool intersectPlane(vec3 ro, vec3 rd, out float t, out vec3 n){
    vec3 pn = vec3(0.0,1.0,0.0);
    float denom = dot(rd,pn);
    if(abs(denom) < 1e-5) return false;
    t = (-0.2 - ro.y) / rd.y;
    if(t < 0.001) return false;
    n = pn;
    return true;
  }

  Hit sceneIntersect(vec3 ro, vec3 rd){
    Hit best; best.t = 1e20; best.idx = -1; best.color=vec3(0.0); best.refl=0.0; best.n=vec3(0.0); best.pos=vec3(0.0);
    for(int i=0;i<16;i++){
      if(i>=u_sphereCount) break;
      float t; vec3 n;
      if(intersectSphere(ro, rd, u_spherePos[i], u_sphereR[i], t, n)){
        if(t < best.t){
          best.t = t; best.n = n; best.idx = i; best.color = u_sphereColor[i]; best.refl = u_sphereRefl[i];
          best.pos = ro + rd * t;
        }
      }
    }
    float tp; vec3 np;
    if(intersectPlane(ro, rd, tp, np)){
      if(tp < best.t){
        best.t = tp; best.n = np; best.idx = 99; best.color = vec3(0.95); best.refl = 0.1; best.pos = ro + rd*tp;
      }
    }
    return best;
  }

  void cameraBasis(vec3 look, vec3 up, out vec3 wz, out vec3 wx, out vec3 wy){
    wz = normalize(look);
    wx = normalize(cross(wz, up));
    wy = normalize(cross(wx, wz));
  }

  vec2 projectPointToUV(vec3 camPos, vec3 camLook, vec3 camUp, vec3 point, float fov, vec2 res){
    vec3 wz, wx, wy; cameraBasis(camLook, camUp, wz, wx, wy);
    vec3 toP = point - camPos;
    float z = dot(toP, wz);
    if(z <= 0.0001) return vec2(-10.0);
    float aspect = res.x / res.y;
    float scale = tan(radians(fov * 0.5));
    float x = dot(toP, wx) / (z * scale * aspect);
    float y = dot(toP, wy) / (z * scale);
    vec2 ndc = vec2(x, -y);
    return ndc * 0.5 + 0.5;
  }

  vec3 traceSample(Ray r, inout uint seed, int maxBounces){
    vec3 accum = vec3(0.0);
    vec3 throughput = vec3(1.0);
    for(int b=0;b<=maxBounces;b++){
      Hit h = sceneIntersect(r.o, r.d);
      if(h.t > 9.9e18){
        accum += throughput * vec3(0.06,0.08,0.12);
        break;
      }
      vec3 hit = h.pos; vec3 N = h.n;
      vec3 lightPos = vec3(0.0, 4.5, 2.0);
      vec3 toL = normalize(lightPos - hit);
      Hit sh = sceneIntersect(hit + N*0.001, toL);
      float inL = 1.0; if(sh.t < length(lightPos - hit)) inL = 0.0;
      float lam = max(dot(N,toL),0.0);
      accum += throughput * h.color * lam * inL * 1.2;

      float p = rnd(seed);
      if(h.refl > 0.0 && p < h.refl){
        vec3 refl = reflectv(r.d, N);
        r.o = hit + N*0.001;
        r.d = normalize(refl + vec3((rnd(seed)-0.5)*0.02));
        throughput *= 0.95;
      } else {
        // cosine-weighted hemisphere (simple)
        float u = rnd(seed), v = rnd(seed);
        float phi = 2.0 * 3.141592653589793 * u;
        float cosT = pow(1.0 - v, 1.0/(1.0+1.0));
        float sinT = sqrt(max(0.0, 1.0 - cosT*cosT));
        vec3 tt = abs(N.z) < 0.999 ? normalize(cross(N, vec3(0,0,1))) : normalize(cross(N, vec3(0,1,0)));
        vec3 bb = normalize(cross(N, tt));
        vec3 newDir = normalize(N * cosT + (tt * cos(phi) + bb * sin(phi)) * sinT);
        r.o = hit + N*0.001; r.d = newDir;
        throughput *= h.color * 0.9;
      }
    }
    return accum;
  }

  void main(){
    ivec2 pix = ivec2(gl_FragCoord.xy);
    uint seed = uint(u_frame) * 9781u + uint(pix.x)*1973u + uint(pix.y)*9277u;
    vec3 wz, wx, wy; cameraBasis(u_camLook, u_camUp, wz, wx, wy);
    float jx = rnd(seed)-0.5, jy = rnd(seed)-0.5;
    float aspect = u_lowRes.x / u_lowRes.y;
    float scale = tan(radians(u_fov * 0.5));
    vec2 ndc = ((vec2(gl_FragCoord.xy) + vec2(jx,jy)) / u_lowRes) * 2.0 - 1.0;
    vec2 screen = vec2(ndc.x * aspect * scale, -ndc.y * scale);
    Ray cam; cam.o = u_camPos; cam.d = normalize(wz + screen.x*wx + screen.y*wy);
    vec3 color = vec3(0.0);
    for(int s=0;s<u_spp;s++) color += traceSample(cam, seed, u_bounces);
    color /= max(1, u_spp);

    Hit last = sceneIntersect(cam.o, cam.d);
    vec3 nOut = vec3(0.0);
    float depth = 1e5;
    vec2 motion = vec2(0.0);
    if(last.t < 1e19){
      nOut = normalize(last.n);
      depth = last.t;
      vec2 curUV = projectPointToUV(u_camPos, u_camLook, u_camUp, last.pos, u_fov, u_lowRes);
      vec2 prevUV = projectPointToUV(u_prevCamPos, u_prevCamLook, u_prevCamUp, last.pos, u_fov, u_lowRes);
      if(prevUV.x >= 0.0 && prevUV.x <= 1.0 && prevUV.y >= 0.0 && prevUV.y <= 1.0){
        motion = prevUV - curUV;
      } else motion = vec2(0.0);
    }

    outColor = vec4(color,1.0);
    outND = vec4(nOut * 0.5 + 0.5, clamp(depth / 100.0, 0.0, 1.0));
    outMotion = motion;
  }`;

  // Temporal reprojection shader: reprojects prevAccum using motion (prevUV = currUV + motion), blends
  const FS_TEMPORAL = `#version 300 es
  precision highp float;
  in vec2 v_uv;
  out vec4 outAccum; // rgb = color, a = sampleCount (unused here but reserved)

  uniform sampler2D u_currColor; // low-res color (current)
  uniform sampler2D u_motion;    // low-res motion (prevUV - currUV)
  uniform sampler2D u_prevAccum; // previous low-res accumulation
  uniform float u_feedback; // blend factor for stable regions
  void main(){
    vec2 uv = v_uv;
    vec3 curr = texture(u_currColor, uv).rgb;
    vec2 motion = texture(u_motion, uv).xy;
    // prevUV = uv + motion
    vec2 prevUV = uv + motion;
    vec3 prevCol = vec3(0.0);
    float valid = 0.0;
    if(all(greaterThanEqual(prevUV, vec2(0.0))) && all(lessThanEqual(prevUV, vec2(1.0)))){
      prevCol = texture(u_prevAccum, prevUV).rgb;
      valid = 1.0;
    }
    // compute difference
    float diff = length(prevCol - curr);
    float adapt = smoothstep(0.0, 0.45, diff); // large diff -> smaller reliance on prev
    float alpha = mix(u_feedback, 0.97, clamp(1.0 - adapt, 0.0, 1.0));
    vec3 outc = mix(curr, prevCol, alpha * valid);
    outAccum = vec4(outc, 1.0);
  }`;

  // Denoise separable (horizontal/vertical)
  const FS_DENOISE = `#version 300 es
  precision highp float;
  in vec2 v_uv;
  out vec4 frag;
  uniform sampler2D u_acc; // low-res accumulated color
  uniform sampler2D u_nd;  // low-res normal+depth
  uniform vec2 u_texel;
  uniform int u_horizontal;
  void main(){
    vec2 uv = v_uv;
    vec3 center = texture(u_acc, uv).rgb;
    vec4 ndc = texture(u_nd, uv); vec3 nC = ndc.rgb; float dC = ndc.a;
    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for(int i=-3;i<=3;i++){
      vec2 off = (u_horizontal==1) ? vec2(float(i)*u_texel.x, 0.0) : vec2(0.0, float(i)*u_texel.y);
      vec3 c = texture(u_acc, uv + off).rgb;
      vec4 nd = texture(u_nd, uv + off); vec3 n = nd.rgb; float d = nd.a;
      float wc = exp(-dot(c-center,c-center)/(2.0*0.18*0.18));
      float wn = exp(-max(0.0,1.0-dot(nC,n))/(2.0*0.2*0.2));
      float wd = exp(-abs(d-dC)/(2.0*0.05*0.05));
      float w = wc * wn * wd;
      sum += c * w;
      wsum += w;
    }
    frag = vec4(sum / max(1e-6, wsum), 1.0);
  }`;

  // VLMS upscaler: bilateral gather in low-res domain + temporal blend with prev full
  const FS_VLMS = `#version 300 es
  precision highp float;
  in vec2 v_uv;
  out vec4 frag;
  uniform sampler2D u_low;  // low-res denoised color
  uniform sampler2D u_nd;   // low-res normal+depth
  uniform sampler2D u_prevFull; // previous final full-res
  uniform vec2 u_lowRes;
  uniform vec2 u_fullRes;
  uniform float u_feedback;
  uniform int u_enable;
  void main(){
    vec2 px = v_uv * u_fullRes;
    vec2 lowPx = floor(px * (u_lowRes / u_fullRes));
    vec2 lowUV = (lowPx + 0.5) / u_lowRes;
    vec3 base = texture(u_low, lowUV).rgb;
    vec4 nd = texture(u_nd, lowUV);
    vec3 n = nd.rgb * 2.0 - 1.0;
    if(u_enable == 0){ frag = vec4(base,1.0); return; }
    vec3 accum = vec3(0.0); float wsum = 0.0;
    for(int oy=-1; oy<=1; oy++) for(int ox=-1; ox<=1; ox++){
      vec2 off = vec2(float(ox), float(oy));
      vec2 sLow = ((floor(lowUV * u_lowRes) + off) + 0.5) / u_lowRes;
      vec3 c = texture(u_low, sLow).rgb;
      vec4 nd2 = texture(u_nd, sLow);
      vec3 n2 = nd2.rgb * 2.0 - 1.0;
      float sc = exp(-dot(c-base,c-base)/(2.0*0.2*0.2));
      float ss = exp(-dot(off,off)/(2.0*1.2*1.2));
      float sn = exp(-max(0.0,1.0-dot(n,n2))/0.2);
      float w = sc * ss * sn;
      accum += c * w; wsum += w;
    }
    vec3 up = accum / max(1e-6, wsum);
    vec3 hist = texture(u_prevFull, v_uv).rgb;
    float diff = length(hist - up);
    float adapt = smoothstep(0.0, 0.4, diff);
    float alpha = mix(u_feedback, 0.97, clamp(1.0 - adapt, 0.0, 1.0));
    vec3 outc = mix(up, hist, alpha);
    frag = vec4(outc,1.0);
  }`;

  const FS_BLIT = `#version 300 es
  precision highp float; in vec2 v_uv; out vec4 frag; uniform sampler2D u_tex; void main(){ frag = texture(u_tex, v_uv); }`;

  // ---------- Scene classes ----------
  class Camera {
    constructor(opts = {}){
      this.pos = opts.pos || [0,1.2,-4];
      this.look = opts.look || [0,0.9,3.0];
      this.up = opts.up || [0,1,0];
      this.fov = opts.fov || 60;
      this.prevPos = [...this.pos];
      this.prevLook = [...this.look];
      this.prevUp = [...this.up];
    }
    copyPrev(){ this.prevPos = [...this.pos]; this.prevLook = [...this.look]; this.prevUp = [...this.up]; }
  }

  class Sphere {
    constructor(opts = {}){
      this.pos = opts.pos || [0,0.5,3];
      this.prevPos = [...this.pos];
      this.r = opts.radius || 0.5;
      this.color = opts.color || [1,0.2,0.2];
      this.refl = (opts.refl !== undefined) ? opts.refl : 0.2;
    }
    copyPrev(){ this.prevPos = [...this.pos]; }
  }

  class Light {
    constructor(opts = {}){
      this.pos = opts.pos || [2,4,2];
      this.color = opts.color || [1,1,1];
    }
  }

  // ---------- Engine ----------
  class Engine {
    constructor(canvasOrSelector, opts = {}){
      this.canvas = (typeof canvasOrSelector === 'string') ? document.querySelector(canvasOrSelector) : (canvasOrSelector || document.createElement('canvas'));
      if(!this.canvas) throw new Error('Canvas not found');
      this.gl = this.canvas.getContext('webgl2', { antialias: false });
      if(!this.gl) throw new Error('WebGL2 required');
      this.gl.getExtension('EXT_color_buffer_float') || console.warn('EXT_color_buffer_float missing — precision lower');

      this.quad = makeFullQuad(this.gl);
      this.programPath = linkProgram(this.gl, VS_FULL, FS_PATH);
      this.programTemporal = linkProgram(this.gl, VS_FULL, FS_TEMPORAL);
      this.programDenoise = linkProgram(this.gl, VS_FULL, FS_DENOISE);
      this.programVLMS = linkProgram(this.gl, VS_FULL, FS_VLMS);
      this.programBlit = linkProgram(this.gl, VS_FULL, FS_BLIT);

      this.camera = new Camera(opts.camera || {});
      this.spheres = [];
      this.lights = [];

      this.scale = clamp(opts.scale || 0.5, 0.25, 1.0);
      this.spp = opts.spp || 1;
      this.bounces = opts.bounces || 2;
      this.enableVLMS = (opts.vlms !== undefined) ? !!opts.vlms : true;
      this.enableDenoise = (opts.denoise !== undefined) ? !!opts.denoise : true;
      this.feedback = opts.feedback || 0.85;

      this._frame = 0; this._accReset = true;
      this._accPing = 0; this._fullPing = 0;

      this._resizeTargets();
      window.addEventListener('resize', ()=>{ this._resizeTargets(); this.resetAccum(); });

      // default demo content (you can remove)
      if(opts.demo !== false){
        this.addSphere({ pos:[-1.1,0.9,1.5], radius:0.9, color:[0.85,0.2,0.2], refl:0.25 });
        this.addSphere({ pos:[0.6,0.6,0.8], radius:0.6, color:[0.15,0.45,0.9], refl:0.35 });
        this.addSphere({ pos:[1.6,0.9,2.2], radius:0.9, color:[0.95,0.95,1.0], refl:0.9 });
        this.addLight({ pos:[5,6,-2], color:[1,1,1] });
      }

      this._tick = this._tick.bind(this);
    }

    // public API
    addSphere(opts){ const s = new Sphere(opts); this.spheres.push(s); return s; }
    addLight(opts){ const l = new Light(opts); this.lights.push(l); return l; }
    resetAccum(){ this._accReset = true; }
    setQuality(level){
      // level: 'low' (0.4), 'medium' (0.6), 'high' (0.9), 'ultra' (1.0)
      const map = { low:0.4, medium:0.6, high:0.9, ultra:1.0 };
      this.scale = map[level] || parseFloat(level) || 0.5;
      this.scale = clamp(this.scale, 0.2, 1.0);
      this._resizeTargets(); this.resetAccum();
    }
    setSPP(n){ this.spp = Math.max(1,n); this.resetAccum(); }
    setBounces(n){ this.bounces = Math.max(0,n); this.resetAccum(); }

    _resizeTargets(){
      const gl = this.gl;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rc = this.canvas.getBoundingClientRect();
      this.fullW = Math.max(2, Math.floor(rc.width * dpr));
      this.fullH = Math.max(2, Math.floor(rc.height * dpr));
      this.lowW = Math.max(2, Math.floor(this.fullW * this.scale));
      this.lowH = Math.max(2, Math.floor(this.fullH * this.scale));

      // create low-res textures: color, nd, motion
      this.texLowColor = makeTexture(gl, this.lowW, this.lowH, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      this.texLowND = makeTexture(gl, this.lowW, this.lowH, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      this.texLowMotion = makeTexture(gl, this.lowW, this.lowH, gl.RG32F, gl.RG, gl.FLOAT);
      this.fboLow = makeFBO(gl, [this.texLowColor, this.texLowND, this.texLowMotion]);

      // accumulation ping-pong (low)
      this.texAccumA = makeTexture(gl, this.lowW, this.lowH, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      this.fboAccumA = makeFBO(gl, [this.texAccumA]);
      this.texAccumB = makeTexture(gl, this.lowW, this.lowH, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      this.fboAccumB = makeFBO(gl, [this.texAccumB]);

      // denoised low
      this.texDenoised = makeTexture(gl, this.lowW, this.lowH, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      this.fboDenoised = makeFBO(gl, [this.texDenoised]);

      // final full-res ping-pong
      this.texFullA = makeTexture(gl, this.fullW, this.fullH, gl.RGBA16F, gl.RGBA, gl.FLOAT);
      this.fboFullA = makeFBO(gl, [this.texFullA]);
      this.texFullB = makeTexture(gl, this.fullW, this.fullH, gl.RGBA16F, gl.RGBA, gl.FLOAT);
      this.fboFullB = makeFBO(gl, [this.texFullB]);

      this._accPing = 0; this._fullPing = 0;
    }

    _bindQuadAndUse(prog){
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.useProgram(prog);
    }

    // pass 1: path trace to low-res (color, ND, motion)
    _passPath(){
      const gl = this.gl;
      this._bindQuadAndUse(this.programPath);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboLow);
      gl.viewport(0,0,this.lowW,this.lowH);

      // uniforms
      gl.uniform2f(gl.getUniformLocation(this.programPath, 'u_lowRes'), this.lowW, this.lowH);
      gl.uniform3fv(gl.getUniformLocation(this.programPath, 'u_camPos'), this.camera.pos);
      gl.uniform3fv(gl.getUniformLocation(this.programPath, 'u_camLook'), this.camera.look);
      gl.uniform3fv(gl.getUniformLocation(this.programPath, 'u_camUp'), this.camera.up);
      gl.uniform3fv(gl.getUniformLocation(this.programPath, 'u_prevCamPos'), this.camera.prevPos);
      gl.uniform3fv(gl.getUniformLocation(this.programPath, 'u_prevCamLook'), this.camera.prevLook);
      gl.uniform3fv(gl.getUniformLocation(this.programPath, 'u_prevCamUp'), this.camera.prevUp);
      gl.uniform1f(gl.getUniformLocation(this.programPath, 'u_fov'), this.camera.fov);
      gl.uniform1i(gl.getUniformLocation(this.programPath, 'u_frame'), this._frame);
      gl.uniform1i(gl.getUniformLocation(this.programPath, 'u_spp'), this.spp);
      gl.uniform1i(gl.getUniformLocation(this.programPath, 'u_bounces'), this.bounces);
      gl.uniform1i(gl.getUniformLocation(this.programPath, 'u_sphereCount'), this.spheres.length);

      // push sphere arrays (up to 16)
      for(let i=0;i<16;i++){
        const posLoc = gl.getUniformLocation(this.programPath, `u_spherePos[${i}]`);
        const prevLoc = gl.getUniformLocation(this.programPath, `u_spherePrevPos[${i}]`);
        const rLoc = gl.getUniformLocation(this.programPath, `u_sphereR[${i}]`);
        const cLoc = gl.getUniformLocation(this.programPath, `u_sphereColor[${i}]`);
        const reflLoc = gl.getUniformLocation(this.programPath, `u_sphereRefl[${i}]`);
        if(i < this.spheres.length){
          const s = this.spheres[i];
          gl.uniform3fv(posLoc, s.pos);
          gl.uniform3fv(prevLoc, s.prevPos);
          gl.uniform1f(rLoc, s.r);
          gl.uniform3fv(cLoc, s.color);
          gl.uniform1f(reflLoc, s.refl);
        } else {
          gl.uniform3fv(posLoc, [0,-9999,0]); gl.uniform3fv(prevLoc, [0,-9999,0]); gl.uniform1f(rLoc,0.0);
          gl.uniform3fv(cLoc,[0,0,0]); gl.uniform1f(reflLoc,0.0);
        }
      }

      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    // pass 2: temporal reprojection (low-res)
    _passTemporal(){
      const gl = this.gl;
      const dstFBO = (this._accPing === 0) ? this.fboAccumA : this.fboAccumB;
      this._bindQuadAndUse(this.programTemporal);
      gl.bindFramebuffer(gl.FRAMEBUFFER, dstFBO);
      gl.viewport(0,0,this.lowW,this.lowH);

      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.texLowColor);
      gl.uniform1i(gl.getUniformLocation(this.programTemporal, 'u_currColor'), 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.texLowMotion);
      gl.uniform1i(gl.getUniformLocation(this.programTemporal, 'u_motion'), 1);
      // previous accumulation
      const prevAccumTex = (this._accPing === 0) ? this.texAccumB : this.texAccumA;
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, prevAccumTex);
      gl.uniform1i(gl.getUniformLocation(this.programTemporal, 'u_prevAccum'), 2);
      gl.uniform1f(gl.getUniformLocation(this.programTemporal, 'u_feedback'), this.feedback);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    // pass 3: denoise separable (horizontal then vertical)
    _passDenoise(){
      const gl = this.gl;
      if(!this.enableDenoise){
        // copy accum -> denoised
        this._bindQuadAndUse(this.programBlit);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboDenoised);
        gl.viewport(0,0,this.lowW,this.lowH);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, (this._accPing===0)?this.texAccumA:this.texAccumB);
        gl.uniform1i(gl.getUniformLocation(this.programBlit, 'u_tex'), 0);
        gl.drawArrays(gl.TRIANGLES,0,6);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        return;
      }
      // horizontal pass: accum -> denoised
      this._bindQuadAndUse(this.programDenoise);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboDenoised);
      gl.viewport(0,0,this.lowW,this.lowH);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, (this._accPing===0)?this.texAccumA:this.texAccumB);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.texLowND);
      gl.uniform1i(gl.getUniformLocation(this.programDenoise, 'u_acc'), 0);
      gl.uniform1i(gl.getUniformLocation(this.programDenoise, 'u_nd'), 1);
      gl.uniform2f(gl.getUniformLocation(this.programDenoise, 'u_texel'), 1.0/this.lowW, 1.0/this.lowH);
      gl.uniform1i(gl.getUniformLocation(this.programDenoise, 'u_horizontal'), 1);
      gl.drawArrays(gl.TRIANGLES,0,6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      // vertical pass: denoised -> accumulation-other
      const dstFBO = (this._accPing===0) ? this.fboAccumB : this.fboAccumA;
      this._bindQuadAndUse(this.programDenoise);
      gl.bindFramebuffer(gl.FRAMEBUFFER, dstFBO);
      gl.viewport(0,0,this.lowW,this.lowH);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.texDenoised);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.texLowND);
      gl.uniform1i(gl.getUniformLocation(this.programDenoise, 'u_acc'), 0);
      gl.uniform1i(gl.getUniformLocation(this.programDenoise, 'u_nd'), 1);
      gl.uniform2f(gl.getUniformLocation(this.programDenoise, 'u_texel'), 1.0/this.lowW, 1.0/this.lowH);
      gl.uniform1i(gl.getUniformLocation(this.programDenoise, 'u_horizontal'), 0);
      gl.drawArrays(gl.TRIANGLES,0,6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    // pass 4: VLMS upsample -> full-res with temporal blend
    _passVLMS(){
      const gl = this.gl;
      const dstFBO = (this._fullPing===0) ? this.fboFullA : this.fboFullB;
      this._bindQuadAndUse(this.programVLMS);
      gl.bindFramebuffer(gl.FRAMEBUFFER, dstFBO);
      gl.viewport(0,0,this.fullW,this.fullH);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, (this._accPing===0)?this.texAccumA:this.texAccumB);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.texLowND);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, (this._fullPing===0)?this.texFullA:this.texFullB);
      gl.uniform1i(gl.getUniformLocation(this.programVLMS, 'u_low'), 0);
      gl.uniform1i(gl.getUniformLocation(this.programVLMS, 'u_nd'), 1);
      gl.uniform1i(gl.getUniformLocation(this.programVLMS, 'u_prevFull'), 2);
      gl.uniform2f(gl.getUniformLocation(this.programVLMS, 'u_lowRes'), this.lowW, this.lowH);
      gl.uniform2f(gl.getUniformLocation(this.programVLMS, 'u_fullRes'), this.fullW, this.fullH);
      gl.uniform1f(gl.getUniformLocation(this.programVLMS, 'u_feedback'), this.feedback);
      gl.uniform1i(gl.getUniformLocation(this.programVLMS, 'u_enable'), this.enableVLMS?1:0);
      gl.drawArrays(gl.TRIANGLES,0,6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    // present final
    _present(){
      const gl = this.gl;
      this._bindQuadAndUse(this.programBlit);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0,0,this.fullW,this.fullH);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, (this._fullPing===0)?this.texFullA:this.texFullB);
      gl.uniform1i(gl.getUniformLocation(this.programBlit, 'u_tex'), 0);
      gl.drawArrays(gl.TRIANGLES,0,6);
    }

    // main loop tick
    _tick(time){
      // handle canvas size change
      const rc = this.canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cw = Math.max(2, Math.floor(rc.width * dpr));
      const ch = Math.max(2, Math.floor(rc.height * dpr));
      if(this.canvas.width !== cw || this.canvas.height !== ch){
        this.canvas.width = cw; this.canvas.height = ch; this._resizeTargets(); this._accReset = true;
      }

      // if accumulation reset, copy prev transforms to avoid ghosting
      if(this._accReset){
        this.camera.copyPrev();
        for(const s of this.spheres) s.copyPrev();
      }

      // user animation example (you can remove)
      if(this.spheres.length>0){
        const t = time * 0.001;
        // gentle motion to generate motion vectors
        this.spheres[0].pos[0] = Math.sin(t*0.6) * 0.9;
        this.spheres[0].pos[2] = 1.4 + Math.cos(t*0.4)*0.3;
      }

      // pipeline
      this._passPath();
      this._passTemporal();
      this._passDenoise();
      this._passVLMS();
      this._present();

      // ping-pong swaps
      this._accPing = 1 - this._accPing;
      this._fullPing = 1 - this._fullPing;
      this._accReset = false;

      // update prev transforms for next frame
      for(const s of this.spheres) s.copyPrev();
      this.camera.copyPrev();

      this._frame++;
      requestAnimationFrame(this._tick);
    }

    start(){ requestAnimationFrame(this._tick); }
    stop(){ /* could implement */ }
  }

  // export
  global.VLMS = { Engine, Camera, Sphere, Light };

})(window);
