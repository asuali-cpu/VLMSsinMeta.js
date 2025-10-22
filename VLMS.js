// vlms.js - VLMS 3.0 Pro (script build)
// Exposes window.VLMS = { App, Camera, Sphere, createApp }
// Minimal dependency: WebGL2 + EXT_color_buffer_float
(function(global){
  'use strict';

  // ---------- Helpers ----------
  function isWebGL2(gl){ return !!gl && typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext; }
  function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }
  function createShader(gl, type, src){
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if(!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error('Shader error:', gl.getShaderInfoLog(s));
      throw new Error('Shader compile failed');
    }
    return s;
  }
  function createProgram(gl, vsSrc, fsSrc){
    const p = gl.createProgram();
    const vs = createShader(gl, gl.VERTEX_SHADER, vsSrc);
    const fs = createShader(gl, gl.FRAGMENT_SHADER, fsSrc);
    gl.attachShader(p, vs); gl.attachShader(p, fs);
    gl.bindAttribLocation(p, 0, 'a_pos');
    gl.linkProgram(p);
    if(!gl.getProgramParameter(p, gl.LINK_STATUS)){
      console.error('Program link failed:', gl.getProgramInfoLog(p));
      throw new Error('Program link failed');
    }
    return p;
  }

  // Fullscreen quad buffer reused
  function makeFullQuad(gl){
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
    return buf;
  }

  // ---------- Small shader sources (shortened, practical) ----------
  const VS_FULL = `#version 300 es
  layout(location=0) in vec2 a_pos;
  out vec2 v_uv;
  void main(){ v_uv = a_pos*0.5 + 0.5; gl_Position = vec4(a_pos,0.0,1.0); }`;

  // Path tracer (low-res) - outputs color (RGBA32F), normal+depth (RGBA32F), motion (RG32F)
  // NOTE: shader simplified for clarity — production engines separate many concerns.
  const FS_PATH = `#version 300 es
  precision highp float;
  in vec2 v_uv;
  layout(location=0) out vec4 outColor;
  layout(location=1) out vec4 outNormalDepth;
  layout(location=2) out vec2 outMotion;

  uniform vec2 u_lowRes;
  uniform vec3 u_camPos, u_camLook, u_camUp;
  uniform vec3 u_prevCamPos, u_prevCamLook, u_prevCamUp;
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

  // RNG
  uint wang(uint s){ s=(s^61u) ^ (s>>16); s *= 9u; s = s ^ (s>>4); s *= 0x27d4eb2du; s = s ^ (s>>15); return s; }
  float rnd(inout uint s){ s = wang(s); return float(s)/4294967296.0; }

  struct Ray{ vec3 o; vec3 d; };
  struct Hit{ float t; vec3 n; int idx; vec3 color; float refl; vec3 pos; };

  vec3 reflectv(vec3 I, vec3 N){ return I - 2.0 * dot(I,N) * N; }

  bool intersectSphere(vec3 ro, vec3 rd, vec3 c, float r, out float t, out vec3 n){
    vec3 oc = ro - c; float b = dot(oc, rd); float c2 = dot(oc,oc) - r*r; float dis = b*b - c2;
    if(dis < 0.0) return false; float sq = sqrt(dis); float t0 = -b - sq; float t1 = -b + sq; t = (t0>0.001? t0 : (t1>0.001? t1 : -1.0));
    if(t<0.0) return false; vec3 p = ro + rd*t; n = normalize(p - c); return true;
  }

  bool intersectPlane(vec3 ro, vec3 rd, out float t, out vec3 n){
    vec3 pn = vec3(0.0,1.0,0.0); float denom = dot(rd,pn); if(abs(denom)<1e-5) return false; t = (-0.2 - ro.y)/rd.y; if(t<0.001) return false; n = pn; return true;
  }

  Hit sceneIntersect(vec3 ro, vec3 rd){
    Hit best; best.t = 1e20; best.idx = -1; best.color = vec3(0.0); best.refl = 0.0; best.n = vec3(0.0); best.pos = vec3(0.0);
    for(int i=0;i<16;i++){ if(i>=u_sphereCount) break; float t; vec3 n; if(intersectSphere(ro,rd,u_spherePos[i],u_sphereR[i],t,n)){ if(t<best.t){ best.t=t; best.n=n; best.idx=i; best.color=u_sphereColor[i]; best.refl=u_sphereRefl[i]; best.pos = ro+rd*t; } } }
    float tp; vec3 np; if(intersectPlane(ro,rd,tp,np)){ if(tp < best.t){ best.t = tp; best.n = np; best.idx = 99; best.color = vec3(0.95); best.refl = 0.1; best.pos = ro+rd*tp; } }
    return best;
  }

  // project point to UV in given camera basis
  vec2 projectToUV(vec3 camPos, vec3 camLook, vec3 camUp, vec3 p, float fov, vec2 res){
    vec3 wz = normalize(camLook);
    vec3 wx = normalize(cross(wz, camUp));
    vec3 wy = normalize(cross(wx, wz));
    vec3 toP = p - camPos;
    float z = dot(toP, wz); if(z<=0.0001) return vec2(-10.0);
    float aspect = res.x/res.y; float scale = tan(radians(fov*0.5));
    float x = dot(toP, wx)/(z*scale*aspect); float y = dot(toP, wy)/(z*scale);
    vec2 ndc = vec2(x, -y); return ndc*0.5 + 0.5;
  }

  vec3 traceSample(Ray r, inout uint seed){
    vec3 accum = vec3(0.0); vec3 throughput = vec3(1.0);
    for(int b=0;b<=u_bounces;b++){
      Hit h = sceneIntersect(r.o, r.d);
      if(h.t > 9.9e18){ accum += throughput * vec3(0.06,0.08,0.12); break; }
      vec3 hit = h.pos; vec3 N = h.n;
      vec3 light = vec3(0.0,4.5,2.0);
      vec3 toL = normalize(light - hit); Hit sh = sceneIntersect(hit + N*0.001, toL);
      float inL = (sh.t < length(light - hit))? 0.0 : 1.0;
      float lam = max(dot(N,toL),0.0); accum += throughput * h.color * lam * inL * 1.2;
      // material simple mix of diff + reflect
      float p = rnd(seed);
      if(h.refl>0.0 && p < h.refl){
        vec3 refl = reflectv(r.d,N); r.o = hit + N*0.001; r.d = normalize(refl + vec3((rnd(seed)-0.5)*0.02));
        throughput *= 0.95;
      } else {
        // cosine-weighted hemisphere
        float u = rnd(seed), v = rnd(seed);
        float phi = 2.0*3.141592653589793*u; float cosT = pow(1.0 - v, 1.0/(1.0+1.0)); float sinT = sqrt(max(0.0,1.0-cosT*cosT));
        vec3 tt = abs(N.z) < 0.999 ? normalize(cross(N,vec3(0,0,1))) : normalize(cross(N,vec3(0,1,0)));
        vec3 bb = normalize(cross(N,tt));
        vec3 newDir = normalize(N*cosT + (tt*cos(phi)+bb*sin(phi))*sinT);
        r.o = hit + N*0.001; r.d = newDir; throughput *= h.color * 0.9;
      }
    }
    return accum;
  }

  void main(){
    ivec2 pix = ivec2(gl_FragCoord.xy);
    uint seed = uint(u_frame) * 9781u + uint(pix.x)*1973u + uint(pix.y)*9277u;
    // build camera basis
    vec3 wz = normalize(u_camLook);
    vec3 wx = normalize(cross(wz, u_camUp));
    vec3 wy = normalize(cross(wx, wz));
    // jitter
    float jx = rnd(seed)-0.5; float jy = rnd(seed)-0.5;
    float aspect = u_lowRes.x/u_lowRes.y;
    float scale = tan(radians(u_fov*0.5));
    vec2 ndc = ((vec2(gl_FragCoord.xy) + vec2(jx,jy)) / u_lowRes) * 2.0 - 1.0;
    vec2 screen = vec2(ndc.x * aspect * scale, -ndc.y * scale);

    Ray cam; cam.o = u_camPos; cam.d = normalize(wz + screen.x*wx + screen.y*wy);
    vec3 col = vec3(0.0);
    for(int s=0;s<u_spp;s++) col += traceSample(cam, seed);
    col /= max(1,u_spp);

    Hit h = sceneIntersect(cam.o, cam.d);
    vec3 nOut = vec3(0.0); float depth = 1e5; vec2 motion = vec2(0.0);
    if(h.t < 1e19){
      nOut = normalize(h.n); depth = h.t;
      vec2 curUV = projectToUV(u_camPos,u_camLook,u_camUp,h.pos,u_fov,u_lowRes);
      vec2 prevUV = projectToUV(u_prevCamPos,u_prevCamLook,u_prevCamUp,h.pos,u_fov,u_lowRes);
      if(all(greaterThanEqual(prevUV,vec2(0.0))) && all(lessThanEqual(prevUV,vec2(1.0)))) motion = prevUV - curUV;
    }
    outColor = vec4(col,1.0);
    outNormalDepth = vec4(nOut*0.5+0.5, clamp(depth/100.0,0.0,1.0));
    outMotion = motion;
  }`;

  // Denoise shader (single pass separable simplified)
  const FS_DENOISE = `#version 300 es
  precision highp float; in vec2 v_uv; out vec4 frag;
  uniform sampler2D u_acc; uniform sampler2D u_nd; uniform vec2 u_texel; uniform int u_horizontal;
  void main(){
    vec3 C = texture(u_acc, v_uv).rgb;
    vec4 ND = texture(u_nd, v_uv);
    vec3 N = ND.rgb; float D = ND.a;
    vec3 acc = vec3(0.0); float wsum = 0.0;
    for(int i=-3;i<=3;i++){
      vec2 off = u_horizontal==1 ? vec2(float(i)*u_texel.x,0.0) : vec2(0.0,float(i)*u_texel.y);
      vec3 c = texture(u_acc, v_uv+off).rgb;
      vec4 nd = texture(u_nd, v_uv+off); vec3 n2 = nd.rgb; float d2 = nd.a;
      float wc = exp(-dot(c-C,c-C)/(2.0*0.18*0.18));
      float wn = exp(-max(0.0,1.0-dot(N,n2))/(2.0*0.2*0.2));
      float wd = exp(-abs(d2-D)/(2.0*0.05*0.05));
      float w = wc*wn*wd; acc += c*w; wsum += w;
    }
    frag = vec4(acc/max(1e-6,wsum),1.0);
  }`;

  // VLMS upscaler (bilateral + temporal)
  const FS_VLMS = `#version 300 es
  precision highp float; in vec2 v_uv; out vec4 frag;
  uniform sampler2D u_low; uniform sampler2D u_nd; uniform sampler2D u_prev; uniform vec2 u_lowRes; uniform vec2 u_fullRes; uniform float u_feedback; uniform int u_enable;
  vec2 mapLow(vec2 uv){ vec2 px = uv*u_fullRes; vec2 lowPx = floor(px*(u_lowRes/u_fullRes)); return (lowPx+0.5)/u_lowRes; }
  void main(){
    vec2 luv = mapLow(v_uv);
    vec3 base = texture(u_low, luv).rgb; vec4 nd = texture(u_nd, luv); vec3 n = nd.rgb*2.0-1.0;
    if(u_enable==0){ frag = vec4(base,1.0); return; }
    vec3 accum = vec3(0.0); float wsum = 0.0;
    for(int oy=-1; oy<=1; oy++){ for(int ox=-1; ox<=1; ox++){
      vec2 off = vec2(float(ox),float(oy)); vec2 s = ((floor(luv*u_lowRes)+off)+0.5)/u_lowRes;
      vec3 c = texture(u_low,s).rgb; vec4 nd2 = texture(u_nd,s); vec3 n2 = nd2.rgb*2.0-1.0;
      float sc = exp(-dot(c-base,c-base)/(2.0*0.2*0.2)); float ss = exp(-dot(off,off)/(2.0*1.2*1.2)); float sn = exp(-max(0.0,1.0-dot(n,n2))/0.2);
      float w = sc*ss*sn; accum += c*w; wsum += w;
    } }
    vec3 up = accum/max(1e-6,wsum);
    vec3 hist = texture(u_prev, v_uv).rgb;
    float diff = length(hist-up); float adapt = smoothstep(0.0,0.4,diff);
    float alpha = mix(u_feedback, 0.95, clamp(1.0-adapt,0.0,1.0));
    vec3 outc = mix(up, hist, alpha);
    frag = vec4(outc,1.0);
  }`;

  // simple blit
  const FS_BLIT = `#version 300 es
  precision highp float; in vec2 v_uv; out vec4 frag; uniform sampler2D u_tex; void main(){ frag = texture(u_tex, v_uv); }`;

  // ---------- Public classes ----------
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
    setPos(x,y,z){ this.pos = [x,y,z]; }
    setLook(x,y,z){ this.look = [x,y,z]; }
  }

  class Sphere {
    constructor(opts = {}){
      this.pos = opts.pos || [0,0.5,3.0];
      this.prevPos = [...this.pos];
      this.r = opts.radius || 0.5;
      this.color = opts.color || [1,0.2,0.2];
      this.refl = (typeof opts.reflect === 'number') ? opts.reflect : 0.2;
    }
  }

  class App {
    constructor(opts = {}){
      this.canvas = opts.canvas ? (typeof opts.canvas === 'string' ? document.querySelector(opts.canvas) : opts.canvas) : (()=>{ const c=document.createElement('canvas'); document.body.appendChild(c); return c; })();
      this.gl = this.canvas.getContext('webgl2', {antialias: false});
      if(!isWebGL2(this.gl)) throw new Error('WebGL2 required');
      if(!this.gl.getExtension('EXT_color_buffer_float')) console.warn('EXT_color_buffer_float missing — precision may be low');

      this.quad = makeFullQuad(this.gl);
      this.programPath = createProgram(this.gl, VS_FULL, FS_PATH);
      this.programDenoise = createProgram(this.gl, VS_FULL, FS_DENOISE);
      this.programVLMS = createProgram(this.gl, VS_FULL, FS_VLMS);
      this.programBlit = createProgram(this.gl, VS_FULL, FS_BLIT);

      this.camera = new Camera(opts.camera||{});
      this.spheres = [];
      this.frame = 0;
      this.anim = true;
      this.scale = clamp(opts.scale||0.5, 0.25, 1.0);
      this.spp = opts.spp || 1;
      this.bounces = opts.bounces || 2;
      this.enableVLMS = (opts.vlms !== undefined) ? !!opts.vlms : true;
      this.enableDenoise = (opts.denoise !== undefined) ? !!opts.denoise : true;

      // textures / FBOs
      this._makeTargets();
      // state
      this._accReset = true;
      this._tick = this._tick.bind(this);
      this._lastT = performance.now();
    }

    // create framebuffers & textures sized by canvas & scale
    _makeTargets(){
      const gl = this.gl;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = this.canvas.getBoundingClientRect();
      this.fullW = Math.max(2, Math.floor(rect.width * dpr));
      this.fullH = Math.max(2, Math.floor(rect.height * dpr));
      this.lowW = Math.max(2, Math.floor(this.fullW * this.scale));
      this.lowH = Math.max(2, Math.floor(this.fullH * this.scale));

      const makeTex = (w,h,internal,format,type)=>{
        const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D,0,internal,w,h,0,format,type,null);
        return t;
      };

      // low-res targets
      this.texLowColor = makeTex(this.lowW,this.lowH, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      this.texLowND = makeTex(this.lowW,this.lowH, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      this.texLowMotion = makeTex(this.lowW,this.lowH, gl.RG32F, gl.RG, gl.FLOAT);
      this.fboLow = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboLow);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texLowColor, 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.texLowND, 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, this.texLowMotion, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      // accumulation ping-pong (low-res)
      this.texAccumA = makeTex(this.lowW,this.lowH, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      this.fboAccumA = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboAccumA); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texAccumA, 0);
      this.texAccumB = makeTex(this.lowW,this.lowH, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      this.fboAccumB = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboAccumB); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texAccumB, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      // denoised
      this.texDenoised = makeTex(this.lowW,this.lowH, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      this.fboDenoised = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboDenoised); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texDenoised, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      // full-res final ping-pong
      this.texFullA = makeTex(this.fullW,this.fullH, gl.RGBA16F, gl.RGBA, gl.FLOAT);
      this.fboFullA = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboFullA); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texFullA, 0);
      this.texFullB = makeTex(this.fullW,this.fullH, gl.RGBA16F, gl.RGBA, gl.FLOAT);
      this.fboFullB = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboFullB); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texFullB, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      // set initial swap
      this._accPing = 0; this._fullPing = 0;
    }

    addSphere(opts){ const s = new Sphere(opts); this.spheres.push(s); return s; }

    resetAccum(){ this._accReset = true; }

    // internal: bind quad & enable attribute 0
    _bindQuad(prog){
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quad); 
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
      gl.useProgram(prog);
    }

    // main render passes (path -> temporal -> denoise -> vlms -> present)
    _passPath(){
      const gl = this.gl; this._bindQuad(this.programPath);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboLow);
      gl.viewport(0,0,this.lowW,this.lowH);

      // set uniforms quickly (manual)
      const set3 = (name, arr)=>{ const loc = gl.getUniformLocation(this.programPath,name); if(loc) gl.uniform3fv(loc, arr); };
      gl.uniform2f(gl.getUniformLocation(this.programPath,'u_lowRes'), this.lowW, this.lowH);
      set3('u_camPos', this.camera.pos); set3('u_camLook', this.camera.look); set3('u_camUp', this.camera.up);
      set3('u_prevCamPos', this.camera.prevPos); set3('u_prevCamLook', this.camera.prevLook); set3('u_prevCamUp', this.camera.prevUp);
      gl.uniform1f(gl.getUniformLocation(this.programPath,'u_fov'), this.camera.fov);
      gl.uniform1i(gl.getUniformLocation(this.programPath,'u_frame'), this.frame);
      gl.uniform1i(gl.getUniformLocation(this.programPath,'u_spp'), this.spp);
      gl.uniform1i(gl.getUniformLocation(this.programPath,'u_bounces'), this.bounces);
      gl.uniform1i(gl.getUniformLocation(this.programPath,'u_sphereCount'), this.spheres.length);

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
          gl.uniform3fv(posLoc, [0,-9999,0]); gl.uniform3fv(prevLoc, [0,-9999,0]); gl.uniform1f(rLoc, 0.0); gl.uniform3fv(cLoc,[0,0,0]); gl.uniform1f(reflLoc,0.0);
        }
      }

      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    _passTemporal(){
      const gl = this.gl; this._bindQuad(this.programBlit); // for simplicity reusing blit with manual shader (could be dedicated)
      // very small reprojection/accum: we do a cheap copy here to accumA
      // In a full implementation we'd have a shader reprojecting prevAccum using motion texture.
      // Here we copy lowColor->accumA as starting step; denoiser and VLMS will use history in full pass.
      const dstFBO = (this._accPing===0) ? this.fboAccumA : this.fboAccumB;
      gl.bindFramebuffer(gl.FRAMEBUFFER, dstFBO); gl.viewport(0,0,this.lowW,this.lowH);
      // use blit shader to copy lowColor to accum
      gl.useProgram(this.programBlit);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.texLowColor);
      gl.uniform1i(gl.getUniformLocation(this.programBlit,'u_tex'), 0);
      gl.drawArrays(gl.TRIANGLES,0,6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    _passDenoise(){
      if(!this.enableDenoise){ // copy accum -> denoised
        const gl = this.gl; this._bindQuad(this.programBlit);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboDenoised); gl.viewport(0,0,this.lowW,this.lowH);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this._accPing===0 ? this.texAccumA : this.texAccumB);
        gl.uniform1i(gl.getUniformLocation(this.programBlit,'u_tex'), 0);
        gl.drawArrays(gl.TRIANGLES,0,6);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        return;
      }
      // separable simplified: horizontal then vertical using same shader with flag
      const gl = this.gl;
      // horizontal -> fboDenoised
      this._bindQuad(this.programDenoise);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboDenoised); gl.viewport(0,0,this.lowW,this.lowH);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this._accPing===0 ? this.texAccumA : this.texAccumB);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.texLowND);
      gl.uniform1i(gl.getUniformLocation(this.programDenoise,'u_acc'), 0);
      gl.uniform1i(gl.getUniformLocation(this.programDenoise,'u_nd'), 1);
      gl.uniform2f(gl.getUniformLocation(this.programDenoise,'u_texel'), 1/this.lowW, 1/this.lowH);
      gl.uniform1i(gl.getUniformLocation(this.programDenoise,'u_horizontal'), 1);
      gl.drawArrays(gl.TRIANGLES,0,6);
      // vertical -> accumB (reuse accumB as final low-res denoised)
      this._bindQuad(this.programDenoise);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this._accPing===0 ? this.fboAccumB : this.fboAccumA);
      gl.viewport(0,0,this.lowW,this.lowH);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.texDenoised);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.texLowND);
      gl.uniform1i(gl.getUniformLocation(this.programDenoise,'u_acc'), 0);
      gl.uniform1i(gl.getUniformLocation(this.programDenoise,'u_nd'), 1);
      gl.uniform2f(gl.getUniformLocation(this.programDenoise,'u_texel'), 1/this.lowW, 1/this.lowH);
      gl.uniform1i(gl.getUniformLocation(this.programDenoise,'u_horizontal'), 0);
      gl.drawArrays(gl.TRIANGLES,0,6);
      // result now in the opposite accum texture
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    _passVLMS(){
      const gl = this.gl; this._bindQuad(this.programVLMS);
      const dstFBO = this._fullPing===0 ? this.fboFullA : this.fboFullB;
      gl.bindFramebuffer(gl.FRAMEBUFFER, dstFBO); gl.viewport(0,0,this.fullW,this.fullH);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this._accPing===0? this.texAccumA : this.texAccumB);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.texLowND);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this._fullPing===0? this.texFullA : this.texFullB);
      gl.uniform1i(gl.getUniformLocation(this.programVLMS,'u_low'), 0);
      gl.uniform1i(gl.getUniformLocation(this.programVLMS,'u_nd'), 1);
      gl.uniform1i(gl.getUniformLocation(this.programVLMS,'u_prev'), 2);
      gl.uniform2f(gl.getUniformLocation(this.programVLMS,'u_lowRes'), this.lowW, this.lowH);
      gl.uniform2f(gl.getUniformLocation(this.programVLMS,'u_fullRes'), this.fullW, this.fullH);
      gl.uniform1f(gl.getUniformLocation(this.programVLMS,'u_feedback'), 0.85);
      gl.uniform1i(gl.getUniformLocation(this.programVLMS,'u_enable'), this.enableVLMS ? 1 : 0);
      gl.drawArrays(gl.TRIANGLES,0,6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    _present(){
      const gl = this.gl; this._bindQuad(this.programBlit);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0,0,this.fullW,this.fullH);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this._fullPing===0 ? this.texFullA : this.texFullB);
      gl.uniform1i(gl.getUniformLocation(this.programBlit,'u_tex'), 0);
      gl.drawArrays(gl.TRIANGLES,0,6);
    }

    // main tick
    _tick(time){
      // handle resize
      const rect = this.canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cw = Math.floor(rect.width * dpr), ch = Math.floor(rect.height * dpr);
      if(this.canvas.width !== cw || this.canvas.height !== ch){ this.canvas.width = cw; this.canvas.height = ch; this._makeTargets(); this._accReset = true; }

      // transfer prev positions (if reset then prev = curr to avoid motion)
      if(this._accReset){
        for(const s of this.spheres) s.prevPos = [...s.pos];
        this.camera.prevPos = [...this.camera.pos]; this.camera.prevLook = [...this.camera.look]; this.camera.prevUp = [...this.camera.up];
      }

      // simple animation placeholder (user can drive their own)
      if(this.anim){
        const t = time * 0.001;
        if(this.spheres.length>0){
          this.spheres[0].pos[0] = Math.sin(t*0.6) * 0.9;
          this.spheres[0].pos[2] = 3.0 + Math.cos(t*0.4)*0.4;
        }
      }

      // path trace into low-res
      this._passPath();

      // temporal accumulation (we simplified to copying current low color into accumulation texture)
      this._passTemporal();

      // denoise
      this._passDenoise();

      // VLMS upscale and temporal blend
      this._passVLMS();

      // present to screen
      this._present();

      // ping-pong swaps
      this._accPing = 1 - this._accPing;
      this._fullPing = 1 - this._fullPing;
      this._accReset = false;
      this.frame++;
      this._lastT = time;
      requestAnimationFrame(this._tick);
    }

    start(){ requestAnimationFrame(this._tick); }
    stop(){ /* not implemented: set flag */ }
  }

  // convenience factory
  function createApp(opts){ return new App(opts); }

  // expose
  const VLMS = { App, Camera, Sphere, createApp };
  global.VLMS = VLMS;
  // also support module consumers if environment supports
  try{ if(typeof module !== 'undefined') module.exports = VLMS; } catch(e){}
})(window);