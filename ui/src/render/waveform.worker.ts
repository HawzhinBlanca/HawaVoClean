/// <reference lib="webworker" />
// Waveform renderer — runs in a dedicated Worker on an OffscreenCanvas with
// WebGL2. Nothing here touches React: the main thread posts data / geometry /
// playhead messages and this worker redraws on its own animation frame.

import { createFbo, createProgram, deleteFbo, hexToRgb, type Fbo } from './glutil';
import { timeTicksIn } from './ticks';
import type { WaveKind, WaveMsg, WaveOutMsg, WaveSlot } from './waveformProtocol';

const ctxSelf = self as unknown as DedicatedWorkerGlobalScope;

// ---------------------------------------------------------------------------
// Shaders

const VS_QUAD = `#version 300 es
precision highp float;
const vec2 P[6] = vec2[6](vec2(-1.,-1.),vec2(1.,-1.),vec2(-1.,1.),vec2(-1.,1.),vec2(1.,-1.),vec2(1.,1.));
out vec2 v_uv;
void main(){ vec2 p = P[gl_VertexID]; v_uv = p*0.5+0.5; gl_Position = vec4(p,0.,1.); }`;

// Display background: graphite gradient, faint amplitude grid, vignette.
const FS_BG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform vec2 u_res;
uniform float u_margin;   // vertical margin fraction (0..0.5)
out vec4 o;
float hline(float y, float target, float px){ return 1.0 - smoothstep(0.0, px, abs(y-target)); }
void main(){
  vec2 uv = v_uv;
  vec3 top = vec3(0.043,0.051,0.063);
  vec3 bot = vec3(0.024,0.027,0.034);
  vec3 c = mix(bot, top, uv.y);
  // vignette
  vec2 d = (uv-0.5)*vec2(1.0,1.35);
  float vig = 1.0 - 0.42*smoothstep(0.35,1.05,length(d));
  c *= vig;
  // amplitude grid: center, ±0.5 (−6 dB), ±0.25 (−12 dB), edges
  float pxy = 1.0/u_res.y;
  float hf = 0.5-u_margin;
  float g = 0.0;
  g += 0.10*hline(uv.y,0.5,pxy*1.0);
  g += 0.045*(hline(uv.y,0.5+hf*0.5,pxy)+hline(uv.y,0.5-hf*0.5,pxy));
  g += 0.03*(hline(uv.y,0.5+hf*0.25,pxy)+hline(uv.y,0.5-hf*0.25,pxy));
  g += 0.05*(hline(uv.y,0.5+hf,pxy)+hline(uv.y,0.5-hf,pxy));
  c += vec3(g);
  // subtle glass sheen near the top edge
  c += vec3(0.012)*smoothstep(0.86,1.0,uv.y);
  o = vec4(c,1.0);
}`;

const VS_WAVE = `#version 300 es
precision highp float;
layout(location=0) in vec2 a_pos;  // device px
layout(location=1) in float a_amp; // signed normalized amplitude at this vertex
uniform vec2 u_res;
out float v_amp;
void main(){
  v_amp = a_amp;
  vec2 ndc = (a_pos / u_res)*2.0-1.0;
  gl_Position = vec4(ndc.x, -ndc.y, 0., 1.);
}`;

const FS_WAVE = `#version 300 es
precision highp float;
in float v_amp;
uniform vec3 u_core;   // colour at the centre line
uniform vec3 u_edge;   // colour at the peaks
uniform float u_aCore; // alpha at centre
uniform float u_aEdge; // alpha at peaks
uniform float u_gamma;
out vec4 o;
void main(){
  float a = clamp(abs(v_amp),0.0,1.0);
  float t = pow(a,u_gamma);
  vec3 c = mix(u_core,u_edge,t);
  float alpha = mix(u_aCore,u_aEdge,t);
  o = vec4(c*alpha, alpha); // premultiplied
}`;

const VS_SOLID = `#version 300 es
precision highp float;
layout(location=0) in vec2 a_pos;
uniform vec2 u_res;
void main(){
  vec2 ndc = (a_pos / u_res)*2.0-1.0;
  gl_Position = vec4(ndc.x, -ndc.y, 0., 1.);
}`;

const FS_SOLID = `#version 300 es
precision highp float;
uniform vec4 u_color; // premultiplied
out vec4 o;
void main(){ o = u_color; }`;

const FS_BLUR = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_tex;
uniform vec2 u_dir; // texel step
out vec4 o;
void main(){
  vec4 c = texture(u_tex, v_uv)*0.2270270270;
  c += texture(u_tex, v_uv + u_dir*1.3846153846)*0.3162162162;
  c += texture(u_tex, v_uv - u_dir*1.3846153846)*0.3162162162;
  c += texture(u_tex, v_uv + u_dir*3.2307692308)*0.0702702703;
  c += texture(u_tex, v_uv - u_dir*3.2307692308)*0.0702702703;
  o = c;
}`;

const FS_COMPOSITE = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_tex;
uniform float u_gain;
out vec4 o;
void main(){ o = texture(u_tex, v_uv)*u_gain; }`;

// ---------------------------------------------------------------------------
// State

interface WaveData {
  min: Float32Array;
  max: Float32Array;
  rms: Float32Array | null;
  /** Seconds this envelope covers. */
  start: number;
  end: number;
}

interface KindData {
  base: WaveData | null;
  detail: WaveData | null;
}

interface Strip {
  buf: WebGLBuffer;
  count: number;
}

interface KindGeom {
  peaks: Strip | null;
  rms: Strip | null;
}

const PALETTE = {
  original: {
    core: hexToRgb('#ffc978'),
    edge: hexToRgb('#ff9a1f'),
    rmsCore: hexToRgb('#ffdfa6'),
    rmsEdge: hexToRgb('#ffb347'),
  },
  cleaned: {
    core: hexToRgb('#6fdcff'),
    edge: hexToRgb('#1fb8ea'),
    rmsCore: hexToRgb('#a9ecff'),
    rmsEdge: hexToRgb('#39d0ff'),
  },
} as const;

let gl: WebGL2RenderingContext | null = null;
let canvas: OffscreenCanvas | null = null;
let cssW = 0;
let cssH = 0;
let dpr = 1;
let W = 0;
let H = 0;

let progBg: WebGLProgram;
let progWave: WebGLProgram;
let progSolid: WebGLProgram;
let progBlur: WebGLProgram;
let progComposite: WebGLProgram;
let vaoEmpty: WebGLVertexArrayObject;
let vaoWave: WebGLVertexArrayObject;
let vaoSolid: WebGLVertexArrayObject;
let solidBuf: WebGLBuffer;
let glowA: Fbo | null = null;
let glowB: Fbo | null = null;

const data: Record<WaveKind, KindData> = {
  original: { base: null, detail: null },
  cleaned: { base: null, detail: null },
};
const geom: Record<WaveKind, KindGeom> = {
  original: { peaks: null, rms: null },
  cleaned: { peaks: null, rms: null },
};
let geomDirty = true;
let viewStart = 0;
let viewEnd = 0;
let playhead = 0;
let playheadVisible = false;
let hoverX: number | null = null;
let highlight: { start: number; end: number } | null = null;
let unitBounds: Float32Array = new Float32Array(0);
let focus: WaveKind = 'original';

let frameReq = 0;
let needsRender = false;

const MARGIN_FRAC = 0.06;

function post(msg: WaveOutMsg): void {
  ctxSelf.postMessage(msg);
}

function scheduleRender(): void {
  needsRender = true;
  if (frameReq) return;
  const raf = (ctxSelf as unknown as { requestAnimationFrame?: (cb: () => void) => number })
    .requestAnimationFrame;
  if (typeof raf === 'function') {
    frameReq = raf.call(ctxSelf, () => {
      frameReq = 0;
      if (needsRender) render();
    });
  } else {
    frameReq = setTimeout(() => {
      frameReq = 0;
      if (needsRender) render();
    }, 16) as unknown as number;
  }
}

// ---------------------------------------------------------------------------
// Init / resize

function init(c: OffscreenCanvas, w: number, h: number, ratio: number): void {
  canvas = c;
  const ctx = c.getContext('webgl2', {
    antialias: true,
    alpha: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: true,
    preserveDrawingBuffer: false,
    powerPreference: 'high-performance',
  });
  if (!ctx) {
    post({ type: 'ready', webgl2: false });
    return;
  }
  gl = ctx;
  progBg = createProgram(gl, VS_QUAD, FS_BG);
  progWave = createProgram(gl, VS_WAVE, FS_WAVE);
  progSolid = createProgram(gl, VS_SOLID, FS_SOLID);
  progBlur = createProgram(gl, VS_QUAD, FS_BLUR);
  progComposite = createProgram(gl, VS_QUAD, FS_COMPOSITE);

  const ve = gl.createVertexArray();
  const vw = gl.createVertexArray();
  const vs = gl.createVertexArray();
  const sb = gl.createBuffer();
  if (!ve || !vw || !vs || !sb) throw new Error('vao alloc failed');
  vaoEmpty = ve;
  vaoWave = vw;
  vaoSolid = vs;
  solidBuf = sb;

  gl.bindVertexArray(vaoSolid);
  gl.bindBuffer(gl.ARRAY_BUFFER, solidBuf);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
  gl.bindVertexArray(null);

  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.CULL_FACE);
  resize(w, h, ratio);
  post({ type: 'ready', webgl2: true });
}

function resize(w: number, h: number, ratio: number): void {
  if (!gl || !canvas) return;
  cssW = Math.max(1, w);
  cssH = Math.max(1, h);
  dpr = Math.max(1, Math.min(3, ratio || 1));
  W = Math.max(1, Math.round(cssW * dpr));
  H = Math.max(1, Math.round(cssH * dpr));
  canvas.width = W;
  canvas.height = H;
  if (glowA) deleteFbo(gl, glowA);
  if (glowB) deleteFbo(gl, glowB);
  const gw = Math.max(1, Math.round(W / 2));
  const gh = Math.max(1, Math.round(H / 2));
  glowA = createFbo(gl, gw, gh);
  glowB = createFbo(gl, gw, gh);
  geomDirty = true;
  scheduleRender();
}

// ---------------------------------------------------------------------------
// Geometry

/**
 * Build a filled strip (triangle strip) for per-column [top,bottom] envelope.
 * Returns interleaved [x, y, amp] per vertex; two vertices per column.
 *
 * The buckets in `lo`/`hi` cover [dataStart, dataEnd] seconds; the strip covers
 * the visible window [vStart, vEnd]. Columns outside the data collapse to the
 * centre line, so a detail window that no longer covers the view simply reads
 * as empty rather than as garbage.
 */
function buildEnvelopeStrip(
  lo: Float32Array,
  hi: Float32Array,
  dataStart: number,
  dataEnd: number,
  vStart: number,
  vEnd: number,
  width: number,
  height: number,
  gain: number,
): Float32Array {
  const n = lo.length;
  const cols = width;
  const out = new Float32Array(cols * 2 * 3);
  const centerY = height / 2;
  const halfH = (height / 2) * (1 - MARGIN_FRAC * 2) * gain;
  const span = Math.max(1e-12, vEnd - vStart);
  const bucketsPerS = n / Math.max(1e-12, dataEnd - dataStart);
  let o = 0;
  for (let x = 0; x < cols; x++) {
    const t0 = vStart + (x / cols) * span;
    const t1 = vStart + ((x + 1) / cols) * span;
    const b0 = (t0 - dataStart) * bucketsPerS;
    const b1 = (t1 - dataStart) * bucketsPerS;
    let vmin = 0;
    let vmax = 0;
    if (b0 >= n || b1 <= 0) {
      vmin = 0;
      vmax = 0;
    } else if (b1 - b0 >= 1) {
      const i0 = Math.max(0, Math.floor(b0));
      const i1 = Math.min(n, Math.ceil(b1));
      vmin = Infinity;
      vmax = -Infinity;
      for (let i = i0; i < i1; i++) {
        const a = lo[i] ?? 0;
        const b = hi[i] ?? 0;
        if (a < vmin) vmin = a;
        if (b > vmax) vmax = b;
      }
      if (!Number.isFinite(vmin)) vmin = 0;
      if (!Number.isFinite(vmax)) vmax = 0;
    } else {
      const bc = (b0 + b1) / 2 - 0.5;
      const i = Math.floor(bc);
      const f = bc - i;
      const ia = Math.min(n - 1, Math.max(0, i));
      const ib = Math.min(n - 1, Math.max(0, i + 1));
      vmin = (lo[ia] ?? 0) * (1 - f) + (lo[ib] ?? 0) * f;
      vmax = (hi[ia] ?? 0) * (1 - f) + (hi[ib] ?? 0) * f;
    }
    // Guarantee at least a hairline so silence still reads as a line.
    const minPx = 0.6;
    let topY = centerY - vmax * halfH;
    let botY = centerY - vmin * halfH;
    if (botY - topY < minPx) {
      const mid = (topY + botY) / 2;
      topY = mid - minPx / 2;
      botY = mid + minPx / 2;
    }
    const px = x + 0.5;
    out[o++] = px;
    out[o++] = topY;
    out[o++] = vmax;
    out[o++] = px;
    out[o++] = botY;
    out[o++] = vmin;
  }
  return out;
}

function uploadStrip(verts: Float32Array, prev: Strip | null): Strip {
  if (!gl) throw new Error('no gl');
  const buf = prev?.buf ?? gl.createBuffer();
  if (!buf) throw new Error('buffer alloc failed');
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, verts, gl.DYNAMIC_DRAW);
  return { buf, count: verts.length / 3 };
}

const COVER_EPS = 1e-4;

/**
 * Pick the envelope to draw for a deck: the windowed detail whenever it still
 * covers the visible range, otherwise the whole-file base. This is what makes
 * a zoom feel instant — the base redraw lands on the same frame as the wheel
 * event and the finer detail swaps in when the fetch returns.
 */
function pickSource(kind: WaveKind): WaveData | null {
  const d = data[kind];
  const det = d.detail;
  if (det && det.start <= viewStart + COVER_EPS && det.end >= viewEnd - COVER_EPS) return det;
  return d.base;
}

function rebuildGeometry(): void {
  if (!gl) return;
  for (const kind of ['original', 'cleaned'] as const) {
    const d = pickSource(kind);
    const g = geom[kind];
    if (!d || viewEnd - viewStart <= 0) {
      if (g.peaks) gl.deleteBuffer(g.peaks.buf);
      if (g.rms) gl.deleteBuffer(g.rms.buf);
      g.peaks = null;
      g.rms = null;
      continue;
    }
    const peaks = buildEnvelopeStrip(d.min, d.max, d.start, d.end, viewStart, viewEnd, W, H, 1);
    g.peaks = uploadStrip(peaks, g.peaks);
    if (d.rms) {
      const neg = new Float32Array(d.rms.length);
      for (let i = 0; i < neg.length; i++) neg[i] = -(d.rms[i] ?? 0);
      const rms = buildEnvelopeStrip(neg, d.rms, d.start, d.end, viewStart, viewEnd, W, H, 1);
      g.rms = uploadStrip(rms, g.rms);
    } else if (g.rms) {
      gl.deleteBuffer(g.rms.buf);
      g.rms = null;
    }
  }
  geomDirty = false;
}

// ---------------------------------------------------------------------------
// Drawing helpers

function drawStrip(
  strip: Strip,
  core: readonly [number, number, number],
  edge: readonly [number, number, number],
  aCore: number,
  aEdge: number,
  gamma: number,
  res: [number, number],
): void {
  if (!gl) return;
  gl.useProgram(progWave);
  gl.uniform2f(gl.getUniformLocation(progWave, 'u_res'), res[0], res[1]);
  gl.uniform3f(gl.getUniformLocation(progWave, 'u_core'), core[0], core[1], core[2]);
  gl.uniform3f(gl.getUniformLocation(progWave, 'u_edge'), edge[0], edge[1], edge[2]);
  gl.uniform1f(gl.getUniformLocation(progWave, 'u_aCore'), aCore);
  gl.uniform1f(gl.getUniformLocation(progWave, 'u_aEdge'), aEdge);
  gl.uniform1f(gl.getUniformLocation(progWave, 'u_gamma'), gamma);
  gl.bindVertexArray(vaoWave);
  gl.bindBuffer(gl.ARRAY_BUFFER, strip.buf);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 12, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 12, 8);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, strip.count);
  gl.bindVertexArray(null);
}

const solidScratch = new Float32Array(4096);
let solidScratchLen = 0;

function solidBegin(): void {
  solidScratchLen = 0;
}
function solidRect(x0: number, y0: number, x1: number, y1: number): void {
  if (solidScratchLen + 12 > solidScratch.length) return;
  const s = solidScratch;
  let o = solidScratchLen;
  s[o++] = x0; s[o++] = y0; s[o++] = x1; s[o++] = y0; s[o++] = x0; s[o++] = y1;
  s[o++] = x0; s[o++] = y1; s[o++] = x1; s[o++] = y0; s[o++] = x1; s[o++] = y1;
  solidScratchLen = o;
}
function solidFlush(r: number, g: number, b: number, a: number): void {
  if (!gl || solidScratchLen === 0) return;
  gl.useProgram(progSolid);
  gl.uniform2f(gl.getUniformLocation(progSolid, 'u_res'), W, H);
  gl.uniform4f(gl.getUniformLocation(progSolid, 'u_color'), r * a, g * a, b * a, a);
  gl.bindVertexArray(vaoSolid);
  gl.bindBuffer(gl.ARRAY_BUFFER, solidBuf);
  gl.bufferData(gl.ARRAY_BUFFER, solidScratch.subarray(0, solidScratchLen), gl.STREAM_DRAW);
  gl.drawArrays(gl.TRIANGLES, 0, solidScratchLen / 2);
  gl.bindVertexArray(null);
  solidScratchLen = 0;
}

function vline(xDev: number, widthDev: number, y0: number, y1: number): void {
  const half = widthDev / 2;
  const xc = Math.round(xDev - half) + half; // pixel-snapped
  solidRect(xc - half, y0, xc + half, y1);
}

function timeToX(t: number): number {
  const span = viewEnd - viewStart;
  return span > 0 ? ((t - viewStart) / span) * W : 0;
}

// ---------------------------------------------------------------------------
// Render

function render(): void {
  needsRender = false;
  if (!gl || !glowA || !glowB) return;
  if (geomDirty) rebuildGeometry();

  const hasCleaned = geom.cleaned.peaks !== null;
  const hasOriginal = geom.original.peaks !== null;

  // 1. background
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, W, H);
  gl.disable(gl.BLEND);
  gl.useProgram(progBg);
  gl.uniform2f(gl.getUniformLocation(progBg, 'u_res'), W, H);
  gl.uniform1f(gl.getUniformLocation(progBg, 'u_margin'), MARGIN_FRAC);
  gl.bindVertexArray(vaoEmpty);
  gl.drawArrays(gl.TRIANGLES, 0, 6);

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  // 2. grid lines (ticks + unit bounds + highlight)
  solidBegin();
  const ticks = timeTicksIn(viewStart, viewEnd, cssW, 72);
  for (const tk of ticks) {
    if (!tk.major) vline(timeToX(tk.time), 1, 0, H);
  }
  solidFlush(1, 1, 1, 0.035);
  solidBegin();
  for (const tk of ticks) {
    if (tk.major) vline(timeToX(tk.time), 1, 0, H);
  }
  solidFlush(1, 1, 1, 0.075);

  if (unitBounds.length > 0) {
    solidBegin();
    for (let i = 0; i < unitBounds.length; i++) {
      const t = unitBounds[i] ?? 0;
      if (t < viewStart || t > viewEnd) continue;
      // Flush in batches: the scratch holds ~340 rects and long clips can
      // have far more unit boundaries than that.
      if (solidScratchLen + 12 > solidScratch.length) solidFlush(0.6, 0.75, 0.95, 0.09);
      vline(timeToX(t), 1, H * 0.08, H * 0.92);
    }
    solidFlush(0.6, 0.75, 0.95, 0.09);
  }

  if (highlight && highlight.end > viewStart && highlight.start < viewEnd) {
    const hx0 = Math.max(0, timeToX(highlight.start));
    const hx1 = Math.min(W, timeToX(highlight.end));
    solidBegin();
    solidRect(hx0, 0, hx1, H);
    solidFlush(1, 1, 1, 0.05);
    solidBegin();
    if (highlight.start >= viewStart) vline(timeToX(highlight.start), 1, 0, H);
    if (highlight.end <= viewEnd) vline(timeToX(highlight.end), 1, 0, H);
    solidFlush(1, 1, 1, 0.16);
  }

  // 3. glow pass (half-res): fills → blur → additive composite
  if (hasOriginal || hasCleaned) {
    const gw = glowA.width;
    const gh = glowA.height;
    gl.bindFramebuffer(gl.FRAMEBUFFER, glowA.fb);
    gl.viewport(0, 0, gw, gh);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    // Geometry is in full-res device px; the vertex shader normalises it to
    // NDC with u_res = [W, H], so the same buffers draw into the half-res
    // target through the smaller viewport.
    const res: [number, number] = [W, H];
    if (hasOriginal && geom.original.peaks) {
      const dim = hasCleaned ? (focus === 'original' ? 0.55 : 0.22) : 1;
      drawStrip(geom.original.peaks, PALETTE.original.core, PALETTE.original.edge, 0.8 * dim, 0.5 * dim, 0.8, res);
    }
    if (hasCleaned && geom.cleaned.peaks) {
      const dim = focus === 'cleaned' ? 1 : 0.5;
      drawStrip(geom.cleaned.peaks, PALETTE.cleaned.core, PALETTE.cleaned.edge, 0.8 * dim, 0.5 * dim, 0.8, res);
    }
    // blur H: A → B
    gl.disable(gl.BLEND);
    gl.useProgram(progBlur);
    gl.bindVertexArray(vaoEmpty);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(gl.getUniformLocation(progBlur, 'u_tex'), 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, glowB.fb);
    gl.bindTexture(gl.TEXTURE_2D, glowA.tex);
    gl.uniform2f(gl.getUniformLocation(progBlur, 'u_dir'), 1.6 / gw, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    // blur V: B → A
    gl.bindFramebuffer(gl.FRAMEBUFFER, glowA.fb);
    gl.bindTexture(gl.TEXTURE_2D, glowB.tex);
    gl.uniform2f(gl.getUniformLocation(progBlur, 'u_dir'), 0, 1.6 / gh);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    // composite additive onto the screen
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(progComposite);
    gl.bindTexture(gl.TEXTURE_2D, glowA.tex);
    gl.uniform1i(gl.getUniformLocation(progComposite, 'u_tex'), 0);
    gl.uniform1f(gl.getUniformLocation(progComposite, 'u_gain'), 0.6);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }

  // 4. crisp fills
  const res: [number, number] = [W, H];
  if (hasOriginal && geom.original.peaks) {
    const dim = hasCleaned ? (focus === 'original' ? 0.7 : 0.4) : 1;
    drawStrip(geom.original.peaks, PALETTE.original.core, PALETTE.original.edge, 0.58 * dim, 0.36 * dim, 0.7, res);
    if (geom.original.rms) {
      drawStrip(geom.original.rms, PALETTE.original.rmsCore, PALETTE.original.rmsEdge, 0.95 * dim, 0.85 * dim, 1.0, res);
    }
  }
  if (hasCleaned && geom.cleaned.peaks) {
    const dim = focus === 'cleaned' ? 1 : 0.55;
    drawStrip(geom.cleaned.peaks, PALETTE.cleaned.core, PALETTE.cleaned.edge, 0.5 * dim, 0.3 * dim, 0.7, res);
    if (geom.cleaned.rms) {
      drawStrip(geom.cleaned.rms, PALETTE.cleaned.rmsCore, PALETTE.cleaned.rmsEdge, 0.9 * dim, 0.8 * dim, 1.0, res);
    }
  }

  // 5. centre line on top (very faint) so silence reads as a hairline
  solidBegin();
  solidRect(0, Math.round(H / 2) - 0.5 * dpr, W, Math.round(H / 2) + 0.5 * dpr);
  solidFlush(1, 1, 1, 0.06);

  // 6. hover cursor
  if (hoverX !== null) {
    solidBegin();
    vline(hoverX * dpr, 1, 0, H);
    solidFlush(1, 1, 1, 0.22);
  }

  // 7. playhead
  if (playheadVisible && viewEnd > viewStart && playhead >= viewStart && playhead <= viewEnd) {
    const x = timeToX(playhead);
    solidBegin();
    vline(x, 5 * dpr, 0, H);
    solidFlush(1, 1, 1, 0.06);
    solidBegin();
    vline(x, 3 * dpr, 0, H);
    solidFlush(1, 1, 1, 0.1);
    solidBegin();
    vline(x, 1 * dpr, 0, H);
    solidFlush(1, 1, 1, 0.92);
    // little head marker
    solidBegin();
    solidRect(x - 3 * dpr, 0, x + 3 * dpr, 2 * dpr);
    solidFlush(1, 1, 1, 0.9);
  }
}

// ---------------------------------------------------------------------------
// Message loop

ctxSelf.onmessage = (ev: MessageEvent<WaveMsg>) => {
  const msg = ev.data;
  try {
    switch (msg.type) {
      case 'init':
        init(msg.canvas, msg.width, msg.height, msg.dpr);
        break;
      case 'resize':
        resize(msg.width, msg.height, msg.dpr);
        break;
      case 'data': {
        const slot: WaveSlot = msg.slot;
        data[msg.kind][slot] = {
          min: msg.min,
          max: msg.max,
          rms: msg.rms,
          start: msg.start,
          end: msg.end,
        };
        // First base envelope with no view yet: show the whole thing.
        if (slot === 'base' && viewEnd - viewStart <= 0) {
          viewStart = msg.start;
          viewEnd = msg.end;
        }
        geomDirty = true;
        scheduleRender();
        break;
      }
      case 'clear':
        if (msg.slot) data[msg.kind][msg.slot] = null;
        else data[msg.kind] = { base: null, detail: null };
        geomDirty = true;
        scheduleRender();
        break;
      case 'view':
        if (viewStart !== msg.start || viewEnd !== msg.end) {
          viewStart = msg.start;
          viewEnd = msg.end;
          geomDirty = true;
          scheduleRender();
        }
        break;
      case 'playhead':
        if (playhead !== msg.time || playheadVisible !== msg.visible) {
          playhead = msg.time;
          playheadVisible = msg.visible;
          scheduleRender();
        }
        break;
      case 'hover':
        if (hoverX !== msg.x) {
          hoverX = msg.x;
          scheduleRender();
        }
        break;
      case 'highlight':
        highlight = msg.range;
        scheduleRender();
        break;
      case 'units':
        unitBounds = msg.bounds;
        scheduleRender();
        break;
      case 'focus':
        if (focus !== msg.kind) {
          focus = msg.kind;
          scheduleRender();
        }
        break;
    }
  } catch (e) {
    post({ type: 'error', message: e instanceof Error ? e.message : String(e) });
  }
};
