/// <reference lib="webworker" />
// Waveform renderer — runs in a dedicated Worker on an OffscreenCanvas with
// WebGL2. Nothing here touches React: the main thread posts data / geometry /
// playhead / palette messages and this worker redraws on its own animation
// frame.
//
// Everything is drawn on the GPU. There is no per-pixel JS: the CPU only
// resamples the envelope into one record per device column (or one point per
// sample at 1:1) and hands those to instanced draw calls whose fragment
// shaders compute exact analytic coverage — that is where the anti-aliasing
// comes from, so edges stay smooth at every zoom without MSAA.

import { createFbo, createProgramU, deleteFbo, type Fbo, type ProgramU } from './glutil';
import { timeTicksIn } from './ticks';
import {
  DEFAULT_WAVE_PALETTE,
  type WaveDeckColors,
  type WaveKind,
  type WaveMsg,
  type WaveOutMsg,
  type WavePalette,
  type WaveRgb,
  type WaveSlot,
} from './waveformProtocol';

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
uniform vec3 u_top;
uniform vec3 u_bot;
uniform vec3 u_grid;
out vec4 o;
float hline(float y, float target, float px){ return 1.0 - smoothstep(0.0, px, abs(y-target)); }
void main(){
  vec2 uv = v_uv;
  vec3 c = mix(u_bot, u_top, uv.y);
  // vignette
  vec2 d = (uv-0.5)*vec2(1.0,1.35);
  float vig = 1.0 - 0.42*smoothstep(0.35,1.05,length(d));
  c *= vig;
  // amplitude grid: ±0.5 (−6 dB), ±0.25 (−12 dB), edges. The centre line is
  // drawn later, on top of the waveform, so it is not repeated here.
  float pxy = 1.0/u_res.y;
  float hf = 0.5-u_margin;
  float g = 0.0;
  g += 0.042*(hline(uv.y,0.5+hf*0.5,pxy)+hline(uv.y,0.5-hf*0.5,pxy));
  g += 0.028*(hline(uv.y,0.5+hf*0.25,pxy)+hline(uv.y,0.5-hf*0.25,pxy));
  g += 0.05*(hline(uv.y,0.5+hf,pxy)+hline(uv.y,0.5-hf,pxy));
  c += u_grid*g;
  // subtle glass sheen near the top edge
  c += vec3(0.012)*smoothstep(0.86,1.0,uv.y);
  o = vec4(c,1.0);
}`;

// One instanced quad per device column. The quad is grown vertically so the
// fragment shader can compute sub-pixel coverage of the [top,bottom] span; the
// `flat` varyings carry the exact edges, which is what keeps the envelope
// smooth instead of hard-stepped.
const VS_ENV = `#version 300 es
precision highp float;
layout(location=0) in vec4 a_col;   // xLeft, top, bottom, width  (device px)
layout(location=1) in vec4 a_band;  // topLo, topHi, botLo, botHi (device px)
uniform vec2 u_res;
uniform float u_grow;
flat out vec4 v_col;
flat out vec4 v_band;
void main(){
  int id = gl_VertexID;
  float fx = float(id & 1);
  float fy = float(id >> 1);
  float lo = min(a_band.x, a_col.y) - u_grow;
  float hi = max(a_band.w, a_col.z) + u_grow;
  vec2 p = vec2(a_col.x + fx*a_col.w, mix(lo, hi, fy));
  v_col = a_col;
  v_band = a_band;
  vec2 ndc = (p / u_res)*2.0-1.0;
  gl_Position = vec4(ndc.x, -ndc.y, 0., 1.);
}`;

const FS_ENV = `#version 300 es
precision highp float;
flat in vec4 v_col;
flat in vec4 v_band;
uniform vec4 u_frame;  // fbW, fbH, fullW/fbW, fullH/fbH
uniform vec3 u_core;
uniform vec3 u_edge;
uniform vec2 u_alpha;  // alpha at the centre line, alpha at the peaks
uniform float u_gamma;
uniform vec2 u_axis;   // centreY, halfHeight (device px)
uniform float u_outline;
uniform float u_lw;    // outline half width (device px)
out vec4 o;
float cover(float lo, float hi, float y, float hp){
  return clamp((min(hi, y+hp) - max(lo, y-hp)) / (2.0*hp), 0.0, 1.0);
}
void main(){
  float hp = 0.5*u_frame.w;
  float y = (u_frame.y - gl_FragCoord.y) * u_frame.w;
  float fill = cover(v_col.y, v_col.z, y, hp);
  float cov = fill;
  if (u_outline > 0.5) {
    float top = cover(v_band.x - u_lw, v_band.y + u_lw, y, hp);
    float bot = cover(v_band.z - u_lw, v_band.w + u_lw, y, hp);
    float edge = max(top, bot);
    cov = edge + fill*0.18*(1.0-edge);
  }
  if (cov <= 0.0) discard;
  float t = pow(clamp(abs(y - u_axis.x)/u_axis.y, 0.0, 1.0), u_gamma);
  vec3 c = mix(u_core, u_edge, t);
  float a = mix(u_alpha.x, u_alpha.y, t) * cov;
  o = vec4(c*a, a);  // premultiplied
}`;

// Continuous sample line (1:1 zoom): a screen-space ribbon with analytic edge
// coverage, so the trace stays a smooth 1.5 px line at any slope.
const VS_LINE = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_v;   // x, y (device px), side (-1..1)
uniform vec2 u_res;
out float v_side;
void main(){
  v_side = a_v.z;
  vec2 ndc = (a_v.xy / u_res)*2.0-1.0;
  gl_Position = vec4(ndc.x, -ndc.y, 0., 1.);
}`;

const FS_LINE = `#version 300 es
precision highp float;
in float v_side;
uniform vec4 u_frame;
uniform vec3 u_core;
uniform vec3 u_edge;
uniform vec2 u_alpha;
uniform float u_gamma;
uniform vec2 u_axis;
out vec4 o;
void main(){
  float d = abs(v_side);
  float w = max(fwidth(v_side), 1e-5);
  float cov = clamp((1.0 - d)/w + 0.5, 0.0, 1.0);
  if (cov <= 0.0) discard;
  float y = (u_frame.y - gl_FragCoord.y) * u_frame.w;
  float t = pow(clamp(abs(y - u_axis.x)/u_axis.y, 0.0, 1.0), u_gamma);
  vec3 c = mix(u_core, u_edge, t);
  float a = mix(u_alpha.x, u_alpha.y, t) * cov;
  o = vec4(c*a, a);
}`;

// Sample dots, drawn only when consecutive samples are far enough apart to read.
const VS_DOT = `#version 300 es
precision highp float;
layout(location=0) in vec2 a_c;  // centre, device px
uniform vec2 u_res;
uniform float u_r;
flat out vec2 v_c;
void main(){
  int id = gl_VertexID;
  vec2 f = vec2(float(id & 1), float(id >> 1))*2.0-1.0;
  vec2 p = a_c + f*(u_r+1.5);
  v_c = a_c;
  vec2 ndc = (p / u_res)*2.0-1.0;
  gl_Position = vec4(ndc.x, -ndc.y, 0., 1.);
}`;

const FS_DOT = `#version 300 es
precision highp float;
flat in vec2 v_c;
uniform vec4 u_frame;
uniform vec3 u_color;
uniform float u_alpha;
uniform float u_r;
out vec4 o;
void main(){
  vec2 fp = vec2(gl_FragCoord.x*u_frame.z, (u_frame.y - gl_FragCoord.y)*u_frame.w);
  float d = length(fp - v_c);
  float hp = 0.75*u_frame.w;
  float cov = 1.0 - smoothstep(u_r-hp, u_r+hp, d);
  if (cov <= 0.0) discard;
  float a = u_alpha*cov;
  o = vec4(u_color*a, a);
}`;

// Every hairline, band and playhead element: one instanced rect with exact
// two-axis coverage, plus optional gaussian / ramp shaping across x.
const VS_RECT = `#version 300 es
precision highp float;
layout(location=0) in vec4 a_r;  // x0,y0,x1,y1 (device px)
uniform vec2 u_res;
flat out vec4 v_r;
void main(){
  int id = gl_VertexID;
  vec2 f = vec2(float(id & 1), float(id >> 1));
  vec2 p = mix(a_r.xy, a_r.zw, f);
  v_r = a_r;
  vec2 ndc = (p / u_res)*2.0-1.0;
  gl_Position = vec4(ndc.x, -ndc.y, 0., 1.);
}`;

const FS_RECT = `#version 300 es
precision highp float;
flat in vec4 v_r;
uniform vec4 u_frame;
uniform vec4 u_color;  // rgb + alpha, unpremultiplied
uniform vec3 u_mode;   // mode, p0, p1  (1 = gaussian about p0 sigma p1, 2 = ramp p0->p1)
out vec4 o;
void main(){
  vec2 fp = vec2(gl_FragCoord.x*u_frame.z, (u_frame.y - gl_FragCoord.y)*u_frame.w);
  vec2 hp = 0.5*u_frame.zw;
  float cx = clamp((min(v_r.z, fp.x+hp.x) - max(v_r.x, fp.x-hp.x))/(2.0*hp.x), 0.0, 1.0);
  float cy = clamp((min(v_r.w, fp.y+hp.y) - max(v_r.y, fp.y-hp.y))/(2.0*hp.y), 0.0, 1.0);
  float a = u_color.a * cx * cy;
  int m = int(u_mode.x + 0.5);
  if (m == 1) {
    float d = (fp.x - u_mode.y)/max(u_mode.z, 1e-4);
    a *= exp(-d*d);
  } else if (m == 2) {
    a *= clamp((fp.x - u_mode.y)/max(u_mode.z - u_mode.y, 1e-4), 0.0, 1.0);
    a *= a;
  }
  if (a <= 0.0) discard;
  o = vec4(u_color.rgb*a, a);
}`;

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

interface Buf {
  buf: WebGLBuffer;
  count: number;
}

/** How a deck is currently represented: min/max envelope, or a sample trace. */
type DeckMode = 'env' | 'line';

interface KindGeom {
  peaks: Buf | null;
  rms: Buf | null;
  line: Buf | null;
  dots: Buf | null;
  mode: DeckMode;
}

let gl: WebGL2RenderingContext | null = null;
let canvas: OffscreenCanvas | null = null;
let cssW = 0;
let cssH = 0;
let dpr = 1;
let W = 0;
let H = 0;

let progBg: ProgramU;
let progEnv: ProgramU;
let progLine: ProgramU;
let progDot: ProgramU;
let progRect: ProgramU;
let progBlur: ProgramU;
let progComposite: ProgramU;
let vaoEmpty: WebGLVertexArrayObject;
let vaoEnv: WebGLVertexArrayObject;
let vaoLine: WebGLVertexArrayObject;
let vaoDot: WebGLVertexArrayObject;
let vaoRect: WebGLVertexArrayObject;
let rectBuf: WebGLBuffer;
let glowA: Fbo | null = null;
let glowB: Fbo | null = null;

let palette: WavePalette = DEFAULT_WAVE_PALETTE;

const data: Record<WaveKind, KindData> = {
  original: { base: null, detail: null },
  cleaned: { base: null, detail: null },
};
const geom: Record<WaveKind, KindGeom> = {
  original: { peaks: null, rms: null, line: null, dots: null, mode: 'env' },
  cleaned: { peaks: null, rms: null, line: null, dots: null, mode: 'env' },
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
/**
 * Device px per bucket beyond which min/max bars give way to a sample trace.
 * The engine's deepest window is one sample per device column, so the switch
 * has to happen at 1:1 itself — below that a bar and a line are the same pixel.
 */
const LINE_MODE_PX = 0.9;
/** CSS px between samples beyond which individual sample dots are drawn. */
const DOT_MODE_CSS_PX = 6;

function post(msg: WaveOutMsg): void {
  ctxSelf.postMessage(msg);
}

// ---------------------------------------------------------------------------
// C1 · what a frame costs, measured rather than asserted.
//
// The claim the goal makes is "the worker never blocks the main thread >16 ms".
// The worker cannot block the main thread at all — it is a different thread —
// so the honest quantity is how long `render()` holds *this* thread, and how
// far apart rendered frames actually land. Both are kept here, at a cost of one
// `performance.now()` pair per frame, and reported at most twice a second.

const FT_WINDOW = 240;
const STATS_MIN_INTERVAL_MS = 500;
const frameMs = new Float32Array(FT_WINDOW);
const frameGapMs = new Float32Array(FT_WINDOW);
const ftSorted = new Float32Array(FT_WINDOW);
let ftIdx = 0;
let ftCount = 0;
let frameTotal = 0;
let frameMaxAll = 0;
let lastFrameAt = 0;
let lastStatsAt = 0;

function percentile(src: Float32Array, n: number, q: number): number {
  if (n <= 0) return 0;
  ftSorted.set(src.subarray(0, n));
  const view = ftSorted.subarray(0, n);
  view.sort();
  const i = Math.min(n - 1, Math.max(0, Math.round(q * (n - 1))));
  return view[i] ?? 0;
}

function statsSnapshot(): WaveOutMsg {
  const n = ftCount;
  let sum = 0;
  let max = 0;
  for (let i = 0; i < n; i++) {
    const v = frameMs[i] ?? 0;
    sum += v;
    if (v > max) max = v;
  }
  return {
    type: 'stats',
    frames: frameTotal,
    last: Number((frameMs[(ftIdx + FT_WINDOW - 1) % FT_WINDOW] ?? 0).toFixed(3)),
    mean: Number((n > 0 ? sum / n : 0).toFixed(3)),
    p95: Number(percentile(frameMs, n, 0.95).toFixed(3)),
    max: Number(max.toFixed(3)),
    maxAll: Number(frameMaxAll.toFixed(3)),
    window: n,
    interval: Number(percentile(frameGapMs, n, 0.5).toFixed(3)),
  };
}

function recordFrame(startedAt: number, endedAt: number): void {
  const dt = endedAt - startedAt;
  frameMs[ftIdx] = dt;
  frameGapMs[ftIdx] = lastFrameAt > 0 ? startedAt - lastFrameAt : 0;
  lastFrameAt = startedAt;
  ftIdx = (ftIdx + 1) % FT_WINDOW;
  if (ftCount < FT_WINDOW) ftCount += 1;
  frameTotal += 1;
  if (dt > frameMaxAll) frameMaxAll = dt;
  if (endedAt - lastStatsAt >= STATS_MIN_INTERVAL_MS) {
    lastStatsAt = endedAt;
    post(statsSnapshot());
  }
}

function renderTimed(): void {
  const t0 = performance.now();
  render();
  recordFrame(t0, performance.now());
}

function scheduleRender(): void {
  needsRender = true;
  if (frameReq) return;
  const raf = (ctxSelf as unknown as { requestAnimationFrame?: (cb: () => void) => number })
    .requestAnimationFrame;
  if (typeof raf === 'function') {
    frameReq = raf.call(ctxSelf, () => {
      frameReq = 0;
      if (needsRender) renderTimed();
    });
  } else {
    frameReq = setTimeout(() => {
      frameReq = 0;
      if (needsRender) renderTimed();
    }, 16) as unknown as number;
  }
}

// ---------------------------------------------------------------------------
// Init / resize

function init(c: OffscreenCanvas, w: number, h: number, ratio: number, pal?: WavePalette): void {
  canvas = c;
  const ctx = c.getContext('webgl2', {
    antialias: false, // coverage is computed analytically in the shaders
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
  if (pal) palette = pal;
  progBg = createProgramU(gl, VS_QUAD, FS_BG, ['u_res', 'u_margin', 'u_top', 'u_bot', 'u_grid']);
  progEnv = createProgramU(gl, VS_ENV, FS_ENV, [
    'u_res', 'u_grow', 'u_frame', 'u_core', 'u_edge', 'u_alpha', 'u_gamma', 'u_axis',
    'u_outline', 'u_lw',
  ]);
  progLine = createProgramU(gl, VS_LINE, FS_LINE, [
    'u_res', 'u_frame', 'u_core', 'u_edge', 'u_alpha', 'u_gamma', 'u_axis',
  ]);
  progDot = createProgramU(gl, VS_DOT, FS_DOT, ['u_res', 'u_r', 'u_frame', 'u_color', 'u_alpha']);
  progRect = createProgramU(gl, VS_RECT, FS_RECT, ['u_res', 'u_frame', 'u_color', 'u_mode']);
  progBlur = createProgramU(gl, VS_QUAD, FS_BLUR, ['u_tex', 'u_dir']);
  progComposite = createProgramU(gl, VS_QUAD, FS_COMPOSITE, ['u_tex', 'u_gain']);

  const ve = gl.createVertexArray();
  const vv = gl.createVertexArray();
  const vl = gl.createVertexArray();
  const vd = gl.createVertexArray();
  const vr = gl.createVertexArray();
  const rb = gl.createBuffer();
  if (!ve || !vv || !vl || !vd || !vr || !rb) throw new Error('vao alloc failed');
  vaoEmpty = ve;
  vaoEnv = vv;
  vaoLine = vl;
  vaoDot = vd;
  vaoRect = vr;
  rectBuf = rb;

  // Attribute layouts. Buffers are bound per draw (there are only a handful of
  // draws per frame), but the divisors live in the VAO and are set once.
  gl.bindVertexArray(vaoEnv);
  gl.enableVertexAttribArray(0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribDivisor(0, 1);
  gl.vertexAttribDivisor(1, 1);
  gl.bindVertexArray(vaoLine);
  gl.enableVertexAttribArray(0);
  gl.bindVertexArray(vaoDot);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribDivisor(0, 1);
  gl.bindVertexArray(vaoRect);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribDivisor(0, 1);
  gl.bindVertexArray(null);
  gl.bindBuffer(gl.ARRAY_BUFFER, rectBuf);
  gl.bufferData(gl.ARRAY_BUFFER, rectScratch.byteLength, gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

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
// Geometry scratch — grown, never reallocated per frame.

let colScratch = new Float32Array(0);
let topScratch = new Float32Array(0);
let botScratch = new Float32Array(0);
let ptScratch = new Float32Array(0);
let lineScratch = new Float32Array(0);
let dotScratch = new Float32Array(0);
let negScratch = new Float32Array(0);

type Scratch = Float32Array<ArrayBuffer>;

function grow(a: Scratch, n: number): Scratch {
  return a.length >= n ? a : new Float32Array(n);
}

function timeToX(t: number): number {
  const span = viewEnd - viewStart;
  return span > 0 ? ((t - viewStart) / span) * W : 0;
}

/**
 * Resample a min/max envelope into one record per device column.
 *
 * The buckets in `lo`/`hi` cover [dataStart, dataEnd] seconds; the columns
 * cover the visible window. Columns outside the data collapse to the centre
 * line, so a detail window that no longer covers the view reads as empty
 * rather than as garbage. Each record is
 * `[xLeft, top, bottom, width, topLo, topHi, botLo, botHi]`, where the four
 * band values span the vertical connection to the neighbouring columns — that
 * is what lets the outline mode draw an unbroken contour.
 */
function buildColumns(
  lo: Float32Array,
  hi: Float32Array,
  dataStart: number,
  dataEnd: number,
  minPx: number,
): number {
  const n = Math.min(lo.length, hi.length);
  const cols = W;
  colScratch = grow(colScratch, cols * 8);
  topScratch = grow(topScratch, cols);
  botScratch = grow(botScratch, cols);
  const centerY = H / 2;
  const halfH = (H / 2) * (1 - MARGIN_FRAC * 2);
  const span = Math.max(1e-12, viewEnd - viewStart);
  const bucketsPerS = n / Math.max(1e-12, dataEnd - dataStart);
  for (let x = 0; x < cols; x++) {
    const t0 = viewStart + (x / cols) * span;
    const t1 = viewStart + ((x + 1) / cols) * span;
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
    let topY = centerY - vmax * halfH;
    let botY = centerY - vmin * halfH;
    // Guarantee a hairline so silence still reads as a line.
    if (minPx > 0 && botY - topY < minPx) {
      const mid = (topY + botY) / 2;
      topY = mid - minPx / 2;
      botY = mid + minPx / 2;
    }
    topScratch[x] = topY;
    botScratch[x] = botY;
  }
  let o = 0;
  for (let x = 0; x < cols; x++) {
    const top = topScratch[x] ?? 0;
    const bot = botScratch[x] ?? 0;
    const tp = topScratch[x > 0 ? x - 1 : 0] ?? top;
    const tn = topScratch[x + 1 < cols ? x + 1 : cols - 1] ?? top;
    const bp = botScratch[x > 0 ? x - 1 : 0] ?? bot;
    const bn = botScratch[x + 1 < cols ? x + 1 : cols - 1] ?? bot;
    const tm0 = (top + tp) * 0.5;
    const tm1 = (top + tn) * 0.5;
    const bm0 = (bot + bp) * 0.5;
    const bm1 = (bot + bn) * 0.5;
    colScratch[o++] = x;
    colScratch[o++] = top;
    colScratch[o++] = bot;
    colScratch[o++] = 1;
    colScratch[o++] = Math.min(top, tm0, tm1);
    colScratch[o++] = Math.max(top, tm0, tm1);
    colScratch[o++] = Math.min(bot, bm0, bm1);
    colScratch[o++] = Math.max(bot, bm0, bm1);
  }
  return cols;
}

/**
 * Build the 1:1 sample trace: a screen-space ribbon through the visible
 * buckets. Returns the vertex count (0 when there is nothing to draw).
 */
function buildRibbon(
  lo: Float32Array,
  hi: Float32Array,
  dataStart: number,
  dataEnd: number,
  halfW: number,
): { verts: number; points: number } {
  const n = Math.min(lo.length, hi.length);
  const bucketsPerS = n / Math.max(1e-12, dataEnd - dataStart);
  const i0 = Math.max(0, Math.floor((viewStart - dataStart) * bucketsPerS) - 1);
  const i1 = Math.min(n, Math.ceil((viewEnd - dataStart) * bucketsPerS) + 2);
  const pts = i1 - i0;
  if (pts < 2) return { verts: 0, points: 0 };
  ptScratch = grow(ptScratch, pts * 2);
  lineScratch = grow(lineScratch, pts * 6);
  const centerY = H / 2;
  const halfH = (H / 2) * (1 - MARGIN_FRAC * 2);
  for (let k = 0; k < pts; k++) {
    const i = i0 + k;
    const t = dataStart + (i + 0.5) / bucketsPerS;
    const v = ((lo[i] ?? 0) + (hi[i] ?? 0)) * 0.5;
    ptScratch[k * 2] = timeToX(t);
    ptScratch[k * 2 + 1] = centerY - v * halfH;
  }
  let o = 0;
  for (let k = 0; k < pts; k++) {
    const kp = k > 0 ? k - 1 : 0;
    const kn = k + 1 < pts ? k + 1 : pts - 1;
    const x = ptScratch[k * 2] ?? 0;
    const y = ptScratch[k * 2 + 1] ?? 0;
    const dx = (ptScratch[kn * 2] ?? x) - (ptScratch[kp * 2] ?? x);
    const dy = (ptScratch[kn * 2 + 1] ?? y) - (ptScratch[kp * 2 + 1] ?? y);
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len;
    const ny = dx / len;
    // Miter compensation so corners keep their width instead of pinching.
    const fx = (ptScratch[kn * 2] ?? x) - x;
    const fy = (ptScratch[kn * 2 + 1] ?? y) - y;
    const flen = Math.hypot(fx, fy);
    let m = 1;
    if (flen > 1e-6) {
      const c = Math.abs(nx * (-fy / flen) + ny * (fx / flen));
      m = 1 / Math.max(0.4, c);
    }
    const ox = nx * halfW * m;
    const oy = ny * halfW * m;
    lineScratch[o++] = x + ox;
    lineScratch[o++] = y + oy;
    lineScratch[o++] = 1;
    lineScratch[o++] = x - ox;
    lineScratch[o++] = y - oy;
    lineScratch[o++] = -1;
  }
  return { verts: pts * 2, points: pts };
}

/** Sample dots reuse the ribbon's point positions. */
function buildDots(points: number): number {
  dotScratch = grow(dotScratch, points * 2);
  for (let k = 0; k < points; k++) {
    dotScratch[k * 2] = ptScratch[k * 2] ?? 0;
    dotScratch[k * 2 + 1] = ptScratch[k * 2 + 1] ?? 0;
  }
  return points;
}

function upload(src: Float32Array, len: number, prev: Buf | null, stride: number): Buf {
  if (!gl) throw new Error('no gl');
  const buf = prev?.buf ?? gl.createBuffer();
  if (!buf) throw new Error('buffer alloc failed');
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, src.subarray(0, len), gl.DYNAMIC_DRAW);
  return { buf, count: len / stride };
}

function dropBuf(b: Buf | null): null {
  if (b && gl) gl.deleteBuffer(b.buf);
  return null;
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

/**
 * One sample per bucket is what `/api/peaks` returns at maximum zoom, and it
 * shows up as min === max. That is the moment a real editor stops drawing bars
 * and starts drawing the waveform itself.
 */
function isRawSamples(d: WaveData): boolean {
  const n = Math.min(d.min.length, d.max.length);
  if (n < 2) return false;
  const bucketsPerS = n / Math.max(1e-12, d.end - d.start);
  const i0 = Math.max(0, Math.floor((viewStart - d.start) * bucketsPerS));
  const i1 = Math.min(n, Math.ceil((viewEnd - d.start) * bucketsPerS));
  if (i1 - i0 < 2) return false;
  const step = Math.max(1, Math.floor((i1 - i0) / 48));
  for (let i = i0; i < i1; i += step) {
    if (Math.abs((d.min[i] ?? 0) - (d.max[i] ?? 0)) > 1e-6) return false;
  }
  return true;
}

function rebuildGeometry(): void {
  if (!gl) return;
  const span = viewEnd - viewStart;
  for (const kind of ['original', 'cleaned'] as const) {
    const d = pickSource(kind);
    const g = geom[kind];
    if (!d || span <= 0 || W <= 0) {
      g.peaks = dropBuf(g.peaks);
      g.rms = dropBuf(g.rms);
      g.line = dropBuf(g.line);
      g.dots = dropBuf(g.dots);
      g.mode = 'env';
      continue;
    }
    const n = Math.min(d.min.length, d.max.length);
    const bucketsPerS = n / Math.max(1e-12, d.end - d.start);
    const pxPerBucket = bucketsPerS > 0 ? W / (span * bucketsPerS) : 0;
    const line = pxPerBucket >= LINE_MODE_PX && isRawSamples(d);
    if (line) {
      const r = buildRibbon(d.min, d.max, d.start, d.end, 1.0 * dpr);
      if (r.verts > 0) {
        g.line = upload(lineScratch, r.verts * 3, g.line, 3);
        if (pxPerBucket >= DOT_MODE_CSS_PX * dpr) {
          const dots = buildDots(r.points);
          g.dots = upload(dotScratch, dots * 2, g.dots, 2);
        } else {
          g.dots = dropBuf(g.dots);
        }
        g.peaks = dropBuf(g.peaks);
        g.rms = dropBuf(g.rms);
        g.mode = 'line';
        continue;
      }
    }
    const cols = buildColumns(d.min, d.max, d.start, d.end, Math.max(1, dpr));
    g.peaks = upload(colScratch, cols * 8, g.peaks, 8);
    if (d.rms) {
      // RMS is a symmetric body about the centre line: mirror it into a
      // min/max pair so it goes through the same resampler.
      const nr = d.rms.length;
      const neg = grow(negScratch, nr);
      negScratch = neg;
      for (let i = 0; i < nr; i++) neg[i] = -(d.rms[i] ?? 0);
      // No hairline floor: a silent RMS body should vanish, leaving only the
      // peak envelope's own hairline on the centre line.
      const rcols = buildColumns(neg, d.rms, d.start, d.end, 0);
      g.rms = upload(colScratch, rcols * 8, g.rms, 8);
    } else {
      g.rms = dropBuf(g.rms);
    }
    g.line = dropBuf(g.line);
    g.dots = dropBuf(g.dots);
    g.mode = 'env';
  }
  geomDirty = false;
}

// ---------------------------------------------------------------------------
// Draw helpers

/** Framebuffer geometry for the fragment shaders: fbW, fbH, and full/fb scale. */
let frameW = 0;
let frameH = 0;

function setFrame(p: ProgramU): void {
  if (!gl) return;
  gl.uniform4f(p.u.u_frame ?? null, frameW, frameH, W / frameW, H / frameH);
}

function drawEnv(
  b: Buf,
  colors: WaveDeckColors,
  aCore: number,
  aEdge: number,
  gamma: number,
  outline: boolean,
  rms: boolean,
): void {
  if (!gl) return;
  const p = progEnv;
  gl.useProgram(p.p);
  gl.uniform2f(p.u.u_res ?? null, W, H);
  gl.uniform1f(p.u.u_grow ?? null, outline ? 1.2 * dpr : 1);
  setFrame(p);
  const core = rms ? colors.rms : colors.core;
  const edge = rms ? colors.core : colors.edge;
  gl.uniform3f(p.u.u_core ?? null, core[0], core[1], core[2]);
  gl.uniform3f(p.u.u_edge ?? null, edge[0], edge[1], edge[2]);
  gl.uniform2f(p.u.u_alpha ?? null, aCore, aEdge);
  gl.uniform1f(p.u.u_gamma ?? null, gamma);
  gl.uniform2f(p.u.u_axis ?? null, H / 2, (H / 2) * (1 - MARGIN_FRAC * 2));
  gl.uniform1f(p.u.u_outline ?? null, outline ? 1 : 0);
  gl.uniform1f(p.u.u_lw ?? null, 0.55 * dpr);
  gl.bindVertexArray(vaoEnv);
  gl.bindBuffer(gl.ARRAY_BUFFER, b.buf);
  gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 32, 0);
  gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 32, 16);
  gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, b.count);
  gl.bindVertexArray(null);
}

function drawLine(b: Buf, colors: WaveDeckColors, aCore: number, aEdge: number): void {
  if (!gl) return;
  const p = progLine;
  gl.useProgram(p.p);
  gl.uniform2f(p.u.u_res ?? null, W, H);
  setFrame(p);
  gl.uniform3f(p.u.u_core ?? null, colors.rms[0], colors.rms[1], colors.rms[2]);
  gl.uniform3f(p.u.u_edge ?? null, colors.edge[0], colors.edge[1], colors.edge[2]);
  gl.uniform2f(p.u.u_alpha ?? null, aCore, aEdge);
  gl.uniform1f(p.u.u_gamma ?? null, 0.65);
  gl.uniform2f(p.u.u_axis ?? null, H / 2, (H / 2) * (1 - MARGIN_FRAC * 2));
  gl.bindVertexArray(vaoLine);
  gl.bindBuffer(gl.ARRAY_BUFFER, b.buf);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, b.count);
  gl.bindVertexArray(null);
}

function drawDots(b: Buf, color: WaveRgb, alpha: number, radius: number): void {
  if (!gl) return;
  const p = progDot;
  gl.useProgram(p.p);
  gl.uniform2f(p.u.u_res ?? null, W, H);
  gl.uniform1f(p.u.u_r ?? null, radius);
  setFrame(p);
  gl.uniform3f(p.u.u_color ?? null, color[0], color[1], color[2]);
  gl.uniform1f(p.u.u_alpha ?? null, alpha);
  gl.bindVertexArray(vaoDot);
  gl.bindBuffer(gl.ARRAY_BUFFER, b.buf);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
  gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, b.count);
  gl.bindVertexArray(null);
}

// --- instanced AA rects (hairlines, bands, playhead) -----------------------

const rectScratch = new Float32Array(4 * 1024);
let rectLen = 0;
let rectR = 1;
let rectG = 1;
let rectB = 1;
let rectA = 1;
let rectMode = 0;
let rectP0 = 0;
let rectP1 = 0;

function rectBegin(color: WaveRgb, alpha: number, mode = 0, p0 = 0, p1 = 0): void {
  rectFlush();
  rectR = color[0];
  rectG = color[1];
  rectB = color[2];
  rectA = alpha;
  rectMode = mode;
  rectP0 = p0;
  rectP1 = p1;
}

function rect(x0: number, y0: number, x1: number, y1: number): void {
  if (rectLen + 4 > rectScratch.length) rectFlush();
  rectScratch[rectLen++] = x0;
  rectScratch[rectLen++] = y0;
  rectScratch[rectLen++] = x1;
  rectScratch[rectLen++] = y1;
}

/** A crisp hairline: snapped to whole device pixels around `center`. */
function vline(center: number, width: number, y0: number, y1: number): void {
  const w = Math.max(1, Math.round(width));
  const x0 = Math.round(center - w / 2);
  rect(x0, y0, x0 + w, y1);
}

function hline(center: number, width: number, x0: number, x1: number): void {
  const w = Math.max(1, Math.round(width));
  const y0 = Math.round(center - w / 2);
  rect(x0, y0, x1, y0 + w);
}

function rectFlush(): void {
  if (!gl || rectLen === 0) return;
  const p = progRect;
  gl.useProgram(p.p);
  gl.uniform2f(p.u.u_res ?? null, W, H);
  setFrame(p);
  gl.uniform4f(p.u.u_color ?? null, rectR, rectG, rectB, rectA);
  gl.uniform3f(p.u.u_mode ?? null, rectMode, rectP0, rectP1);
  gl.bindVertexArray(vaoRect);
  gl.bindBuffer(gl.ARRAY_BUFFER, rectBuf);
  // Fixed-size store written in place: no subarray view, no allocation, on a
  // path that runs several times per frame.
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, rectScratch, 0, rectLen);
  gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 16, 0);
  gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, rectLen / 4);
  gl.bindVertexArray(null);
  rectLen = 0;
}

// ---------------------------------------------------------------------------
// Render

/**
 * The tick ladder only changes with the view or the width, but render() runs
 * on every playhead update during playback — so it is memoised rather than
 * rebuilt (and re-allocated) sixty times a second.
 */
let tickCache: ReturnType<typeof timeTicksIn> = [];
let tickKeyStart = NaN;
let tickKeyEnd = NaN;
let tickKeyW = -1;

function cachedTicks(): ReturnType<typeof timeTicksIn> {
  if (tickKeyStart !== viewStart || tickKeyEnd !== viewEnd || tickKeyW !== cssW) {
    tickCache = timeTicksIn(viewStart, viewEnd, cssW, 72);
    tickKeyStart = viewStart;
    tickKeyEnd = viewEnd;
    tickKeyW = cssW;
  }
  return tickCache;
}

interface DeckLook {
  aCore: number;
  aEdge: number;
  outline: boolean;
}

function deckLook(focused: boolean, both: boolean): DeckLook {
  if (!both) return { aCore: 0.72, aEdge: 0.46, outline: false };
  if (focused) return { aCore: 0.7, aEdge: 0.42, outline: false };
  // The unfocused deck keeps its contour at nearly full strength so it stays
  // readable through the focused deck's body — an outline, not a ghost.
  return { aCore: 0.72, aEdge: 0.62, outline: true };
}

function drawDeck(kind: WaveKind, focused: boolean, both: boolean, glowPass: boolean): void {
  const g = geom[kind];
  const colors = kind === 'original' ? palette.original : palette.cleaned;
  const look = deckLook(focused, both);
  if (g.mode === 'line' && g.line) {
    const a = glowPass ? 1.0 : focused || !both ? 0.95 : 0.6;
    drawLine(g.line, colors, a, a * 0.85);
    if (g.dots && !glowPass) drawDots(g.dots, colors.rms, focused || !both ? 0.9 : 0.5, 1.6 * dpr);
    return;
  }
  if (!g.peaks) return;
  if (glowPass) {
    // Only the body blooms; the outline deck contributes nothing so the halo
    // never doubles up around the focused trace.
    if (look.outline) return;
    drawEnv(g.peaks, colors, 0.9, 0.5, 0.8, false, false);
    return;
  }
  drawEnv(g.peaks, colors, look.aCore, look.aEdge, 0.72, look.outline, false);
  if (!look.outline && g.rms) {
    drawEnv(g.rms, colors, 0.82, 0.62, 1.0, false, true);
  }
}

function render(): void {
  needsRender = false;
  if (!gl || !glowA || !glowB) return;
  if (geomDirty) rebuildGeometry();

  const hasOriginal = geom.original.peaks !== null || geom.original.line !== null;
  const hasCleaned = geom.cleaned.peaks !== null || geom.cleaned.line !== null;
  const both = hasOriginal && hasCleaned;
  const grid = palette.grid;

  // 1. background
  frameW = W;
  frameH = H;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, W, H);
  gl.disable(gl.BLEND);
  gl.useProgram(progBg.p);
  gl.uniform2f(progBg.u.u_res ?? null, W, H);
  gl.uniform1f(progBg.u.u_margin ?? null, MARGIN_FRAC);
  gl.uniform3f(progBg.u.u_top ?? null, palette.bgTop[0], palette.bgTop[1], palette.bgTop[2]);
  gl.uniform3f(
    progBg.u.u_bot ?? null,
    palette.bgBottom[0],
    palette.bgBottom[1],
    palette.bgBottom[2],
  );
  gl.uniform3f(progBg.u.u_grid ?? null, grid[0], grid[1], grid[2]);
  gl.bindVertexArray(vaoEmpty);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.bindVertexArray(null);

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  // 2. time grid — two weights, both sub-pixel accurate
  const ticks = cachedTicks();
  rectBegin(grid, 0.03);
  for (let i = 0; i < ticks.length; i++) {
    const tk = ticks[i];
    if (tk && !tk.major) vline(timeToX(tk.time), dpr, 0, H);
  }
  rectBegin(grid, 0.07);
  for (let i = 0; i < ticks.length; i++) {
    const tk = ticks[i];
    if (tk && tk.major) vline(timeToX(tk.time), dpr, 0, H);
  }

  // 3. unit boundaries: a hairline through the body plus brighter end caps, so
  // they read as structure rather than as more gridlines.
  if (unitBounds.length > 0) {
    const cap = Math.round(5 * dpr);
    rectBegin(palette.unit, 0.11);
    for (let i = 0; i < unitBounds.length; i++) {
      const t = unitBounds[i] ?? 0;
      if (t < viewStart || t > viewEnd) continue;
      vline(timeToX(t), dpr, H * 0.06, H * 0.94);
    }
    rectBegin(palette.unit, 0.4);
    for (let i = 0; i < unitBounds.length; i++) {
      const t = unitBounds[i] ?? 0;
      if (t < viewStart || t > viewEnd) continue;
      const x = timeToX(t);
      vline(x, dpr, H * 0.06, H * 0.06 + cap);
      vline(x, dpr, H * 0.94 - cap, H * 0.94);
    }
  }

  // 4. selection band
  if (highlight && highlight.end > viewStart && highlight.start < viewEnd) {
    const hx0 = Math.max(0, timeToX(highlight.start));
    const hx1 = Math.min(W, timeToX(highlight.end));
    rectBegin(palette.highlight, 0.055);
    rect(hx0, 0, hx1, H);
    rectBegin(palette.highlight, 0.32);
    if (highlight.start >= viewStart) vline(hx0, dpr, 0, H);
    if (highlight.end <= viewEnd) vline(hx1, dpr, 0, H);
  }
  rectFlush();

  // 5. glow pass (half-res): focused body → blur → additive composite
  if (hasOriginal || hasCleaned) {
    const gw = glowA.width;
    const gh = glowA.height;
    frameW = gw;
    frameH = gh;
    gl.bindFramebuffer(gl.FRAMEBUFFER, glowA.fb);
    gl.viewport(0, 0, gw, gh);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    // Geometry is in full-res device px; the vertex shaders normalise it with
    // u_res = [W, H], so the same buffers draw into the half-res target
    // through the smaller viewport.
    if (hasOriginal && (!both || focus === 'original')) drawDeck('original', true, both, true);
    if (hasCleaned && (!both || focus === 'cleaned')) drawDeck('cleaned', true, both, true);
    // blur H: A → B
    gl.disable(gl.BLEND);
    gl.useProgram(progBlur.p);
    gl.bindVertexArray(vaoEmpty);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(progBlur.u.u_tex ?? null, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, glowB.fb);
    gl.bindTexture(gl.TEXTURE_2D, glowA.tex);
    gl.uniform2f(progBlur.u.u_dir ?? null, 1.5 / gw, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    // blur V: B → A
    gl.bindFramebuffer(gl.FRAMEBUFFER, glowA.fb);
    gl.bindTexture(gl.TEXTURE_2D, glowB.tex);
    gl.uniform2f(progBlur.u.u_dir ?? null, 0, 1.5 / gh);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    // composite additively onto the screen
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    frameW = W;
    frameH = H;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(progComposite.p);
    gl.bindTexture(gl.TEXTURE_2D, glowA.tex);
    gl.uniform1i(progComposite.u.u_tex ?? null, 0);
    gl.uniform1f(progComposite.u.u_gain ?? null, 0.5);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindVertexArray(null);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }

  // 6. decks — focused body first, unfocused contour drawn over it. An A/B
  // comparison is only useful if the reference deck stays readable, so it wins
  // the z-order as a thin contour rather than being buried under the fill.
  if (both) {
    const back: WaveKind = focus === 'cleaned' ? 'original' : 'cleaned';
    drawDeck(focus, true, true, false);
    drawDeck(back, false, true, false);
  } else if (hasOriginal) {
    drawDeck('original', true, false, false);
  } else if (hasCleaned) {
    drawDeck('cleaned', true, false, false);
  }

  // 7. centre line on top, so silence still reads as a hairline
  rectBegin(grid, 0.09);
  hline(H / 2, dpr, 0, W);

  // 8. hover cursor
  if (hoverX !== null) {
    rectBegin(grid, 0.2);
    vline(hoverX * dpr, dpr, 0, H);
  }
  rectFlush();

  // 9. playhead: leading gradient, bloom halo, crisp core, head marker
  if (playheadVisible && viewEnd > viewStart && playhead >= viewStart && playhead <= viewEnd) {
    const x = timeToX(playhead);
    const ph = palette.playhead;
    const lead = 56 * dpr;
    rectBegin(ph, 0.09, 2, x - lead, x);
    rect(x - lead, 0, x, H);
    rectBegin(ph, 0.22, 1, x, 4.5 * dpr);
    rect(x - 14 * dpr, 0, x + 14 * dpr, H);
    rectFlush();
    rectBegin(ph, 0.95);
    vline(x, dpr, 0, H);
    const hw = 3.5 * dpr;
    rect(x - hw, 0, x + hw, 2.5 * dpr);
    rect(x - hw, H - 2.5 * dpr, x + hw, H);
    rectFlush();
  }
  rectFlush();
}

// ---------------------------------------------------------------------------
// Message loop

ctxSelf.onmessage = (ev: MessageEvent<WaveMsg>) => {
  const msg = ev.data;
  try {
    switch (msg.type) {
      case 'init':
        init(msg.canvas, msg.width, msg.height, msg.dpr, msg.palette);
        break;
      case 'resize':
        resize(msg.width, msg.height, msg.dpr);
        break;
      case 'theme':
        palette = msg.palette;
        scheduleRender();
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
      case 'stats':
        lastStatsAt = performance.now();
        post(statsSnapshot());
        break;
    }
  } catch (e) {
    post({ type: 'error', message: e instanceof Error ? e.message : String(e) });
  }
};
