// Tiny WebGL2 helpers shared by the waveform worker.

export function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type);
  if (!sh) throw new Error('createShader failed');
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh) ?? '';
    gl.deleteShader(sh);
    throw new Error(`shader compile failed: ${log}\n${src}`);
  }
  return sh;
}

export function createProgram(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const prog = gl.createProgram();
  if (!prog) throw new Error('createProgram failed');
  const v = compileShader(gl, gl.VERTEX_SHADER, vs);
  const f = compileShader(gl, gl.FRAGMENT_SHADER, fs);
  gl.attachShader(prog, v);
  gl.attachShader(prog, f);
  gl.linkProgram(prog);
  gl.deleteShader(v);
  gl.deleteShader(f);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog) ?? '';
    gl.deleteProgram(prog);
    throw new Error(`program link failed: ${log}`);
  }
  return prog;
}

/**
 * A program plus its uniform locations, resolved once at link time.
 * `getUniformLocation` is a synchronous driver round trip; doing it per frame
 * for a dozen uniforms is measurable, so every draw path here uses this.
 */
export interface ProgramU {
  p: WebGLProgram;
  u: Record<string, WebGLUniformLocation | null>;
}

export function createProgramU(
  gl: WebGL2RenderingContext,
  vs: string,
  fs: string,
  names: readonly string[],
): ProgramU {
  const p = createProgram(gl, vs, fs);
  const u: Record<string, WebGLUniformLocation | null> = {};
  for (const n of names) u[n] = gl.getUniformLocation(p, n);
  return { p, u };
}

export interface Fbo {
  fb: WebGLFramebuffer;
  tex: WebGLTexture;
  width: number;
  height: number;
}

export function createFbo(gl: WebGL2RenderingContext, width: number, height: number): Fbo {
  const tex = gl.createTexture();
  const fb = gl.createFramebuffer();
  if (!tex || !fb) throw new Error('fbo alloc failed');
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return { fb, tex, width, height };
}

export function deleteFbo(gl: WebGL2RenderingContext, fbo: Fbo): void {
  gl.deleteFramebuffer(fbo.fb);
  gl.deleteTexture(fbo.tex);
}

export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

const SRGB_RE = /^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/i;

/**
 * Parse whatever `getComputedStyle` hands back for a colour — the browser
 * resolves custom properties to `rgb()`/`rgba()`/`color(srgb …)`, but a raw
 * token read can still be a hex literal. Falls back to `fallback` on anything
 * unrecognised so a missing design token can never blank the display.
 */
export function parseCssRgb(
  value: string | null | undefined,
  fallback: [number, number, number],
): [number, number, number] {
  if (!value) return fallback;
  const s = value.trim();
  if (s.startsWith('#')) {
    const h = s.slice(1);
    if (h.length === 3 || h.length === 6 || h.length === 8) {
      const rgb = hexToRgb(h.length === 8 ? h.slice(0, 6) : h);
      if (rgb.every((c) => Number.isFinite(c))) return rgb;
    }
    return fallback;
  }
  const m = SRGB_RE.exec(s);
  if (m) {
    const r = Number(m[1]);
    const g = Number(m[2]);
    const b = Number(m[3]);
    if (Number.isFinite(r) && Number.isFinite(g) && Number.isFinite(b)) {
      return [r / 255, g / 255, b / 255];
    }
  }
  // color(srgb 0.5 0.2 0.1 …) — the modern computed form on wide-gamut screens.
  if (s.startsWith('color(')) {
    const nums = s
      .slice(6, -1)
      .replace(/^[a-z-]+\s*/i, '')
      .split(/[\s/]+/)
      .map(Number)
      .filter((n) => Number.isFinite(n));
    if (nums.length >= 3) return [nums[0] ?? 0, nums[1] ?? 0, nums[2] ?? 0];
  }
  return fallback;
}

/** Mix two colours in linear-ish sRGB space; used for derived tints. */
export function mixRgb(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
  t: number,
): [number, number, number] {
  return [
    (a[0] ?? 0) + ((b[0] ?? 0) - (a[0] ?? 0)) * t,
    (a[1] ?? 0) + ((b[1] ?? 0) - (a[1] ?? 0)) * t,
    (a[2] ?? 0) + ((b[2] ?? 0) - (a[2] ?? 0)) * t,
  ];
}
