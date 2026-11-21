// render/spectrum.test.ts — Regression test for shallow-height y-axis dB label stride.

import { afterEach, describe, expect, it, vi } from 'vitest';
import { SpectrumRenderer } from './spectrum';

function makeMock2dCtx(drawnTexts: string[]): CanvasRenderingContext2D {
  const noop = () => undefined;
  const gradient = { addColorStop: noop };
  return {
    filter: 'none',
    font: '',
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 1,
    lineJoin: 'miter',
    lineCap: 'butt',
    globalAlpha: 1,
    globalCompositeOperation: 'source-over',
    textAlign: 'start',
    textBaseline: 'alphabetic',
    imageSmoothingEnabled: true,
    imageSmoothingQuality: 'high',
    setTransform: noop,
    setLineDash: noop,
    clearRect: noop,
    fillRect: noop,
    strokeRect: noop,
    beginPath: noop,
    closePath: noop,
    moveTo: noop,
    lineTo: noop,
    bezierCurveTo: noop,
    rect: noop,
    clip: noop,
    fill: noop,
    stroke: noop,
    save: noop,
    restore: noop,
    drawImage: noop,
    createLinearGradient: () => gradient,
    createRadialGradient: () => gradient,
    createPattern: () => null,
    measureText: (text: string) => ({ width: text.length * 6 }),
    fillText: (text: string) => {
      drawnTexts.push(String(text));
    },
  } as unknown as CanvasRenderingContext2D;
}

describe('SpectrumRenderer y-axis dB label spacing', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('uses a 48 dB label step (0, -48, -96) when plot height is under 80px', () => {
    const drawnTexts: string[] = [];
    const origGetContext = HTMLCanvasElement.prototype.getContext;
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (
      this: HTMLCanvasElement,
      type: string,
    ) {
      if (type === '2d') return makeMock2dCtx(drawnTexts);
      return origGetContext.call(this, type as '2d');
    } as typeof HTMLCanvasElement.prototype.getContext);

    const canvas = document.createElement('canvas');
    // At 60px canvas height, plot height is 60 - 11 - 17 = 32px (< 80px)
    Object.defineProperty(canvas, 'offsetWidth', { configurable: true, value: 326 });
    Object.defineProperty(canvas, 'offsetHeight', { configurable: true, value: 60 });
    document.body.appendChild(canvas);

    const renderer = new SpectrumRenderer(canvas);
    try {
      drawnTexts.length = 0;
      renderer.frame(1 / 60);
      const dbLabels = drawnTexts.filter((t) => t === '0' || /^-\d+$/.test(t));
      expect(dbLabels).toEqual(['0', '-48', '-96']);
    } finally {
      renderer.dispose();
      canvas.remove();
    }
  });

  it('uses a 24 dB label step when plot height is between 80px and 135px', () => {
    const drawnTexts: string[] = [];
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(() =>
      makeMock2dCtx(drawnTexts),
    );

    const canvas = document.createElement('canvas');
    // At 128px canvas height, plot height is 128 - 11 - 17 = 100px (>= 80px, spacing = 12 < 17)
    Object.defineProperty(canvas, 'offsetWidth', { configurable: true, value: 326 });
    Object.defineProperty(canvas, 'offsetHeight', { configurable: true, value: 128 });
    document.body.appendChild(canvas);

    const renderer = new SpectrumRenderer(canvas);
    try {
      drawnTexts.length = 0;
      renderer.frame(1 / 60);
      const dbLabels = drawnTexts.filter((t) => t === '0' || /^-\d+$/.test(t));
      expect(dbLabels).toEqual(['0', '-24', '-48', '-72', '-96']);
    } finally {
      renderer.dispose();
      canvas.remove();
    }
  });
});
