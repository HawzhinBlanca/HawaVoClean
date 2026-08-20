import { defineConfig } from 'vitest/config';

// Unit tests for the UI logic that is not a pixel: the store's state machine,
// the SSE client, the API client, the keyboard map, and the pure render maths
// (tick ladder, view window, peaks cache). Everything that needs a canvas, a
// WebGL context, a real audio element or a real engine stays in the scripted
// browser verification instead — see docs/web-perfection-log.md.
export default defineConfig({
  test: {
    environment: 'happy-dom',
    include: ['src/**/*.test.ts'],
    clearMocks: true,
    restoreMocks: true,
    unstubEnvs: true,
    unstubGlobals: true,
  },
});
