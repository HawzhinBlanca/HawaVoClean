import { readFileSync } from 'node:fs';
import { defineConfig } from 'vitest/config';

// Unit tests for the UI logic that is not a pixel: the store's state machine,
// the SSE client, the API client, the keyboard map, and the pure render maths
// (tick ladder, view window, peaks cache). Everything that needs a canvas, a
// WebGL context, a real audio element or a real engine stays in the scripted
// browser verification instead — see docs/web-perfection-log.md.
//
// dev/ holds the one exception to "tests live under src/": the mock engine's
// contract test runs in a node environment (it spawns the mock as a child
// process), and the app tsconfig deliberately keeps node's ambient types out
// of src — so the test lives beside the mock it verifies.

// Mirrors the `define` in vite.config.ts so a test can assert the version the
// wordmark will actually paint. See that file for why the version is derived.
const UI_VERSION: string = JSON.parse(
  readFileSync(new URL('./package.json', import.meta.url), 'utf8'),
).version;

export default defineConfig({
  define: {
    __UI_VERSION__: JSON.stringify(UI_VERSION),
  },
  test: {
    environment: 'happy-dom',
    // Both extensions: the glob was `*.test.ts` alone, so a contributor who
    // reached for JSX in a component test would have written a file that ran
    // zero tests and reported nothing — a silent gap, not a failure.
    include: ['src/**/*.test.{ts,tsx}', 'dev/**/*.test.ts'],
    clearMocks: true,
    restoreMocks: true,
    unstubEnvs: true,
    unstubGlobals: true,
  },
});
