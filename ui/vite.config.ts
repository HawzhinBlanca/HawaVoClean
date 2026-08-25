import { readFileSync } from 'node:fs'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The version the wordmark shows. It is read from package.json rather than
// typed into the header, because a string typed into the header drifts: the
// wordmark said v3.2 from the commit that created it ("v3.3.0-dev: first
// screen") until this was wired up, and three adversarial audits walked past
// it. `scripts/sync_release_identity.py` already gates ui/package.json against
// src/hawavoclean/release.json, so deriving from it puts the pixel on the same
// chain of trust as every other version mirror in the product.
const UI_VERSION: string = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf8')).version

// The bundle is loaded from file:// inside Resolve's Electron and from
// http://127.0.0.1:{port}/ when served by `hawavoclean serve --ui-dir`.
// Relative base + a single entry keeps both working without rewrites.
export default defineConfig({
  base: './',
  plugins: [react()],
  define: {
    __UI_VERSION__: JSON.stringify(UI_VERSION),
  },
  // Every asset this app ships goes through the bundler. There is no public/
  // directory and there must not be one: anything dropped there is copied into
  // dist/ verbatim, unhashed and unreferenced, and `resolve-plugin/install.sh`
  // then packages the whole directory. A 580 kB debug script reached dist/ that
  // way while this audit was being written.
  publicDir: false,
  build: {
    outDir: 'dist',
    target: 'chrome136',
    sourcemap: false,
    assetsInlineLimit: 0,
  },
  worker: {
    // WaveformHost constructs the production worker as a *classic* script
    // (render/WaveformHost.ts:112) so it also loads from file:// inside
    // Resolve's Electron. Emitting 'es' happened to work only because the
    // worker bundles to a single chunk with no import/export surviving — an
    // invariant that was checked by hand in a doc. 'iife' makes it hold by
    // construction; measured at 22,634 B against 22,656 B for 'es'.
    format: 'iife',
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    // bridge/index.ts resolves the engine origin from a hardcoded list of dev
    // ports. Letting Vite wander to 5174 when 5173 is busy drops the UI off
    // that list, so it silently aims its API client at itself and reports the
    // engine as unreachable. Fail loudly on a busy port instead.
    strictPort: true,
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
  },
})
