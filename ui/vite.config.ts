import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The bundle is loaded from file:// inside Resolve's Electron and from
// http://127.0.0.1:{port}/ when served by `hawavoclean serve --ui-dir`.
// Relative base + a single entry keeps both working without rewrites.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    outDir: 'dist',
    target: 'chrome136',
    sourcemap: false,
    assetsInlineLimit: 0,
  },
  worker: {
    format: 'es',
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: false,
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
  },
})
