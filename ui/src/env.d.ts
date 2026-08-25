/**
 * Build-time constants injected by `define` in vite.config.ts (and mirrored in
 * vitest.config.ts so the tests see the same value the bundle will).
 */

/** The UI's own version, read from ui/package.json at build time. */
declare const __UI_VERSION__: string;
