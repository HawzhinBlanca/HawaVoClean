// render/spectrum.ts — the FALLBACK palette, pinned to styles/tokens.css.
//
// A canvas has no cascade. When `getComputedStyle` cannot resolve a custom
// property — before the stylesheet has applied, or in any context where the
// element is not in a styled tree — spectrum.ts paints with a hardcoded copy
// of the design tokens instead. Duplicated values drift, and these did: --fg-3
// and --fg-4 sat at #6f7886 / #454c58 in spectrum.ts long after tokens.css was
// re-cut to #949daa / #86909f, and they were older even than the #7c8593 /
// #545c69 the D2 contrast pass had already rejected as illegible. Nothing
// noticed, because nothing compared the two files.
//
// This lives in dev/ rather than beside spectrum.ts for a mechanical reason:
// it has to read tokens.css as text, and vitest stubs CSS modules to the empty
// string (`css: false` by default), so `?raw`, `?inline` and import.meta.glob
// all return '' from inside src/. dev/ is the one place in this project with
// node's fs, and since tsconfig.node.json includes it, this file is
// typechecked like everything else.
//
// Reading both files as source text is also the stronger assertion: it checks
// what a reader of the repository sees, adds no export to production code just
// to be testable, and cannot be satisfied by a value that only exists at
// runtime.
//
// Scoped to hex-valued entries on purpose. `--font-ui` is `var(--font-sans)`
// in tokens.css — an indirection, not a value — so a naive "every key must
// match" assertion would fail on it while telling us nothing.

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const read = (rel: string): string =>
  readFileSync(new URL(rel, import.meta.url), 'utf8');

const spectrumSrc = read('../src/render/spectrum.ts');
const tokensCss = read('../src/styles/tokens.css');

/** The `'--name': '#rrggbb',` entries of spectrum.ts's FALLBACK table. */
function fallbackHexes(src: string): Map<string, string> {
  const start = src.indexOf('const FALLBACK');
  expect(start, 'spectrum.ts must still declare a FALLBACK table').toBeGreaterThan(-1);
  const body = src.slice(start, src.indexOf('};', start));
  const out = new Map<string, string>();
  for (const m of body.matchAll(/'(--[a-z0-9-]+)':\s*'(#[0-9a-fA-F]{3,8})'/g)) {
    out.set(m[1] as string, (m[2] as string).toLowerCase());
  }
  return out;
}

/** `--name: #rrggbb;` declarations from tokens.css. */
function tokenHexes(css: string): Map<string, string> {
  const out = new Map<string, string>();
  for (const m of css.matchAll(/(--[a-z0-9-]+):\s*(#[0-9a-fA-F]{3,8})\s*;/g)) {
    out.set(m[1] as string, (m[2] as string).toLowerCase());
  }
  return out;
}

describe('spectrum FALLBACK palette', () => {
  const fallback = fallbackHexes(spectrumSrc);
  const tokens = tokenHexes(tokensCss);

  it('finds both tables to compare', () => {
    // Guards the regexes above: if either file is restructured so nothing
    // matches, this fails loudly rather than passing on an empty comparison —
    // which is exactly how an earlier draft of this test passed vacuously.
    expect(fallback.size).toBeGreaterThan(4);
    expect(tokens.size).toBeGreaterThan(10);
  });

  it('every hex fallback equals the token it copies', () => {
    const drifted: string[] = [];
    for (const [name, hex] of fallback) {
      const token = tokens.get(name);
      if (token === undefined) continue; // covered by the next case
      if (token !== hex) drifted.push(`${name}: spectrum.ts ${hex} vs tokens.css ${token}`);
    }
    expect(drifted).toEqual([]);
  });

  it('every hex fallback names a token that still exists', () => {
    // A fallback for a deleted token is a value nothing can ever override.
    expect([...fallback.keys()].filter((n) => !tokens.has(n))).toEqual([]);
  });

  it('keeps the two text greys at the values the contrast pass settled on', () => {
    // Named explicitly because these are the two that drifted, and because
    // legibility depends on them: tokens.css records --fg-3 at 6.4:1 and
    // --fg-4 at 5.6:1 on --panel, both above the 4.5:1 floor.
    expect(fallback.get('--fg-3')).toBe('#949daa');
    expect(fallback.get('--fg-4')).toBe('#86909f');
  });
});
