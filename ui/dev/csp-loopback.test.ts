// src/bridge/index.ts and ui/index.html hold two lists that have to agree:
// the hosts a `?engine=` override may name, and the hosts the page's own
// Content-Security-Policy will let it contact.
//
// They did not agree. The bridge accepted `127.0.0.1`, `localhost`, `[::1]`
// and `::1`; the CSP names only `http://127.0.0.1:*`. CSP host matching is
// literal — it does not match `localhost` — and CSP's host grammar has no
// IPv6-literal form at all, so no source expression can ever cover `[::1]`.
// Three of the four accepted hostnames therefore produced a base URL the page
// was forbidden from contacting, and every fetch, EventSource and <audio src>
// against it failed and was reported to the user as "Engine unreachable": a
// true sentence about a cause nobody could guess.
//
// This lives in dev/ for the same mechanical reason as palette.test.ts — it
// reads repository files as text, and only dev/ has node's fs in this project.
// Reading the source rather than importing it also means no export is added to
// production code purely to be testable.

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const read = (rel: string): string => readFileSync(new URL(rel, import.meta.url), 'utf8');

const bridgeSrc = read('../src/bridge/index.ts');
const indexHtml = read('../index.html');

/** The literal members of `LOOPBACK_HOSTS`. */
function acceptedHosts(src: string): string[] {
  // Non-greedy to `])`, not `[^\]]*` — an IPv6 literal like '[::1]' contains a
  // `]`, so a character-class version stops inside the very entry this test
  // exists to reject and fails as a parse error instead of as the finding.
  const m = src.match(/const LOOPBACK_HOSTS = new Set\(\[([\s\S]*?)\]\);/);
  expect(m, 'bridge/index.ts must still declare LOOPBACK_HOSTS as a Set literal').not.toBeNull();
  return [...(m?.[1] ?? '').matchAll(/'([^']+)'/g)].map((x) => x[1] as string);
}

/** The value of one CSP directive from the meta tag. */
function directive(html: string, name: string): string {
  const m = html.match(new RegExp(`${name} ([^;"]+)`));
  expect(m, `index.html must still declare a ${name} directive`).not.toBeNull();
  return (m?.[1] ?? '').trim();
}

describe('the ?engine= allowlist and the CSP agree', () => {
  const hosts = acceptedHosts(bridgeSrc);
  const connect = directive(indexHtml, 'connect-src');
  const media = directive(indexHtml, 'media-src');

  it('finds both lists to compare', () => {
    expect(hosts.length).toBeGreaterThan(0);
    expect(connect).toContain('127.0.0.1');
    expect(media).toContain('127.0.0.1');
  });

  it('every host the bridge accepts is one the page may contact', () => {
    // `connect-src` covers fetch and EventSource; `media-src` covers the
    // <audio src> each deck plays. An engine reachable by one and not the
    // other is worse than one reachable by neither.
    const unreachable = hosts.filter((h) => !connect.includes(h) || !media.includes(h));
    expect(unreachable).toEqual([]);
  });

  it('accepts no host CSP cannot express', () => {
    // CSP's host-part grammar is ALPHA / DIGIT / "-" (plus dots and a leading
    // wildcard). An IPv6 literal cannot appear in a source expression at all,
    // so a bracketed host in this set can never be permitted, however the CSP
    // is widened.
    const inexpressible = hosts.filter((h) => h.includes(':') || h.includes('['));
    expect(inexpressible).toEqual([]);
  });
});
