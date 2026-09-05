// @vitest-environment node
// Security regression checks over the renderer sources. Authentication must
// remain owned by Electron main (or an HttpOnly same-origin web cookie), never
// reconstructed from URLs or browser storage.

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const read = (rel: string): string => readFileSync(new URL(rel, import.meta.url), 'utf8');

const bridgeSrc = read('../src/bridge/index.ts');
const bridgeTypes = read('../src/bridge/types.ts');
const clientSrc = read('../src/api/client.ts');
const indexHtml = read('../index.html');

function directive(html: string, name: string): string {
  const match = html.match(new RegExp(`${name} ([^;\"]+)`));
  expect(match, `index.html must still declare a ${name} directive`).not.toBeNull();
  return (match?.[1] ?? '').trim();
}

describe('renderer authentication stays out of URLs and storage', () => {
  it('has no query/localStorage credential fallback', () => {
    const executableSource = `${bridgeSrc}\n${clientSrc}`
      .split('\n')
      .filter((line) => !line.trim().startsWith('//'))
      .join('\n');
    expect(executableSource).not.toMatch(/localStorage|sessionStorage|hawa\.token/);
    expect(executableSource).not.toMatch(/searchParams\.(?:get|set)\(['"]token/);
    expect(executableSource).not.toMatch(/X-Hawa-Token|Authorization\s*:/);
  });

  it('exposes only baseUrl through the bridge endpoint contract', () => {
    expect(bridgeTypes).toMatch(/getEndpoint\(\): Promise<\{ baseUrl: string \}>/);
    expect(bridgeTypes).not.toMatch(/token|authorization/i);
  });

  it('makes unsupported cross-origin development authentication explicit', () => {
    expect(bridgeSrc).toContain('Web development authentication is unavailable');
    expect(bridgeSrc).toContain('window.location.href');
    expect(bridgeSrc).not.toContain("searchParams.get('engine')");
  });
});

describe('the renderer CSP keeps engine traffic on self or numeric loopback', () => {
  it('confines fetch/EventSource and media without a wildcard', () => {
    const connect = directive(indexHtml, 'connect-src');
    const media = directive(indexHtml, 'media-src');
    expect(connect).toContain("'self'");
    expect(connect).toContain('http://127.0.0.1:*');
    expect(media).toContain("'self'");
    expect(media).toContain('http://127.0.0.1:*');
    expect(connect).not.toContain('*://');
    expect(media).not.toContain('*://');
  });
});
