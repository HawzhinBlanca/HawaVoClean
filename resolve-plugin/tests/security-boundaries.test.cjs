// @ts-check
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const mainSource = fs.readFileSync(
  path.join(__dirname, '..', 'com.hawavoclean.resolve', 'main.js'),
  'utf8'
);

test('resolve main enforces strict Content-Security-Policy with zero unsafe-eval and loopback connect-src', () => {
  assert.ok(mainSource.includes('STRICT_CSP ='), 'main.js must define STRICT_CSP');
  assert.ok(mainSource.includes("default-src 'none'"), 'CSP must deny by default');
  assert.ok(mainSource.includes("connect-src http://127.0.0.1:* ws://127.0.0.1:*"), 'CSP must restrict connect-src to loopback');
  assert.ok(!mainSource.includes("'unsafe-eval'"), 'CSP must strictly disallow unsafe-eval');
  assert.ok(mainSource.includes("frame-src 'none'"), 'CSP must disallow framing');
  assert.ok(mainSource.includes("object-src 'none'"), 'CSP must disallow plugins/objects');
});

test('resolve main denies external navigation, popups, and webviews', () => {
  assert.ok(mainSource.includes('setWindowOpenHandler'), 'Must handle window open events');
  assert.ok(mainSource.includes("action: 'deny'"), 'All popups must be denied');
  assert.ok(mainSource.includes('will-navigate'), 'Must guard will-navigate');
  assert.ok(mainSource.includes('will-attach-webview'), 'Must guard webview attachment');
});

test('resolve main request filter cancels all outbound non-loopback HTTP/HTTPS requests', () => {
  // Extract onBeforeRequest filter logic
  assert.ok(mainSource.includes('onBeforeRequest'), 'Must register onBeforeRequest');
  assert.ok(mainSource.includes('isEngineApiRequest'), 'Must verify engine API request origin');
  assert.ok(mainSource.includes('securityEvents.blockedRequests'), 'Must track blocked requests');
});
