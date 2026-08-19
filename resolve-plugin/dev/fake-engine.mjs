#!/usr/bin/env node
/*
 * fake-engine.mjs — a stand-in for `hawavoclean serve` used to test the Resolve/desktop shell
 * without the Python engine. Node built-ins only.
 *
 * Mimics the subset of docs/ui-contract.md section 1 the shell depends on:
 *   - argv: serve [--host 127.0.0.1] [--port 0] --token TOKEN [--ui-dir DIR]
 *   - prints exactly one JSON "ready" line to stdout, everything else to stderr
 *   - GET  /api/health     (token via X-Hawa-Token header or ?token=)
 *   - POST /api/shutdown   (responds {"ok":true}, exits within 1 s)
 *   - CORS: * / X-Hawa-Token, Content-Type
 *   - static files from --ui-dir at / (404 JSON when absent)
 *
 * Failure modes for exercising the shell's error handling (env FAKE_ENGINE_MODE):
 *   ok        normal (default)
 *   crash     write a traceback-ish message to stderr and exit 3 before ready
 *   hang      never print ready (exits normally on SIGTERM)
 *   stubborn  ready, but ignores /api/shutdown and SIGTERM (forces the shell's SIGKILL path)
 */

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const VERSION = '0.0.0-fake';
const mode = process.env.FAKE_ENGINE_MODE || 'ok';

function parseArgs(argv) {
  const out = { host: '127.0.0.1', port: 0, token: null, uiDir: null, sub: null };
  const args = argv.slice();
  if (args.length && !args[0].startsWith('--')) out.sub = args.shift();
  for (let i = 0; i < args.length; i += 1) {
    const a = args[i];
    const next = () => {
      i += 1;
      return args[i];
    };
    if (a === '--host') out.host = next();
    else if (a === '--port') out.port = Number(next());
    else if (a === '--token') out.token = next();
    else if (a === '--ui-dir') out.uiDir = next();
    else if (a.startsWith('--port=')) out.port = Number(a.slice(7));
    else if (a.startsWith('--token=')) out.token = a.slice(8);
    else if (a.startsWith('--ui-dir=')) out.uiDir = a.slice(9);
    else if (a.startsWith('--host=')) out.host = a.slice(7);
  }
  return out;
}

const opts = parseArgs(process.argv.slice(2));
const log = (...a) => console.error('[fake-engine]', ...a);

if (opts.sub !== 'serve') {
  log(`unexpected subcommand ${JSON.stringify(opts.sub)}; expected "serve"`);
  process.exit(2);
}
if (!opts.token) {
  log('--token is required');
  process.exit(2);
}

if (mode === 'crash') {
  console.error('Traceback (most recent call last):');
  console.error('  File "hawavoclean/server/app.py", line 1, in <module>');
  console.error('    import fastapi');
  console.error("ModuleNotFoundError: No module named 'fastapi' (simulated by FAKE_ENGINE_MODE=crash)");
  process.exit(3);
}

if (mode === 'hang') {
  log('mode=hang: never reporting ready');
  setInterval(() => {}, 1 << 30);
  process.on('SIGTERM', () => {
    log('SIGTERM in hang mode; exiting');
    process.exit(0);
  });
} else {
  startServer();
}

function sendJson(res, status, body, extraHeaders = {}) {
  const data = JSON.stringify(body);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(data),
    ...extraHeaders,
  });
  res.end(data);
}

function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'X-Hawa-Token, Content-Type');
  res.setHeader('Access-Control-Max-Age', '600');
}

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.wasm': 'application/wasm',
};

function serveStatic(req, res, urlPath) {
  if (!opts.uiDir) return sendJson(res, 404, { error: 'not_found', message: 'no --ui-dir configured' });
  const rel = decodeURIComponent(urlPath === '/' ? '/index.html' : urlPath);
  const root = path.resolve(opts.uiDir);
  const file = path.resolve(root, '.' + rel);
  if (!file.startsWith(root + path.sep) && file !== root) return sendJson(res, 403, { error: 'forbidden', message: 'path escapes ui dir' });
  fs.stat(file, (err, st) => {
    if (err || !st.isFile()) return sendJson(res, 404, { error: 'not_found', message: rel });
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream', 'Content-Length': st.size });
    if (req.method === 'HEAD') return res.end();
    fs.createReadStream(file).pipe(res);
  });
}

function startServer() {
  let stubborn = mode === 'stubborn';
  const server = http.createServer((req, res) => {
    cors(res);
    try {
      handleRequest(req, res);
    } catch (err) {
      // A malformed request (bad %-encoding, garbage URL, NUL in path) must not kill the server.
      log(`request error: ${err.message}`);
      if (!res.headersSent) sendJson(res, 400, { error: 'bad_request', message: String(err.message || err) });
      else res.destroy();
    }
  });

  function handleRequest(req, res) {
    const url = new URL(req.url, 'http://127.0.0.1');
    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      return res.end();
    }
    if (url.pathname.startsWith('/api/')) {
      const token = req.headers['x-hawa-token'] || url.searchParams.get('token');
      if (token !== opts.token) return sendJson(res, 401, { error: 'unauthorized' });
      if (url.pathname === '/api/health' && req.method === 'GET') {
        return sendJson(res, 200, { ok: true, version: VERSION, profiles: ['studio', 'production'], engine_pid: process.pid });
      }
      if (url.pathname === '/api/shutdown' && req.method === 'POST') {
        req.resume();
        sendJson(res, 200, { ok: true });
        if (stubborn) {
          log('mode=stubborn: ignoring /api/shutdown');
        } else {
          log('shutdown requested; exiting');
          setTimeout(() => process.exit(0), 100);
        }
        return undefined;
      }
      return sendJson(res, 404, { error: 'not_found', message: `${req.method} ${url.pathname}` });
    }
    return serveStatic(req, res, url.pathname);
  }

  server.on('error', (err) => {
    log(`server error: ${err.message}`);
    process.exit(1);
  });

  server.listen(opts.port, opts.host, () => {
    const { port } = server.address();
    log(`listening on http://${opts.host}:${port} (mode=${mode}, ui-dir=${opts.uiDir || '-'})`);
    // The one and only stdout line.
    process.stdout.write(JSON.stringify({ event: 'ready', port, pid: process.pid, version: VERSION }) + '\n');
  });

  process.on('SIGTERM', () => {
    if (stubborn) {
      log('mode=stubborn: ignoring SIGTERM');
      return;
    }
    log('SIGTERM; exiting');
    process.exit(0);
  });
  process.on('SIGINT', () => process.exit(0));
}
