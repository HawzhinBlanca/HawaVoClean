'use strict';

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

function fail(message) {
  process.stderr.write(`desktop proof packaging failed: ${message}\n`);
  process.exit(2);
}

function parseArgs(values) {
  const parsed = { engine: null, output: null, sourceSha: null, shellOnly: false };
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value === '--shell-only') {
      parsed.shellOnly = true;
      continue;
    }
    if (!['--engine', '--output', '--source-sha'].includes(value)) fail(`unknown argument ${value}`);
    const next = values[index + 1];
    if (!next) fail(`${value} requires a value`);
    index += 1;
    if (value === '--engine') parsed.engine = next;
    if (value === '--output') parsed.output = next;
    if (value === '--source-sha') parsed.sourceSha = next;
  }
  if (!parsed.output || !parsed.sourceSha) fail('--output and --source-sha are required');
  if (!/^[0-9a-f]{40}$/i.test(parsed.sourceSha)) fail('--source-sha must be a full Git SHA');
  if (parsed.shellOnly === Boolean(parsed.engine)) {
    fail('choose exactly one of --engine PATH or --shell-only');
  }
  return parsed;
}

if (process.platform !== 'darwin') fail('macOS desktop proof packaging requires a macOS host');
const args = parseArgs(process.argv.slice(2));
const desktopRoot = path.resolve(__dirname, '..');
const repositoryRoot = path.resolve(desktopRoot, '..');
const output = path.resolve(args.output);
if (fs.existsSync(output)) fail(`output already exists: ${output}`);
fs.mkdirSync(path.dirname(output), { recursive: true });

let engineSource;
if (args.shellOnly) {
  engineSource = path.join(desktopRoot, 'resources', 'engine', 'darwin-arm64');
} else {
  engineSource = path.resolve(args.engine);
  const launcher = path.join(engineSource, 'hawavoclean-engine');
  if (!fs.existsSync(launcher) || !fs.statSync(launcher).isFile()) {
    fail(`full engine launcher is missing: ${launcher}`);
  }
  try {
    fs.accessSync(launcher, fs.constants.X_OK);
  } catch {
    fail(`full engine launcher is not executable: ${launcher}`);
  }
  if (os.arch() !== 'arm64') fail('a full macOS arm64 app proof must be built on Apple silicon');
}

for (const required of [
  path.join(repositoryRoot, 'ui', 'dist', 'index.html'),
  path.join(desktopRoot, 'dist', 'main.js'),
]) {
  if (!fs.existsSync(required) || !fs.statSync(required).isFile()) fail(`build input is missing: ${required}`);
}

const toolchainBin = path.join(repositoryRoot, 'resolve-plugin', 'toolchain', 'node_modules', '.bin');
const pathDelimiter = process.platform === 'win32' ? ';' : ':';
const currentPath = process.env.PATH || '';
const augmentedPath = fs.existsSync(toolchainBin) && !currentPath.includes(toolchainBin)
  ? `${toolchainBin}${pathDelimiter}${currentPath}`
  : currentPath;

const cli = require.resolve('electron-builder/out/cli/cli.js');
const env = {
  ...process.env,
  PATH: augmentedPath,
  CSC_IDENTITY_AUTO_DISCOVERY: 'false',
  HAWAVOCLEAN_RELEASE_BUILD: '0',
  HAWAVOCLEAN_DESKTOP_PROOF_BUILD: '1',
  HAWAVOCLEAN_DESKTOP_ENGINE_MODE: args.shellOnly ? 'shell-only' : 'full',
  HAWAVOCLEAN_DESKTOP_ENGINE_SOURCE: engineSource,
  HAWAVOCLEAN_DESKTOP_OUTPUT: output,
  HAWAVOCLEAN_RELEASE_SOURCE_SHA: args.sourceSha.toLowerCase(),
  HAWAVOCLEAN_TARGET_PLATFORM: 'darwin',
  HAWAVOCLEAN_TARGET_ARCH: 'arm64',
};
const result = spawnSync(
  process.execPath,
  [cli, '--dir', '--mac', '--arm64', '--config', 'electron-builder.config.cjs'],
  { cwd: desktopRoot, env, stdio: 'inherit' },
);
if (result.error) fail(result.error.message);
if (result.status !== 0) process.exit(result.status ?? 1);

const app = path.join(output, 'mac-arm64', 'HawaVoClean.app');
if (!fs.existsSync(app) || !fs.statSync(app).isDirectory()) fail(`packaged app is missing: ${app}`);
const seal = spawnSync(
  'codesign',
  ['--force', '--deep', '--sign', '-', '--timestamp=none', app],
  { cwd: desktopRoot, env: process.env, encoding: 'utf8' },
);
if (seal.error) fail(seal.error.message);
if (seal.status !== 0) fail((seal.stderr || seal.stdout || 'ad-hoc sealing failed').trim());
process.stdout.write(`${JSON.stringify({ app, engineMode: env.HAWAVOCLEAN_DESKTOP_ENGINE_MODE })}\n`);
