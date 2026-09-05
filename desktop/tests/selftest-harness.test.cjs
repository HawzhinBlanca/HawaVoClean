'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const { assertSelfTestResult } = require('../scripts/selftest-harness.cjs');

function resultFixture() {
  return {
    host: 'electron',
    healthStatus: 200,
    enginePid: 123,
    authenticatedPostStatus: 400,
    authenticatedMediaStatus: 400,
    mediaNoStore: true,
    sessionClearing: {
      ok: true,
      clearedItems: ['session_http_cache', 'session_storage_data', 'session_code_cache'],
      retainedItems: ['exported_wav_masters', 'exported_processing_records', 'user_source_media', 'engine_jobs_database'],
    },
    endpointKeys: ['baseUrl'],
    endpointHasSecret: false,
    endpointUrlHasAuth: false,
    remoteBlocked: true,
    popupBlocked: true,
    nodeHidden: true,
    topLevelKeys: ['app', 'diagnostics', 'engine', 'files', 'host', 'session', 'updates'],
    app: { name: 'HawaVoClean', version: '3.3.0', packaged: true },
    renderer: {
      contract: 'hawavoclean-react-v1',
      challengeVerified: true,
      uiVersion: '3.3.0',
      domContract: 'hawavoclean-react-v1',
      domUiVersion: '3.3.0',
      rootChildCount: 1,
      shellIsOnlyRootChild: true,
      shellTag: 'DIV',
      shellClass: 'app',
      missingSelectors: [],
      brandText: 'HAWAVOCLEANv3.3.0',
      openFileText: 'Open file…',
      processLabel: 'PROCESS',
      display: 'grid',
      width: 1280,
      height: 820,
      stylesheetCount: 2,
      moduleScriptCount: 1,
      documentTitle: 'HawaVoClean',
    },
    diagnostics: {
      status: 'ready',
      optIn: false,
      canExport: false,
      reason: 'Opt-in required to retain diagnostics',
      pendingErrorCount: 0,
      telemetryEgress: 'none',
    },
    updates: { status: 'disabled', reason: 'release_feed_not_configured', canCheck: false },
  };
}

test('desktop self-test accepts the complete committed React renderer evidence', () => {
  assert.doesNotThrow(() => assertSelfTestResult(resultFixture(), true));
});

test('desktop self-test rejects the former false green with no renderer evidence', () => {
  const result = resultFixture();
  delete result.renderer;
  assert.throws(() => assertSelfTestResult(result, true));
});

test('desktop self-test rejects a static DOM marker without a verified App challenge', () => {
  const result = resultFixture();
  result.renderer.challengeVerified = false;
  assert.throws(() => assertSelfTestResult(result, true), /challengeVerified/);
});

test('desktop self-test rejects blank, partial, unstyled, or version-skewed renderers', () => {
  for (const renderer of [
    { ...resultFixture().renderer, shellIsOnlyRootChild: false },
    { ...resultFixture().renderer, missingSelectors: ['main.main'] },
    { ...resultFixture().renderer, display: 'block' },
    { ...resultFixture().renderer, width: 0, height: 0 },
    { ...resultFixture().renderer, stylesheetCount: 1 },
    { ...resultFixture().renderer, moduleScriptCount: 0 },
    { ...resultFixture().renderer, uiVersion: '3.2.0' },
  ]) {
    const result = resultFixture();
    result.renderer = renderer;
    assert.throws(() => assertSelfTestResult(result, true));
  }
});
