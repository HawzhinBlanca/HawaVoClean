"""Static ownership tests for the Electron/Resolve trust boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.audit_resolve_runtime import _summarize_advisories

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "resolve-plugin" / "com.hawavoclean.resolve"


def test_browser_window_enables_every_required_isolation_control() -> None:
    main = (PLUGIN / "main.js").read_text(encoding="utf-8")

    for setting in (
        "sandbox: true",
        "contextIsolation: true",
        "nodeIntegration: false",
        "partition: SESSION_PARTITION",
        "webSecurity: true",
        "allowRunningInsecureContent: false",
        "webviewTag: false",
        "navigateOnDragDrop: false",
        "safeDialogs: true",
    ):
        assert setting in main
    assert "setWindowOpenHandler" in main
    assert "will-navigate" in main
    assert "will-attach-webview" in main
    assert "shell.openExternal(" not in main


def test_permissions_network_and_ipc_fail_closed() -> None:
    main = (PLUGIN / "main.js").read_text(encoding="utf-8")
    auth = (PLUGIN / "session-auth.js").read_text(encoding="utf-8")

    assert "setPermissionRequestHandler" in main
    assert "session.fromPartition(SESSION_PARTITION)" in main
    assert "const SESSION_PARTITION = 'hawavoclean-isolated'" in main
    assert "protocol.registerSchemesAsPrivileged" in main
    assert "appSession.protocol.handle(APP_SCHEME" in main
    assert "corsEnabled: true" in main
    assert "allowServiceWorkers: false" in main
    assert "bypassCSP: false" in main
    assert "fs.realpathSync(candidate)" in main
    assert "pathTraversalBlocked" in main
    assert "serviceWorkerBlocked" in main
    assert "setPermissionCheckHandler" in main
    assert "setDevicePermissionHandler(() => false)" in main
    assert "permission === 'clipboard-sanitized-write'" in main
    assert "webRequest.onBeforeRequest" in main
    assert "webRequest.onBeforeSendHeaders" in main
    assert "{ urls: ['http://127.0.0.1:*/*'] }" in main
    assert "isEngineApiRequest(details.url, engineOrigin())" in main
    assert "url.pathname !== '/api/session'" in main
    assert "withEngineAuthorization(details.requestHeaders, authorization)" in main
    assert "url.hostname !== '127.0.0.1'" in auth
    assert "url.origin === engineOrigin" in auth
    assert "url.pathname.startsWith('/api/')" in auth
    assert "IPC sender is not the trusted HawaVoClean renderer" in main
    assert main.count("requireTrustedIpcSender(event);") == 9


def test_preload_exposes_only_the_declared_narrow_bridge() -> None:
    preload = (PLUGIN / "preload.js").read_text(encoding="utf-8")

    assert "contextBridge.exposeInMainWorld('hawa', bridge)" in preload
    assert "ipcRenderer.sendSync('hawa:host')" in preload
    assert "ipcRenderer.invoke(channel" in preload
    assert "ipcRenderer.send(" not in preload
    assert "ipcRenderer.on(" not in preload
    assert "const { contextBridge, ipcRenderer, webUtils } = require('electron')" in preload
    assert "child_process" not in preload
    for secret_marker in (
        "X-Hawa-Token",
        "Authorization",
        "sessionToken",
        "hawa_session",
        "localStorage",
        "sessionStorage",
    ):
        assert secret_marker not in preload


def test_ui_csp_allows_only_local_assets_loopback_engine_and_declared_workers() -> None:
    html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")

    assert 'http-equiv="Content-Security-Policy"' in html
    for directive in (
        "default-src 'self'",
        "script-src 'self'",
        "connect-src 'self' http://127.0.0.1:*",
        "worker-src 'self' blob:",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-src 'none'",
    ):
        assert directive in html
    assert "'unsafe-eval'" not in html
    assert '<meta name="referrer" content="no-referrer"' in html


def test_runtime_version_is_captured_without_conflating_resolve_and_standalone() -> None:
    main = (PLUGIN / "main.js").read_text(encoding="utf-8")
    readme = (ROOT / "resolve-plugin" / "README.md").read_text(encoding="utf-8")
    risk = (ROOT / "docs" / "resolve-runtime-risk.md").read_text(encoding="utf-8")

    assert "process.versions.electron" in main
    assert "process.versions.chrome" in main
    assert "runtime evidence:" in main
    assert "runtime owned and shipped" in readme
    assert "does not infer" in readme
    assert "Electron 36.3.2" in risk
    assert "7 high, 20 medium and 6 low" in risk
    assert "dependency estate is therefore **not clean**" in risk
    assert "explicit user acceptance required" in risk


def test_advisory_capture_keeps_exact_ranges_and_high_findings() -> None:
    raw: list[dict[str, Any]] = [
        {
            "ghsa_id": "GHSA-high",
            "cve_id": "CVE-high",
            "severity": "high",
            "summary": "high finding",
            "published_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-02T00:00:00Z",
            "html_url": "https://github.com/advisories/GHSA-high",
            "vulnerabilities": [
                {
                    "package": {"ecosystem": "npm", "name": "electron"},
                    "vulnerable_version_range": "< 39.8.9",
                    "first_patched_version": {"identifier": "39.8.9"},
                }
            ],
        },
        {
            "ghsa_id": "GHSA-medium",
            "cve_id": "CVE-medium",
            "severity": "medium",
            "summary": "medium finding",
            "published_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-02T00:00:00Z",
            "html_url": "https://github.com/advisories/GHSA-medium",
            "vulnerabilities": [],
        },
    ]

    report = _summarize_advisories("36.3.2", "https://example.test/query", raw)

    assert report["total"] == 2
    assert report["severity_counts"] == {"high": 1, "medium": 1}
    assert [item["ghsa_id"] for item in report["high_or_critical"]] == ["GHSA-high"]
    assert report["high_or_critical"][0]["electron_ranges"] == [
        {"vulnerable_version_range": "< 39.8.9", "first_patched_version": "39.8.9"}
    ]
    assert len(report["response_canonical_sha256"]) == 64
