# HawaVoClean desktop foundation

This package is the standalone Electron host for the existing `../ui` renderer. It does not
fork or copy the React application. Development uses the repository `.venv/bin/hawavoclean`
engine by default; an alternative command must be supplied as a JSON array in
`HAWAVOCLEAN_DESKTOP_ENGINE_COMMAND`.

```sh
pnpm install --frozen-lockfile
pnpm test
pnpm typecheck
pnpm verify:config
pnpm package:dir
```

`package:dir` deliberately produces an unsigned, unpacked development application. Release
scripts run only on the matching native host and fail before packaging unless the platform engine,
full `HAWAVOCLEAN_RELEASE_SOURCE_SHA`, and signing credentials are present. macOS additionally
requires Apple notarization credentials. Release scripts never publish artifacts.

The exact release gate uses `scripts/package-proof.cjs` to create a separate source-bound macOS
arm64 `.app`. Its full mode refuses a missing/non-executable engine; hosted CI uses the explicit
`--shell-only` mode only to test packaging mechanics. `scripts/validate_desktop_app.py` verifies the
complete canonical tree, package provenance, ASAR-integrity metadata, unsigned/ad-hoc identity and,
in full mode, every engine checksum and symlink before and after the engine smoke. The integrity check
recomputes Electron's exact Chromium-Pickle ASAR header SHA-256; the full exact gate also launches the
packaged executable and repeats the source shell's bridge, health, authentication and network-sandbox
assertions. That hidden packaged self-test is enabled only by the full, non-distributable proof marker;
shell-only and ordinary release packages cannot enable it. This proof carries
`distribution_eligible: false`: it is not a substitute for Developer ID signing, notarization,
stapling or a final DMG/ZIP.

The preload exposes only fixed operations: base-URL-only engine endpoint retrieval, native audio/folder/export
dialogs, registered-file drops, reveal-in-folder, immutable app metadata, and explicit diagnostics/update placeholders.
It exposes no Node object, raw filesystem API, generic IPC method, URL launcher, or arbitrary
channel subscription.

The engine bootstrap secret never crosses preload IPC. Electron main exchanges it for a bounded
15-minute engine session, renews before expiry, and injects the short-lived Bearer header only into
the exact spawned `127.0.0.1` origin under `/api/`. Renderer-supplied auth and cookies are stripped;
session-minting requests from the renderer, foreign origins, ports, and non-API engine paths are
blocked. Audio and SSE URLs contain no credential.

Native selections do not make renderer-supplied paths trustworthy. Main registers each dialog or
dropped path with the broker through the root-only `/api/v1/native-sources` route before returning
the canonical path to the compatibility UI. The renderer session may then read/process only that
exact still-identical regular file, a marker-owned managed upload, or a completed job's verified
immutable artifact. If a dropped `File` cannot be registered, the UI uploads its bytes into managed
storage. The legacy synchronous `pathForFile` bridge always returns `null`.

The native media picker and renderer validation accept only WAV/WAVE, AIF/AIFF/AIFC, FLAC, MP3,
M4A and MP4. MP4 supplies extracted audio; the desktop app does not silently rewrite video.
