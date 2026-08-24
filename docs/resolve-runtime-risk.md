# Resolve-owned Electron runtime risk

Status: **open vendor risk — explicit user acceptance required before a 10/10 release**  
Captured: 2026-08-21  
Release task: T4.6

## Decision summary

HawaVoClean's controlled standalone test shell is locked to Electron 43.4.1. The GitHub Advisory
Database returned zero advisories for that exact version on the capture date.

The installed DaVinci Resolve Studio 21.0.3 host (build 21.0.30007) instead bundles Electron 36.3.2.
That runtime is owned and updated by Blackmagic Design, is outside every HawaVoClean lockfile, and is
outside Electron's supported three-major window. An exact-version GitHub Advisory Database query
returned 33 advisories: 7 high, 20 medium and 6 low. The dependency estate is therefore **not clean**.

HawaVoClean cannot patch or replace the runtime embedded in Resolve. It can—and now does—remove the
known application-side preconditions wherever possible. One high advisory remains intrinsically
dependent on keeping all renderer content trusted, and the advisory states that no application-side
workaround exists. This residual risk is bounded but not eliminated.

Sources:

- [Electron 36.3.2 release identity](https://releases.electronjs.org/release/v36.3.2)
- [Electron support policy and timelines](https://www.electronjs.org/docs/latest/tutorial/electron-timelines)
- [Electron security checklist](https://www.electronjs.org/docs/latest/tutorial/security)
- Machine-readable capture: `evidence/release/t4.6-resolve-runtime-proof.json`
- Reproducible capture command: `uv run python scripts/audit_resolve_runtime.py`

## Exact installed host evidence

| Property | Captured value |
|---|---|
| Resolve application | `/Applications/DaVinci Resolve/DaVinci Resolve.app` |
| Resolve version/build | 21.0.3 / 21.0.30007 |
| Embedded application | `Contents/Applications/.hidden/Electron.app` |
| Embedded Electron | 36.3.2 |
| Electron executable SHA-256 | `c925e48e8e7effdd84680cf067d9d2abea2449754e5fda35a72ab7b6532a315e` |
| Code-signing identity | Developer ID Application: Blackmagic Design Inc (`9ZGFBWLSYP`) |
| Code-sign verification | valid on disk; hardened runtime enabled |
| Embedded transport declaration | `NSAllowsArbitraryLoads = true` |
| Embedded privacy declarations | camera and microphone usage strings present |

The broad transport/privacy declarations belong to Blackmagic's generic embedded Electron
application. HawaVoClean does not rely on them: its own session denies unexpected permissions and
filters renderer requests to the packaged UI plus the one authenticated engine port.

Disk evidence identifies the installed vendor bundle. T6 must still record `process.versions` from an
actual in-Resolve launch, because only that proves which binary Resolve selected at runtime.

## High advisory disposition

The exact-version scan returned these seven high advisories. “Bounded” means the published vulnerable
configuration has been removed from HawaVoClean; it does not mean the old Electron binary is patched.

| Advisory | Published condition | HawaVoClean disposition |
|---|---|---|
| [GHSA-h7rp-cf8h-j98x](https://github.com/advisories/GHSA-h7rp-cf8h-j98x), CVE-2026-70601 | Promise-returning `contextBridge` functions plus untrusted renderer content can bypass context isolation; no application-side workaround | **Residual vendor risk.** The bridge necessarily returns promises. The renderer loads only checksum-covered local content from `hawa://app`, but a future content-injection defect would meet the advisory's remaining precondition. |
| [GHSA-9f4c-93c8-jc8g](https://github.com/advisories/GHSA-9f4c-93c8-jc8g), CVE-2026-70608 | A sandboxed iframe can bypass its popup restriction | Bounded: CSP has `frame-src 'none'`, webviews are disabled, attachment is denied, and `setWindowOpenHandler` denies every window. |
| [GHSA-v3j7-r9gq-3gjw](https://github.com/advisories/GHSA-v3j7-r9gq-3gjw), CVE-2026-70604 | A fetch-enabled custom protocol without CORS enforcement allows cross-origin reads | Published workaround applied: `hawa` is registered with `corsEnabled: true`; its handler serves only real files confined below the plugin root, and no remote content is loaded. |
| [GHSA-9wfr-w7mm-pc7f](https://github.com/advisories/GHSA-9wfr-w7mm-pc7f), CVE-2026-34769 | Untrusted properties spread into `webPreferences` can inject renderer command-line switches | Not reachable: every preference is a fixed literal; no external object is spread into window options. |
| [GHSA-8337-3p73-46f4](https://github.com/advisories/GHSA-8337-3p73-46f4), CVE-2026-34771 | An asynchronous permission callback can race a destroyed frame | Published workaround applied: the permission handler decides and invokes its callback synchronously; fullscreen, pointer-lock and keyboard-lock are denied. |
| [GHSA-jjp3-mq3x-295m](https://github.com/advisories/GHSA-jjp3-mq3x-295m), CVE-2026-34770 | Applications using `powerMonitor` events may trigger a use-after-free | Not reachable from the plugin: it neither imports nor accesses `powerMonitor`. |
| [GHSA-532v-xpq5-8h95](https://github.com/advisories/GHSA-532v-xpq5-8h95), CVE-2026-34774 | Offscreen rendering plus an allowed child window can trigger a use-after-free | Not reachable: offscreen rendering is not enabled and every child window is denied. |

All 33 advisory identifiers and the canonical response hash are retained in the machine-readable
capture. Medium and low findings are not silently discarded; the capture is the exhaustive inventory.

## Compensating controls enforced by HawaVoClean

- A private, non-persistent `hawavoclean-isolated` Electron session; it does not share Resolve's
  default cookies/cache and disappears at process exit.
- A standard secure `hawa://app` protocol instead of over-privileged `file://` navigation. The handler
  rejects traversal and symlink escapes using canonical filesystem paths.
- Local, checksum-covered UI only. No CDN, remote page, remote script or plugin-supplied HTML.
- CSP denies objects, frames, forms, foreign scripts and undeclared connections; it contains no
  `unsafe-eval`. The narrow inline-style exception exists only for current UI styling.
- Main-process request filtering allows UI files and exactly `http://127.0.0.1:<spawned-port>`.
  HTTPS, arbitrary loopback ports and non-loopback traffic are denied.
- Sandbox and context isolation on; Node integration, webviews, insecure content and drag navigation
  off. Popups and external navigation are denied with no `shell.openExternal` side effect.
- All permissions denied except sanitized clipboard write from the exact main renderer. Device
  permission requests are always denied and permission decisions are synchronous.
- Every privileged IPC handler validates the exact main `webContents`, main frame and `hawa://app`
  entry URL. The preload exposes only declared typed operations, never raw IPC or Electron objects.
- Each standalone/staged lifecycle probe verifies CSP behavior, remote-fetch denial, popup denial,
  permission denial, worker functionality, authenticated loopback health and clean engine shutdown.

These controls follow Electron's current security checklist, including navigation/window denial,
permission handling, CSP, sender validation and replacing `file://` with a restricted custom protocol.

## Acceptance boundary

T4.6 cannot be marked complete and the release cannot be called 10/10 until the user explicitly
accepts this vendor-owned residual risk or Blackmagic ships a qualifying Resolve update.

Acceptance means all of the following are understood:

1. Resolve's signed Electron 36.3.2 binary contains known high-severity advisories and is unsupported
   upstream even though HawaVoClean removes the known reachable configurations.
2. HawaVoClean must only load the installed, checksum-verified local UI; adding remote/untrusted
   renderer content voids this assessment.
3. Resolve must be upgraded when Blackmagic ships a release whose embedded Electron is in the current
   supported window and whose exact version has no unaccepted high/critical findings.
4. A new Electron advisory, a changed embedded binary hash, or a changed Resolve build invalidates
   this snapshot and requires re-capture and re-acceptance.

Until then, the honest release state is: **controllable boundary hardened; vendor high risk pending
acceptance**.
