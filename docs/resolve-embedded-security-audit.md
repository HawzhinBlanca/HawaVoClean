# DaVinci Resolve Embedded Electron Security & Accessibility Audit

**Status:** Certified Audit & Hardening Specification  
**Task Reference:** Task Q5.6 (`docs/true-10-readiness-task-sheet.md`)  
**Date:** 2026-09-06  

---

## 1. Executive Summary

DaVinci Resolve Studio hosts Workflow Integration plugins inside an embedded Electron runtime owned and maintained by Blackmagic Design (version 36.3.2 in DaVinci Resolve Studio 21.0.3). Because this vendor runtime cannot be directly updated by plugin authors, HawaVoClean enforces an airtight defense-in-depth security perimeter and accessibility standards to neutralize all potential threat vectors.

### Core Security Invariants:
1. **Zero Cloud Credentials & Zero External Network Egress:** The embedded Electron runtime is strictly air-gapped. Network requests are filtered at the Chromium session layer; all non-loopback and all non-engine-port traffic is synchronously cancelled.
2. **Strict Network & Response CSP:** Content Security Policy (`Content-Security-Policy`) is enforced on all loaded documents and assets with `default-src 'none'`, `script-src 'self'`, and `connect-src http://127.0.0.1:*`.
3. **No Dynamic Navigation / No Popups:** All window opening (`setWindowOpenHandler`), external navigation (`will-navigate`), and webview attachments (`will-attach-webview`) are denied.
4. **Sandboxed & Context-Isolated Process Model:** `sandbox: true`, `contextIsolation: true`, and `nodeIntegration: false` strictly prevent renderer execution of Node.js APIs.
5. **IPC Sender Validation:** All IPC invocations are validated via `isTrustedIpcSender`, verifying the main frame and exact `hawa://app/index.html` URL origin.
6. **Full Keyboard & Screen-Reader Accessibility:** UI adheres to WCAG AA contrast, keyboard focus rings, `axe-core` accessibility audits with zero serious or critical violations, and global reduced motion support.

---

## 2. Host Runtime Inventory

| Host Property | Qualified Specification | Enforced Value / Behavior |
|---|---|---|
| **Host Application** | DaVinci Resolve Studio 20.1+ | Tested against 21.0.3 |
| **Embedded Runtime** | Blackmagic Electron | Version 36.3.2 (hardened runtime enabled) |
| **Electron Session** | Private in-memory partition | `SESSION_PARTITION = 'hawavoclean-isolated'` (never persists to disk) |
| **App Protocol** | Custom secure scheme | `hawa://app/` (confined to installed plugin root, path-traversal proof) |
| **Origin Isolation** | `http://127.0.0.1:<port>` | Dynamic random loopback port, authenticated by root token in main |

---

## 3. Network Boundary & Egress Prevention

The embedded runtime has **zero external egress**:

```mermaid
flowchart LR
    subgraph Embedded Electron Renderer
        UI["HawaVoClean UI (hawa://app)"]
    end
    
    subgraph Main Process Session Interceptor
        Filter{"onBeforeRequest Filter"}
        HeaderAuth["onBeforeSendHeaders Injector"]
    end
    
    subgraph Loopback Engine Boundary
        Engine["Python Engine (127.0.0.1:port)"]
    end
    
    subgraph Blocked Targets
        Internet["External Internet / Cloud (HTTPS)"]
        Arbitrary["Other Loopback Ports"]
        Filesystem["Arbitrary Local Files"]
    end

    UI -->|"Request"| Filter
    Filter -- "hawa://app/*" --> UI
    Filter -- "127.0.0.1:engine_port/api/*" --> HeaderAuth --> Engine
    Filter -- "https://* (Cloud/Remote)" -->|"DENIED"| Internet
    Filter -- "127.0.0.1:other_port" -->|"DENIED"| Arbitrary
    Filter -- "file://* outside plugin" -->|"DENIED"| Filesystem
```

### Compensating Controls:
- **`onBeforeRequest` Rule:** Explicitly cancels any URL whose protocol is not `hawa:` (host `app`), `file:` (strictly inside plugin directory), or `http:` (matching `exactEngineOrigin()` and non-session API endpoint).
- **No Cloud SDKs:** The Resolve plugin shell contains no AWS, GCP, or external cloud SDKs or credential caches. Cloud acceleration (C6) is isolated to backend infrastructure and requires explicit opt-in.

---

## 4. Content Security Policy (CSP)

The session injects and enforces the following strict Content Security Policy across all responses:

```http
Content-Security-Policy: default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob: http://127.0.0.1:*; connect-src http://127.0.0.1:* ws://127.0.0.1:*; font-src 'self'; object-src 'none'; frame-src 'none'; base-uri 'none'; form-action 'none';
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
```

---

## 5. Accessibility & Interaction Qualification

1. **Keyboard Navigation:**
   - Interactive elements (`button`, `input`, `select`, summary disclosures) are fully reachable via <kbd>Tab</kbd> / <kbd>Shift</kbd>+<kbd>Tab</kbd>.
   - Clear high-contrast focus rings (`outline: 2px solid var(--ix-focus); outline-offset: 2px`).
   - Modal dialogs and disclosures support <kbd>Escape</kbd> and <kbd>Enter</kbd>/<kbd>Space</kbd> activation.
2. **Screen Reader (VoiceOver) Support:**
   - ARIA live regions for batch status announcements (`aria-live="polite"`).
   - Explicit `aria-label` names on all progress indicators and icon buttons.
   - Zero serious or critical axe findings (`axe-core` audited).
3. **Reduced Motion:**
   - Global collapse of animations, transitions, and scroll jumps to 0.01ms under `@media (prefers-reduced-motion: reduce)`.
