# Security policy

## Reporting a vulnerability

Use the private contact channel provided by the repository/distribution owner and include the affected
version, reproduction steps, impact, and the smallest non-sensitive fixture possible. Do not publish a
working exploit before a fix is available, and do not send private dialogue audio, consent records,
credentials, or loopback tokens. This source snapshot does not claim a monitored security inbox or a
response-time SLA.

## Security boundary

HawaVoClean processes hostile local media and exposes an optional authenticated loopback service. It
does not claim to sandbox FFmpeg, libsndfile, PyTorch, DeepFilterNet, Electron, or DaVinci Resolve from
the operating system. Run it as a non-administrator and keep those prerequisites patched.

The supported processing path is offline-first: audio processing performs no telemetry, cloud API
call, remote model lookup, or dynamic weight download. `hawavoclean serve` binds loopback only and is
the sole network-facing runtime surface. Build, audit, and dependency-install commands do use package
registries/advisory services and are outside that offline processing claim.

## Enforced controls

### Media, processes, and resources

- Inputs are never opened for writing. Probe/decode commands use fixed argument arrays, no shell
  interpretation, closed stdin, and timeouts/resource checks.
- Enhancement executes in disposable child processes with deadlines, memory/device constraints,
  parent-death watchdogs, and request/reply identity. Crash, hang, invalid length, NaN, silence, or
  uncertain guard state falls back to the original unit.
- Scratch and publication bundles are mode `0700` where created. Successful output is committed as an
  immutable WAV/report/summary generation through one verified pointer; unexpected symlinks,
  ownership markers, or mixed legacy files are refused.
- The CPU container runs as UID/GID 10001 and is qualified with a read-only root plus explicit bounded
  tmpfs/work mounts. Windows and GPU/studio containers are not security-qualified for v3.3.

### Models and supply chain

- Production has no learned weights. Studio/lowband use the one vendored DeepFilterNet3 config and
  checkpoint named in the core locks; processing recomputes their SHA-256 values before loading.
- The vendored upstream artifact is `model_120.ckpt.best` and is deserialized by DeepFilterNet 0.5.6.
  It is not `safetensors`, and HawaVoClean does not claim `weights_only=True`. The control is stricter
  trust scope: no user-supplied checkpoint path, no runtime download, exact committed hash, isolated
  worker, and release dependency/audit gates. A checkpoint-hash change requires deliberate review and
  a new lock.
- Exact Python, JavaScript, build-tool, container and model locks are audited. Release artifacts carry
  source/build identity, hashes and a deterministic CycloneDX 1.6 SBOM. A final release must rebuild
  and rescan the exact eventual commit; historical scans are not silently current.

### Loopback API and uploads

- `hawavoclean serve` accepts only loopback addresses and requires a non-empty per-launch token on
  every `/api` request. The Resolve shell generates a fresh random token and never exposes raw IPC.
- Client paths must be absolute and resolve under the user's home, `/Volumes`, or HawaVoClean's work
  root. NUL/unencodable names, escapes, and unauthorized output parents fail closed.
- Active jobs, terminal history, concurrent uploads, one-upload bytes, total upload bytes, retention
  age, and minimum free space are finite. Bodies stream to disk rather than memory. Cleanup is limited
  to marker-scoped managed uploads and never deletes committed outputs.
- API errors have bounded structured messages. Logs can still contain operator paths and dependency
  stderr; treat logs as sensitive operational metadata.

### Electron and Resolve

- The renderer loads checksum-covered local content from a private non-persistent `hawa://app`
  session. Sandbox/context isolation are enabled; Node integration, webviews, foreign navigation,
  popups, unexpected permissions, arbitrary network requests, and unvalidated IPC senders are denied.
- The controlled standalone shell is exact Electron 43.4.1 and is lock-audited. DaVinci Resolve
  21.0.3 embeds vendor-owned Electron 36.3.2 with known high-severity advisories. Application controls
  reduce reachable configurations but cannot patch that binary. This remains an explicit release
  blocker documented in [the runtime-risk assessment](docs/resolve-runtime-risk.md).

## Privacy and data retention

Ordinary processing reports contain technical metadata, source/output paths, hashes, measurements,
decisions, and review timecodes. They do not transcribe dialogue or create speaker embeddings. Corpus
evaluation manifests are a separate workflow and can contain pseudonymous speaker IDs and Sorani
transcripts; keep those manifests and all raw/consent material outside public Git according to the
approved evaluation protocol.

Uploaded inputs are local temporary data, deleted when their job reaches a terminal state or after the
bounded retention interval. A process crash can leave scratch material for recovery/forensics until
startup cleanup or operator review. Committed masters are user data and are never retention-cleaned.

## Known limits

- Signal guards detect timing/spectral/integrity failures, not linguistic meaning. Human Sorani
  review is a separate, currently open release gate.
- Hash verification establishes artifact identity, not that upstream checkpoint serialization is
  intrinsically safe or that dependencies contain no unknown vulnerability.
- Path policy protects the loopback API, but a process running as the user can access files that user
  can access. OS account isolation and file permissions remain part of the security boundary.
- The project has not completed external penetration testing, real in-Resolve security acceptance, or
  the final protected-CI/release-signing gate.
