# Security Policy

## Reporting Security Issues

Please report security vulnerabilities by emailing `security@hawzhin.ai`. We will acknowledge receipt within 24 hours and provide an evaluation within 72 hours.

## Security Controls & Invariants

1. **Subprocess Isolation**: Subprocess calls (FFmpeg, FFprobe, isolated model workers) are executed strictly via argument lists without shell interpretation (`shell=False`).
2. **Untrusted Checkpoint Policy**: Checkpoints are loaded using safe formats (`safetensors` preferred, or hash-verified `.pt` files loaded strictly with `weights_only=True`).
3. **No Network Operations at Runtime**: The production engine operates strictly offline. No telemetry, dynamic weight downloads, or cloud API calls are made during audio processing.
4. **Filesystem Safety**: Private workspaces enforce restricted permissions (0700), sanitize input filenames to prevent path traversal attacks, and never modify the input file.
5. **Privacy by Default**: Dialogue transcripts and speaker embeddings are excluded from audit reports by default.
