# Deploying HawaVoClean Studio 4.0 on the Web

HawaVoClean ships with a hardened **`--web-public`** mode designed for hosting the Studio 4.0 browser interface behind your website's HTTPS reverse proxy and authentication gateway.

## What `--web-public` Locks Down

When started with `--web-public` (or `HAWAVOCLEAN_WEB_PUBLIC=1`), the FastAPI server enforces strict server-side isolation:

1. **Upload-Only Filesystem Boundary**: `allowed_roots()` is reduced exclusively to `HAWAVOCLEAN_WORK_DIR` (uploaded clips, previews, and outputs). Any request referencing host home directories, `/Volumes`, Windows drives, or arbitrary filesystem paths fails closed with HTTP `403`.
2. **Native Source Registration Disabled**: `POST /api/v1/sources/register` returns HTTP `403`.
3. **Model Installer Admin Route Disabled**: `POST /api/v1/models/{model_id}/install` returns HTTP `403` so web visitors cannot trigger server-side downloads. Pre-bake optional weights into `/var/lib/hawavoclean/data/models` in your container image.
4. **Explicit HTTPS Origin Allowlist**: Pass `--cors-origins https://studio.example.com` (or `HAWAVOCLEAN_CORS_ORIGINS`) when embedding or serving from a separate origin. Wildcards (`*`) and `null` origins are rejected.
5. **Bounded Disk & Upload Quotas**: `UploadStore` enforces per-file max size, total directory byte budget, TTL expiration, and emergency free-space reserves.

## Quick Start with Docker Compose

```bash
export HAWAVOCLEAN_WEB_TOKEN="$(openssl rand -hex 32)"
docker compose -f deploy/docker-compose.yml up --build -d
```

Then open `http://localhost:8080/api/health` and verify `"deployment_mode": "web_public"`.

## Integrating into Your Website

1. **Authentication Gateway (Required for Multi-User Sites)**: Place your existing auth middleware (Clerk, Auth0, Supabase, or Nginx `auth_request`) in front of `/api/session` so only authenticated users receive a short-lived session bearer token.
2. **Subdomain or Subpath**: Serve on `https://studio.yourdomain.com` for full-bleed Studio 4.0 layout and zero CSS conflicts.
3. **Iframe Embed**: If embedding inside a page, grant microphone permission for the 5-second Mic Test:
   ```html
   <iframe
     src="https://studio.yourdomain.com"
     allow="microphone"
     style="width:100%;height:860px;border:0;border-radius:12px;"
   ></iframe>
   ```
4. **GPU Acceleration**: On Linux servers with NVIDIA GPUs, install CUDA-enabled PyTorch in the runtime stage and launch Docker with `--gpus all` for 15–30× real-time processing.
5. **Local LLM Chat on the Server**: Point `HAWAVOCLEAN_OLLAMA_URL` to a loopback Ollama instance (`127.0.0.1:11434`) with `qwen2.5:7b` or `qwen3:8b` pulled to give every web user full open-ended conversational audio control alongside the built-in deterministic studio commands.
