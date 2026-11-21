# HawaVoClean Studio 4.0 — Production Web-Public Container (CPU / GPU-ready)
# Build:  docker build -f deploy/Dockerfile -t hawavoclean-web:3.3.0 .
# Run:    docker run -p 8000:8000 -e HAWAVOCLEAN_WEB_TOKEN=change-me hawavoclean-web:3.3.0

FROM node:22-slim AS ui-builder
WORKDIR /build/ui
RUN corepack enable
COPY ui/package.json ui/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY ui/ ./
RUN pnpm run build

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HAWAVOCLEAN_WEB_PUBLIC=1 \
    HAWAVOCLEAN_WORK_DIR=/var/lib/hawavoclean/work \
    HAWAVOCLEAN_APP_DATA_DIR=/var/lib/hawavoclean/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 hawavoclean \
    && mkdir -p /var/lib/hawavoclean/work /var/lib/hawavoclean/data /app/ui/dist \
    && chown -R hawavoclean:hawavoclean /var/lib/hawavoclean /app

WORKDIR /app
COPY pyproject.toml README.md THIRD_PARTY_LICENSES.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[ui]"

COPY --from=ui-builder /build/ui/dist /app/ui/dist
RUN chown -R hawavoclean:hawavoclean /app/ui/dist

USER hawavoclean
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

ENTRYPOINT ["/bin/sh", "-c", "printf '%s\n' \"${HAWAVOCLEAN_WEB_TOKEN:?set HAWAVOCLEAN_WEB_TOKEN}\" | exec hawavoclean serve --host 0.0.0.0 --port 8000 --ui-dir /app/ui/dist --token-stdin --web-public"]
