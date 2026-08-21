# syntax=docker/dockerfile:1.7

# HawaVoClean v3.3 CPU reference image. Every base and Wolfi package is pinned;
# build evidence records the platform-specific manifests BuildKit resolves.
FROM ghcr.io/astral-sh/uv:0.11.14@sha256:1025398289b62de8269e70c45b91ffa37c373f38118d7da036fb8bb8efc85d97 AS uv
FROM cgr.dev/chainguard/wolfi-base@sha256:bfcffaf1336b26a3fd33c8cb31a86a09324d2048420d7f49b983f323b0d33e8d AS build

COPY --from=uv /uv /bin/uv
COPY docker/wolfi-packages.lock /tmp/wolfi-packages.lock
RUN xargs apk add --no-cache < /tmp/wolfi-packages.lock
WORKDIR /app

ARG SOURCE_REVISION
ARG SOURCE_DATE_EPOCH

# Build from the exact Python lock. The project wheel is installed without a
# second resolver pass, so the runtime environment cannot drift from uv.lock.
COPY pyproject.toml uv.lock README.md THIRD_PARTY_LICENSES.md hatch_build.py ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    HAWAVOCLEAN_SOURCE_REVISION="${SOURCE_REVISION}" \
    SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH}" \
    uv build --wheel --out-dir /dist \
    && uv sync --frozen --no-dev --no-install-project \
    && uv pip install --python /app/.venv/bin/python --no-deps /dist/*.whl

FROM cgr.dev/chainguard/wolfi-base@sha256:bfcffaf1336b26a3fd33c8cb31a86a09324d2048420d7f49b983f323b0d33e8d AS runtime

# The lock includes every transitive package, not only top-level runtime names.
COPY docker/wolfi-packages.lock /tmp/wolfi-packages.lock
RUN xargs apk add --no-cache < /tmp/wolfi-packages.lock \
    && rm /tmp/wolfi-packages.lock

ARG SOURCE_REVISION
ARG SOURCE_DATE_EPOCH
ARG SOURCE_DATE
LABEL org.opencontainers.image.title="HawaVoClean CPU reference" \
      org.opencontainers.image.version="3.3.0" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.created="${SOURCE_DATE}" \
      org.opencontainers.image.source="https://github.com/hawzhin/HawaVoClean"

RUN addgroup -g 10001 -S hawavoclean \
    && adduser -S -D -u 10001 -G hawavoclean -h /home/hawavoclean hawavoclean \
    && install -d -o hawavoclean -g hawavoclean -m 0750 /work /cache/work

COPY --from=build --chown=root:root /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    HOME="/home/hawavoclean" \
    HAWAVOCLEAN_WORK_DIR="/cache/work" \
    HAWAVOCLEAN_DEVICE="cpu" \
    PYTHONDONTWRITEBYTECODE="1" \
    PYTHONHASHSEED="0" \
    OMP_NUM_THREADS="1" \
    MKL_NUM_THREADS="1"

USER 10001:10001
WORKDIR /work

HEALTHCHECK --interval=30s --timeout=20s --start-period=5s --retries=3 \
    CMD ["hawavoclean", "doctor"]

ENTRYPOINT ["hawavoclean"]
CMD ["doctor"]
