# syntax=docker/dockerfile:1.7

# HawaVoClean v3.3 CPU reference image. The multi-platform indexes are pinned;
# build evidence records the platform-specific manifests BuildKit resolves.
FROM ghcr.io/astral-sh/uv:0.11.14@sha256:1025398289b62de8269e70c45b91ffa37c373f38118d7da036fb8bb8efc85d97 AS uv
FROM python:3.12-slim-bookworm@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134 AS build

COPY --from=uv /uv /bin/uv
WORKDIR /app

# Build from the exact Python lock. The project wheel is installed without a
# second resolver pass, so the runtime environment cannot drift from uv.lock.
COPY pyproject.toml uv.lock README.md THIRD_PARTY_LICENSES.md hatch_build.py ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel --out-dir /dist \
    && uv sync --frozen --no-dev --no-install-project \
    && uv pip install --python /app/.venv/bin/python --no-deps /dist/*.whl

FROM python:3.12-slim-bookworm@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134 AS runtime

# Exact versions from the pinned Bookworm snapshot visible to the base image.
# A missing version is a hard build failure instead of a silent package drift.
COPY docker/debian.sources /etc/apt/sources.list.d/debian.sources
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg=7:5.1.9-0+deb12u1 \
        libsndfile1=1.2.0-1+deb12u1 \
    && rm -rf /var/lib/apt/lists/*

ARG SOURCE_REVISION
ARG SOURCE_DATE_EPOCH
ARG SOURCE_DATE
LABEL org.opencontainers.image.title="HawaVoClean CPU reference" \
      org.opencontainers.image.version="3.3.0" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.created="${SOURCE_DATE}" \
      org.opencontainers.image.source="https://github.com/hawzhin/HawaVoClean"

RUN groupadd --gid 10001 hawavoclean \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/hawavoclean hawavoclean \
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
