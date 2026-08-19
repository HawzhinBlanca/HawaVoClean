# Pinned Reference Container for HawaVoClean v1
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04@sha256:39a5a7dc50b7ec1287c7118335b71db4bc699cf3a60e0a5c2d3a3d24ea25d7ef

# Install system dependencies & ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pinned uv
COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /bin/uv

WORKDIR /app

# Copy dependency definitions and lockfile
COPY pyproject.toml uv.lock ./

# Install locked dependencies
RUN uv sync --frozen --no-cache

# Copy application source
COPY . .

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

ENTRYPOINT ["hawavoclean"]
CMD ["doctor"]
