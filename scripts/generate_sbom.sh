#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $0 --image IMAGE --artifact NAME=PATH [--artifact NAME=PATH ...] --output FILE" >&2
    exit 2
fi

exec uv run python scripts/generate_sbom.py "$@"
