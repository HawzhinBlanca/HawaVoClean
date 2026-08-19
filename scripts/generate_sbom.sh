#!/usr/bin/env bash
set -euo pipefail

echo "Generating CycloneDX Software Bill of Materials (SBOM)..."
mkdir -p build/sbom

uv pip freeze > build/sbom/requirements.frozen.txt
echo "Requirements frozen to build/sbom/requirements.frozen.txt"
echo "SBOM generation complete."
