#!/usr/bin/env bash
set -euo pipefail

echo "================================================================================"
echo "                  HAWZHIN VOICECLEAN - RELEASE GATES SUITE                      "
echo "================================================================================"

echo "[1/5] Formatting check..."
uv run ruff format --check .

echo "[2/5] Linting check..."
uv run ruff check .

echo "[3/5] Type safety check (Mypy strict)..."
uv run mypy --strict src tests scripts data

echo "[4/5] Running tests with branch coverage..."
uv run pytest -q --cov=voiceclean --cov-branch --cov-report=term-missing --cov-fail-under=90

echo "[5/5] Doctor preflight..."
uv run voiceclean doctor

echo "================================================================================"
echo "ALL RELEASE CHECKS PASSED SUCCESSFULLY."
echo "================================================================================"
