#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
exec ./.venv/bin/python scripts/run_local_audio_app.py
