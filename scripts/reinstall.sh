#!/usr/bin/env bash
set -euo pipefail

# Reinstall the package locally in editable mode for quick testing.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m pip uninstall -y tmux-llm >/dev/null 2>&1 || true
rm -rf build dist tmux_llm.egg-info
python3 -m pip install -e .

echo "tmux-llm reinstalled in editable mode."
