#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${1:-}"
if [ -z "${ROOT_DIR}" ]; then
  echo "Usage: ./scripts/run_module_a.sh /path/to/EEG_48sounds"
  exit 1
fi
pip install -e ".[analysis]" > /dev/null
eeg48 module-a --root-dir "${ROOT_DIR}"
echo "[OK] Open Results Card: ${ROOT_DIR}/moduleA_outputs/report/results_card.html"
