#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-}"
if [ -z "${ROOT_DIR}" ]; then
  echo "Usage: ./scripts/run_module_b.sh /path/to/EEG_48sounds"
  exit 1
fi

pip install -e ".[analysis]" > /dev/null
eeg48 module-b --root-dir "${ROOT_DIR}"

echo "[OK] Open Results Card: ${ROOT_DIR}/moduleB_outputs/report/results_card.html"
