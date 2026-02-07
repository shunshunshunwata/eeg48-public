#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-}"
if [ -z "${ROOT_DIR}" ]; then
  echo "Usage: ./scripts/run_all.sh /path/to/EEG_48sounds"
  exit 1
fi

pip install -e ".[analysis]" > /dev/null
eeg48 run-all --root-dir "${ROOT_DIR}"

echo "[OK] Master report: ${ROOT_DIR}/eeg48_reports/index.html"
