#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${1:-}"
REPO_ROOT="${2:-.}"
if [ -z "${ROOT_DIR}" ]; then
  echo "Usage: ./scripts/run_module_d.sh /path/to/EEG_48sounds [repo_root]"
  exit 1
fi
pip install -e ".[analysis,notebook]" > /dev/null
eeg48 module-d --root-dir "${ROOT_DIR}" --repo-root "${REPO_ROOT}"
echo "[OK] Open Results Card: ${ROOT_DIR}/moduleD_outputs/report/results_card.html"
