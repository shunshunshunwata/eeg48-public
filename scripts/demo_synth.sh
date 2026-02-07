#!/usr/bin/env bash
set -euo pipefail
pip install -e . > /dev/null
eeg48 make-synth --out examples/synthetic_project
eeg48 results-card --run-dir examples/synthetic_project/moduleB_outputs
echo "[OK] Open: examples/synthetic_project/moduleB_outputs/report/results_card.html"
