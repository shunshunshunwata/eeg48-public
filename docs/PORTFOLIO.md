# How to showcase EEG48 (portfolio / hiring)

This repository is designed to be **public-friendly** even when your real EEG data cannot be shared.

## What a reviewer can verify in 3 minutes
1. **It installs cleanly** (`pip install -e .`)
2. **A demo run produces artifacts** (synthetic project)
3. **Artifacts are reviewable in one page** (Results Card HTML)
4. **The real pipeline has a stable entrypoint** (`eeg48 module-b ...`)
5. **Core research practices are present** (LOSO CV, permutation test, interpretable linear model)

## Suggested GitHub README highlights (copy-paste ideas)
- Reproducible research utilities for EEG + audio emotion tasks
- LOSO cross-validation + permutation tests (primary evaluation)
- Outputs standardized via `metrics.json` and a one-page HTML Results Card
- Synthetic demo to validate the pipeline wiring without private data

## Resume bullets (example)
- Built a reproducible EEG ML pipeline (ERP/TFR feature engineering) with LOSO cross-validation and permutation testing for robust evaluation.
- Packaged research code into an installable Python project with CLI entrypoints, synthetic demo outputs, and automated HTML reporting for third-party review.
- Implemented privacy-safe publication workflow (private-data exclusion + repository privacy scan).

## What to attach in applications
- Link to the repo
- 1 screenshot of `results_card.html` (demo or real)
- A short note: “Real data is private; demo shows full reproducibility of reporting and execution.”
