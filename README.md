# EEG48 (public) — Reproducible EEG + Audio Affect Pipeline

A public-ready research codebase for an EEG × audio affect study workflow (ERP/TFR features, LOSO CV, permutation tests),
packaged as **one-command pipelines** with **single-page HTML reports**.

> 🇯🇵 Note: This repository does **not** include private EEG data.  
> A **synthetic demo** is included so anyone can verify the full plumbing end-to-end.

---

## What you get (why this repo is strong)

- **One-command execution**: run modules end-to-end with `eeg48 run-all`
- **Auto reporting**: each run produces `metrics.json` + `report/results_card.html` (one-page HTML)
- **Onboardable**: a newcomer can run the synthetic demo in minutes
- **Publish-safe**: includes `eeg48 privacy-scan` to detect common private identifiers
- **Traceable**: legacy scripts are preserved (for auditability) while public entrypoints standardize execution

---

## Quickstart (no private data)

```bash
pip install -e .
eeg48 make-synth --out examples/synthetic_project
eeg48 results-card --run-dir examples/synthetic_project/moduleB_outputs
open examples/synthetic_project/moduleB_outputs/report/results_card.html
