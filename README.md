# EEG48 — Public‑ready Research Codebase

This repository is a **public‑ready** packaging of an EEG + audio emotion research workflow.

Design goals:
- **Reproducible**: clear run directories, deterministic artifacts where possible
- **Onboardable**: a newcomer can run a demo in minutes
- **Publishable**: no private EEG data required to validate the plumbing

---

## Quickstart (no private data)

```bash
pip install -e .

# 1) Create synthetic demo outputs
eeg48 make-synth --out examples/synthetic_project

# 2) Generate a 1‑page Results Card (HTML)
eeg48 results-card --run-dir examples/synthetic_project/moduleB_outputs

# Open:
# examples/synthetic_project/moduleB_outputs/report/results_card.html
```

---

## Results Card

A single HTML page that aggregates:
- Key metrics from `metrics.json` (optional)
- Figures (first 24)
- Tables and other artifacts
- A complete file index (`artifacts_index.csv`)

Generate:

```bash
eeg48 results-card --run-dir /path/to/run_dir
```

---

## Repo structure

- `src/eeg48/` — reusable utilities (CLI, reporting, synthetic demo)
- `src/eeg48/legacy/` — original scripts copied from the archive (kept intact)
- `notebooks/` — cleaned notebook names + originals in `notebooks/_original`
- `archive_raw/` — extracted archive (traceability)
- `examples/` — demo project
- `tests/` — lightweight tests for reporting
- `.github/workflows/` — CI skeleton

---

## Data policy

This repository is intended to be publishable without sharing private EEG data.
See: `data/README.md`

---

## License / citation

- License: MIT (see `LICENSE`)
- Citation: `CITATION.cff`


---

## Run Module B on your local data (real pipeline)

Install analysis dependencies:

```bash
pip install -e ".[analysis]"
```

Run Module B and auto-generate `metrics.json` + Results Card:

```bash
eeg48 module-b --root-dir /path/to/EEG_48sounds
# -> /path/to/EEG_48sounds/moduleB_outputs/metrics.json
# -> /path/to/EEG_48sounds/moduleB_outputs/report/results_card.html
```


### Passing legacy script flags (advanced)

You can pass any additional flags supported by the legacy Module B script by appending them after `--`:

```bash
eeg48 module-b --root-dir /path/to/EEG_48sounds -- --tasks emo_arousal,emo_valence --n-perm 200 --n-jobs -1
```


---

## Privacy scan (recommended before publishing)

Scan the repo for common private identifiers (absolute user paths, emails):

```bash
eeg48 privacy-scan --root .
# or write JSON:
eeg48 privacy-scan --root . --out report/privacy_scan.json
```


---

## Module A/C/D entrypoints (public-friendly)

### Module A (extras): figures + threshold table
```bash
pip install -e ".[analysis]"
eeg48 module-a --root-dir /path/to/EEG_48sounds
```

### Module C (scriptified pipeline)
```bash
pip install -e ".[analysis]"
eeg48 module-c --root-dir /path/to/EEG_48sounds
```

### Module D (scriptified pipeline)
```bash
pip install -e ".[analysis]"
eeg48 module-d --root-dir /path/to/EEG_48sounds
```

### Generic notebook runner
```bash
pip install -e ".[notebook]"
eeg48 notebook-run --notebook notebooks/99_misc__moduleC_prime_full_script.ipynb --out-dir runs/tmp_moduleC --timeout 1200
```


---

## Run everything (A/B/C/D) and generate a master report

```bash
pip install -e ".[analysis]"
eeg48 run-all --root-dir /path/to/EEG_48sounds
# open: /path/to/EEG_48sounds/eeg48_reports/index.html
```

Pass Module B legacy flags via `--` (advanced):

```bash
eeg48 run-all --root-dir /path/to/EEG_48sounds --modules B,C,D,A -- --tasks emo_arousal,emo_valence --n-perm 200
```

See also: `docs/FIGURE_INDEX.md`
