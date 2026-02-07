# Data policy (important)

This repository is intended to be publishable without sharing private EEG data.

## Recommended approach
- Keep raw EEG, participant identifiers, and sensitive metadata **outside** this repository.
- Provide a **data contract** (expected folder layout, file formats) here.
- Provide **synthetic demo outputs** that prove the pipeline wiring and the reporting work.

## Minimal data contract (example)
```
PROJECT_ROOT/
  derivatives/
  moduleB_outputs/
    figs/
    tables/
    metrics.json   # optional but recommended
```
