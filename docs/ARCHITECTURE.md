# Architecture (public-ready)

## Philosophy
- Keep **legacy scripts intact** (traceability).
- Add thin wrappers that expose **stable CLI entrypoints**.
- Standardize outputs (`metrics.json`) and provide a one-page review artifact (Results Card).

## Components
- `src/eeg48/legacy/`: original scripts from the archive (no refactor here)
- `src/eeg48/pipelines/`: wrappers (Module B today; extend to C/D/A)
- `src/eeg48/report.py`: Results Card generator (HTML + artifacts index)
- `src/eeg48/synth.py`: synthetic demo generator (no private data)
