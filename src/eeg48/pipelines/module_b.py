from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _require_analysis_stack() -> None:
    # Import lazily so lightweight commands (results-card) work without heavy deps.
    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import mne  # noqa: F401
        import sklearn  # noqa: F401
        import joblib  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing analysis dependencies.\n"
            "Install with: pip install -e '.[analysis]'\n"
            f"Original error: {type(e).__name__}: {e}"
        )


def _read_moduleb_summaries(tables_dir: Path) -> Dict[str, Any]:
    import pandas as pd  # type: ignore

    tasks: Dict[str, Any] = {}
    for p in sorted(tables_dir.glob("moduleB_summary_*_linear.csv")):
        try:
            df = pd.read_csv(p)
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            task = str(row.get("task", p.stem))
            tasks[task] = {
                "kind": row.get("kind"),
                "model": row.get("model"),
                "score_mean": float(row.get("score_mean")) if row.get("score_mean") is not None else None,
                "p_perm": float(row.get("p_perm")) if row.get("p_perm") is not None else None,
                "n_perm": int(row.get("n_perm")) if row.get("n_perm") is not None else None,
                "n_trials": int(row.get("n_trials")) if row.get("n_trials") is not None else None,
                "n_subjects": int(row.get("n_subjects")) if row.get("n_subjects") is not None else None,
                "n_features": int(row.get("n_features")) if row.get("n_features") is not None else None,
                "apply_qc": bool(row.get("apply_qc")) if row.get("apply_qc") is not None else None,
                "perm_within_subject": bool(row.get("perm_within_subject")) if row.get("perm_within_subject") is not None else None,
                "with_maps": bool(row.get("with_maps")) if row.get("with_maps") is not None else None,
                "source": str(p.name),
            }
        except Exception:
            continue
    return tasks


def _aggregate_metrics(tasks: Dict[str, Any]) -> Dict[str, Any]:
    scores = [(k, v.get("score_mean")) for k, v in tasks.items() if isinstance(v.get("score_mean"), (int, float))]
    scores_clean = [(k, float(s)) for k, s in scores if s is not None]
    out: Dict[str, Any] = {}
    if scores_clean:
        vals = [s for _, s in scores_clean]
        out["mean_score"] = float(sum(vals) / len(vals))
        best = max(scores_clean, key=lambda x: x[1])
        worst = min(scores_clean, key=lambda x: x[1])
        out["best_task"] = {"task": best[0], "score_mean": best[1]}
        out["worst_task"] = {"task": worst[0], "score_mean": worst[1]}
        out["n_tasks_scored"] = len(vals)
    else:
        out["mean_score"] = None
        out["best_task"] = None
        out["worst_task"] = None
        out["n_tasks_scored"] = 0
    return out


def run_module_b(
    root_dir: Path,
    passthrough_args: Optional[List[str]] = None,
    write_results_card: bool = True,
    overwrite_metrics: bool = True,
) -> Path:
    """Run legacy Module B and standardize outputs.

    Writes:
      - <root>/moduleB_outputs/metrics.json
      - <root>/moduleB_outputs/report/results_card.html (optional)

    Returns:
      run_dir (= <root>/moduleB_outputs)
    """
    _require_analysis_stack()

    root_dir = root_dir.expanduser().resolve()
    passthrough_args = passthrough_args or []

    # Import legacy only after deps are present
    from ..legacy import moduleB_phaseB_EEG_only as legacy_b  # type: ignore

    argv = ["--root-dir", str(root_dir)] + passthrough_args
    legacy_b.main(argv)

    run_dir = root_dir / "moduleB_outputs"
    tables_dir = run_dir / "tables"

    tasks = _read_moduleb_summaries(tables_dir)

    metrics: Dict[str, Any] = {
        "module": "B",
        "primary_model": "linear",
        "cv": "LOSO",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "tasks": tasks,
    }
    metrics.update(_aggregate_metrics(tasks))

    metrics_path = run_dir / "metrics.json"
    if overwrite_metrics or (not metrics_path.exists()):
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if write_results_card:
        from ..report import generate_results_card
        generate_results_card(run_dir=run_dir, out_dir=run_dir / "report", title="EEG48 Module B — Results Card")

    return run_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eeg48 module-b",
        description="Run Module B (legacy pipeline) and generate metrics.json + Results Card."
    )
    p.add_argument("--root-dir", required=True, help="Project root directory (contains derivatives/, etc.).")
    p.add_argument("--no-results-card", action="store_true", help="Do not generate Results Card HTML.")
    p.add_argument("--no-overwrite-metrics", action="store_true", help="Do not overwrite existing metrics.json.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argparser()
    args, unknown = parser.parse_known_args(argv)

    run_dir = run_module_b(
        root_dir=Path(args.root_dir),
        passthrough_args=unknown,
        write_results_card=(not args.no_results_card),
        overwrite_metrics=(not args.no_overwrite_metrics),
    )
    print(f"[OK] Module B run dir: {run_dir}")
    print(f"[OK] Metrics: {run_dir / 'metrics.json'}")
    if not args.no_results_card:
        print(f"[OK] Results Card: {run_dir / 'report' / 'results_card.html'}")
    return 0
