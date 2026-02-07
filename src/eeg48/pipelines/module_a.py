from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _auto_find_sound_level(root_dir: Path) -> Optional[Path]:
    candidates = [
        root_dir / "derivatives" / "master_tables" / "master_sound_level_with_PC.csv",
        root_dir / "derivatives" / "master_tables" / "master_participant_sound_level_with_PC.csv",
        root_dir / "master_sound_level_with_PC.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    hits = list(root_dir.rglob("master_sound_level_with_PC*.csv"))
    return hits[0] if hits else None


def run_module_a_extras(
    root_dir: Path,
    in_sound_level: Optional[Path] = None,
    in_corr: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    write_results_card: bool = True,
    overwrite_metrics: bool = True,
) -> Path:
    root_dir = root_dir.expanduser().resolve()
    in_sound_level = (in_sound_level.expanduser().resolve() if in_sound_level else _auto_find_sound_level(root_dir))
    if in_sound_level is None or not in_sound_level.exists():
        raise FileNotFoundError(
            "Could not find master_sound_level_with_PC.csv.\n"
            "Provide it with --in-sound-level or place it under:\n"
            "  <root>/derivatives/master_tables/master_sound_level_with_PC.csv"
        )

    out_dir = (out_dir.expanduser().resolve() if out_dir else (root_dir / "moduleA_outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    from ..legacy import make_moduleA_extras as legacy  # type: ignore

    argv = [
        "make_moduleA_extras.py",
        "--in-sound-level", str(in_sound_level),
        "--out-dir", str(out_dir),
    ]
    if in_corr:
        argv += ["--in-corr", str(in_corr)]

    old_argv = sys.argv
    try:
        sys.argv = argv
        legacy.main()
    finally:
        sys.argv = old_argv

    metrics: Dict[str, Any] = {
        "module": "A",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "root_dir": str(root_dir),
        "in_sound_level": str(in_sound_level),
        "out_dir": str(out_dir),
        "artifacts": {
            "corr_matrix_fig": "Fig4-1e_subjective_corr_matrix.png",
            "pc_scatter_fig": "Fig4-1f_PC1_PC2_scatter.png",
            "threshold_table": "Tbl4-1g_binary_thresholds_and_counts.csv",
        },
    }

    metrics_path = out_dir / "metrics.json"
    if overwrite_metrics or (not metrics_path.exists()):
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if write_results_card:
        from ..report import generate_results_card
        generate_results_card(run_dir=out_dir, out_dir=out_dir / "report", title="EEG48 Module A — Results Card")

    return out_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eeg48 module-a",
        description="Run Module A extras and generate metrics.json + Results Card."
    )
    p.add_argument("--root-dir", required=True, help="Project root directory.")
    p.add_argument("--in-sound-level", default=None, help="Path to master_sound_level_with_PC.csv (auto-detected if omitted).")
    p.add_argument("--in-corr", default=None, help="Optional phaseA_subjective_corr_matrix.csv")
    p.add_argument("--out-dir", default=None, help="Output directory (default: <root>/moduleA_outputs)")
    p.add_argument("--no-results-card", action="store_true", help="Do not generate Results Card HTML.")
    p.add_argument("--no-overwrite-metrics", action="store_true", help="Do not overwrite existing metrics.json.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    out_dir = run_module_a_extras(
        root_dir=Path(args.root_dir),
        in_sound_level=(Path(args.in_sound_level) if args.in_sound_level else None),
        in_corr=(Path(args.in_corr) if args.in_corr else None),
        out_dir=(Path(args.out_dir) if args.out_dir else None),
        write_results_card=(not args.no_results_card),
        overwrite_metrics=(not args.no_overwrite_metrics),
    )
    print(f"[OK] Module A outputs: {out_dir}")
    print(f"[OK] Metrics: {out_dir / 'metrics.json'}")
    if not args.no_results_card:
        print(f"[OK] Results Card: {out_dir / 'report' / 'results_card.html'}")
    return 0
