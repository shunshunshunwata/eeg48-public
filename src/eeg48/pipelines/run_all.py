from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional


def _parse_modules(s: str) -> List[str]:
    items = [x.strip().upper() for x in (s or "").split(",") if x.strip()]
    # allow full names too
    norm: List[str] = []
    for x in items:
        if x in {"A", "B", "C", "D"}:
            norm.append(x)
        elif x in {"MODULE-A", "MODULE_A"}:
            norm.append("A")
        elif x in {"MODULE-B", "MODULE_B"}:
            norm.append("B")
        elif x in {"MODULE-C", "MODULE_C"}:
            norm.append("C")
        elif x in {"MODULE-D", "MODULE_D"}:
            norm.append("D")

    # de-dup while preserving order
    out: List[str] = []
    for m in norm:
        if m not in out:
            out.append(m)
    return out


def run_all(
    root_dir: Path,
    modules: List[str],
    b_legacy_args: Optional[List[str]] = None,
    write_results_cards: bool = True,
    overwrite_metrics: bool = True,
    out_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Run selected modules and return mapping module->run_dir."""
    root_dir = root_dir.expanduser().resolve()
    module_run_dirs: Dict[str, Path] = {}

    # Execute in the provided order (important for reproducibility narratives)
    for m in modules:
        if m == "A":
            from .module_a import run_module_a_extras

            module_run_dirs["A"] = run_module_a_extras(
                root_dir=root_dir,
                write_results_card=write_results_cards,
                overwrite_metrics=overwrite_metrics,
            )

        elif m == "B":
            from .module_b import run_module_b

            module_run_dirs["B"] = run_module_b(
                root_dir=root_dir,
                passthrough_args=b_legacy_args or [],
                write_results_card=write_results_cards,
                overwrite_metrics=overwrite_metrics,
            )

        elif m == "C":
            from .module_c import run_module_c

            module_run_dirs["C"] = run_module_c(
                root_dir=root_dir,
                out_dir=None,
                write_results_card=write_results_cards,
                overwrite_metrics=overwrite_metrics,
            )

        elif m == "D":
            from .module_d import run_module_d

            module_run_dirs["D"] = run_module_d(
                root_dir=root_dir,
                out_dir=None,
                write_results_card=write_results_cards,
                overwrite_metrics=overwrite_metrics,
            )

    # Master report
    from ..master_report import generate_master_report

    generate_master_report(
        root_dir=root_dir,
        module_run_dirs=module_run_dirs,
        out_dir=(out_dir.expanduser().resolve() if out_dir else None),
        title="EEG48 — Master Report",
    )

    return module_run_dirs


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eeg48 run-all",
        description="Run selected modules and generate a master report linking all Results Cards.",
    )
    p.add_argument("--root-dir", required=True, help="Project root directory (contains derivatives/).")
    p.add_argument("--modules", default="A,B,C,D", help="Comma-separated modules to run, e.g., A,B or B,C,D,A.")
    p.add_argument("--out-dir", default=None, help="Master report output directory (default: <root>/eeg48_reports).")
    p.add_argument("--no-results-cards", action="store_true", help="Do not generate module Results Cards.")
    p.add_argument("--no-overwrite-metrics", action="store_true", help="Do not overwrite existing metrics.json.")
    p.add_argument(
        "b_legacy_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to Module B legacy script (use `--` before them).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    modules = _parse_modules(args.modules) or ["A", "B", "C", "D"]

    run_all(
        root_dir=Path(args.root_dir),
        modules=modules,
        b_legacy_args=(args.b_legacy_args or []),
        write_results_cards=(not args.no_results_cards),
        overwrite_metrics=(not args.no_overwrite_metrics),
        out_dir=(Path(args.out_dir) if args.out_dir else None),
    )

    root = Path(args.root_dir).expanduser().resolve()
    report_dir = (Path(args.out_dir).expanduser().resolve() if args.out_dir else (root / "eeg48_reports"))
    print(f"[OK] Master report: {report_dir / 'index.html'}")
    return 0
