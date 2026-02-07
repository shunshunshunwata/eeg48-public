from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _make_metrics_from_outputs(out_dir: Path) -> Dict[str, Any]:
    tab = out_dir / "tables"
    fig = out_dir / "figs"
    log = out_dir / "logs"
    rep = out_dir / "report"

    metrics: Dict[str, Any] = {
        "module": "D",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "out_dir": str(out_dir),
        "keynumbers": {},
        "notes": [
            "Module D extracts ambiguous sounds and quantifies individual differences (PC2), then tests association with EEG features.",
            "metrics.json is derived from moduleD_KEYNUMBERS.json when available.",
        ],
    }

    key_path = log / "moduleD_KEYNUMBERS.json"
    if key_path.exists():
        try:
            metrics["keynumbers"] = json.loads(key_path.read_text(encoding="utf-8"))
        except Exception:
            metrics["keynumbers"] = {}

    # lightweight file index
    def list_files(p: Path, exts: tuple[str, ...], limit: int = 50) -> List[str]:
        if not p.exists():
            return []
        files = [str(x.relative_to(out_dir)) for x in sorted(p.rglob("*")) if x.is_file() and x.suffix.lower() in exts]
        return files[:limit]

    metrics["files"] = {
        "tables": list_files(tab, (".csv",)),
        "figs": list_files(fig, (".png",)),
        "reports": list_files(rep, (".md", ".html")),
        "logs": list_files(log, (".log", ".json")),
    }
    return metrics


def run_module_d(
    root_dir: Path,
    out_dir: Optional[Path] = None,
    trial_csv: Optional[str] = None,
    subject_csv: Optional[str] = None,
    pc2_col: str = "PC2_emotion",
    enough_ratio: float = 0.80,
    neutral_th: float = 0.35,
    top_k: int = 10,
    n_perm: int = 5000,
    n_boot: int = 5000,
    seed: int = 42,
    make_figures: bool = True,
    minimal_load: bool = True,
    write_results_card: bool = True,
    overwrite_metrics: bool = True,
) -> Path:
    root_dir = root_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve() if out_dir else (root_dir / "moduleD_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    from ..legacy import moduleD_ambiguous_individual_diff as legacy

    argv: List[str] = ["--root-dir", str(root_dir), "--out-dir", str(out_dir)]
    if trial_csv:
        argv += ["--trial-csv", trial_csv]
    if subject_csv:
        argv += ["--subject-csv", subject_csv]

    argv += [
        "--pc2-col", pc2_col,
        "--enough-ratio", str(float(enough_ratio)),
        "--neutral-th", str(float(neutral_th)),
        "--top-k", str(int(top_k)),
        "--n-perm", str(int(n_perm)),
        "--n-boot", str(int(n_boot)),
        "--seed", str(int(seed)),
    ]
    if not minimal_load:
        argv.append("--no-minimal-load")
    if not make_figures:
        argv.append("--no-fig")

    legacy.cli_main(argv)

    metrics = _make_metrics_from_outputs(out_dir)
    metrics_path = out_dir / "metrics.json"
    if overwrite_metrics or (not metrics_path.exists()):
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if write_results_card:
        from ..report import generate_results_card
        generate_results_card(run_dir=out_dir, out_dir=out_dir / "report", title="EEG48 Module D — Results Card")

    return out_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eeg48 module-d",
        description="Run Module D (scriptified) and generate metrics + Results Card."
    )
    p.add_argument("--root-dir", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--trial-csv", default=None)
    p.add_argument("--subject-csv", default=None)

    p.add_argument("--pc2-col", default="PC2_emotion")
    p.add_argument("--enough-ratio", type=float, default=0.80)
    p.add_argument("--neutral-th", type=float, default=0.35)
    p.add_argument("--top-k", type=int, default=10)

    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--no-minimal-load", action="store_true")
    p.add_argument("--no-fig", action="store_true")
    p.add_argument("--no-results-card", action="store_true")
    p.add_argument("--no-overwrite-metrics", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    out = run_module_d(
        root_dir=Path(args.root_dir),
        out_dir=(Path(args.out_dir) if args.out_dir else None),
        trial_csv=args.trial_csv,
        subject_csv=args.subject_csv,
        pc2_col=args.pc2_col,
        enough_ratio=args.enough_ratio,
        neutral_th=args.neutral_th,
        top_k=args.top_k,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        seed=args.seed,
        make_figures=(not args.no_fig),
        minimal_load=(not args.no_minimal_load),
        write_results_card=(not args.no_results_card),
        overwrite_metrics=(not args.no_overwrite_metrics),
    )
    print(f"[OK] Module D outputs: {out}")
    print(f"[OK] Metrics: {out / 'metrics.json'}")
    if not args.no_results_card:
        print(f"[OK] Results Card: {out / 'report' / 'results_card.html'}")
    return 0
