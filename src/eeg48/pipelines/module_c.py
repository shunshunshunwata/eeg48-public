from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def _make_metrics_from_outputs(out_dir: Path) -> Dict[str, Any]:
    tab = out_dir / "tables"
    fig = out_dir / "figs"
    log = out_dir / "logs"
    rep = out_dir / "report"

    metrics: Dict[str, Any] = {
        "module": "C",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "out_dir": str(out_dir),
        "artifacts": {
            "tables_dir": str(tab),
            "figs_dir": str(fig),
            "logs_dir": str(log),
            "report_dir": str(rep),
        },
        "targets": [],
        "encoding": {},
        "notes": [
            "Module C implements residualization-based EEG increment (LOSO) and acoustic→EEG encoding (LOO), with permutation tests.",
            "metrics.json is derived from output CSVs; if files are missing, fields may be empty.",
        ],
    }

    # Residualization summary
    summary = _read_csv_safe(tab / "moduleC_summary.csv")
    if summary is not None and len(summary) > 0:
        for _, r in summary.iterrows():
            metrics["targets"].append({
                "target": str(r.get("target")),
                "n_folds": int(r.get("n_folds", 0)) if pd.notna(r.get("n_folds", None)) else None,
                "mean_r2_acoustic": float(r.get("mean_r2_acoustic")) if pd.notna(r.get("mean_r2_acoustic", None)) else None,
                "mean_r2_acoustic_plus_bias": float(r.get("mean_r2_acoustic_plus_bias")) if pd.notna(r.get("mean_r2_acoustic_plus_bias", None)) else None,
                "mean_r2_full": float(r.get("mean_r2_full")) if pd.notna(r.get("mean_r2_full", None)) else None,
                "mean_delta_full": float(r.get("mean_delta_full")) if pd.notna(r.get("mean_delta_full", None)) else None,
                "median_delta_full": float(r.get("median_delta_full")) if pd.notna(r.get("median_delta_full", None)) else None,
                "n_delta_pos": int(r.get("n_delta_pos", 0)) if pd.notna(r.get("n_delta_pos", None)) else None,
                "n_delta_neg": int(r.get("n_delta_neg", 0)) if pd.notna(r.get("n_delta_neg", None)) else None,
                "perm_p_two_sided": float(r.get("p_two_sided")) if pd.notna(r.get("p_two_sided", None)) else None,
            })

    # Encoding summary
    enc_sum = _read_csv_safe(tab / "encoding_summary.csv")
    if enc_sum is not None and len(enc_sum) > 0:
        r = enc_sum.iloc[0].to_dict()
        # keep only serializable primitives
        metrics["encoding"] = {k: (float(v) if isinstance(v, (int, float)) and pd.notna(v) else v) for k, v in r.items()}

    # TopK table (optional)
    enc_top = _read_csv_safe(tab / "encoding_topk_with_p.csv")
    if enc_top is not None and len(enc_top) > 0:
        # store top 10
        top10 = enc_top.head(10)[["eeg_feature", "r2_loocv", "p_perm_two_sided"]].to_dict(orient="records")
        metrics["encoding"]["top10_features"] = top10

    # Quick file index (lightweight)
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


def run_module_c(
    root_dir: Path,
    out_dir: Optional[Path] = None,
    wipe_outdir: bool = True,
    moduleb_trial_csv: str = "moduleB_outputs/tables/moduleB_trial_eeg_features.csv",
    n_perm: int = 200,
    enc_n_perm: int = 300,
    enc_topk: int = 50,
    write_results_card: bool = True,
    overwrite_metrics: bool = True,
) -> Path:
    root_dir = root_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve() if out_dir else (root_dir / "moduleC_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run legacy script in-process (sys.argv compatible)
    from ..legacy import moduleC_finalize as legacy

    argv = [
        "moduleC_finalize.py",
        "--root-dir", str(root_dir),
        "--out-dir", str(out_dir),
        "--wipe_outdir", "1" if wipe_outdir else "0",
        "--moduleb_trial_csv", moduleb_trial_csv,
        "--n_perm", str(int(n_perm)),
        "--enc_n_perm", str(int(enc_n_perm)),
        "--enc_topk", str(int(enc_topk)),
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        legacy.main()
    finally:
        sys.argv = old_argv

    metrics = _make_metrics_from_outputs(out_dir)
    metrics_path = out_dir / "metrics.json"
    if overwrite_metrics or (not metrics_path.exists()):
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if write_results_card:
        from ..report import generate_results_card
        generate_results_card(run_dir=out_dir, out_dir=out_dir / "report", title="EEG48 Module C — Results Card")

    return out_dir


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eeg48 module-c",
        description="Run Module C (scriptified) and generate metrics + Results Card."
    )
    p.add_argument("--root-dir", required=True, help="Project root directory (contains derivatives/).")
    p.add_argument("--out-dir", default=None, help="Optional absolute output directory (default: <root>/moduleC_outputs)")
    p.add_argument("--no-wipe", action="store_true", help="Do not delete output directory before running.")
    p.add_argument("--moduleb-trial-csv", default="moduleB_outputs/tables/moduleB_trial_eeg_features.csv")
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--enc-n-perm", type=int, default=300)
    p.add_argument("--enc-topk", type=int, default=50)
    p.add_argument("--no-results-card", action="store_true")
    p.add_argument("--no-overwrite-metrics", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    out = run_module_c(
        root_dir=Path(args.root_dir),
        out_dir=(Path(args.out_dir) if args.out_dir else None),
        wipe_outdir=(not args.no_wipe),
        moduleb_trial_csv=args.moduleb_trial_csv,
        n_perm=args.n_perm,
        enc_n_perm=args.enc_n_perm,
        enc_topk=args.enc_topk,
        write_results_card=(not args.no_results_card),
        overwrite_metrics=(not args.no_overwrite_metrics),
    )
    print(f"[OK] Module C outputs: {out}")
    print(f"[OK] Metrics: {out / 'metrics.json'}")
    if not args.no_results_card:
        print(f"[OK] Results Card: {out / 'report' / 'results_card.html'}")
    return 0
