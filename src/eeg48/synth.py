from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

@dataclass
class SynthConfig:
    n_images: int = 6
    n_tables: int = 2
    seed: int = 42

def _write_dummy_png(path: Path) -> None:
    # Minimal 1x1 PNG to avoid external deps.
    tiny_png = bytes.fromhex(
        "89504E470D0A1A0A0000000D4948445200000001000000010802000000907753DE"
        "0000000A49444154789C636000000200015DFB02160000000049454E44AE426082"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tiny_png)

def make_synthetic_project(out_dir: Path, cfg: SynthConfig) -> Path:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = out_dir / "moduleB_outputs"
    figs = run_dir / "figs"
    tables = run_dir / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    random.seed(cfg.seed)

    for i in range(1, cfg.n_images + 1):
        _write_dummy_png(figs / f"FIG{i:02d}_dummy.png")

    for i in range(1, cfg.n_tables + 1):
        (tables / f"table{i:02d}_summary.csv").write_text(
            "metric,value\n" + "\n".join([f"m{j},{random.random():.4f}" for j in range(1, 6)]) + "\n",
            encoding="utf-8"
        )

    metrics = {
        "mean_auc": round(0.55 + random.random() * 0.1, 4),
        "p_perm": round(random.random() * 0.1, 4),
        "cv": "LOSO",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": "Synthetic demo (no private data). Replace with real outputs to generate a real Results Card."
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "README.txt").write_text(
        "Synthetic demo created by `eeg48 make-synth`.\n"
        "Contains dummy figures/tables/metrics for validating reporting.\n",
        encoding="utf-8"
    )
    return run_dir

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eeg48 make-synth", description="Create a synthetic project directory (no private data).")
    p.add_argument("--out", required=True, type=str, help="Output directory. A moduleB_outputs folder will be created under it.")
    p.add_argument("--n-images", default=6, type=int, help="Number of dummy images.")
    p.add_argument("--n-tables", default=2, type=int, help="Number of dummy tables.")
    p.add_argument("--seed", default=42, type=int, help="Random seed.")
    return p

def main(argv: Optional[list[str]] = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)
    run_dir = make_synthetic_project(Path(args.out), SynthConfig(args.n_images, args.n_tables, args.seed))
    print(f"[OK] Synthetic run dir created: {run_dir}")
    return 0
