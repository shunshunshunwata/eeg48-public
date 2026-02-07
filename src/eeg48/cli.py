from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eeg48",
        description="EEG48: public-ready research utilities (reporting, demos, legacy pipelines).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # Demo
    syn = sub.add_parser("make-synth", help="Create synthetic demo outputs (no private data).")
    syn.add_argument("--out", required=True, help="Output dir (creates <out>/moduleB_outputs).")
    syn.add_argument("--n-images", type=int, default=6)
    syn.add_argument("--n-tables", type=int, default=2)
    syn.add_argument("--seed", type=int, default=42)

    # Reporting
    rc = sub.add_parser("results-card", help="Generate a single-page HTML Results Card from a run directory.")
    rc.add_argument("--run-dir", required=True, help="Run dir containing outputs.")
    rc.add_argument("--out-dir", default=None, help="Output dir for report (default: <run-dir>/report).")
    rc.add_argument("--title", default=None, help="Optional page title.")

    # Notebook runner
    nr = sub.add_parser("notebook-run", help="Execute a notebook into an output directory (logs included).")
    nr.add_argument("--notebook", required=True, help="Path to .ipynb")
    nr.add_argument("--out-dir", required=True, help="Directory to execute notebook in")
    nr.add_argument("--timeout", type=int, default=600, help="Execution timeout seconds.")
    nr.add_argument("--executed-name", default=None, help="Optional executed notebook filename.")

    # Module A
    ma = sub.add_parser("module-a", help="Run Module A extras and generate metrics + Results Card.")
    ma.add_argument("--root-dir", required=True, help="Project root directory.")
    ma.add_argument("--in-sound-level", default=None, help="Path to master_sound_level_with_PC.csv (auto if omitted).")
    ma.add_argument("--in-corr", default=None, help="Optional phaseA_subjective_corr_matrix.csv")
    ma.add_argument("--out-dir", default=None, help="Output directory (default: <root>/moduleA_outputs)")
    ma.add_argument("--no-results-card", action="store_true")
    ma.add_argument("--no-overwrite-metrics", action="store_true")

    # Module B (legacy pipeline wrapper)
    mb = sub.add_parser("module-b", help="Run Module B (legacy) and generate metrics + Results Card.")
    mb.add_argument("--root-dir", required=True, help="Project root directory (contains derivatives/, etc.).")
    mb.add_argument("--no-results-card", action="store_true", help="Do not generate Results Card HTML.")
    mb.add_argument("--no-overwrite-metrics", action="store_true", help="Do not overwrite existing metrics.json.")
    mb.add_argument(
        "legacy_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the legacy script (use `--` before them).",
    )

    # Module C (scriptified)
    mc = sub.add_parser("module-c", help="Run Module C (scriptified) and generate outputs + Results Card.")
    mc.add_argument("--root-dir", required=True, help="Project root directory (contains derivatives/).")
    mc.add_argument("--out-dir", default=None, help="Optional absolute output directory (default: <root>/moduleC_outputs)")
    mc.add_argument("--no-wipe", action="store_true", help="Do not delete output directory before running.")
    mc.add_argument("--moduleb-trial-csv", default="moduleB_outputs/tables/moduleB_trial_eeg_features.csv")
    mc.add_argument("--n-perm", type=int, default=200)
    mc.add_argument("--enc-n-perm", type=int, default=300)
    mc.add_argument("--enc-topk", type=int, default=50)
    mc.add_argument("--no-results-card", action="store_true")
    mc.add_argument("--no-overwrite-metrics", action="store_true")

    # Module D (scriptified)
    md = sub.add_parser("module-d", help="Run Module D (scriptified) and generate outputs + Results Card.")
    md.add_argument("--root-dir", required=True, help="Project root directory (contains derivatives/).")
    md.add_argument("--out-dir", default=None, help="Optional absolute output directory (default: <root>/moduleD_outputs)")
    md.add_argument("--trial-csv", default=None)
    md.add_argument("--subject-csv", default=None)
    md.add_argument("--pc2-col", default="PC2_emotion")
    md.add_argument("--enough-ratio", type=float, default=0.80)
    md.add_argument("--neutral-th", type=float, default=0.35)
    md.add_argument("--top-k", type=int, default=10)
    md.add_argument("--n-perm", type=int, default=5000)
    md.add_argument("--n-boot", type=int, default=5000)
    md.add_argument("--seed", type=int, default=42)
    md.add_argument("--no-minimal-load", action="store_true")
    md.add_argument("--no-fig", action="store_true")
    md.add_argument("--no-results-card", action="store_true")
    md.add_argument("--no-overwrite-metrics", action="store_true")

    # Run all + master report
    ra = sub.add_parser("run-all", help="Run selected modules and generate a master report.")
    ra.add_argument("--root-dir", required=True, help="Project root directory (contains derivatives/).")
    ra.add_argument("--modules", default="A,B,C,D", help="Comma-separated modules to run, e.g., A,B or B,C,D,A.")
    ra.add_argument("--out-dir", default=None, help="Master report output dir (default: <root>/eeg48_reports).")
    ra.add_argument("--no-results-cards", action="store_true")
    ra.add_argument("--no-overwrite-metrics", action="store_true")
    ra.add_argument(
        "b_legacy_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to Module B legacy script (use `--` before them).",
    )

    # Privacy scan
    ps = sub.add_parser("privacy-scan", help="Scan repository for private identifiers (paths/emails).")
    ps.add_argument("--root", default=".", help="Root dir to scan.")
    ps.add_argument("--out", default=None, help="Optional JSON output file.")

    # Tree
    tree = sub.add_parser("tree", help="Print a compact tree (onboarding helper).")
    tree.add_argument("--root", default=".", help="Root dir to inspect.")
    tree.add_argument("--max-depth", type=int, default=3)

    return p


def cmd_make_synth(args: argparse.Namespace) -> int:
    from .synth import make_synthetic_project, SynthConfig
    run_dir = make_synthetic_project(Path(args.out), SynthConfig(args.n_images, args.n_tables, args.seed))
    print(f"[OK] Synthetic run dir: {run_dir}")
    return 0


def cmd_results_card(args: argparse.Namespace) -> int:
    from .report import generate_results_card
    out_dir = Path(args.out_dir) if args.out_dir else None
    html_path, index_path = generate_results_card(Path(args.run_dir), out_dir=out_dir, title=args.title)
    print(f"[OK] Results Card: {html_path}")
    print(f"[OK] Artifact index: {index_path}")
    return 0


def cmd_tree(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    max_depth = args.max_depth
    print(root)

    def walk(p: Path, depth: int) -> None:
        if depth > max_depth:
            return
        entries = sorted(
            [x for x in p.iterdir() if not x.name.startswith(".")],
            key=lambda x: (x.is_file(), x.name.lower()),
        )
        for i, e in enumerate(entries):
            prefix = "└── " if i == len(entries) - 1 else "├── "
            print("    " * depth + prefix + e.name + ("/" if e.is_dir() else ""))
            if e.is_dir():
                walk(e, depth + 1)

    walk(root, 0)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = build_parser()
    args = p.parse_args(argv)

    if args.cmd == "make-synth":
        return cmd_make_synth(args)

    if args.cmd == "results-card":
        return cmd_results_card(args)

    if args.cmd == "notebook-run":
        from .notebook import main as nb_main
        nb_argv = ["--notebook", args.notebook, "--out-dir", args.out_dir, "--timeout", str(args.timeout)]
        if args.executed_name:
            nb_argv += ["--executed-name", args.executed_name]
        return nb_main(nb_argv)

    if args.cmd == "module-a":
        from .pipelines.module_a import main as module_a_main
        ma_argv = ["--root-dir", args.root_dir]
        if args.in_sound_level:
            ma_argv += ["--in-sound-level", args.in_sound_level]
        if args.in_corr:
            ma_argv += ["--in-corr", args.in_corr]
        if args.out_dir:
            ma_argv += ["--out-dir", args.out_dir]
        if args.no_results_card:
            ma_argv.append("--no-results-card")
        if args.no_overwrite_metrics:
            ma_argv.append("--no-overwrite-metrics")
        return module_a_main(ma_argv)

    if args.cmd == "module-b":
        from .pipelines.module_b import main as module_b_main
        mb_argv = ["--root-dir", args.root_dir] + (args.legacy_args or [])
        if args.no_results_card:
            mb_argv.append("--no-results-card")
        if args.no_overwrite_metrics:
            mb_argv.append("--no-overwrite-metrics")
        return module_b_main(mb_argv)

    if args.cmd == "module-c":
        from .pipelines.module_c import main as module_c_main
        mc_argv = ["--root-dir", args.root_dir]
        if args.out_dir:
            mc_argv += ["--out-dir", args.out_dir]
        if args.no_wipe:
            mc_argv.append("--no-wipe")
        if args.moduleb_trial_csv:
            mc_argv += ["--moduleb-trial-csv", args.moduleb_trial_csv]
        mc_argv += ["--n-perm", str(args.n_perm), "--enc-n-perm", str(args.enc_n_perm), "--enc-topk", str(args.enc_topk)]
        if args.no_results_card:
            mc_argv.append("--no-results-card")
        if args.no_overwrite_metrics:
            mc_argv.append("--no-overwrite-metrics")
        return module_c_main(mc_argv)

    if args.cmd == "module-d":
        from .pipelines.module_d import main as module_d_main
        md_argv = ["--root-dir", args.root_dir]
        if args.out_dir:
            md_argv += ["--out-dir", args.out_dir]
        if args.trial_csv:
            md_argv += ["--trial-csv", args.trial_csv]
        if args.subject_csv:
            md_argv += ["--subject-csv", args.subject_csv]
        md_argv += ["--pc2-col", args.pc2_col]
        md_argv += ["--enough-ratio", str(args.enough_ratio), "--neutral-th", str(args.neutral_th)]
        md_argv += ["--top-k", str(args.top_k), "--n-perm", str(args.n_perm), "--n-boot", str(args.n_boot), "--seed", str(args.seed)]
        if args.no_minimal_load:
            md_argv.append("--no-minimal-load")
        if args.no_fig:
            md_argv.append("--no-fig")
        if args.no_results_card:
            md_argv.append("--no-results-card")
        if args.no_overwrite_metrics:
            md_argv.append("--no-overwrite-metrics")
        return module_d_main(md_argv)

    if args.cmd == "run-all":
        from .pipelines.run_all import main as run_all_main
        ra_argv = ["--root-dir", args.root_dir, "--modules", args.modules]
        if args.out_dir:
            ra_argv += ["--out-dir", args.out_dir]
        if args.no_results_cards:
            ra_argv.append("--no-results-cards")
        if args.no_overwrite_metrics:
            ra_argv.append("--no-overwrite-metrics")
        ra_argv += (args.b_legacy_args or [])
        return run_all_main(ra_argv)

    if args.cmd == "privacy-scan":
        from .privacy import main as privacy_main
        ps_argv = ["--root", args.root]
        if args.out:
            ps_argv += ["--out", args.out]
        return privacy_main(ps_argv)

    if args.cmd == "tree":
        return cmd_tree(args)

    return 2
