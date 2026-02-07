#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""moduleB_roi_loso.py

ROI別のLOSO性能を追加で算出する（4.2補強）。
- 前頭のみ / 頭頂のみ など「領域だけで当たるか」を検証
- ERPは19chをROIに束ねてチャネル選択
- TFRは特徴量がTFR_<band>_<roi>_* なので、そのroiで選択
- モダリティ（ERPのみ / TFRのみ / 両方）を切り替え可能
- 置換検定（permutation）も回す（主張強度アップ）

前提
- moduleB_phaseB_EEG_only.py を同じディレクトリに置き、importできること
- moduleB_outputs/tables/moduleB_trial_eeg_features.csv が存在すること

実行例（1行推奨）
python moduleB_roi_loso.py --root-dir . --trial-features moduleB_outputs/tables/moduleB_trial_eeg_features.csv --out-dir moduleB_outputs/figures/roi_loso_0to1000 --tasks "emo_arousal,emo_approach,emo_valence,emo_arousal_high,emo_approach_high,emo_valence_high,is_ambiguous,category_3" --rois "frontal,central,parietal,occipital,temporal" --modality both --n-perm 1000 --seed 42

メモ
- ROIの定義はここで固定（10-20法19ch想定）。必要なら編集してOK。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 同階層の moduleB_phaseB_EEG_only.py を利用
import moduleB_phaseB_EEG_only as mb


ROI_ERP_CHS: Dict[str, List[str]] = {
    # 19ch想定（あなたのセットに合わせて調整してOK）
    "frontal":  ["Fp1","Fp2","F7","F3","Fz","F4","F8"],
    "central":  ["C3","Cz","C4"],
    "parietal": ["P3","Pz","P4"],
    "occipital": ["O1","O2"],
    "temporal": ["T3","T4","T5","T6"],
}


def parse_list(s: str) -> List[str]:
    return [t.strip() for t in s.split(",") if t.strip()]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_erp_ch(feat: str) -> str:
    # ERP_<ch>_<t0>_<t1>ms
    try:
        return feat.split("_")[1]
    except Exception:
        return ""


def parse_tfr_roi(feat: str) -> str:
    # TFR_<band>_<roi>_<t0>_<t1>ms
    try:
        return feat.split("_")[2]
    except Exception:
        return ""


def select_features(all_feats: List[str], roi: str, modality: str) -> List[str]:
    roi = roi.strip()
    if roi not in ROI_ERP_CHS:
        raise ValueError(f"Unknown ROI: {roi}. available={list(ROI_ERP_CHS.keys())}")

    feats: List[str] = []
    if modality in ["erp", "both"]:
        chs = set(ROI_ERP_CHS[roi])
        feats += [f for f in all_feats if f.startswith("ERP_") and parse_erp_ch(f) in chs]
    if modality in ["tfr", "both"]:
        feats += [f for f in all_feats if f.startswith("TFR_") and parse_tfr_roi(f) == roi]

    # 重複除去（順序保持）
    seen = set()
    out = []
    for f in feats:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def plot_fold_scatter(task: str, kind: str, roi: str, modality: str, fold_df: pd.DataFrame, out_png: Path) -> None:
    y = fold_df["score_primary"].to_numpy(dtype=float)
    plt.figure(figsize=(7.5, 4.0))
    plt.scatter(np.arange(len(y)) + 1, y)
    plt.axhline(float(np.nanmean(y)), linestyle="--")
    plt.grid(True)
    plt.xlabel("被験者（LOSOのfold）")
    if kind == "binary":
        ylabel = "AUC"
    elif kind == "multiclass":
        ylabel = "balanced accuracy"
    else:
        ylabel = "Pearson r"
    plt.ylabel(ylabel)
    plt.title(f"ROI別LOSO性能：{task} / {roi} / {modality}")
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=250)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", type=str, default=".")
    ap.add_argument("--trial-features", type=str, default="moduleB_outputs/tables/moduleB_trial_eeg_features.csv")
    ap.add_argument("--out-dir", type=str, default="moduleB_outputs/figures/roi_loso")
    ap.add_argument("--tasks", type=str, default="emo_arousal,emo_approach,emo_valence,emo_arousal_high,emo_approach_high,emo_valence_high,is_ambiguous,category_3")
    ap.add_argument("--rois", type=str, default="frontal,central,parietal,occipital,temporal")
    ap.add_argument("--modality", type=str, default="both", choices=["erp","tfr","both"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--perm-global", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=-1)

    args = ap.parse_args()

    root = Path(args.root_dir)
    trial_csv = (root / args.trial_features).resolve()
    out_dir = (root / args.out_dir).resolve()
    ensure_dir(out_dir)

    df = pd.read_csv(trial_csv)
    df = mb.normalize_category_column(df)

    tasks = parse_list(args.tasks)
    rois = parse_list(args.rois)

    specs = mb.build_task_specs(df, tasks)
    if not specs:
        raise RuntimeError("No valid tasks found in the trial_features table. Check column names.")

    all_feats = mb.eeg_feature_cols(df)

    rows_summary = []

    for spec in specs:
        for roi in rois:
            feats = select_features(all_feats, roi, args.modality)
            if len(feats) < 5:
                # あまりにも少ないと不安定
                continue

            X, y, groups, sub = mb.prepare_xy(df, spec, feats)
            if len(np.unique(groups)) < 2:
                continue
            if spec.kind in ["binary","multiclass"] and np.unique(y).size < 2:
                continue

            folds, _imp_df = mb.loso_evaluate_linear(X, y, groups, spec.kind, feats, seed=args.seed)
            fold_df = pd.DataFrame([{
                "task": spec.name,
                "kind": spec.kind,
                "roi": roi,
                "modality": args.modality,
                "subject_id": f.subject_id,
                "score_primary": f.score_primary,
                "score_aux": f.score_aux,
                "n_test": f.n_test,
            } for f in folds])

            obs_mean = float(np.nanmean(fold_df["score_primary"].to_numpy(dtype=float)))
            obs_sem  = float(np.nanstd(fold_df["score_primary"].to_numpy(dtype=float), ddof=1) / np.sqrt(max(1, fold_df.shape[0])))

            p_perm, null, _ = mb.permutation_test_mean_score_linear(
                X, y, groups, spec.kind, feats,
                seed=args.seed,
                n_perm=int(args.n_perm),
                within_subject=(not bool(args.perm_global)),
                n_jobs=int(args.n_jobs),
                obs_mean=obs_mean,
            )

            # 保存
            fold_path = out_dir / f"Tbl_ROI_LOSO_folds_{spec.name}_{roi}_{args.modality}.csv"
            fold_df.to_csv(fold_path, index=False)
            np.save(out_dir / f"Null_ROI_{spec.name}_{roi}_{args.modality}.npy", null)

            plot_fold_scatter(spec.name, spec.kind, roi, args.modality, fold_df,
                              out_dir / f"Fig_ROI_LOSO_{spec.name}_{roi}_{args.modality}.png")

            rows_summary.append({
                "task": spec.name,
                "kind": spec.kind,
                "roi": roi,
                "modality": args.modality,
                "n_trials": int(len(sub)),
                "n_subjects": int(np.unique(groups).size),
                "n_features": int(len(feats)),
                "score_mean": obs_mean,
                "score_sem": obs_sem,
                "p_perm": float(p_perm),
                "n_perm": int(args.n_perm),
                "perm_within_subject": bool(not args.perm_global),
            })

    if not rows_summary:
        raise RuntimeError("No ROI results computed. Check ROI definition / feature names.")

    summ = pd.DataFrame(rows_summary)
    summ.to_csv(out_dir / "Tbl_ROI_LOSO_summary.csv", index=False)

    # 全体俯瞰の図（kindごとに分ける：指標混在を避ける）
    for kind in sorted(summ["kind"].unique()):
        d = summ[summ["kind"] == kind].copy()
        # タスク×ROI の表
        pivot = d.pivot_table(index="roi", columns="task", values="score_mean", aggfunc="mean")
        plt.figure(figsize=(max(10, 1.1 * pivot.shape[1]), 4.5))
        im = plt.imshow(pivot.to_numpy(dtype=float), aspect="auto")
        plt.colorbar(im)
        plt.yticks(np.arange(pivot.shape[0]), pivot.index.tolist())
        plt.xticks(np.arange(pivot.shape[1]), pivot.columns.tolist(), rotation=45, ha="right")
        if kind == "binary":
            ttl = "ROI別LOSO性能（AUC）"
        elif kind == "multiclass":
            ttl = "ROI別LOSO性能（balanced accuracy）"
        else:
            ttl = "ROI別LOSO性能（Pearson r）"
        plt.title(ttl + f" / modality={args.modality}")
        plt.tight_layout()
        plt.savefig(out_dir / f"Fig_ROI_LOSO_heatmap_{kind}_{args.modality}.png", dpi=250)
        plt.close()

    print(f"[OK] saved: {out_dir}")


if __name__ == "__main__":
    main()
