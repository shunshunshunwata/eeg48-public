#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module A extras: correlation matrix, PC scatter, and threshold/class-balance tables.

This script is intentionally "robust" to slightly different column names.
Expected inputs (recommended):
  - master_sound_level_with_PC.csv  (48 sounds x ratings + PC scores)
Optionally:
  - phaseA_subjective_corr_matrix.csv (8x8)

Outputs (saved into --out-dir):
  - Fig4-1e_subjective_corr_matrix.png
  - Fig4-1f_PC1_PC2_scatter.png
  - Tbl4-1g_binary_thresholds_and_counts.csv
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABELS = ["驚き","緊急感","脅威感","圧倒感","接近","興味","没入","退屈"]

def find_col(df, keywords):
    cols = list(df.columns)
    for kw in keywords:
        # exact
        if kw in cols:
            return kw
    # partial match
    for c in cols:
        for kw in keywords:
            if kw in str(c):
                return c
    # regex
    pat = re.compile("|".join([re.escape(k) for k in keywords]))
    for c in cols:
        if pat.search(str(c)):
            return c
    return None

def zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-sound-level", required=True, help="master_sound_level_with_PC.csv")
    ap.add_argument("--in-corr", default=None, help="phaseA_subjective_corr_matrix.csv (optional)")
    ap.add_argument("--out-dir", default="moduleA_outputs", help="output directory")
    ap.add_argument("--pc1-col", default=None, help="PC1 column name (auto if omitted)")
    ap.add_argument("--pc2-col", default=None, help="PC2 column name (auto if omitted)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_sound_level)

    # resolve PC columns
    pc1 = args.pc1_col or find_col(df, ["PC1_emotion","PC1","pc1"])
    pc2 = args.pc2_col or find_col(df, ["PC2_emotion","PC2","pc2"])
    if pc1 is None or pc2 is None:
        raise ValueError(f"PC columns not found. pc1={pc1}, pc2={pc2}. Use --pc1-col/--pc2-col.")

    # resolve rating columns
    col_map = {
        "驚き": find_col(df, ["驚き"]),
        "緊急感": find_col(df, ["緊急感"]),
        "脅威感": find_col(df, ["脅威感"]),
        "圧倒感": find_col(df, ["圧倒感","圧倒"]),
        "接近": find_col(df, ["接近"]),
        "興味": find_col(df, ["興味"]),
        "没入": find_col(df, ["没入"]),
        "退屈": find_col(df, ["退屈"]),
    }
    rating_cols = [col_map[k] for k in LABELS]
    if any(c is None for c in rating_cols):
        missing = [k for k,c in col_map.items() if c is None]
        raise ValueError(f"Rating columns not found for: {missing}. Please check input CSV header.")

    df_r = df[rating_cols].copy()
    df_r.columns = LABELS

    # 1) correlation matrix (either read or compute)
    if args.in_corr is not None:
        corr = pd.read_csv(args.in_corr, index_col=0)
        # ensure order
        corr = corr.loc[LABELS, LABELS]
    else:
        corr = df_r.corr()

    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Subjective ratings correlation (sound-level, n=48)")
    fig.tight_layout()
    fig.savefig(out_dir / "Fig4-1e_subjective_corr_matrix.png", dpi=200)
    plt.close(fig)

    # 2) PC scatter
    fig, ax = plt.subplots(figsize=(7,6))
    ax.scatter(df[pc1], df[pc2], s=40)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel(pc1)
    ax.set_ylabel(pc2)
    ax.set_title("PC1 vs PC2 (sound-level, n=48)")
    fig.tight_layout()
    fig.savefig(out_dir / "Fig4-1f_PC1_PC2_scatter.png", dpi=200)
    plt.close(fig)

    # 3) thresholds + class balance for derived targets
    proxy_arousal  = df_r[["緊急感","脅威感","驚き","圧倒感"]].mean(axis=1)
    proxy_approach = df_r["接近"] - df_r["退屈"]
    proxy_valence  = (df_r["興味"] + df_r["没入"]) - df_r["退屈"]
    emo_arousal  = df[pc1].astype(float).to_numpy()
    emo_approach = df[pc2].astype(float).to_numpy()
    emo_valence  = zscore(proxy_valence.to_numpy())

    med_arousal  = float(np.nanmedian(emo_arousal))
    med_approach = float(np.nanmedian(emo_approach))

    df_thr = pd.DataFrame([
        ["emo_arousal_high",  "median(emo_arousal)",  med_arousal,  int(np.sum(emo_arousal>=med_arousal)),  len(emo_arousal)],
        ["emo_approach_high", "median(emo_approach)", med_approach, int(np.sum(emo_approach>=med_approach)), len(emo_approach)],
        ["emo_valence_high",  "0 (z-scored proxy)",   0.0,          int(np.sum(emo_valence>=0.0)),          len(emo_valence)],
    ], columns=["task","threshold_rule","threshold_value","n_positive","n_total"])
    df_thr.to_csv(out_dir / "Tbl4-1g_binary_thresholds_and_counts.csv", index=False, encoding="utf-8-sig")

    print("Saved outputs to:", out_dir.resolve())

if __name__ == "__main__":
    main()
