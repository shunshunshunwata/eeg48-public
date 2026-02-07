#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.family"] = "Hiragino Sans"   # まずこれで固定
mpl.rcParams["axes.unicode_minus"] = False      # “−”が化ける対策

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


BAND_ORDER = ["theta", "alpha", "beta", "gamma"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tables", type=str, default="moduleB_outputs/tables")
    p.add_argument("--out", type=str, default="moduleB_outputs/figures/maps")
    p.add_argument("--erp-style", type=str, default="line", choices=["line", "bar"])
    p.add_argument("--alpha", type=float, default=0.05, help="fallback alpha if q_fdr not present")
    return p.parse_args()

def _window_label(t0, t1):
    return f"{int(t0)}-{int(t1)}"

def plot_erp(csv_path: Path, out_dir: Path, style: str):
    df = pd.read_csv(csv_path)
    # expected cols: t0_ms,t1_ms,score,(q_fdr/significant_fdr) or p_perm
    if not {"t0_ms","t1_ms","score"}.issubset(df.columns):
        raise RuntimeError(f"ERP csv missing required columns: {csv_path.name}")

    df = df.sort_values(["t0_ms","t1_ms"]).reset_index(drop=True)
    x = np.arange(len(df))
    y = df["score"].to_numpy(dtype=float)
    labels = [_window_label(a,b) for a,b in zip(df["t0_ms"], df["t1_ms"])]

    # significance
    if "significant_fdr" in df.columns:
        sig = df["significant_fdr"].astype(bool).to_numpy()
    elif "q_fdr" in df.columns:
        sig = (df["q_fdr"].to_numpy(dtype=float) <= 0.05)
    elif "p_perm" in df.columns:
        sig = (df["p_perm"].to_numpy(dtype=float) <= 0.05)
    else:
        sig = np.zeros(len(df), dtype=bool)

    task = csv_path.stem.replace("moduleB_map_ERP_", "")
    out_png = out_dir / f"MAP_ERP_{task}_score_{style}.png"

    plt.figure(figsize=(12, 4.5))
    if style == "bar":
        plt.bar(x, y)
    else:
        plt.plot(x, y, marker="o")

    # mark significant windows with star above point/bar
    ymax = np.nanmax(y) if np.isfinite(y).any() else 0
    ypad = (abs(ymax) * 0.05) + 1e-6
    for i, is_sig in enumerate(sig):
        if is_sig and np.isfinite(y[i]):
            plt.text(i, y[i] + ypad, "*", ha="center", va="bottom")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Time window (ms)")
    plt.ylabel("Score (LOSO mean)")
    plt.title(f"ERP map: {task}  (* = significant)")
    plt.grid(True, axis="y")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=250)
    plt.close()

def plot_tfr(csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path)
    # expected cols: band,t0_ms,t1_ms,score,(q_fdr/significant_fdr) or p_perm
    if not {"band","t0_ms","t1_ms","score"}.issubset(df.columns):
        raise RuntimeError(f"TFR csv missing required columns: {csv_path.name}")

    df["window"] = [_window_label(a,b) for a,b in zip(df["t0_ms"], df["t1_ms"])]

    # order windows by t0_ms
    win_order = (
        df[["window","t0_ms","t1_ms"]]
        .drop_duplicates()
        .sort_values(["t0_ms","t1_ms"])
        ["window"].tolist()
    )

    # band order
    bands = df["band"].astype(str).unique().tolist()
    bands_sorted = [b for b in BAND_ORDER if b in bands] + [b for b in bands if b not in BAND_ORDER]

    mat = df.pivot_table(index="band", columns="window", values="score", aggfunc="mean")
    mat = mat.reindex(index=bands_sorted, columns=win_order)

    # significance matrix (optional)
    if "significant_fdr" in df.columns:
        sig_df = df.pivot_table(index="band", columns="window", values="significant_fdr", aggfunc="max")
        sig_df = sig_df.reindex(index=bands_sorted, columns=win_order).fillna(False).astype(bool)
    elif "q_fdr" in df.columns:
        q_df = df.pivot_table(index="band", columns="window", values="q_fdr", aggfunc="min")
        q_df = q_df.reindex(index=bands_sorted, columns=win_order)
        sig_df = (q_df <= 0.05)
    elif "p_perm" in df.columns:
        p_df = df.pivot_table(index="band", columns="window", values="p_perm", aggfunc="min")
        p_df = p_df.reindex(index=bands_sorted, columns=win_order)
        sig_df = (p_df <= 0.05)
    else:
        sig_df = pd.DataFrame(False, index=bands_sorted, columns=win_order)

    task = csv_path.stem.replace("moduleB_map_TFR_", "")
    out_png = out_dir / f"MAP_TFR_{task}_heatmap.png"

    arr = mat.to_numpy(dtype=float)
    plt.figure(figsize=(14, 4.5))
    im = plt.imshow(arr, aspect="auto")
    plt.colorbar(im, shrink=0.9, label="Score (LOSO mean)")

    plt.yticks(np.arange(len(mat.index)), mat.index.tolist())
    plt.xticks(np.arange(len(mat.columns)), mat.columns.tolist(), rotation=45, ha="right")
    plt.xlabel("Time window (ms)")
    plt.ylabel("Band")
    plt.title(f"TFR map: {task}  (* = significant)")

    # annotate significant cells
    sig_arr = sig_df.to_numpy(dtype=bool)
    for i in range(sig_arr.shape[0]):
        for j in range(sig_arr.shape[1]):
            if sig_arr[i, j] and np.isfinite(arr[i, j]):
                plt.text(j, i, "*", ha="center", va="center")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=250)
    plt.close()

def main():
    args = parse_args()
    tables = Path(args.tables)
    out_dir = Path(args.out)

    erp_files = sorted(tables.glob("moduleB_map_ERP_*.csv"))
    tfr_files = sorted(tables.glob("moduleB_map_TFR_*.csv"))

    if not erp_files and not tfr_files:
        raise SystemExit(f"No map csv found in: {tables}")

    for f in erp_files:
        plot_erp(f, out_dir, args.erp_style)

    for f in tfr_files:
        plot_tfr(f, out_dir)

    print(f"[DONE] saved figures to: {out_dir}")

if __name__ == "__main__":
    main()
