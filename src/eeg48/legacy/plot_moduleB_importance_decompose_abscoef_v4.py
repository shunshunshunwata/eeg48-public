#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# -----------------------------
# font helper（文字化け＆マイナス記号対策）
# -----------------------------
def set_jp_font(font_name: str | None):
    matplotlib.rcParams["axes.unicode_minus"] = False
    if font_name:
        matplotlib.rcParams["font.family"] = font_name
    else:
        # fallback (Mac想定)
        matplotlib.rcParams["font.family"] = "Hiragino Sans"


# -----------------------------
# feature parsers
# -----------------------------
ERP_RE = re.compile(r"^ERP_([^_]+)_(-?\d+)_(-?\d+)ms$")
TFR_RE = re.compile(r"^TFR_([^_]+)_([^_]+)_(-?\d+)_(-?\d+)ms$")


def parse_feature(name: str):
    m = ERP_RE.match(name)
    if m:
        ch = m.group(1)
        t0 = int(m.group(2))
        t1 = int(m.group(3))
        return ("ERP", {"ch": ch, "t0": t0, "t1": t1})

    m = TFR_RE.match(name)
    if m:
        band = m.group(1)
        roi = m.group(2)
        t0 = int(m.group(3))
        t1 = int(m.group(4))
        return ("TFR", {"band": band, "roi": roi, "t0": t0, "t1": t1})

    return ("OTHER", {})


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(path: Path, dpi=300):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def task_metric(task: str):
    # 論文向けの“読み手に優しい”推定
    if task == "category_3":
        return "bAcc（バランス精度）"
    if task.endswith("_high") or task == "is_ambiguous":
        return "AUC"
    return "相関（r）"


# -----------------------------
# plots
# -----------------------------
def plot_group_bar(task: str, sums: dict, out_png: Path, fontsize: int):
    labels = ["ERP", "TFR短(0–1000ms)", "TFR長(1000–5000ms)"]
    vals = [sums.get("ERP", 0.0), sums.get("TFR_SHORT", 0.0), sums.get("TFR_LONG", 0.0)]

    plt.figure(figsize=(7.5, 4.8))
    plt.bar(labels, vals)
    plt.title(f"重要度の内訳（{task}）", fontsize=fontsize)
    plt.ylabel("重要度（|係数|の合計）", fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    savefig(out_png)


def heatmap_erp(task: str, df: pd.DataFrame, out_png: Path, fontsize: int):
    # df columns: ch, t0, t1, val
    if df.empty:
        return

    # time windows sorted
    wins = sorted(df[["t0", "t1"]].drop_duplicates().apply(tuple, axis=1).tolist())
    chs = sorted(df["ch"].drop_duplicates().tolist())

    mat = np.zeros((len(chs), len(wins)), dtype=float)
    for i, ch in enumerate(chs):
        sub = df[df["ch"] == ch]
        for j, (t0, t1) in enumerate(wins):
            mat[i, j] = sub[(sub["t0"] == t0) & (sub["t1"] == t1)]["val"].sum()

    plt.figure(figsize=(max(9, 0.9 * len(wins)), max(7, 0.35 * len(chs))))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="重要度（|係数|）")
    plt.title(f"ERP重要度（チャネル×時間窓）: {task}", fontsize=fontsize)

    xlabels = [f"{t0}–{t1}" for (t0, t1) in wins]
    plt.xticks(range(len(wins)), xlabels, rotation=45, ha="right", fontsize=max(10, fontsize - 8))
    plt.yticks(range(len(chs)), chs, fontsize=max(10, fontsize - 8))
    plt.xlabel("時間窓（ms）", fontsize=fontsize - 2)
    plt.ylabel("チャネル", fontsize=fontsize - 2)
    savefig(out_png)


def heatmap_tfr(task: str, df: pd.DataFrame, out_png: Path, fontsize: int, title_suffix: str):
    # df columns: band, t0, t1, val (roiは集約済み)
    if df.empty:
        return

    wins = sorted(df[["t0", "t1"]].drop_duplicates().apply(tuple, axis=1).tolist())
    bands = ["theta", "alpha", "beta", "gamma"]
    bands = [b for b in bands if b in set(df["band"])]

    mat = np.zeros((len(bands), len(wins)), dtype=float)
    for i, band in enumerate(bands):
        sub = df[df["band"] == band]
        for j, (t0, t1) in enumerate(wins):
            mat[i, j] = sub[(sub["t0"] == t0) & (sub["t1"] == t1)]["val"].sum()

    plt.figure(figsize=(max(9, 0.9 * len(wins)), 4.8))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="重要度（|係数|）")
    plt.title(f"TFR重要度（帯域×時間窓）{title_suffix}: {task}", fontsize=fontsize)

    xlabels = [f"{t0}–{t1}" for (t0, t1) in wins]
    plt.xticks(range(len(wins)), xlabels, rotation=45, ha="right", fontsize=max(10, fontsize - 8))
    plt.yticks(range(len(bands)), bands, fontsize=max(10, fontsize - 8))
    plt.xlabel("時間窓（ms）", fontsize=fontsize - 2)
    plt.ylabel("帯域", fontsize=fontsize - 2)
    savefig(out_png)


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--tfr-split-ms", type=int, default=1000)
    ap.add_argument("--tfr-max-ms", type=int, default=5000)
    ap.add_argument("--tmin-ms", type=int, default=0)
    ap.add_argument("--fontsize", type=int, default=22)
    ap.add_argument("--font", default="Hiragino Sans")
    args = ap.parse_args()

    set_jp_font(args.font)

    in_dir = Path(args.in_dir)
    out_dir = ensure_dir(Path(args.out_dir))
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    summary_rows = []

    for task in tasks:
        p = in_dir / f"moduleB_importance_{task}_linear.csv"
        if not p.exists():
            print(f"[SKIP] not found: {p}")
            continue

        df = pd.read_csv(p)
        if "feature" not in df.columns:
            print(f"[SKIP] bad columns: {p}")
            continue

        val_col = None
        for c in df.columns:
            if c != "feature":
                val_col = c
                break
        if val_col is None:
            print(f"[SKIP] no value col: {p}")
            continue

        df["feature"] = df["feature"].astype(str)
        df["val"] = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)

        kinds = []
        parsed = []
        for s in df["feature"]:
            k, d = parse_feature(s)
            kinds.append(k)
            parsed.append(d)
        df["kind"] = kinds
        df["parsed"] = parsed

        # remove pre-stim (t1 <= 0)
        def valid_time(d):
            if "t0" not in d or "t1" not in d:
                return False
            return (d["t1"] > args.tmin_ms)

        df = df[df["parsed"].map(valid_time)].copy()

        # ---------------- group sums ----------------
        sums = {"ERP": 0.0, "TFR_SHORT": 0.0, "TFR_LONG": 0.0}

        # ERP
        erp = df[df["kind"] == "ERP"].copy()
        if not erp.empty:
            sums["ERP"] = float(erp["val"].sum())

        # TFR short / long
        tfr = df[df["kind"] == "TFR"].copy()
        if not tfr.empty:
            tfr["t0"] = tfr["parsed"].map(lambda d: d["t0"])
            tfr["t1"] = tfr["parsed"].map(lambda d: d["t1"])
            tfr_short = tfr[(tfr["t0"] >= 0) & (tfr["t1"] <= args.tfr_split_ms)]
            tfr_long = tfr[(tfr["t0"] >= args.tfr_split_ms) & (tfr["t1"] <= args.tfr_max_ms)]
            sums["TFR_SHORT"] = float(tfr_short["val"].sum())
            sums["TFR_LONG"] = float(tfr_long["val"].sum())

        # save group bar
        plot_group_bar(task, sums, out_dir / f"Fig_imp_group_{task}.png", args.fontsize)

        # ---------------- ERP heatmap (ch×time) ----------------
        if not erp.empty:
            erp2 = erp.copy()
            erp2["ch"] = erp2["parsed"].map(lambda d: d["ch"])
            erp2["t0"] = erp2["parsed"].map(lambda d: d["t0"])
            erp2["t1"] = erp2["parsed"].map(lambda d: d["t1"])
            # keep 0–1000ms only for ERP figure
            erp2 = erp2[(erp2["t0"] >= 0) & (erp2["t1"] <= 1000)]
            heatmap_erp(task, erp2[["ch", "t0", "t1", "val"]], out_dir / f"Fig_imp_ERP_ch_time_{task}.png", args.fontsize)

        # ---------------- TFR heatmaps (band×time, roi集約) ----------------
        if not tfr.empty:
            tfr2 = tfr.copy()
            tfr2["band"] = tfr2["parsed"].map(lambda d: d["band"])
            tfr2["roi"] = tfr2["parsed"].map(lambda d: d["roi"])
            # short
            tfrs = tfr2[(tfr2["t0"] >= 0) & (tfr2["t1"] <= args.tfr_split_ms)].copy()
            if not tfrs.empty:
                # roiを合算
                tfrs_agg = tfrs.groupby(["band", "t0", "t1"], as_index=False)["val"].sum()
                heatmap_tfr(task, tfrs_agg, out_dir / f"Fig_imp_TFRshort_band_time_{task}.png", args.fontsize, "（0–1000ms）")

            # long
            tfrl = tfr2[(tfr2["t0"] >= args.tfr_split_ms) & (tfr2["t1"] <= args.tfr_max_ms)].copy()
            if not tfrl.empty:
                tfrl_agg = tfrl.groupby(["band", "t0", "t1"], as_index=False)["val"].sum()
                heatmap_tfr(task, tfrl_agg, out_dir / f"Fig_imp_TFRlong_band_time_{task}.png", args.fontsize, "（1000–5000ms）")

        # ---------------- Top features CSV（論文本文に強い） ----------------
        top = df.sort_values("val", ascending=False).head(20).copy()
        top_out = out_dir / f"Tbl_top20_features_{task}.csv"
        top[["feature", "val"]].to_csv(top_out, index=False, encoding="utf-8-sig")

        summary_rows.append({
            "task": task,
            "metric_hint": task_metric(task),
            "sum_ERP": sums["ERP"],
            "sum_TFR_short": sums["TFR_SHORT"],
            "sum_TFR_long": sums["TFR_LONG"],
        })

        print(f"[OK] {task}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "Tbl_importance_group_sums_all_tasks.csv", index=False, encoding="utf-8-sig")

    print("DONE: importance decompose + heatmaps + top20 tables")


if __name__ == "__main__":
    main()
