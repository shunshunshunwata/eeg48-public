#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModuleB 4.2 強化：Temporal Generalization（時間汎化）
- trial_features（moduleB_trial_eeg_features.csv）から、時間窓ごとに特徴量を束ねる
- ある時間窓で学習 → 別の時間窓でテスト（LOSOで平均）
- ERP と TFR のどちらでも実行可能

出力：
- taskごとに「学習窓×テスト窓」のヒートマップPNG
- 集計CSV（平均スコア）

使い方例：
python moduleB_temporal_generalization_v2.py \
  --root-dir . \
  --out-dir moduleB_outputs/figures/temporal_gen_TFR_0to5000 \
  --tasks "emo_arousal,emo_approach,emo_valence,emo_arousal_high,emo_approach_high,emo_valence_high,is_ambiguous,category_3" \
  --mod TFR --tmin-ms 0 --tmax-ms 5000 --n-jobs -1 --fontsize 18
"""
from __future__ import annotations
import argparse, os, re, math
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.stats import pearsonr

# ---- optional Japanese font support ----
def setup_japanese_fonts():
    try:
        import japanize_matplotlib  # noqa: F401
        return
    except Exception:
        pass
    # fallback: try common JP fonts if visible
    try:
        import matplotlib
        cand = [
            "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", "Meiryo", "MS Gothic",
            "Noto Sans CJK JP", "IPAexGothic", "IPAGothic"
        ]
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = cand + matplotlib.rcParams.get("font.sans-serif", [])
    except Exception:
        pass

ERP_RE = re.compile(r"^ERP_([A-Za-z0-9]+)_(-?\d+)_(-?\d+)ms$")
TFR_RE = re.compile(r"^TFR_([A-Za-z0-9]+)_([A-Za-z0-9]+)_(-?\d+)_(-?\d+)ms$")

REG_TASKS = {"emo_arousal","emo_approach","emo_valence"}
BIN_TASKS = {"emo_arousal_high","emo_approach_high","emo_valence_high","is_ambiguous"}
MULTI_TASKS = {"category_3"}

def infer_task_kind(task: str) -> str:
    if task in REG_TASKS: return "reg"
    if task in BIN_TASKS: return "bin"
    if task in MULTI_TASKS: return "multi"
    # fallback: treat *_high as bin
    if task.endswith("_high"): return "bin"
    return "reg"

def fit_model(kind: str, seed: int):
    if kind == "reg":
        # Ridge: robust / fast
        return make_pipeline(StandardScaler(with_mean=True, with_std=True), Ridge(alpha=1.0, random_state=seed))
    if kind == "bin":
        return make_pipeline(StandardScaler(with_mean=True, with_std=True),
                             LogisticRegression(
                                 penalty="l2", C=1.0, solver="lbfgs",
                                 max_iter=2000, random_state=seed))
    if kind == "multi":
        return make_pipeline(StandardScaler(with_mean=True, with_std=True),
                             LogisticRegression(
                                 penalty="l2", C=1.0, solver="lbfgs",
                                 max_iter=3000, random_state=seed,
                                 multi_class="multinomial"))
    raise ValueError(kind)

def score_pred(kind: str, y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None):
    if kind == "reg":
        if np.std(y_pred) == 0 or np.std(y_true) == 0:
            return np.nan
        return pearsonr(y_true, y_pred)[0]
    if kind == "bin":
        # use proba of positive class if available; else decision scores
        if proba is None:
            # try interpret y_pred as score
            scores = y_pred
        else:
            scores = proba
        # guard single-class in fold
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, scores)
    if kind == "multi":
        return balanced_accuracy_score(y_true, y_pred)
    raise ValueError(kind)

def parse_window(col: str, mod: str):
    m = ERP_RE.match(col) if mod == "ERP" else TFR_RE.match(col)
    if not m:
        return None
    if mod == "ERP":
        ch, t0, t1 = m.group(1), int(m.group(2)), int(m.group(3))
        return (t0, t1)
    else:
        band, roi, t0, t1 = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        return (t0, t1)

def group_columns_by_window(cols, mod: str, tmin: int, tmax: int):
    win2cols = {}
    for c in cols:
        w = parse_window(c, mod)
        if w is None:
            continue
        t0, t1 = w
        if t1 <= tmin:  # strictly before analysis start
            continue
        if t0 < tmin:
            # partially overlapping: skip to keep interpretation clean
            continue
        if t1 > tmax:
            continue
        win2cols.setdefault((t0, t1), []).append(c)
    wins = sorted(win2cols.keys())
    return wins, win2cols

def loso_splits(df: pd.DataFrame):
    subs = df["subject_id"].astype(str).unique().tolist()
    for s in subs:
        test = df["subject_id"].astype(str) == s
        yield s, np.where(~test)[0], np.where(test)[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", default=".")
    ap.add_argument("--trial-features", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--mod", choices=["ERP","TFR"], required=True)
    ap.add_argument("--tmin-ms", type=int, default=0)
    ap.add_argument("--tmax-ms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--fontsize", type=int, default=18)
    args = ap.parse_args()

    root = Path(args.root_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    setup_japanese_fonts()
    import matplotlib.pyplot as plt

    trial_path = Path(args.trial_features) if args.trial_features else root/"moduleB_outputs"/"tables"/"moduleB_trial_eeg_features.csv"
    df = pd.read_csv(trial_path)

    # tasks
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    # feature columns
    feat_cols = [c for c in df.columns if c.startswith(args.mod + "_")]
    wins, win2cols = group_columns_by_window(feat_cols, args.mod, args.tmin_ms, args.tmax_ms)

    if len(wins) == 0:
        raise SystemExit(f"[ERROR] {args.mod} の時間窓が見つかりませんでした（tmin={args.tmin_ms}, tmax={args.tmax_ms}）。\n"
                         f"trial_features: {trial_path}")

    # prepare window labels
    win_labels = [f"{t0}-{t1}" for (t0,t1) in wins]

    rows = []
    for task in tasks:
        kind = infer_task_kind(task)
        if task not in df.columns:
            print(f"[SKIP] task列がありません: {task}")
            continue

        y_all = df[task].values
        # drop NaN targets
        keep = ~pd.isna(y_all)
        dft = df.loc[keep].reset_index(drop=True)
        y_all = dft[task].values

        M = np.full((len(wins), len(wins)), np.nan, dtype=float)

        for i_tr, w_tr in enumerate(wins):
            Xtr_cols = win2cols[w_tr]
            Xtr_all = dft[Xtr_cols].values

            for j_te, w_te in enumerate(wins):
                Xte_cols = win2cols[w_te]
                Xte_all = dft[Xte_cols].values

                fold_scores = []
                for sid, idx_tr, idx_te in loso_splits(dft):
                    X_train = Xtr_all[idx_tr]
                    y_train = y_all[idx_tr]
                    X_test  = Xte_all[idx_te]
                    y_test  = y_all[idx_te]

                    # skip if y_train is degenerate
                    if kind in ("bin","multi") and len(np.unique(y_train)) < 2:
                        continue

                    model = fit_model(kind, seed=args.seed)
                    model.fit(X_train, y_train)

                    if kind == "reg":
                        y_pred = model.predict(X_test)
                        sc = score_pred(kind, y_test, y_pred, None)
                    elif kind == "bin":
                        # probability of positive class (assume label 1)
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(X_test)[:,1]
                            y_pred = (proba >= 0.5).astype(int)
                            sc = score_pred(kind, y_test, y_pred, proba)
                        else:
                            y_pred = model.predict(X_test)
                            sc = score_pred(kind, y_test, y_pred, None)
                    else:  # multi
                        y_pred = model.predict(X_test)
                        sc = score_pred(kind, y_test, y_pred, None)

                    if not (sc is None or np.isnan(sc)):
                        fold_scores.append(sc)

                M[i_tr, j_te] = float(np.nanmean(fold_scores)) if len(fold_scores) else np.nan

        # plot
        fig = plt.figure(figsize=(10, 8))
        ax = plt.gca()
        im = ax.imshow(M, aspect="auto")
        ax.set_xticks(range(len(win_labels)))
        ax.set_xticklabels(win_labels, rotation=45, ha="right", fontsize=max(10, args.fontsize-4))
        ax.set_yticks(range(len(win_labels)))
        ax.set_yticklabels(win_labels, fontsize=max(10, args.fontsize-4))
        ax.set_xlabel("テスト時間窓 (ms)", fontsize=args.fontsize)
        ax.set_ylabel("学習時間窓 (ms)", fontsize=args.fontsize)

        title_mod = "ERP" if args.mod == "ERP" else "TFR"
        title_kind = {"reg":"回帰(相関r)","bin":"二値分類(AUC)","multi":"多クラス(bAcc)"}[kind]
        ax.set_title(f"時間汎化：{title_mod} / {task}（{title_kind}）", fontsize=args.fontsize+2)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=max(10, args.fontsize-4))

        fig.tight_layout()
        fig_path = out / f"Fig_temporal_gen_{args.mod}_{task}.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

        rows.append({"task": task, "mod": args.mod, "tmin_ms": args.tmin_ms, "tmax_ms": args.tmax_ms,
                     "score_mean": float(np.nanmean(M)), "score_diag_mean": float(np.nanmean(np.diag(M)))})
        print(f"[OK] {task} -> {fig_path}")

    if rows:
        pd.DataFrame(rows).to_csv(out/"Tbl_temporal_gen_summary.csv", index=False)
        print("[SAVE] summary:", out/"Tbl_temporal_gen_summary.csv")

if __name__ == "__main__":
    main()
