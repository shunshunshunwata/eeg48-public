#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RE_ERP = re.compile(r"^ERP_(?P<ch>[^_]+)_(?P<t0>-?\d+)_(?P<t1>-?\d+)ms$")
RE_TFR = re.compile(r"^TFR_(?P<band>[^_]+)_(?P<roi>[^_]+)_(?P<t0>-?\d+)_(?P<t1>-?\d+)ms$")

ROI_ERP = {
    "前頭": ["Fp1","Fp2","F7","F3","Fz","F4","F8"],
    "中心": ["C3","Cz","C4"],
    "頭頂": ["P3","Pz","P4"],
    "後頭": ["O1","O2"],
    "側頭": ["T3","T4","T5","T6","T7","T8"],
}
ROI_TFR = ["frontal","central","parietal","occipital","temporal"]

def set_jp_font():
    plt.rcParams["font.family"] = [
        "Hiragino Sans","Hiragino Kaku Gothic ProN","Yu Gothic","Meiryo",
        "MS Gothic","Noto Sans CJK JP","IPAexGothic"
    ]
    plt.rcParams["axes.unicode_minus"] = False

def parse_tasks(s):
    return [t.strip() for t in s.split(",") if t.strip()]

def select_cols(df, mod, roi, tmin, tmax, split, tfr_max):
    cols=[]
    for c in df.columns:
        if not isinstance(c,str): 
            continue

        m = RE_ERP.match(c)
        if m:
            if mod != "ERP":
                continue
            ch = m.group("ch"); t0=int(m.group("t0")); t1=int(m.group("t1"))
            if t0 < tmin or t1 > tmax: 
                continue
            if ch in ROI_ERP.get(roi, []):
                cols.append(c)
            continue

        m = RE_TFR.match(c)
        if m:
            if mod not in ("TFR_SHORT","TFR_LONG"):
                continue
            roi_tag = m.group("roi"); t0=int(m.group("t0")); t1=int(m.group("t1"))
            if t0 < tmin or t1 > tmax:
                continue
            if mod=="TFR_SHORT" and t1>split:
                continue
            if mod=="TFR_LONG" and not (t1>split and t1<=tfr_max):
                continue
            if roi_tag == roi:
                cols.append(c)
    return cols

def loso_mean(modB, X, y, groups, kind, seed):
    scores=[]
    for sid in np.unique(groups):
        te = (groups==sid); tr=~te
        if tr.sum()<10 or te.sum()<5:
            continue
        sc, _aux, _coef = modB._fit_predict_score_fold_linear(X[tr], y[tr], X[te], y[te], kind, seed)
        if np.isfinite(sc):
            scores.append(float(sc))
    return float(np.mean(scores)) if scores else float("nan")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root-dir", default=".")
    ap.add_argument("--trial-features", default="moduleB_outputs/tables/moduleB_trial_eeg_features.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--mod", choices=["ERP","TFR_SHORT","TFR_LONG"], required=True)
    ap.add_argument("--tmin-ms", type=int, default=0)
    ap.add_argument("--tmax-ms", type=int, default=5000)
    ap.add_argument("--tfr-split-ms", type=int, default=1000)
    ap.add_argument("--tfr-max-ms", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()

    set_jp_font()
    root=Path(args.root_dir)
    df=pd.read_csv(root/args.trial_features)

    import importlib
    modB=importlib.import_module("moduleB_phaseB_EEG_only")
    specs = modB.build_task_specs(df, parse_tasks(args.tasks))

    out_dir = root/args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rois = list(ROI_ERP.keys()) if args.mod=="ERP" else ROI_TFR

    dummy = [c for c in df.columns if isinstance(c,str) and (c.startswith("ERP_") or c.startswith("TFR_"))][:3]

    for spec in specs:
        _Xd, y, _g2, sub = modB.prepare_xy(df, spec, dummy)
        idx = sub.index.to_numpy()
        groups = sub["subject_id"].astype(int).to_numpy()
        y = np.asarray(y)

        rows=[]
        for roi in rois:
            cols = select_cols(df, args.mod, roi, args.tmin_ms, args.tmax_ms, args.tfr_split_ms, args.tfr_max_ms)
            if len(cols) < 5:
                rows.append({"roi":roi,"n_features":len(cols),"score":np.nan})
                continue
            X = df[cols].to_numpy(dtype=float)[idx]
            sc = loso_mean(modB, X, y, groups, spec.kind, args.seed)
            rows.append({"roi":roi,"n_features":len(cols),"score":sc})

        out = pd.DataFrame(rows).sort_values("score", ascending=False)
        out.to_csv(out_dir/f"Tbl_roi_loso_{args.mod}_{spec.name}.csv", index=False, encoding="utf-8-sig")

        metric = "AUC" if spec.kind=="binary" else ("bAcc" if spec.kind=="multiclass" else "r")
        fig = plt.figure(figsize=(10, 0.45*len(rois)+3))
        plt.barh(out["roi"], out["score"])
        plt.xlabel(f"LOSO性能（指標={metric}）")
        plt.title(f"ROI別LOSO（{args.mod}）: {spec.name}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        fig.savefig(out_dir/f"Fig_roi_loso_{args.mod}_{spec.name}.png", dpi=220)
        plt.close(fig)

        print("[OK]", spec.name)

if __name__=="__main__":
    main()
