# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pandas as pd
import matplotlib as mpl
mpl.rcParams["font.family"] = "Hiragino Sans"   # まずこれで固定
mpl.rcParams["axes.unicode_minus"] = False      # “−”が化ける対策

IN_DIR = Path("moduleB_outputs/tables")
OUT = Path("moduleB_outputs/figures/importance_decompose_0to5000")
OUT.mkdir(parents=True, exist_ok=True)

TASKS = "emo_arousal,emo_approach,emo_valence,emo_arousal_high,emo_approach_high,emo_valence_high,is_ambiguous,category_3".split(",")

tfr_split=1000
tfr_max=5000
tmin=0

ERP_RE = re.compile(r"^ERP_(?P<ch>[^_]+)_(?P<t0>-?\d+)_(?P<t1>-?\d+)ms$")
TFR_RE = re.compile(r"^TFR_(?P<band>[^_]+)_(?P<roi>[^_]+)_(?P<t0>-?\d+)_(?P<t1>-?\d+)ms$")

def group_of(feat: str):
    m=ERP_RE.match(feat)
    if m:
        t1=int(m.group("t1"))
        if t1<=tmin: return None
        return "ERP"
    m=TFR_RE.match(feat)
    if m:
        t1=int(m.group("t1"))
        if t1<=tmin: return None
        if t1<=tfr_split: return f"TFR_0-{tfr_split}"
        if t1<=tfr_max:   return f"TFR_{tfr_split}-{tfr_max}"
        return None
    return None

rows=[]
for task in TASKS:
    p=IN_DIR/f"moduleB_importance_{task}_linear.csv"
    df=pd.read_csv(p)
    df["group"]=df["feature"].astype(str).map(group_of)
    df=df.dropna(subset=["group"])
    for g in ["ERP", f"TFR_0-{tfr_split}", f"TFR_{tfr_split}-{tfr_max}"]:
        sub=df[df["group"]==g].sort_values("importance_abscoef", ascending=False).head(10)
        for _,r in sub.iterrows():
            rows.append({"task":task,"group":g,"feature":r["feature"],"importance_abscoef":r["importance_abscoef"]})

out=pd.DataFrame(rows)
out.to_csv(OUT/"Tbl_top10_features_by_group.csv", index=False, encoding="utf-8-sig")
print("saved:", OUT/"Tbl_top10_features_by_group.csv")
