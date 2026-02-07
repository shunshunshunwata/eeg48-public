#!/usr/bin/env python
# coding: utf-8

# In[6]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
Module B (BF: EEG本体 / B+F統合)  再現性プロトコル完全実装（最終版）
============================================================

【主目的】
- EEG特徴量（ERP + TFR）からターゲット（主観/PC/カテゴリ）を予測
- 教授の「EEGが薄い」を封殺するため、LOSO + 置換検定（必須）を主軸に
- 必要な場合のみ「いつ/どの帯域が効くか」(maps) と「AIっぽい補助モデル」を追加

【入力】
- epochs_all-epo.fif（trialごとのEpochs; 例: derivatives/epochs_all/epochs_all-epo.fif）
- trial_table_all_runs.csv（trial ↔ sound ↔ subject ↔ run ↔ trial_in_run）
- master_sound_level_with_PC.csv（sound-level targets）
- qc_by_trial_all.csv（trial-level QC; qc_pass True のみ残す）

【出力】
- moduleB_outputs/
  - tables/: 主要結果（fold別・summary・置換・重要特徴） + 任意で maps
  - figures/: 任意（mapsや重要特徴）
  - logs/: 実行ログ
  - moduleB_run_metadata.json

============================================================
5) 次に“最強”にするなら（手戻り最小で効く順）
============================================================
(1) QCをONにして主結果を確定（デフォルトで apply_qc=True 推奨）
(2) 特徴量CSVを再利用して再実行を高速化（--reuse-built）
(3) 置換を並列化して時間短縮（--n-jobs -1 など）
(4) “いつ効くか”を図で出す（--with-maps, --map-tasks で最小限のタスクだけ）
(5) 「AI要素」を補助で追加（--model-family linear+mlp など）
    ※主結果は線形（解釈可能）で勝ち、AIは“補助結果”として見栄えを足すのが最強

============================================================
実行例（推奨：QCあり・主結果のみ）
============================================================
python moduleB_BF_final.py --root-dir /path/to/EEG_48sounds

（再実行を速く）
python moduleB_BF_final.py --root-dir /path/to/EEG_48sounds --reuse-built

（mapsも出す：重要タスクだけ）
python moduleB_BF_final.py --root-dir /path/to/EEG_48sounds --with-maps \
  --map-tasks emo_arousal_high,category_3 --n-perm-map 200

（置換を並列）
python moduleB_BF_final.py --root-dir /path/to/EEG_48sounds --n-jobs -1

============================================================
"""

from __future__ import annotations

import re
import sys
import json
import time
import zlib
import argparse
import logging
import platform
import inspect
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import mne
mne.set_log_level("ERROR")

import matplotlib.pyplot as plt

from joblib import Parallel, delayed

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, r2_score

# Optional (AIっぽい補助モデル)
try:
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    _HAS_MLP = True
except Exception:
    _HAS_MLP = False

import matplotlib as mpl
mpl.rcParams["font.family"] = "Hiragino Sans"   # まずこれで固定
mpl.rcParams["axes.unicode_minus"] = False      # “−”が化ける対策

# ============================================================
# Utils
# ============================================================

U32_MOD = 2**32

def seed_u32(x: int) -> int:
    return int(x % U32_MOD)

def stable_hash_u32(text: str) -> int:
    # pythonのhash()はセッションで変わるのでCRC32を使う
    return int(zlib.crc32(text.encode("utf-8")) % U32_MOD)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_logging(log_path: Path) -> None:
    ensure_dir(log_path.parent)
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

def setup_matplotlib(auto_font: bool = True) -> None:
    import matplotlib as mpl
    mpl.rcParams["axes.unicode_minus"] = False
    if not auto_font:
        return
    # macOS想定（無ければ勝手にフォールバック）
    for f in ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "Arial Unicode MS"]:
        try:
            mpl.rcParams["font.family"] = f
            break
        except Exception:
            pass

def json_dump(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def safe_read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(p)

def to_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def extract_int_prefix(s: pd.Series) -> pd.Series:
    txt = s.astype(str)
    m = txt.str.extract(r"^\s*0*(\d+)", expand=False)
    return pd.to_numeric(m, errors="coerce").astype("Int64")

def extract_any_int(s: pd.Series) -> pd.Series:
    txt = s.astype(str)
    m = txt.str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(m, errors="coerce").astype("Int64")

def pearsonr_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2:
        return np.nan
    if np.allclose(np.std(a), 0) or np.allclose(np.std(b), 0):
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def fdr_bh(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR. returns (reject_bool, qvals)"""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    qvals = np.empty_like(q)
    qvals[order] = q
    reject = qvals <= alpha
    return reject, qvals

def _first_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    return None

def normalize_category_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    mergeでカテゴリーが カテゴリー_x / カテゴリー_y に割れたときに
    最終的に df['カテゴリー'] を確実に作る。
    ついでに不要な列は落とす。
    """
    df = df.copy()

    # 候補（あなたの現状に合わせて）
    cand = [c for c in ["カテゴリー", "カテゴリー_y", "カテゴリー_x", "category", "Category", "カテゴリ"] if c in df.columns]
    if not cand:
        return df

    # 「不快/中間/快」を含む列を優先して採用（最も堅い）
    def looks_like_3cat(s: pd.Series) -> bool:
        u = set(s.dropna().astype(str).str.strip().unique().tolist())
        return len(u & {"不快", "中間", "快"}) > 0

    pick = None
    for c in ["カテゴリー_y", "カテゴリー_x", "カテゴリー", "category", "Category", "カテゴリ"]:
        if c in df.columns and looks_like_3cat(df[c]):
            pick = c
            break

    # それでも決まらない場合は、存在する最優先列を使う
    if pick is None:
        pick = cand[0]

    df["カテゴリー"] = df[pick].astype(str).str.strip()

    # いらない派生列を削除（事故防止）
    drop_cols = [c for c in ["カテゴリー_x", "カテゴリー_y"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


# ============================================================
# Args
# ============================================================

def parse_windows_ms(s: str) -> List[Tuple[int, int]]:
    """
    例: "-500-0,0-200,200-400,5000-12000"
    "-2000--500" のような負数もOK
    """
    out: List[Tuple[int, int]] = []
    for chunk in [c.strip() for c in s.split(",") if c.strip()]:
        m = re.match(r"^\s*(-?\d+)\s*-\s*(-?\d+)\s*$", chunk)
        if not m:
            raise ValueError(f"Bad window format: {chunk}")
        out.append((int(m.group(1)), int(m.group(2))))
    return out

def parse_bands(s: str) -> List[Tuple[str, int, int]]:
    # 例: "4-7,8-12,13-30,31-40" → theta/alpha/beta/gamma
    names = ["theta", "alpha", "beta", "gamma", "band5", "band6"]
    chunks = [c.strip() for c in s.split(",") if c.strip()]
    bands: List[Tuple[str, int, int]] = []
    for i, ch in enumerate(chunks):
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", ch)
        if not m:
            raise ValueError(f"Bad band format: {ch}")
        f0, f1 = int(m.group(1)), int(m.group(2))
        bands.append((names[i] if i < len(names) else f"band{i+1}", f0, f1))
    return bands

def parse_list(s: str) -> List[str]:
    return [t.strip() for t in s.split(",") if t.strip()]

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Module B (BF: EEG本体 / B+F統合) - QC + LOSO + permutation (+ optional maps/AI)",
        add_help=True,
    )
    p.add_argument("--root-dir", type=str, default=".")

    # Inputs
    p.add_argument("--trial-features-csv", type=str, default="")  # 任意（無ければ build）
    p.add_argument("--reuse-built", action="store_true")          # moduleB_trial_eeg_features.csv があれば再利用
    p.add_argument("--epochs-all", type=str, default="")          # 任意（無ければ ROOT/derivatives/epochs_all/epochs_all-epo.fif）
    p.add_argument("--trial-table", type=str, default="")         # 任意（無ければ ROOT/derivatives/trial_table/trial_table_all_runs.csv）
    p.add_argument("--master-sound-pc", type=str, default="")     # 任意（無ければ ROOT/derivatives/master_tables/master_sound_level_with_PC.csv）

    # QC（デフォルトON）
    p.add_argument("--no-qc", action="store_true") 
    p.add_argument("--qc-by-trial", type=str, default="")  # 任意（無ければ ROOT/derivatives/qc_all/qc_by_trial_all.csv）
    p.add_argument("--qc-pass-col", type=str, default="")  # 任意（空なら自動で qc_pass / qc_autoreject_pass を探す）

    # Repro / compute
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=1)

    # Permutation（主結果）
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--perm-global", action="store_true")  # Trueなら全体シャッフル（デフォルトは被験者内）

    # Optional maps（重い：必要時だけ）
    p.add_argument("--with-maps", action="store_true")
    p.add_argument("--no-maps", action="store_true") 
    p.add_argument("--map-tasks", type=str, default="")      # 空なら全taskにmaps（推奨は重要taskだけ）
    p.add_argument("--n-perm-map", type=int, default=200)    # mapsセル置換（軽め推奨）

    # Feature build
    p.add_argument("--resample", type=float, default=250.0)
    p.add_argument("--crop-tmin", type=float, default=-2.0)
    p.add_argument("--crop-tmax", type=float, default=12.0)
    p.add_argument("--baseline-tmin", type=float, default=-0.2)
    p.add_argument("--baseline-tmax", type=float, default=0.0)

    # Windows (ms)
    p.add_argument(
        "--erp-windows",
        type=str,
        default="-2000--1500,-1500--1000,-1000--500,-500-0,0-200,200-400,400-800,800-1200,1200-2000,2000-3000,3000-4000,4000-5000,5000-12000"
    )
    p.add_argument(
        "--tfr-windows",
        type=str,
        default="-2000--1500,-1500--1000,-1000--500,-500-0,0-200,200-400,400-800,800-1200,1200-2000,2000-3000,3000-4000,4000-5000,5000-12000"
    )
    p.add_argument("--tfr-freqs", type=str, default="4-7,8-12,13-30,31-40")

    # Tasks
    p.add_argument(
        "--tasks",
        type=str,
        default="emo_arousal,emo_approach,emo_valence,emo_arousal_high,emo_approach_high,emo_valence_high,is_ambiguous,category_3"
    )

    # Model family
    p.add_argument(
        "--model-family",
        type=str,
        default="linear",
        choices=["linear", "linear+mlp"],
        help="主結果は線形。mlpは補助（AIっぽさ用）"
    )

    p.add_argument("--no-auto-font", action="store_true")

    if argv is None:
        argv = sys.argv[1:]
    args, _unknown = p.parse_known_args(argv)  # notebook の -f kernel.json を無視
    return args


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    root_dir: Path
    seed: int
    n_jobs: int
    n_perm: int
    n_perm_map: int
    perm_within_subject: bool

    apply_qc: bool
    qc_by_trial_csv: Path
    qc_pass_col: str

    with_maps: bool
    map_tasks: List[str]

    resample_sfreq: float
    crop_tmin: float
    crop_tmax: float
    baseline_tmin: float
    baseline_tmax: float

    erp_windows_ms: List[Tuple[int, int]]
    tfr_windows_ms: List[Tuple[int, int]]
    bands: List[Tuple[str, int, int]]

    tasks: List[str]
    model_family: str

    # Paths
    derivatives_dir: Path
    trial_table_csv: Path
    master_sound_pc_csv: Path
    epochs_all_path: Path

    out_dir: Path
    tables_dir: Path
    figures_dir: Path
    logs_dir: Path

    trial_features_csv_out: Path
    run_metadata_json: Path

def make_config(args: argparse.Namespace) -> Config:
    root = Path(args.root_dir).expanduser()
    derivatives = root / "derivatives"

    trial_table = Path(args.trial_table).expanduser() if args.trial_table else derivatives / "trial_table" / "trial_table_all_runs.csv"
    master_pc  = Path(args.master_sound_pc).expanduser() if args.master_sound_pc else derivatives / "master_tables" / "master_sound_level_with_PC.csv"
    epochs_all = Path(args.epochs_all).expanduser() if args.epochs_all else derivatives / "epochs_all" / "epochs_all-epo.fif"
    qc_by_trial = Path(args.qc_by_trial).expanduser() if args.qc_by_trial else derivatives / "qc_all" / "qc_by_trial_all.csv"

    out_dir = root / "moduleB_outputs"
    tables  = out_dir / "tables"
    figs    = out_dir / "figures"
    logs    = out_dir / "logs"
    for p in [out_dir, tables, figs, logs]:
        ensure_dir(p)

    erp_windows = parse_windows_ms(args.erp_windows)
    tfr_windows = parse_windows_ms(args.tfr_windows)
    bands       = parse_bands(args.tfr_freqs)
    tasks       = parse_list(args.tasks)

    # ★ここで「代入」を済ませる（Config呼び出しの中に書かない）
    apply_qc  = (not bool(args.no_qc))   # デフォルトON、--no-qcでOFF
    with_maps = (bool(args.with_maps) and (not bool(args.no_maps)))
    map_tasks = parse_list(args.map_tasks) if args.map_tasks else []

    cfg = Config(
        root_dir=root,
        seed=int(args.seed),
        n_jobs=int(args.n_jobs),
        n_perm=int(args.n_perm),
        n_perm_map=int(args.n_perm_map),
        perm_within_subject=(not bool(args.perm_global)),

        apply_qc=apply_qc,
        qc_by_trial_csv=qc_by_trial,
        qc_pass_col=str(args.qc_pass_col).strip(),

        with_maps=with_maps,
        map_tasks=map_tasks,

        resample_sfreq=float(args.resample),
        crop_tmin=float(args.crop_tmin),
        crop_tmax=float(args.crop_tmax),
        baseline_tmin=float(args.baseline_tmin),
        baseline_tmax=float(args.baseline_tmax),

        erp_windows_ms=erp_windows,
        tfr_windows_ms=tfr_windows,
        bands=bands,

        tasks=tasks,
        model_family=str(args.model_family),

        derivatives_dir=derivatives,
        trial_table_csv=trial_table,
        master_sound_pc_csv=master_pc,
        epochs_all_path=epochs_all,

        out_dir=out_dir,
        tables_dir=tables,
        figures_dir=figs,
        logs_dir=logs,

        trial_features_csv_out=tables / "moduleB_trial_eeg_features.csv",
        run_metadata_json=tables / "moduleB_run_metadata.json",
    )

    # QCファイルが無ければOFFに落とす（ログはmainのsetup_logging後が望ましいが、挙動はOK）
    if cfg.apply_qc and (not cfg.qc_by_trial_csv.exists()):
        cfg.apply_qc = False

    return cfg



# ============================================================
# Load tables: trial_table & master_sound
# ============================================================

def load_trial_table(cfg: Config) -> pd.DataFrame:
    tt = safe_read_csv(cfg.trial_table_csv)

    if "subject_id" not in tt.columns:
        c = _first_col(tt, ["subject", "Subject", "participant", "participant_id", "sub", "sub_id"])
        if c:
            tt["subject_id"] = tt[c]
    if "subject_id" in tt.columns and not pd.api.types.is_numeric_dtype(tt["subject_id"]):
        sid = to_int_series(tt["subject_id"])
        if sid.isna().mean() > 0.5:
            sid = extract_int_prefix(tt["subject_id"])
        tt["subject_id"] = sid
    tt["subject_id"] = to_int_series(tt["subject_id"])

    if "run_id" not in tt.columns:
        c = _first_col(tt, ["run", "Run", "block", "session"])
        if c:
            tt["run_id"] = tt[c]
    if "run_id" in tt.columns and not pd.api.types.is_numeric_dtype(tt["run_id"]):
        tt["run_id"] = extract_any_int(tt["run_id"])
    tt["run_id"] = to_int_series(tt["run_id"])

    if "trial_in_run" not in tt.columns:
        c = _first_col(tt, ["trial_in_run", "trial", "Trial", "trial_num", "trial_index"])
        if c:
            tt["trial_in_run"] = tt[c]
    tt["trial_in_run"] = to_int_series(tt["trial_in_run"])

    if "number" not in tt.columns:
        c = _first_col(tt, ["sound_id", "SoundID", "sound", "stim_id", "stimulus_id"])
        if c:
            tt["number"] = tt[c]
    if "number" in tt.columns and not pd.api.types.is_numeric_dtype(tt["number"]):
        nn = to_int_series(tt["number"])
        if nn.isna().mean() > 0.5:
            nn = extract_any_int(tt["number"])
        tt["number"] = nn
    tt["number"] = to_int_series(tt["number"])

    if "カテゴリー" not in tt.columns:
        c = _first_col(tt, ["category", "Category", "カテゴリ"])
        if c:
            tt["カテゴリー"] = tt[c]

    keep = ["subject_id", "run_id", "trial_in_run", "number"]
    if "カテゴリー" in tt.columns:
        keep.append("カテゴリー")
    tt = tt[keep].copy()
    # uniqueness for merge keys
    return tt

def load_master_sound_pc(cfg: Config) -> pd.DataFrame:
    ms = safe_read_csv(cfg.master_sound_pc_csv)
    if "number" not in ms.columns:
        c = _first_col(ms, ["sound_id", "SoundID", "sound", "stim_id"])
        if c:
            ms["number"] = ms[c]
    ms["number"] = to_int_series(ms["number"])
    return ms

def add_highlow_and_ambiguous(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # continuous emo -> high/low median split
    for col in ["emo_arousal", "emo_approach", "emo_valence"]:
        if col in df.columns and f"{col}_high" not in df.columns:
            med = float(pd.to_numeric(df[col], errors="coerce").median())
            df[f"{col}_high"] = (pd.to_numeric(df[col], errors="coerce") >= med).astype(int)

    # ambiguous
    if "is_ambiguous" not in df.columns:
        if "is_ambiguous_approach_sd_top10" in df.columns:
            x = df["is_ambiguous_approach_sd_top10"]
            if x.dtype != bool:
                x = x.fillna(False).astype(bool)
            df["is_ambiguous"] = x.astype(int)
        else:
            df["is_ambiguous"] = 0

    return df


# ============================================================
# Metadata recovery: subject_id, run_id, trial_in_run, number
# ============================================================

def ensure_keys_in_metadata(meta: pd.DataFrame, tt: pd.DataFrame) -> pd.DataFrame:
    """
    epochs.metadata 行数を変えずに keys を復元する（dropしない）
    keys: subject_id, run_id, trial_in_run, number
    """
    meta = meta.copy()
    n = len(meta)
    meta["_row_id"] = np.arange(n)

    # subject_id
    if "subject_id" not in meta.columns:
        c = _first_col(meta, ["subject_id", "subject", "Subject", "participant", "participant_id", "sub", "sub_id"])
        if c:
            meta["subject_id"] = meta[c]
    if "subject_id" in meta.columns and not pd.api.types.is_numeric_dtype(meta["subject_id"]):
        sid = to_int_series(meta["subject_id"])
        if sid.isna().mean() > 0.5:
            sid = extract_int_prefix(meta["subject_id"])
        meta["subject_id"] = sid
    if "subject_id" in meta.columns:
        meta["subject_id"] = to_int_series(meta["subject_id"])

    # run_id
    if "run_id" not in meta.columns:
        c = _first_col(meta, ["run_id", "run", "Run", "block", "session"])
        if c:
            meta["run_id"] = meta[c]
    if "run_id" in meta.columns and not pd.api.types.is_numeric_dtype(meta["run_id"]):
        meta["run_id"] = extract_any_int(meta["run_id"])
    if "run_id" in meta.columns:
        meta["run_id"] = to_int_series(meta["run_id"])

    # trial_in_run
    if "trial_in_run" not in meta.columns:
        c = _first_col(meta, ["trial_in_run", "trial", "Trial", "trial_num", "trial_index"])
        if c:
            meta["trial_in_run"] = meta[c]
    if "trial_in_run" in meta.columns and not pd.api.types.is_numeric_dtype(meta["trial_in_run"]):
        meta["trial_in_run"] = extract_any_int(meta["trial_in_run"])
    if "trial_in_run" in meta.columns:
        meta["trial_in_run"] = to_int_series(meta["trial_in_run"])

    # number
    if "number" not in meta.columns:
        c = _first_col(meta, ["number", "sound_id", "SoundID", "sound", "stim_id", "stimulus_id"])
        if c:
            meta["number"] = meta[c]
    if "number" in meta.columns and not pd.api.types.is_numeric_dtype(meta["number"]):
        nn = to_int_series(meta["number"])
        if nn.isna().mean() > 0.5:
            nn = extract_any_int(meta["number"])
        meta["number"] = nn
    if "number" in meta.columns:
        meta["number"] = to_int_series(meta["number"])

    # merge with trial_table to fill missing keys (strong: subject_id+run_id+trial_in_run)
    keys = ["subject_id", "run_id", "trial_in_run"]
    if all(k in meta.columns for k in keys):
        tmp = meta.merge(
            tt[keys + ["number", "カテゴリー"]].drop_duplicates(keys),
            on=keys,
            how="left",
            suffixes=("", "_tt"),
        )
        if "number_tt" in tmp.columns:
            if "number" in tmp.columns:
                tmp["number"] = tmp["number"].fillna(tmp["number_tt"])
            else:
                tmp["number"] = tmp["number_tt"]
        if "カテゴリー_tt" in tmp.columns and "カテゴリー" not in tmp.columns:
            tmp["カテゴリー"] = tmp["カテゴリー_tt"]
        meta = tmp.drop(columns=[c for c in ["number_tt", "カテゴリー_tt"] if c in tmp.columns])

    # final fallback: row align only if N一致
    needed = ["subject_id", "run_id", "trial_in_run", "number"]
    if any((k not in meta.columns) for k in needed) or any(meta[k].isna().all() for k in needed if k in meta.columns):
        if len(meta) == len(tt):
            logging.warning("[meta-fix] fallback row-align: epochs order assumed identical to trial_table (N一致).")
            tt2 = tt.reset_index(drop=True)
            meta = meta.reset_index(drop=True)
            for k in ["subject_id", "run_id", "trial_in_run", "number"]:
                meta[k] = tt2[k].values
            if "カテゴリー" in tt2.columns and "カテゴリー" not in meta.columns:
                meta["カテゴリー"] = tt2["カテゴリー"].values
        else:
            raise RuntimeError("[meta-fix] cannot recover keys (merge failed and N mismatch).")

    # validate
    for k in ["subject_id", "run_id", "trial_in_run", "number"]:
        if k not in meta.columns:
            raise RuntimeError(f"[meta-fix] missing {k} in metadata. cols={list(meta.columns)}")
        if meta[k].isna().any():
            raise RuntimeError(f"[meta-fix] still NA in {k}: {int(meta[k].isna().sum())}/{len(meta)}")

    meta = meta.drop(columns=["_row_id"], errors="ignore")
    return meta


# ============================================================
# EEG channel canonicalization (19ch)
# ============================================================

CANON_19 = ["Fp1","Fp2","F7","F3","Fz","F4","F8","T7","C3","Cz","C4","T8","P7","P3","Pz","P4","P8","O1","O2"]

def _norm_ch(name: str) -> str:
    s = name.upper()
    s = s.replace("EEG", "")
    s = s.replace(" ", "")
    s = s.replace("-REF", "").replace("REF", "")
    s = s.replace("-", "")
    # 旧命名互換
    s = s.replace("T3", "T7").replace("T4", "T8").replace("T5", "P7").replace("T6", "P8")
    return s

def pick_and_rename_19ch(epochs: mne.Epochs) -> mne.Epochs:
    # pick EEG only
    try:
        epochs = epochs.copy().pick_types(eeg=True, eog=False, stim=False, misc=False)
    except Exception:
        epochs = epochs.copy().pick("eeg")

    cur = list(epochs.ch_names)
    norm_map = {_norm_ch(c): c for c in cur}
    rename = {}
    keep = []
    for canon in CANON_19:
        key = canon.upper()
        if key in norm_map:
            orig = norm_map[key]
            rename[orig] = canon
            keep.append(orig)

    if not keep:
        raise RuntimeError(f"[channels] none of 19ch matched. available={cur}")

    epochs = epochs.copy().pick_channels(keep, ordered=True)
    epochs.rename_channels(rename)
    ordered = [c for c in CANON_19 if c in epochs.ch_names]
    epochs = epochs.copy().reorder_channels(ordered)
    return epochs


# ============================================================
# Feature extraction: ERP + TFR
# ============================================================

def compute_erp_features(epochs: mne.Epochs, windows_ms: List[Tuple[int,int]]) -> pd.DataFrame:
    """
    ERP: channel × time-window mean amplitude
    features: ERP_<ch>_<t0>_<t1>ms
    ※ DataFrame断片化を避けるため、最後に一括concat
    """
    data = epochs.get_data()  # (n, ch, t)
    times_ms = epochs.times * 1000.0
    chs = epochs.ch_names

    md = epochs.metadata.reset_index(drop=True).copy()
    col_names: List[str] = []
    col_arrays: List[np.ndarray] = []

    for (t0, t1) in windows_ms:
        m = (times_ms >= t0) & (times_ms < t1)
        if not m.any():
            continue
        seg = data[:, :, m].mean(axis=2)  # (n, ch)
        for ci, ch in enumerate(chs):
            col_names.append(f"ERP_{ch}_{t0}_{t1}ms")
            col_arrays.append(seg[:, ci])

    if not col_arrays:
        return md

    feat = np.column_stack(col_arrays)
    feat_df = pd.DataFrame(feat, columns=col_names)
    return pd.concat([md, feat_df], axis=1)

def make_rois_from_19ch(ch_names: List[str]) -> Dict[str, List[str]]:
    rois = {
        "frontal":  [c for c in ["F3","F4","F7","F8","Fp1","Fp2","Fz"] if c in ch_names],
        "central":  [c for c in ["C3","C4","Cz"] if c in ch_names],
        "parietal": [c for c in ["P3","P4","Pz","P7","P8"] if c in ch_names],
        "occipital":[c for c in ["O1","O2"] if c in ch_names],
    }
    return {k:v for k,v in rois.items() if len(v)>0}

def compute_tfr_features(
    epochs: mne.Epochs,
    rois: Dict[str,List[str]],
    bands: List[Tuple[str,int,int]],
    windows_ms: List[Tuple[int,int]],
    baseline: Tuple[float, float],
) -> pd.DataFrame:
    """
    TFR: ROI平均（仮想1ch）→ Morlet → baseline ratio → band×time-window平均
    features: TFR_<band>_<roi>_<t0>_<t1>ms
    """
    from mne.time_frequency import tfr_morlet

    md_base = epochs.metadata.reset_index(drop=True).copy()
    n = len(epochs)

    # freqs: union of bands (step=2Hz)
    fmin = min(b[1] for b in bands)
    fmax = max(b[2] for b in bands)
    freqs = np.arange(float(fmin), float(fmax) + 0.1, 2.0)
    n_cycles = freqs / 2.0

    feat_names: List[str] = []
    feat_arrays: List[np.ndarray] = []

    for roi, chs in rois.items():
        ep = epochs.copy().pick_channels(chs)
        data = ep.get_data().mean(axis=1, keepdims=True).astype(np.float32)  # (n,1,t)

        info = mne.create_info([f"ROI_{roi}"], sfreq=ep.info["sfreq"], ch_types=["eeg"])
        events = np.c_[np.arange(n), np.zeros(n, dtype=int), np.ones(n, dtype=int)]
        ep_roi = mne.EpochsArray(data, info, events=events, tmin=ep.tmin, verbose="ERROR")

        power = tfr_morlet(
            ep_roi,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=False,
            average=False,
            decim=5,
            n_jobs=1,
            verbose="ERROR",
        )
        try:
            power.apply_baseline(baseline=baseline, mode="ratio")
        except Exception:
            pass

        pow_data = power.data[:, 0, :, :]  # (n, f, t)
        times_ms = power.times * 1000.0

        for (bname, f0, f1) in bands:
            fm = (freqs >= f0) & (freqs <= f1)
            if not fm.any():
                continue
            for (t0, t1) in windows_ms:
                tm = (times_ms >= t0) & (times_ms < t1)
                if not tm.any():
                    continue
                vals = pow_data[:, fm][:, :, tm].mean(axis=(1, 2))
                feat_names.append(f"TFR_{bname}_{roi}_{t0}_{t1}ms")
                feat_arrays.append(vals)

    if not feat_arrays:
        return md_base

    feat = np.column_stack(feat_arrays)
    feat_df = pd.DataFrame(feat, columns=feat_names)
    return pd.concat([md_base, feat_df], axis=1)

def eeg_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("ERP_") or c.startswith("TFR_")]


# ============================================================
# QC
# ============================================================

def load_qc_by_trial(cfg: Config) -> pd.DataFrame:
    qc = safe_read_csv(cfg.qc_by_trial_csv)

    # keys
    if "subject_id" not in qc.columns:
        c = _first_col(qc, ["subject", "Subject", "participant", "participant_id"])
        if c:
            qc["subject_id"] = qc[c]
    if "subject_id" in qc.columns and not pd.api.types.is_numeric_dtype(qc["subject_id"]):
        sid = to_int_series(qc["subject_id"])
        if sid.isna().mean() > 0.5:
            sid = extract_int_prefix(qc["subject_id"])
        qc["subject_id"] = sid
    qc["subject_id"] = to_int_series(qc["subject_id"])

    if "run_id" not in qc.columns:
        c = _first_col(qc, ["run", "Run", "block", "session"])
        if c:
            qc["run_id"] = qc[c]
    if "run_id" in qc.columns and not pd.api.types.is_numeric_dtype(qc["run_id"]):
        qc["run_id"] = extract_any_int(qc["run_id"])
    qc["run_id"] = to_int_series(qc["run_id"])

    if "trial_in_run" not in qc.columns:
        c = _first_col(qc, ["trial_in_run", "trial", "Trial", "trial_num", "trial_index"])
        if c:
            qc["trial_in_run"] = qc[c]
    qc["trial_in_run"] = to_int_series(qc["trial_in_run"])

    # pass col auto-detect
    pass_col = cfg.qc_pass_col
    if not pass_col:
        for cand in ["qc_pass", "qc_autoreject_pass", "pass", "is_pass"]:
            if cand in qc.columns:
                pass_col = cand
                break
    if not pass_col or pass_col not in qc.columns:
        raise RuntimeError(f"[QC] pass column not found. candidates: qc_pass/qc_autoreject_pass. cols={list(qc.columns)}")

    qc["qc_pass"] = qc[pass_col].astype(bool)

    keep = ["subject_id", "run_id", "trial_in_run", "qc_pass"]
    qc = qc[keep].copy()
    qc = qc.drop_duplicates(["subject_id", "run_id", "trial_in_run"])
    return qc

def apply_qc_filter(df: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    keys = ["subject_id", "run_id", "trial_in_run"]
    for k in keys:
        if k not in df.columns:
            raise RuntimeError(f"[QC] feature df missing key {k}. cols={list(df.columns)}")

    # ★衝突回避：df側にqc_passが既にあるなら捨てる（QCファイルを正とする）
    df2 = df.drop(columns=["qc_pass"], errors="ignore").copy()

    # ★サフィックスを明示して、qc側の判定列を必ず拾えるようにする
    merged = df2.merge(qc, on=keys, how="left", suffixes=("", "_qc"))

    # ここで qc_pass が無いケースも潰す（保険）
    if "qc_pass" not in merged.columns:
        if "qc_pass_qc" in merged.columns:
            merged["qc_pass"] = merged["qc_pass_qc"]
        else:
            raise RuntimeError(f"[QC] qc_pass missing after merge. cols={list(merged.columns)}")

    matched = int(merged["qc_pass"].notna().sum())
    kept = int((merged["qc_pass"] == True).sum())
    logging.info(f"[QC] keys={keys} | matched={matched}/{len(merged)} | kept(pass)={kept}/{len(merged)}")

    merged = merged.loc[merged["qc_pass"] == True].drop(columns=["qc_pass"], errors="ignore")
    merged = merged.drop(columns=["qc_pass_qc"], errors="ignore")
    return merged.reset_index(drop=True)



# ============================================================
# Build / Load trial features
# ============================================================

def load_or_build_features(cfg: Config, trial_features_in: Optional[Path], reuse_built: bool) -> pd.DataFrame:
    tt = load_trial_table(cfg)
    ms = load_master_sound_pc(cfg)

    # 0) reuse built
    if reuse_built and cfg.trial_features_csv_out.exists() and (trial_features_in is None or not trial_features_in.exists()):
        logging.info(f"[LOAD] reuse built trial_features: {cfg.trial_features_csv_out}")
        df = pd.read_csv(cfg.trial_features_csv_out)
        df = add_highlow_and_ambiguous(df)
        df = normalize_category_column(df)
        return df

    # 1) if provided CSV exists
    if trial_features_in and trial_features_in.exists():
        logging.info(f"[LOAD] trial_features_csv_in: {trial_features_in}")
        df = pd.read_csv(trial_features_in)
        # minimal key normalization
        if "subject_id" not in df.columns:
            c = _first_col(df, ["subject", "Subject", "participant", "participant_id", "sub", "sub_id"])
            if c:
                df["subject_id"] = df[c]
        if "subject_id" in df.columns and not pd.api.types.is_numeric_dtype(df["subject_id"]):
            sid = to_int_series(df["subject_id"])
            if sid.isna().mean() > 0.5:
                sid = extract_int_prefix(df["subject_id"])
            df["subject_id"] = sid
        df["subject_id"] = to_int_series(df["subject_id"])

        if "run_id" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["run_id"]):
                df["run_id"] = extract_any_int(df["run_id"])
            df["run_id"] = to_int_series(df["run_id"])
        if "trial_in_run" in df.columns:
            df["trial_in_run"] = to_int_series(extract_any_int(df["trial_in_run"]) if not pd.api.types.is_numeric_dtype(df["trial_in_run"]) else df["trial_in_run"])

        if "number" not in df.columns:
            c = _first_col(df, ["sound_id", "SoundID", "sound", "stim_id", "stimulus_id"])
            if c:
                df["number"] = df[c]
        if "number" in df.columns and not pd.api.types.is_numeric_dtype(df["number"]):
            nn = to_int_series(df["number"])
            if nn.isna().mean() > 0.5:
                nn = extract_any_int(df["number"])
            df["number"] = nn
        df["number"] = to_int_series(df["number"])

        # attach category/keys from trial_table if possible
        if all(k in df.columns for k in ["subject_id", "run_id", "trial_in_run", "number"]):
            df = df.merge(tt, on=["subject_id","run_id","trial_in_run","number"], how="left")
        else:
            tmp = tt[["subject_id","number","カテゴリー"]].drop_duplicates(["subject_id","number"])
            if "カテゴリー" not in df.columns:
                df = df.merge(tmp, on=["subject_id","number"], how="left")

        # attach master targets
        keep = ["number"]
        for c in ["emo_arousal","emo_approach","emo_valence","PC1_emotion","PC2_emotion","PC3_emotion","is_ambiguous_approach_sd_top10"]:
            if c in ms.columns:
                keep.append(c)
        df = df.merge(ms[keep].drop_duplicates("number"), on="number", how="left")
        df = add_highlow_and_ambiguous(df)
        df = normalize_category_column(df)

        return df

    # 2) build from epochs_all
    if not cfg.epochs_all_path.exists():
        raise FileNotFoundError(f"epochs_all not found: {cfg.epochs_all_path}")

    logging.info(f"[BUILD] epochs_all: {cfg.epochs_all_path}")
    epochs = mne.read_epochs(cfg.epochs_all_path, preload=True, verbose="ERROR")
    epochs = pick_and_rename_19ch(epochs)

    # resample
    if abs(epochs.info["sfreq"] - cfg.resample_sfreq) > 1e-6:
        epochs.resample(cfg.resample_sfreq)

    # crop
    try:
        epochs.crop(tmin=cfg.crop_tmin, tmax=cfg.crop_tmax, include_tmax=False)
    except TypeError:
        epochs.crop(tmin=cfg.crop_tmin, tmax=cfg.crop_tmax - (1.0 / epochs.info["sfreq"]))

    # baseline
    try:
        epochs.apply_baseline(baseline=(cfg.baseline_tmin, cfg.baseline_tmax))
    except Exception:
        pass

    # metadata keys
    md = epochs.metadata.reset_index(drop=True).copy() if epochs.metadata is not None else pd.DataFrame(index=np.arange(len(epochs)))
    md = ensure_keys_in_metadata(md, tt)
    epochs.metadata = md  # length一致

    # features
    md_erp = compute_erp_features(epochs, cfg.erp_windows_ms)
    rois = make_rois_from_19ch(epochs.ch_names)
    md_tfr = compute_tfr_features(
        epochs,
        rois=rois,
        bands=cfg.bands,
        windows_ms=cfg.tfr_windows_ms,
        baseline=(cfg.baseline_tmin, cfg.baseline_tmax),
    )

    df = md_erp.copy()
    # add TFR cols (一括代入)
    tfr_cols = [c for c in md_tfr.columns if c.startswith("TFR_")]
    if tfr_cols:
        df = pd.concat([df, md_tfr[tfr_cols].reset_index(drop=True)], axis=1)

    # attach trial_table category (強結合)
    df = df.merge(tt, on=["subject_id","run_id","trial_in_run","number"], how="left")

    # attach master targets
    keep = ["number"]
    for c in ["emo_arousal","emo_approach","emo_valence","PC1_emotion","PC2_emotion","PC3_emotion","is_ambiguous_approach_sd_top10"]:
        if c in ms.columns:
            keep.append(c)
    df = df.merge(ms[keep].drop_duplicates("number"), on="number", how="left")
    df = add_highlow_and_ambiguous(df)
    df = normalize_category_column(df)
    df.to_csv(cfg.trial_features_csv_out, index=False)
    logging.info(f"[SAVE] built trial_features: {cfg.trial_features_csv_out}")

    return df


# ============================================================
# Task specs
# ============================================================

@dataclass
class TaskSpec:
    name: str
    kind: str  # "regression" | "binary" | "multiclass"
    target_col: str
    n_classes: Optional[int] = None

def build_task_specs(df: pd.DataFrame, tasks: List[str]) -> List[TaskSpec]:
    specs: List[TaskSpec] = []
    for t in tasks:
        if t in ["emo_arousal","emo_approach","emo_valence","PC1_emotion","PC2_emotion","PC3_emotion"]:
            if t in df.columns:
                specs.append(TaskSpec(name=t, kind="regression", target_col=t))
        elif t.endswith("_high") or t == "is_ambiguous":
            if t in df.columns:
                specs.append(TaskSpec(name=t, kind="binary", target_col=t))
        elif t == "category_3":
            if "カテゴリー" in df.columns:
                specs.append(TaskSpec(name="category_3", kind="multiclass", target_col="カテゴリー", n_classes=3))
    return specs

def encode_category_3(y: pd.Series) -> np.ndarray:
    # 不快=0, 中間=1, 快=2
    m = {"不快":0, "中間":1, "快":2}
    yy = y.astype(str).map(m)
    return pd.to_numeric(yy, errors="coerce").to_numpy(dtype=float)


# ============================================================
# Models (robust across sklearn versions)
# ============================================================

def make_logreg(**kwargs) -> LogisticRegression:
    sig = inspect.signature(LogisticRegression)
    clean = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            clean[k] = v
    return LogisticRegression(**clean)

def make_mlp_reg(seed: int) -> Any:
    if not _HAS_MLP:
        return None
    return MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        random_state=seed_u32(seed),
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=20,
    )

def make_mlp_clf(seed: int) -> Any:
    if not _HAS_MLP:
        return None
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        random_state=seed_u32(seed),
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=20,
    )


# ============================================================
# LOSO core
# ============================================================

@dataclass
class FoldResult:
    subject_id: int
    score_primary: float
    score_aux: float
    n_test: int

def _fit_predict_score_fold_linear(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    kind: str,
    rng_seed: int,
) -> Tuple[float, float, np.ndarray]:
    """
    primary:
      - regression: Pearson r
      - binary: AUC
      - multiclass: balanced accuracy
    aux:
      - regression: R2
      - multiclass: accuracy
      - binary: nan
    coef importance:
      - regression: coef
      - binary: coef
      - multiclass: L2 norm across classes
    """
    if kind == "regression":
        model = RidgeCV(alphas=np.logspace(-3, 3, 13))
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        r = pearsonr_safe(pred, y_te)
        r2 = float(r2_score(y_te, pred)) if np.isfinite(pred).all() else np.nan
        coef = pipe.named_steps["model"].coef_.astype(float).ravel()
        return float(r), float(r2), coef

    if kind == "binary":
        model = make_logreg(
            max_iter=4000,
            solver="liblinear",
            class_weight="balanced",
            random_state=seed_u32(rng_seed),
        )
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1]
        try:
            auc = float(roc_auc_score(y_te, prob))
        except Exception:
            auc = np.nan
        coef = pipe.named_steps["model"].coef_.astype(float).ravel()
        return float(auc), float("nan"), coef

    # multiclass
    model = make_logreg(
        max_iter=5000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=seed_u32(rng_seed),
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    bacc = float(balanced_accuracy_score(y_te, pred))
    acc = float(accuracy_score(y_te, pred))
    coef = pipe.named_steps["model"].coef_.astype(float)  # (K,F)
    coef_vec = np.linalg.norm(coef, axis=0)
    return bacc, acc, coef_vec

def _fit_predict_score_fold_mlp(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    kind: str,
    rng_seed: int,
) -> Tuple[float, float]:
    """
    MLPは“AIっぽさ用の補助結果”。
    - regression: primary=r, aux=R2
    - binary: primary=AUC, aux=nan
    - multiclass: primary=bAcc, aux=acc
    """
    if not _HAS_MLP:
        return np.nan, np.nan

    if kind == "regression":
        model = make_mlp_reg(rng_seed)
        if model is None:
            return np.nan, np.nan
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        r = pearsonr_safe(pred, y_te)
        r2 = float(r2_score(y_te, pred)) if np.isfinite(pred).all() else np.nan
        return float(r), float(r2)

    if kind == "binary":
        model = make_mlp_clf(rng_seed)
        if model is None:
            return np.nan, np.nan
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1]
        try:
            auc = float(roc_auc_score(y_te, prob))
        except Exception:
            auc = np.nan
        return float(auc), float("nan")

    model = make_mlp_clf(rng_seed)
    if model is None:
        return np.nan, np.nan
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    bacc = float(balanced_accuracy_score(y_te, pred))
    acc = float(accuracy_score(y_te, pred))
    return float(bacc), float(acc)

def loso_evaluate_linear(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    kind: str,
    feature_names: List[str],
    seed: int,
) -> Tuple[List[FoldResult], pd.DataFrame]:
    logo = LeaveOneGroupOut()
    fold_rows: List[FoldResult] = []

    imp_accum = np.zeros(len(feature_names), dtype=float)
    imp_n = 0

    for tr_idx, te_idx in logo.split(X, y, groups=groups):
        sid = int(groups[te_idx][0])
        rs = seed_u32(seed + sid)

        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # validity check
        if kind in ["binary", "multiclass"]:
            if np.unique(y_tr).size < 2:
                fold_rows.append(FoldResult(subject_id=sid, score_primary=np.nan, score_aux=np.nan, n_test=int(len(te_idx))))
                continue
            if kind == "binary" and np.unique(y_te).size < 2:
                primary, aux, coef = _fit_predict_score_fold_linear(X_tr, y_tr, X_te, y_te, kind, rs)
                primary = np.nan
            else:
                primary, aux, coef = _fit_predict_score_fold_linear(X_tr, y_tr, X_te, y_te, kind, rs)
        else:
            primary, aux, coef = _fit_predict_score_fold_linear(X_tr, y_tr, X_te, y_te, kind, rs)

        fold_rows.append(FoldResult(subject_id=sid, score_primary=float(primary), score_aux=float(aux), n_test=int(len(te_idx))))
        if np.all(np.isfinite(coef)):
            imp_accum += np.abs(coef)
            imp_n += 1

    imp = (imp_accum / max(1, imp_n)).astype(float)
    imp_df = pd.DataFrame({"feature": feature_names, "importance_abscoef": imp})
    imp_df = imp_df.sort_values("importance_abscoef", ascending=False).reset_index(drop=True)

    return fold_rows, imp_df

def loso_evaluate_mlp(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    kind: str,
    seed: int,
) -> List[FoldResult]:
    logo = LeaveOneGroupOut()
    fold_rows: List[FoldResult] = []
    for tr_idx, te_idx in logo.split(X, y, groups=groups):
        sid = int(groups[te_idx][0])
        rs = seed_u32(seed + 100000 + sid)

        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        if kind in ["binary", "multiclass"] and np.unique(y_tr).size < 2:
            fold_rows.append(FoldResult(subject_id=sid, score_primary=np.nan, score_aux=np.nan, n_test=int(len(te_idx))))
            continue
        if kind == "binary" and np.unique(y_te).size < 2:
            primary, aux = _fit_predict_score_fold_mlp(X_tr, y_tr, X_te, y_te, kind, rs)
            primary = np.nan
        else:
            primary, aux = _fit_predict_score_fold_mlp(X_tr, y_tr, X_te, y_te, kind, rs)

        fold_rows.append(FoldResult(subject_id=sid, score_primary=float(primary), score_aux=float(aux), n_test=int(len(te_idx))))
    return fold_rows


# ============================================================
# Permutation
# ============================================================

def permute_y(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator, within_subject: bool) -> np.ndarray:
    y2 = y.copy()
    if within_subject:
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            y2[idx] = rng.permutation(y2[idx])
    else:
        y2 = rng.permutation(y2)
    return y2

def permutation_test_mean_score_linear(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    kind: str,
    feature_names: List[str],
    seed: int,
    n_perm: int,
    within_subject: bool,
    n_jobs: int,
    obs_mean: Optional[float] = None,   # ★追加
) -> Tuple[float, np.ndarray, float]:
    """
    returns: (p_perm, null, obs)
    obs_mean を渡せば、観測値のためのLOSO再計算をスキップ（高速化）
    """
    if obs_mean is None:
        folds, _ = loso_evaluate_linear(X, y, groups, kind, feature_names, seed)
        obs = float(np.nanmean([f.score_primary for f in folds]))
    else:
        obs = float(obs_mean)

    def one_perm(i: int) -> float:
        rng = np.random.default_rng(seed_u32(seed + 20000 + i))
        yp = permute_y(y, groups, rng, within_subject)
        fr, _ = loso_evaluate_linear(X, yp, groups, kind, feature_names, seed_u32(seed + 50000 + i))
        return float(np.nanmean([f.score_primary for f in fr]))

    null = Parallel(n_jobs=n_jobs)(delayed(one_perm)(i) for i in range(int(n_perm)))
    null = np.asarray(null, dtype=float)
    p = float((1 + np.sum(null >= obs)) / (1 + len(null)))
    return p, null, obs




# ============================================================
# Maps: feature subsets
# ============================================================

def select_erp_window_features(df: pd.DataFrame, t0: int, t1: int) -> List[str]:
    suffix = f"_{t0}_{t1}ms"
    return [c for c in df.columns if c.startswith("ERP_") and c.endswith(suffix)]

def select_tfr_band_window_features(df: pd.DataFrame, band_name: str, t0: int, t1: int) -> List[str]:
    prefix = f"TFR_{band_name}_"
    suffix = f"_{t0}_{t1}ms"
    return [c for c in df.columns if c.startswith(prefix) and c.endswith(suffix)]

# ============================================================
# Plotting (optional)
# ============================================================

def plot_loso_subject_points(task_name: str, kind: str, fold_df: pd.DataFrame, out_png: Path) -> None:
    y = fold_df["score_primary"].to_numpy(dtype=float)
    plt.figure()
    plt.scatter(np.arange(len(y)) + 1, y)
    plt.axhline(float(np.nanmean(y)), linestyle="--")
    plt.title(f"LOSO performance: {task_name} ({kind})")
    plt.xlabel("Subject (fold index)")
    plt.ylabel("AUC" if kind=="binary" else ("bAcc" if kind=="multiclass" else "Pearson r"))
    plt.grid(True)
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_top_importances(task_name: str, imp_df: pd.DataFrame, out_png: Path, top_k: int = 20) -> None:
    d = imp_df.head(top_k).copy().iloc[::-1]
    plt.figure()
    plt.barh(d["feature"].astype(str), d["importance_abscoef"].to_numpy(dtype=float))
    plt.title(f"Feature importance (|coef| mean): {task_name} top{top_k}")
    plt.xlabel("mean |coef|")
    plt.tight_layout()
    ensure_dir(out_png.parent)
    plt.savefig(out_png, dpi=200)
    plt.close()


# ============================================================
# Prepare X/y
# ============================================================

def prepare_xy(df: pd.DataFrame, spec: TaskSpec, features: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    base_cols = ["subject_id", "run_id", "trial_in_run", "number", spec.target_col]
    sub = df[base_cols + features].copy()
    sub = sub.dropna()

    sub["subject_id"] = to_int_series(sub["subject_id"])
    groups = sub["subject_id"].astype(int).to_numpy()
    X = sub[features].to_numpy(dtype=float)

    if spec.kind == "regression":
        y = pd.to_numeric(sub[spec.target_col], errors="coerce").to_numpy(dtype=float)
    elif spec.kind == "binary":
        y = pd.to_numeric(sub[spec.target_col], errors="coerce").astype(int).to_numpy()
    else:
        y0 = encode_category_3(sub[spec.target_col])
        ok = np.isfinite(y0)
        sub = sub.loc[ok].copy()
        X = X[ok]
        groups = groups[ok]
        y = y0[ok].astype(int)

    return X, y, groups, sub


# ============================================================
# Run one task (main results + optional maps + optional AI model)
# ============================================================

def run_task(cfg: Config, df: pd.DataFrame, spec: TaskSpec) -> None:
    all_feats = eeg_feature_cols(df)
    if len(all_feats) < 2:
        logging.warning(f"[task:{spec.name}] not enough EEG features -> skip")
        return

    X, y, groups, sub = prepare_xy(df, spec, all_feats)
    if len(np.unique(groups)) < 2:
        logging.warning(f"[task:{spec.name}] too few subjects -> skip")
        return
    if spec.kind in ["binary","multiclass"] and np.unique(y).size < 2:
        logging.warning(f"[task:{spec.name}] single class -> skip")
        return

    # ---------- Linear (PRIMARY) ----------
    folds, imp_df = loso_evaluate_linear(X, y, groups, spec.kind, all_feats, cfg.seed)
    fold_df = pd.DataFrame([{
        "task": spec.name,
        "kind": spec.kind,
        "model": "linear",
        "subject_id": f.subject_id,
        "score_primary": f.score_primary,
        "score_aux": f.score_aux,
        "n_test": f.n_test,
    } for f in folds])
    obs_mean = float(np.nanmean(fold_df["score_primary"].to_numpy(dtype=float)))

    p_perm, null, _obs = permutation_test_mean_score_linear(
        X, y, groups, spec.kind, all_feats,
        seed=cfg.seed,
        n_perm=cfg.n_perm,
        within_subject=cfg.perm_within_subject,
        n_jobs=cfg.n_jobs,
        obs_mean=obs_mean,  # ★ここ
    )



    # save main tables
    fold_df.to_csv(cfg.tables_dir / f"moduleB_LOSO_folds_{spec.name}_linear.csv", index=False)
    imp_df.to_csv(cfg.tables_dir / f"moduleB_importance_{spec.name}_linear.csv", index=False)
    np.save(cfg.tables_dir / f"moduleB_null_{spec.name}_linear.npy", null)

    summary = pd.DataFrame([{
        "task": spec.name,
        "kind": spec.kind,
        "model": "linear",
        "n_trials": int(len(sub)),
        "n_subjects": int(np.unique(groups).size),
        "n_features": int(len(all_feats)),
        "score_mean": obs_mean,
        "p_perm": float(p_perm),
        "n_perm": int(cfg.n_perm),
        "perm_within_subject": bool(cfg.perm_within_subject),
        "apply_qc": bool(cfg.apply_qc),
        "with_maps": bool(cfg.with_maps),
    }])
    summary.to_csv(cfg.tables_dir / f"moduleB_summary_{spec.name}_linear.csv", index=False)

    logging.info(f"[task:{spec.name}] LINEAR mean={obs_mean:.4f} p_perm={p_perm:.4f}")

    # optional figures (軽いので主結果でも出してOK)
    plot_loso_subject_points(spec.name, spec.kind, fold_df, cfg.figures_dir / f"FIG1_LOSO_{spec.name}_linear.png")
    plot_top_importances(spec.name, imp_df, cfg.figures_dir / f"FIG3_importance_{spec.name}_linear.png", top_k=20)

    # ---------- Optional: MLP (AIっぽい補助) ----------
    if cfg.model_family == "linear+mlp":
        if not _HAS_MLP:
            logging.warning("[MLP] sklearn MLP not available -> skip")
        else:
            folds_mlp = loso_evaluate_mlp(X, y, groups, spec.kind, cfg.seed)
            fold_mlp_df = pd.DataFrame([{
                "task": spec.name,
                "kind": spec.kind,
                "model": "mlp",
                "subject_id": f.subject_id,
                "score_primary": f.score_primary,
                "score_aux": f.score_aux,
                "n_test": f.n_test,
            } for f in folds_mlp])
            mlp_mean = float(np.nanmean(fold_mlp_df["score_primary"].to_numpy(dtype=float)))
            fold_mlp_df.to_csv(cfg.tables_dir / f"moduleB_LOSO_folds_{spec.name}_mlp.csv", index=False)
            pd.DataFrame([{
                "task": spec.name,
                "kind": spec.kind,
                "model": "mlp",
                "n_trials": int(len(sub)),
                "n_subjects": int(np.unique(groups).size),
                "score_mean": mlp_mean,
                "note": "AI-like supplementary (not primary). No permutation by default.",
            }]).to_csv(cfg.tables_dir / f"moduleB_summary_{spec.name}_mlp.csv", index=False)
            logging.info(f"[task:{spec.name}] MLP (supp) mean={mlp_mean:.4f}")

        # ---------- Optional maps ----------
    if not cfg.with_maps:
        return
    if cfg.map_tasks and (spec.name not in cfg.map_tasks):
        return

    logging.info(f"[maps] task={spec.name} enabled | map_tasks={cfg.map_tasks or 'ALL'} | n_perm_map={cfg.n_perm_map}")
    logging.info(f"[maps] eeg_feature example: {all_feats[:5]}")

    # ERP window map
    erp_rows = []
    pvals = []

    for (t0, t1) in cfg.erp_windows_ms:
        feats = select_erp_window_features(df, t0, t1)
        if len(feats) < 2:
            logging.info(f"[maps-ERP] window {t0}-{t1}: feats={len(feats)} (skip)")
            continue

        Xw, yw, gw, _subw = prepare_xy(df, spec, feats)
        if len(np.unique(gw)) < 2:
            continue
        if spec.kind in ["binary","multiclass"] and np.unique(yw).size < 2:
            continue

        fr, _ = loso_evaluate_linear(Xw, yw, gw, spec.kind, feats, cfg.seed)
        mean_score = float(np.nanmean([f.score_primary for f in fr]))

        local_seed = seed_u32(cfg.seed + 1000 + t0 + 17)
        p_perm_w, _null_w, _obs_w = permutation_test_mean_score_linear(
            Xw, yw, gw, spec.kind, feats,
            seed=local_seed,
            n_perm=cfg.n_perm_map,
            within_subject=cfg.perm_within_subject,
            n_jobs=cfg.n_jobs,
            obs_mean=mean_score,  # ★入れておくと高速化にもなる
        )


        erp_rows.append({
            "task": spec.name,
            "window": f"{t0}-{t1}",
            "t0_ms": t0,
            "t1_ms": t1,
            "score": mean_score,
            "p_perm": float(p_perm_w),
            "n_perm": int(cfg.n_perm_map),
            "n_features": int(len(feats)),
        })
        pvals.append(float(p_perm_w))

    if erp_rows:
        erp_map = pd.DataFrame(erp_rows).sort_values(["t0_ms","t1_ms"]).reset_index(drop=True)
        rej, q = fdr_bh(np.asarray(pvals, dtype=float), alpha=0.05)
        erp_map["q_fdr"] = q
        erp_map["significant_fdr"] = rej
        erp_map.to_csv(cfg.tables_dir / f"moduleB_map_ERP_{spec.name}.csv", index=False)

    # TFR band × time map
    tfr_rows = []
    cell_p = []

    for (bname, _f0, _f1) in cfg.bands:
        bseed = stable_hash_u32(bname)
        for (t0, t1) in cfg.tfr_windows_ms:
            feats = select_tfr_band_window_features(df, bname, t0, t1)
            if len(feats) < 2:
                logging.info(f"[maps-TFR] {bname} {t0}-{t1}: feats={len(feats)} (skip)")
                continue

            Xw, yw, gw, _subw = prepare_xy(df, spec, feats)
            if len(np.unique(gw)) < 2:
                continue
            if spec.kind in ["binary","multiclass"] and np.unique(yw).size < 2:
                continue

            fr, _ = loso_evaluate_linear(Xw, yw, gw, spec.kind, feats, cfg.seed)
            mean_score = float(np.nanmean([f.score_primary for f in fr]))

            local_seed = seed_u32(cfg.seed + 2000 + t0 + int(bseed % 10000))
            p_perm_w, _null_w, _obs_w = permutation_test_mean_score_linear(
                Xw, yw, gw, spec.kind, feats,
                seed=local_seed,
                n_perm=cfg.n_perm_map,
                within_subject=cfg.perm_within_subject,
                n_jobs=cfg.n_jobs,
                obs_mean=mean_score,
            )


            tfr_rows.append({
                "task": spec.name,
                "band": bname,
                "window": f"{t0}-{t1}",
                "t0_ms": t0,
                "t1_ms": t1,
                "score": mean_score,
                "p_perm": float(p_perm_w),
                "n_perm": int(cfg.n_perm_map),
                "n_features": int(len(feats)),
            })
            cell_p.append(float(p_perm_w))

    if tfr_rows:
        tfr_map = pd.DataFrame(tfr_rows).sort_values(["band","t0_ms","t1_ms"]).reset_index(drop=True)
        rej, q = fdr_bh(np.asarray(cell_p, dtype=float), alpha=0.05)
        tfr_map["q_fdr"] = q
        tfr_map["significant_fdr"] = rej
        tfr_map.to_csv(cfg.tables_dir / f"moduleB_map_TFR_{spec.name}.csv", index=False)


# ============================================================
# Metadata / snapshot
# ============================================================

def save_run_metadata(cfg: Config, df: pd.DataFrame, trial_features_in: Optional[Path]) -> None:
    meta = {
        "module": "ModuleB_BF_EEGcore_final",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "root_dir": str(cfg.root_dir),
        "inputs": {
            "trial_features_csv_in": str(trial_features_in) if trial_features_in else "",
            "epochs_all": str(cfg.epochs_all_path),
            "trial_table": str(cfg.trial_table_csv),
            "master_sound_pc": str(cfg.master_sound_pc_csv),
            "qc_by_trial": str(cfg.qc_by_trial_csv) if cfg.apply_qc else "",
        },
        "outputs": {
            "out_dir": str(cfg.out_dir),
            "tables": str(cfg.tables_dir),
            "figures": str(cfg.figures_dir),
            "logs": str(cfg.logs_dir),
        },
        "params": {
            "seed": cfg.seed,
            "n_jobs": cfg.n_jobs,
            "n_perm": cfg.n_perm,
            "n_perm_map": cfg.n_perm_map,
            "perm_within_subject": cfg.perm_within_subject,
            "apply_qc": cfg.apply_qc,
            "with_maps": cfg.with_maps,
            "map_tasks": cfg.map_tasks,
            "resample_sfreq": cfg.resample_sfreq,
            "crop_tmin": cfg.crop_tmin,
            "crop_tmax": cfg.crop_tmax,
            "baseline": [cfg.baseline_tmin, cfg.baseline_tmax],
            "erp_windows_ms": cfg.erp_windows_ms,
            "tfr_windows_ms": cfg.tfr_windows_ms,
            "bands": cfg.bands,
            "tasks": cfg.tasks,
            "model_family": cfg.model_family,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": getattr(np, "__version__", "unknown"),
            "pandas": getattr(pd, "__version__", "unknown"),
            "mne": getattr(mne, "__version__", "unknown"),
        },
        "data_snapshot": {
            "n_rows": int(len(df)),
            "n_subjects": int(df["subject_id"].nunique()) if "subject_id" in df.columns else None,
            "n_sounds": int(df["number"].nunique()) if "number" in df.columns else None,
            "n_eeg_features": int(len(eeg_feature_cols(df))),
            "has_category": bool("カテゴリー" in df.columns),
            "target_cols_present": [c for c in ["emo_arousal","emo_approach","emo_valence","is_ambiguous","category_3"] if (c in df.columns or (c=="category_3" and "カテゴリー" in df.columns))],
        },
    }
    json_dump(cfg.run_metadata_json, meta)
    logging.info(f"[SAVE] run metadata: {cfg.run_metadata_json}")


# ============================================================
# Main
# ============================================================

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    setup_matplotlib(auto_font=not args.no_auto_font)

    # logging needs cfg.logs_dir; first make cfg, then setup logging
    cfg = make_config(args)

    log_path = cfg.logs_dir / f"moduleB_{now_tag()}.log"
    setup_logging(log_path)

    t0 = time.time()
    logging.info("=== Module B (BF: EEG本体 / B+F統合) start ===")
    logging.info(f"ROOT_DIR     : {cfg.root_dir}")
    logging.info(f"OUT_DIR      : {cfg.out_dir}")
    logging.info(f"trial_table  : {cfg.trial_table_csv}")
    logging.info(f"master_sound : {cfg.master_sound_pc_csv}")
    logging.info(f"epochs_all   : {cfg.epochs_all_path}")
    logging.info(f"qc_by_trial  : {cfg.qc_by_trial_csv if cfg.apply_qc else ''}")
    logging.info(f"seed={cfg.seed} n_perm={cfg.n_perm} n_perm_map={cfg.n_perm_map} n_jobs={cfg.n_jobs} perm_within={cfg.perm_within_subject}")
    logging.info(f"crop         : {cfg.crop_tmin}..{cfg.crop_tmax} sec | ERP/TFR baseline=({cfg.baseline_tmin},{cfg.baseline_tmax})")
    logging.info(f"erp_windows_ms={cfg.erp_windows_ms}")
    logging.info(f"tfr_windows_ms={cfg.tfr_windows_ms}")
    logging.info(f"bands={cfg.bands}")
    logging.info(f"tasks={cfg.tasks}")
    logging.info(f"with_maps={cfg.with_maps} map_tasks={cfg.map_tasks if cfg.map_tasks else 'ALL'}")
    logging.info(f"model_family={cfg.model_family}")
    logging.info(f"log          : {log_path}")

    trial_features_in = Path(args.trial_features_csv).expanduser() if args.trial_features_csv else None
    if trial_features_in and not trial_features_in.exists():
        logging.warning(f"[input] trial_features_csv not found -> build from epochs. path={trial_features_in}")
        trial_features_in = None

    # build/load
    df = load_or_build_features(cfg, trial_features_in, reuse_built=bool(args.reuse_built))

    # QC
    if cfg.apply_qc:
        qc = load_qc_by_trial(cfg)
        df = apply_qc_filter(df, qc)

    df = normalize_category_column(df)

    logging.info(f"[DATA] shape={df.shape} subjects={df['subject_id'].nunique()} sounds={df['number'].nunique()} eeg_features={len(eeg_feature_cols(df))}")

    # save metadata early
    save_run_metadata(cfg, df, trial_features_in)

    cat_cols = [c for c in df.columns if ("カテゴリ" in c) or (c.lower() == "category")]
    logging.info(f"[CAT] category-like cols={cat_cols} | has_カテゴリー={'カテゴリー' in df.columns}")

    # tasks
    specs = build_task_specs(df, cfg.tasks)
    if not specs:
        raise RuntimeError("No valid tasks found (targets/labels missing).")

    for spec in specs:
        logging.info(f"--- Task: {spec.name} ({spec.kind}) ---")
        run_task(cfg, df, spec)

    logging.info("=== Module B complete ===")
    logging.info(f"- tables : {cfg.tables_dir}")
    logging.info(f"- figures: {cfg.figures_dir}")
    logging.info(f"- log    : {log_path}")
    logging.info(f"- elapsed: {time.time() - t0:.1f} sec")


if __name__ == "__main__":
    main()


# In[ ]:




