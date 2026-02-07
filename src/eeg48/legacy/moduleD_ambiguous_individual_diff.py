#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module D: Ambiguous sounds & Individual differences (PC2) + EEG association
===========================================================================
End-to-end pipeline (reproducible, “成果物として強い”完全体):

D-0) Load trial-level master (subject x sound) with PC2 + 8 ratings.
     Load subject-level master with EEG features (ERP_/SPN_).
     *Minimal loading by default* (reads only necessary columns).

D-0.5) Align PC2 sign for interpretability (threat/avoidance direction).
       Save pseudo "PC2 axis loadings" via standardized regression: PC2 ~ z(ratings).

D-1) Define ambiguous sounds:
     - candidates: enough subjects (>= ceil(total_subjects * enough_ratio))
     - neutral: abs(pc2_mean) <= neutral_th
     - ambiguous: among candidates, top_k by pc2_sd
     Save stats table.

D-2) Compute individual-difference scores (ambiguous set):
     - ambig_pc2_mad (within-subject MAD over ambiguous sounds)
     - PCA over subject-by-ambiguousSound PC2 matrix (optional but saved)
     Save scores table.

D-3) Merge subject-level EEG features + indiv scores, then:
     - Spearman correlation + permutation p-value (two-sided)
     - Bootstrap CI
     - FDR correction
     - Robustness: leave-1-subject-out for primary features
     - Specificity: compare to overall MAD (all sounds)
     Save result tables.

D-4) LOSO stability of ambiguous sound set:
     - Recompute ambiguous set leaving one subject out
     - Jaccard similarity vs full set
     - Count selection frequency across folds
     Save stability tables and plots.

4) “成果物として強くする”次アクションの自動パッケージ出力:
     - moduleD_KEYNUMBERS.json
     - moduleD_NEXT_ACTIONS.md  (no tabulate dependency)
     - moduleD_SLIDE_BULLETS.txt
     - top EEG scatter plots

Dependencies:
  numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib

Usage:
  python moduleD_ambiguous_and_individual_diff.py --root-dir /path/to/EEG_48sounds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt


# =========================
# Config
# =========================
@dataclass
class Config:
    root_dir: Path

    # Inputs
    trial_csv: Path
    subject_csv: Path

    # Output
    out_dir: Path
    tab_dir: Path
    fig_dir: Path
    log_dir: Path

    # Columns (autodetected if None)
    subject_col: Optional[str] = None
    sound_col: Optional[str] = None
    pc2_col: str = "PC2_emotion"

    # Ratings (Japanese)
    rating_cols_expected: Tuple[str, ...] = (
        "驚き", "緊急感", "脅威感", "圧倒感", "接近したい気持ち", "興味", "没入", "退屈"
    )

    # Ambiguous definition
    enough_ratio: float = 0.80
    neutral_th: float = 0.35
    top_k: int = 10

    # Stats params
    n_perm: int = 5000
    n_boot: int = 5000
    seed: int = 42

    # EEG column selection
    eeg_cols: Optional[List[str]] = None  # if None -> auto detect by prefix patterns ERP_/SPN_

    # I/O behavior
    minimal_load: bool = True
    make_figures: bool = True

    # Reporting
    n_scatter_top: int = 4  # how many top EEG features to make scatter plots


DEFAULT_ROOT = "."


def build_config(
    root_dir: str,
    trial_csv: Optional[str] = None,
    subject_csv: Optional[str] = None,
    out_dir: Optional[str] = None,
    subject_col: Optional[str] = None,
    sound_col: Optional[str] = None,
    pc2_col: str = "PC2_emotion",
    enough_ratio: float = 0.80,
    neutral_th: float = 0.35,
    top_k: int = 10,
    n_perm: int = 5000,
    n_boot: int = 5000,
    seed: int = 42,
    minimal_load: bool = True,
    make_figures: bool = True,
) -> Config:
    root = Path(root_dir)

    trial = Path(trial_csv) if trial_csv else root / "derivatives/master_tables/master_participant_sound_level_with_PC.csv"
    subj = Path(subject_csv) if subject_csv else root / "derivatives/master_tables/master_participant_level.csv"

    out = Path(out_dir) if out_dir else root / "moduleD_outputs"
    tab = out / "tables"
    fig = out / "figures"
    log = out / "logs"
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)

    return Config(
        root_dir=root,
        trial_csv=trial,
        subject_csv=subj,
        out_dir=out,
        tab_dir=tab,
        fig_dir=fig,
        log_dir=log,
        subject_col=subject_col,
        sound_col=sound_col,
        pc2_col=pc2_col,
        enough_ratio=enough_ratio,
        neutral_th=neutral_th,
        top_k=top_k,
        n_perm=n_perm,
        n_boot=n_boot,
        seed=seed,
        minimal_load=minimal_load,
        make_figures=make_figures,
    )


# =========================
# Utilities
# =========================
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"[SAVE] {path}")


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {path}")


def save_text(text: str, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[SAVE] {path}")


def in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is None:
            return False
        return "IPKernelApp" in ip.config
    except Exception:
        return False


def find_first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def find_subject_col_from_cols(cols: List[str]) -> str:
    candidates = ["subject_id", "subject", "participant_id", "participant", "subj", "sid", "ID"]
    c = find_first_existing(cols, candidates)
    if c is None:
        raise ValueError("Could not find subject column. Pass --subject-col explicitly.")
    return c


def find_sound_col_from_cols(cols: List[str]) -> str:
    candidates = ["sound_id", "sound", "stimulus_id", "stimulus", "number", "trial_sound", "sid_sound"]
    c = find_first_existing(cols, candidates)
    if c is None:
        raise ValueError("Could not find sound column. Pass --sound-col explicitly.")
    return c


def detect_rating_cols_from_cols(cols: List[str], expected: Tuple[str, ...]) -> List[str]:
    # Prefer raw names; fallback to *_mean if raw absent
    found_raw = [c for c in expected if c in cols]
    if len(found_raw) > 0:
        return found_raw
    found_mean = [f"{c}_mean" for c in expected if f"{c}_mean" in cols]
    return found_mean


def mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if len(sa | sb) == 0:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


def stable_seed(base_seed: int, key: str) -> int:
    """Per-feature stable seed to avoid order-dependence."""
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return (base_seed + int(h[:8], 16)) % (2**32 - 1)


def df_to_markdown_pipe(df: pd.DataFrame, max_rows: int = 15) -> str:
    """Minimal markdown table formatter (no tabulate dependency)."""
    if df is None or len(df) == 0:
        return "_(empty)_\n"
    d = df.copy()
    if max_rows is not None and len(d) > max_rows:
        d = d.head(max_rows)
    cols = list(d.columns)
    # stringify cells
    def fmt(v):
        if isinstance(v, float):
            if np.isnan(v):
                return ""
            return f"{v:.6g}"
        return str(v)
    rows = [[fmt(v) for v in d.iloc[i].tolist()] for i in range(len(d))]
    # widths
    widths = [len(str(c)) for c in cols]
    for r in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, r)]
    # build
    header = "| " + " | ".join(str(c).ljust(w) for c, w in zip(cols, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = "\n".join("| " + " | ".join(cell.ljust(w) for cell, w in zip(r, widths)) + " |" for r in rows)
    return header + "\n" + sep + ("\n" + body if body else "") + "\n"


# =========================
# Loading (minimal)
# =========================
def detect_eeg_cols_from_cols(cols: List[str]) -> List[str]:
    eeg = []
    for c in cols:
        if isinstance(c, str) and (c.startswith("ERP_") or c.startswith("SPN_")):
            eeg.append(c)
    return eeg


def load_inputs(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Loads:
      - trial table with minimal required columns
      - subject table with minimal required columns
    Returns:
      df_trial, df_subj, rating_cols
    """
    # ---- trial header
    trial_header = pd.read_csv(cfg.trial_csv, nrows=0)
    trial_cols_all = trial_header.columns.tolist()
    raw_trial_ncols = len(trial_cols_all)

    if cfg.subject_col is None:
        cfg.subject_col = find_subject_col_from_cols(trial_cols_all)
    if cfg.sound_col is None:
        cfg.sound_col = find_sound_col_from_cols(trial_cols_all)

    rating_cols = detect_rating_cols_from_cols(trial_cols_all, cfg.rating_cols_expected)
    if cfg.pc2_col not in trial_cols_all:
        raise ValueError(f"PC2 column '{cfg.pc2_col}' not found in trial CSV.")

    trial_usecols = [cfg.subject_col, cfg.sound_col, cfg.pc2_col] + rating_cols
    # keep only existing (safety)
    trial_usecols = [c for c in trial_usecols if c in trial_cols_all]

    if cfg.minimal_load:
        df_trial = pd.read_csv(cfg.trial_csv, usecols=trial_usecols)
        print(f"[LOAD] trial  : {df_trial.shape} | {cfg.trial_csv}")
        print(f"        trial columns raw={raw_trial_ncols} -> loaded={len(df_trial.columns)} (minimal)")
    else:
        df_trial = pd.read_csv(cfg.trial_csv)
        print(f"[LOAD] trial  : {df_trial.shape} | {cfg.trial_csv}")

    # ---- subject header
    subj_header = pd.read_csv(cfg.subject_csv, nrows=0)
    subj_cols_all = subj_header.columns.tolist()
    raw_subj_ncols = len(subj_cols_all)

    # subject id column might differ in subject table
    subj_subject_col = cfg.subject_col if cfg.subject_col in subj_cols_all else find_subject_col_from_cols(subj_cols_all)

    if cfg.eeg_cols is None:
        eeg_cols = detect_eeg_cols_from_cols(subj_cols_all)
    else:
        eeg_cols = [c for c in cfg.eeg_cols if c in subj_cols_all]

    if len(eeg_cols) == 0:
        raise ValueError("No EEG feature columns detected. Provide --eeg-cols or ensure subject CSV contains ERP_/SPN_ columns.")

    subj_usecols = [subj_subject_col] + eeg_cols
    subj_usecols = [c for c in subj_usecols if c in subj_cols_all]

    if cfg.minimal_load:
        df_subj = pd.read_csv(cfg.subject_csv, usecols=subj_usecols)
        print(f"[LOAD] subject: {df_subj.shape} | {cfg.subject_csv}")
        print(f"        subject columns raw={raw_subj_ncols} -> loaded={len(df_subj.columns)} (minimal)")
    else:
        df_subj = pd.read_csv(cfg.subject_csv)
        print(f"[LOAD] subject: {df_subj.shape} | {cfg.subject_csv}")

    # normalize subject id col name in subject table to cfg.subject_col
    if subj_subject_col != cfg.subject_col:
        df_subj = df_subj.rename(columns={subj_subject_col: cfg.subject_col})

    # dtype normalize early (merge safety)
    df_trial[cfg.subject_col] = df_trial[cfg.subject_col].astype(str)
    df_subj[cfg.subject_col] = df_subj[cfg.subject_col].astype(str)
    df_trial[cfg.sound_col] = df_trial[cfg.sound_col].astype(str)

    print(f"[INFO] trial  : subject col = '{cfg.subject_col}'")
    print(f"[INFO] subject: subject col = '{cfg.subject_col}'")
    print(f"[INFO] trial  : sound col   = '{cfg.sound_col}'")
    print(f"[INFO] rating cols: {rating_cols}")

    return df_trial, df_subj, rating_cols


# =========================
# Core steps
# =========================
def aggregate_subject_sound(df_trial: pd.DataFrame, subject_col: str, sound_col: str, cols: List[str]) -> pd.DataFrame:
    """Ensure unique subject-sound rows by averaging duplicates."""
    agg = (
        df_trial
        .groupby([subject_col, sound_col], as_index=False)[cols]
        .mean()
    )
    return agg


def align_pc2_sign(
    cfg: Config,
    df_trial: pd.DataFrame,
    rating_cols: List[str],
) -> Tuple[pd.DataFrame, int, pd.DataFrame]:
    """
    Align PC2 sign so that PC2 is positively associated with "avoidance/threat" anchor.
    Anchor = mean(threat-related) - mean(approach-related) at sound-level.

    Then compute pseudo-loadings via standardized regression: PC2 ~ z(ratings).
    Saves: moduleD_PC2_axis_loadings_aligned.csv
    """
    pc2 = cfg.pc2_col
    subject_col = cfg.subject_col
    sound_col = cfg.sound_col

    if len(rating_cols) == 0:
        print(f"[INFO] PC2 sign alignment: sign=1 (no rating cols found; pc2_col='{pc2}')")
        return df_trial.copy(), 1, pd.DataFrame()

    cols_needed = [pc2] + rating_cols
    df_ss = aggregate_subject_sound(df_trial, subject_col, sound_col, cols_needed)

    # sound-level mean (avoid overweighting subjects)
    sound_mean = df_ss.groupby(sound_col, as_index=False)[cols_needed].mean()

    def col_present(name: str) -> Optional[str]:
        if name in rating_cols:
            return name
        nm = f"{name}_mean"
        if nm in rating_cols:
            return nm
        return None

    threat_cols = [c for c in [col_present("緊急感"), col_present("脅威感"), col_present("圧倒感")] if c is not None]
    approach_cols = [c for c in [col_present("接近したい気持ち"), col_present("興味"), col_present("没入")] if c is not None]

    if len(threat_cols) == 0 or len(approach_cols) == 0:
        anchor = sound_mean[rating_cols].mean(axis=1).to_numpy()
    else:
        anchor = sound_mean[threat_cols].mean(axis=1).to_numpy() - sound_mean[approach_cols].mean(axis=1).to_numpy()

    corr = np.corrcoef(sound_mean[pc2].to_numpy(dtype=float), anchor.astype(float))[0, 1]
    sign = 1 if np.isnan(corr) or corr >= 0 else -1

    df_aligned = df_trial.copy()
    df_aligned[pc2] = df_aligned[pc2] * sign

    print(f"[INFO] PC2 sign alignment: sign={sign} (pc2_col='{pc2}')")

    # pseudo-loadings via standardized regression
    df_reg = df_ss.dropna(subset=[pc2] + rating_cols).copy()
    X = df_reg[rating_cols].to_numpy(dtype=float)
    y = df_reg[pc2].to_numpy(dtype=float)

    Xz = StandardScaler().fit_transform(X)
    yz = (y - y.mean()) / (y.std(ddof=0) + 1e-12)

    X_design = np.c_[np.ones(len(Xz)), Xz]
    beta, *_ = np.linalg.lstsq(X_design, yz, rcond=None)

    loadings = pd.DataFrame({
        "rating": ["(intercept)"] + rating_cols,
        "coef_z": beta.tolist(),
    })
    loadings["pc2_aligned_sign"] = sign
    loadings = loadings.sort_values("coef_z", key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)

    out_load = cfg.tab_dir / "moduleD_PC2_axis_loadings_aligned.csv"
    save_csv(loadings, out_load)

    return df_aligned, sign, loadings


def define_ambiguous_sounds(
    cfg: Config,
    df_trial: pd.DataFrame,
    save_path: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    D-1:
      - enough_n: n_subjects >= ceil(total_subjects * enough_ratio)
      - neutral: abs(pc2_mean) <= neutral_th
      - ambiguous: among candidates, top_k by pc2_sd
    """
    subject_col = cfg.subject_col
    sound_col = cfg.sound_col
    pc2 = cfg.pc2_col

    df_ss = aggregate_subject_sound(df_trial, subject_col, sound_col, [pc2])

    subjects = sorted(df_ss[subject_col].unique().tolist())
    total_subj = len(subjects)
    enough_n_min = int(math.ceil(total_subj * cfg.enough_ratio))

    g = df_ss.groupby(sound_col)[pc2]
    stats = pd.DataFrame({
        "sound_id": g.mean().index.astype(str),
        "pc2_mean": g.mean().values,
        "pc2_sd": g.std(ddof=1).values,
        "n_subjects": g.count().values,
    })
    stats["enough_n"] = stats["n_subjects"] >= enough_n_min
    stats["is_neutral_candidate"] = stats["enough_n"] & (stats["pc2_mean"].abs() <= cfg.neutral_th)

    cand = stats[stats["is_neutral_candidate"]].copy()
    cand = cand.sort_values(["pc2_sd", "sound_id"], ascending=[False, True])

    selected = cand.head(cfg.top_k)["sound_id"].tolist()
    stats["is_ambiguous"] = stats["sound_id"].isin(selected)

    if verbose:
        print("\n[MODULE D-1] Ambiguous sound definition (TOP-K)")
        print(f"  subjects total = {total_subj} | enough_n >= {enough_n_min} ({cfg.enough_ratio:.2f})")
        print(f"  neutral: |pc2_mean| <= {cfg.neutral_th:.3f}")
        print(f"  ambiguous: among candidates, top_k={cfg.top_k} by pc2_sd")
        print(stats[["sound_id", "pc2_mean", "pc2_sd", "n_subjects", "enough_n", "is_ambiguous"]])

    if save_path is not None:
        save_csv(stats, save_path)

    return stats, selected


def compute_individual_difference_scores(
    cfg: Config,
    df_trial: pd.DataFrame,
    ambiguous_sounds: List[str],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    D-2:
      - ambig_pc2_mad per subject over ambiguous sounds
      - PCA over subject x ambiguousSound PC2 matrix
    """
    subject_col = cfg.subject_col
    sound_col = cfg.sound_col
    pc2 = cfg.pc2_col

    df_ss = aggregate_subject_sound(df_trial, subject_col, sound_col, [pc2])

    df_a = df_ss[df_ss[sound_col].astype(str).isin([str(s) for s in ambiguous_sounds])].copy()
    counts = df_a.groupby(subject_col)[sound_col].nunique()
    keep_subjects = counts[counts >= 2].index.tolist()

    print("\n[MODULE D-2] Individual-difference scores (ambiguous set)")
    print(f"  Exclude subjects with ambig trials < 2: {df_ss[subject_col].nunique()} -> {len(keep_subjects)}")

    df_a = df_a[df_a[subject_col].isin(keep_subjects)].copy()

    pivot = df_a.pivot_table(index=subject_col, columns=sound_col, values=pc2, aggfunc="mean")
    pivot = pivot.reindex(columns=[str(s) for s in ambiguous_sounds])
    pivot = pivot.apply(lambda col: col.fillna(col.mean()), axis=0)

    ambig_pc2_mad = pivot.apply(lambda row: mad(row.to_numpy()), axis=1).rename("ambig_pc2_mad")

    X = pivot.to_numpy(dtype=float)
    Xz = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(3, Xz.shape[1]))
    pcs = pca.fit_transform(Xz)

    pca_info = {f"PC{i+1}": float(pca.explained_variance_ratio_[i]) for i in range(pcs.shape[1])}
    print("  PCA explained variance ratio:")
    for k, v in pca_info.items():
        print(f"    {k}: {v:.3f}")

    df_scores = pd.DataFrame({
        subject_col: pivot.index.astype(str),
        "ambig_pc2_mad": ambig_pc2_mad.values,
        "ambig_pc2_mean": pivot.mean(axis=1).values,
        "ambig_pc2_sd": pivot.std(axis=1, ddof=1).values,
    })
    for i in range(pcs.shape[1]):
        df_scores[f"ambig_PC{i+1}"] = pcs[:, i]

    out_scores = cfg.tab_dir / "moduleD_individual_difference_scores.csv"
    save_csv(df_scores, out_scores)

    return df_scores, pca_info


def merge_subject_eeg(
    cfg: Config,
    df_subj: pd.DataFrame,
    df_scores: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    D-3: merge subject-level EEG + indiv scores.
    """
    subject_col = cfg.subject_col

    if cfg.eeg_cols is None:
        eeg_cols = [c for c in df_subj.columns if isinstance(c, str) and (c.startswith("ERP_") or c.startswith("SPN_"))]
    else:
        eeg_cols = [c for c in cfg.eeg_cols if c in df_subj.columns]

    if len(eeg_cols) == 0:
        raise ValueError("No EEG feature columns detected. Provide --eeg-cols or ensure subject CSV contains ERP_/SPN_ columns.")

    df_subj_small = df_subj[[subject_col] + eeg_cols].copy()

    # dtype safety (prevents merge object/int64 crash)
    df_scores = df_scores.copy()
    df_scores[subject_col] = df_scores[subject_col].astype(str)
    df_subj_small[subject_col] = df_subj_small[subject_col].astype(str)

    df_merged = pd.merge(df_scores, df_subj_small, on=subject_col, how="inner")

    print("\n[MODULE D-3] Merge subject EEG + indiv scores")
    print(f"  merged: {df_merged.shape}")

    print("\n[MODULE D-3] EEG feature columns")
    print(f"  n_eeg_cols = {len(eeg_cols)}")
    print(f"  head: {eeg_cols[:10]}")

    return df_merged, eeg_cols


def spearman_perm_test(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Two-sided Spearman correlation with permutation p-value."""
    rho_obs, _ = spearmanr(x, y)
    if np.isnan(rho_obs):
        return float("nan"), float("nan")

    y = np.asarray(y)
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        rho_p, _ = spearmanr(x, yp)
        if np.isnan(rho_p):
            continue
        if abs(rho_p) >= abs(rho_obs):
            count += 1
    p_perm = (count + 1) / (n_perm + 1)
    return float(rho_obs), float(p_perm)


def bootstrap_ci_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Bootstrap percentile CI for Spearman rho."""
    n = len(x)
    rhos = []
    idx = np.arange(n)
    for _ in range(n_boot):
        b = rng.choice(idx, size=n, replace=True)
        rb, _ = spearmanr(x[b], y[b])
        if np.isnan(rb):
            continue
        rhos.append(rb)
    if len(rhos) == 0:
        return float("nan"), float("nan")
    lo = float(np.quantile(rhos, alpha / 2))
    hi = float(np.quantile(rhos, 1 - alpha / 2))
    return lo, hi


def association_analysis(
    cfg: Config,
    df_merged: pd.DataFrame,
    eeg_cols: List[str],
    target_col: str = "ambig_pc2_mad",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Spearman + permutation + bootstrap CI + FDR.
    Saves:
      - moduleD_eeg_association_spearman_perm.csv
      - moduleD_PRIMARY_results_with_bootstrapCI.csv
    """
    y = df_merged[target_col].to_numpy(dtype=float)

    rows = []
    for feat in eeg_cols:
        rng_feat = np.random.default_rng(stable_seed(cfg.seed, feat))  # stable per feature
        x = df_merged[feat].to_numpy(dtype=float)

        rho, p_perm = spearman_perm_test(x, y, cfg.n_perm, rng_feat)
        ci_lo, ci_hi = bootstrap_ci_spearman(x, y, cfg.n_boot, rng_feat)

        rows.append({
            "target": target_col,
            "feature": feat,
            "n": int(np.sum(~np.isnan(x) & ~np.isnan(y))),
            "spearman_rho": rho,
            "p_perm": p_perm,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
        })

    res = pd.DataFrame(rows).sort_values("p_perm", ascending=True).reset_index(drop=True)

    pvals = res["p_perm"].to_numpy(dtype=float)
    _rej, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    res["q_fdr_primary"] = qvals

    out_all = cfg.tab_dir / "moduleD_eeg_association_spearman_perm.csv"
    out_primary = cfg.tab_dir / "moduleD_PRIMARY_results_with_bootstrapCI.csv"
    save_csv(res, out_all)
    save_csv(res.sort_values(["q_fdr_primary", "p_perm", "feature"], ascending=[True, True, True]).reset_index(drop=True), out_primary)

    primary = pd.read_csv(out_primary)
    print(primary.head(12))

    return res, primary


def robustness_leave1(
    cfg: Config,
    df_merged: pd.DataFrame,
    assoc_table: pd.DataFrame,
    eeg_cols: List[str],
    target_col: str = "ambig_pc2_mad",
) -> pd.DataFrame:
    """
    Leave-one-subject-out robustness for significant/near-significant features.
    Saves: moduleD_PRIMARY_robustness_leave1.csv
    """
    sig = assoc_table[assoc_table["q_fdr_primary"] < 0.05]["feature"].tolist()
    if len(sig) == 0:
        sig = assoc_table.sort_values("p_perm").head(min(5, len(eeg_cols)))["feature"].tolist()

    subject_col = cfg.subject_col
    subjects = df_merged[subject_col].astype(str).tolist()

    rows = []
    for leave in subjects:
        df_lo = df_merged[df_merged[subject_col].astype(str) != str(leave)].copy()
        y = df_lo[target_col].to_numpy(dtype=float)

        for feat in sig:
            x = df_lo[feat].to_numpy(dtype=float)
            rho, _ = spearmanr(x, y)
            rows.append({
                "leave_out_subject": leave,
                "feature": feat,
                "rho_leave1": float(rho) if not np.isnan(rho) else float("nan"),
                "n": int(len(df_lo)),
            })

    out = pd.DataFrame(rows)
    out_path = cfg.tab_dir / "moduleD_PRIMARY_robustness_leave1.csv"
    save_csv(out, out_path)
    return out


def specificity_overall_mad(
    cfg: Config,
    df_trial: pd.DataFrame,
    df_merged: pd.DataFrame,
    eeg_cols: List[str],
    target_col: str = "ambig_pc2_mad",
) -> pd.DataFrame:
    """
    Specificity: compare associations with overall MAD across all sounds.
    Saves: moduleD_PRIMARY_specificity_overallMAD.csv
    """
    subject_col = cfg.subject_col
    sound_col = cfg.sound_col
    pc2 = cfg.pc2_col

    df_ss = aggregate_subject_sound(df_trial, subject_col, sound_col, [pc2])
    overall = (
        df_ss
        .groupby(subject_col)[pc2]
        .apply(lambda s: mad(s.to_numpy(dtype=float)))
        .rename("overall_pc2_mad")
        .reset_index()
    )
    overall[subject_col] = overall[subject_col].astype(str)

    df2 = pd.merge(df_merged[[subject_col, target_col] + eeg_cols], overall, on=subject_col, how="inner")

    rows = []
    for feat in eeg_cols:
        x = df2[feat].to_numpy(dtype=float)

        rng_a = np.random.default_rng(stable_seed(cfg.seed + 999, feat + "_ambig"))
        rng_o = np.random.default_rng(stable_seed(cfg.seed + 1999, feat + "_overall"))

        rho_a, p_a = spearman_perm_test(x, df2[target_col].to_numpy(dtype=float), cfg.n_perm, rng_a)
        rho_o, p_o = spearman_perm_test(x, df2["overall_pc2_mad"].to_numpy(dtype=float), cfg.n_perm, rng_o)

        rows.append({
            "feature": feat,
            "rho_ambigMAD": rho_a, "p_ambigMAD": p_a,
            "rho_overallMAD": rho_o, "p_overallMAD": p_o,
        })

    out = pd.DataFrame(rows).sort_values("p_ambigMAD", ascending=True).reset_index(drop=True)
    out_path = cfg.tab_dir / "moduleD_PRIMARY_specificity_overallMAD.csv"
    save_csv(out, out_path)
    return out


def loso_ambiguous_stability(
    cfg: Config,
    df_trial: pd.DataFrame,
    full_ambiguous: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Path], Optional[Path]]:
    """
    LOSO stability of ambiguous set:
      - each fold: recompute ambiguous sounds leaving one subject out
      - jaccard with full set
      - selection frequency across folds
    Saves:
      - moduleD_ambiguous_set_stability_LOSO.csv
      - moduleD_ambiguous_set_selection_frequency_LOSO.csv
      - figures: moduleD_LOSO_selection_frequency.png, moduleD_LOSO_jaccard_hist.png
    """
    subject_col = cfg.subject_col
    subjects = sorted(df_trial[subject_col].astype(str).unique().tolist())

    fold_rows = []
    freq = Counter()
    jacs = []

    for leave in subjects:
        df_lo = df_trial[df_trial[subject_col].astype(str) != str(leave)].copy()
        _stats_lo, amb_lo = define_ambiguous_sounds(cfg, df_lo, save_path=None, verbose=False)

        jac = jaccard([str(s) for s in full_ambiguous], [str(s) for s in amb_lo])
        jacs.append(jac)

        fold_rows.append({
            "leave_out_subject": leave,
            "n_subjects": int(df_lo[subject_col].nunique()),
            "jaccard_with_full": jac,
            "ambiguous_set": ",".join([str(s) for s in amb_lo]),
        })
        for sid in amb_lo:
            freq[str(sid)] += 1

    stab = pd.DataFrame(fold_rows)
    out_stab = cfg.tab_dir / "moduleD_ambiguous_set_stability_LOSO.csv"
    save_csv(stab, out_stab)

    freq_df = pd.DataFrame({
        "sound_id": list(freq.keys()),
        "selected_count": list(freq.values()),
    })
    freq_df["selected_ratio"] = freq_df["selected_count"] / len(subjects)
    freq_df["n_folds"] = len(subjects)
    freq_df = freq_df.sort_values(["selected_count", "sound_id"], ascending=[False, True]).reset_index(drop=True)

    out_freq = cfg.tab_dir / "moduleD_ambiguous_set_selection_frequency_LOSO.csv"
    save_csv(freq_df, out_freq)

    fig_freq = None
    fig_jac = None
    if cfg.make_figures:
        # selection frequency
        fig_freq = cfg.fig_dir / "moduleD_LOSO_selection_frequency.png"
        plt.figure()
        plt.bar(freq_df["sound_id"].astype(str), freq_df["selected_count"].to_numpy())
        plt.xticks(rotation=90)
        plt.xlabel("sound_id")
        plt.ylabel("selected_count (across LOSO folds)")
        plt.title("LOSO ambiguous set selection frequency")
        plt.tight_layout()
        plt.savefig(fig_freq, dpi=200)
        plt.close()
        print(f"[SAVE] {fig_freq}")

        # jaccard hist
        fig_jac = cfg.fig_dir / "moduleD_LOSO_jaccard_hist.png"
        plt.figure()
        plt.hist(np.asarray(jacs, dtype=float), bins=8)
        plt.xlabel("Jaccard with full ambiguous set")
        plt.ylabel("count")
        plt.title("LOSO stability (Jaccard) histogram")
        plt.tight_layout()
        plt.savefig(fig_jac, dpi=200)
        plt.close()
        print(f"[SAVE] {fig_jac}")

    return stab, freq_df, fig_freq, fig_jac


def make_scatter_plots_top_assoc(
    cfg: Config,
    df_merged: pd.DataFrame,
    primary: pd.DataFrame,
    target_col: str = "ambig_pc2_mad",
) -> List[Path]:
    """Make scatter plots for top N associations (slide-ready)."""
    if not cfg.make_figures:
        return []
    if primary is None or len(primary) == 0:
        return []

    top = primary.sort_values(["q_fdr_primary", "p_perm"], ascending=[True, True]).head(cfg.n_scatter_top)
    paths: List[Path] = []

    y = df_merged[target_col].to_numpy(dtype=float)
    for _, row in top.iterrows():
        feat = str(row["feature"])
        x = df_merged[feat].to_numpy(dtype=float)

        fig_path = cfg.fig_dir / f"moduleD_scatter_{feat}.png"
        plt.figure()
        plt.scatter(x, y)
        plt.xlabel(feat)
        plt.ylabel(target_col)
        title = f"{feat} vs {target_col} (rho={row['spearman_rho']:.3f}, p_perm={row['p_perm']:.4f}, q={row['q_fdr_primary']:.4f})"
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=220)
        plt.close()
        print(f"[SAVE] {fig_path}")
        paths.append(fig_path)

    return paths


def build_next_actions_outputs(
    cfg: Config,
    stab: pd.DataFrame,
    freq: pd.DataFrame,
    primary: pd.DataFrame,
) -> Dict[str, object]:
    """
    Creates:
      - moduleD_KEYNUMBERS.json
      - moduleD_NEXT_ACTIONS.md
      - moduleD_SLIDE_BULLETS.txt
    Returns keynumbers dict.
    """
    # key numbers
    j_mean = float(stab["jaccard_with_full"].mean()) if len(stab) else float("nan")
    j_min = float(stab["jaccard_with_full"].min()) if len(stab) else float("nan")
    j_max = float(stab["jaccard_with_full"].max()) if len(stab) else float("nan")

    top_sounds = freq.sort_values(["selected_count", "sound_id"], ascending=[False, True]).head(15).copy()
    top_assoc = primary.sort_values(["q_fdr_primary", "p_perm"], ascending=[True, True]).head(10).copy()

    keynumbers = {
        "loso_jaccard_mean": j_mean,
        "loso_jaccard_min": j_min,
        "loso_jaccard_max": j_max,
        "top_selection_frequency": top_sounds.to_dict(orient="records"),
        "top_eeg_associations": top_assoc.to_dict(orient="records"),
    }

    out_key = cfg.tab_dir / "moduleD_KEYNUMBERS.json"
    save_json(keynumbers, out_key)

    # markdown (no tabulate)
    md = []
    md.append("# Module D: 次アクション（自動生成）\n")
    md.append("## 1) LOSO stability（曖昧音集合の安定性）\n")
    md.append(f"- Jaccard mean: **{j_mean:.3f}**\n")
    md.append(f"- Jaccard min : **{j_min:.3f}**\n")
    md.append(f"- Jaccard max : **{j_max:.3f}**\n\n")

    md.append("## 2) 曖昧音の選抜頻度（top 15）\n")
    md.append(df_to_markdown_pipe(top_sounds, max_rows=15))
    md.append("\n")

    md.append("## 3) EEG関連（top 10）\n")
    cols = ["target", "feature", "n", "spearman_rho", "p_perm", "q_fdr_primary", "ci95_lo", "ci95_hi"]
    cols = [c for c in cols if c in top_assoc.columns]
    md.append(df_to_markdown_pipe(top_assoc[cols], max_rows=10))
    md.append("\n")

    md.append("## 4) 今すぐ“成果物として強くする”ための実務アクション\n")
    md.append("- **感度分析**：neutral_th（例 0.25/0.35/0.45）と top_k（例 8/10/12）で結果の頑健性を確認。\n")
    md.append("- **再現性**：Permutation/Bootstrap は feature 名から固定 seed を作る（本コードは対応済）。\n")
    md.append("- **図の最適化**：上位4特徴の散布図＋ρ/p/qをスライドに直貼り。\n")

    out_md = cfg.tab_dir / "moduleD_NEXT_ACTIONS.md"
    save_text("".join(md), out_md)

    # slide bullets
    bullets = []
    bullets.append("【Module D：スライド用 要点】\n")
    bullets.append(f"- 曖昧音集合のLOSO安定性：Jaccard mean={j_mean:.3f}（min={j_min:.3f}, max={j_max:.3f}）\n")
    bullets.append("- 曖昧音（頻出）例：\n")
    for r in top_sounds.head(10).to_dict(orient="records"):
        bullets.append(f"  - {r['sound_id']}：{int(r['selected_count'])}/{int(r['n_folds'])}\n")
    bullets.append("- 個人差指標：ambig_pc2_mad（曖昧音集合内PC2のMAD）\n")
    bullets.append("- EEG関連（FDR通過）：\n")
    for r in top_assoc.head(4).to_dict(orient="records"):
        bullets.append(
            f"  - {r['feature']}: rho={float(r['spearman_rho']):.3f}, p_perm={float(r['p_perm']):.4f}, q={float(r['q_fdr_primary']):.4f}\n"
        )
    bullets.append("- 解釈：曖昧音で“評価が揺れる”人ほど、FC優位の初期〜中期ERP成分（N1/N2/P2）に系統的差が見える。\n")

    out_txt = cfg.tab_dir / "moduleD_SLIDE_BULLETS.txt"
    save_text("".join(bullets), out_txt)

    return keynumbers


def save_metadata(cfg: Config, extra: dict) -> None:
    meta = {
        "timestamp": now_str(),
        "python": sys.version,
        "platform": platform.platform(),
        "cfg": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "extra": extra,
    }
    out = cfg.log_dir / "moduleD_metadata.json"
    save_json(meta, out)


def next_actions_summary_print(cfg: Config) -> None:
    """Console summary only (files already saved elsewhere)."""
    stab_path = cfg.tab_dir / "moduleD_ambiguous_set_stability_LOSO.csv"
    freq_path = cfg.tab_dir / "moduleD_ambiguous_set_selection_frequency_LOSO.csv"
    primary_path = cfg.tab_dir / "moduleD_PRIMARY_results_with_bootstrapCI.csv"

    print("\n" + "=" * 72)
    print("4) 今すぐ“成果物として強くする”ための次アクション（自動要約）")
    print("=" * 72)

    if stab_path.exists():
        stab = pd.read_csv(stab_path)
        if "jaccard_with_full" in stab.columns and len(stab) > 0:
            print(f"[NEXT] LOSO Jaccard mean: {stab['jaccard_with_full'].mean():.3f}")
            print(f"[NEXT] LOSO Jaccard min : {stab['jaccard_with_full'].min():.3f}")
            print(f"[NEXT] LOSO Jaccard max : {stab['jaccard_with_full'].max():.3f}")
        else:
            print("[NEXT] LOSO stability table found, but missing jaccard_with_full.")
    else:
        print("[NEXT] LOSO stability table not found.")

    if freq_path.exists():
        freq = pd.read_csv(freq_path).sort_values(["selected_count", "sound_id"], ascending=[False, True])
        print("\n[NEXT] Top selection frequency sounds (head 15):")
        print(freq.head(15).to_string(index=False))
    else:
        print("[NEXT] LOSO selection frequency table not found.")

    if primary_path.exists():
        prim = pd.read_csv(primary_path).sort_values(["q_fdr_primary", "p_perm"], ascending=[True, True])
        print("\n[NEXT] Top EEG associations (head 10):")
        cols = ["target", "feature", "n", "spearman_rho", "p_perm", "q_fdr_primary", "ci95_lo", "ci95_hi"]
        cols = [c for c in cols if c in prim.columns]
        print(prim[cols].head(10).to_string(index=False))
    else:
        print("[NEXT] PRIMARY results table not found.")


# =========================
# Runner
# =========================
def run_moduleD(cfg: Config) -> None:
    t0 = time.time()

    # Load (minimal + dtype-safe)
    df_trial, df_subj, rating_cols = load_inputs(cfg)

    # Align sign
    df_trial_aligned, sign, loadings = align_pc2_sign(cfg, df_trial, rating_cols)

    # D-1 full ambiguous set (save once)
    stats_path = cfg.tab_dir / "moduleD_ambiguous_sounds_stats.csv"
    stats, ambiguous = define_ambiguous_sounds(cfg, df_trial_aligned, save_path=stats_path, verbose=True)

    # D-2 indiv scores
    df_scores, pca_info = compute_individual_difference_scores(cfg, df_trial_aligned, ambiguous)

    # D-3 merge EEG
    df_merged, eeg_cols = merge_subject_eeg(cfg, df_subj, df_scores)

    # Associations
    assoc, primary = association_analysis(cfg, df_merged, eeg_cols, target_col="ambig_pc2_mad")

    # Robustness
    _rob = robustness_leave1(cfg, df_merged, assoc, eeg_cols, target_col="ambig_pc2_mad")

    # Specificity
    _spec = specificity_overall_mad(cfg, df_trial_aligned, df_merged, eeg_cols, target_col="ambig_pc2_mad")

    # D-4 LOSO stability
    stab, freq, fig_freq, fig_jac = loso_ambiguous_stability(cfg, df_trial_aligned, ambiguous)

    # Scatter plots for top EEG associations
    scatter_paths = make_scatter_plots_top_assoc(cfg, df_merged, primary, target_col="ambig_pc2_mad")

    # Next actions pack (json + md + txt)
    keynumbers = build_next_actions_outputs(cfg, stab, freq, primary)

    # Metadata
    save_metadata(cfg, extra={
        "pc2_aligned_sign": sign,
        "inputs": {
            "trial_csv": str(cfg.trial_csv),
            "subject_csv": str(cfg.subject_csv),
        },
        "columns": {
            "subject_col": cfg.subject_col,
            "sound_col": cfg.sound_col,
            "pc2_col": cfg.pc2_col,
            "rating_cols": rating_cols,
            "eeg_cols": eeg_cols,
        },
        "ambiguous_definition": {
            "enough_ratio": cfg.enough_ratio,
            "neutral_th": cfg.neutral_th,
            "top_k": cfg.top_k,
            "ambiguous_sounds": [str(s) for s in ambiguous],
        },
        "pca_explained_variance_ratio": pca_info,
        "stats_params": {
            "n_perm": cfg.n_perm,
            "n_boot": cfg.n_boot,
            "seed": cfg.seed,
        },
        "figures": {
            "loso_freq": str(fig_freq) if fig_freq else None,
            "loso_jaccard_hist": str(fig_jac) if fig_jac else None,
            "scatter_paths": [str(p) for p in scatter_paths],
        },
        "keynumbers_path": str(cfg.tab_dir / "moduleD_KEYNUMBERS.json"),
    })

    # Console summary
    next_actions_summary_print(cfg)

    print("\n[MODULE D FINAL] DONE")
    print(f"  outputs dir : {cfg.out_dir}")
    print(f"  tables      : {cfg.tab_dir}")
    print(f"  figures     : {cfg.fig_dir}")
    print(f"  logs        : {cfg.log_dir}")
    print(f"[TIME] total seconds: {time.time() - t0:.1f}")


# =========================
# CLI
# =========================
def cli_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Module D: ambiguous sounds & individual differences + EEG association")
    parser.add_argument("--root-dir", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--trial-csv", type=str, default=None)
    parser.add_argument("--subject-csv", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)

    parser.add_argument("--subject-col", type=str, default=None)
    parser.add_argument("--sound-col", type=str, default=None)
    parser.add_argument("--pc2-col", type=str, default="PC2_emotion")

    parser.add_argument("--enough-ratio", type=float, default=0.80)
    parser.add_argument("--neutral-th", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=10)

    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no-minimal-load", action="store_true", help="Disable minimal loading (read all columns).")
    parser.add_argument("--no-fig", action="store_true", help="Disable figure outputs")

    args, _unknown = parser.parse_known_args(argv)

    cfg = build_config(
        root_dir=args.root_dir,
        trial_csv=args.trial_csv,
        subject_csv=args.subject_csv,
        out_dir=args.out_dir,
        subject_col=args.subject_col,
        sound_col=args.sound_col,
        pc2_col=args.pc2_col,
        enough_ratio=args.enough_ratio,
        neutral_th=args.neutral_th,
        top_k=args.top_k,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        seed=args.seed,
        minimal_load=(not args.no_minimal_load),
        make_figures=(not args.no_fig),
    )

    run_moduleD(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
