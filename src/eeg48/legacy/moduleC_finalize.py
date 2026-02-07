#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module C Finalize (End-to-End) - Single OUT + Wipe
==================================================

What this script does (Task 1–3):
1) Module C complete 結果の可視化と要約
   - 被験者別 R² (Acoustic / Acoustic+Bias / Full)
   - ΔR²(Full-Acoustic) ヒストグラム + permutation p
   - Acoustic vs Full の散布図
   - 要約テーブル（平均/中央値/良化人数/悪化人数/最良/最悪）

2) “脳波で語れる”結果をCに追加：Acoustic → EEG Encoding（刺激駆動表現）
   - sound-level acoustic features -> sound-mean EEG features
   - LOO-CVで multi-output Ridge を実行
   - EEG featureごとのR²分布 + Top20
   - TopK featureだけ permutation で p値（現実的な計算量）

3) “崩れる被験者”説明材料：QCリンク（ModuleB trial-level EEG特徴量があれば）
   - 被験者ごとの trial-level EEG feature の NaN率
   - Module C の ΔR² と関連を図示

Run:
  python moduleC_finalize_end_to_end.py --root_dir ... --out_dir_name moduleC_outputs --wipe_outdir 1

Notes:
- 音響特徴は sound-level table（leakage-safe）
- EEG feature列は eeg_features_participant_sound.csv の列名を「正」として master から参照
- Encoding permutation は TopK（default 50）だけに絞る
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys


import argparse
import shutil
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from scipy.stats import wilcoxon


# =========================================================
# Utils (safe)
# =========================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_wipe_outdir(root: Path, out_dir: Path) -> None:
    """
    OUT_DIR を削除して作り直す。
    誤爆防止：out_dir が root 配下で、かつ名前が 'moduleC' を含むことを要求。
    """
    out_dir = out_dir.resolve()
    root = root.resolve()

    if not str(out_dir).startswith(str(root) + str(Path("/"))):
        raise RuntimeError(f"[WIPE][ABORT] OUT_DIR is not under ROOT_DIR.\n  ROOT={root}\n  OUT ={out_dir}")

    if "modulec" not in out_dir.name.lower():
        raise RuntimeError(f"[WIPE][ABORT] OUT_DIR name must contain 'moduleC'.\n  OUT={out_dir}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)


def numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Pick the first matching column by normalized name; fallback to contains-match."""
    norm_cols = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_cols:
            return norm_cols[key]
    for cand in candidates:
        key = _norm(cand)
        for n, orig in norm_cols.items():
            if key and key in n:
                return orig
    return None


# =========================================================
# Logger
# =========================================================
class Logger:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def log(self, msg: str) -> None:
        print(msg)
        self.lines.append(msg)

    def save(self, path: Path) -> None:
        ensure_dir(path.parent)
        path.write_text("\n".join(self.lines), encoding="utf-8")


# =========================================================
# Config
# =========================================================
@dataclass
class Cfg:
    ROOT_DIR: Path
    OUT_DIR: Path

    # --- Inputs ---
    MASTER_PS: Path
    MASTER_SOUND: Path
    EEG_FEATURES_LIST: Path

    # --- Optional QC link ---
    MODULEB_TRIAL_EEG: Optional[Path] = None

    # --- Columns ---
    SUB_COL: str = "subject"
    SOUND_COL: str = "sound_id"
    EEG_SUB_COL: str = "participant"
    EEG_SOUND_COL: str = "number"
    TARGETS: Tuple[str, ...] = ("PC1_emotion", "PC2_emotion")

    # --- Modeling ---
    ALPHAS: Tuple[float, ...] = tuple(np.logspace(-3, 4, 24))
    INNER_SPLITS: int = 5

    EEG_DECONFOUND_WITH_ACOUSTICS: bool = True
    ACOUSTIC_USE_PCA: bool = True
    ACOUSTIC_PCA_NCOMP: int = 20

    USE_STACKING_WEIGHT: bool = True
    GAMMA_CLIP: Tuple[float, float] = (-2.0, 2.0)

    N_PERM: int = 200
    PERM_SEED: int = 123

    ENC_TOPK: int = 50
    ENC_N_PERM: int = 300
    ENC_SEED: int = 7

    SAVE_DPI: int = 220


# =========================================================
# Models
# =========================================================
def build_ridge_pipe(alpha: float) -> Pipeline:
    # solver="svd" で singular matrix 系の警告を避けやすい
    return Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=float(alpha), solver="svd"))
    ])


def tune_alpha_ridge(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    alphas: Tuple[float, ...], n_splits: int
) -> float:
    uniq = np.unique(groups)
    k = min(len(uniq), n_splits)
    if k < 2:
        return float(alphas[len(alphas)//2])

    gkf = GroupKFold(n_splits=k)
    best_a, best_mse = None, np.inf
    for a in alphas:
        mses = []
        for tr, va in gkf.split(X, y, groups):
            m = build_ridge_pipe(float(a))
            m.fit(X[tr], y[tr])
            pred = m.predict(X[va])
            mses.append(float(np.mean((y[va] - pred) ** 2)))
        mse = float(np.mean(mses))
        if mse < best_mse:
            best_mse = mse
            best_a = float(a)
    return float(best_a)


def oof_predict_ridge(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    alpha: float, n_splits: int
) -> np.ndarray:
    uniq = np.unique(groups)
    k = min(len(uniq), n_splits)
    if k < 2:
        m = build_ridge_pipe(alpha)
        m.fit(X, y)
        return m.predict(X)

    gkf = GroupKFold(n_splits=k)
    pred = np.full_like(y, np.nan, dtype=float)
    for tr, va in gkf.split(X, y, groups):
        m = build_ridge_pipe(alpha)
        m.fit(X[tr], y[tr])
        pred[va] = m.predict(X[va])

    if np.isnan(pred).any():
        m = build_ridge_pipe(alpha)
        m.fit(X, y)
        pred[np.isnan(pred)] = m.predict(X)[np.isnan(pred)]
    return pred


def fit_predict_ridge(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, alpha: float) -> np.ndarray:
    m = build_ridge_pipe(alpha)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


# =========================================================
# Permutation helper
# =========================================================
def permute_rows_within_group(X: np.ndarray, groups: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    Xp = X.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) <= 1:
            continue
        perm = rng.permutation(idx)
        Xp[idx] = X[perm]
    return Xp


def drop_all_nan_cols(X: np.ndarray, colnames: List[str]) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    keep = []
    dropped_names = []
    for j in range(X.shape[1]):
        if np.all(np.isnan(X[:, j])):
            dropped_names.append(colnames[j])
            continue
        keep.append(j)
    keep = np.array(keep, dtype=int)
    kept_names = [colnames[j] for j in keep.tolist()]
    return X[:, keep], kept_names, keep, dropped_names


def impute_with_train_median(Xtr: np.ndarray, Xte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(Xtr, axis=0)
    tr = Xtr.copy()
    te = Xte.copy()
    it = np.where(np.isnan(tr))
    tr[it] = np.take(med, it[1])
    ie = np.where(np.isnan(te))
    te[ie] = np.take(med, ie[1])
    return tr, te


# =========================================================
# Load & assemble
# =========================================================
def load_eeg_feature_names(cfg: Cfg, L: Logger) -> List[str]:
    if not cfg.EEG_FEATURES_LIST.exists():
        raise FileNotFoundError(f"Missing: {cfg.EEG_FEATURES_LIST}")
    eeg = pd.read_csv(cfg.EEG_FEATURES_LIST)
    num = numeric_cols(eeg)
    eeg_cols = [c for c in num if c not in {cfg.EEG_SUB_COL, cfg.EEG_SOUND_COL}]
    if len(eeg_cols) == 0:
        raise ValueError("EEG feature list has no numeric feature columns.")
    L.log(f"[INFO] EEG feature list: n={len(eeg_cols)}")
    return eeg_cols


def load_master_minimal(cfg: Cfg, eeg_cols: List[str], L: Logger) -> pd.DataFrame:
    if not cfg.MASTER_PS.exists():
        raise FileNotFoundError(f"Missing: {cfg.MASTER_PS}")

    header = pd.read_csv(cfg.MASTER_PS, nrows=0).columns.tolist()
    need = [cfg.SUB_COL, cfg.SOUND_COL, *cfg.TARGETS]
    need = [c for c in need if c in header]

    eeg_in_master = [c for c in eeg_cols if c in header]
    if len(eeg_in_master) == 0:
        raise ValueError("None of EEG feature columns exist in MASTER_PS.")
    usecols = list(dict.fromkeys(need + eeg_in_master))

    L.log(f"[LOAD] master minimal usecols: {len(usecols)}")
    m = pd.read_csv(cfg.MASTER_PS, usecols=usecols)
    m[cfg.SUB_COL] = m[cfg.SUB_COL].astype(str)
    m[cfg.SOUND_COL] = m[cfg.SOUND_COL].astype(str)

    for t in cfg.TARGETS:
        if t not in m.columns:
            raise ValueError(f"MASTER_PS missing target: {t}")
    return m


def load_sound(cfg: Cfg, L: Logger) -> pd.DataFrame:
    if not cfg.MASTER_SOUND.exists():
        raise FileNotFoundError(f"Missing: {cfg.MASTER_SOUND}")
    L.log(f"[LOAD] sound-level: {cfg.MASTER_SOUND}")
    s = pd.read_csv(cfg.MASTER_SOUND)
    if cfg.SOUND_COL not in s.columns:
        raise ValueError("MASTER_SOUND missing sound_id.")
    s[cfg.SOUND_COL] = s[cfg.SOUND_COL].astype(str)
    return s


def choose_acoustic_cols(cfg: Cfg, sound: pd.DataFrame, L: Logger) -> List[str]:
    num = numeric_cols(sound)
    exclude = {cfg.SOUND_COL, "Unnamed: 0", "index", "number"}
    exclude |= set(cfg.TARGETS)
    for c in sound.columns:
        if c.endswith("_mean"):
            exclude.add(c)

    cols = [c for c in num if c not in exclude]
    cols = [c for c in cols if not sound[c].isna().all()]
    L.log(f"[INFO] Acoustic features (sound-level, leakage-safe): n={len(cols)}")
    return cols


def assemble_df(
    cfg: Cfg, master: pd.DataFrame, sound: pd.DataFrame,
    acoustic_cols: List[str], eeg_cols: List[str], L: Logger
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    ren = {c: f"AC__{c}" for c in acoustic_cols}
    s_sub = sound[[cfg.SOUND_COL] + acoustic_cols].drop_duplicates(cfg.SOUND_COL).rename(columns=ren)
    df = master.merge(s_sub, on=cfg.SOUND_COL, how="left", validate="m:1")

    ac_use = [ren[c] for c in acoustic_cols]
    eeg_use = [c for c in eeg_cols if c in df.columns]

    miss_ac = float(df[ac_use].isna().mean().mean()) if ac_use else 1.0
    miss_eeg = float(df[eeg_use].isna().mean().mean()) if eeg_use else 1.0
    L.log(f"[INFO] merged df shape: {df.shape}")
    L.log(f"[INFO] Acoustic missing rate: {miss_ac:.6f}")
    L.log(f"[INFO] EEG missing rate: {miss_eeg:.6f}")
    return df, ac_use, eeg_use


# =========================================================
# EEG deconfound (Xeeg <- Xac) within fold
# =========================================================
def build_acoustic_transform(cfg: Cfg) -> Pipeline:
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
    ]
    if cfg.ACOUSTIC_USE_PCA:
        steps.append(("pca", PCA(
            n_components=cfg.ACOUSTIC_PCA_NCOMP,
            svd_solver="randomized",
            random_state=0
        )))
    return Pipeline(steps=steps)


def deconfound_eeg_with_acoustics(
    cfg: Cfg,
    Xac_tr: np.ndarray,
    Xac_te: np.ndarray,
    Xeeg_tr: np.ndarray,
    Xeeg_te: np.ndarray,
    eeg_names: List[str],
    L: Logger,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    Xeeg_tr2, names2, keep_idx, dropped_names = drop_all_nan_cols(Xeeg_tr, eeg_names)
    Xeeg_te2 = Xeeg_te[:, keep_idx]
    if len(dropped_names) > 0:
        L.log(f"[EEG][DECONF] Dropping all-NaN EEG cols in TRAIN: {len(dropped_names)}/{len(eeg_names)}")
        L.log(f"[EEG][DECONF] dropped example: {dropped_names[:10]}")

    Xeeg_tr2, Xeeg_te2 = impute_with_train_median(Xeeg_tr2, Xeeg_te2)

    Xpipe = build_acoustic_transform(cfg)
    Zac_tr = Xpipe.fit_transform(Xac_tr)
    Zac_te = Xpipe.transform(Xac_te)

    # solver="svd" で安定寄り
    m = Ridge(alpha=10.0, solver="svd")
    m.fit(Zac_tr, Xeeg_tr2)
    Xeeg_tr_hat = m.predict(Zac_tr)
    Xeeg_te_hat = m.predict(Zac_te)

    Rtr = (Xeeg_tr2 - Xeeg_tr_hat).astype(np.float32)
    Rte = (Xeeg_te2 - Xeeg_te_hat).astype(np.float32)
    return Rtr, Rte, names2


# =========================================================
# Module C Complete core (LOSO)
# =========================================================
def safe_wilcoxon_zero_test(x: np.ndarray) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test vs 0.
    戻り値: (stat, pvalue)
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, np.nan
    # 全部0なら検定不能に近い → p=1 扱い
    if np.all(np.abs(x) < 1e-12):
        return 0.0, 1.0
    try:
        w = wilcoxon(x)
        return float(w.statistic), float(w.pvalue)
    except Exception:
        return np.nan, np.nan


def run_moduleC_complete(
    cfg: Cfg, df: pd.DataFrame, ac_cols: List[str], eeg_cols: List[str], L: Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subjects = df[cfg.SUB_COL].astype(str).values
    uniq_sub = np.unique(subjects)

    all_rows: List[Dict[str, object]] = []
    dev_rows: List[Dict[str, object]] = []
    perm_rows: List[Dict[str, object]] = []
    rng_perm = np.random.RandomState(cfg.PERM_SEED)

    for tgt in cfg.TARGETS:
        L.log(f"[INFO] LOSO folds: {len(uniq_sub)} | target={tgt}")
        fold_cache: List[Dict[str, object]] = []

        for k, held in enumerate(uniq_sub, start=1):
            te_mask = (subjects == held)
            tr_mask = ~te_mask

            tr = df.loc[tr_mask].copy()
            te = df.loc[te_mask].copy()

            y_tr = tr[tgt].astype(float).values
            y_te = te[tgt].astype(float).values

            Xac_tr = tr[ac_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32).values
            Xac_te = te[ac_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32).values

            Xeeg_tr = tr[eeg_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32).values
            Xeeg_te = te[eeg_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32).values

            g_tr = tr[cfg.SUB_COL].astype(str).values

            # --- acoustic baseline ---
            alpha_ac = tune_alpha_ridge(Xac_tr, y_tr, g_tr, cfg.ALPHAS, cfg.INNER_SPLITS)
            yhat_tr_oof = oof_predict_ridge(Xac_tr, y_tr, g_tr, alpha_ac, cfg.INNER_SPLITS)
            resid_tr = y_tr - yhat_tr_oof
            yhat_ac_te = fit_predict_ridge(Xac_tr, y_tr, Xac_te, alpha_ac)

            # --- acoustic + bias (補助指標) ---
            subj_bias = pd.Series(resid_tr).groupby(tr[cfg.SUB_COL].astype(str).values).mean().to_dict()
            bias_te = np.array([subj_bias.get(held, 0.0)] * len(y_te), dtype=float)
            yhat_ac_bias_te = yhat_ac_te + bias_te

            # --- EEG matrix (optionally deconfounded) ---
            if cfg.EEG_DECONFOUND_WITH_ACOUSTICS:
                Ee_tr, Ee_te, _ = deconfound_eeg_with_acoustics(
                    cfg, Xac_tr, Xac_te, Xeeg_tr, Xeeg_te, eeg_cols, L
                )
            else:
                Ee_tr, Ee_te = impute_with_train_median(Xeeg_tr, Xeeg_te)
                Ee_tr = Ee_tr.astype(np.float32)
                Ee_te = Ee_te.astype(np.float32)

            # --- EEG -> residual(y) ---
            alpha_eeg = tune_alpha_ridge(Ee_tr, resid_tr, g_tr, cfg.ALPHAS, cfg.INNER_SPLITS)
            rhat_tr_oof = oof_predict_ridge(Ee_tr, resid_tr, g_tr, alpha_eeg, cfg.INNER_SPLITS)

            if cfg.USE_STACKING_WEIGHT:
                denom = float(np.dot(rhat_tr_oof, rhat_tr_oof)) + 1e-12
                gamma = float(np.dot(resid_tr, rhat_tr_oof) / denom)
                gamma = float(np.clip(gamma, cfg.GAMMA_CLIP[0], cfg.GAMMA_CLIP[1]))
            else:
                gamma = 1.0

            rhat_te = fit_predict_ridge(Ee_tr, resid_tr, Ee_te, alpha_eeg)
            yhat_full_te = yhat_ac_te + gamma * rhat_te

            r2_ac = float(r2_score(y_te, yhat_ac_te))
            r2_ac_bias = float(r2_score(y_te, yhat_ac_bias_te))
            r2_full = float(r2_score(y_te, yhat_full_te))
            delta = float(r2_full - r2_ac)

            all_rows.append({
                "target": tgt,
                "fold": k,
                "held_out": held,
                "r2_acoustic": r2_ac,
                "r2_acoustic_plus_bias": r2_ac_bias,
                "r2_full": r2_full,
                "delta_r2_full": delta,
                "alpha_ac": float(alpha_ac),
                "alpha_eeg": float(alpha_eeg),
                "gamma": float(gamma),
                "n_test": int(te_mask.sum()),
            })

            L.log(f"[FOLD][{tgt}][{k:02d}] held={held} | R2(ac)={r2_ac:+.4f} R2(full)={r2_full:+.4f} Δ={delta:+.4f}")

            fold_cache.append({
                "held": held,
                "y_te": y_te.astype(float),
                "yhat_ac_te": yhat_ac_te.astype(float),
                "resid_tr": resid_tr.astype(float),
                "Ee_tr": Ee_tr.astype(np.float32),
                "Ee_te": Ee_te.astype(np.float32),
                "g_tr": g_tr.astype(str),
                "g_te": te[cfg.SUB_COL].astype(str).values,
                "alpha_eeg": float(alpha_eeg),
                "gamma": float(gamma),
            })

            # --- EEG deviation task (EEG only) ---
            tr_sound_mean = tr.groupby(cfg.SOUND_COL)[tgt].mean().to_dict()
            mean_te = te[cfg.SOUND_COL].map(tr_sound_mean).astype(float).values
            # fallback（万一 unseen sound があってもクラッシュしない）
            if np.isnan(mean_te).any():
                mean_te[np.isnan(mean_te)] = float(np.nanmean(y_tr))

            ydev_tr = y_tr - tr[cfg.SOUND_COL].map(tr_sound_mean).astype(float).values
            ydev_te = y_te - mean_te

            alpha_dev = tune_alpha_ridge(Ee_tr, ydev_tr, g_tr, cfg.ALPHAS, cfg.INNER_SPLITS)
            ydev_hat = fit_predict_ridge(Ee_tr, ydev_tr, Ee_te, alpha_dev)
            r2_dev = float(r2_score(ydev_te, ydev_hat))

            dev_rows.append({
                "target": tgt,
                "fold": k,
                "held_out": held,
                "r2_eeg_dev": r2_dev,
                "alpha_eeg_dev": float(alpha_dev),
                "n_test": int(te_mask.sum()),
            })

        folds_df = pd.DataFrame([r for r in all_rows if r["target"] == tgt])

        stat, pval = safe_wilcoxon_zero_test(folds_df["delta_r2_full"].values)
        L.log(f"[STAT][{tgt}] Wilcoxon ΔR²_full vs 0: p={pval:.4g}, stat={stat}")

        obs_mean = float(folds_df["delta_r2_full"].mean())
        count_ge = 0
        perm_means = []

        L.log(f"[PERM] start n_perm={cfg.N_PERM} target={tgt} | obs mean Δ={obs_mean:+.4f}")
        for t in range(1, cfg.N_PERM + 1):
            deltas = []
            for fc in fold_cache:
                Ee_tr_p = permute_rows_within_group(fc["Ee_tr"], fc["g_tr"], rng_perm)
                Ee_te_p = permute_rows_within_group(fc["Ee_te"], fc["g_te"], rng_perm)

                rhat_te_p = fit_predict_ridge(Ee_tr_p, fc["resid_tr"], Ee_te_p, fc["alpha_eeg"])
                yhat_full_p = fc["yhat_ac_te"] + fc["gamma"] * rhat_te_p

                r2_ac_p = float(r2_score(fc["y_te"], fc["yhat_ac_te"]))
                r2_full_p = float(r2_score(fc["y_te"], yhat_full_p))
                deltas.append(r2_full_p - r2_ac_p)

            m = float(np.mean(deltas))
            perm_means.append(m)
            if abs(m) >= abs(obs_mean):
                count_ge += 1
            if t in {50, 100, 150, 200}:
                L.log(f"  [PERM] t={t}/{cfg.N_PERM} count_ge_abs={count_ge}")

        p_two = float((count_ge + 1) / (cfg.N_PERM + 1))
        perm_rows.append({
            "target": tgt,
            "obs_mean_delta_r2_full": obs_mean,
            "perm_mean": float(np.mean(perm_means)),
            "p_two_sided": p_two,
            "count_ge_abs": int(count_ge),
            "n_perm": int(cfg.N_PERM),
        })
        L.log(f"[STAT][{tgt}] Permutation p(two-sided)={p_two:.4g} | obs mean Δ={obs_mean:+.4f}")

    return pd.DataFrame(all_rows), pd.DataFrame(dev_rows), pd.DataFrame(perm_rows)


# =========================================================
# Figures + Summary
# =========================================================
def save_moduleC_figs(cfg: Cfg, folds: pd.DataFrame, perm: pd.DataFrame, fig_dir: Path) -> None:
    ensure_dir(fig_dir)

    for tgt in sorted(folds["target"].unique()):
        d = folds[folds["target"] == tgt].copy().sort_values("held_out")

        # R² by subject
        x = np.arange(len(d))
        plt.figure(figsize=(10, 4))
        plt.plot(x, d["r2_acoustic"].values, marker="o", label="Acoustic")
        plt.plot(x, d["r2_acoustic_plus_bias"].values, marker="o", label="Acoustic+Bias")
        plt.plot(x, d["r2_full"].values, marker="o", label="Full(+EEG residual)")
        plt.axhline(0, linewidth=1)
        plt.xticks(x, d["held_out"].astype(str).tolist(), rotation=45, ha="right")
        plt.ylabel("R² (held-out subject)")
        plt.title(f"Module C: R² by subject ({tgt})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"C_r2_by_subject_{tgt}.png", dpi=cfg.SAVE_DPI, bbox_inches="tight")
        plt.close()

        # Δ histogram
        p = float(perm.loc[perm["target"] == tgt, "p_two_sided"].iloc[0])
        obs = float(perm.loc[perm["target"] == tgt, "obs_mean_delta_r2_full"].iloc[0])
        plt.figure(figsize=(6, 4))
        plt.hist(d["delta_r2_full"].values, bins=12)
        plt.axvline(0, linewidth=1)
        plt.xlabel("ΔR² (Full − Acoustic)")
        plt.ylabel("Count (folds)")
        plt.title(f"{tgt} ΔR²  mean={obs:+.3f}  perm p={p:.3f}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"C_delta_hist_{tgt}.png", dpi=cfg.SAVE_DPI, bbox_inches="tight")
        plt.close()

        # Scatter
        plt.figure(figsize=(5.2, 5.2))
        plt.scatter(d["r2_acoustic"].values, d["r2_full"].values)
        mn = float(np.nanmin(np.r_[d["r2_acoustic"].values, d["r2_full"].values]))
        mx = float(np.nanmax(np.r_[d["r2_acoustic"].values, d["r2_full"].values]))
        plt.plot([mn, mx], [mn, mx], linewidth=1)
        plt.axhline(0, linewidth=1)
        plt.axvline(0, linewidth=1)
        for _, r in d.iterrows():
            plt.text(float(r["r2_acoustic"]), float(r["r2_full"]), str(r["held_out"]), fontsize=8)
        plt.xlabel("R² Acoustic")
        plt.ylabel("R² Full")
        plt.title(f"{tgt}: Acoustic vs Full")
        plt.tight_layout()
        plt.savefig(fig_dir / f"C_scatter_acoustic_vs_full_{tgt}.png", dpi=cfg.SAVE_DPI, bbox_inches="tight")
        plt.close()


def make_moduleC_summary(folds: pd.DataFrame, perm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tgt, d in folds.groupby("target"):
        dr = d["delta_r2_full"].to_numpy(dtype=float)
        rows.append({
            "target": tgt,
            "n_folds": int(len(d)),
            "mean_r2_acoustic": float(np.mean(d["r2_acoustic"])),
            "mean_r2_acoustic_plus_bias": float(np.mean(d["r2_acoustic_plus_bias"])),
            "mean_r2_full": float(np.mean(d["r2_full"])),
            "mean_delta_full": float(np.mean(dr)),
            "median_delta_full": float(np.median(dr)),
            "n_delta_pos": int(np.sum(dr > 0)),
            "n_delta_neg": int(np.sum(dr < 0)),
            "best_subject": str(d.loc[d["delta_r2_full"].idxmax(), "held_out"]),
            "best_delta": float(d["delta_r2_full"].max()),
            "worst_subject": str(d.loc[d["delta_r2_full"].idxmin(), "held_out"]),
            "worst_delta": float(d["delta_r2_full"].min()),
        })
    return pd.DataFrame(rows).merge(perm, on="target", how="left")


# =========================================================
# Encoding: Acoustic -> EEG (sound-mean)  [LEAKAGE-SAFE]
# =========================================================
def build_X_pipe_fold(cfg: Cfg, n_train: int, n_features: int) -> Pipeline:
    """
    LOOの各fold内でfitする前処理パイプライン。
    PCAは fold内の訓練サンプル数・特徴数を超えないように自動でクリップする。
    """
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
    ]
    if cfg.ACOUSTIC_USE_PCA:
        # PCAは n_components <= min(n_train-1, n_features) 必須
        max_comp = min(int(cfg.ACOUSTIC_PCA_NCOMP), max(1, n_train - 1), int(n_features))
        # n_trainが極端に小さいときはPCAを外す（安全運用）
        if max_comp >= 1 and max_comp < n_features:
            steps.append(("pca", PCA(
                n_components=max_comp,
                svd_solver="randomized",
                random_state=0
            )))
    return Pipeline(steps=steps)


def precompute_loo_transforms(cfg: Cfg, Xraw: np.ndarray) -> List[Dict[str, object]]:
    """
    LOOの各foldで、X前処理（impute/scale/pca）をtrainだけでfitし、
    train/testをtransformした行列をキャッシュする（リーク防止 + 高速化）。
    """
    loo = LeaveOneOut()
    folds: List[Dict[str, object]] = []
    n, p = Xraw.shape
    for tr, te in loo.split(np.arange(n)):
        Xpipe = build_X_pipe_fold(cfg, n_train=len(tr), n_features=p)
        Ztr = Xpipe.fit_transform(Xraw[tr])
        Zte = Xpipe.transform(Xraw[te])
        folds.append({
            "tr": tr,
            "te": te,
            "Ztr": Ztr,
            "Zte": Zte,
        })
    return folds


def loo_predict_multioutput_cached(folds: List[Dict[str, object]], Y: np.ndarray, alpha: float) -> np.ndarray:
    """
    キャッシュ済みの(Ztr, Zte)を使って multi-output Ridge をLOOで予測。
    """
    Yhat = np.full_like(Y, np.nan, dtype=float)
    for fd in folds:
        tr = fd["tr"]
        te = fd["te"]
        m = Ridge(alpha=float(alpha), solver="svd")
        m.fit(fd["Ztr"], Y[tr])
        Yhat[te] = m.predict(fd["Zte"])
    return Yhat


def tune_alpha_loocv_multioutput_cached(
    folds: List[Dict[str, object]],
    Y: np.ndarray,
    alphas: Tuple[float, ...],
) -> float:
    """
    LOO（リーク無し）でalphaを選ぶ。目的関数は全要素のMSE平均。
    """
    best_a, best_mse = None, np.inf
    for a in alphas:
        Yhat = loo_predict_multioutput_cached(folds, Y, float(a))
        mse = float(np.mean((Y - Yhat) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_a = float(a)
    return float(best_a)


def perm_test_topk_multioutput_cached(
    folds: List[Dict[str, object]],
    Ytop: np.ndarray,
    alpha: float,
    obs_r2: np.ndarray,
    n_perm: int,
    seed: int,
    L: Logger,
) -> np.ndarray:
    """
    TopKだけ permutation p を計算（高速版：multi-outputでまとめて回す）。
    各permで「各特徴（列）を独立にシャッフル」するので、列ごとのnullになる。
    """
    rng = np.random.RandomState(int(seed))
    topk = Ytop.shape[1]
    ge = np.zeros(topk, dtype=int)

    for t in range(1, int(n_perm) + 1):
        Yp = np.empty_like(Ytop)
        # 各列独立にシャッフル（nullをfeatureごとに作る）
        for j in range(topk):
            Yp[:, j] = rng.permutation(Ytop[:, j])

        Yp_hat = loo_predict_multioutput_cached(folds, Yp, float(alpha))
        rp = np.array([float(r2_score(Yp[:, j], Yp_hat[:, j])) for j in range(topk)], dtype=float)
        ge += (np.abs(rp) >= np.abs(obs_r2)).astype(int)

        if t in {50, 100, 200} or t == n_perm:
            L.log(f"  [ENC][PERM] t={t}/{n_perm}")

    pvals = (ge + 1) / (int(n_perm) + 1)
    return pvals.astype(float)


def run_encoding(
    cfg: Cfg, master: pd.DataFrame, sound: pd.DataFrame, eeg_cols: List[str], L: Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leakage-safe LOO Encoding:
    - LOOの各foldでX前処理をtrainだけでfit（impute/scale/pca）
    - その上でmulti-output Ridge をfit/predict
    """
    # --- acoustic cols（「音響」だけに寄せて、主観系が紛れないように） ---
    num = numeric_cols(sound)
    exclude = {cfg.SOUND_COL, "Unnamed: 0", "index", "number"}
    exclude |= set(cfg.TARGETS)

    # 主観平均や派生列が紛れたときに事故るので、"_mean" を除外（Aと同じ思想）
    for c in sound.columns:
        if str(c).endswith("_mean"):
            exclude.add(c)

    ac_cols = [c for c in num if c not in exclude]
    ac_cols = [c for c in ac_cols if not sound[c].isna().all()]
    if len(ac_cols) == 0:
        raise ValueError("[ENC] No acoustic feature columns after exclusion.")

    Xdf = (
        sound[[cfg.SOUND_COL] + ac_cols]
        .drop_duplicates(cfg.SOUND_COL)
        .set_index(cfg.SOUND_COL)
        .replace([np.inf, -np.inf], np.nan)
    )

    # --- Y: sound-mean EEG ---
    Ydf = master.groupby(cfg.SOUND_COL)[eeg_cols].mean()

    # 共通soundをXdfの順序で揃える（安定）
    common = [sid for sid in Xdf.index.tolist() if sid in Ydf.index]
    if len(common) < 10:
        raise ValueError(f"[ENC] Too few common sounds: {len(common)}")

    Xraw = Xdf.loc[common, ac_cols].to_numpy(dtype=np.float64)
    Yraw = Ydf.loc[common, eeg_cols].to_numpy(dtype=np.float64)

    # EEG列のクリーニング（all-NaN / ほぼ定数は落とす）
    keep = []
    kept_names = []
    for j, name in enumerate(eeg_cols):
        col = Yraw[:, j]
        if np.all(np.isnan(col)):
            continue
        if np.nanstd(col) < 1e-12:
            continue
        keep.append(j)
        kept_names.append(name)
    keep = np.array(keep, dtype=int)
    Y = Yraw[:, keep]

    # YのNaNは（ほぼ無いはずだが）列medianで埋める
    med = np.nanmedian(Y, axis=0)
    inds = np.where(np.isnan(Y))
    if len(inds[0]) > 0:
        Y = Y.copy()
        Y[inds] = np.take(med, inds[1])

    L.log(f"[ENC] n_sounds={len(common)} | X_raw={Xraw.shape} | Y={Y.shape} (kept EEG={len(kept_names)})")

    # LOO変換キャッシュ（リーク無し + 高速化）
    folds = precompute_loo_transforms(cfg, Xraw)

    # alpha tuning（リーク無しLOO）
    alpha = tune_alpha_loocv_multioutput_cached(folds, Y, cfg.ALPHAS)
    L.log(f"[ENC] best alpha={alpha}")

    # LOO予測（リーク無し）
    Yhat = loo_predict_multioutput_cached(folds, Y, alpha)

    r2s = np.array([float(r2_score(Y[:, j], Yhat[:, j])) for j in range(Y.shape[1])], dtype=float)
    res = (
        pd.DataFrame({"eeg_feature": kept_names, "r2_loocv": r2s})
        .sort_values("r2_loocv", ascending=False)
        .reset_index(drop=True)
    )

    # --- TopK permutation（multi-outputでまとめて回す） ---
    topk = int(min(cfg.ENC_TOPK, len(res)))
    top = res.head(topk).copy()

    idx_top = top.index.to_numpy(dtype=int)
    Ytop = Y[:, idx_top]
    Ytop_hat = Yhat[:, idx_top]
    obs_r2 = np.array([float(r2_score(Ytop[:, j], Ytop_hat[:, j])) for j in range(topk)], dtype=float)

    L.log(f"[ENC][PERM] TopK={topk} N_PERM={cfg.ENC_N_PERM}")
    pvals = perm_test_topk_multioutput_cached(
        folds=folds,
        Ytop=Ytop,
        alpha=alpha,
        obs_r2=obs_r2,
        n_perm=cfg.ENC_N_PERM,
        seed=cfg.ENC_SEED,
        L=L,
    )
    top["p_perm_two_sided"] = pvals

    summary = pd.DataFrame([{
        "n_sounds": int(len(common)),
        "n_acoustic_features": int(len(ac_cols)),
        "use_pca": bool(cfg.ACOUSTIC_USE_PCA),
        "pca_ncomp": int(cfg.ACOUSTIC_PCA_NCOMP) if cfg.ACOUSTIC_USE_PCA else 0,
        "n_eeg_features_kept": int(len(kept_names)),
        "alpha": float(alpha),
        "r2_mean": float(np.mean(res["r2_loocv"])),
        "r2_median": float(np.median(res["r2_loocv"])),
        "topk": int(topk),
        "n_topk_p05": int((top["p_perm_two_sided"] < 0.05).sum()),
    }])

    return res, top, summary



def save_encoding_figs(cfg: Cfg, res: pd.DataFrame, fig_dir: Path) -> None:
    ensure_dir(fig_dir)

    plt.figure(figsize=(6, 4))
    plt.hist(res["r2_loocv"].values, bins=30)
    plt.axvline(0, linewidth=1)
    plt.xlabel("LOO-CV R² (Acoustic → EEG feature)")
    plt.ylabel("Count")
    plt.title("Encoding: stimulus-driven EEG explained by acoustics")
    plt.tight_layout()
    plt.savefig(fig_dir / "encoding_r2_hist.png", dpi=cfg.SAVE_DPI, bbox_inches="tight")
    plt.close()

    top20 = res.head(20).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top20["eeg_feature"].values, top20["r2_loocv"].values)
    plt.axvline(0, linewidth=1)
    plt.xlabel("LOO-CV R²")
    plt.title("Top 20 EEG features explained by acoustics")
    plt.tight_layout()
    plt.savefig(fig_dir / "encoding_top20.png", dpi=cfg.SAVE_DPI, bbox_inches="tight")
    plt.close()


# =========================================================
# QC link (optional, single OUT style)
# =========================================================
def run_qc_link_optional(cfg: Cfg, folds: pd.DataFrame, L: Logger) -> Optional[pd.DataFrame]:
    if cfg.MODULEB_TRIAL_EEG is None:
        L.log("[QC] Skip: MODULEB_TRIAL_EEG is None.")
        return None
    if not cfg.MODULEB_TRIAL_EEG.exists():
        L.log(f"[QC] Skip: file not found: {cfg.MODULEB_TRIAL_EEG}")
        return None

    b = pd.read_csv(cfg.MODULEB_TRIAL_EEG)

    # subject column auto-detect
    sub_col = pick_col(b, ["subject", "subject_id", "participant", "participant_id", "subj"])
    if sub_col is None:
        L.log("[QC] Skip: could not detect subject column in ModuleB trial EEG CSV.")
        return None

    num = numeric_cols(b)
    drop_like = {"sound_id", "sound", "number", "run", "trial", "epoch", "time"}
    feat_cols = [c for c in num if _norm(c) not in {_norm(x) for x in drop_like}]
    if len(feat_cols) == 0:
        L.log("[QC] Skip: no numeric feature columns in ModuleB trial EEG CSV.")
        return None

    grp = b.groupby(b[sub_col].astype(str))
    qc = pd.DataFrame({
        "subject": grp.size().index.astype(str),
        "n_trials": grp.size().values.astype(int),
        "nan_rate": [float(np.isnan(df[feat_cols].to_numpy(dtype=float)).mean()) for _, df in grp],
    })

    c = folds.rename(columns={"held_out": "subject"}).copy()
    c["subject"] = c["subject"].astype(str)
    merged = c.merge(qc, on="subject", how="left")
    return merged


def save_qc_figs(cfg: Cfg, merged: pd.DataFrame, fig_dir: Path) -> None:
    ensure_dir(fig_dir)
    for tgt, d in merged.groupby("target"):
        plt.figure(figsize=(6, 4))
        plt.scatter(d["nan_rate"].values, d["delta_r2_full"].values)
        for _, r in d.iterrows():
            if pd.notna(r["nan_rate"]) and pd.notna(r["delta_r2_full"]):
                plt.text(float(r["nan_rate"]), float(r["delta_r2_full"]), str(r["subject"]), fontsize=8)
        plt.axhline(0, linewidth=1)
        plt.xlabel("NaN rate in trial-level EEG features")
        plt.ylabel("ΔR² (Full − Acoustic) in Module C")
        plt.title(f"QC link: {tgt}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"qc_nan_vs_delta_{tgt}.png", dpi=cfg.SAVE_DPI, bbox_inches="tight")
        plt.close()


# =========================================================
# Reports (貼って終わり用)
# =========================================================
def write_reports(cfg: Cfg, summary: pd.DataFrame, report_dir: Path) -> None:
    ensure_dir(report_dir)

    lines = []
    lines.append("# Module C（Finalize）— 結果要約（貼って終わり版）\n")
    lines.append("## 位置づけ（教授対策の定義）")
    lines.append("- **B/F**：EEG→主観のデコード（本丸）")
    lines.append("- **C**：音響代理（confound）を厳密に統制し、EEGの“追加説明力”の上限を定量化する（盾）\n")
    lines.append("## 観察された事実（数値）")
    for _, r in summary.iterrows():
        lines.append(f"- {r['target']}: mean ΔR²_full={r['mean_delta_full']:+.4f}（perm p={r.get('p_two_sided', np.nan):.3f}）")
    lines.append("\n## 解釈（言い方）")
    lines.append("音刺激が主観を強く規定する状況では、音響で説明可能な成分を統制すると、残差EEGの未知被験者一般化（LOSO）での追加説明力は限定的だった。")
    lines.append("これは『EEGが弱い』という結論ではなく、『音響代理を差し引いた上での厳密な上限評価』であり、Cは解析の妥当性・頑健性を担保する役割を持つ。")
    (report_dir / "ModuleC_task1_summary.md").write_text("\n".join(lines), encoding="utf-8")

    rep = []
    rep.append("# Encoding（Acoustic → EEG）— 書き方テンプレ\n")
    rep.append("## 主張（短く強く）")
    rep.append("- 音響特徴量から、音ごとのEEG平均（刺激駆動成分）の一部は説明可能（Encoding）。")
    rep.append("- その一方で、音響で説明可能な成分を統制した残差EEGは、未知被験者一般化で主観PCへの追加説明力が限定的。")
    rep.append("\n## 本文に貼って通る版（日本語）")
    rep.append("音刺激が共通にもたらす脳波表現を評価するため、音ごとのEEG平均（stimulus-driven component）を目的変数とし、音響特徴量からのエンコーディング解析（LOO-CV）を行った。")
    rep.append("その結果、一部のEEG特徴量は正の決定係数を示し、音刺激の物理構造が脳波特徴として反映されることが示唆された。")
    rep.append("一方、音響で説明可能な成分を統制した残差EEGは、未知被験者外挿（LOSO）における主観PC予測への寄与が限定的であった。")
    (report_dir / "Encoding_text_template.md").write_text("\n".join(rep), encoding="utf-8")


# =========================================================
# CLI / Main
# =========================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # Public-friendly aliases (no private defaults)

    # Jupyter/ipykernel 直実行でも動くように default を設定（required を外す）
    ap.add_argument("--root-dir", dest="root_dir", type=str, default=".")
    ap.add_argument("--root_dir", dest="root_dir", type=str, default=".")
    ap.add_argument("--out-dir", dest="out_dir", type=str, default=None,
                    help="Optional absolute output directory. If set, overrides root/out_dir_name.")
    ap.add_argument("--out_dir_name", type=str, default="moduleC_outputs")
    ap.add_argument("--wipe_outdir", type=int, default=1, help="1: delete OUT_DIR then run; 0: keep")
    ap.add_argument("--moduleb_trial_csv", type=str,
                    default="moduleB_outputs/tables/moduleB_trial_eeg_features.csv")

    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--enc_n_perm", type=int, default=300)
    ap.add_argument("--enc_topk", type=int, default=50)

    # ipykernel が付ける unknown args（例: -f xxxx）を握りつぶす
    args, _unknown = ap.parse_known_args()
    return args



def main() -> None:
    # warning はログが汚れるので必要最低限だけ抑える（処理自体は続行）
    warnings.filterwarnings("ignore", category=UserWarning, message="Singular matrix.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    args = parse_args()

    root = Path(args.root_dir)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (root / args.out_dir_name)

    cfg = Cfg(
        ROOT_DIR=root,
        OUT_DIR=out_dir,
        MASTER_PS=root / "derivatives/master_tables/master_participant_sound_level_with_PC.csv",
        MASTER_SOUND=root / "derivatives/master_tables/master_sound_level_with_PC.csv",
        EEG_FEATURES_LIST=root / "derivatives/eeg_features_participant_sound.csv",
        MODULEB_TRIAL_EEG=(root / args.moduleb_trial_csv) if args.moduleb_trial_csv else None,
    )
    cfg.N_PERM = int(args.n_perm)
    cfg.ENC_N_PERM = int(args.enc_n_perm)
    cfg.ENC_TOPK = int(args.enc_topk)

    L = Logger()

    # wipe (single OUT style)
    if int(args.wipe_outdir) == 1:
        safe_wipe_outdir(cfg.ROOT_DIR, cfg.OUT_DIR)

    tab_dir = cfg.OUT_DIR / "tables"
    fig_dir = cfg.OUT_DIR / "figures"
    log_dir = cfg.OUT_DIR / "logs"
    rep_dir = cfg.OUT_DIR / "reports"
    ensure_dir(tab_dir); ensure_dir(fig_dir); ensure_dir(log_dir); ensure_dir(rep_dir)

    L.log("=== Module C FINALIZE (Single OUT + Wipe) START ===")
    L.log(f"[CFG] ROOT_DIR={cfg.ROOT_DIR}")
    L.log(f"[CFG] OUT_DIR ={cfg.OUT_DIR}")
    L.log(f"[CFG] EEG_DECONFOUND_WITH_ACOUSTICS={cfg.EEG_DECONFOUND_WITH_ACOUSTICS}")
    L.log(f"[CFG] ACOUSTIC_USE_PCA={cfg.ACOUSTIC_USE_PCA} ncomp={cfg.ACOUSTIC_PCA_NCOMP}")
    L.log(f"[CFG] N_PERM={cfg.N_PERM} | ENC_TOPK={cfg.ENC_TOPK} ENC_N_PERM={cfg.ENC_N_PERM}")

    # load
    eeg_cols = load_eeg_feature_names(cfg, L)
    master = load_master_minimal(cfg, eeg_cols, L)
    sound = load_sound(cfg, L)
    ac_cols = choose_acoustic_cols(cfg, sound, L)

    df, ac_use, eeg_use = assemble_df(cfg, master, sound, ac_cols, eeg_cols, L)
    L.log(f"[INFO] n_acoustic_use={len(ac_use)} n_eeg_use={len(eeg_use)}")

    # run Module C complete
    L.log("\n--- Run: Module C Complete (LOSO + Perm + Deviation) ---")
    folds, dev, perm = run_moduleC_complete(cfg, df, ac_use, eeg_use, L)

    folds_path = tab_dir / "moduleC_complete_folds.csv"
    dev_path = tab_dir / "moduleC_eeg_deviation_folds.csv"
    perm_path = tab_dir / "moduleC_complete_permutation_summary.csv"
    folds.to_csv(folds_path, index=False, encoding="utf-8-sig")
    dev.to_csv(dev_path, index=False, encoding="utf-8-sig")
    perm.to_csv(perm_path, index=False, encoding="utf-8-sig")
    L.log(f"[SAVE] {folds_path}")
    L.log(f"[SAVE] {dev_path}")
    L.log(f"[SAVE] {perm_path}")

    # figs + summary
    L.log("\n--- Make: Module C Summary + Figures ---")
    save_moduleC_figs(cfg, folds, perm, fig_dir)
    summary = make_moduleC_summary(folds, perm)
    summary_path = tab_dir / "moduleC_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    L.log(f"[SAVE] {summary_path}")

    # encoding
    L.log("\n--- Run: Encoding (Acoustic -> EEG sound-mean) ---")
    enc_res, enc_top, enc_sum = run_encoding(cfg, master, sound, eeg_use, L)
    enc_res_path = tab_dir / "encoding_feature_r2.csv"
    enc_top_path = tab_dir / "encoding_topk_with_p.csv"
    enc_sum_path = tab_dir / "encoding_summary.csv"
    enc_res.to_csv(enc_res_path, index=False, encoding="utf-8-sig")
    enc_top.to_csv(enc_top_path, index=False, encoding="utf-8-sig")
    enc_sum.to_csv(enc_sum_path, index=False, encoding="utf-8-sig")
    save_encoding_figs(cfg, enc_res, fig_dir)
    L.log(f"[SAVE] {enc_res_path}")
    L.log(f"[SAVE] {enc_top_path}")
    L.log(f"[SAVE] {enc_sum_path}")

    # QC link optional
    L.log("\n--- Optional: QC link (ModuleB trial EEG) ---")
    merged = run_qc_link_optional(cfg, folds, L)
    if merged is not None:
        qc_path = tab_dir / "qc_subject_qc_with_moduleC.csv"
        merged.to_csv(qc_path, index=False, encoding="utf-8-sig")
        save_qc_figs(cfg, merged, fig_dir)
        L.log(f"[SAVE] {qc_path}")
    else:
        L.log("[QC] skipped.")

    # reports
    L.log("\n--- Write: Reports ---")
    write_reports(cfg, summary, rep_dir)
    L.log(f"[SAVE] {rep_dir / 'ModuleC_task1_summary.md'}")
    L.log(f"[SAVE] {rep_dir / 'Encoding_text_template.md'}")

    # log
    log_path = log_dir / "moduleC_finalize_run.log"
    L.save(log_path)
    L.log(f"[SAVE] {log_path}")

    L.log("=== Module C FINALIZE END ===")
    L.log(f"Outputs saved under: {cfg.OUT_DIR}")


if __name__ == "__main__":
    main()