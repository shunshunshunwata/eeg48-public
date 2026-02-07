#!/usr/bin/env python3
import os
import argparse
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.family"] = "Hiragino Sans"
mpl.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け対策


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", required=True, help="Path to epochs .fif (e.g., epochs_all-epo.fif)")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--ch", default="Fz", help="Channel name to plot (default: Fz)")
    ap.add_argument("--epoch-idx", type=int, default=0, help="Epoch index (0-based)")
    ap.add_argument("--tmin", type=float, default=-0.5, help="Plot start time [s]")
    ap.add_argument("--tmax", type=float, default=10.0, help="Plot end time [s]")
    ap.add_argument("--shade", nargs=2, type=float, default=None, metavar=("S0","S1"),
                    help="Shade interval [s] (e.g., 0 5 for stimulus duration)")
    args = ap.parse_args()

    if not os.path.exists(args.epochs):
        raise FileNotFoundError(f"epochs file not found: {args.epochs}")

    epochs = mne.read_epochs(args.epochs, preload=True, verbose="ERROR")

    # channel existence check (case-sensitive in many datasets)
    chs = epochs.ch_names
    if args.ch not in chs:
        # try case-insensitive match
        cand = [c for c in chs if c.lower() == args.ch.lower()]
        if cand:
            args.ch = cand[0]
        else:
            raise ValueError(f"Channel '{args.ch}' not found. Available example: {chs[:10]} ...")

    if args.epoch_idx < 0 or args.epoch_idx >= len(epochs):
        raise IndexError(f"epoch-idx out help range: 0..{len(epochs)-1}")

    ep = epochs[args.epoch_idx].copy().pick([args.ch])

    # crop to desired window (clip within available range)
    tmin = max(args.tmin, float(ep.tmin))
    tmax = min(args.tmax, float(ep.tmax))
    ep.crop(tmin=tmin, tmax=tmax)

    data = ep.get_data()[0, 0, :]  # (n_times,)
    times = ep.times               # (n_times,)
    data_uV = data * 1e6           # V -> µV

    # --- plot ---
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig = plt.figure(figsize=(10, 3.2), dpi=200)
    ax = fig.add_subplot(111)

    ax.plot(times, data_uV, linewidth=1.8)

    # stimulus onset
    ax.axvline(0, linewidth=2.2)
    ax.text(0, ax.get_ylim()[1]*0.95, "音提示開始", va="top", ha="left")

    # optional shading (stim duration)
    if args.shade is not None:
        s0, s1 = args.shade
        ax.axvspan(s0, s1, alpha=0.12)
        ax.text((s0+s1)/2, ax.get_ylim()[1]*0.95, "音提示(0–5s)", va="top", ha="center")

    ax.set_xlabel("時間（秒）", fontsize=12)
    ax.set_ylabel("電位（µV）", fontsize=12)
    ax.set_title(f"EEG生波形例：{args.ch}（1試行）", fontsize=13)

    ax.tick_params(axis="both", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.out, transparent=True)
    plt.close(fig)

    print(f"[OK] saved: {args.out}")

if __name__ == "__main__":
    main()
