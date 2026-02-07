#!/usr/bin/env python3
import argparse
import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def pick_label_column(md):
    # よくあるラベル名を順に探す（あなたのデータに合わせて増やしてOK）
    for c in ["is_ambiguous", "ambiguous", "y", "label", "target"]:
        if c in md.columns:
            return c
    return None

def ensure_jp_font():
    # 文字化け対策：まず「日本語フォントを使う」だけでOK（MacならHiraginoで通ることが多い）
    import matplotlib
    matplotlib.rcParams["font.family"] = [
        "Hiragino Sans", "Hiragino Kaku Gothic ProN", "IPAexGothic", "Noto Sans CJK JP", "Meiryo", "MS Gothic"
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

def save_erp(epochs, out_png, ch="Fp2", tmin=0.0, tmax=1.0,
             highlight_ms=((300, 500), (500, 900))):
    ensure_jp_font()

    ep = epochs.copy().pick(ch).crop(tmin=tmin, tmax=tmax)
    # ERPはベースライン入れておくと見やすい（-200〜0msがあれば）
    if ep.tmin < 0:
        ep.apply_baseline((ep.tmin, 0))

    md = ep.metadata
    if md is not None:
        col = pick_label_column(md)
    else:
        col = None

    fig, ax = plt.subplots(figsize=(10, 4))

    if (md is not None) and (col is not None):
        # 0/1を想定：曖昧=1, 非曖昧=0
        ep0 = ep[md[col].astype(int) == 0]
        ep1 = ep[md[col].astype(int) == 1]

        ev0 = ep0.average()
        ev1 = ep1.average()

        times_ms = ev0.times * 1000
        y0 = ev0.data[0] * 1e6  # V→µV
        y1 = ev1.data[0] * 1e6

        ax.plot(times_ms, y0, label="非曖昧(0)")
        ax.plot(times_ms, y1, label="曖昧(1)")
        ax.set_title(f"ERP波形（{ch}）：曖昧 vs 非曖昧")
        ax.legend(loc="upper right")
    else:
        # ラベルが無い場合：単純に全試行平均のERPを出す（“実データ例”としては十分）
        ev = ep.average()
        times_ms = ev.times * 1000
        y = ev.data[0] * 1e6
        ax.plot(times_ms, y, label="平均ERP")
        ax.set_title(f"ERP波形（{ch}）")
        ax.legend(loc="upper right")

    for a, b in highlight_ms:
        ax.axvspan(a, b, alpha=0.15)

    ax.set_xlabel("時間 (ms)")
    ax.set_ylabel("電位 (µV)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def save_tfr(epochs, out_png, chs=("Cz", "C3", "C4"),
             freqs=np.arange(2, 46, 1),
             baseline=(-0.2, 0.0),
             tmin=0.0, tmax=1.0,
             highlight=(600, 1000, 30, 45)):  # (t_ms1,t_ms2,f1,f2)
    ensure_jp_font()

    ep = epochs.copy()
    # 指定chが無い場合に備えて、存在するchだけ使う
    have = [c for c in chs if c in ep.ch_names]
    if len(have) == 0:
        raise RuntimeError(f"TFR用チャンネルが見つかりません: {chs}")

    ep = ep.pick(have)

    # MorletでTFR（平均power）
    n_cycles = freqs / 2.0
    power = mne.time_frequency.tfr_morlet(
        ep, freqs=freqs, n_cycles=n_cycles,
        use_fft=True, return_itc=False,
        average=True, decim=2, n_jobs=1, verbose=False
    )

    # チャンネル平均（freq x time）
    data = power.data.mean(axis=0)
    times = power.times
    f = power.freqs

    # ベースラインdB化（log10）
    bmask = (times >= baseline[0]) & (times <= baseline[1])
    if bmask.sum() < 2:
        # ベースライン区間が含まれないepochsの場合は、0〜100msを仮ベースラインにする
        bmask = (times >= 0.0) & (times <= 0.1)

    base = data[:, bmask].mean(axis=1, keepdims=True)
    eps = 1e-12
    data_db = 10 * np.log10((data + eps) / (base + eps))

    # 表示は0〜1000msに絞る
    tmask = (times >= tmin) & (times <= tmax)
    times_ms = times[tmask] * 1000
    data_db = data_db[:, tmask]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(
        data_db, origin="lower", aspect="auto",
        extent=[times_ms[0], times_ms[-1], f[0], f[-1]]
    )
    ax.set_title(f"TFR（{'+'.join(have)} 平均）：ベースライン補正(dB)")
    ax.set_xlabel("時間 (ms)")
    ax.set_ylabel("周波数 (Hz)")

    # ハイライト枠
    t1, t2, f1, f2 = highlight
    ax.add_patch(Rectangle((t1, f1), t2 - t1, f2 - f1, fill=False, linewidth=2))

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Power変化 (dB)")

    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", required=True, help="epochs fif path (e.g., *-epo.fif)")
    ap.add_argument("--outdir", default="figs_real", help="output dir")
    ap.add_argument("--erp_ch", default="Fp2")
    args = ap.parse_args()

    epochs = mne.read_epochs(args.epochs, preload=True, verbose=False)

    import os
    os.makedirs(args.outdir, exist_ok=True)

    erp_png = os.path.join(args.outdir, "ERP_real.png")
    tfr_png = os.path.join(args.outdir, "TFR_real.png")

    save_erp(epochs, erp_png, ch=args.erp_ch)
    save_tfr(epochs, tfr_png)

    print("saved:", erp_png)
    print("saved:", tfr_png)

if __name__ == "__main__":
    main()
