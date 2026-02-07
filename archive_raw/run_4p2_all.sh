#!/usr/bin/env bash
set -euo pipefail

cd /Users/shunsuke/EEG_48sounds

TASKS="emo_arousal,emo_approach,emo_valence,emo_arousal_high,emo_approach_high,emo_valence_high,is_ambiguous,category_3"

# 0) matplotlib font cache（文字化け対策：一度消す）
rm -f ~/.matplotlib/fontlist-v*.json 2>/dev/null || true

# 1) ModuleB本体（表・重要度CSV・MAP用CSVを生成）
python moduleB_phaseB_EEG_only.py \
  --root-dir . \
  --crop-tmin -0.2 --crop-tmax 5.0 \
  --baseline-tmin -0.2 --baseline-tmax 0.0 \
  --erp-windows "0-200,100-300,200-400,300-500,400-600,500-700,600-800,700-900,800-1000" \
  --tfr-windows "0-400,200-600,400-800,600-1000,1000-2000,2000-3000,3000-4000,4000-5000" \
  --tasks "$TASKS" \
  --with-maps \
  --n-perm 1000 --n-jobs -1

# 2) MAP図（ERP/TFR）を“0ms以降”で作る（既存plotを使用）
python plot_moduleB_maps.py \
  --tables moduleB_outputs/tables \
  --out moduleB_outputs/figures/maps_0to5000_v1 \
  --erp-style line

# 3) 追加：重要度の分解（ERP vs TFR短 vs TFR長）＋ヒートマップ＋Top特徴CSV
python plot_moduleB_importance_decompose_abscoef_v4.py \
  --in-dir moduleB_outputs/tables \
  --out-dir moduleB_outputs/figures/importance_decompose_0to5000_v1 \
  --tasks "$TASKS" \
  --tfr-split-ms 1000 --tfr-max-ms 5000 --tmin-ms 0 \
  --font "Hiragino Sans" --fontsize 22

# 4) 追加：ROI別LOSO（“どの領域だけで当たるか”＝主張が硬くなる）
python moduleB_roi_loso.py \
  --root-dir . \
  --out-dir moduleB_outputs/figures/roi_loso_both_0to5000_v1 \
  --tasks "$TASKS" \
  --modality both \
  --n-perm 1000 --n-jobs -1

# 5) まとめてZIP
ts=$(date +%Y%m%d_%H%M%S)
zip -r "moduleB_outputs_thesis_pack_${ts}.zip" moduleB_outputs/tables moduleB_outputs/figures >/dev/null
echo "[OK] created: moduleB_outputs_thesis_pack_${ts}.zip"
