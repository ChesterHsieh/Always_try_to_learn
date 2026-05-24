#!/usr/bin/env bash
# 在 RunPod pod 上執行：用 kohya_ss / sd-scripts 訓練 SDXL LoRA。
# 超參數全由環境變數帶入（launcher 經 pod env 注入），不需改本腳本。
# 前提：Network Volume 掛在 /workspace；資料集已上傳到 /workspace/datasets/<concept>/。
set -euo pipefail

VOL="${COMFY_VOLUME:-/workspace}"
CONCEPT="${TRAIN_CONCEPT:-stcklnd}"

DATASET_DIR="$VOL/datasets/$CONCEPT"
OUTPUT_DIR="$VOL/models/loras"
LOG_DIR="$VOL/training/logs/$CONCEPT"
DONE_MARKER="$VOL/training/$CONCEPT.train.done"     # launcher 輪詢此標記
FAIL_MARKER="$VOL/training/$CONCEPT.train.failed"

BASE_MODEL="$VOL/${TRAIN_BASE_MODEL:-models/checkpoints/dreamshaper_xl_v2_turbo.safetensors}"
RANK="${TRAIN_RANK:-16}"
ALPHA="${TRAIN_ALPHA:-8}"
LR="${TRAIN_LR:-1e-4}"
STEPS="${TRAIN_STEPS:-1500}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$VOL/training"
rm -f "$DONE_MARKER" "$FAIL_MARKER"

# 任一步失敗：寫失敗標記後以非零碼結束（launcher 視為訓練失敗）。
fail() {
  echo "訓練失敗：$1" >&2
  echo "$1" > "$FAIL_MARKER"
  exit 1
}

echo "==> 檢查輸入"
[ -d "$DATASET_DIR" ] || fail "找不到資料集目錄：$DATASET_DIR"
[ -f "$BASE_MODEL" ] || fail "找不到 base 模型：$BASE_MODEL（先跑 setup_volume.sh）"

# 取得 kohya_ss / sd-scripts。多數 ComfyUI/訓練 template 已內建；否則 clone。
SD_SCRIPTS="${SD_SCRIPTS_DIR:-/workspace/sd-scripts}"
if [ ! -d "$SD_SCRIPTS" ]; then
  echo "==> 取得 sd-scripts"
  git clone --depth 1 https://github.com/kohya-ss/sd-scripts "$SD_SCRIPTS" \
    || fail "clone sd-scripts 失敗"
  pip install -q -r "$SD_SCRIPTS/requirements.txt" || fail "安裝 sd-scripts 相依失敗"
fi

echo "==> 開始訓練 SDXL LoRA（concept=$CONCEPT rank=$RANK alpha=$ALPHA lr=$LR steps=$STEPS）"
cd "$SD_SCRIPTS"
# sd-scripts 的 SDXL LoRA 訓練。資料夾用 kohya 的 "<repeats>_<concept>" 慣例由呼叫端準備，
# 這裡直接指向 DATASET_DIR；caption 採同名 .txt。
python sdxl_train_network.py \
  --pretrained_model_name_or_path "$BASE_MODEL" \
  --train_data_dir "$DATASET_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --output_name "$CONCEPT" \
  --resolution 1024,1024 \
  --network_module networks.lora \
  --network_dim "$RANK" \
  --network_alpha "$ALPHA" \
  --learning_rate "$LR" \
  --max_train_steps "$STEPS" \
  --mixed_precision bf16 \
  --save_model_as safetensors \
  --caption_extension .txt \
  --logging_dir "$LOG_DIR" \
  2>&1 | tee "$LOG_DIR/train.log" \
  || fail "sd-scripts 訓練以非零碼結束"

echo "==> 訓練完成，輸出：$OUTPUT_DIR/$CONCEPT.safetensors"
echo "$OUTPUT_DIR/$CONCEPT.safetensors" > "$DONE_MARKER"
