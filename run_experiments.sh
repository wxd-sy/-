#!/bin/bash
# Dual-Mapping Experiments Runner
# Usage: bash run_experiments.sh

set -e

CKPT_BB="data/weights/rtdetr_r18.pth"
CKPT_DET="data/weights/rtdetr_r18vd_coco.pth"
DATASET="voc"
DATA_DIR="data"
NUM_CLASSES=20
EPOCHS=30
BATCH_SIZE=8
DEVICE="cuda"

echo "============================================"
echo " Dual-Mapping RT-DETR Experiments"
echo "============================================"
echo ""

# ---- 0. Pre-check ----
if [ ! -f "$CKPT_BB" ]; then
    echo "ERROR: Backbone checkpoint not found: $CKPT_BB"
    exit 1
fi
if [ ! -f "$CKPT_DET" ]; then
    echo "ERROR: Detection checkpoint not found: $CKPT_DET"
    exit 1
fi
if [ ! -d "$DATA_DIR/VOCdevkit/VOC2007/JPEGImages" ]; then
    echo "ERROR: VOC data not found at $DATA_DIR/VOCdevkit/VOC2007/JPEGImages"
    exit 1
fi
echo "All prerequisites found."
echo ""

# ---- A: VOC Baseline Frozen ----
echo ">>> Experiment A: VOC Baseline Frozen"
python rtdetr/train.py \
    --pretrained_ckpt "$CKPT_BB" --det_ckpt "$CKPT_DET" \
    --dataset "$DATASET" --data_dir "$DATA_DIR" --num_classes "$NUM_CLASSES" \
    --epochs 1 --baseline --batch_size "$BATCH_SIZE" --device "$DEVICE" \
    --exp_name expA_frozen
echo ""

# ---- B: VOC Full Fine-tune ----
echo ">>> Experiment B: VOC Full Fine-tune"
python rtdetr/train.py \
    --pretrained_ckpt "$CKPT_BB" --det_ckpt "$CKPT_DET" \
    --dataset "$DATASET" --data_dir "$DATA_DIR" --num_classes "$NUM_CLASSES" \
    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr 1e-4 --device "$DEVICE" \
    --full_finetune --exp_name expB_fullft
echo ""

# ---- C: VOC LoRA r=16 ----
echo ">>> Experiment C: VOC LoRA r=16"
python rtdetr/train.py \
    --pretrained_ckpt "$CKPT_BB" --det_ckpt "$CKPT_DET" \
    --dataset "$DATASET" --data_dir "$DATA_DIR" --num_classes "$NUM_CLASSES" \
    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr 1e-3 --device "$DEVICE" \
    --use_lora --lora_rank 16 --exp_name expC_lora16
echo ""

# ---- D: VOC Dual-Mapping ----
echo ">>> Experiment D: VOC Dual-Mapping"
python rtdetr/train.py \
    --pretrained_ckpt "$CKPT_BB" --det_ckpt "$CKPT_DET" \
    --dataset "$DATASET" --data_dir "$DATA_DIR" --num_classes "$NUM_CLASSES" \
    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr 1e-2 --device "$DEVICE" \
    --use_query_mapping --exp_name expD_dualmap
echo ""

# ---- E: VOC Backbone-Only Mapping ----
echo ">>> Experiment E: VOC Backbone-Only Mapping"
python rtdetr/train.py \
    --pretrained_ckpt "$CKPT_BB" --det_ckpt "$CKPT_DET" \
    --dataset "$DATASET" --data_dir "$DATA_DIR" --num_classes "$NUM_CLASSES" \
    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr 1e-2 --device "$DEVICE" \
    --no_query_mapping --exp_name expE_backbone
echo ""

# ---- F: VOC Latent Dim Scaling ----
echo ">>> Experiment F: VOC Latent Dim Scaling"
for dim in 256 512 1024 2048; do
    echo "  --- z_dim = $dim ---"
    python rtdetr/train.py \
        --pretrained_ckpt "$CKPT_BB" --det_ckpt "$CKPT_DET" \
        --dataset "$DATASET" --data_dir "$DATA_DIR" --num_classes "$NUM_CLASSES" \
        --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr 1e-2 --device "$DEVICE" \
        --use_query_mapping --latent_dim $dim --query_latent_dim $dim \
        --exp_name expF_z${dim}
done
echo ""

echo "============================================"
echo " All experiments complete!"
echo "============================================"
echo "Checkpoints saved under: checkpoints_rtdetr/"
echo "Results log: see each experiment's std output above."
