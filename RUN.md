# Dual-Mapping Networks for RT-DETR — 实验运行指南

## 一、项目概述

复现 Mapping Networks (arXiv:2602.19134v1) 在 RT-DETR 目标检测架构上的扩展：
**Dual-Mapping** — 同时用 latent vector 映射 Backbone 卷积权重和 Decoder 的 query_pos_head。

```
z_backbone(5×1024d) → OnTheFlyProjection → ΔW_conv → W_pretrained + ΔW (1,843x 压缩)
z_query(1×1024d)   → OnTheFlyProjection → ΔW_qp   → query_pos_head + ΔW
```

**总可训练参数: 6,144 | 目标参数: 11.3M | 压缩比: 1,843x**

---

## 二、环境要求

- Python 3.10~3.12, PyTorch >= 2.0, CUDA (推荐 48GB GPU)
- pip install torch torchvision tqdm numpy matplotlib scipy peft

```bash
pip install torch torchvision tqdm numpy matplotlib scipy peft
```

---

## 三、数据准备

### 3.1 下载权重文件

下载到 `data/weights/` 目录：

| 文件 | 大小 | 用途 |
|------|------|------|
| `rtdetr_r18.pth` | 43MB | Backbone 预训练权重 |
| `rtdetr_r18vd_coco.pth` | 91MB | 完整 RT-DETR 检测 checkpoint (encoder+decoder) |

```bash
mkdir -p data/weights
# 从原始 mappingNetwork 项目复制权重到此处
```

### 3.2 COCO128 数据集（验证用）

```bash
mkdir -p data/coco128
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
unzip -o coco128.zip -d data && mv data/coco128/images data/coco128/ 2>/dev/null; true
rm coco128.zip
```

### 3.3 PASCAL VOC 2007 数据集（主实验）

```bash
mkdir -p data
# 将 VOCtrainval.tar 放到 data/ 目录下
tar xf data/VOCtrainval.tar -C data/
# 数据路径: data/VOCdevkit/VOC2007/JPEGImages/ (5011 张)
```

---

## 四、快速验证链路

### 4.1 纯代码验证（无数据需求）

```bash
python rtdetr/train.py --pretrained_ckpt data/weights/rtdetr_r18.pth \
    --dummy --dummy_samples 4 --batch_size 2 --epochs 1 \
    --use_query_mapping
```

期望输出: `Compression ratio: 1843.1x`，不报错。

### 4.2 COCO128 完整链路验证（需要 COCO128 数据）

```bash
python rtdetr/train.py --pretrained_ckpt data/weights/rtdetr_r18.pth \
    --det_ckpt data/weights/rtdetr_r18vd_coco.pth \
    --epochs 3 --batch_size 4 --use_query_mapping \
    --exp_name coco128_verify
```

---

## 五、正式实验矩阵

以下实验使用 VOC 数据集（需准备 COCO128 用于 Full Fine-tune 的预训练验证）。

### 实验 A: VOC Baseline Frozen（精度下限）

```bash
python rtdetr/train.py --pretrained_ckpt data/weights/rtdetr_r18.pth \
    --det_ckpt data/weights/rtdetr_r18vd_coco.pth \
    --dataset voc --data_dir data --num_classes 20 \
    --epochs 1 --baseline --batch_size 8 \
    --exp_name expA_frozen
```

### 实验 B: VOC Full Fine-tune（精度上限, 30 epochs, ~5-8h GPU）

```bash
python rtdetr/train.py --pretrained_ckpt data/weights/rtdetr_r18.pth \
    --det_ckpt data/weights/rtdetr_r18vd_coco.pth \
    --dataset voc --data_dir data --num_classes 20 \
    --epochs 30 --batch_size 8 --lr 1e-4 \
    --full_finetune --exp_name expB_fullft
```

### 实验 C: VOC LoRA r=16（PEFT 对标）

```bash
python rtdetr/train.py --pretrained_ckpt data/weights/rtdetr_r18.pth \
    --det_ckpt data/weights/rtdetr_r18vd_coco.pth \
    --dataset voc --data_dir data --num_classes 20 \
    --epochs 30 --batch_size 8 --lr 1e-3 \
    --use_lora --lora_rank 16 --exp_name expC_lora16
```

### 实验 D: VOC Dual-Mapping（主实验）

```bash
python rtdetr/train.py --pretrained_ckpt data/weights/rtdetr_r18.pth \
    --det_ckpt data/weights/rtdetr_r18vd_coco.pth \
    --dataset voc --data_dir data --num_classes 20 \
    --epochs 30 --batch_size 8 --lr 1e-2 \
    --use_query_mapping --exp_name expD_dualmap
```

### 实验 E: VOC Backbone-Only Mapping（消融 — query贡献）

```bash
python rtdetr/train.py --pretrained_ckpt data/weights/rtdetr_r18.pth \
    --det_ckpt data/weights/rtdetr_r18vd_coco.pth \
    --dataset voc --data_dir data --num_classes 20 \
    --epochs 30 --batch_size 8 --lr 1e-2 \
    --no_query_mapping --exp_name expE_backbone
```

### 实验 F: VOC Latent Dim Scaling（论文关键消融）

```bash
for dim in 256 512 1024 2048; do
    python rtdetr/train.py --pretrained_ckpt data/weights/rtdetr_r18.pth \
        --det_ckpt data/weights/rtdetr_r18vd_coco.pth \
        --dataset voc --data_dir data --num_classes 20 \
        --epochs 30 --batch_size 8 --lr 1e-2 \
        --use_query_mapping --latent_dim $dim --query_latent_dim $dim \
        --exp_name expF_z${dim}
done
```

### 实验 G: COCO→COCO128 同域 Baseline

```bash
python rtdetr/train.py --pretrained_ckpt data/weights/rtdetr_r18.pth \
    --det_ckpt data/weights/rtdetr_r18vd_coco.pth \
    --epochs 10 --batch_size 8 \
    --use_query_mapping --exp_name expG_coco128
```

---

## 六、结果记录模板

每完成一个实验，记录以下指标：

| 实验 | mAP@50 | mAP@50:95 | Trainable Params | Loss(末epoch) | 训练时间 |
|------|--------|-----------|-----------------|---------------|----------|
| A: Frozen | ? | ? | 0 | ? | 2min |
| B: Full FT | ? | ? | 40M | ? | 5-8h |
| C: LoRA r=16 | ? | ? | ~400K | ? | 3-5h |
| D: Dual-Map | ? | ? | 6,144 | ? | 2-4h |
| E: Backbone only | ? | ? | 5,120 | ? | 2-4h |
| F: z=256 | ? | ? | 1,536 | ? | 2h |
| F: z=512 | ? | ? | 3,072 | ? | 2h |
| F: z=1024 | ? | ? | 6,144 | ? | 2-4h |
| F: z=2048 | ? | ? | 12,288 | ? | 4-6h |

---

## 七、文件结构

```
dual_mapping_deliverable/
├── config.py / config_rtdetr.py     # 配置文件
├── mapping_network.py               # OnTheFlyProjection + MappingNetwork
├── bn_fuse.py                       # BN 融合工具
├── losses.py                        # KL stability + alignment + smoothness
├── data_loader.py / train.py / trainer.py / eval.py  # MNIST 实验
├── functional_cnn.py / target_cnn_spec.py
├── requirements.txt
├── rtdetr/
│   ├── train.py / trainer.py        # RT-DETR 训练入口
│   ├── mapping_backbone.py          # ⭐ MappingBackbone (核心)
│   ├── encoder_decoder.py           # 加载 frozen encoder/decoder + wrapper
│   ├── functional_presnet.py        # PResNet forward (conv + fused)
│   ├── spec.py / query_pos_spec.py  # 层规格生成
│   ├── coco_loader.py               # COCO/YOLO 数据加载
│   ├── voc_loader.py                # VOC 数据加载
│   └── voc_eval.py                  # mAP 评估
├── RT-DETR-main/                    # 官方 RT-DETR 源码（导入用）
└── data/weights/                    # 权重文件（需手动放入）
```

---

## 八、故障排查

| 症状 | 解决 |
|------|------|
| `No module named 'torch'` | `pip install torch` |
| `No module named 'scipy'` | `pip install scipy` |
| `No module named 'peft'` | `pip install peft` |
| `FileNotFoundError: data/weights/rtdetr_r18.pth` | 确认权重已放入正确路径 |
| `parameter name can't contain "."` | 用最新 mapping_backbone.py |
| CUDA OOM | 减小 `--batch_size`（试 4 或 2） |
| VOC 数据未找到 | 确认 `data/VOCdevkit/VOC2007/JPEGImages/` 存在 |

---

## 九、一键运行所有实验

```bash
bash run_experiments.sh
```
