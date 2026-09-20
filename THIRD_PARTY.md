# 第三方组件与许可声明

本仓库中的自有代码（`rtdetr/`、`mapping_network.py`、`config*.py`、`losses.py`、`bn_fuse.py`、
`functional_cnn.py`、`target_cnn_spec.py`、`data_loader.py`、`train.py`、`trainer.py`、`eval.py` 等）
为本人开发，用于软件著作权登记。

## 1. RT-DETR 官方开源实现（`RT-DETR-main/`）

- 来源：<https://github.com/lyuwenyu/RT-DETR>（`rtdetr_pytorch` 目录）
- 许可：Apache License 2.0
- 本仓库中的用途：作为**被冻结的检测模型**（编码器、解码器、匹配器、损失函数）与依赖实现，
  代码基本保持原样，版权归原作者所有。
- 注意：本仓库收录的该目录副本**未包含其 Apache-2.0 LICENSE 文件**，如需再分发，请从上游仓库
  获取并随附 LICENSE 与 NOTICE。上游引用信息：

  ```
  @misc{lyuwenyu2023rtdetr,
    title={RT-DETR: DETRs Beat YOLOs on Real-time Object Detection},
    author={Lv, Wenyu and others},
    year={2023},
    howpublished={\url{https://github.com/lyuwenyu/RT-DETR}}
  }
  ```

## 2. 预训练权重（未入库）

| 文件 | 来源 | 说明 |
|---|---|---|
| `rtdetr_r18.pth` | RT-DETR 官方发布的 ResNet-18-vd 主干权重 | 需自行下载放入 `data/weights/` |
| `rtdetr_r18vd_coco.pth` | RT-DETR 官方发布的 R18 COCO 检测器权重 | 需自行下载放入 `data/weights/` |

## 3. 数据集（未入库）

- PASCAL VOC 2007（`VOCtrainval_06-Nov-2007`），版权归原作者与发布方所有，仅用于实验。

## 4. Python 依赖

PyTorch、torchvision、NumPy、SciPy、tqdm、Pillow 等遵循各自开源许可协议。
