"""RT-DETR 推理引擎。

两个可切换的引擎：
  official : 官方 RT-DETR-R18-vd（COCO 80 类，mAP 高，作为质检主力）
  dualmap  : 实验 D 的 Dual-Mapping 压缩模型（仅 6144 个可训练参数，压缩 1843×）

模型权重全部复用桌面上已有的文件，不需要下载。
"""
from __future__ import annotations

import base64
import io
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchvision.transforms as T
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent                 # .../质检前端
PROJ_DIR = BASE_DIR.parent                                 # .../智能质检
VENDOR = BASE_DIR / "vendor"                               # 自带的推理依赖
RTDETR_SRC = VENDOR / "RT-DETR-main" / "RT-DETR-main" / "rtdetr_pytorch"

WEIGHTS = {
    "backbone": PROJ_DIR / "测试文件" / "rtdetr_r18.pth",
    "detector": PROJ_DIR / "测试文件" / "rtdetr_r18vd_coco.pth",
    "dualmap": PROJ_DIR / "测试文件" / "expD_results" / "expD_dualmap_best.pt",
}

for p in (str(VENDOR), str(RTDETR_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

COCO_CLASSES = [
    ("person", "人"), ("bicycle", "自行车"), ("car", "汽车"), ("motorcycle", "摩托车"),
    ("airplane", "飞机"), ("bus", "公交车"), ("train", "火车"), ("truck", "卡车"),
    ("boat", "船"), ("traffic light", "交通灯"), ("fire hydrant", "消防栓"), ("stop sign", "停止标志"),
    ("parking meter", "停车计时器"), ("bench", "长椅"), ("bird", "鸟"), ("cat", "猫"),
    ("dog", "狗"), ("horse", "马"), ("sheep", "羊"), ("cow", "牛"),
    ("elephant", "大象"), ("bear", "熊"), ("zebra", "斑马"), ("giraffe", "长颈鹿"),
    ("backpack", "背包"), ("umbrella", "雨伞"), ("handbag", "手提包"), ("tie", "领带"),
    ("suitcase", "行李箱"), ("frisbee", "飞盘"), ("skis", "滑雪板"), ("snowboard", "单板滑雪"),
    ("sports ball", "球"), ("kite", "风筝"), ("baseball bat", "棒球棒"), ("baseball glove", "棒球手套"),
    ("skateboard", "滑板"), ("surfboard", "冲浪板"), ("tennis racket", "网球拍"), ("bottle", "瓶子"),
    ("wine glass", "酒杯"), ("cup", "杯子"), ("fork", "叉子"), ("knife", "刀"),
    ("spoon", "勺子"), ("bowl", "碗"), ("banana", "香蕉"), ("apple", "苹果"),
    ("sandwich", "三明治"), ("orange", "橙子"), ("broccoli", "西兰花"), ("carrot", "胡萝卜"),
    ("hot dog", "热狗"), ("pizza", "披萨"), ("donut", "甜甜圈"), ("cake", "蛋糕"),
    ("chair", "椅子"), ("couch", "沙发"), ("potted plant", "盆栽"), ("bed", "床"),
    ("dining table", "餐桌"), ("toilet", "马桶"), ("tv", "电视"), ("laptop", "笔记本"),
    ("mouse", "鼠标"), ("remote", "遥控器"), ("keyboard", "键盘"), ("cell phone", "手机"),
    ("microwave", "微波炉"), ("oven", "烤箱"), ("toaster", "烤面包机"), ("sink", "水槽"),
    ("refrigerator", "冰箱"), ("book", "书"), ("clock", "时钟"), ("vase", "花瓶"),
    ("scissors", "剪刀"), ("teddy bear", "泰迪熊"), ("hair drier", "吹风机"), ("toothbrush", "牙刷"),
]

IMG_SIZE = 640
MIN_CONF_FLOOR = 0.01


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cuda_name() -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "CUDA"


def _letterbox(img: Image.Image, size: int = IMG_SIZE):
    """等比缩放 + 灰边填充，返回 (tensor, scale, pad_x, pad_y)。"""
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), ((size - nw) // 2, (size - nh) // 2))
    tensor = T.ToTensor()(canvas).unsqueeze(0)
    return tensor, scale, (size - nw) // 2, (size - nh) // 2


def _postprocess(output, conf: float, iou: float, scale: float, pad_x: int, pad_y: int,
                 orig_w: int, orig_h: int, class_filter: Optional[List[int]] = None) -> List[dict]:
    logits = output["pred_logits"][0].sigmoid().float()
    boxes = output["pred_boxes"][0].float()          # cxcywh, 归一化到 640 letterbox 画布
    if class_filter is not None:
        logits = logits[:, class_filter]

    scores, labels = logits.max(dim=-1)
    keep = scores > conf
    if not bool(keep.any()):
        return []
    scores, labels, boxes = scores[keep], labels[keep], boxes[keep]

    # cxcywh -> xyxy（画布像素）
    cx, cy, bw, bh = boxes.unbind(-1)
    xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1) * IMG_SIZE
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, orig_w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, orig_h)

    # 同一实例的重复框：RT-DETR 本身是集合预测，这里按类别做轻量 NMS
    if iou < 1.0 and len(scores) > 1:
        from torchvision.ops import batched_nms
        idx = batched_nms(xyxy, scores, labels, iou)
        xyxy, scores, labels = xyxy[idx], scores[idx], labels[idx]

    if class_filter is not None:
        inv = {new: old for new, old in enumerate(class_filter)}
        labels = torch.tensor([inv[i.item()] for i in labels], dtype=torch.long)

    area_total = float(orig_w * orig_h)
    dets = []
    order = torch.argsort(scores, descending=True)
    for i in order.tolist():
        x1, y1, x2, y2 = [round(float(v), 1) for v in xyxy[i].tolist()]
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        cid = int(labels[i].item())
        en, zh = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else ("object", "目标")
        area = max(0.0, (x2 - x1) * (y2 - y1))
        dets.append({
            "cls_id": cid, "cls": en, "cls_zh": zh,
            "conf": round(float(scores[i].item()), 4),
            "xyxy": [x1, y1, x2, y2],
            "area_ratio": round(area / area_total * 100, 2) if area_total else 0.0,
        })
    return dets


class _OfficialEngine:
    id = "official"
    label = "RT-DETR-R18 官方"
    note = "COCO 80 类 · 完整权重 · 质检主力"

    def __init__(self):
        self.device = _device()
        self.model = None

    def load(self):
        if self.model is not None:
            return
        from rtdetr.encoder_decoder import create_encoder_decoder
        from src.nn.backbone.presnet import PResNet

        ckpt = torch.load(WEIGHTS["detector"], map_location="cpu", weights_only=False)
        state = ckpt["ema"]["module"]
        backbone = PResNet(depth=18, variant="d", num_stages=4, return_idx=[1, 2, 3],
                           act="relu", freeze_at=-1, freeze_norm=True, pretrained=False)
        backbone.load_state_dict({k[len("backbone."):]: v for k, v in state.items()
                                  if k.startswith("backbone.")})
        encoder, decoder = create_encoder_decoder(torch.device("cpu"))
        encoder.load_state_dict({k[len("encoder."):]: v for k, v in state.items()
                                 if k.startswith("encoder.")})
        decoder.load_state_dict({k[len("decoder."):]: v for k, v in state.items()
                                 if k.startswith("decoder.")})
        backbone.to(self.device).eval()
        encoder.to(self.device).eval()
        decoder.to(self.device).eval()
        self.model = (backbone, encoder, decoder)

    def infer(self, img: Image.Image, conf: float, iou: float) -> dict:
        self.load()
        backbone, encoder, decoder = self.model
        tensor, scale, px, py = _letterbox(img)
        tensor = tensor.to(self.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            output = decoder(encoder(backbone(tensor)), None)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000
        dets = _postprocess(output, conf, iou, scale, px, py, img.width, img.height)
        return {"detections": dets, "infer_ms": round(infer_ms, 1)}


class _DualMapEngine:
    id = "dualmap"
    label = "Dual-Mapping 实验模型"
    note = "6,144 参数 · 压缩 1843× · 研究对照"

    def __init__(self):
        self.device = _device()
        self.model = None

    def load(self):
        if self.model is not None:
            return
        from config_rtdetr import RTDETRConfig
        from rtdetr.mapping_backbone import MappingBackbone
        from rtdetr.encoder_decoder import load_frozen_ed, apply_querypos_fused

        cfg = RTDETRConfig(device=str(self.device), num_classes=80)
        pretrained = torch.load(WEIGHTS["backbone"], map_location="cpu", weights_only=False)
        mapping = MappingBackbone(
            pretrained_state=pretrained if isinstance(pretrained, dict) else {},
            depth=cfg.backbone_depth, latent_dim=cfg.latent_dim, alpha=cfg.alpha,
            layerwise=cfg.layerwise, block_size=cfg.block_size,
            cache_projections=cfg.cache_projections, use_tanh=cfg.use_tanh,
            output_gain=cfg.output_gain, bn_fuse=cfg.bn_fuse,
            use_query_mapping=True, query_latent_dim=cfg.query_latent_dim,
            query_output_gain=cfg.query_output_gain,
        )
        ckpt = torch.load(WEIGHTS["dualmap"], map_location="cpu", weights_only=False)
        # _ref_feats_* 是训练期对齐用的参考特征 buffer，推理不需要，且形状随 batch 变化
        model_state = {k: v for k, v in ckpt["model"].items()
                       if not k.startswith("_ref_feats")}
        missing, unexpected = mapping.load_state_dict(model_state, strict=False)
        print(f"[dualmap] missing={len(missing)} unexpected={len(unexpected)}")
        mapping.to(self.device).eval()

        encoder, decoder, _, _ = load_frozen_ed(str(WEIGHTS["detector"]), self.device, num_classes=80)
        apply_querypos_fused(decoder, mapping.generate_querypos_delta(return_smoothness=False))
        self.model = (mapping, encoder, decoder)

    def infer(self, img: Image.Image, conf: float, iou: float) -> dict:
        self.load()
        mapping, encoder, decoder = self.model
        tensor, scale, px, py = _letterbox(img)
        tensor = tensor.to(self.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            output = decoder(encoder(mapping(tensor)), None)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000
        dets = _postprocess(output, conf, iou, scale, px, py, img.width, img.height)
        return {"detections": dets, "infer_ms": round(infer_ms, 1)}


ENGINES = {"official": _OfficialEngine(), "dualmap": _DualMapEngine()}


def engine_info() -> dict:
    return {
        "device": str(_device()),
        "gpu": _cuda_name(),
        "torch": torch.__version__,
        "image_size": IMG_SIZE,
        "classes": [{"id": i, "en": en, "zh": zh} for i, (en, zh) in enumerate(COCO_CLASSES)],
        "engines": [
            {"id": e.id, "label": e.label, "note": e.note, "loaded": e.model is not None}
            for e in ENGINES.values()
        ],
        "weights": {k: {"path": str(v), "exists": v.exists()} for k, v in WEIGHTS.items()},
    }


def detect(image_bytes: bytes, conf: float = 0.25, iou: float = 0.55,
           engine: str = "official") -> dict:
    conf = max(MIN_CONF_FLOOR, min(0.99, float(conf)))
    iou = max(0.05, min(1.0, float(iou)))
    eng = ENGINES.get(engine) or ENGINES["official"]

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")

    t0 = time.perf_counter()
    result = eng.infer(img, conf, iou)
    total_ms = (time.perf_counter() - t0) * 1000

    classes: Dict[str, dict] = {}
    for d in result["detections"]:
        c = classes.setdefault(d["cls"], {"en": d["cls"], "zh": d["cls_zh"], "count": 0,
                                          "best": 0.0, "area": 0.0})
        c["count"] += 1
        c["best"] = max(c["best"], d["conf"])
        c["area"] += d["area_ratio"]

    return {
        "engine": {"id": eng.id, "label": eng.label},
        "image": {"width": img.width, "height": img.height},
        "detections": result["detections"],
        "summary": {
            "total": len(result["detections"]),
            "classes": len(classes),
            "avg_conf": round(sum(d["conf"] for d in result["detections"]) / max(1, len(result["detections"])), 4),
            "infer_ms": result["infer_ms"],
            "total_ms": round(total_ms, 1),
            "breakdown": sorted(classes.values(), key=lambda c: c["count"], reverse=True),
        },
        "params": {"conf": conf, "iou": iou, "size": IMG_SIZE},
    }
