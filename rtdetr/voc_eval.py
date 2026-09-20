"""PASCAL VOC evaluation: class-balanced sampling, mAP@50/95, post-processing."""
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Tuple

from rtdetr.voc_loader import VOCDetectionDataset, VOC_TO_COCO, VOC_CLASSES

COCO_TO_VOC = {v: k for k, v in VOC_TO_COCO.items()}


def class_balanced_sample(
    root: str,
    image_set: str = "trainval",
    max_samples: int = 256,
    seed: int = 42,
) -> List[int]:
    """Sample indices class-balanced from VOC dataset."""
    dataset = VOCDetectionDataset(root, image_set)
    class_counts = defaultdict(list)

    for idx in range(len(dataset)):
        _, target = dataset[idx]
        labels = target["labels"].tolist()
        # filter to VOC classes only
        voc_labels = [l for l in labels if l in COCO_TO_VOC]
        seen = set()
        for l in voc_labels:
            if l not in seen:
                class_counts[l].append(idx)
                seen.add(l)

    rng = random.Random(seed)
    sampled = set()
    # per-class allocation
    num_classes = len(VOC_CLASSES)
    per_class = max(1, max_samples // num_classes)

    for coco_id in COCO_TO_VOC:
        indices = class_counts.get(coco_id, [])
        if indices:
            n = min(per_class, len(indices))
            sampled.update(rng.sample(indices, n))

    # fill remaining slots
    remaining = max_samples - len(sampled)
    all_idx = set(range(len(dataset))) - sampled
    if remaining > 0 and all_idx:
        sampled.update(rng.sample(sorted(all_idx), min(remaining, len(all_idx))))

    return sorted(sampled)


def filter_voc_predictions(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    orig_sizes: torch.Tensor,
    score_thresh: float = 0.05,  # lowered for cross-domain transfer
    nms_thresh: float = 0.5,
) -> List[Dict[str, torch.Tensor]]:
    """Filter COCO predictions (80-class) to VOC (20-class), apply NMS."""
    voc_indices = sorted(COCO_TO_VOC.keys())  # 20 VOC class indices

    results = []
    for b in range(pred_logits.shape[0]):
        scores = pred_logits[b, :, voc_indices].sigmoid()
        boxes = pred_boxes[b].clone()

        # convert boxes from cxcywh to xyxy for NMS
        orig_h, orig_w = orig_sizes[b].tolist()
        boxes_xyxy = cxcywh_to_xyxy(boxes)
        boxes_xyxy[:, [0, 2]] *= orig_w
        boxes_xyxy[:, [1, 3]] *= orig_h

        all_boxes = []
        all_scores = []
        all_labels = []

        for vi, coco_idx in enumerate(voc_indices):
            cls_scores = scores[:, vi]
            keep = cls_scores > score_thresh
            if not keep.any():
                continue
            cls_boxes = boxes_xyxy[keep]
            cls_s = cls_scores[keep]
            nms_keep = torchvision_nms(cls_boxes, cls_s, nms_thresh)
            all_boxes.append(cls_boxes[nms_keep])
            all_scores.append(cls_s[nms_keep])
            all_labels.append(torch.full_like(cls_s[nms_keep], coco_idx, dtype=torch.long))

        if all_boxes:
            results.append({
                "boxes": torch.cat(all_boxes),
                "scores": torch.cat(all_scores),
                "labels": torch.cat(all_labels),
            })
        else:
            results.append({
                "boxes": torch.zeros(0, 4),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long),
            })
    return results


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def torchvision_nms(boxes, scores, iou_thresh):
    try:
        from torchvision.ops import nms
        return nms(boxes, scores, iou_thresh)
    except ImportError:
        # fallback: sort by score, no NMS
        return torch.arange(len(scores), device=scores.device)


def compute_voc_ap(
    pred: Dict[str, torch.Tensor],
    gt: Dict[str, torch.Tensor],
    iou_thresh: float = 0.5,
) -> float:
    """Compute AP for a single image and a specific class."""
    if len(pred["scores"]) == 0 and len(gt["boxes"]) == 0:
        return 1.0
    if len(pred["scores"]) == 0:
        return 0.0

    sorted_idx = torch.argsort(pred["scores"], descending=True)
    pred_boxes = pred["boxes"][sorted_idx]
    pred_labels = pred["labels"][sorted_idx]

    gt_boxes = gt["boxes"]
    gt_labels = gt["labels"]

    tp = torch.zeros(len(pred_boxes))
    fp = torch.zeros(len(pred_boxes))
    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

    for i, (box, label) in enumerate(zip(pred_boxes, pred_labels)):
        gt_mask = (gt_labels == label) & ~gt_matched
        if not gt_mask.any():
            fp[i] = 1
            continue
        gt_box = gt_boxes[gt_mask]
        ious = box_iou(box.unsqueeze(0), gt_box)
        max_iou, max_idx = ious.max(dim=1)
        if max_iou >= iou_thresh:
            gt_indices = torch.where(gt_mask)[0]
            gt_matched[gt_indices[max_idx]] = True
            tp[i] = 1
        else:
            fp[i] = 1

    tp_cumsum = tp.cumsum(dim=0)
    fp_cumsum = fp.cumsum(dim=0)
    eps = 1e-8
    recalls = tp_cumsum / max(len(gt_boxes), 1)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + eps)

    # 11-point interpolation
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max() / 11.0
    return ap.item()


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes (xyxy format)."""
    x1 = torch.max(boxes1[:, 0:1], boxes2[:, 0:1].T)
    y1 = torch.max(boxes1[:, 1:2], boxes2[:, 1:2].T)
    x2 = torch.min(boxes1[:, 2:3], boxes2[:, 2:3].T)
    y2 = torch.min(boxes1[:, 3:4], boxes2[:, 3:4].T)

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
    return inter / (union + 1e-8)


def evaluate_voc_map(
    mapping,
    encoder,
    decoder,
    val_loader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute VOC mAP@50 and mAP@50:95 using mapping backbone."""
    mapping.eval()
    encoder.eval()
    decoder.eval()

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            feats = mapping(images)
            encoded = encoder(feats)
            decoder_out = decoder(encoded)

            pred_logits = decoder_out["pred_logits"]
            pred_boxes = decoder_out["pred_boxes"]
            orig_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)

            preds = filter_voc_predictions(pred_logits, pred_boxes, orig_sizes)
            for p, t in zip(preds, targets):
                # GT: cxcywh normalized → xyxy absolute pixels (match pred format)
                gt_box = t["boxes"].cpu()
                cx, cy, w, h = gt_box[:, 0], gt_box[:, 1], gt_box[:, 2], gt_box[:, 3]
                orig_h, orig_w = t["orig_size"].tolist()
                x1 = (cx - w / 2) * orig_w
                y1 = (cy - h / 2) * orig_h
                x2 = (cx + w / 2) * orig_w
                y2 = (cy + h / 2) * orig_h
                gt_voc = {
                    "boxes": torch.stack([x1, y1, x2, y2], dim=-1),
                    "labels": t["labels"].cpu(),
                }
                # Move preds to CPU for consistent AP computation
                p_cpu = {k: v.cpu() for k, v in p.items()}
                all_preds.append(p_cpu)
                all_gts.append(gt_voc)

    # Per-class AP
    class_aps = defaultdict(list)
    for pred, gt in zip(all_preds, all_gts):
        unique_labels = set(pred["labels"].tolist() + gt["labels"].tolist())
        for label in unique_labels:
            p_cls = {k: v[pred["labels"] == label] for k, v in pred.items()}
            g_cls = {k: v[gt["labels"] == label] for k, v in gt.items()}
            ap = compute_voc_ap(p_cls, g_cls, iou_thresh=0.5)
            class_aps[label].append(ap)

    mAP50 = np.mean([np.mean(aps) for aps in class_aps.values()]) if class_aps else 0.0
    return {"mAP@50": mAP50, "num_classes": len(class_aps)}
