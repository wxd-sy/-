import os
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import json
from typing import List, Tuple, Optional


class YoloDetectionDataset(Dataset):
    """YOLO-format dataset: images in img_dir/*.jpg, labels in label_dir/*.txt.
       Each label line: class_id cx cy w h (all normalized 0-1)."""

    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        img_size: int = 640,
        transform: Optional[transforms.Compose] = None,
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size

        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        self.img_paths = []
        for ext in exts:
            self.img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
        self.img_paths = sorted(self.img_paths)

        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")

        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image = self.transform(image)

        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, f"{basename}.txt")

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls_id = int(float(parts[0]))
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    boxes.append([cx, cy, w, h])
                    labels.append(cls_id)

        if len(boxes) == 0:
            boxes = [[0.5, 0.5, 0.01, 0.01]]
            labels = [0]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "orig_size": torch.tensor([orig_h, orig_w]),
            "size": torch.tensor([self.img_size, self.img_size]),
            "img_id": idx,
        }
        return image, target


class CocoDetectionDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        img_size: int = 640,
        transform: Optional[transforms.Compose] = None,
    ):
        self.img_dir = img_dir
        self.img_size = img_size

        with open(ann_file) as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            self.img_to_anns.setdefault(img_id, []).append(ann)

        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        orig_w, orig_h = image.size
        image = self.transform(image)

        anns = self.img_to_anns.get(img_info["id"], [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x = x / orig_w
            y = y / orig_h
            w = w / orig_w
            h = h / orig_h
            boxes.append([x, y, w, h])
            labels.append(ann["category_id"] - 1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "orig_size": torch.tensor([orig_h, orig_w]),
            "size": torch.tensor([self.img_size, self.img_size]),
            "img_id": img_info["id"],
        }
        return image, target


def build_coco_loader(
    img_dir: str,
    ann_file: str = "",
    batch_size: int = 4,
    img_size: int = 640,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    parent = os.path.dirname(img_dir.rstrip("/"))
    label_dir = img_dir.replace("images", "labels") if "images" in img_dir else os.path.join(parent, "labels", os.path.basename(img_dir))

    if os.path.exists(ann_file):
        dataset = CocoDetectionDataset(img_dir, ann_file, img_size)
    elif os.path.isdir(label_dir) and len(glob.glob(os.path.join(label_dir, "*.txt"))) > 0:
        dataset = YoloDetectionDataset(img_dir, label_dir, img_size)
    else:
        raise FileNotFoundError(f"No COCO annotation at {ann_file}, and no YOLO labels at {label_dir}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=_collate_fn)


def _collate_fn(batch: list) -> Tuple[torch.Tensor, List[dict]]:
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets
