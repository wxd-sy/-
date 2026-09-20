"""PASCAL VOC dataset loader that outputs SetCriterion-compatible targets
with VOC classes mapped to COCO class indices."""
import os
import glob
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Optional

# VOC class name → COCO 0-indexed class ID (80 classes total)
VOC_TO_COCO = {
    'aeroplane': 4, 'bicycle': 1, 'bird': 14, 'boat': 8,
    'bottle': 39, 'bus': 5, 'car': 2, 'cat': 15,
    'chair': 56, 'cow': 19, 'diningtable': 60, 'dog': 16,
    'horse': 17, 'motorbike': 3, 'person': 0, 'pottedplant': 58,
    'sheep': 18, 'sofa': 57, 'train': 6, 'tvmonitor': 62,
}

VOC_CLASSES = list(VOC_TO_COCO.keys())


class VOCDetectionDataset(Dataset):
    """PASCAL VOC detection dataset. Expects VOCdevkit/VOC2007/ structure."""

    def __init__(
        self,
        root: str,
        image_set: str = "trainval",
        img_size: int = 640,
        transform: Optional[transforms.Compose] = None,
    ):
        self.img_size = img_size
        self.root = root

        img_dir = os.path.join(root, "VOCdevkit", "VOC2007", "JPEGImages")
        ann_dir = os.path.join(root, "VOCdevkit", "VOC2007", "Annotations")
        splits_dir = os.path.join(root, "VOCdevkit", "VOC2007", "ImageSets", "Main")

        split_file = os.path.join(splits_dir, f"{image_set}.txt")
        if not os.path.exists(split_file):
            # Fallback: just use all JPEG images
            self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        else:
            with open(split_file) as f:
                ids = [line.strip() for line in f if line.strip()]
            self.img_paths = [os.path.join(img_dir, f"{img_id}.jpg") for img_id in ids]

        self.ann_dir = ann_dir
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
        ann_path = os.path.join(self.ann_dir, f"{basename}.xml")

        boxes = []
        labels = []

        if os.path.exists(ann_path):
            tree = ET.parse(ann_path)
            root_elem = tree.getroot()
            for obj in root_elem.findall("object"):
                name = obj.find("name").text.strip().lower()
                if name not in VOC_TO_COCO:
                    continue
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                cx = ((xmin + xmax) / 2) / orig_w
                cy = ((ymin + ymax) / 2) / orig_h
                w = (xmax - xmin) / orig_w
                h = (ymax - ymin) / orig_h
                boxes.append([cx, cy, w, h])
                labels.append(VOC_TO_COCO[name])

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


def build_voc_loader(
    root: str,
    image_set: str = "trainval",
    batch_size: int = 4,
    img_size: int = 640,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    dataset = VOCDetectionDataset(root, image_set, img_size)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )


def _collate_fn(batch: list) -> Tuple[torch.Tensor, List[dict]]:
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets
