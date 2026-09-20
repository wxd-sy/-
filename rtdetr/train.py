import argparse
import os
import sys
# Allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from config_rtdetr import RTDETRConfig
from rtdetr.spec import get_total_target_params
from rtdetr.mapping_backbone import MappingBackbone
from rtdetr.trainer import train_rtdetr_backbone


def _download_coco128(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    images_dir = os.path.join(data_dir, "images", "train2017")
    if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0:
        print(f"COCO128 already present in {data_dir}")
        return

    print(f"Downloading COCO128 to {data_dir}...")
    import zipfile, requests, io
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
    r = requests.get(url, timeout=120)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(data_dir)
    print("COCO128 download complete.")


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100, img_size=640):
        self.num_samples = num_samples
        self.img_size = img_size
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        img = torch.randn(3, self.img_size, self.img_size)
        return img, {}


def _voc_collate(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def main():
    parser = argparse.ArgumentParser(description="RT-DETR Backbone Mapping Training")
    parser.add_argument("--pretrained_ckpt", type=str, default="")
    parser.add_argument("--det_ckpt", type=str, default="", help="Full RT-DETR detection checkpoint (encoder+decoder)")
    parser.add_argument("--depth", type=int, default=18, choices=[18, 34, 50, 101])
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--layerwise", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--max_samples", type=int, default=0, help="Limit dataset size for quick test")
    parser.add_argument("--baseline", action="store_true", help="Run frozen baseline without mapping")
    parser.add_argument("--eval_every", type=int, default=0, help="Evaluate mAP every N epochs (0=disabled)")
    parser.add_argument("--data_dir", type=str, default="./data/coco128")
    parser.add_argument("--dataset", type=str, default="coco128", choices=["coco128", "voc"])
    parser.add_argument("--output_gain", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp_name", type=str, default="mapping_rtdetr")
    parser.add_argument("--dummy", action="store_true", help="Use random dummy data for quick test")
    parser.add_argument("--dummy_samples", type=int, default=64)
    parser.add_argument("--use_query_mapping", action="store_true", default=True, help="Enable query_pos_head mapping")
    parser.add_argument("--no_query_mapping", action="store_true", help="Disable query_pos_head mapping")
    parser.add_argument("--query_latent_dim", type=int, default=1024)
    parser.add_argument("--query_output_gain", type=float, default=0.01)
    parser.add_argument("--no_bn_fuse", action="store_true", help="Disable BN fuse")
    parser.add_argument("--lambda_align", type=float, default=0.01)
    parser.add_argument("--num_classes", type=int, default=80, help="Number of target classes (20 for VOC)")
    parser.add_argument("--full_finetune", action="store_true", help="Full fine-tune (unfreeze all params)")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA PEFT")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    args = parser.parse_args()

    cfg = RTDETRConfig(
        backbone_depth=args.depth,
        latent_dim=args.latent_dim,
        layerwise=args.layerwise,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        img_size=args.img_size,
        data_dir=args.data_dir,
        output_gain=args.output_gain,
        device=args.device,
        exp_name=args.exp_name,
        pretrained_ckpt=args.pretrained_ckpt,
        use_query_mapping=args.use_query_mapping and not args.no_query_mapping,
        query_latent_dim=args.query_latent_dim,
        query_output_gain=args.query_output_gain,
        bn_fuse=not args.no_bn_fuse,
        lambda_align=args.lambda_align,
        num_classes=args.num_classes,
        full_finetune=args.full_finetune,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
    )

    print("=" * 60)
    print("RT-DETR Backbone Mapping Training")
    print("=" * 60)
    print(f"  Backbone: PResNet-{cfg.backbone_depth}-vd")
    print(f"  Num classes: {cfg.num_classes}")
    print(f"  Latent dim: {cfg.latent_dim}")
    print(f"  Layerwise: {cfg.layerwise}")
    print(f"  Target conv params: {get_total_target_params(cfg.backbone_depth):,}")
    print(f"  BN fuse: {cfg.bn_fuse}")
    print(f"  Mode: {'full_finetune' if cfg.full_finetune else 'LoRA r=' + str(cfg.lora_rank) if cfg.use_lora else 'mapping'}")
    print(f"  Query mapping: {cfg.use_query_mapping}")
    if cfg.use_query_mapping:
        print(f"    query_latent_dim: {cfg.query_latent_dim}")
        print(f"    query_output_gain: {cfg.query_output_gain}")
    print(f"  Lambda align: {cfg.lambda_align}")
    print(f"  Dummy data: {args.dummy}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Detection ckpt: {args.det_ckpt or 'none'}")
    print("=" * 60)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    pretrained_state = {}
    if cfg.pretrained_ckpt and os.path.exists(cfg.pretrained_ckpt):
        print(f"\nLoading pretrained checkpoint: {cfg.pretrained_ckpt}")
        ckpt = torch.load(cfg.pretrained_ckpt, map_location="cpu", weights_only=False)
        pretrained_state = ckpt if isinstance(ckpt, dict) else {}

    mapping = MappingBackbone(
        pretrained_state=pretrained_state,
        depth=cfg.backbone_depth,
        latent_dim=cfg.latent_dim,
        alpha=cfg.alpha,
        layerwise=cfg.layerwise,
        block_size=cfg.block_size,
        cache_projections=cfg.cache_projections,
        use_tanh=cfg.use_tanh,
        output_gain=cfg.output_gain,
        bn_fuse=cfg.bn_fuse,
        use_query_mapping=cfg.use_query_mapping and not args.full_finetune and not args.use_lora,
        query_latent_dim=cfg.query_latent_dim,
        query_output_gain=cfg.query_output_gain,
        trainable_backbone=args.full_finetune or args.use_lora,
    ).to(device)

    n_trainable = mapping.count_trainable_params()
    n_target = mapping.count_target_params()
    n_pretrained = sum(v.numel() for v in mapping.pretrained_conv.values())
    print(f"\n  Pretrained conv params loaded: {n_pretrained:,}")
    print(f"  Trainable params: {n_trainable:,}")
    print(f"  Target conv params: {n_target:,}")
    if n_trainable > 0:
        print(f"  Compression ratio: {n_target / n_trainable:.1f}x")

    encoder, decoder, criterion = None, None, None
    qp_orig_state = None
    if args.det_ckpt and os.path.exists(args.det_ckpt) and not args.dummy:
        from rtdetr.encoder_decoder import load_frozen_ed
        encoder, decoder, criterion, qp_orig_state = load_frozen_ed(args.det_ckpt, device, num_classes=cfg.num_classes)

    if args.dummy:
        print(f"\nUsing dummy dataset ({args.dummy_samples} random images)")
        encoder = decoder = criterion = None
        train_ds = DummyDataset(num_samples=args.dummy_samples, img_size=cfg.img_size)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True
        )
    elif args.dataset == "voc":
        from rtdetr.voc_loader import build_voc_loader
        if args.max_samples > 0:
            from rtdetr.voc_eval import class_balanced_sample
            indices = class_balanced_sample(cfg.data_dir, "trainval", args.max_samples)
            full_ds = build_voc_loader(cfg.data_dir, "trainval", cfg.batch_size, cfg.img_size, cfg.num_workers).dataset
            train_ds = torch.utils.data.Subset(full_ds, indices)
            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=cfg.batch_size, shuffle=True,
                num_workers=cfg.num_workers,
                collate_fn=_voc_collate,
            )
        else:
            train_loader = build_voc_loader(
                cfg.data_dir, "trainval", cfg.batch_size, cfg.img_size, cfg.num_workers
            )
    else:
        from rtdetr.coco_loader import build_coco_loader
        img_dir = os.path.join(cfg.data_dir, "images", "train2017")
        ann_file = os.path.join(cfg.data_dir, "annotations", "instances_train2017.json")
        if os.path.exists(img_dir):
            train_loader = build_coco_loader(
                img_dir, ann_file, cfg.batch_size, cfg.img_size, cfg.num_workers
            )
        else:
            print("Image directory not found, falling back to dummy data")
            encoder = decoder = criterion = None
            train_ds = DummyDataset(num_samples=128, img_size=cfg.img_size)
            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=cfg.batch_size, shuffle=True
            )

    eval_fn = None
    if args.eval_every > 0 and encoder is not None:
        from rtdetr.voc_eval import evaluate_voc_map
        eval_fn = evaluate_voc_map

    if cfg.full_finetune:
        from rtdetr.trainer import train_full_finetune
        train_full_finetune(mapping, encoder, decoder, criterion,
                            train_loader, cfg, device,
                            eval_fn=eval_fn, eval_every=args.eval_every)
    elif cfg.use_lora:
        from rtdetr.trainer import train_lora
        train_lora(mapping, encoder, decoder, criterion,
                   train_loader, cfg, device,
                   eval_fn=eval_fn, eval_every=args.eval_every)
    else:
        train_rtdetr_backbone(
            mapping, encoder, decoder, criterion,
            train_loader, cfg, device,
            eval_fn=eval_fn,
            eval_every=args.eval_every,
            is_baseline=args.baseline,
            qp_orig_state=qp_orig_state,
        )


if __name__ == "__main__":
    main()
