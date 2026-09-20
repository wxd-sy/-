from dataclasses import dataclass


@dataclass
class RTDETRConfig:
    dataset: str = "coco128"
    data_dir: str = "./data/coco128"
    num_classes: int = 80
    batch_size: int = 4
    num_workers: int = 2
    img_size: int = 640

    backbone_depth: int = 18
    backbone_variant: str = "d"

    latent_dim: int = 1024
    alpha: float = 0.001
    layerwise: bool = True
    block_size: int = 8192
    cache_projections: bool = False
    use_tanh: bool = True
    output_gain: float = 1.0

    epochs: int = 20
    lr: float = 1e-2
    weight_decay: float = 1e-5

    use_query_mapping: bool = True
    query_latent_dim: int = 1024
    query_output_gain: float = 0.01
    bn_fuse: bool = True
    lambda_align: float = 0.01

    full_finetune: bool = False
    use_lora: bool = False
    lora_rank: int = 16

    lambda_stability: float = 0.01
    stability_sigma: float = 0.01
    lambda_smoothness: float = 0.1

    device: str = "cuda"
    seed: int = 42

    log_interval: int = 20
    save_dir: str = "./checkpoints_rtdetr"
    exp_name: str = "mapping_rtdetr_r18"
    pretrained_ckpt: str = ""


RESNET_VD_CFG = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
}

STAGE_CHANNELS = {
    18: {"base": [64, 128, 256, 512], "expansion": 1},
    34: {"base": [64, 128, 256, 512], "expansion": 1},
    50: {"base": [64, 128, 256, 512], "expansion": 4},
    101: {"base": [64, 128, 256, 512], "expansion": 4},
}
