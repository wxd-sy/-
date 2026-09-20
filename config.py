from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Config:
    dataset: str = "mnist"
    batch_size: int = 128
    num_workers: int = 4

    latent_dim: int = 1024
    cnn_spec: str = "large"

    epochs: int = 100
    lr: float = 1e-2
    weight_decay: float = 1e-5

    alpha: float = 0.001
    use_tanh: bool = True
    block_size: int = 8192
    cache_projections: bool = True
    latent_init_std: float = 1.0
    output_gain: float = 1.0

    lambda_stability: float = 0.01
    stability_sigma: float = 0.01
    lambda_smoothness: float = 0.1
    lambda_alignment: float = 0.001

    device: str = "cuda"
    seed: int = 42

    log_interval: int = 100
    save_dir: str = "./checkpoints"
    exp_name: str = "mapping_mnist"


CNN_SPECS: Dict[str, List[Tuple[str, List[int], List[int]]]] = {
    "default": [
        ("conv1", [16, 1, 3, 3], [16]),
        ("conv2", [32, 16, 3, 3], [32]),
        ("fc1", [128, 32 * 7 * 7], [128]),
        ("fc2", [10, 128], [10]),
    ],
    "large": [
        ("conv1", [32, 1, 3, 3], [32]),
        ("conv2", [64, 32, 3, 3], [64]),
        ("conv3", [128, 64, 3, 3], [128]),
        ("fc1", [256, 128 * 3 * 3], [256]),
        ("fc2", [10, 256], [10]),
    ],
}
