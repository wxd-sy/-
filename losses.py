import torch
import torch.nn.functional as F
from mapping_network import MappingNetwork
from typing import Dict, Optional


def task_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def stability_loss(
    mapping: MappingNetwork,
    x: torch.Tensor,
    functional_forward,
    cnn_spec_name: str,
    sigma: float = 0.01,
) -> torch.Tensor:
    z_orig = mapping.z.data.clone()

    noise = torch.randn_like(z_orig) * sigma
    mapping.z.data = z_orig + noise
    with torch.no_grad():
        weights_noisy = mapping()
    logits_noisy = functional_forward(x, weights_noisy, cnn_spec_name)

    mapping.z.data = z_orig
    weights_clean = mapping()
    logits_clean = functional_forward(x, weights_clean, cnn_spec_name)

    return F.kl_div(
        F.log_softmax(logits_noisy, dim=1),
        F.softmax(logits_clean.detach(), dim=1),
        reduction='batchmean',
    )


def alignment_loss(
    mapping: MappingNetwork,
) -> torch.Tensor:
    z = mapping.z
    z_norm = z / (z.norm(p=2) + 1e-8)
    d = z.shape[1]
    total_loss = 0.0

    for name, proj in mapping.projections.items():
        g = torch.Generator(device=z.device).manual_seed(proj._seed.item())
        Wo = torch.randn(proj.out_dim, d, generator=g, device=z.device, dtype=z.dtype)
        Wo = Wo / (Wo.norm(p=2, dim=1, keepdim=True) + 1e-8)

        M = Wo + mapping.alpha * z

        w_bar = M.mean(dim=0)
        w_bar_norm = w_bar / (w_bar.norm(p=2) + 1e-8)

        cos_sim = F.cosine_similarity(z_norm, w_bar_norm.unsqueeze(0), dim=1)
        total_loss = total_loss + (1.0 - cos_sim.mean())

    return total_loss / len(mapping.projections)


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mapping: MappingNetwork,
    x: torch.Tensor,
    functional_forward,
    cnn_spec_name: str,
    lambda_stability: float = 0.1,
    stability_sigma: float = 0.01,
    lambda_smoothness: float = 0.01,
    lambda_alignment: float = 0.001,
    smoothness_val: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    loss_task = task_loss(logits, targets)
    loss_stab = stability_loss(mapping, x, functional_forward, cnn_spec_name, stability_sigma)
    loss_align = alignment_loss(mapping)

    total = loss_task + lambda_stability * loss_stab + lambda_alignment * loss_align
    result = {
        "total": total,
        "task": loss_task,
        "stability": loss_stab,
        "alignment": loss_align,
        "smoothness": torch.tensor(0.0, device=logits.device),
    }

    if smoothness_val is not None:
        loss_smooth = smoothness_val
        total = total + lambda_smoothness * loss_smooth
        result["total"] = total
        result["smoothness"] = loss_smooth

    return result
