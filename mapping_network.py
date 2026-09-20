import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from target_cnn_spec import get_layer_specs, get_layer_param_counts


def _hash_seed(base_seed: int, offset: int) -> int:
    return (base_seed * 1000003 + offset) & 0x7FFFFFFF


class OnTheFlyProjection(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        seed: int,
        block_size: int = 8192,
        use_tanh: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.register_buffer("_seed", torch.tensor(seed, dtype=torch.long))
        self.scale = in_dim**-0.5
        self.block_size = block_size if out_dim > block_size else 0
        self.use_tanh = use_tanh

    def _generate_block(self, rows: int, block_seed: int, device: torch.device,
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
        g = torch.Generator(device=device)
        g.manual_seed(block_seed)
        W_block = torch.randn(rows, self.in_dim, generator=g, device=device, dtype=dtype)
        return W_block / (W_block.norm(p=2, dim=1, keepdim=True) + 1e-8)

    def forward(
        self,
        z: torch.Tensor,
        alpha: float,
        return_smoothness: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        z_norm_mean = z.square().mean(dim=1, keepdim=True)
        modulation = alpha * z_norm_mean

        if self.block_size == 0:
            g = torch.Generator(device=z.device)
            g.manual_seed(self._seed.item())
            W = torch.randn(self.out_dim, self.in_dim, generator=g,
                            device=z.device, dtype=z.dtype)
            W = W / (W.norm(p=2, dim=1, keepdim=True) + 1e-8)
            raw_woz = F.linear(z, W)
            raw = raw_woz + modulation

            if self.use_tanh:
                theta = torch.tanh(raw)
            else:
                theta = raw

            smooth_val = None
            if return_smoothness:
                coef = alpha / self.in_dim
                z_sq_sum = z.square().sum()
                s_val = 1.0 + 4.0 * coef * raw_woz + 4.0 * coef * coef * z_sq_sum
                smooth_val = s_val.mean()

            return (theta, smooth_val) if return_smoothness else theta

        theta = torch.empty(1, self.out_dim, device=z.device, dtype=z.dtype)
        smooth_sum = 0.0

        for start in range(0, self.out_dim, self.block_size):
            end = min(start + self.block_size, self.out_dim)
            rows = end - start
            block_seed = _hash_seed(self._seed.item(), start)
            W_block = self._generate_block(rows, block_seed, z.device, z.dtype)

            raw_woz_block = F.linear(z, W_block)
            raw_block = raw_woz_block + modulation

            if self.use_tanh:
                theta_block = torch.tanh(raw_block)
            else:
                theta_block = raw_block

            theta[:, start:end] = theta_block

            if return_smoothness:
                coef = alpha / self.in_dim
                z_sq_sum = z.square().sum()
                s_block = 1.0 + 4.0 * coef * raw_woz_block + 4.0 * coef * coef * z_sq_sum
                smooth_sum = smooth_sum + s_block.sum()

        if return_smoothness:
            smooth_val = smooth_sum / self.out_dim
            return theta, smooth_val
        return theta


class MappingNetwork(nn.Module):
    def __init__(
        self,
        latent_dim: int = 1024,
        cnn_spec_name: str = "large",
        alpha: float = 0.01,
        block_size: int = 8192,
        use_tanh: bool = True,
        cache_projections: bool = False,
        latent_init_std: float = 1.0,
        output_gain: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.output_gain = output_gain

        self.z = nn.Parameter(torch.randn(1, latent_dim) * latent_init_std)
        self.layer_specs = get_layer_specs(cnn_spec_name)
        self.layer_param_counts = get_layer_param_counts(cnn_spec_name)

        self.projections = nn.ModuleDict()
        for i, (name, n_params) in enumerate(self.layer_param_counts.items()):
            self.projections[name] = OnTheFlyProjection(
                in_dim=latent_dim,
                out_dim=n_params,
                seed=i,
                block_size=block_size,
                use_tanh=use_tanh,
            )

        self._cached = cache_projections
        self._cache: Optional[Dict[str, torch.Tensor]] = None

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_target_params(self) -> int:
        return sum(self.layer_param_counts.values())

    def forward(
        self,
        return_smoothness: bool = False,
    ) -> Union[Dict[str, Dict[str, torch.Tensor]],
               Tuple[Dict[str, Dict[str, torch.Tensor]], torch.Tensor]]:
        weights = {}
        smooth_vals = []

        for name, proj in self.projections.items():
            out = proj(self.z, self.alpha, return_smoothness)
            if return_smoothness:
                theta, s = out
                smooth_vals.append(s)
            else:
                theta = out

            spec = self.layer_specs[name]
            w_shape = spec["weight"]
            b_shape = spec["bias"]
            n_w = int(np.prod(w_shape))
            fan_in = int(np.prod(w_shape[1:])) if len(w_shape) > 1 else w_shape[1]
            weight_scale = self.output_gain / (fan_in ** 0.5)
            bias_scale = weight_scale * 0.01
            w_flat = theta[:, :n_w] * weight_scale
            b_flat = theta[:, n_w:] * bias_scale
            weights[name] = {
                "weight": w_flat.view(w_shape),
                "bias": b_flat.view(b_shape),
            }

        if return_smoothness:
            return weights, torch.stack(smooth_vals).mean()
        return weights
