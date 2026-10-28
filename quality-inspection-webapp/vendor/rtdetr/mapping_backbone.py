import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from mapping_network import OnTheFlyProjection
from rtdetr.spec import build_backbone_specs, get_stage_groups, build_querypos_spec
from rtdetr.functional_presnet import presnet_forward, presnet_forward_fused
from bn_fuse import fuse_backbone


def _sanitize(name: str) -> str:
    return name.replace(".", "_")


class MappingBackbone(nn.Module):
    def __init__(
        self,
        pretrained_state: Dict[str, torch.Tensor],
        depth: int = 18,
        latent_dim: int = 1024,
        alpha: float = 0.001,
        layerwise: bool = True,
        block_size: int = 8192,
        cache_projections: bool = False,
        use_tanh: bool = True,
        output_gain: float = 1.0,
        bn_fuse: bool = True,
        use_query_mapping: bool = False,
        query_latent_dim: int = 1024,
        query_output_gain: float = 0.01,
        trainable_backbone: bool = False,
    ):
        super().__init__()
        self.depth = depth
        self.alpha = alpha
        self.output_gain = output_gain
        self.layerwise = layerwise
        self.use_tanh = use_tanh
        self.use_fused_bn = bn_fuse
        self.use_query_mapping = use_query_mapping
        self.query_output_gain = query_output_gain
        self.block_size = block_size
        self.trainable_backbone = trainable_backbone

        self.conv_specs = build_backbone_specs(depth)
        stage_groups = get_stage_groups(depth)

        self.pretrained_conv: Dict[str, torch.Tensor] = {}
        self.bn_buffers: Dict[str, Dict[str, torch.Tensor]] = {}
        self.pretrained_bias: Dict[str, torch.Tensor] = {}
        self.conv_shapes: Dict[str, Tuple[int, ...]] = {}
        self.conv_stage: Dict[str, str] = {}
        self.conv_fan_in: Dict[str, int] = {}
        self._name_list: List[str] = []

        for name, (w_shape, _) in self.conv_specs.items():
            self.conv_shapes[name] = w_shape
            fan_in = int(np.prod(w_shape[1:])) if len(w_shape) > 1 else w_shape[1]
            self.conv_fan_in[name] = fan_in
            self._name_list.append(name)

            for group in stage_groups:
                if name.startswith(group) or (group == "conv1" and name.startswith("conv1")):
                    self.conv_stage[name] = group
                    break

        self._extract_pretrained(pretrained_state, depth)

        if bn_fuse:
            fused_w, fused_b = fuse_backbone(self.pretrained_conv, self.bn_buffers)
            self.pretrained_conv = fused_w
            self.pretrained_bias = fused_b
            self.bn_buffers = {}
        else:
            for name in self.conv_specs:
                self.pretrained_bias[name] = torch.zeros(self.conv_shapes[name][0])

        if trainable_backbone:
            self._trainable_conv = nn.ParameterDict()
            self._trainable_bias = nn.ParameterDict()
            self._name_to_safe = {}
            for name in self.conv_specs:
                safe_name = _sanitize(name)
                self._name_to_safe[name] = safe_name
                self._trainable_conv[safe_name] = nn.Parameter(self.pretrained_conv[name].clone())
                self._trainable_bias[safe_name] = nn.Parameter(self.pretrained_bias[name].clone())

        self.projections = nn.ModuleDict()
        seed_counter = 0

        if layerwise:
            self.latents = nn.ParameterList(
                [nn.Parameter(torch.randn(1, latent_dim) * 1.0) for _ in stage_groups]
            )
            for group_idx, group in enumerate(stage_groups):
                group_layers = [n for n in self.conv_specs if self.conv_stage.get(n) == group]
                for name in group_layers:
                    self.projections[_sanitize(name)] = OnTheFlyProjection(
                        in_dim=latent_dim,
                        out_dim=int(np.prod(self.conv_shapes[name])),
                        seed=seed_counter,
                        block_size=block_size,
                        use_tanh=use_tanh,
                    )
                    seed_counter += 1
        else:
            self.latent = nn.Parameter(torch.randn(1, latent_dim) * 1.0)
            for name in self._name_list:
                self.projections[_sanitize(name)] = OnTheFlyProjection(
                    in_dim=latent_dim,
                    out_dim=int(np.prod(self.conv_shapes[name])),
                    seed=seed_counter,
                    block_size=block_size,
                    use_tanh=use_tanh,
                )
                seed_counter += 1

        if use_query_mapping:
            self.z_query = nn.Parameter(torch.randn(1, query_latent_dim) * 0.01)
            self.qp_projections = nn.ModuleDict()
            self._qp_specs = build_querypos_spec()
            for i, (name, (w_shape, b_shape)) in enumerate(self._qp_specs.items()):
                n_params = int(np.prod(w_shape)) + int(np.prod(b_shape))
                self.qp_projections[name] = OnTheFlyProjection(
                    in_dim=query_latent_dim,
                    out_dim=n_params,
                    seed=200 + i,
                    block_size=block_size,
                    use_tanh=use_tanh,
                )

            self.register_buffer("_ref_feats_c3", torch.tensor(0.0), persistent=True)
            self.register_buffer("_ref_feats_c4", torch.tensor(0.0), persistent=True)
            self.register_buffer("_ref_feats_c5", torch.tensor(0.0), persistent=True)
            self._has_ref = False

    def _extract_pretrained(self, state: Dict[str, torch.Tensor], depth: int):
        for name, (w_shape, _) in self.conv_specs.items():
            out_ch = w_shape[0]
            self.bn_buffers[name] = {
                "bn_w": torch.ones(out_ch),
                "bn_b": torch.zeros(out_ch),
                "bn_mean": torch.zeros(out_ch),
                "bn_var": torch.ones(out_ch),
                "bn_eps": torch.tensor(1e-5),
            }

        if not state:
            return

        for name in self.conv_specs:
            conv_key, bn_prefix = _find_ckpt_keys(name, depth, state)
            if conv_key and conv_key in state:
                self.pretrained_conv[name] = state[conv_key].detach().clone().float()
            if bn_prefix:
                w_key = f"{bn_prefix}.weight"
                b_key = f"{bn_prefix}.bias"
                m_key = f"{bn_prefix}.running_mean"
                v_key = f"{bn_prefix}.running_var"
                if w_key in state:
                    self.bn_buffers[name] = {
                        "bn_w": state[w_key].detach().clone().float(),
                        "bn_b": state[b_key].detach().clone().float(),
                        "bn_mean": state[m_key].detach().clone().float(),
                        "bn_var": state[v_key].detach().clone().float(),
                        "bn_eps": torch.tensor(1e-5),
                    }

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_target_params(self) -> int:
        n = sum(int(np.prod(w)) for w, _ in self.conv_specs.values())
        if self.use_query_mapping:
            for w_shape, b_shape in self._qp_specs.values():
                n += int(np.prod(w_shape)) + int(np.prod(b_shape))
        return n

    def _get_latent(self, stage: str) -> torch.Tensor:
        if not self.layerwise:
            return self.latent
        stage_groups = get_stage_groups(self.depth)
        idx = stage_groups.index(stage)
        return self.latents[idx]

    def _generate_conv_weights(
        self, return_smoothness: bool = False
    ):
        adapted_w = {}
        adapted_b = {}
        smooth_vals = []

        for name in self._name_list:
            if self.trainable_backbone:
                safe_name = self._name_to_safe.get(name, _sanitize(name))
                adapted_w[name] = self._trainable_conv[safe_name]
                adapted_b[name] = self._trainable_bias[safe_name]
                continue

            proj = self.projections[_sanitize(name)]
            stage = self.conv_stage[name]
            z = self._get_latent(stage)
            out = proj(z, self.alpha, return_smoothness)

            if return_smoothness:
                theta, s = out
                smooth_vals.append(s)
            else:
                theta = out

            w_shape = self.conv_shapes[name]
            fan_in = self.conv_fan_in[name]
            delta_w = theta.view(w_shape) * self.output_gain / (fan_in ** 0.5)

            pretrained_w = self.pretrained_conv.get(name)
            if pretrained_w is not None:
                adapted_w[name] = pretrained_w.to(delta_w.device) + delta_w
            else:
                adapted_w[name] = delta_w

            pretrained_b = self.pretrained_bias.get(name)
            if pretrained_b is not None:
                adapted_b[name] = pretrained_b.to(delta_w.device)
            else:
                adapted_b[name] = torch.zeros(w_shape[0], device=delta_w.device)

        if return_smoothness and smooth_vals:
            return adapted_w, adapted_b, torch.stack(smooth_vals).mean()
        return adapted_w, adapted_b, None

    def generate_querypos_delta(self, return_smoothness: bool = False):
        delta = {}
        smooth_vals = []

        for name, proj in self.qp_projections.items():
            out = proj(self.z_query, self.alpha, return_smoothness)
            if return_smoothness:
                theta, s = out
                smooth_vals.append(s)
            else:
                theta = out

            w_shape, b_shape = self._qp_specs[name]
            n_w = int(np.prod(w_shape))
            w_delta = theta[:, :n_w].view(w_shape) * self.query_output_gain
            b_delta = theta[:, n_w:].view(b_shape) * self.query_output_gain * 0.1
            delta[name] = {"weight": w_delta, "bias": b_delta}

        if return_smoothness and smooth_vals:
            return delta, torch.stack(smooth_vals).mean()
        return delta

    def set_ref_features(self, feats: List[torch.Tensor]):
        if not self.use_query_mapping:
            return
        if isinstance(self._ref_feats_c3, torch.Tensor) and self._ref_feats_c3.numel() == 1:
            self._ref_feats_c3 = feats[0].detach().clone()
            self._ref_feats_c4 = feats[1].detach().clone()
            self._ref_feats_c5 = feats[2].detach().clone()
            self._has_ref = True

    def compute_align_loss(self, mapped_feats: List[torch.Tensor]) -> torch.Tensor:
        if not self._has_ref or not self.use_query_mapping:
            return torch.tensor(0.0, device=mapped_feats[0].device)

        ref = [self._ref_feats_c3, self._ref_feats_c4, self._ref_feats_c5]
        loss = 0.0
        for mf, rf in zip(mapped_feats, ref):
            rf = rf.to(mf.device)
            loss += (1.0 - F.cosine_similarity(
                mf.flatten(1), rf.flatten(1), dim=1
            )).mean()
        return loss / len(mapped_feats)

    def forward(
        self, x: torch.Tensor, return_smoothness: bool = False,
        return_qdelta: bool = False,
    ):
        adapted_w, adapted_b, s_val = self._generate_conv_weights(
            return_smoothness=return_smoothness
        )

        qp_delta = None
        if self.use_query_mapping and return_qdelta:
            qp_delta = self.generate_querypos_delta(return_smoothness=False)

        device = x.device
        conv_w = {k: v.to(device) for k, v in adapted_w.items()}
        conv_b = {k: v.to(device) for k, v in adapted_b.items()}

        if self.use_fused_bn:
            feats = presnet_forward_fused(x, conv_w, conv_b, self.depth)
        else:
            bn = {}
            for k, v in self.bn_buffers.items():
                bn[k] = {kk: vv if isinstance(vv, float) else vv.to(device) for kk, vv in v.items()}
            feats = presnet_forward(x, conv_w, bn, self.depth)

        if return_qdelta:
            return feats, s_val, qp_delta
        if return_smoothness:
            return feats, s_val
        return feats


def _find_ckpt_keys(
    name: str, depth: int, state: Dict[str, torch.Tensor]
) -> Tuple[Optional[str], Optional[str]]:
    parts = name.split(".")
    block_nums = [2, 2, 2, 2] if depth == 18 else [3, 4, 6, 3]
    stage_names = ["res2", "res3", "res4", "res5"]
    sub = None

    if parts[0] == "conv1":
        tail = f"conv1.{parts[1]}.conv"
    else:
        for si, stage_name in enumerate(stage_names):
            if parts[0] == stage_name:
                block_idx = int(parts[1].replace("block", ""))
                sub = parts[2]
                if sub == "shortcut":
                    tail = f"res_layers.{si}.blocks.{block_idx}.short.conv"
                    tail_alt = f"res_layers.{si}.blocks.{block_idx}.short.conv.conv"
                else:
                    tail = f"res_layers.{si}.blocks.{block_idx}.{sub}.conv"
                break
        else:
            return None, None

    candidates_conv = [tail, f"backbone.{tail}"]
    if sub == "shortcut":
        candidates_conv += [tail_alt, f"backbone.{tail_alt}"]
    conv_key = None
    for c in candidates_conv:
        if f"{c}.weight" in state:
            conv_key = f"{c}.weight"
            break
    if conv_key is None:
        for c in candidates_conv:
            if c in state:
                conv_key = c
                break

    bn_prefix = tail.replace(".conv", ".norm")
    bn_candidates = [bn_prefix, f"backbone.{bn_prefix}"]
    if sub == "shortcut":
        bn_alt = tail_alt.replace(".conv", ".norm")
        bn_candidates += [bn_alt, f"backbone.{bn_alt}"]
    bn_key = None
    for c in bn_candidates:
        if f"{c}.weight" in state:
            bn_key = c
            break

    return conv_key, bn_key
