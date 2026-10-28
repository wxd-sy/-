import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


def _frozen_bn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    w = weight.view(1, -1, 1, 1)
    b = bias.view(1, -1, 1, 1)
    rv = running_var.view(1, -1, 1, 1)
    rm = running_mean.view(1, -1, 1, 1)
    scale = w * (rv + eps).rsqrt()
    shift = b - rm * scale
    return x * scale + shift


def _conv_bn_relu(
    x: torch.Tensor,
    conv_w: torch.Tensor,
    bn_w: torch.Tensor,
    bn_b: torch.Tensor,
    bn_mean: torch.Tensor,
    bn_var: torch.Tensor,
    bn_eps: float,
    stride: int = 1,
    padding: int = None,
    groups: int = 1,
    act: bool = True,
) -> torch.Tensor:
    if padding is None:
        k = conv_w.shape[-1]
        padding = (k - 1) // 2
    x = F.conv2d(x, conv_w, None, stride=stride, padding=padding, groups=groups)
    x = _frozen_bn(x, bn_w, bn_b, bn_mean, bn_var, bn_eps)
    if act:
        x = F.relu(x)
    return x


def _basic_block_forward(
    x: torch.Tensor,
    w: Dict[str, torch.Tensor],
    bn: Dict[str, Dict[str, torch.Tensor]],
    prefix: str,
    stride: int,
    shortcut: bool,
    ch_out: int,
) -> torch.Tensor:
    identity = x

    x = _conv_bn_relu(x, w[f"{prefix}.branch2a"], **bn[f"{prefix}.branch2a"], stride=stride)
    x = _conv_bn_relu(x, w[f"{prefix}.branch2b"], **bn[f"{prefix}.branch2b"], stride=1, act=False)

    if shortcut:
        short = identity
    else:
        short = _conv_bn_relu(
            identity, w[f"{prefix}.shortcut"],
            **bn[f"{prefix}.shortcut"],
            stride=stride, act=False,
        )

    x = F.relu(x + short)
    return x


def _bottleneck_forward(
    x: torch.Tensor,
    w: Dict[str, torch.Tensor],
    bn: Dict[str, Dict[str, torch.Tensor]],
    prefix: str,
    stride: int,
    shortcut: bool,
    expansion: int = 4,
) -> torch.Tensor:
    identity = x
    ch_in = w[f"{prefix}.branch2a"].shape[1] if f"{prefix}.branch2a" in w else bn[f"{prefix}.branch2a"]["bn_w"].shape[0]
    width = w[f"{prefix}.branch2a"].shape[0] if f"{prefix}.branch2a" in w else bn[f"{prefix}.branch2a"]["bn_w"].shape[0]

    stride1, stride2 = (1, stride)

    x = _conv_bn_relu(x, w[f"{prefix}.branch2a"], **bn[f"{prefix}.branch2a"], stride=stride1, padding=0)
    x = _conv_bn_relu(x, w[f"{prefix}.branch2b"], **bn[f"{prefix}.branch2b"], stride=stride2)
    x = _conv_bn_relu(x, w[f"{prefix}.branch2c"], **bn[f"{prefix}.branch2c"], stride=1, padding=0, act=False)

    if shortcut:
        short = identity
    else:
        short = _conv_bn_relu(
            identity, w[f"{prefix}.shortcut"],
            **bn[f"{prefix}.shortcut"],
            stride=stride, padding=0, act=False,
        )

    x = F.relu(x + short)
    return x


def presnet_forward(
    x: torch.Tensor,
    conv_weights: Dict[str, torch.Tensor],
    bn_buffers: Dict[str, Dict[str, torch.Tensor]],
    depth: int = 18,
    return_indices: Optional[List[int]] = None,
) -> List[torch.Tensor]:
    if return_indices is None:
        return_indices = [1, 2, 3]

    block_nums = [2, 2, 2, 2] if depth == 18 else [3, 4, 6, 3]
    stage_names = ["res2", "res3", "res4", "res5"]
    is_bottleneck = depth >= 50
    expansion = 4 if is_bottleneck else 1
    ch_base = [64, 128, 256, 512]

    # conv1 stem
    x = _conv_bn_relu(x, conv_weights["conv1.conv1_1"], **bn_buffers["conv1.conv1_1"], stride=2)
    x = _conv_bn_relu(x, conv_weights["conv1.conv1_2"], **bn_buffers["conv1.conv1_2"], stride=1)
    x = _conv_bn_relu(x, conv_weights["conv1.conv1_3"], **bn_buffers["conv1.conv1_3"], stride=1)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

    outs = []
    for stage_idx, (stage_name, num_blocks) in enumerate(zip(stage_names, block_nums)):
        for block_idx in range(num_blocks):
            stride_val = 2 if (block_idx == 0 and stage_idx > 0) else 1
            has_shortcut = block_idx > 0
            prefix = f"{stage_name}.block{block_idx}"

            if is_bottleneck:
                x = _bottleneck_forward(x, conv_weights, bn_buffers, prefix, stride_val, has_shortcut, expansion)
            else:
                ch_out = ch_base[stage_idx]
                x = _basic_block_forward(x, conv_weights, bn_buffers, prefix, stride_val, has_shortcut, ch_out)

        if stage_idx in return_indices:
            outs.append(x)

    return outs


def _conv_relu(
    x: torch.Tensor,
    conv_w: torch.Tensor,
    conv_bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = None,
    groups: int = 1,
    act: bool = True,
) -> torch.Tensor:
    if padding is None:
        padding = (conv_w.shape[-1] - 1) // 2
    x = F.conv2d(x, conv_w, conv_bias, stride=stride, padding=padding, groups=groups)
    if act:
        x = F.relu(x)
    return x


def _basic_block_fused_forward(
    x: torch.Tensor,
    w: Dict[str, torch.Tensor],
    bias: Dict[str, torch.Tensor],
    prefix: str,
    stride: int,
    shortcut: bool,
    ch_out: int,
) -> torch.Tensor:
    identity = x

    x = _conv_relu(x, w[f"{prefix}.branch2a"], bias.get(f"{prefix}.branch2a"), stride=stride)
    x = _conv_relu(x, w[f"{prefix}.branch2b"], bias.get(f"{prefix}.branch2b"), stride=1, act=False)

    if shortcut:
        short = identity
    else:
        short = _conv_relu(identity, w[f"{prefix}.shortcut"], bias.get(f"{prefix}.shortcut"), stride=stride, act=False)

    x = F.relu(x + short)
    return x


def _bottleneck_fused_forward(
    x: torch.Tensor,
    w: Dict[str, torch.Tensor],
    bias: Dict[str, torch.Tensor],
    prefix: str,
    stride: int,
    shortcut: bool,
    expansion: int = 4,
) -> torch.Tensor:
    identity = x

    x = _conv_relu(x, w[f"{prefix}.branch2a"], bias.get(f"{prefix}.branch2a"), stride=1, padding=0)
    x = _conv_relu(x, w[f"{prefix}.branch2b"], bias.get(f"{prefix}.branch2b"), stride=stride)
    x = _conv_relu(x, w[f"{prefix}.branch2c"], bias.get(f"{prefix}.branch2c"), stride=1, padding=0, act=False)

    if shortcut:
        short = identity
    else:
        short = _conv_relu(identity, w[f"{prefix}.shortcut"], bias.get(f"{prefix}.shortcut"), stride=stride, padding=0, act=False)

    x = F.relu(x + short)
    return x


def presnet_forward_fused(
    x: torch.Tensor,
    conv_weights: Dict[str, torch.Tensor],
    conv_bias: Dict[str, torch.Tensor],
    depth: int = 18,
    return_indices: Optional[List[int]] = None,
) -> List[torch.Tensor]:
    if return_indices is None:
        return_indices = [1, 2, 3]

    block_nums = [2, 2, 2, 2] if depth == 18 else [3, 4, 6, 3]
    stage_names = ["res2", "res3", "res4", "res5"]
    is_bottleneck = depth >= 50
    ch_base = [64, 128, 256, 512]

    x = _conv_relu(x, conv_weights["conv1.conv1_1"], conv_bias.get("conv1.conv1_1"), stride=2)
    x = _conv_relu(x, conv_weights["conv1.conv1_2"], conv_bias.get("conv1.conv1_2"), stride=1)
    x = _conv_relu(x, conv_weights["conv1.conv1_3"], conv_bias.get("conv1.conv1_3"), stride=1)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

    outs = []
    for stage_idx, (stage_name, num_blocks) in enumerate(zip(stage_names, block_nums)):
        for block_idx in range(num_blocks):
            stride_val = 2 if (block_idx == 0 and stage_idx > 0) else 1
            has_shortcut = block_idx > 0
            prefix = f"{stage_name}.block{block_idx}"

            if is_bottleneck:
                x = _bottleneck_fused_forward(x, conv_weights, conv_bias, prefix, stride_val, has_shortcut)
            else:
                ch_out = ch_base[stage_idx]
                x = _basic_block_fused_forward(x, conv_weights, conv_bias, prefix, stride_val, has_shortcut, ch_out)

        if stage_idx in return_indices:
            outs.append(x)

    return outs
