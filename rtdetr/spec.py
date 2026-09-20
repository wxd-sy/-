import numpy as np
from typing import Dict, List, Tuple

from config_rtdetr import RESNET_VD_CFG, STAGE_CHANNELS


def build_backbone_specs(depth: int) -> Dict[str, Tuple[int, ...]]:
    """Returns dict of conv layer name -> (weight_shape, bias_shape or None)."""
    block_nums = RESNET_VD_CFG[depth]
    ch_base = STAGE_CHANNELS[depth]["base"]
    expansion = STAGE_CHANNELS[depth]["expansion"]
    is_bottleneck = depth >= 50

    specs = {}

    # conv1 stem (variant d: 3 convs, always)
    specs["conv1.conv1_1"] = ((32, 3, 3, 3), None)
    specs["conv1.conv1_2"] = ((32, 32, 3, 3), None)
    specs["conv1.conv1_3"] = ((64, 32, 3, 3), None)

    ch_in = 64
    stage_names = ["res2", "res3", "res4", "res5"]

    for stage_idx, (stage_name, num_blocks) in enumerate(zip(stage_names, block_nums)):
        ch_out = ch_base[stage_idx]
        for block_idx in range(num_blocks):
            stride = 2 if (block_idx == 0 and stage_idx > 0) else 1
            shortcut = block_idx > 0
            prefix = f"{stage_name}.block{block_idx}"

            if is_bottleneck:
                width = ch_out
                # branch2a: 1x1, ch_in -> width
                specs[f"{prefix}.branch2a"] = ((width, ch_in, 1, 1), None)
                # branch2b: 3x3, width -> width
                specs[f"{prefix}.branch2b"] = ((width, width, 3, 3), None)
                # branch2c: 1x1, width -> ch_out*expansion
                specs[f"{prefix}.branch2c"] = ((ch_out * expansion, width, 1, 1), None)
                if not shortcut:
                    specs[f"{prefix}.shortcut"] = ((ch_out * expansion, ch_in, 1, 1), None)
                ch_in = ch_out * expansion
            else:
                # BasicBlock
                specs[f"{prefix}.branch2a"] = ((ch_out, ch_in, 3, 3), None)
                specs[f"{prefix}.branch2b"] = ((ch_out, ch_out, 3, 3), None)
                if not shortcut:
                    specs[f"{prefix}.shortcut"] = ((ch_out, ch_in, 1, 1), None)
                ch_in = ch_out

    return specs


def get_stage_groups(depth: int) -> List[str]:
    """Groups conv layers by stage for layerwise mapping."""
    block_nums = RESNET_VD_CFG[depth]
    groups = ["conv1"]
    stage_names = ["res2", "res3", "res4", "res5"]
    for stage_name in stage_names:
        groups.append(stage_name)
    return groups


def get_layer_param_count(depth: int) -> Dict[str, int]:
    specs = build_backbone_specs(depth)
    counts: Dict[str, int] = {}
    for name, (w_shape, _) in specs.items():
        counts[name] = int(np.prod(w_shape))
    return counts


def get_total_target_params(depth: int) -> int:
    return sum(get_layer_param_count(depth).values())


def build_querypos_spec():
    from rtdetr.query_pos_spec import QUERY_POS_SPEC
    return {k: (v["weight"], v["bias"]) for k, v in QUERY_POS_SPEC.items()}
