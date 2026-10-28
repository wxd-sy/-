"""query_pos_head: MLP(4, 512, 256, num_layers=2).
   layer0: Linear(4, 512) + ReLU
   layer1: Linear(512, 256)
"""

QUERY_POS_SPEC = {
    "qpos_l0": {"weight": (512, 4), "bias": (512,)},
    "qpos_l1": {"weight": (256, 512), "bias": (256,)},
}


def get_qpos_param_count():
    counts = {}
    for name, shapes in QUERY_POS_SPEC.items():
        n_w = shapes["weight"][0] * shapes["weight"][1]
        n_b = shapes["bias"][0]
        counts[name] = n_w + n_b
    return counts
