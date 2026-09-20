from typing import Dict, List, Tuple
import numpy as np
from config import CNN_SPECS


def get_layer_specs(cnn_spec_name: str) -> Dict[str, Dict[str, Tuple[int, ...]]]:
    raw = CNN_SPECS[cnn_spec_name]
    specs = {}
    for name, w_shape, b_shape in raw:
        specs[name] = {"weight": tuple(w_shape), "bias": tuple(b_shape)}
    return specs


def get_layer_param_counts(cnn_spec_name: str) -> Dict[str, int]:
    specs = get_layer_specs(cnn_spec_name)
    counts = {}
    for name, shapes in specs.items():
        n_w = int(np.prod(shapes["weight"]))
        n_b = int(np.prod(shapes["bias"]))
        counts[name] = n_w + n_b
    return counts


def get_total_param_count(cnn_spec_name: str) -> int:
    return sum(get_layer_param_counts(cnn_spec_name).values())
