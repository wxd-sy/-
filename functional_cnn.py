import torch
import torch.nn.functional as F
from typing import Dict


def functional_cnn_forward(
    x: torch.Tensor,
    weights: Dict[str, Dict[str, torch.Tensor]],
    cnn_spec_name: str = "default",
) -> torch.Tensor:
    if cnn_spec_name == "default":
        w = weights["conv1"]
        x = F.conv2d(x, w["weight"], w["bias"], padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        w = weights["conv2"]
        x = F.conv2d(x, w["weight"], w["bias"], padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        w = weights["fc1"]
        x = F.linear(x, w["weight"], w["bias"])
        x = F.relu(x)

        w = weights["fc2"]
        x = F.linear(x, w["weight"], w["bias"])

    elif cnn_spec_name == "large":
        w = weights["conv1"]
        x = F.conv2d(x, w["weight"], w["bias"], padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        w = weights["conv2"]
        x = F.conv2d(x, w["weight"], w["bias"], padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        w = weights["conv3"]
        x = F.conv2d(x, w["weight"], w["bias"], padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        w = weights["fc1"]
        x = F.linear(x, w["weight"], w["bias"])
        x = F.relu(x)

        w = weights["fc2"]
        x = F.linear(x, w["weight"], w["bias"])

    return x
