from pathlib import Path

import torch


DEFAULT_MODEL_KWARGS = {
    "in_channels": 1,
    "base_filters": 64,
    "ratio": 1,
    "filter_list": [64, 160, 160, 400, 400, 1024, 1024],
    "m_blocks_list": [2, 2, 2, 3, 3, 4, 4],
    "kernel_size": 16,
    "stride": 2,
    "groups_width": 16,
    "n_classes": 1,
    "use_bn": False,
    "use_do": False,
}


def project_root():
    return Path(__file__).resolve().parent.parent


def resolve_device(device_arg=None):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
