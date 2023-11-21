import hashlib
import logging
import subprocess

import torch
from torch import backends, cuda

_ENABLE_APPLE_GPU = False


def get_free_gpu():
    max_free = 0
    max_idx = 0

    rows = (
        subprocess.check_output(
            ["nvidia-smi", "--format=csv", "--query-gpu=memory.free"]
        )
        .decode("utf-8")
        .split("\n")
    )
    for i, row in enumerate(rows[1:-1]):
        mb = float(row.rstrip(" [MiB]"))

        if mb > max_free:
            max_idx = i
            max_free = mb

    return max_idx


def get_device():
    if _ENABLE_APPLE_GPU and backends.mps.is_available():
        device = "mps"
    elif cuda.is_available():
        device = f"cuda:{get_free_gpu()}"
    else:
        device = "cpu"
    logging.info(f"Using device {device}")
    return torch.device(device)


def kwargs_to_id(kwargs) -> str:
    s = ""
    for key, val in sorted(kwargs.items()):
        s += f"{key} {val}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]
