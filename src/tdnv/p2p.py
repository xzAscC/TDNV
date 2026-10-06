"""Route GPU-to-GPU copies through host memory for models split with accelerate.

On OSC Ascend 2-GPU nodes (AMD EPYC, GPUs on different sockets, topology SYS) CUDA reports
peer access as available, but direct copies are slow (~100 ms each) and corrupt data: every
model split across the two GPUs produced NaN hidden states from layer 1 on. Copying through
the CPU avoids the peer path.
"""

from __future__ import annotations

from collections.abc import Mapping

import accelerate.hooks
import torch
from accelerate.utils.operations import honor_type

_original = accelerate.hooks.send_to_device


def _staged(obj, device, non_blocking=False, skip_keys=None):
    if isinstance(obj, torch.Tensor):
        target = torch.device(device)
        if obj.is_cuda and target.type == "cuda" and obj.device != target:
            return obj.to("cpu").to(target)
        return obj.to(target)
    if isinstance(obj, (tuple, list)):
        return honor_type(obj, (_staged(t, device, non_blocking, skip_keys) for t in obj))
    if isinstance(obj, Mapping):
        skip = [skip_keys] if isinstance(skip_keys, str) else (skip_keys or [])
        return type(obj)({k: v if k in skip else _staged(v, device, non_blocking, skip_keys)
                          for k, v in obj.items()})
    return _original(obj, device, non_blocking=non_blocking, skip_keys=skip_keys)


def stage_cross_gpu_copies() -> None:
    """Make accelerate's dispatch hooks copy between GPUs via the CPU."""
    accelerate.hooks.send_to_device = _staged
