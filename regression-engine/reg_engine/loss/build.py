# Copyright (c) Facebook, Inc. and its affiliates.
from fvcore.common.registry import Registry
import torch.nn as nn

LOSS_REGISTRY = Registry("LOSS")  # noqa F401 isort:skip
LOSS_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_loss(cfg, loss_list=None):
    """
    Build the whole model architecture, defined by ``cfg.SOLVER.LOSS``.
    Note that it does not load any weights from ``cfg``.
    """
    if not loss_list:
        loss_list = cfg.SOLVER.LOSS.NAME
    else:
        loss_list = [loss_list]
    loss_dict = {}
    for loss_nm in loss_list:
        loss_tmp = LOSS_REGISTRY.get(loss_nm)
        if issubclass(loss_tmp, nn.Module):
            loss_dict[loss_nm] = loss_tmp(cfg).to(cfg.DEVICE)

    return loss_dict
