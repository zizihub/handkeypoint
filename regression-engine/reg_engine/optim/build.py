# Copyright (c) Facebook, Inc. and its affiliates.
from fvcore.common.registry import Registry

OPTIMIZER_REGISTRY = Registry("OPTIMIZER")
OPTIMIZER_REGISTRY.__doc__ = """
Registry for optimizer, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

LR_SCHEDULER_REGISTRY = Registry("LR_SCHEDULER")
LR_SCHEDULER_REGISTRY.__doc__ = """
Registry for lr scheduler, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_optimizer(cfg, net_params):
    """
    Build the whole model architecture, defined by ``cfg.SOLVER.LOSS``.
    Note that it does not load any weights from ``cfg``.
    """
    optimizer_name = cfg.SOLVER.OPTIMIZER.NAME
    optim = OPTIMIZER_REGISTRY.get(optimizer_name)(cfg, net_params)
    return optim


def build_scheduler(cfg, optim, step_per_epoch):
    """
    Build the whole model architecture, defined by ``cfg.SOLVER.LOSS``.
    Note that it does not load any weights from ``cfg``.
    """
    lr_scheduler_name = cfg.SOLVER.LR_SCHEDULER.NAME
    lr_scheduler = LR_SCHEDULER_REGISTRY.get(lr_scheduler_name)(
        cfg, optim, step_per_epoch) if lr_scheduler_name else None

    return lr_scheduler
