# Copyright (c) Facebook, Inc. and its affiliates.
from fvcore.common.registry import Registry
OPTIMIZER_REGISTRY = Registry('OPTIMIZER')  # noqa F401 isort:skip
OPTIMIZER_REGISTRY.__doc__ = '''
Registry for OPTIMIZER, i.e. the whole optim.

The registered object will be called with `obj(cfg)`
'''

LR_SCHEDULER_REGISTRY = Registry('Learning Rate Scheduler')  # noqa F401 isort:skip
LR_SCHEDULER_REGISTRY.__doc__ = '''
Registry for Learning Rate Scheduler, i.e. the whole lrs.

The registered object will be called with `obj(cfg)`
'''


def build_optimizer(cfg, param_dict, step_per_epoch):
    '''
    Build the whole model architecture, defined by ``cfg.MODEL.OPTIMIZER``.
    Note that it does not load any weights from ``cfg``.
    '''
    optim_cfg = cfg.SOLVER.OPTIMIZER.NAME
    optim = OPTIMIZER_REGISTRY.get(optim_cfg)(cfg, param_dict)
    lrs = build_lrs(cfg, optim, step_per_epoch+10)
    return optim, lrs


def build_lrs(cfg, optim, step_per_epoch):
    lr_scheduler_name = cfg.SOLVER.LR_SCHEDULER.NAME
    lr_scheduler = LR_SCHEDULER_REGISTRY.get(lr_scheduler_name)(
        cfg, optim, step_per_epoch) if lr_scheduler_name else None
    return lr_scheduler
