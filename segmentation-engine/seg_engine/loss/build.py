# Copyright (c) Facebook, Inc. and its affiliates.
from fvcore.common.registry import Registry

LOSS_REGISTRY = Registry('LOSS')  # noqa F401 isort:skip
LOSS_REGISTRY.__doc__ = '''
Registry for Loss, i.e. the whole model.

The registered object will be called with `obj(cfg)`
'''


def build_loss(cfg):
    '''
    Build the whole model architecture, defined by ``cfg.SOLVER.LOSS``.
    Note that it does not load any weights from ``cfg``.
    '''
    loss_arch = cfg.SOLVER.LOSS
    loss = LOSS_REGISTRY.get(loss_arch)(cfg)
    return loss
