# Copyright (c) Facebook, Inc. and its affiliates.
from fvcore.common.registry import Registry
import torch.nn as nn

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

BACKBONE_REGISTRY = Registry("BACKBONE")  # noqa F401 isort:skip
BACKBONE_REGISTRY.__doc__ = """
Registry for BLOCK, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and exp
"""

BLOCK_REGISTRY = Registry("BLOCK")  # noqa F401 isort:skip
BLOCK_REGISTRY.__doc__ = """
Registry for BLOCK, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def convert_relu_to_prelu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.PReLU())
        else:
            convert_relu_to_prelu(child)
    return model


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.DECODER.NAME
    block = BLOCK_REGISTRY.get(cfg.MODEL.BLOCK) if cfg.MODEL.BLOCK else None
    try:
        backbone = BACKBONE_REGISTRY.get(cfg.MODEL.ENCODER.NAME)(cfg, block)
    except:
        backbone = None
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg, backbone)
    # replace ReLU with PReLU
    model = convert_relu_to_prelu(model)

    return model
