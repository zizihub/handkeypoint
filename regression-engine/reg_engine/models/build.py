# Copyright (c) Facebook, Inc. and its affiliates.
from fvcore.common.registry import Registry
import torch
from .models import Model
from .efficientnet import EfficientNet
from .timm_model import get_timm_model
import pdb


META_ARCH_REGISTRY = Registry("META_ARCHITECTURE")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

HEAD_REGISTRY = Registry("HEAD")  # noqa F401 isort:skip
HEAD_REGISTRY.__doc__ = """
Registry for head.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""

METRICS_REGISTRY = Registry("METRICS")  # noqa F401 isort:skip
METRICS_REGISTRY.__doc__ = """
Registry for METRICS.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    backbone_name = cfg.MODEL.NAME
    if 'efficient' in backbone_name:
        backbone = EfficientNet.from_pretrained(model_name=backbone_name)
    elif 'timm' in backbone_name:
        backbone = get_timm_model(cfg)
    else:
        backbone = META_ARCH_REGISTRY.get(backbone_name)(pretrained=cfg.MODEL.PRETRAINED)
    # auto get backbone output shape
    fake_x = torch.zeros((1, 3, cfg.INPUT.SIZE[0], cfg.INPUT.SIZE[1]))
    out = backbone(fake_x)
    
    if isinstance(out, list):
        out = out[-1]
    
    
    _, C, _, _ = out.shape
    ################################
    metric_product = METRICS_REGISTRY.get(cfg.MODEL.METRIC.NAME) if cfg.MODEL.METRIC.NAME else None
    heads = build_head(cfg, C)
    model = Model(
        backbone=backbone,
        heads=heads,
        cfg=cfg,
    )
    return model


def build_head(cfg, C):
    """
    Build the whole model head, defined by ``cfg.MODEL.HEAD``.
    Note that it does not load any weights from ``cfg``.
    """
    head_archs = cfg.MODEL.HEAD.NAME
    model = {}
    for head_arch in head_archs:
        model[head_arch] = HEAD_REGISTRY.get(head_arch)(cfg, C) if head_arch else None
    return model
