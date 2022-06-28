# Copyright (c) Facebook, Inc. and its affiliates.
from fvcore.common.registry import Registry

FPN_REGISTRY = Registry('FPN')  # noqa F401 isort:skip
FPN_REGISTRY.__doc__ = '''

Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
'''

POSTPROCESS_REGISTRY = Registry('POSTPROCESS')  # noqa F401 isort:skip
POSTPROCESS_REGISTRY.__doc__ = '''

Registry for post processing, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
'''

CONV2D_REGISTRY = Registry('CONV2D')  # noqa F401 isort:skip
CONV2D_REGISTRY.__doc__ = '''

Registry for Conv2d, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
'''

SPP_REGISTRY = Registry('SPP')  # noqa F401 isort:skip
SPP_REGISTRY.__doc__ = '''

Registry for SPP, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
'''


def build_fpn(cfg, in_channels):
    '''
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    '''
    fpn_arch = cfg.MODEL.FPN.NAME
    conv2d = build_conv2d(cfg)
    model = FPN_REGISTRY.get(fpn_arch)(cfg, in_channels, conv2d) if fpn_arch else None
    return model


def build_post(cfg, encoder_channels):
    '''
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    '''
    post_archs = cfg.MODEL.POSTPROCESS.NAME
    model = {}
    for post_arch in post_archs:
        model[post_arch] = POSTPROCESS_REGISTRY.get(post_arch)(cfg, encoder_channels) if post_arch else None
    return model


def build_conv2d(cfg):
    conv2d = CONV2D_REGISTRY.get(cfg.MODEL.CONV2D)
    return conv2d


def build_spp(cfg):
    spp = SPP_REGISTRY.get(cfg.MODEL.DECODER.SPP)
    return spp
