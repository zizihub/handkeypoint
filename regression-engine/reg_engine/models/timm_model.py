import timm


def get_timm_model(cfg, **kwargs):
    _, model_name = cfg.MODEL.NAME.split('-')
    backbone = timm.create_model(model_name,
                                 pretrained=cfg.MODEL.PRETRAINED,
                                 features_only=True,
                                 scriptable=cfg.DEPLOY,
                                 exportable=cfg.DEPLOY,
                                 **kwargs)
    return backbone
