import torch


def create_model(cfg, logger, **kwargs):
    """
    Creates model.
    Args:
        cfg: An EasyDict object containing configuration information.
        logger: logger.
    Returns:
        Model.
    """
    if cfg.MODEL.NAME == 'pose_hrnet':
        from .pose_hrnet import PoseHighResolutionNet
        model = PoseHighResolutionNet(cfg, **kwargs)
        model.init_weights(
            pretrained=cfg.MODEL.PRETRAINED,
            imagenet_pretrained=cfg.MODEL.IMAGENET_PRETRAINED
        )
    else:
        raise ValueError('Model name {} is invalid.'.format(cfg.MODEL.NAME))

    return model
