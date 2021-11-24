import numpy as np
import torch
import yaml
from easydict import EasyDict
from model.hrnet import PoseHighResolutionNet


def read_config(file):
    """
    Reads configuration from yaml file.
    Args:
        file: Path to configuration file.
    Returns:
        An EasyDcit object containing configuration information.
    """
    with open(str(file), 'r') as f:
        config = EasyDict(yaml.load(f, Loader=yaml.Loader))
    return config


def get_max_preds(batch_heatmaps, sides):
    '''
    get predictions from score maps
        batch_heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        sides: list of str, "left" or "right"
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    assert batch_size == len(sides)
    for i in range(batch_size):
        if sides[i] == "left":
            preds[i, :, 0] = width - (preds[i, :, 0]) % width
        else:
            preds[i, :, 0] = (preds[i, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


class HandPoseEstimator(object):
    def __init__(self, config):
        self.config = config.POSE
        cfg = read_config(self.config.CFG_FILE)

        self.model = PoseHighResolutionNet(cfg)
        state = torch.load(self.config.TRAINED_MODEL)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()
        print("Finished loading hand pose model: {}".format(self.config.TRAINED_MODEL))

        if self.config.CUDA:
            self.model = self.model.cuda()

    def __call__(self, images, sides, top_lefts, scale_ratios):
        stride_size = 4
        batch_size = images.shape[0]

        if self.config.CUDA:
            images = images.float().cuda(non_blocking=True)

        heatmap = self.model(images)

        heatmap_np = heatmap.detach().cpu().numpy()
        predict_pose, maxvals = get_max_preds(heatmap_np, sides)

        predict_pose[:, :, 0] *= stride_size
        predict_pose[:, :, 1] *= stride_size

        top_left = np.asarray(top_lefts).reshape(batch_size, 1, 2)
        scale_ratio = np.asarray(scale_ratios).reshape(batch_size, 1, 2)
        final_pose = top_left + predict_pose * scale_ratio

        score = np.max(maxvals.reshape(batch_size, -1), axis=1)

        return final_pose, score

