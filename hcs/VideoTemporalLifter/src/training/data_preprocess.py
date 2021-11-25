import numpy as np
import torch
import pickle
import sys
import os
import time
sys.path.append('../')
sys.path.append("./VideoTemporalLifter/src/")
sys.path.append("./VideoTemporalLifter/")

from util.misc import load, assert_shape
from util.pose import remove_root, remove_root_keepscore, combine_pose_and_trans
from databases.joint_sets import HandJoints
from src.model.videopose import TemporalModel



def load_model(model_folder):
    config = load(os.path.join( model_folder, 'config.json'))
    path = os.path.join(model_folder, 'model_params.pkl')

    # Input/output size calculation is hacky
    weights = torch.load(path)
    num_in_features = weights['expand_conv.weight'].shape[1]

    m = TemporalModel(num_in_features, HandJoints.NUM_JOINTS, config['model']['filter_widths'],
                      dropout=config['model']['dropout'], channels=config['model']['channels'])

    m.cuda()
    m.load_state_dict(weights)
    m.eval()

    return config, m

def preprocess_2d(data, fx, cx, fy, cy, joint_set):

    assert_shape(data, ("*", None, joint_set.NUM_JOINTS, 3))
    assert not isinstance(fx, np.ndarray) or len(fx) == len(data)
    assert not isinstance(fy, np.ndarray) or len(fy) == len(data)

    # negligible
    if isinstance(fx, np.ndarray):
        N = len(data)
        shape = [1] * (data.ndim - 1)
        shape[0] = N
        fx = fx.reshape(shape)
        fy = fy.reshape(shape)
        cx = cx.reshape(shape)
        cy = cy.reshape(shape)



    # This is 100ms
    data[..., :, 0] -= cx
    data[..., :, 1] -= cy
    data[..., :, 0] /= fx
    data[..., :, 1] /= fy

    root_ind = 0
    root2d = data[..., root_ind, :].copy()  # negligible
    # 70ms
    data = remove_root_keepscore(data, root_ind)  # (nPoses, 13, 3), modifies data


    # negligible
    bad_frames = data[..., 2] < 0.1


    if isinstance(fx, np.ndarray):
        fx = np.tile(fx, (1,) + data.shape[1:-1])
        fy = np.tile(fy, (1,) + data.shape[1:-1])
        data[bad_frames, 0] = -1700 / fx[bad_frames]
        data[bad_frames, 1] = -1700 / fy[bad_frames]
    else:
        data[bad_frames, 0] = -1700 / fx
        data[bad_frames, 1] = -1700 / fy

    # stack root next to the pose
    data = data.reshape(data.shape[:-2] + (-1,))  
    data = np.concatenate([data, root2d], axis=-1)  
    return data

class BaseNormalizer(object):
    """
    Baseclass for preprocessors that normalize a field.

    Subclasses must set the field_name field by themselves, outside the constructor.
    They must also have the constructor to accept a single 'None' argument, that does
    not preload the parameters.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @classmethod
    def from_file(cls, path):
        state = load(path)
        return cls.from_state(state)

    @classmethod
    def from_state(cls, state):
        """
        Path is a pkl file that contains mean and std.
        """
        instance = cls()
        instance.mean = state['mean']
        instance.std = state['std']

        return instance

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std, 'field_name': self.field_name}

    def __call__(self, sample):

        sample[self.field_name] = (sample[self.field_name] - self.mean) / self.std
        return sample

class MeanNormalize2D(BaseNormalizer):
    """
    Normalizes the input 2D pose with mean and std.
    """

    def __init__(self):

        self.field_name = "pose2d"
        
        return


class MeanNormalize3D(BaseNormalizer):
    """
    Normalizes the input 3D pose with mean and std.
    """

    def __init__(self):

        self.field_name = 'pose3d'

        return

        
class DepthposeNormalize2D(object):
    """
    Normalizes the 2D pose using the technique in Depthpose.
    """

    def __init__(self, normalizer=None):

        self.normalizer = normalizer
        return

    @classmethod
    def from_file(cls, path, dataset):
        state = load(path)
        return cls.from_state(state, dataset)

    @classmethod
    def from_state(cls, state):
        instance = cls(MeanNormalize2D.from_state(state))
        return instance

    def state_dict(self):
        return self.normalizer.state_dict()

    def __call__(self, sample):

        pose2d = sample['pose2d'] 

        single_item = sample['pose2d'].ndim == 2

        if single_item:
            pose2d = np.expand_dims(pose2d, axis=0)

        preprocessed = preprocess_2d(pose2d.copy(), sample["fx"], sample["cx"],
                                        sample["fy"], sample["cy"],
                                        HandJoints())
        if single_item:
            preprocessed = preprocessed[0]
        
        # print("processed2d:{}".format(preprocessed))
        sample['pose2d'] = preprocessed
        sample = self.normalizer(sample)
        return sample

if __name__ == "__main__":
    with open ( './res_hrnet.pkl', 'rb' ) as file:
        hrnet_poses2d = pickle.load(file)
    hrnet_poses2d = np.asarray(hrnet_poses2d)
    # print(hrnet_poses2d.shape)
    score = np.ones((hrnet_poses2d.shape[0], hrnet_poses2d.shape[1], 1))*0.9
    hrnet_poses2d = np.stack((hrnet_poses2d[:,:,0], hrnet_poses2d[:,:,1], score[:,:,0]), axis = 2)
    # print(hrnet_poses2d.shape)
    # print(np.asarray(hrnet_poses2d).shape)
    poses2d = []
    poses3d = []

    poses2d.append(hrnet_poses2d)
    poses3d.append(hrnet_poses2d)

    pose2d = np.concatenate(poses2d).astype("float32")
    sample = {}

    sample["pose2d"] = pose2d

    #TODO: calibration_mx
    calibration_mx = [[1.72081829e+03,0.00000000e+00,5.58267104e+02],
            [0.00000000e+00,1.72023839e+03,1.04675427e+03],
            [0.00000000e+00,0.00000000e+00,1.00000000e+00]]
    calibration_mx = np.array(calibration_mx)
    N = hrnet_poses2d.shape[0]

    fx = []
    fy = []
    cx = []
    cy = []

    fx.extend([calibration_mx[0, 0]] * N)
    fy.extend([calibration_mx[1, 1]] * N)
    cx.extend([calibration_mx[0, 2]] * N)
    cy.extend([calibration_mx[1, 2]] * N)

    fx = np.array(fx, dtype='float32')
    fy = np.array(fy, dtype='float32')
    cx = np.array(cx, dtype='float32')
    cy = np.array(cy, dtype='float32')

    sample["fx"] = fx
    sample["fy"] = fy
    sample["cx"] = cx
    sample["cy"] = cy

    state = load("../../checkpoints/preprocess_params.pkl")
    # print(state)
    # depthposeNormalize2D_state = state[0]
    transform1 = DepthposeNormalize2D.from_state(state[0]['state'])
    sample1 = transform1(sample.copy())
    print(sample["pose2d"].shape)
    print(sample1["pose2d"].shape)
    pad = 40
    batch_2d = np.expand_dims(np.pad(sample1['pose2d'], ((pad, pad), (0, 0)), 'edge'), axis=0)
    print(batch_2d.shape)
    print(batch_2d[0][1])
    model_folder = "../../checkpoints"
    config, model = load_model(model_folder)
    start = time.time()
    pred3d = model(torch.from_numpy(batch_2d).cuda()).detach().cpu().numpy()
    end = time.time()

    print("time:{}".format(end - start))
    print(pred3d.shape)



    #correct
    


