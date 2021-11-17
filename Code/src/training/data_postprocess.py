import numpy as np
import sys
sys.path.append('../')
from databases.joint_sets import HandJoints
from util.misc import assert_shape

def combine_pose_and_trans(data3d, std3d, mean3d, joint_set, root_name, log_root_z=True):
    """
    3D result postprocess: unnormalizes data3d and reconstructs the absolute pose from relative + absolute split.

    Parameters:
        data3d: output of the PyTorch model, ndarray(nPoses, 3*nJoints), in the format created by preprocess3d
        std3d: normalization standard deviations
        mean3d: normalization means
        root_name: name of the root joint
        log_root_z: The z coordinate of the depth is in logarithms

    Returns:
        ndarray(nPoses, nJoints, 3)
    """
    assert_shape(data3d, (None, joint_set.NUM_JOINTS * 3))

    # print("mean:{}".format(mean3d))
    # print("std:{}".format(std3d))
    data3d = data3d * std3d + mean3d
    # print("data:{}".format(data3d[0][0]))
    root = data3d[:, -3:]
    rel_pose = data3d[:, :-3].reshape((len(data3d), joint_set.NUM_JOINTS - 1, 3))
    # print("real_pose:{}".format(rel_pose[0]))

    if log_root_z:
        root[:, 2] = np.exp(root[:, 2])

    rel_pose += root[:, np.newaxis, :]
    # print("real_pose:{}".format(rel_pose[0]))

    result = np.zeros((len(data3d), joint_set.NUM_JOINTS, 3), dtype='float32')
    root_ind = joint_set.index_of(root_name)
    result[:, :root_ind, :] = rel_pose[:, :root_ind, :]
    result[:, root_ind, :] = root
    result[:, root_ind + 1:, :] = rel_pose[:, root_ind:, :]
    # print("result:{}".format(result[0]))
    return result

def get_postprocessor(config, normalizer3d):
    if config['preprocess_3d'] == 'SplitToRelativeAbsAndMeanNormalize3D':
        def f(x):
            scale = 1 
            return scale * combine_pose_and_trans(x, normalizer3d.std, normalizer3d.mean, HandJoints(), "hip")

        return f

    else:
        raise NotImplementedError('No unconverter for 3D preprocessing: ' + config['preprocess_3d'])