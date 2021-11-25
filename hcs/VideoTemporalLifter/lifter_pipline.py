import numpy as np
import sys
import torch
import os
import pickle
import time
import json
import matplotlib.pyplot as plt

from VideoTemporalLifter.src.model.videopose import TemporalModel
from VideoTemporalLifter.src.util.misc import load
from VideoTemporalLifter.src.databases.joint_sets import HandJoints
from VideoTemporalLifter.src.training.data_preprocess import DepthposeNormalize2D, MeanNormalize3D
from VideoTemporalLifter.src.training.data_postprocess import get_postprocessor


def plotHand3d(poses, colors, fig):
    """Plot the 3D pose showing the joint connections.
    """
    import mpl_toolkits.mplot3d.axes3d as p3
    # R = np.array ( [[1, 0, 0], [0, 0, 1], [0, -1, 0]] )
    R = np.array ( [[1, 0, 0], [0, 1, 0], [0, 0, 1]] )
    poses = [R @ i for i in poses]
    _CONNECTION = [[0,1], [0,5], [0,9], [0,13], [0,17], [1,2], [2,3], [3,4], [5,6], [6,7], [7,8], [9,10], [10,11], [11,12], [13,14], [14,15], [15,16], [17,18], [18,19],[19,20]]

    # fig = plt.figure ()
    import math
    rows = math.ceil ( math.sqrt ( len ( poses ) ) )

    ax = fig.gca ( projection='3d' )

    # smallest = [min ( [i[idx].min () for i in poses] ) for idx in range ( 3 )]
    # largest = [max ( [i[idx].max () for i in poses] ) for idx in range ( 3 )]

    # smallest = [min ( [i[idx].min () for i in [poses[0]]] ) for idx in range ( 3 )]
    # largest = [max ( [i[idx].max () for i in [poses[0]]] ) for idx in range ( 3 )]

    smallest = [min ( [i[idx].min () for i in [poses[0]]] ) for idx in range ( 3 )]
    largest = [max ( [i[idx].max () for i in [poses[0]]] ) for idx in range ( 3 )]

    plt.axis("auto")
    ax.set_xlim3d ( smallest[0], largest[0] )
    ax.set_ylim3d ( smallest[1], largest[1] )
    ax.set_zlim3d ( smallest[2], largest[2] )

    x_len = largest[0] - smallest[0]
    y_len = largest[1] - smallest[1]
    z_len = largest[2] - smallest[2]
    ax.set_box_aspect(aspect=(x_len, y_len, z_len))

    for i, pose in enumerate ( poses ):
        # if i != 0:
        #     continue
        # if i != 1:
        #     continue
        assert (pose.ndim == 2)
        assert (pose.shape[0] == 3)
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % (colors[i][0], colors[i][1], colors[i][2])
            ax.plot ( [pose[0, c[0]], pose[0, c[1]]],
                      [pose[1, c[0]], pose[1, c[1]]],
                      [pose[2, c[0]], pose[2, c[1]]], c=col )
        for j in range ( pose.shape[1] ):
            col = '#%02x%02x%02x' % (colors[i][0], colors[i][1], colors[i][2])
            ax.scatter ( pose[0, j], pose[1, j], pose[2, j],
                         c=col, marker='o', edgecolor=col )
        # ax.set_label ( f'#{i}' )
    return fig



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

class VideoTemporalLifter(object):
    def __init__(self, model_folder, calibration_mx, frames_num):

        self.frames_num = frames_num
        self.calibration_mx = np.asarray(calibration_mx)
        self.model_folder = model_folder

        config, model = load_model(self.model_folder)

        self.config = config
        self.model = model

        self.pad = (model.receptive_field() - 1) // 2

        self.params_path = os.path.join(self.model_folder, 'preprocess_params.pkl')

        fx = []
        fy = []
        cx = []
        cy = []

        fx.extend([self.calibration_mx[0, 0]] * frames_num)
        fy.extend([self.calibration_mx[1, 1]] * frames_num)
        cx.extend([self.calibration_mx[0, 2]] * frames_num)
        cy.extend([self.calibration_mx[1, 2]] * frames_num)

        self.fx = np.array(fx, dtype='float32')
        self.fy = np.array(fy, dtype='float32')
        self.cx = np.array(cx, dtype='float32')
        self.cy = np.array(cy, dtype='float32')

        self.sample = {}

        self.sample["fx"] = self.fx
        self.sample["fy"] = self.fy
        self.sample["cx"] = self.cx
        self.sample["cy"] = self.cy

        self.state = load(os.path.join(model_folder, "preprocess_params.pkl"))
        # print(self.state)
        self.pre_process_transform = DepthposeNormalize2D.from_state(self.state[0]['state'])

        self.normalizer3d = MeanNormalize3D.from_state(self.state[1]['state'])
        self.post_precess_transform = get_postprocessor(self.config, self.normalizer3d)

    def forword(self, keypoints_2d):
        """
        Params:
        - keypoints_2d : 2d keypoints of hrnet predict, shape : (frames_num, 21, 3), the last dim of "3" is the score of the keypoints;
        Return:
        - keypoints_3d : 3d keypoints of tpn predict, shape: (frames_num, 21, 3)
        """
        assert keypoints_2d.shape[0] == self.frames_num
        poses2d = []
        poses2d.append(keypoints_2d)
        poses2d = np.concatenate(poses2d).astype("float32")

        self.sample["pose2d"] = poses2d

        self.sample = self.pre_process_transform(self.sample.copy())
        batch_2d = np.expand_dims(np.pad(self.sample['pose2d'], ((self.pad, self.pad), (0, 0)), 'edge'), axis=0)

        pred3d = self.model(torch.from_numpy(batch_2d).cuda()).detach().cpu().numpy()

        poses3d = self.post_precess_transform(pred3d[0])
        return poses3d 


if __name__ == '__main__':
    frames_num = 400
    model_folder = './checkpoints'
    calibration_mx = [[
                1720.8182903492593,
                0.0,
                558.2671035775594
            ],
            [
                0.0,
                1720.2383941539035,
                1046.7542654266829
            ],
            [
                0.0,
                0.0,
                1.0
            ]]
    lifter = VideoTemporalLifter(model_folder, calibration_mx, frames_num)
    with open ( './014.json', 'rb' ) as file:
        pose_dic = json.load(file)

    hrnet_poses2d = []
    key_list = sorted(pose_dic.keys())

    for key in key_list:
        hrnet_poses2d.append(pose_dic[key])

    hrnet_poses2d = np.asarray(hrnet_poses2d)
    print("ori_pose_2d:{}".format(hrnet_poses2d[0]))
    print(hrnet_poses2d.shape)
    # score = np.ones((hrnet_poses2d.shape[0], hrnet_poses2d.shape[1], 1))*0.9
    # hrnet_poses2d = np.stack((hrnet_poses2d[:,:,0], hrnet_poses2d[:,:,1], score[:,:,0]), axis = 2)
    start = time.time()
    poses3d = lifter.forword(np.asarray(hrnet_poses2d))
    end = time.time()
    fig = plt.figure()
    all_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 215, 0), (0, 255, 255), (255, 255, 0)]
    for i in range(poses3d.shape[0]):
        
        pose_3d = poses3d[i]

        pose_3d = np.asarray([pose_3d])
        pose_3d = pose_3d.transpose(0, 2, 1)
        pose_3d.tolist()

        fig = plotHand3d(pose_3d, all_color, fig)

        plt.ion()
        plt.pause(0.001)
        plt.cla()

    print("time:{}".format(end-start))
    # print(poses3d.shape)
    print(poses3d[0])
    
    