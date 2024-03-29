"""
Hand Model Fitting
"""
import sys
# sys.path.append('/workspace/hand-complete-w-Handtailor/hcs/')
# print(sys.path)
import torch
import numpy as np
import cv2
import jax.numpy as npj
import PIL.Image as Image
from jax import grad, jit, vmap
from jax.experimental import optimizers
from torchvision.transforms import functional
import pickle


from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

import sys
import torch
import numpy as np
import cv2
import json
import os
import math
import time

from VideoTemporalLifter.config import cfg_hrnet as cfg_hrnet
from VideoTemporalLifter.HRNet.pose_detect import HandKeypointEstimator

from VideoTemporalLifter.model.hrnet.pose_hrnet import PoseHighResolutionNet
from VideoTemporalLifter.inference_util import crop_image_with_static_size, get_max_preds, flip_img, plot_hand
from VideoTemporalLifter.lifter_pipline import VideoTemporalLifter
from VideoTemporalLifter.core import config

CONFIG_PATH = "./VideoTemporalLifter/core/w32_256x256_adam_lr1e-3.yaml"
MODEL_PATH = "./VideoTemporalLifter/checkpoints/pose_hrnet_w32_256x256.pth"
INPUT_WIDTH = 256
NUM_JOINTS = 16


mano2cmu = [
    0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20
]

valid2full = [5, 0, 6, 7, 8, 9, 10, 11, 12,  # index
              13, 1, 14, 15, 16, 17, 18, 19, 20,  # middle
              21, 2, 22, 23, 24, 25, 26, 27, 28,  # pinky
              29, 3, 30, 31, 32, 33, 34, 35, 36,  # ring
              37, 4, 38, 39, 40, 41, 42, 43, 44,  # thumb
              ]




class Solver(object):

    def __init__(self, calibration_mx, frames_num = 10, mode = 1):

        model_floder = './VideoTemporalLifter/checkpoints'

        self.hrnet_detector = PoseKeypointEstimator()
        self.hrnet_hand_detector = HandPoseDetector_HRNet(cfg_hrnet)
        self.lifter = VideoTemporalLifter(model_floder, calibration_mx, frames_num)

        frame_index = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # init scale of hand bbox
        self.scale = [350, 350]

        right_pose_list = []
        left_pose_list = []
        self.pose_list = []

        self.pose_list.append(right_pose_list)
        self.pose_list.append(left_pose_list)

        self.fig = plt.figure()

        self.all_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 215, 0), (0, 255, 255), (255, 255, 0)]

        self.mode = mode

        self.init_pose = []
        self.left_angles = []
        self.right_angles = []

    @torch.no_grad()
    def __call__(self, img, mode = 1, left=True):

        if self.mode != mode:
            self.init_pose = []
            self.mode = mode
            self.left_angles = []
            self.right_angles = []
        output = {
            "angle": [0, 0],
        }

        frame = img

        width = frame.shape[1]
        # frame.shape (480, 640, 3)

        final_pose, score = self.hrnet_detector.forward(frame)
        # bbox_list = getBbox(final_pose[0], frame.shape, self.scale)
        bbox_list = [[0, 320, 80, 400], [320, 640, 80, 400]]

        image_list = []
        top_left_list = []
        hand_side_list = ['left', 'right']


        for bbox in bbox_list:
            cropped_image = frame[bbox[2]:bbox[3], bbox[0]:bbox[1], :]
            image_list.append(cropped_image)
            top_left_list.append(np.asarray([bbox[0], bbox[2]]))

            cv2.rectangle(frame, (bbox[0], bbox[2]), (bbox[1], bbox[3]), (0, 255, 0), 2)

        final_poses, _, pose_with_score = self.hrnet_hand_detector.forward(image_list, top_left_list, hand_side_list)

        pose_3d_list = []

        for i in range(final_poses.shape[0]):

            pose = final_poses[i]

            self.scale[i] = max(250, get_scale_from_pose(pose))
            plot_hand(pose, frame)

            pose_2d_score = pose_with_score[i]
            if hand_side_list[i] == 'left':
                fliped_pose = flip_pose(pose_2d_score, width)
                pose_2d_score = fliped_pose
                # plot_hand(pose_2d_score[:,:2], frame)

            self.pose_list[i].append(pose_2d_score)
            self.pose_list[i] = self.pose_list[i][-10:]

            if len(self.pose_list[i]) < 10:
                return output

            pose_3d = self.lifter.forword(np.asarray(self.pose_list[i]))

            pose_3d = pose_3d[-1]
            pose_3d = np.asarray([pose_3d])
            pose_3d = pose_3d.transpose(0, 2, 1)
            pose_3d.tolist()

            # # TODO: plot 3d pose
            # if i == 1:
            #     self.fig = plotHand3d(pose_3d, self.all_color, self.fig)
            #
            #     plt.ion()
            #     plt.pause(0.001)
            #     plt.cla()

            # TODO: calculate vector angle
            pose_3d = pose_3d.transpose(0, 2, 1)

            pose_3d_list.append(pose_3d)

        if len(pose_3d_list) != 2:
            pose_3d_list.append(pose_3d_list[0])

        if not len(self.init_pose):
            self.init_pose = pose_3d_list
            sickside_angles = 0
            goodside_angles = 0
        else:

            self.left_angles.append(diff_pose([pose_3d_list[0],self.init_pose[0]], self.mode))
            self.right_angles.append(diff_pose([pose_3d_list[1],self.init_pose[1]], self.mode))

            self.left_angles = self.left_angles[-10:]
            self.right_angles = self.right_angles[-10:]
            left_angle = sum(self.left_angles)/len(self.left_angles)
            right_angle = sum(self.right_angles)/len(self.right_angles)

            if left:
                sickside_angles = left_angle
                goodside_angles = right_angle
            else:
                sickside_angles = right_angle
                goodside_angles = left_angle


        output.update({
            "angle": [int(sickside_angles), int(goodside_angles)],
        })


        cv2.putText(frame, str([int(sickside_angles), int(goodside_angles)]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

        cv2.imshow('frame',frame)
        cv2.waitKey(2)

        return output



if __name__ == "__main__":
    # test codes

    color = np.array(Image.open('000000.jpg'))
    Ks = pickle.load(open('000000.pkl', "rb"))['ks']


    # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    H, W, C = color.shape
    print(color.shape)
    color_left = color[:, :H, :]
    color_right = color[:, -1:-H:-1, :]
    output = {
        "opt_params": [],
        "vertices": [],
        "hand_joints": []
    }
    solver = Solver(Ks=Ks, size=H)

    dd = pickle.load(open("./mano/MANO_RIGHT.pkl", 'rb'), encoding='latin1')
    face = np.array(dd['f'])
    renderer = utils.MeshRenderer(face, img_size=256)

    for i, img in enumerate([color_left, color_right]):
        frame = img.copy()
        frame = cv2.resize(frame, (256, 256), cv2.INTER_LINEAR)

        _ = solver(img, Ks, i)
        output["opt_params"].append(_["opt_params"])
        output["vertices"].append(_["vertices"])
        output["hand_joints"].append(_["hand_joints"])

        # frame1 = renderer(np.multiply(_["vertices"], [-1, 1, 1]), solver.intr[0].cpu(), frame)
        frame1 = renderer(_["vertices"], solver.intr[0].cpu(), frame)

        cv2.imwrite(f"img{i}.jpg", np.flip(frame1, -1))


def convert_cam2img(samples, K):
    K = np.asarray(K)
    samples = np.asarray(samples)
    samples = samples.reshape((-1, 21, 3))
    res = []
    # print(samples.shape)
    for cam_points in samples:
        img_points = []
        for cam_point in cam_points:
            img_point = np.dot(K, cam_point)
            # print(img_point.shape)
            img_point[0] = img_point[0] / img_point[2]
            img_point[1] = img_point[1] / img_point[2]
            img_points.append(img_point[:2])
        res.append(img_points)
    res = np.asarray(res)
    # print(res.shape)
    return res


def plotHand3d(poses, colors, fig):
    """Plot the 3D pose showing the joint connections.
    """
    import mpl_toolkits.mplot3d.axes3d as p3
    # R = np.array ( [[1, 0, 0], [0, 0, 1], [0, -1, 0]] )
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    poses = [R @ i for i in poses]
    _CONNECTION = [[0, 1], [0, 5], [0, 9], [0, 13], [0, 17], [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10],
                   [10, 11], [11, 12], [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20]]

    # fig = plt.figure ()
    import math
    rows = math.ceil(math.sqrt(len(poses)))

    ax = fig.gca(projection='3d')

    # smallest = [min ( [i[idx].min () for i in poses] ) for idx in range ( 3 )]
    # largest = [max ( [i[idx].max () for i in poses] ) for idx in range ( 3 )]

    # smallest = [min ( [i[idx].min () for i in [poses[0]]] ) for idx in range ( 3 )]
    # largest = [max ( [i[idx].max () for i in [poses[0]]] ) for idx in range ( 3 )]

    smallest = [min([i[idx].min() for i in [poses[0]]]) for idx in range(3)]
    largest = [max([i[idx].max() for i in [poses[0]]]) for idx in range(3)]

    plt.axis("auto")
    ax.set_xlim3d(smallest[0], largest[0])
    ax.set_ylim3d(smallest[1], largest[1])
    ax.set_zlim3d(smallest[2], largest[2])

    x_len = largest[0] - smallest[0]
    y_len = largest[1] - smallest[1]
    z_len = largest[2] - smallest[2]
    ax.set_box_aspect(aspect=(x_len, y_len, z_len))

    for i, pose in enumerate(poses):
        # if i != 0:
        #     continue
        # if i != 1:
        #     continue
        assert (pose.ndim == 2)
        assert (pose.shape[0] == 3)
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % (colors[i][0], colors[i][1], colors[i][2])
            ax.plot([pose[0, c[0]], pose[0, c[1]]],
                    [pose[1, c[0]], pose[1, c[1]]],
                    [pose[2, c[0]], pose[2, c[1]]], c=col)
        for j in range(pose.shape[1]):
            col = '#%02x%02x%02x' % (colors[i][0], colors[i][1], colors[i][2])
            ax.scatter(pose[0, j], pose[1, j], pose[2, j],
                       c=col, marker='o', edgecolor=col)
        # ax.set_label ( f'#{i}' )
    return fig


class HandPoseDetector_HRNet(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.hrnet_detector = HandKeypointEstimator(self.cfg)

    def forward(self, image_list, top_left_list, hand_side_list):
        image_list = image_list
        top_left_list = np.asarray(top_left_list)
        poses, score, pose_with_score = self.hrnet_detector.forward(image_list, hand_side_list)

        final_poses = []

        for i in range(top_left_list.shape[0]):
            final_pose = poses[i] + top_left_list[i]
            final_poses.append(final_pose)
        final_poses = np.asarray(final_poses)
        return final_poses, score, pose_with_score


COCO_KP_ORDER = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]


def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right _wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines


class HandPoseDetector_HRNet(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.hrnet_detector = HandKeypointEstimator(self.cfg)

    def forward(self, image_list, top_left_list, hand_side_list):
        image_list = image_list
        top_left_list = np.asarray(top_left_list)
        poses, score, pose_with_score = self.hrnet_detector.forward(image_list, hand_side_list)

        final_poses = []

        for i in range(top_left_list.shape[0]):
            final_pose = poses[i] + top_left_list[i]
            final_poses.append(final_pose)

            pose_with_score[i, :, :2] = pose_with_score[i, :, :2] + top_left_list[i]

        final_poses = np.asarray(final_poses)
        return final_poses, score, pose_with_score


class PoseKeypointEstimator():

    def __init__(self):

        # 模型加载代码
        self.cfg = config.read_config(CONFIG_PATH)
        self.model = PoseHighResolutionNet(self.cfg)

        # cuda or cpu
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.model = self.model.cuda()
        else:
            self.device = 'cpu'

        state = torch.load(MODEL_PATH)
        self.model.load_state_dict(state)
        self.model.eval()

        print("Finished loading hand pose model: {}".format(MODEL_PATH))

    def preprocess(self, image):

        # 图像前处理代码
        cropped_image = image
        scale = np.asarray(cropped_image.shape[:2], dtype='int')

        scale_ratio = np.array([cropped_image.shape[1] / INPUT_WIDTH,
                                cropped_image.shape[0] / INPUT_WIDTH])
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = (cropped_image / 255.0 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
        cropped_image = cv2.resize(cropped_image, (INPUT_WIDTH, INPUT_WIDTH))

        cropped_image = np.expand_dims(cropped_image.transpose(2, 0, 1), 0)
        cropped_image = torch.from_numpy(cropped_image).float().to(self.device)

        return cropped_image, scale, scale_ratio

    def forward(self, ori_image, hand_side="right"):

        stride_size = 4

        # 图像预处理
        cropped_image, scale, scale_ratio = self.preprocess(ori_image)

        heatmap = self.model(cropped_image)
        heatmap_np = heatmap.detach().cpu().numpy()

        predict_pose, maxvals = get_max_preds(heatmap_np, [hand_side])
        predict_pose[:, :, 0] *= stride_size
        predict_pose[:, :, 1] *= stride_size

        final_pose = predict_pose * scale_ratio
        score = np.max(maxvals)

        return final_pose, score

    def visualization(self, img, final_pose):
        points = []
        for i in range(NUM_JOINTS):
            points.append((int(final_pose[0][i][0]), int(final_pose[0][i][1])))
        POSE_PAIRS = kp_connections(COCO_KP_ORDER)
        for pair in POSE_PAIRS:
            idFrom = pair[0]
            idTo = pair[1]

            if points[idFrom] and points[idTo]:
                cv2.line(img, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(img, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(img, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
        return img


def getBbox(pose, frame_shape, scale=[350, 350]):
    left_center = pose[10]
    right_center = pose[15]
    bbox_list = []

    index = 0

    for center in [left_center, right_center]:
        xmin = int(center[0] - scale[index] / 2)
        xmax = int(center[0] + scale[index] / 2)
        ymin = int(center[1] - scale[index] / 2)
        ymax = int(center[1] + scale[index] / 2)

        index += 1

        x_min = np.maximum(xmin, 0)
        x_max = np.minimum(xmax, frame_shape[1])
        y_min = np.maximum(ymin, 0)
        y_max = np.minimum(ymax, frame_shape[0])

        bbox_list.append([x_min, x_max, y_min, y_max])

    return bbox_list


def get_scale_from_pose(pose):
    x_min = np.min(pose[:, 0])
    x_max = np.max(pose[:, 0])
    y_min = np.min(pose[:, 1])
    y_max = np.max(pose[:, 1])

    x_len = x_max - x_min
    y_len = y_max - y_min
    scale = max(x_len, y_len)

    return scale * 2


def flip_pose(pose, width):
    fliped_pose = pose.copy()
    x = width - (pose[:, 0] % width)
    fliped_pose[:, 0] = x

    return fliped_pose


def calc_angle(vector1, vector2):
    vector_length1 = np.sqrt(vector1.dot(vector1))
    vector_length2 = np.sqrt(vector2.dot(vector2))

    cos_angle = vector1.dot(vector2) / (vector_length1 * vector_length2)
    theta = np.arccos(cos_angle)
    angle = theta * 360 / 2 / np.pi

    return angle


def diff_pose(pose_3d_list, mode=1):
    """
    Params:
        -pose_3d_list : list of pose_3d, 2 * 1 * 21 * 3
        -mode : 1 for palm and 2 for fist
    Return:
        -angle
    """

    if mode == 1:

        vector1 = pose_3d_list[0][0][9] - pose_3d_list[0][0][0]
        vector2 = pose_3d_list[1][0][9] - pose_3d_list[1][0][0]

        angle = calc_angle(vector1, vector2)

    elif mode == 2:

        max_angle_list = []
        for i in range(2):
            max_angle = 0
            # for finger_num in [5, 9, 13, 17]:
            for finger_num in [9, 13, 17]:
                vector1 = pose_3d_list[i][0][0] - pose_3d_list[i][0][finger_num]
                vector2 = pose_3d_list[i][0][finger_num + 1] - pose_3d_list[i][0][finger_num]
                tmp_angle = calc_angle(vector1, vector2)
                max_angle = max(tmp_angle, max_angle)

            max_angle_list.append(max_angle)

        angle = np.abs(max_angle_list[0] - max_angle_list[1])

    return angle