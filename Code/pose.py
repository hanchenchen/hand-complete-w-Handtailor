from core import config
from model.hrnet.pose_hrnet import PoseHighResolutionNet
from inference_util import crop_image_with_static_size, get_max_preds, flip_img, plot_hand
import matplotlib.pyplot as plt
 
import sys
import torch
import numpy as np
import cv2
import json
import os
import math
import time


from lifter_pipline import VideoTemporalLifter
from utils.inference_util import plot_hand
from config import cfg_hrnet as cfg_hrnet
from HRNet.pose_detect import HandKeypointEstimator
# from tqdm import tqdm
CONFIG_PATH = "./core/w32_256x256_adam_lr1e-3.yaml"
MODEL_PATH = "./checkpoints/pose_hrnet_w32_256x256.pth"
INPUT_WIDTH = 256
NUM_JOINTS = 16

def convert_cam2img(samples, K):

    K = np.asarray(K)
    samples = np.asarray(samples)
    samples = samples.reshape((-1,21,3))
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
        final_poses = np.asarray(final_poses)
        return final_poses, score, pose_with_score

class PoseKeypointEstimator():
    def __init__(self):
        #模型加载代码
        self.cfg = config.read_config(CONFIG_PATH)
        self.model = PoseHighResolutionNet(self.cfg)
        #cuda or cpu 
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.model = self.model.cuda()
        else:
            self.device = 'cpu'
        state = torch.load(MODEL_PATH)
        self.model.load_state_dict(state)
        self.model.eval()
        print("Finished loading hand pose model: {}".format(MODEL_PATH))

        # #cuda or cpu 
        # if torch.cuda.is_available():
        #     self.device = 'cuda'
        #     self.model = self.model.cuda()
        # else:
        #     self.device = 'cpu'
        
        # #show model construct
        # summary(self.model, input_size = (3,224,224))

    def preprocess(self, image):

        #图像前处理代码
        cropped_image = image

        scale = np.asarray(cropped_image.shape[:2], dtype='int')
        # print(scale)
        # scale_ratio = np.array([cropped_image.shape[1] / self.cfg.MODEL.INPUT_WIDTH,
        #                             cropped_image.shape[0] / self.cfg.MODEL.INPUT_HEIGHT])
        scale_ratio = np.array([cropped_image.shape[1] / INPUT_WIDTH,
                                    cropped_image.shape[0] / INPUT_WIDTH])
        # print(scale_ratio)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = (cropped_image / 255.0 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
        # cropped_image = cv2.resize(cropped_image, (self.cfg.MODEL.INPUT_HEIGHT, self.cfg.MODEL.INPUT_WIDTH))
        cropped_image = cv2.resize(cropped_image, (INPUT_WIDTH, INPUT_WIDTH))
        # cv2.imshow('1',cropped_image)
        # cv2.waitKey(0)
        
        #TODO: right or left hand?
        # if hand_side == 'left':
        #     cropped_image = flip_img(cropped_image).copy()

        cropped_image = np.expand_dims(cropped_image.transpose(2, 0, 1), 0)
        cropped_image = torch.from_numpy(cropped_image).float().to(self.device)

        return cropped_image, scale, scale_ratio


    def forward(self, ori_image, hand_side = "right"):
        #模型推理代码，输入为图像，输出即为手部关键点

        #HRNet中heatmap的尺寸为原图的1/4, stride_size = 4
        stride_size = 4

        #图像预处理
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


    def plot_hand(self, coords_xy, img):
        #可视化代码
        # plot_hand(coords_xy, img)
        print(coords_xy)
        
def getBbox(pose, frame_shape, scale = [350, 350] ):
    
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
    print(pose.shape)
    x_min = np.min(pose[:,0])
    x_max = np.max(pose[:,0])
    y_min = np.min(pose[:,1])
    y_max = np.max(pose[:,1])

    x_len = x_max - x_min
    y_len = y_max - y_min
    scale = max(x_len, y_len)

    return scale*2
def flip_pose(pose, width):
    fliped_pose = pose.copy()
    x = width - (pose[:, 0] % width)
    fliped_pose[:, 0] = x
    
    return fliped_pose


if __name__ == '__main__':

    calibration_mx = [[383.33, 0, 315.864],
            [0, 383.33, 239.169],
            [0, 0, 1]]
    model_floder = './checkpoints'
    frames_num = 10

    hrnet_detector = PoseKeypointEstimator()
    hrnet_hand_detector = HandPoseDetector_HRNet(cfg_hrnet)
    lifter = VideoTemporalLifter(model_floder, calibration_mx, frames_num)

    capture = cv2.VideoCapture(1)
    frame_index = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = [350, 350]

    right_pose_list = []
    left_pose_list = []
    pose_list = []
    pose_list.append(right_pose_list)
    pose_list.append(left_pose_list)

    fig = plt.figure()


    all_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 215, 0), (0, 255, 255), (255, 255, 0)]

    # pic_list = os.listdir("./pic_test")
    # for pic_name in pic_list:
    #     pic_path = os.path.join("./pic_test", pic_name)
    #     pic = cv2.imread(pic_path)
        
    #     frame = pic

    #     width = frame.shape[1]
    #     # print(frame.shape) (480, 640, 3)
    #     frame_index += 1
    #     final_pose, score = hrnet_detector.forward(frame)
    #     bbox_list = getBbox(final_pose[0], frame.shape, scale)
    #     # print(bbox_list)

    #     # index = 0
    #     # for pose in final_pose[0]:
    #     #     cv2.circle(frame, (int(pose[0]), int(pose[1])), 3, (0,255,0), 3)
    #     #     cv2.putText(frame, str(index),  (int(pose[0]), int(pose[1])), font, 1.2, (0,255,0), 3)
    #     #     index += 1 

    #     cv2.circle(frame, (int(final_pose[0][10][0]), int(final_pose[0][10][1])), 3, (0,255,0), 3)
    #     cv2.circle(frame, (int(final_pose[0][15][0]), int(final_pose[0][15][1])), 3, (0,255,0), 3)
    #     image_list = []
    #     top_left_list = []
    #     hand_side_list = ['right', 'left']

    #     for bbox in bbox_list:
    #         cropped_image = frame[bbox[2]:bbox[3], bbox[0]:bbox[1], :]
    #         image_list.append(cropped_image)
    #         top_left_list.append(np.asarray([bbox[0],bbox[2]]))

    #         cv2.rectangle(frame, (bbox[0], bbox[2]), (bbox[1], bbox[3]), (0,255,0), 2)
    #         # print(cropped_image.shape)
        
    #     final_poses, _, pose_with_score = hrnet_hand_detector.forward(image_list, top_left_list, hand_side_list)

    #     for i in range(final_poses.shape[0]):
            
    #         pose = final_poses[i]    

    #         scale[i] = max(50,get_scale_from_pose(pose))
    #         plot_hand(pose, frame)

    #         # pose_2d_score  = pose_with_score[i]
    #         # if hand_side_list[i] == 'left':
    #         #     fliped_pose = flip_pose(pose_2d_score, width)
    #         #     pose_2d_score = fliped_pose

    #         # #TODO: 0.9 score test
    #         # print(final_poses.shape)
    #         # # score = np.ones((final_poses.shape[1])) * 0.9
    #         # # pose_2d_score[:,2] = score

    #         # #TODO:

    #         # pose_list[i].append(pose_2d_score)
    #         # pose_list[i] = pose_list[i][-10:]
    #         # print(len(pose_list[i]))

    #         # if len(pose_list[i]) < 10:
    #         #     continue

    #         # # print(np.asarray(pose_list[i]).shape)
    #         # start = time.time()

    #         # pose_3d = lifter.forword(np.asarray(pose_list[i]))
    #         # end = time.time()
    #         # print("time:{}".format(end-start))
    #         # # print(pose_3d.shape)
    #         # pose_3d = pose_3d[-1]
    #         # pose_3d = np.asarray([pose_3d])
    #         # pose_3d = pose_3d.transpose(0, 2, 1)
    #         # pose_3d.tolist()

    #         # if i == 0:

    #         #     fig = plotHand3d(pose_3d, all_color, fig)

    #         #     plt.ion()
    #         #     plt.pause(0.001)
    #         #     plt.cla()

    #         #     # pose_2d = convert_cam2img( np.asarray([pose_3d]), np.asarray(calibration_mx))
    #         #     # plot_hand(pose_2d[0], frame)
            
        
            

    #     cv2.imshow('frame',frame)
    #     cv2.waitKey()


    
    while True:

        ret, frame = capture.read()

        width = frame.shape[1]
        # print(frame.shape) (480, 640, 3)
        frame_index += 1
        final_pose, score = hrnet_detector.forward(frame)
        bbox_list = getBbox(final_pose[0], frame.shape, scale)
        # print(bbox_list)

        # index = 0
        # for pose in final_pose[0]:
        #     cv2.circle(frame, (int(pose[0]), int(pose[1])), 3, (0,255,0), 3)
        #     cv2.putText(frame, str(index),  (int(pose[0]), int(pose[1])), font, 1.2, (0,255,0), 3)
        #     index += 1 

        # cv2.circle(frame, (int(final_pose[0][10][0]), int(final_pose[0][10][1])), 3, (0,255,0), 3)
        # cv2.circle(frame, (int(final_pose[0][15][0]), int(final_pose[0][15][1])), 3, (0,255,0), 3)
        image_list = []
        top_left_list = []
        hand_side_list = ['right', 'left']

        for bbox in bbox_list:
            cropped_image = frame[bbox[2]:bbox[3], bbox[0]:bbox[1], :]
            image_list.append(cropped_image)
            top_left_list.append(np.asarray([bbox[0],bbox[2]]))

            cv2.rectangle(frame, (bbox[0], bbox[2]), (bbox[1], bbox[3]), (0,255,0), 2)
            # print(cropped_image.shape)
        
        final_poses, _, pose_with_score = hrnet_hand_detector.forward(image_list, top_left_list, hand_side_list)

        

        vector_list = []
        vector_length = []

        for i in range(final_poses.shape[0]):
            
            
            pose = final_poses[i]    

            scale[i] = max(250,get_scale_from_pose(pose))
            plot_hand(pose, frame)

            pose_2d_score  = pose_with_score[i]
            if hand_side_list[i] == 'left':
                fliped_pose = flip_pose(pose_2d_score, width)
                pose_2d_score = fliped_pose

            # #TODO: 0.9 score test
            # print(final_poses.shape)
            # # score = np.ones((final_poses.shape[1])) * 0.9
            # # pose_2d_score[:,2] = score

            # #TODO:

            pose_list[i].append(pose_2d_score)
            pose_list[i] = pose_list[i][-10:]
            print(len(pose_list[i]))

            if len(pose_list[i]) < 10:
                continue

            # # print(np.asarray(pose_list[i]).shape)
            start = time.time()

            pose_3d = lifter.forword(np.asarray(pose_list[i]))
            end = time.time()
            print("time:{}".format(end-start))
            # print(pose_3d.shape)
            pose_3d = pose_3d[-1]
            pose_3d = np.asarray([pose_3d])
            pose_3d = pose_3d.transpose(0, 2, 1)
            pose_3d.tolist()


            if i == 0:

                fig = plotHand3d(pose_3d, all_color, fig)

                plt.ion()
                plt.pause(0.001)
                plt.cla()

                # pose_2d = convert_cam2img( np.asarray([pose_3d]), np.asarray(calibration_mx))
                # plot_hand(pose_2d[0], frame)
            
            #TODO: vector

            pose_3d = pose_3d.transpose(0, 2, 1)
            print("pose_3d.shape:{}".format(pose_3d.shape))
            vector = pose_3d[0][9] - pose_3d[0][0]
            vector_list.append(vector)
            vector_length.append(np.sqrt(vector.dot(vector)))

        # print("len_vector_list:{}".format(len(vector_list)))
        if len(vector_list) != 2 or len(vector_length) != 2:
            continue
        cos_angle = vector_list[0].dot(vector_list[1])/(vector_length[0]*vector_length[1])
        theta = np.arccos(cos_angle)
        angle=theta*360/2/np.pi
        cv2.putText(frame, str(angle), (50, 50), font, 1.2, (0,255,0), 3)    
        

            
        
            

        cv2.imshow('frame',frame)
        cv2.waitKey(2)

