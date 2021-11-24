from logging import handlers
from collections import OrderedDict
import logging
import cv2
import numpy as np
# import torch


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)
    # handler = logging.FileHandler("log.txt")
    handler = handlers.TimedRotatingFileHandler(filename="log.txt", when="W6", backupCount=168, encoding='utf-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(ch)
    return logger


def get_matrix(pdata):
    K = np.asarray(pdata['intrinsic'])
    R = np.asarray(pdata['rot'])
    T = np.asarray(pdata['tran']).reshape(3, 1)
    RT = np.concatenate([R, T], -1)
    M = K.dot(RT)
    dist = np.asarray(pdata['dist'])
    return M, K, dist


def getA(u1, v1, u2, v2, M1, M2):
    row1 = np.array([u1*M1[2][0]-M1[0][0], u1*M1[2][1]-M1[0][1], u1*M1[2][2]-M1[0][2]])
    row2 = np.array([v1*M1[2][0]-M1[1][0], v1*M1[2][1]-M1[1][1], v1*M1[2][2]-M1[1][2]])
    row3 = np.array([u2*M2[2][0]-M2[0][0], u2*M2[2][1]-M2[0][1], u2*M2[2][2]-M2[0][2]])
    row4 = np.array([v2*M2[2][0]-M2[1][0], v2*M2[2][1]-M2[1][1], v2*M2[2][2]-M2[1][2]])
    return np.vstack([row1, row2, row3, row4])


def getB(u1, v1, u2, v2, M1, M2):
    row1 = np.array([M1[0][3] - u1*M1[2][3]])
    row2 = np.array([M1[1][3] - v1*M1[2][3]])
    row3 = np.array([M2[0][3] - u2*M2[2][3]])
    row4 = np.array([M2[1][3] - v2*M2[2][3]])
    return np.vstack([row1, row2, row3, row4])


def rectify(u, v, K, distCoeffs=np.zeros(5)):
     src = np.array([[[u, v]]], np.float64)
     dst = cv2.undistortPoints(src, K, distCoeffs)
     return K.dot(np.vstack([dst[0,0].reshape((-1,1)), np.array([[1]])])).reshape(3)[:2]


def get_wrist3D_multiview(wrist_locations, camera_param_dict):
    confidence = {}
    views = []
    for view, v in wrist_locations.items():
        for handside in v.keys():
            if handside not in confidence:
                confidence[handside] = []
            confidence[handside].append(v[handside][:, 2])
        views.append(view)
    M_list, K_list, d_list = [], [], []
    for view in views:
        M, K, dist = get_matrix(camera_param_dict[view])
        M_list.append(M)
        K_list.append(K)
        d_list.append(dist)
    wrists_3d = OrderedDict()
    for handside in confidence.keys():
        wrist_3d = []
        top2index = np.argsort(np.stack(confidence[handside], -1), -1)[:, -2:]
        for i in range(top2index.shape[0]):
            M1, K1, d1 = M_list[top2index[i, 0]], K_list[top2index[i, 0]], d_list[top2index[i, 0]]
            M2, K2, d2 = M_list[top2index[i, 1]], K_list[top2index[i, 1]], d_list[top2index[i, 1]]
            u1, v1 = rectify(wrist_locations[views[top2index[i, 0]]][handside][i, 0], wrist_locations[views[top2index[i, 0]]][handside][i, 1], K1, d1)
            u2, v2 = rectify(wrist_locations[views[top2index[i, 1]]][handside][i, 0], wrist_locations[views[top2index[i, 1]]][handside][i, 1], K2, d2)
            x, y, z = np.linalg.lstsq(getA(u1, v1, u2, v2, M1, M2), getB(u1, v1, u2, v2, M1, M2), rcond=None)[0]
            wrist_3d.append([x, y, z])
        wrists_3d[handside] = np.array(wrist_3d).astype('float32').reshape(-1, 3)
    return wrists_3d


# def get_init_trans(wrists_3d):
#     mano_wrists = {"right": np.array([0.0832, 0.0056, 0.0055]),
#                    "left": np.array([-0.1082, 0.0055, 0.0054])}
#     init_trans = OrderedDict()
#     for handside in wrists_3d.keys():
#         print(wrists_3d[handside].shape)
#         init_trans[handside] = wrists_3d[handside] / 1000.0 - mano_wrists[handside]
#     return init_trans


# def get_init_params(device, num_frames, correct_shape=False, handsides=["right"], initparams=None, inittrans=None, fix_trans=False):
#     param_dict = {}
#     if initparams is not None:
#         for k, v in initparams.items():
#             assert v.shape[0] == num_frames
#             init_pose = torch.from_numpy(v[:, :45]).float().to(device)
#             init_quat = torch.from_numpy(v[:, 45:49]).float().to(device)
#             if inittrans is None:
#                 fix_trans = False
#                 init_trans = torch.from_numpy(v[:, 49:52]).float().to(device)
#             else:
#                 init_trans = torch.from_numpy(inittrans[k]).float().to(device)
#             param_list = [init_pose, init_quat, init_trans]
#             for i in range(len(param_list)):
#                 param_list[i].requires_grad = True
#             if fix_trans:
#                 param_list[2].requires_grad = False
#             param_dict[k] = param_list
#         return param_dict

#     for item in handsides:
#         init_pose = torch.zeros(num_frames, 45).to(device)
#         init_quat = torch.zeros(num_frames, 4).to(device)
#         init_quat[:, 0] = -0.707
#         init_quat[:, 1] = 0.707
#         if inittrans is None:
#             fix_trans = False
#             init_trans = torch.zeros(num_frames, 3).to(device)
#             init_trans[:, -1] = -0.5
#         else:
#             init_trans = torch.from_numpy(inittrans[item]).float().to(device)
#         param_list = [init_pose, init_quat, init_trans]
#         if correct_shape:
#             init_shape_0 = torch.zeros(1, 1).to(device)
#             param_list.append(init_shape_0)
#         for i in range(len(param_list)):
#             param_list[i].requires_grad = True
#         if fix_trans:
#             param_list[2].requires_grad = False
#         param_dict[item] = param_list
#     return param_dict


def visualisze_joint(hand_joints, img, filename):
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])
    colors = np.uint8(colors*255)
    # define connections and colors of the bones
    bones = [((0, 1), colors[0, :]),
             ((1, 2), colors[1, :]),
             ((2, 3), colors[2, :]),
             ((3, 4), colors[3, :]),

             ((0, 5), colors[4, :]),
             ((5, 6), colors[5, :]),
             ((6, 7), colors[6, :]),
             ((7, 8), colors[7, :]),

             ((0, 9), colors[8, :]),
             ((9, 10), colors[9, :]),
             ((10, 11), colors[10, :]),
             ((11, 12), colors[11, :]),

             ((0, 13), colors[12, :]),
             ((13, 14), colors[13, :]),
             ((14, 15), colors[14, :]),
             ((15, 16), colors[15, :]),

             ((0, 17), colors[16, :]),
             ((17, 18), colors[17, :]),
             ((18, 19), colors[18, :]),
             ((19, 20), colors[19, :])]

    for connection, color in bones:
        for _, v in hand_joints.items():
            v = v.astype('int')
            coord1 = v[connection[0], :2]
            coord2 = v[connection[1], :2]
            cv2.line(img, tuple(coord1), tuple(coord2), color=tuple(color.tolist()), thickness=2)

    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
