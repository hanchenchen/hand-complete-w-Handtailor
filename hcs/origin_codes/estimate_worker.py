import numpy as np
import cv2
import torch
import heapq
from collections import OrderedDict
from .estimate_solve import Solver
from scipy import ndimage
from utils import Completeness
from .hrnet import inference, postprocess


# def Worker(inputs_queue, output_queue, proc_id, init_params,
#            wrist_rot_pitch, wrist_rot_yaw, gesture,
#            left_wrist_loc, right_wrist_loc, y_mtip,
#            height, width, step_size, num_iters,
#            threshold, lefthand, righthand,
#            w_silhouette, w_pointcloud, w_poseprior, left, use_pcaprior=True):
# hand_joints 每个循环更新
def Worker(inputs_queue, output_queue, proc_id, init_params, hand_joints, extra_verts,
           wrist_rot_pitch, wrist_rot_yaw, gesture,
           height, width, step_size, num_iters,
           threshold, lefthand, righthand,
           w_silhouette, w_pointcloud, w_poseprior, w_reprojection,  w_temporalprior, left, use_pcaprior=True):
    solver = Solver(init_params=init_params,
                    step_size=step_size,
                    num_iters=num_iters,
                    threshold=threshold,
                    w_poseprior=w_poseprior,
                    w_pointcloud=w_pointcloud,
                    w_silhouette=w_silhouette,
                    w_reprojection=w_reprojection,
                    w_temporalprior=w_temporalprior,
                    lefthand=lefthand,
                    righthand=righthand,
                    verbose=False,
                    use_pcaprior=use_pcaprior)
    # learnable_params = get_init_params(init_params, solver.device, gesture)
    learnable_params = get_init_params(init_params, solver.device, gesture)
    v = np.linspace(0, height - 1, height)
    u = np.linspace(0, width - 1, width)
    coords_u, coords_v = np.meshgrid(u, v)
    # optimizer = torch.optim.SGD(learnable_params, lr=step_size, momentum=0.0, nesterov=False)
    completeness_estimator = Completeness(gesture, wrist_rot_pitch, wrist_rot_yaw)
    hrnet = inference.get_hrnet()
    count = 0
    while True:
        # if not inputs_queue.empty():
        meta = inputs_queue.get()
        if meta == 'STOP':
            print("Quit Estimate Process.")
            break
        else:
            color, depth, Ks = meta['color'][0], meta['depth'][0], meta['ks'][0]
            # outputs = get_silhouettes_and_pointclouds(depth, color, Ks, coords_u, coords_v,
            #                                           left_wrist_loc, right_wrist_loc, y_mtip)
            outputs = get_silhouettes_and_pointclouds_tracking(depth, color, Ks, coords_u, coords_v,
                                                               hand_joints, hrnet, offset=20, idx=count)
            count += 1
            # if type(outputs) == int:
            #     output_queue.put(outputs)
            # else:
            silhouettes, pointclouds, pred_pose = outputs
            # count += 1
            # visualisze_joint(pred_pose, color, count)
            if len(pointclouds[0]) == 0 or len(pointclouds[1]) == 0:
                output_queue.put(1)
            else:
                # make target
                fit_target = OrderedDict()
                fit_target['silhouette_l'] = silhouettes[0]
                fit_target['silhouette_r'] = silhouettes[1]
                fit_target['pointcloud_l'] = pointclouds[0]
                fit_target['pointcloud_r'] = pointclouds[1]
                if gesture == "fist":
                    fit_target['hand_joints_l'] = pred_pose[0]
                    fit_target['hand_joints_r'] = pred_pose[1]
                # opt = solver(learnable_params, fit_target, gesture, optimizer)
                opt = solver(learnable_params, fit_target, Ks, gesture)
                # completeness = completeness_estimator(opt["opt_params"].astype('float32'),
                #                                       init_params, left)
                completeness, sickside_angle, goodside_angle = completeness_estimator(opt["glb_rot"].astype('float32'),
                                                                                      opt["Rs"].astype('float32'), left)
                hand_joints = opt['hand_joints'].astype('int')

                mismatchness = np.mean(np.sqrt(np.sum(np.square(pred_pose - hand_joints), -1)), -1)
                vertices = np.concatenate(
                    (opt["vertices"].astype('float32').reshape(2, -1, 3), extra_verts.reshape(2, -1, 3)), 1).reshape(-1, 3) 
                output = {
                    "completeness": completeness,
                    "vertices": vertices,
                    "opt_params": opt["opt_params"].astype('float32'),
                    # "vertices": opt["vertices"][0].astype('float32'),
                    "hand_joints": hand_joints,
                    "angles": [sickside_angle, goodside_angle],
                    "mismatchness": mismatchness
                }
                output_queue.put(output)
        # else:
        #     print("Empty queue")


def get_init_params(init_params, device, gesture):

    # Add gradient
    if gesture == "fist":
        init_pose_l = torch.from_numpy(init_params[:1, :45]).float().to(device)
        init_pose_r = torch.from_numpy(init_params[1:, :45]).float().to(device)
        # init_pose_l = torch.zeros((1, 13)).to(device)
        # init_pose_r = torch.zeros((1, 13)).to(device)
        init_quat_l = torch.zeros((1, 4)).to(device)
        init_quat_l[0, 0] = 1.0
        init_quat_r = torch.zeros((1, 4)).to(device)
        init_quat_r[0, 0] = 1.0
        init_pose_l.requires_grad = True
        init_pose_r.requires_grad = True
        param_list = [init_pose_l, init_quat_l, init_pose_r, init_quat_r]
    else:
        init_pose_l = torch.from_numpy(init_params[:1, :45]).float().to(device)
        init_pose_r = torch.from_numpy(init_params[1:, :45]).float().to(device)
        init_quat_l = torch.zeros((1, 2)).to(device)
        init_quat_l[0, 0] = 1.0
        init_quat_l.requires_grad = True
        init_quat_r = torch.zeros((1, 2)).to(device)
        init_quat_r[0, 0] = 1.0
        init_quat_r.requires_grad = True
        param_list = [init_pose_l, init_quat_l, init_pose_r, init_quat_r]
    return param_list


# def get_init_params(init_params, device, gesture):
#     init_pose_l = torch.from_numpy(init_params[0, :45]).float().to(device).unsqueeze(0)
#     init_pose_r = torch.from_numpy(init_params[1, :45]).float().to(device).unsqueeze(0)
#     init_shape_l = torch.from_numpy(init_params[0, 45:55]).float().to(device).unsqueeze(0)
#     init_shape_r = torch.from_numpy(init_params[1, 45:55]).float().to(device).unsqueeze(0)
#     init_quat_l = torch.from_numpy(init_params[0, 55:59]).float().to(device).unsqueeze(0)
#     init_quat_r = torch.from_numpy(init_params[1, 55:59]).float().to(device).unsqueeze(0)
#     init_cam_l = torch.from_numpy(init_params[0, 59:]).float().to(device).unsqueeze(0)
#     init_cam_r = torch.from_numpy(init_params[1, 59:]).float().to(device).unsqueeze(0)
#     # Add gradient
#     param_list = [init_pose_l, init_shape_l, init_quat_l, init_cam_l,
#                   init_pose_r, init_shape_r, init_quat_r, init_cam_r]
#     if gesture == "fist":
#         param_list[0].requires_grad = True
#         param_list[4].requires_grad = True
#     else:
#         param_list[2].requires_grad = True
#         param_list[6].requires_grad = True
#     return param_list


def get_silhouettes_and_pointclouds(depth, color, ks, coords_u, coords_v, left_wrist_loc, right_wrist_loc, y_mtip):
    # Using skin detector get silhouettes and pointclouds
    if len(depth.shape) == 3:
        depth = np.squeeze(depth)
    # middle_u = (self.left_wrist_loc[0] + self.right_wrist_loc[0]) / 2
    minv = max(min(left_wrist_loc[1], right_wrist_loc[1]) - 10, 0)
    maxv = min(y_mtip + 10, coords_v.shape[0])

    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([100, 255, 255])
    roi_v = np.logical_and(coords_v > minv, coords_v < maxv)
    color_mask = cv2.inRange(hsv_color, lower_green, upper_green)
    color_mask = np.logical_and(np.logical_not(color_mask), roi_v)

    # r = color[:, :, 2]
    # g = color[:, :, 1]
    # b = color[:, :, 0]
    # color_mask = np.logical_and(np.logical_and(r > 115, b > 40), g > 40)

    labeled_array, num_objects = ndimage.measurements.label(color_mask > 0)
    # if num_objects < 2:
    #     # 手部丢失
    #     print('1')
    #     return num_objects
    # else:
    num_of_label = [np.sum(labeled_array == (i + 1)) for i in range(num_objects)]
    top2index = list(map(num_of_label.index, heapq.nlargest(2, num_of_label)))

    mask1 = color_mask * (labeled_array == (top2index[0] + 1))
    center1 = ndimage.measurements.center_of_mass(mask1)
    mask2 = color_mask * (labeled_array == (top2index[1] + 1))
    center2 = ndimage.measurements.center_of_mass(mask2)

    if center1[1] > center2[1]:
        silhouette_l = mask1
        silhouette_r = mask2
    else:
        silhouette_l = mask2
        silhouette_r = mask1

    # roi_v = np.logical_and(coords_v > minv, coords_v < maxv)
    # silhouette_l = np.logical_and(np.logical_and(silhouette_l, roi_v), depth > 0)
    # silhouette_r = np.logical_and(np.logical_and(silhouette_r, roi_v), depth > 0)
    silhouette_l = np.logical_and(silhouette_l, depth > 0)
    silhouette_r = np.logical_and(silhouette_r, depth > 0)

    # left
    silhouette_l = silhouette_l.squeeze()
    lefthand_u = coords_u[silhouette_l]
    lefthand_v = coords_v[silhouette_l]
    lefthand_d = depth[silhouette_l]
    lefthand_uvd = np.vstack((lefthand_u, lefthand_v, lefthand_d)).T
    lefthand_uvd[:, :2] = lefthand_uvd[:, :2] * (lefthand_uvd[:, 2:] + 1e-9)
    pointcloud_l = np.dot(lefthand_uvd, np.linalg.inv(ks.T)) / 1000
    # right
    silhouette_r = silhouette_r.squeeze()
    righthand_u = coords_u[silhouette_r]
    righthand_v = coords_v[silhouette_r]
    righthand_d = depth[silhouette_r]
    righthand_uvd = np.vstack((righthand_u, righthand_v, righthand_d)).T
    righthand_uvd[:, :2] = righthand_uvd[:, :2] * (righthand_uvd[:, 2:] + 1e-9)
    pointcloud_r = np.dot(righthand_uvd, np.linalg.inv(ks.T)) / 1000

    return [silhouette_l.astype('int'), silhouette_r.astype('int')], [pointcloud_l, pointcloud_r]


def get_silhouettes_and_pointclouds_tracking(depth, color, ks, coords_u, coords_v, hand_joints, hrnet, 
                                             R=None, offset=7, idx=0):
    if len(depth.shape) == 3:
        depth = np.squeeze(depth)
    lefthand_joints = hand_joints[0, :, :]
    righthand_joints = hand_joints[1, :, :]
    lefthand_bbox = [min(lefthand_joints[:, 0]) - offset, min(lefthand_joints[:, 1]) - offset,
                     max(lefthand_joints[:, 0]) + offset, max(lefthand_joints[:, 1]) + offset]
    righthand_bbox = [min(righthand_joints[:, 0]) - offset, min(righthand_joints[:, 1]) - offset,
                      max(righthand_joints[:, 0]) + offset, max(righthand_joints[:, 1]) + offset]
    pred_pose = get_handpose(color, hrnet, [lefthand_bbox, righthand_bbox])

    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 43, 46])
    upper_green = np.array([100, 255, 255])
    roi_lefthand = np.logical_and(np.logical_and(coords_u > lefthand_bbox[0], coords_u < lefthand_bbox[2]),
                                  np.logical_and(coords_v > lefthand_bbox[1], coords_v < lefthand_bbox[3]))
    roi_righthand = np.logical_and(np.logical_and(coords_u > righthand_bbox[0], coords_u < righthand_bbox[2]),
                                   np.logical_and(coords_v > righthand_bbox[1], coords_v < righthand_bbox[3]))
    color_mask = cv2.inRange(hsv_color, lower_green, upper_green)

    silhouette_l = np.logical_and(np.logical_and(np.logical_not(color_mask), roi_lefthand), depth > 0)
    silhouette_r = np.logical_and(np.logical_and(np.logical_not(color_mask), roi_righthand), depth > 0)

    # left
    silhouette_l = silhouette_l.squeeze()
    lefthand_u = coords_u[silhouette_l]
    lefthand_v = coords_v[silhouette_l]
    lefthand_d = depth[silhouette_l]
    lefthand_uvd = np.vstack((lefthand_u, lefthand_v, lefthand_d)).T
    lefthand_uvd[:, :2] = lefthand_uvd[:, :2] * (lefthand_uvd[:, 2:] + 1e-9)
    pointcloud_l = np.dot(lefthand_uvd, np.linalg.inv(ks.T)) / 1000
    # right
    silhouette_r = silhouette_r.squeeze()
    righthand_u = coords_u[silhouette_r]
    righthand_v = coords_v[silhouette_r]
    righthand_d = depth[silhouette_r]
    righthand_uvd = np.vstack((righthand_u, righthand_v, righthand_d)).T
    righthand_uvd[:, :2] = righthand_uvd[:, :2] * (righthand_uvd[:, 2:] + 1e-9)
    pointcloud_r = np.dot(righthand_uvd, np.linalg.inv(ks.T)) / 1000

    if R is not None:
        pointcloud_l = np.dot(pointcloud_l, np.linalg.inv(R))
        pointcloud_r = np.dot(pointcloud_r, np.linalg.inv(R))

    # return [silhouette_l.astype('int'), silhouette_r.astype('int')], [pointcloud_l, pointcloud_r]
    return [np.vstack((lefthand_u, lefthand_v)).T, np.vstack((righthand_u, righthand_v)).T], [pointcloud_l, pointcloud_r], pred_pose


def get_handpose(color, hrnet, bboxes, pose_inputsize=224):
    scales = []
    top_lefts = []
    scale_ratios = []
    images = []
    centers = []
    for bbox in bboxes:
        center_u = (bbox[0] + bbox[2]) // 2
        center_v = (bbox[1] + bbox[3]) // 2
        centers.append(np.array([center_u, center_v], dtype='float'))
        cropped_img, top_left = crop_image_with_static_size(color, center_u, center_v, pose_inputsize)
        scales.append(np.array(cropped_img.shape[:2], dtype='int'))
        scale_ratio = np.array([cropped_img.shape[1] / pose_inputsize,
                                cropped_img.shape[0] / pose_inputsize])
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        cropped_img = (cropped_img / 255.0 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
        cropped_img = cv2.resize(cropped_img, (pose_inputsize, pose_inputsize))
        top_lefts.append(np.asarray(top_left, np.float32))
        scale_ratios.append(scale_ratio)
        images.append(cropped_img)
    hands = ['left', 'right']
    images[0] = np.fliplr(images[0]).transpose(2, 0, 1)
    images[1] = images[1].transpose(2, 0, 1)
    images = torch.from_numpy(np.array(images)).float().cuda()
    centers = torch.from_numpy(np.array(centers)).float().cuda()
    scales = torch.from_numpy(np.array(scales)).float().cuda()
    pred_pose = postprocess.get_output(hrnet, images, hands, top_lefts, scale_ratios)

    return pred_pose


def crop_image_with_static_size(img, center_x, center_y, size, value=(127, 127, 127)):
    center_x, center_y = int(round(center_x)), int(round(center_y))
    size = int(round(size))
    H = img.shape[0]
    W = img.shape[1]
    half_size = size // 2
    pad_left = max(0, half_size - center_x)
    pad_top = max(0, half_size - center_y)
    pad_right = max(0, center_x + half_size - W)
    pad_bottom = max(0, center_y + half_size - H)
    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=value)
    img_crop = img[center_y + pad_top - half_size:center_y + pad_top + half_size,
                   center_x + pad_left - half_size:center_x + pad_left + half_size, :]
    return img_crop, [center_x-half_size, center_y-half_size]


def visualisze_joint(hand_joints, color_img, idx):
    hand_joints = hand_joints.astype('int')
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
        for i in range(hand_joints.shape[0]):
            coord1 = hand_joints[i, connection[0], :]
            coord2 = hand_joints[i, connection[1], :]
            cv2.line(color_img, tuple(coord1), tuple(coord2), color=tuple(color.tolist()), thickness=2)

    cv2.imwrite("./.cache/pose/%05d.png" % idx, color_img)