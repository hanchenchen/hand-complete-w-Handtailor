import cv2
import heapq
import numpy as np
import torch
from .prepare_solve import Solver
from collections import OrderedDict
from torch.autograd import Variable
# from utils.util import write_ply
from scipy import ndimage
from .ssd import build_ssd
from .hrnet import inference, postprocess


# Using Multiprocessing to do this
def Worker(input_queue, output_queue, proc_id,
           height, width,
           detect_threshold, detect_inputsize, pose_inputsize,
           verbose, detectmodel_path, step_size, num_iters,
           threshold, lefthand, righthand, w_silhouette,
           w_pointcloud, w_poseprior, w_shapeprior, w_reprojection,
           use_pcaprior=True, init_params=None, usage="getshape", all_pose=False, method=0):
    # Load Detect Model
    detect_net = build_ssd('test', size=300, num_classes=2)
    detect_net.load_state_dict(torch.load(detectmodel_path))
    detect_net.eval()
    detect_net = detect_net.cuda()
    # Solver object
    solver = Solver(step_size=step_size,
                    num_iters=num_iters,
                    threshold=threshold,
                    w_poseprior=w_poseprior,
                    w_shapeprior=w_shapeprior,
                    w_pointcloud=w_pointcloud,
                    w_reprojection=w_reprojection,
                    w_silhouette=w_silhouette,
                    lefthand=lefthand,
                    righthand=righthand,
                    verbose=verbose,
                    fit_camera=True,
                    use_pcaprior=use_pcaprior)

    # Get fit target
    meta = input_queue.get()
    color, depth, Ks = meta['color'][0], meta['depth'][0], meta['ks'][0]
    # color, depth, Ks = inputs['color'][0], inputs['depth'][0], inputs['ks'][0]
    v = np.linspace(0, height - 1, height)
    u = np.linspace(0, width - 1, width)
    coords_u, coords_v = np.meshgrid(u, v)

    outputs = get_handpose(color, detect_inputsize, detect_net, detect_threshold, pose_inputsize)
    if len(outputs) == 1:
        # 当检测到的手少于两个时报错
        output_queue.put(outputs)
    else:
        hand_joints, hand_sides = outputs
        silhouettes, pointclouds = get_silhouettes_and_pointclouds(depth, color, hand_joints, Ks, coords_u, coords_v)
        visualisze_joint(hand_joints, color, depth)
        # make target
        fit_target = OrderedDict()
        fit_target['silhouette_l'] = silhouettes[0]
        fit_target['silhouette_r'] = silhouettes[1]
        fit_target['pointcloud_l'] = pointclouds[0]
        fit_target['pointcloud_r'] = pointclouds[1]
        fit_target['hand_joints_l'] = hand_joints[hand_sides.index("left")]
        fit_target['hand_joints_r'] = hand_joints[hand_sides.index("right")]
        # learnable parameters
        if method == 0:
            learnable_params = get_init_param(init_params, solver.device, usage, all_pose)
        else:
            learnable_params = get_init_param_v2(init_params, solver.device, usage)
        # Do Model fitting
        opt = solver(learnable_params, fit_target, Ks, method=method)
        # Put results to output_queue
        hand_joints = np.stack((fit_target['hand_joints_l'], fit_target['hand_joints_r']), 0)
        output = {
            "opt_params": opt["opt_params"].astype('float32'),
            "vertices": opt["vertices"].astype('float32'),
            "extra_verts": opt["extra_verts"].astype('float32'),
            "hand_joints": hand_joints}
        output_queue.put(output)


def get_handpose(color, detect_inputsize, detect_net, detect_threshold, pose_inputsize):
    h, w, c = color.shape
    # Resize color image
    color_rz = cv2.resize(color, (detect_inputsize, detect_inputsize))
    norm_color = (color_rz.astype(np.float32) / 255.0 - 0.5) * 2
    batch_norm_color = Variable(torch.from_numpy(norm_color).permute(2, 0, 1).unsqueeze(0)).cuda()
    detections = detect_net(batch_norm_color).data
    dets = detections[0, 1, :]
    mask = dets[:, 0].gt(0.).expand(11, dets.size(0)).t()
    dets = torch.masked_select(dets, mask).view(-1, 11)
    bboxes = dets[:, 1:5]
    bboxes[:, 0] *= w
    bboxes[:, 2] *= w
    bboxes[:, 1] *= h
    bboxes[:, 3] *= h
    scores = dets[:, 0].cpu().numpy()
    cls_dets = np.hstack((bboxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32, copy=False)
    cls_dets = cls_dets[cls_dets[:, -1] >= detect_threshold][:2]
    # Crop hand area by detection result
    centers = []
    scales = []
    top_lefts = []
    scale_ratios = []
    images = []
    for det in cls_dets:
        x1, y1, x2, y2 = list(map(lambda x: int(round(x)), det[:4]))
        center_u = (x1 + x2) // 2
        center_v = (y1 + y2) // 2
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

    if len(images) == 2:
        if centers[0][0] > centers[1][0]:
            images[0] = np.fliplr(images[0])
            hands = ['left', 'right']
        else:
            images[1] = np.fliplr(images[1])
            hands = ['right', 'left']
        for i in range(len(images)):
            images[i] = np.expand_dims(images[i].transpose(2, 0, 1), 0)
            centers[i] = np.expand_dims(centers[i], 0)
            scales[i] = np.expand_dims(scales[i], 0)
        images = torch.from_numpy(np.concatenate(images, 0)).float().cuda()
        centers = torch.from_numpy(np.concatenate(centers, 0)).float().cuda()
        scales = torch.from_numpy(np.concatenate(scales, 0)).float().cuda()
        # Estimate joint locations
        pred_pose = postprocess.get_output(inference.get_hrnet(), images, hands, top_lefts, scale_ratios)

        return pred_pose, hands
    else:
        return [len(images)]


def get_init_param_v2(init_params, device, usage="getshape"):
    if init_params is not None:
        init_pose_l = torch.from_numpy(init_params[0, [1, 10, 19, 28, 37]]).float().to(device).unsqueeze(0)
        init_pose_r = torch.from_numpy(init_params[1, [1, 10, 19, 28, 37]]).float().to(device).unsqueeze(0)
        init_shape_l = torch.from_numpy(init_params[0, 45:55]).float().to(device).unsqueeze(0)
        init_shape_r = torch.from_numpy(init_params[1, 45:55]).float().to(device).unsqueeze(0)
        init_quat_l = torch.from_numpy(init_params[0, [55, 57]]).float().to(device).unsqueeze(0)
        init_quat_r = torch.from_numpy(init_params[1, [55, 57]]).float().to(device).unsqueeze(0)
        init_trans_l = torch.from_numpy(init_params[0, [59]]).float().to(device).unsqueeze(0)
        init_trans_r = torch.from_numpy(init_params[1, [59]]).float().to(device).unsqueeze(0)
        if init_params.shape[-1] > 62:
            glb_quat = torch.from_numpy(init_params[0, 62:66]).float().to(device).unsqueeze(0)
            glb_trans = torch.from_numpy(init_params[0, 66:]).float().to(device).unsqueeze(0)
        else:
            glb_quat = torch.zeros((1, 4)).to(device)
            glb_quat[:, 0] = 1.0
            glb_quat[:, 1] = 1.0
            glb_trans = torch.zeros((1, 3)).to(device)
    else:
        init_pose_l = torch.zeros((1, 5)).to(device)
        init_pose_r = torch.zeros((1, 5)).to(device)
        init_shape_l = torch.zeros((1, 10)).to(device)
        init_shape_r = torch.zeros((1, 10)).to(device)
        init_quat_l = torch.ones((1, 2)).to(device)
        init_quat_r = torch.ones((1, 2)).to(device)
        glb_quat = torch.zeros((1, 4)).to(device)
        glb_quat[:, 0] = 1.0
        glb_quat[:, 1] = 1.0
        init_trans_l = torch.zeros((1, 1)).to(device)
        init_trans_r = torch.zeros((1, 1)).to(device)
        glb_trans = torch.zeros((1, 3)).to(device)
    
    # Add gradient
    param_list = [init_pose_l, init_shape_l, init_quat_l, init_trans_l,
                  init_pose_r, init_shape_r, init_quat_r, init_trans_r,
                  glb_quat, glb_trans]
    for i in range(len(param_list)):
        param_list[i].requires_grad = True
    return param_list


def get_init_param(init_params, device, usage="getshape", all_pose=True):
    if init_params is not None:
        if not all_pose:
            init_pose_l = torch.from_numpy(init_params[0, [1, 10, 19, 28, 37]]).float().to(device).unsqueeze(0)
            init_pose_r = torch.from_numpy(init_params[1, [1, 10, 19, 28, 37]]).float().to(device).unsqueeze(0)
        else:
            init_pose_l = torch.from_numpy(init_params[0, :45]).float().to(device).unsqueeze(0)
            init_pose_r = torch.from_numpy(init_params[1, :45]).float().to(device).unsqueeze(0)
        init_shape_l = torch.from_numpy(init_params[0, 45:55]).float().to(device).unsqueeze(0)
        init_shape_r = torch.from_numpy(init_params[1, 45:55]).float().to(device).unsqueeze(0)
        init_quat_l = torch.from_numpy(init_params[0, 55:59]).float().to(device).unsqueeze(0)
        init_quat_r = torch.from_numpy(init_params[1, 55:59]).float().to(device).unsqueeze(0)
        init_trans_l = torch.from_numpy(init_params[0, 59:]).float().to(device).unsqueeze(0)
        init_trans_r = torch.from_numpy(init_params[1, 59:]).float().to(device).unsqueeze(0)
    else:
        if not all_pose:
            init_pose_l = torch.zeros((1, 5)).to(device)
            init_pose_r = torch.zeros((1, 5)).to(device)
        else:
            init_pose_l = torch.zeros((1, 45)).to(device)
            init_pose_r = torch.zeros((1, 45)).to(device)
        init_shape_l = torch.zeros((1, 10)).to(device)
        init_shape_r = torch.zeros((1, 10)).to(device)
        init_trans_l = torch.zeros((1, 3)).to(device)
        init_trans_r = torch.zeros((1, 3)).to(device)
        init_quat_l = torch.zeros((1, 4)).to(device)
        init_quat_l[:, 0] = 1.0
        init_quat_l[:, 1] = 1.0
        init_quat_r = torch.zeros((1, 4)).to(device)
        init_quat_r[:, 0] = 1.0
        init_quat_r[:, 1] = 1.0

    # Add gradient
    param_list = [init_pose_l, init_shape_l, init_quat_l, init_trans_l,
                  init_pose_r, init_shape_r, init_quat_r, init_trans_r]
    param_list[0].requires_grad = True
    param_list[4].requires_grad = True
    param_list[2].requires_grad = True
    param_list[6].requires_grad = True
    param_list[3].requires_grad = True
    param_list[7].requires_grad = True
    # if usage == "getshape":
    param_list[1].requires_grad = True
    param_list[5].requires_grad = True

    return param_list

# def get_init_param(init_params, device, learnable=[True, True, True, True]):
#     # Initialization
#     if init_params is None:
#         init_pose_l = torch.zeros((1, 45)).to(device)
#         init_pose_r = torch.zeros((1, 45)).to(device)
#         init_quat_l = torch.zeros((1, 4)).to(device)
#         init_quat_l[:, 0] = 1.0
#         init_shape_l = torch.zeros((1, 10)).to(device)
#         init_cam_l = torch.zeros((1, 3)).to(device)
#         init_cam_l[:, 2] = 1.0
#         init_quat_r = torch.zeros((1, 4)).to(device)
#         init_quat_r[:, 0] = 1.0
#         init_shape_r = torch.zeros((1, 10)).to(device)
#         init_cam_r = torch.zeros((1, 3)).to(device)
#         init_cam_r[:, 2] = 1.0
#     else:
#         init_params = init_params.reshape(2, -1)
#         init_pose_l = torch.from_numpy(init_params[0, :45]).float().to(device).unsqueeze(0)
#         init_pose_r = torch.from_numpy(init_params[1, :45]).float().to(device).unsqueeze(0)
#         init_shape_l = torch.from_numpy(init_params[0, 45:55]).float().to(device).unsqueeze(0)
#         init_shape_r = torch.from_numpy(init_params[1, 45:55]).float().to(device).unsqueeze(0)
#         init_quat_l = torch.from_numpy(init_params[0, 55:59]).float().to(device).unsqueeze(0)
#         init_quat_r = torch.from_numpy(init_params[1, 55:59]).float().to(device).unsqueeze(0)
#         init_cam_l = torch.from_numpy(init_params[0, 59:]).float().to(device).unsqueeze(0)
#         init_cam_r = torch.from_numpy(init_params[1, 59:]).float().to(device).unsqueeze(0)
#     # Add gradient
#     param_list = [init_pose_l, init_shape_l, init_quat_l, init_cam_l,
#                   init_pose_r, init_shape_r, init_quat_r, init_cam_r]
#     for i in range(len(param_list)):
#         param_list[i].requires_grad = learnable[i % 4]
#     return param_list


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


def get_silhouettes_and_pointclouds(depth, color, hand_joints, ks, coords_u, coords_v):
    if len(depth.shape) == 3:
        depth = np.squeeze(depth)
    # middle_u = (hand_joints[0, 0, 0] + hand_joints[1, 0, 0]) / 2
    minv = min(hand_joints[0, 0, 1], hand_joints[1, 0, 1])
    maxv = np.max(hand_joints[:, :, 1])

    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    # r = color[:, :, 2]
    # g = color[:, :, 1]
    # b = color[:, :, 0]
    # color_mask = np.logical_and(np.logical_and(r > 115, b > 40), g > 40)
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([100, 255, 255])
    roi_v = np.logical_and(coords_v > minv, coords_v < maxv)
    color_mask = cv2.inRange(hsv_color, lower_green, upper_green)
    color_mask = np.logical_and(np.logical_not(color_mask), roi_v)

    labeled_array, num_objects = ndimage.measurements.label(color_mask > 0)

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
    lefthand_u = coords_u[silhouette_l]
    lefthand_v = coords_v[silhouette_l]
    lefthand_d = depth[silhouette_l]
    lefthand_uvd = np.vstack((lefthand_u, lefthand_v, lefthand_d)).T
    lefthand_uvd[:, :2] = lefthand_uvd[:, :2] * (lefthand_uvd[:, 2:] + 1e-9)
    pointcloud_l = np.dot(lefthand_uvd, np.linalg.inv(ks.T)) / 1000
    # write_ply("./.cache/prepare_lefthand.ply", pointcloud_l)
    # right
    righthand_u = coords_u[silhouette_r]
    righthand_v = coords_v[silhouette_r]
    righthand_d = depth[silhouette_r]
    righthand_uvd = np.vstack((righthand_u, righthand_v, righthand_d)).T
    righthand_uvd[:, :2] = righthand_uvd[:, :2] * (righthand_uvd[:, 2:] + 1e-9)
    pointcloud_r = np.dot(righthand_uvd, np.linalg.inv(ks.T)) / 1000
    # write_ply("./.cache/prepare_righthand.ply", pointcloud_r)

    return [silhouette_l.astype('int'), silhouette_r.astype('int')], [pointcloud_l, pointcloud_r]


def visualisze_joint(hand_joints, color_img, depth_img):
    cv2.imwrite("./.cache/org_color.png", color_img)
    hand_joints = hand_joints.astype('int')
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
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
            cv2.line(depth_colormap, tuple(coord1), tuple(coord2), color=tuple(color.tolist()), thickness=2)

    cv2.imwrite("./.cache/prepare_color.png", color_img)
    cv2.imwrite("./.cache/prepare_depth.png", depth_colormap)


def pixel2world(xyz, K, eps=1e-9):
    if len(xyz.shape) == 1:
        xyz = np.expand_dims(xyz, 0)
    xyz[:, :2] = xyz[:, :2] * (xyz[:, 2:] + eps)
    xyz = np.dot(xyz, np.linalg.inv(K.T))
    xyz = np.squeeze(xyz, 0)
    return xyz
