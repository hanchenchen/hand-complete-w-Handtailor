import cv2
import numpy as np
import torch


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


def Worker(inputs_queue, output_queue, proc_id, hand_joints, pose_inputsize):
    from .hrnet import inference, postprocess
    hrnet = inference.get_hrnet()
    while True:
        if not inputs_queue.empty():
            meta = inputs_queue.get()
            if meta == 'STOP':
                print("Quit Pose Estimate Process.")
                break
            else:
                color = meta['color'][0]
                hand_joints = get_handpose(color, pose_inputsize, hand_joints, hrnet, postprocess)
                output = {"hand_joints": hand_joints}
                output_queue.put(output)


def get_handpose(color, pose_inputsize, hand_joints, hrnet, postprocess):
    h, w, c = color.shape
    centers = []
    scales = []
    top_lefts = []
    scale_ratios = []
    images = []
    for i in range(len(hand_joints)):
        x1, y1 = min(hand_joints[i, :, 0]), min(hand_joints[i, :, 1])
        x2, y2 = max(hand_joints[i, :, 0]), max(hand_joints[i, :, 1])
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
    images[0] = np.fliplr(images[0])
    hands = ['left', 'right']
    for i in range(len(images)):
        images[i] = np.expand_dims(images[i].transpose(2, 0, 1), 0)
        centers[i] = np.expand_dims(centers[i], 0)
        scales[i] = np.expand_dims(scales[i], 0)
    images = torch.from_numpy(np.concatenate(images, 0)).float().cuda()
    centers = torch.from_numpy(np.concatenate(centers, 0)).float().cuda()
    scales = torch.from_numpy(np.concatenate(scales, 0)).float().cuda()

    pred_pose = postprocess.get_output(hrnet, images, hands, top_lefts, scale_ratios)

    return pred_pose
