import sys

sys.path.append('..')

import time
import os

import numpy as np
import cv2

# 导入 TensorRT
import tensorrt as trt

# 导入 TensorRT 封装好的库
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '. '))
import common

INPUT_HEIGHT = 256
INPUT_WIDTH = 192
NUM_JOINTS = 17
STRIDE_SIZE = 4
TRT_LOGGER = trt.Logger()


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
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
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


# 用于加载 TensorRT 模型
def load_engine(trt_path):
    with open(trt_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

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

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
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

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

class HRNet:
    def __init__(self, model_path):
        # 模型初始化
        self.engine = load_engine(model_path)
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()
        self.expand_radio = 1.5

    def preprocess(self, img_path, bboxes=None):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs = []
        top_left_coords = []
        scale_ratios = []
        if bboxes is not None:
            for bbox in bboxes:
                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

                center_x = (xmin + xmax) // 2
                center_y = (ymin + ymax) // 2
                center = np.asarray([center_x, center_y])
                norm_size = max(abs(xmax - xmin), abs(ymax - ymin)) * self.expand_radio

                cropped_image, top_left = crop_image_with_static_size(img, center_x, center_y, norm_size)

                cropped_image = (cropped_image / 255.0 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
                scale = np.asarray(cropped_image.shape[:2], dtype='int')
                scale_ratio = np.array([cropped_image.shape[1] / INPUT_WIDTH,
                                            cropped_image.shape[0] / INPUT_HEIGHT])

                cropped_image = cv2.resize(cropped_image, (INPUT_WIDTH, INPUT_HEIGHT))
                cropped_image = np.expand_dims(cropped_image.transpose(2, 0, 1), 0)
                imgs.append(cropped_image)
                top_left_coords.append(top_left)
                scale_ratios.append(scale_ratio)
        return imgs, top_left_coords, scale_ratios

    def inference(self, img):
        img = img.flatten()
        np.copyto(self.inputs[0].host, img)

        start_time = time.time()
        results = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        print('HRNet time: ', str((time.time() - start_time) * 1000), 'ms')

        return results
    
    def postprocess(self, results):
        res = results[0]
        res = res.reshape((1, 17, INPUT_HEIGHT // STRIDE_SIZE, INPUT_WIDTH // STRIDE_SIZE))
        preds, maxvals = get_max_preds(res)
        return preds, maxvals

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


    def forward(self, img_path, bbox=None):
        bbox_imgs, top_left_coords, scale_ratios = self.preprocess(img_path, bbox)
        raw_image = cv2.imread(img_path)
        filename = os.path.split(img_path)[-1]
        save_path = os.path.join('/workspace/HumanPoseEstimation/outputs/', filename)
        for i, img in enumerate(bbox_imgs):
            results = self.inference(img)
            preds, maxvals = self.postprocess(results)
            preds[:, :, 0] *= STRIDE_SIZE
            preds[:, :, 1] *= STRIDE_SIZE
            
            top_left = top_left_coords[i]
            final_pose = top_left + preds * scale_ratios[i]
            visualized_img = self.visualization(raw_image, final_pose)
            # for i in range(NUM_JOINTS):
            #     cv2.circle(raw_image, (int(final_pose[0][i][0]), int(final_pose[0][i][1])), 1, [0, 0, 255], 1)
            cv2.imwrite(save_path, visualized_img)

    def warmup(self):
        print('-----warm up start-----')
        img = np.zeros([INPUT_HEIGHT, INPUT_WIDTH, 3], dtype=np.uint8)
        for _ in range(10):
            self.inference(img)
        print('-----warm up end-----')

if __name__ == '__main__':

    model = HRNet('/workspace/HumanPoseEstimation/models/hrnet/hrnet_human_pose_estimation_w32.trt')
    model.warmup()
    model.forward('/workspace/HumanPoseEstimation/imgs/bus.jpg')
