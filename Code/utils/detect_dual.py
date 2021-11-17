import torch
import os
import numpy as np
import cv2
from .imghelper import crop_pad_img_from_bounding_rect, flip_img
from model.ssd import build_ssd
from torch.autograd import Variable
import sys
sys.path.append("..")


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


class HandDetector(object):

    def __init__(self, config):
        self.config = config.DETECTOR
        self.num_classes = 3  # background, lefthand, righthand
        self.net = build_ssd('test', size=self.config.INPUT_DIM, num_classes=self.num_classes)
        self.net.load_state_dict(torch.load(self.config.TRAINED_MODEL))
        self.net.eval()
        print("Finished loading hand detect model: {}".format(self.config.TRAINED_MODEL))
        if self.config.CUDA:
            self.device = 'cuda'
            self.net = self.net.cuda()
        else:
            self.device = 'cpu'
        self.confidence_threshold = self.config.THRESHOLD
        self.top_k_each = 1  # left hand + right hand

    def __call__(self, frame, keypoints2d=None, handsides=None):
        centers = []
        scales = []
        bbox = []
        images = []
        top_lefts = []
        scale_ratios = []
        hands = []

        if handsides is None:
            handsides = []
        detect_classes = []
        if 'left' not in handsides:
            detect_classes.append((1, 'left'))
        if 'right' not in handsides:
            detect_classes.append((2, 'right'))

        if len(detect_classes)>0:
            print("restart detection for", [c[1] for c in detect_classes])
            h, w, c = frame.shape
            input_size = (self.config.INPUT_DIM, self.config.INPUT_DIM)
            img_rz = cv2.resize(frame, input_size)
            x = (img_rz.astype(np.float32) / 255.0 - 0.5) * 2
            x = Variable(torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0))
            if self.config.CUDA:
                x = x.cuda()

            detections = self.net(x).data

            for j, side in detect_classes:
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(11, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 11)
                if dets.dim() == 0:
                    return None
                bboxes = dets[:, 1:5]
                bboxes[:, 0] *= w
                bboxes[:, 2] *= w
                bboxes[:, 1] *= h
                bboxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((bboxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32, copy=False)
                cls_dets = cls_dets[cls_dets[:, -1] >= self.confidence_threshold][:self.top_k_each]

                for det in cls_dets:
                    x1, y1, x2, y2 = list(map(lambda x: int(round(x)), det[:4]))
                    # score = float(det[-1])
                    center_u = (x1 + x2) // 2
                    center_v = (y1 + y2) // 2
                    centers.append(np.array([center_u, center_v], dtype='float'))
                    norm_size = max(abs(x2 - x1), abs(y2 - y1)) * self.config.EXPAND_RATIO
                    bb = [center_u - norm_size // 2,
                          center_v - norm_size // 2,
                          center_u + norm_size // 2,
                          center_v + norm_size // 2]
                    bb = np.array(bb, dtype='int')
                    if self.config.CROP_METHOD == 0:
                        cropped_img, bb = crop_pad_img_from_bounding_rect(frame, bb)
                        scales.append(np.array(cropped_img.shape[:2], dtype='int'))
                        scale_ratio = np.array([cropped_img.shape[1]/self.config.OUTPUT_DIM,
                                                cropped_img.shape[0]/self.config.OUTPUT_DIM])
                        cropped_img = cv2.resize(cropped_img, (self.config.OUTPUT_DIM, self.config.OUTPUT_DIM))
                        top_left = [center_u - norm_size // 2, center_v - norm_size // 2]
                        bbox.append(bb)
                    else:
                        cropped_img, top_left = crop_image_with_static_size(frame, center_u, center_v,
                                                                            norm_size)
                        scales.append(np.array(cropped_img.shape[:2], dtype='int'))
                        scale_ratio = np.array([cropped_img.shape[1]/self.config.OUTPUT_DIM,
                                                cropped_img.shape[0]/self.config.OUTPUT_DIM])
                        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                        cropped_img = (cropped_img / 255.0 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
                        cropped_img = cv2.resize(cropped_img, (self.config.OUTPUT_DIM, self.config.OUTPUT_DIM))

                    top_lefts.append(np.asarray(top_left, np.float32))
                    scale_ratios.append(scale_ratio)
                    images.append(cropped_img)
                    hands.append(side)

        if keypoints2d is not None and len(keypoints2d)>0:
            # Track by keypoints estimated by previous frame
            for i in range(keypoints2d.shape[0]):
                k2d = keypoints2d[i]  # 21 x 2
                minu, maxu = np.min(k2d[:, 0]), np.max(k2d[:, 0])
                minv, maxv = np.min(k2d[:, 1]), np.max(k2d[:, 1])
                center_u = (minu + maxu) // 2
                center_v = (minv + maxv) // 2
                centers.append(np.array([center_u, center_v], dtype='float'))
                norm_size = max((maxu - minu), (maxv - minv)) * self.config.EXPAND_RATIO
                bb = [center_u - norm_size // 2,
                      center_v - norm_size // 2,
                      center_u + norm_size // 2,
                      center_v + norm_size // 2]
                bb = np.array(bb, dtype='int')
                cropped_img, top_left = crop_image_with_static_size(frame, center_u, center_v, norm_size)
                scales.append(np.array(cropped_img.shape[:2], dtype='int'))
                scale_ratio = np.array([cropped_img.shape[1] / self.config.OUTPUT_DIM,
                                        cropped_img.shape[0] / self.config.OUTPUT_DIM])
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                cropped_img = (cropped_img / 255.0 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
                cropped_img = cv2.resize(cropped_img, (self.config.OUTPUT_DIM, self.config.OUTPUT_DIM))
                top_lefts.append(np.asarray(top_left, np.float32))
                scale_ratios.append(scale_ratio)
                images.append(cropped_img)
                bbox.append(bb)
            hands.extend(handsides)
        try:
            # do flip to left hand
            for idx in range(len(hands)):
                if hands[idx]=='left':
                    images[idx] = flip_img(images[idx])

            for i in range(len(images)):
                images[i] = np.expand_dims(images[i].transpose(2, 0, 1), 0)
                centers[i] = np.expand_dims(centers[i], 0)
                scales[i] = np.expand_dims(scales[i], 0)
                # bbox[i] = np.expand_dims(bbox[i], 0)
            images = torch.from_numpy(np.concatenate(images, 0)).float().to(self.device)
            centers = torch.from_numpy(np.concatenate(centers, 0)).float().to(self.device)
            scales = torch.from_numpy(np.concatenate(scales, 0)).float().to(self.device)
            # bbox = torch.from_numpy(np.concatenate(bbox, 0)).float().to(self.device)
        except Exception as e:
            print(e)
        finally:
            return {"images": images, "centers": centers,
                    "scales": scales, "bbox": bbox,
                    "hands": hands, # "dets": cls_dets,
                    "top_lefts": top_lefts, "scale_ratios": scale_ratios}

    def draw_detections(self, img, dets):
        for det in dets:
            x1, y1, x2, y2 = list(map(lambda x: int(round(x)), det[:4]))
            score = float(det[-1])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "%.4f" % score, (x1, y1+10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255))
        return img


if __name__ == "__main__":
    from config import cfg_demo
    import json
    cfg_demo.merge_from_file("../cfg_files/demo.yaml")
    cfg_demo.freeze()
    hand_detector = HandDetector(cfg_demo)
    root = "/HDD/jimmydai/dataset/Realsense-image"
    output_root = "../detection_result"
    ann_file = os.path.join(root, "annotation2d.json")
    with open(ann_file, 'r') as f:
        data_dict = json.load(f)
        for imgname, _ in data_dict.items():
            img_file = os.path.join(root, imgname)
            img = cv2.imread(img_file)
            _, _, _, _, _, dets = hand_detector(img)
            img = hand_detector.draw_detections(img, dets)
            output_file = os.path.join(output_root, imgname)
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            cv2.imwrite(output_file, img)
