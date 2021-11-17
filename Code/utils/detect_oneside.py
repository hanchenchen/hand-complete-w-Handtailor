import torch
import numpy as np
import cv2
import sys
sys.path.append("..")
from model.ssd import build_ssd
from torch.autograd import Variable




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


def flip_img(img):
    """
    Flip images or masks.
    Channels come last, e.g. (256, 256, 3)
    """
    img = np.fliplr(img)
    return img


class HandDetector(object):

    def __init__(self, config, handside='right', top_k=4):
        self.config = config.DETECTOR
        num_classes = 2  # hand + not hand
        self.net = build_ssd('test', size=self.config.INPUT_DIM, num_classes=num_classes)
        self.net.load_state_dict(torch.load(self.config.TRAINED_MODEL))
        self.net.eval()
        print("Finished loading hand detect model: {}".format(self.config.TRAINED_MODEL))
        if self.config.CUDA:
            self.device = 'cuda'
            self.net = self.net.cuda()
        else:
            self.device = 'cpu'
        self.confidence_threshold = self.config.THRESHOLD
        self.top_k = top_k
        self.handside = handside

    def get_state(self, pred_pose, maxvals, sides):
        lost_threshold = self.config.LOST_THRESH
        previous_pose = pred_pose[maxvals >= lost_threshold].copy()
        previous_sides = [side for side, maxval in zip(sides, maxvals) if maxval >= lost_threshold]
        return {"poses": previous_pose, "sides": previous_sides}

    def __call__(self, frame, previous_state=None):
        centers = []
        scales = []
        boxs = []
        images = []
        top_lefts = []
        scale_ratios = []
        if previous_state is None or len(previous_state["poses"])==0:
            # print("restart detection")
            h, w, c = frame.shape
            input_size = (self.config.INPUT_DIM, self.config.INPUT_DIM)
            img_rz = cv2.resize(frame, input_size)
            x = (img_rz.astype(np.float32) / 255.0 - 0.5) * 2
            x = Variable(torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0))
            if self.config.CUDA:
                x = x.cuda()

            detections = self.net(x).data

            dets = detections[0, 1, :]
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
            cls_dets = cls_dets[cls_dets[:, -1] >= self.confidence_threshold][:self.top_k]

            for det in cls_dets:
                x1, y1, x2, y2 = list(map(lambda x: int(round(x)), det[:4]))
                boxs.append([x1, y1, x2, y2])

        else:
            # Track by keypoints estimated by previous frame
            keypoints2d = previous_state["poses"]
            for i in range(keypoints2d.shape[0]):
                k2d = keypoints2d[i]  # 21 x 2
                minu, maxu = np.min(k2d[:, 0]), np.max(k2d[:, 0])
                minv, maxv = np.min(k2d[:, 1]), np.max(k2d[:, 1])
                boxs.append([minu, minv, maxu, maxv])

        for box in boxs:
            x1,y1,x2,y2 = box

            center_u = (x1 + x2) // 2
            center_v = (y1 + y2) // 2
            centers.append(np.array([center_u, center_v], dtype='float'))
            norm_size = max(abs(x2 - x1), abs(y2 - y1)) * self.config.EXPAND_RATIO

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

        # do flip to left hand
        if self.handside=='left':
            for idx in range(len(images)):
                images[idx] = flip_img(images[idx])

        for i in range(len(images)):
            images[i] = np.expand_dims(images[i].transpose(2, 0, 1), 0)

        if len(images)>0:
            images = torch.from_numpy(np.concatenate(images, 0)).float().to(self.device)

        return {"images": images,
                "centers": centers, "scales": scales,
                "hands": [self.handside for _ in images],  # only support one handside
                "top_lefts": top_lefts, "scale_ratios": scale_ratios}

    # def draw_detections(self, img, dets):
    #     for det in dets:
    #         x1, y1, x2, y2 = list(map(lambda x: int(round(x)), det[:4]))
    #         score = float(det[-1])
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(img, "%.4f" % score, (x1, y1+10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255))
    #     return img


if __name__ == "__main__":
    pass
