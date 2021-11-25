from VideoTemporalLifter.core import config
from VideoTemporalLifter.model.hrnet.pose_hrnet import PoseHighResolutionNet
from VideoTemporalLifter.utils.inference_util import crop_image_with_static_size, get_max_preds, flip_img, plot_hand
 
import sys
import torch
import numpy as np
import cv2
import json
import os
from tqdm import tqdm

class HandKeypointEstimator():
    def __init__(self, cfg):
        #模型加载代码
        self.cfg = config.read_config(cfg.CONFIG_PATH)
        self.model = PoseHighResolutionNet(self.cfg)
        
        model_path = cfg.MODEL_PATH
        state = torch.load(model_path)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()
        print("Finished loading hand pose model: {}".format(model_path))

        #cuda or cpu 
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.model = self.model.cuda()
        else:
            self.device = 'cpu'
        
        # #show model construct
        # summary(self.model, input_size = (3,224,224))

    def preprocess(self, image, hand_side = "right"):

        #图像前处理代码
        cropped_image = image

        scale = np.asarray(cropped_image.shape[:2], dtype='int')
        scale_ratio = np.array([cropped_image.shape[1] / self.cfg.MODEL.INPUT_WIDTH,
                                    cropped_image.shape[0] / self.cfg.MODEL.INPUT_HEIGHT])

        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = (cropped_image / 255.0 - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
        cropped_image = cv2.resize(cropped_image, (self.cfg.MODEL.INPUT_HEIGHT, self.cfg.MODEL.INPUT_WIDTH))
        
        #TODO: right or left hand?
        if hand_side == 'left':
            cropped_image = flip_img(cropped_image).copy()

        # cropped_image = np.expand_dims(cropped_image.transpose(2, 0, 1), 0)
        cropped_image = cropped_image.transpose(2, 0, 1)
        # cropped_image = torch.from_numpy(cropped_image).float().to(self.device)

        return cropped_image, scale, scale_ratio


    def forward(self, ori_image_list, hand_side_list):
        #模型推理代码，输入为图像，输出即为手部关键点

        #HRNet中heatmap的尺寸为原图的1/4, stride_size = 4
        stride_size = 4

        #图像预处理
        cropped_image_list = []
        scale_ratio_list = []
        for i in range(len(ori_image_list)):
            ori_image = ori_image_list[i]
            hand_side = hand_side_list[i]
            cropped_image, scale, scale_ratio = self.preprocess(ori_image, hand_side)
            cropped_image_list.append(cropped_image)
            scale_ratio_list.append(scale_ratio)
        
        
        cropped_image_list = np.asarray(cropped_image_list)
        scale_ratio_list = np.asarray(scale_ratio_list)
        cropped_image_list = torch.from_numpy(cropped_image_list).float().to(self.device)

        heatmap = self.model(cropped_image_list)
        heatmap_np = heatmap.detach().cpu().numpy() 
        # print("heatmap_np.shape = {}".format(heatmap_np.shape))

        predict_pose, maxvals = get_max_preds(heatmap_np, hand_side_list)
        predict_pose[:, :, 0] *= stride_size
        predict_pose[:, :, 1] *= stride_size
        # print("predict_pose:{}".format(predict_pose.shape))

        final_poses = []
        for i in range(predict_pose.shape[0]):

            final_pose = predict_pose[i] * scale_ratio_list[i]
            final_poses.append(final_pose)
        final_poses = np.asarray(final_poses)
        score = np.max(maxvals)

        # print("final_poses.shape: {}".format(final_poses.shape))
        # print("maxvals.shape:{}".format(maxvals.shape))

        final_poses_with_score = np.stack((final_poses[:,:,0], final_poses[:,:,1], maxvals[:,:,0]), axis = 2)
        # print("final_poses_with_score.shape = {}".format(final_poses_with_score.shape))
        # print(final_poses_with_score)

        return final_poses, score, final_poses_with_score

        
    def plot_hand(self, coords_xy, img):
        #可视化代码
        plot_hand(coords_xy, img)



if __name__ == '__main__':


    detect_res = {}
    with open('./detect_result.json', 'r') as jsonfile:

        detect_res = json.load(jsonfile)
    # #测试代码
    # img_root = "/HDD_sdc/HandDataset/dataset_gather/"
    # with open ("./self_inference.json", 'r' ) as jsonfile:
    #     anno_dic = json.load(jsonfile)

    #TODO:从配置文件和模型路径初始化检测器
    keypoint_detector = HandKeypointEstimator(cfg_file, model_path)

    for key, value in tqdm(detect_res.items()):
        file = '/HDD_sdc/HW_Hand/'+key.split('/')[3]+'/'+key.split('/')[4]+'/'+key.split('/')[5]+'/'+key.split('/')[6]+'/'+key.split('/')[7]
        try:
            bbox = detect_res[key]['bbox']
            center = [bbox[0], bbox[1]]
            scale = max(bbox[2],bbox[3])

            xmin = int(center[0] - scale / 2)
            xmax = int(center[0] + scale / 2)
            ymin = int(center[1] - scale / 2)
            ymax = int(center[1] + scale / 2)

            #TODO: 获得image 和 bbox
            img = cv2.imread(file)
            bbox = [xmin, ymin, xmax, ymax]

            #检测
            handpose, score = keypoint_detector.forward(img, bbox)

            #可视化
            keypoint_detector.plot_hand(handpose[0], img)

            #keshihua
            scale = scale*1.4

            xmin = int(center[0] - scale / 2)
            xmax = int(center[0] + scale / 2)
            ymin = int(center[1] - scale / 2)
            ymax = int(center[1] + scale / 2)

            cv2.rectangle(img,(xmin,ymin),(xmax, ymax),(0,255,0),2)

            right = detect_res[key]['right']
            center = [right[0], right[1]]
            scale = right[2]*2

            xmin = int(center[0] - scale / 2)
            xmax = int(center[0] + scale / 2)
            ymin = int(center[1] - scale / 2)
            ymax = int(center[1] + scale / 2)

            cv2.rectangle(img,(xmin,ymin),(xmax, ymax),(0,0,255),2)
            
            
            save_path = os.path.join('./preds',key.split('/')[-1])
            cv2.imwrite(save_path, img)
        except:
            continue



    # img = cv2.imread('samples/test_image.jpg')
    # keypoint_detector = HandKeypointEstimator(cfg_file, model_path)
    # bbox = [xmin, ymin, xmax, ymax]   # set the bbox of the hand
    # handpose, score = keypoint_detector.forward(img, bbox)
    # keypoint_detector.plot_hand(handpose[0], img)
    # cv2.imwrite("output.jpg", img)
