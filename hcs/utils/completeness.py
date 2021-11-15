"""
完成度计算函数，包含两个功能
1.不同手势的完成度计算不同
"""
import numpy as np


class Completeness(object):

    # def __init__(self, left_comp, right_comp, gesture,
    #              wrist_rot_pitch=60,
    #              wrist_rot_yaw=45):
    #     self.gesture = gesture
    #     self.left_comp = left_comp
    #     self.right_comp = right_comp

    #     # standard pose parameters
    #     self.wrist_rot_pitch = wrist_rot_pitch / 180 * np.pi  # 腕背伸和腕前屈
    #     self.wrist_rot_yaw = wrist_rot_yaw / 180 * np.pi  # 尺偏和桡偏
    #     self.pose_angle = np.ones(8) * np.pi / 2  # 握拳
    def __init__(self, gesture, wrist_rot_pitch=60, wrist_rot_yaw=45, complete_method=0):
        self.gesture = gesture
        # self.init_params = init_params
        # standard pose parameters
        self.wrist_rot_pitch = wrist_rot_pitch / 180 * np.pi  # 腕背伸和腕前屈
        self.wrist_rot_yaw = wrist_rot_yaw / 180 * np.pi  # 尺偏和桡偏
        self.pose_angle = np.ones(8) * np.pi / 2  # 握拳
        self.complete_method = complete_method

    def update_gesture(self, gesture):
        self.gesture = gesture

    def __call__(self, glb_rot, Rs, left=True):
        if self.gesture == 'fist':
            init_Rs = Rs[:2, :, :, :]  # 4 x 15 x 3 x 3
            cur_Rs = Rs[2:, :, :, :]
            delta_pose = np.einsum('klij,kljm->klim', np.linalg.inv(init_Rs), cur_Rs)
            delta_angles = np.arccos(np.einsum('jkii', delta_pose) * 0.5 - 0.5)
            if left:
                sickside_angles = delta_angles[0]
                goodside_angles = delta_angles[1]
            else:
                sickside_angles = delta_angles[1]
                goodside_angles = delta_angles[0]
            sickside_angle = np.mean(sickside_angles[[0, 1, 3, 4, 6, 7, 9, 10]])
            goodside_angle = np.mean(goodside_angles[[0, 1, 3, 4, 6, 7, 9, 10]])
            # sickside_angle = np.mean(np.linalg.norm(sickside_pose_param.reshape(-1, 3), axis=1)[[0, 1, 3, 4, 6, 7, 9, 10]])
            # goodside_angle = np.mean(np.linalg.norm(goodside_pose_param.reshape(-1, 3), axis=1)[[0, 1, 3, 4, 6, 7, 9, 10]])
            if self.complete_method == 0:
                completeness = abs(sickside_angle) / (np.pi / 2)
            else:
                completeness = abs(sickside_angle) / abs(goodside_angle)
        else:
            init_glb_rot = glb_rot[:2, :]
            cur_glb_rot = glb_rot[2:, :] 
            glb_rot_angles = 2 * np.arccos(cur_glb_rot[:, 0] / np.linalg.norm(cur_glb_rot, axis=1)) - \
                2 * np.arccos(init_glb_rot[:, 0] / np.linalg.norm(init_glb_rot, axis=1))
            if left:
                sickside_angle = glb_rot_angles[0]
                goodside_angle = glb_rot_angles[1]
            else:
                sickside_angle = glb_rot_angles[1]
                goodside_angle = glb_rot_angles[0]
            # glb_rotation = opt_params[0, 45:] if left else opt_params[1, 45:]
            # sickside_angle = 2 * np.arccos(sickside_glb_rot_param[0] / np.linalg.norm(sickside_glb_rot_param))
            # goodside_angle = 2 * np.arccos(goodside_glb_rot_param[0] / np.linalg.norm(goodside_glb_rot_param))
            if self.complete_method == 0:
                completeness = abs(sickside_angle) / (self.wrist_rot_pitch if self.gesture != "wristside" else self.wrist_rot_yaw)
            else:
                completeness = abs(sickside_angle) / abs(goodside_angle)
        return completeness, int(sickside_angle / np.pi * 180), int(goodside_angle / np.pi * 180) 

    # def __call__(self, opt_params, init_params, left=True):
    #     if self.gesture == 'fist':
    #         parameters = opt_params[0, :45] if left else opt_params[1, :45]
    #         init = init_params[0, :45] if left else opt_params[1, :45]
    #         rot_vector = np.dot(parameters, self.left_comp if left else self.right_comp)
    #         selected_pose_angles = rot_vector[[2, 5, 11, 14, 20, 23, 29, 32]]
    #         completeness = np.mean(np.abs(selected_pose_angles) / self.pose_angle)
    #     elif self.gesture == "wristbackward" or "wristforward":
    #         parameters = opt_params[0, 55:59] if left else opt_params[1, 55:59]
    #         init = init_params[0, 55:59] if left else init_params[1, 55:59]
    #         wrist_rot_pitch = 2 * np.arccos(np.dot(parameters, init) / (np.linalg.norm(parameters) * np.linalg.norm(init)))
    #         completeness = abs(wrist_rot_pitch) / self.wrist_rot_pitch
    #     else:
    #         parameters = opt_params[0, 55:59] if left else opt_params[1, 55:59]
    #         init = init_params[0, 55:59] if left else init_params[1, 55:59]
    #         wrist_rot_yaw = 2 * np.arccos(np.dot(parameters, init) / (np.linalg.norm(parameters) * np.linalg.norm(init)))
    #         completeness = abs(wrist_rot_yaw) / self.wrist_rot_yaw
    #     return completeness
