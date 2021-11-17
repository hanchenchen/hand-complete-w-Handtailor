"""
完成度计算函数，包含两个功能
1.不同手势的完成度计算不同
"""
import numpy as np


class Completeness(object):

    def __init__(self, gesture, wrist_rot_pitch=60, wrist_rot_yaw=45, complete_method=0):
        self.gesture = gesture

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
            if self.complete_method == 0:
                completeness = abs(sickside_angle) / (np.pi / 2)
            else:
                completeness = abs(sickside_angle) / abs(goodside_angle)
        else:
            init_glb_rot = glb_rot[:2, :]
            cur_glb_rot = glb_rot[2:, :] 
            glb_rot_angles = 2*np.arccos(np.abs(np.sum(init_glb_rot * cur_glb_rot, axis=1)))
            if left:
                sickside_angle = glb_rot_angles[0]
                goodside_angle = glb_rot_angles[1]
            else:
                sickside_angle = glb_rot_angles[1]
                goodside_angle = glb_rot_angles[0]

            if self.complete_method == 0:
                completeness = abs(sickside_angle) / (self.wrist_rot_pitch if self.gesture != "wristside" else self.wrist_rot_yaw)
            else:
                completeness = abs(sickside_angle) / abs(goodside_angle)
        return completeness, int(sickside_angle / np.pi * 180), int(goodside_angle / np.pi * 180) 
