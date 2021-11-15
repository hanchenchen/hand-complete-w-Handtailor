"""
Get theta and temporal constraints
"""
import torch
import numpy as np


MINBOUND = -5.0
MAXBOUND = 5.0


class Constraints(object):

    def __init__(self, right=True):
        if right:
            # 右手模型
            self.minTheta = torch.from_numpy(np.array([MINBOUND, MINBOUND, MINBOUND,  # global rot
                                                       0, -0.15, 0.1, -0.3, MINBOUND, -0.0, MINBOUND, MINBOUND, 0,  # index
                                                       MINBOUND, -0.15, 0.1, -0.5, MINBOUND, -0.0, MINBOUND, MINBOUND, 0,  # middle
                                                       -1.5, -0.15, -0.1, MINBOUND, -0.5, -0.0, MINBOUND, MINBOUND, 0,  # pinky
                                                       -0.5, -0.25, 0.1, -0.4, MINBOUND, -0.0, MINBOUND, MINBOUND, 0,   # ring
                                                       0.0, -0.83, -0.0, -0.15, MINBOUND, 0, MINBOUND, -0.5, -1.57])).float()   # thumb
            self.maxTheta = torch.from_numpy(np.array([MAXBOUND, MAXBOUND, MAXBOUND,  # global rot
                                                       0.45, 0.2, 1.8, 0.2, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25,  # index
                                                       MAXBOUND, 0.15, 2.0, -0.2, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25,  # middle
                                                       -0.2, 0.6, 1.6, MAXBOUND, 0.6, 2.0, MAXBOUND, MAXBOUND, 1.25,  # pinky
                                                       -0.4, 0.10, 1.8, -0.2, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25,  # ring
                                                       2.0, 0.66, 0.5, 1.6, MAXBOUND, 0.5, MAXBOUND, 0, 1.08])).float()  # thumb
        else:
            # 左手模型
            self.minTheta = torch.from_numpy(np.array([MINBOUND, MINBOUND, MINBOUND,  # global rot
                                                       0, -0.15, -1.8, -0.3, MINBOUND, -2.0, MINBOUND, MINBOUND, -1.25,  # index
                                                       MINBOUND, -0.15, -2.0, -0.5, MINBOUND, -2.0, MINBOUND, MINBOUND, -1.25,  # middle
                                                       -1.5, -0.15, -1.6, MINBOUND, -0.5, -2.0, MINBOUND, MINBOUND, -1.25,  # pinky
                                                       -0.5, -0.25, -1.8, -0.4, MINBOUND, -2.0, MINBOUND, MINBOUND, -1.25,  # ring
                                                       0.0, -0.83, -0.5, -0.15, MINBOUND, -0.5, MINBOUND, -0.5, -1.08])).float()  # thumb
            self.maxTheta = torch.from_numpy(np.array([MAXBOUND, MAXBOUND, MAXBOUND,  # global rot
                                                       0.45, 0.2, -0.1, 0.2, MAXBOUND, -0.0, MAXBOUND, MAXBOUND, -0.0,  # index
                                                       MAXBOUND, 0.15, -0.1, -0.2, MAXBOUND, -0.0, MAXBOUND, MAXBOUND, -0.0,  # middle
                                                       -0.2, 0.6, 0.1, MAXBOUND, 0.6, -0.0, MAXBOUND, MAXBOUND, -0.0,  # pinky
                                                       -0.4, 0.10, -0.1, -0.2, MAXBOUND, -0.0, MAXBOUND, MAXBOUND, -0.0,  # ring
                                                       2.0, 0.66, -0.0, 1.6, MAXBOUND, -0.0, MAXBOUND, 0, 1.57])).float()  # thumb

        self.validThetaIDs = [0, 1, 2, 3, 4, 5, 6, 8, 11, 13, 14, 15, 17, 20, 21, 22, 23, 25, 26, 29,
                              30, 31, 32, 33, 35, 38, 39, 40, 41, 42, 44, 46, 47]
        invalidThetaIDsList = []
        for i in range(48):
            if i not in self.validThetaIDs:
                invalidThetaIDsList.append(i)
        self.invalidThetaIDs = np.array(invalidThetaIDsList)

    def getPCAPoseConstraints(self, pose_param):
        # loss = torch.nn.MSELoss()(pose_param, torch.zeros_like(pose_param).to(pose_param.device))
        loss = torch.sum(torch.pow(pose_param, 2))
        return loss

    def getAnglePoseConstraints(self, joint_angles):
        assert joint_angles.shape[-1] == 45, "Invalid shape of joint angles, {}".format(joint_angles.shape)
        device = joint_angles.device
        validThetaIDs = [idx - 3 for idx in self.validThetaIDs[3:]]
        valid_joint_angles = joint_angles[:, validThetaIDs].contiguous()
        min_joint_angles = self.minTheta[self.validThetaIDs[3:]].reshape(1, -1).repeat(valid_joint_angles.shape[0], 1).to(device)
        max_joint_angles = self.maxTheta[self.validThetaIDs[3:]].reshape(1, -1).repeat(valid_joint_angles.shape[0], 1).to(device)
        phyConst1 = torch.pow(torch.max(min_joint_angles - valid_joint_angles, torch.zeros_like(valid_joint_angles).to(device)), 2)
        phyConst2 = torch.pow(torch.max(valid_joint_angles - max_joint_angles, torch.zeros_like(valid_joint_angles).to(device)), 2)
        phyConst = phyConst1 + phyConst2
        loss = torch.sum(phyConst)
        return loss

    def getShapeConstraints(self, shape_param):
        # loss = torch.nn.MSELoss()(shape_param, torch.zeros_like(shape_param).to(shape_param.device))
        loss = torch.sum(torch.pow(shape_param, 2))
        return loss

    def getTemporalConstraints(self, pose_param):
        assert len(pose_param.shape) == 2
        loss_vel = torch.sum(torch.pow(pose_param[:-1, :] - pose_param[1:, :], 2))
        vel = pose_param[1:, :] - pose_param[:-1, :]
        loss_acc = torch.sum(torch.pow(vel[:-1, :] - vel[1:, :], 2))

        return loss_vel + loss_acc
