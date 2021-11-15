"""
Hand Model Fitting
"""
import torch
import numpy as np
from collections import OrderedDict
from utils import Constraints
from utils.loss import chamfer_loss
from utils.optimizer import create_optimizer
from utils.quaternion import qmul
# from utils.util import export_ply, add_arm_vertices
from utils.util import add_arm_vertices
import time

torch.set_num_threads(1)
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = False
import cv2
import numpy as np
import torch
import time
import torch.backends.cudnn as cudnn
import jax.numpy as npj
import PIL.Image as Image
import glob
import argparse
from jax import grad, jit, vmap
from jax.experimental import optimizers
from torchvision.transforms import functional
import pickle

from manolayer import ManoLayer
from model import HandNet
from checkpoints import CheckpointIO
import utils
import pickle as pkl
import os

import numpy as np
import logging
import matplotlib.pyplot as plt
import json
from sklearn.metrics import auc

import numpy as np
import math
from scipy.spatial.transform import Rotation as R

mano2cmu = [
    0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20
]

valid2full = [5, 0, 6, 7, 8, 9, 10, 11, 12,  # index
              13, 1, 14, 15, 16, 17, 18, 19, 20,  # middle
              21, 2, 22, 23, 24, 25, 26, 27, 28,  # pinky
              29, 3, 30, 31, 32, 33, 34, 35, 36,  # ring
              37, 4, 38, 39, 40, 41, 42, 43, 44,  # thumb
              ]


class Solver(object):

    def __init__(self,
                 mano_root="./mano",
                 ):
        mano_layer_right = ManoLayer(center_idx=9, side="right", mano_root=mano_root, use_pca=False,
                                     flat_hand_mean=True, )
        mano_layer_left = ManoLayer(center_idx=9, side="left", mano_root=mano_root, use_pca=False,
                                    flat_hand_mean=True, )
        self.mano_layer_right = jit(mano_layer_right)
        self.mano_layer_left = jit(mano_layer_left)

        self.hand_side = "right"
        self.mano_layer = {"right": self.mano_layer_right, "left": self.mano_layer_left}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HandNet()
        self.model = self.model.to(self.device)
        checkpoint_io = CheckpointIO('.', model=self.model)
        load_dict = checkpoint_io.load('./checkpoints/model.pt')
        self.model.eval()

        dd = pickle.load(open("./mano/MANO_RIGHT.pkl", 'rb'), encoding='latin1')
        face = np.array(dd['f'])
        self.renderer = utils.MeshRenderer(face, img_size=256)

        self.gr = jit(grad(self.residuals))
        lr = 0.03
        opt_init, opt_update, get_params = optimizers.adam(lr, b1=0.5, b2=0.5)
        self.opt_init = jit(opt_init)
        self.opt_update = jit(opt_update)
        self.get_params = jit(get_params)


        def __call__(self, img, Ks, hand_side):
            img = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)
            frame = img.copy()

            intr = torch.from_numpy(np.array(Ks, dtype=np.float32)).unsqueeze(0).to(self.device)
            intr[:, :2, :] *= 256 / img.shape[0]

            _intr = intr.cpu().numpy()
            camparam = np.zeros((1, 21, 4))
            camparam[:, :, 0] = _intr[:, 0, 0]
            camparam[:, :, 1] = _intr[:, 1, 1]
            camparam[:, :, 2] = _intr[:, 0, 2]
            camparam[:, :, 3] = _intr[:, 1, 2]

            img = functional.to_tensor(img).float()
            img = functional.normalize(img, [0.5, 0.5, 0.5], [1, 1, 1])
            img = img.unsqueeze(0).to(self.device)

            hm, so3, beta, joint_root, bone = self.model(img, intr)
            kp2d = self.hm_to_kp2d(hm.detach().cpu().numpy()) * 4
            so3 = so3[0].detach().cpu().float().numpy()
            beta = beta[0].detach().cpu().float().numpy()

            bone = bone[0].detach().cpu().numpy()
            joint_root = joint_root[0].detach().cpu().numpy()
            so3 = npj.array(so3)
            beta = npj.array(beta)
            bone = npj.array(bone)
            joint_root = npj.array(joint_root)
            kp2d = npj.array(kp2d)
            so3_init = so3
            beta_init = beta
            joint_root = self.reinit_root(joint_root, kp2d, camparam)
            joint = self.mano_de_j(so3, beta)
            bone = self.reinit_scale(joint, kp2d, camparam, bone, joint_root)
            params = {'so3': so3, 'beta': beta, 'bone': bone}
            opt_state = self.opt_init(params)
            n = 0
            while n < 20:
                n = n + 1
                params = self.get_params(opt_state)
                grads = self.gr(params, so3_init, beta_init, joint_root, kp2d, camparam)
                opt_state = self.opt_update(n, grads, opt_state)
            params = self.get_params(opt_state)

            pred_v, pred_joint_3d, result = self.mano_de(params, joint_root, bone)
            frame1 = self.renderer(pred_v, intr[0].cpu(), frame)
            if not os.path.exists(f"workspace/hand-complete/{dire}/"):
                os.makedirs(f"workspace/hand-complete/{dire}/")
            # cv2.imwrite(f"workspace/hand-complete/{dire}/{img_path.split('/')[-1]}_pred.jpg", np.flip(frame1, -1))

            root_rot_matrix = result['root_rot_matrix']
            root_rot_matrix = R.from_matrix(root_rot_matrix)
            root_quat = root_rot_matrix.as_quat()
            euler = root_rot_matrix.as_euler('zxy', degrees=True)
            if not i:
                angle = 0
                init_root_quat = root_quat
                init_euler = euler
                print(init_root_quat)
            else:
                print('root_quat', root_quat)
                angle = 2 * np.arccos(np.abs(np.sum(init_root_quat * root_quat))) * 180 / np.pi
                print('angle', angle)
                print('euler', euler - init_euler)
                cv2.imwrite(f"workspace/hand-complete/{dire}/{img_path.split('/')[-1]}_pred_{angle}.jpg",
                            np.flip(frame1, -1))
            print()

            gt_v, gt_joint_3d, result = self.mano_de(
                {'so3': np.concatenate((meta_info["mano_params_r"][-3:], meta_info["mano_params_r"][0:45],), axis=-1),
                 'beta': meta_info["mano_params_r"][45:55], 'bone': bone, 'quat': meta_info["mano_params_r"][55:59]},
                joint_root, bone)
            frame1 = self.renderer(gt_v, intr[0].cpu(), frame)
            # cv2.imwrite(f"workspace/hand-complete/{dire}/{img_path.split('/')[-1]}_gt.jpg", np.flip(frame1, -1))
            # mano2cmu = [
            #     0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20
            # ]
            # gt_3d = meta_info["joints_3d_normed_r"][mano2cmu, :]

            root_rot_matrix = result['root_rot_matrix']
            root_rot_matrix = R.from_matrix(root_rot_matrix)
            root_quat = root_rot_matrix.as_quat()
            euler = root_rot_matrix.as_euler('zxy', degrees=True)
            if not i:
                angle = 0
                init_root_quat = root_quat
                init_euler = euler
                print(init_root_quat)
            else:
                print('root_quat', root_quat)
                angle = 2 * np.arccos(np.abs(np.sum(init_root_quat * root_quat))) * 180 / np.pi
                print('angle', angle)
                print('euler', euler - init_euler)
                cv2.imwrite(f"workspace/hand-complete/{dire}/{img_path.split('/')[-1]}_gt_{angle}.jpg",
                            np.flip(frame1, -1))
            print()
            predict_labels_dict[img_path] = {}
            predict_labels_dict[img_path]["prd_label"] = pred_joint_3d[0] * 1000
            predict_labels_dict[img_path]["resol"] = 480
            gt_labels[img_path] = gt_joint_3d[0] * 1000

            return output


        @jit
        def hm_to_kp2d(hm):
            b, c, w, h = hm.shape
            hm = hm.reshape(b, c, -1)
            hm = hm / npj.sum(hm, -1, keepdims=True)
            coord_map_x = npj.tile(npj.arange(0, w).reshape(-1, 1), (1, h))
            coord_map_y = npj.tile(npj.arange(0, h).reshape(1, -1), (w, 1))
            coord_map_x = coord_map_x.reshape(1, 1, -1)
            coord_map_y = coord_map_y.reshape(1, 1, -1)
            x = npj.sum(coord_map_x * hm, -1, keepdims=True)
            y = npj.sum(coord_map_y * hm, -1, keepdims=True)
            kp_2d = npj.concatenate((y, x), axis=-1)
            return kp_2d

        @jit
        def reinit_root(joint_root, kp2d, camparam):
            uv = kp2d[0, 9, :]
            xy = joint_root[..., :2]
            z = joint_root[..., 2]
            joint_root = ((uv - camparam[0, 0, 2:4]) / camparam[0, 0, :2]) * z
            joint_root = npj.concatenate((joint_root, z))
            return joint_root

        @jit
        def reinit_scale(joint, kp2d, camparam, bone, joint_root):
            z0 = joint_root[2:]
            xy0 = joint_root[:2]
            xy = joint[:, :2] * bone
            z = joint[:, 2:] * bone
            kp2d = kp2d[0]
            s1 = npj.sum(
                ((kp2d - camparam[0, 0, 2:4]) * xy) / (camparam[0, 0, :2] * (z0 + z)) - (xy0 * xy) / ((z0 + z) ** 2))
            s2 = npj.sum((xy ** 2) / ((z0 + z) ** 2))
            s = s1 / s2
            bone = bone * npj.max(npj.array([s, 0.9]))
            return bone

        @jit
        def geo(joint):
            idx_a = npj.array([1, 5, 9, 13, 17])
            idx_b = npj.array([2, 6, 10, 14, 18])
            idx_c = npj.array([3, 7, 11, 15, 19])
            idx_d = npj.array([4, 8, 12, 16, 20])
            p_a = joint[:, idx_a, :]
            p_b = joint[:, idx_b, :]
            p_c = joint[:, idx_c, :]
            p_d = joint[:, idx_d, :]
            v_ab = p_a - p_b  # (B, 5, 3)
            v_bc = p_b - p_c  # (B, 5, 3)
            v_cd = p_c - p_d  # (B, 5, 3)
            loss_1 = npj.abs(npj.sum(npj.cross(v_ab, v_bc, -1) * v_cd, -1)).mean()
            loss_2 = - npj.clip(npj.sum(npj.cross(v_ab, v_bc, -1) * npj.cross(v_bc, v_cd, -1)), -npj.inf, 0).mean()
            loss = 10000 * loss_1 + 100000 * loss_2

            return loss

        @jit
        def residuals(input_list, so3_init, beta_init, joint_root, kp2d, camparam):
            so3 = input_list['so3']
            beta = input_list['beta']
            bone = input_list['bone']
            so3 = so3[npj.newaxis, ...]
            beta = beta[npj.newaxis, ...]
            _, joint_mano, _, _ = self.mano_layer[self.hand_side](
                pose_coeffs=so3,
                betas=beta
            )
            bone_pred = npj.linalg.norm(joint_mano[:, 0, :] - joint_mano[:, 9, :], axis=1, keepdims=True)
            bone_pred = bone_pred[:, npj.newaxis, ...]
            reg = ((so3 - so3_init) ** 2)
            reg_beta = ((beta - beta_init) ** 2)
            joint_mano = joint_mano / bone_pred
            joint_mano = joint_mano * bone + joint_root
            geo_reg = self.geo(joint_mano)
            xy = (joint_mano[..., :2] / joint_mano[..., 2:])
            uv = (xy * camparam[:, :, :2]) + camparam[:, :, 2:4]
            errkp = ((uv - kp2d) ** 2)
            err = 0.01 * reg.mean() + 0.01 * reg_beta.mean() + 1 * errkp.mean() + 100 * geo_reg.mean()
            return err

        @jit
        def mano_de(params, joint_root, bone):
            so3 = params['so3']
            beta = params['beta']
            if 'quat' in params:
                quat = params['quat'][npj.newaxis, ...]
            else:
                quat = None
            verts_mano, joint_mano, _, result = self.mano_layer[self.hand_side](
                pose_coeffs=so3[npj.newaxis, ...],
                betas=beta[npj.newaxis, ...],
                quat=quat
            )

            bone_pred = npj.linalg.norm(joint_mano[:, 0, :] - joint_mano[:, 9, :], axis=1, keepdims=True)
            bone_pred = bone_pred[:, npj.newaxis, ...]
            verts_mano = verts_mano / bone_pred
            verts_mano = verts_mano * bone + joint_root
            v = verts_mano[0]
            return v, joint_mano, result

        @jit
        def mano_de_j(so3, beta):
            _, joint_mano, _, _ = self.mano_layer[self.hand_side](
                pose_coeffs=so3[npj.newaxis, ...],
                betas=beta[npj.newaxis, ...]
            )

            bone_pred = npj.linalg.norm(joint_mano[:, 0, :] - joint_mano[:, 9, :], axis=1, keepdims=True)
            bone_pred = bone_pred[:, npj.newaxis, ...]
            joint_mano = joint_mano / bone_pred
            j = joint_mano[0]
            return j



    def projectPoints_batch(xyz, K, eps=1e-9):
        """
        Project 3D coordinates into image space.
        K: intrinsic camera matrix batch_size x 3 x 3 or 3 x 3
        """
        if not isinstance(K, torch.Tensor):
            K = torch.from_numpy(K).float().to(xyz.device)
        if len(xyz.shape) == 2:
            xyz = xyz.unsqueeze(0)
        if len(K.shape) == 2:
            K = K.reshape(-1, 3, 3).repeat(xyz.shape[0], 1, 1)
        uv = torch.matmul(xyz, K.transpose(1, 2))
        z = uv[:, :, 2:]
        return uv[:, :, :2] / (z + eps)

    def gmof(x, sigma):
        """
        Geman-McClure error function
        """
        x_squared = x ** 2
        sigma_squared = sigma ** 2
        return (sigma_squared * x_squared) / (sigma_squared + x_squared)
