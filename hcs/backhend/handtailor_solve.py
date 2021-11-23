"""
Hand Model Fitting
"""
import sys
# sys.path.append('/workspace/hand-complete-w-Handtailor/hcs/')
# print(sys.path)
import torch
import numpy as np
from utils.util import add_arm_vertices

# torch.set_num_threads(1)

import cv2
import jax.numpy as npj
import PIL.Image as Image
from jax import grad, jit, vmap
from jax.experimental import optimizers
from torchvision.transforms import functional
import pickle

from handtailor.manolayer import ManoLayer
from handtailor.model import HandNet
from handtailor.checkpoints import CheckpointIO
import handtailor.utils as utils

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

mano_root="./mano"
hand_side = "right"
mano_layer_right = ManoLayer(center_idx=9, side="right", mano_root=mano_root, use_pca=False,
                             flat_hand_mean=True, )
mano_layer_left = ManoLayer(center_idx=9, side="left", mano_root=mano_root, use_pca=False,
                            flat_hand_mean=True, )
mano_layer_right = jit(mano_layer_right)
mano_layer_left = jit(mano_layer_left)



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
    if hand_side == "right":
        _, joint_mano, _, _ = mano_layer_right(
            pose_coeffs=so3,
            betas=beta
        )
    else:
        _, joint_mano, _, _ = mano_layer_left(
            pose_coeffs=so3,
            betas=beta
        )

    bone_pred = npj.linalg.norm(joint_mano[:, 0, :] - joint_mano[:, 9, :], axis=1, keepdims=True)
    bone_pred = bone_pred[:, npj.newaxis, ...]
    reg = ((so3 - so3_init) ** 2)
    reg_beta = ((beta - beta_init) ** 2)
    joint_mano = joint_mano / bone_pred
    joint_mano = joint_mano * bone + joint_root
    geo_reg = geo(joint_mano)
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

    if hand_side == "right":
        verts_mano, joint_mano, _, result = mano_layer_right(
            pose_coeffs=so3[npj.newaxis, ...],
            betas=beta[npj.newaxis, ...],
            quat=quat
        )
    else:
        verts_mano, joint_mano, _, result = mano_layer_left(
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
    if hand_side == "right":
        _, joint_mano, _, _ = mano_layer_right(
            pose_coeffs=so3[npj.newaxis, ...],
            betas=beta[npj.newaxis, ...]
        )
    else:
        _, joint_mano, _, _ = mano_layer_left(
            pose_coeffs=so3[npj.newaxis, ...],
            betas=beta[npj.newaxis, ...]
        )
    bone_pred = npj.linalg.norm(joint_mano[:, 0, :] - joint_mano[:, 9, :], axis=1, keepdims=True)
    bone_pred = bone_pred[:, npj.newaxis, ...]
    joint_mano = joint_mano / bone_pred
    j = joint_mano[0]
    return j


class Solver(object):

    def __init__(self, Ks, size):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = HandNet()
        self.model = self.model.to(self.device)
        checkpoint_io = CheckpointIO('.', model=self.model)
        load_dict = checkpoint_io.load('./checkpoints/model.pt')
        self.model.eval()

        dd = pickle.load(open("./mano/MANO_RIGHT.pkl", 'rb'), encoding='latin1')
        face = np.array(dd['f'])
        self.renderer = utils.MeshRenderer(face, img_size=256)

        self.gr = jit(grad(residuals))
        lr = 0.03
        opt_init, opt_update, get_params = optimizers.adam(lr, b1=0.5, b2=0.5)
        self.opt_init = jit(opt_init)
        self.opt_update = jit(opt_update)
        self.get_params = jit(get_params)

        self.intr = torch.from_numpy(np.array(Ks, dtype=np.float32)).unsqueeze(0).to(self.device)
        self.intr[:, :2, :] *= 256 / size

        _intr = self.intr.cpu().numpy()
        self.camparam = np.zeros((1, 21, 4))
        self.camparam[:, :, 0] = _intr[:, 0, 0]
        self.camparam[:, :, 1] = _intr[:, 1, 1]
        self.camparam[:, :, 2] = _intr[:, 0, 2]
        self.camparam[:, :, 3] = _intr[:, 1, 2]


    @torch.no_grad()
    def __call__(self, img, Ks, hand_side):
        if hand_side == 0:
            self.hand_side = "right"
        else:
            self.hand_side = "left"
        # test codes
        # color = np.array(Image.open('000000.jpg'))
        # H, W, C = color.shape
        # img = color[:, :H, :]
        #
        # Ks = pickle.load(open('000000.pkl', "rb"))['ks']


        img = cv2.resize(img, (256, 256), cv2.INTER_LINEAR)

        img = functional.to_tensor(img).float()
        img = functional.normalize(img, [0.5, 0.5, 0.5], [1, 1, 1])
        img = img.unsqueeze(0).to(self.device)

        hm, so3, beta, joint_root, bone = self.model(img, self.intr)
        kp2d = hm_to_kp2d(hm.detach().cpu().numpy()) * 4
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
        joint_root = reinit_root(joint_root, kp2d, self.camparam)
        joint = mano_de_j(so3, beta)
        bone = reinit_scale(joint, kp2d, self.camparam, bone, joint_root)
        params = {'so3': so3, 'beta': beta, 'bone': bone}
        opt_state = self.opt_init(params)
        n = 0
        while n < 5: # default 20
            n = n + 1
            params = self.get_params(opt_state)
            grads = self.gr(params, so3_init, beta_init, joint_root, kp2d, self.camparam)
            opt_state = self.opt_update(n, grads, opt_state)
        params = self.get_params(opt_state)

        pred_v, pred_joint_3d, result = mano_de(params, joint_root, bone)
        root_rot_matrix = result['root_rot_matrix']
        root_rot_matrix = R.from_matrix(root_rot_matrix)
        root_quat = root_rot_matrix.as_quat()

        opt_params = np.zeros((62))
        opt_params[0:45] = params['so3'][3:]
        opt_params[45:55] = params['beta']
        opt_params[55:59] = root_quat

        output = {
            "opt_params": opt_params.astype('float32'),
            "vertices": pred_v.astype('float32'),
            "hand_joints": kp2d,
            "glb_rot": root_quat,
            "Rs": result['pose_rot_matrix']
        }

        return output



if __name__ == "__main__":
    # test codes

    color = np.array(Image.open('000000.jpg'))
    Ks = pickle.load(open('000000.pkl', "rb"))['ks']


    # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    H, W, C = color.shape
    print(color.shape)
    color_left = color[:, :H, :]
    color_right = color[:, -1:-H:-1, :]
    output = {
        "opt_params": [],
        "vertices": [],
        "hand_joints": []
    }
    solver = Solver(Ks=Ks, size=H)

    dd = pickle.load(open("./mano/MANO_RIGHT.pkl", 'rb'), encoding='latin1')
    face = np.array(dd['f'])
    renderer = utils.MeshRenderer(face, img_size=256)

    for i, img in enumerate([color_left, color_right]):
        frame = img.copy()
        frame = cv2.resize(frame, (256, 256), cv2.INTER_LINEAR)

        _ = solver(img, Ks, i)
        output["opt_params"].append(_["opt_params"])
        output["vertices"].append(_["vertices"])
        output["hand_joints"].append(_["hand_joints"])

        # frame1 = renderer(np.multiply(_["vertices"], [-1, 1, 1]), solver.intr[0].cpu(), frame)
        frame1 = renderer(_["vertices"], solver.intr[0].cpu(), frame)

        cv2.imwrite(f"img{i}.jpg", np.flip(frame1, -1))
