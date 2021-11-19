import cv2
import numpy as np
import pygame
import torch
import time
import torch.backends.cudnn as cudnn
# import pyrealsense2 as rs
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

mano_layer = ManoLayer(center_idx=9, side="right", mano_root="workspace/mano", use_pca=False, flat_hand_mean=True,)
mano_layer = jit(mano_layer)

@jit
def hm_to_kp2d(hm):
    b, c, w, h = hm.shape
    hm = hm.reshape(b,c,-1)
    hm = hm/npj.sum(hm,-1,keepdims=True)
    coord_map_x = npj.tile(npj.arange(0,w).reshape(-1,1), (1,h))
    coord_map_y = npj.tile(npj.arange(0,h).reshape(1,-1), (w,1))
    coord_map_x = coord_map_x.reshape(1,1,-1)
    coord_map_y = coord_map_y.reshape(1,1,-1)
    x = npj.sum(coord_map_x * hm,-1,keepdims=True)
    y = npj.sum(coord_map_y * hm,-1,keepdims=True)
    kp_2d = npj.concatenate((y,x),axis=-1)
    return kp_2d

@jit
def reinit_root(joint_root,kp2d,camparam):
    uv = kp2d[0,9,:]
    xy = joint_root[...,:2]
    z = joint_root[...,2]
    joint_root = ((uv - camparam[0, 0, 2:4])/camparam[0, 0, :2]) * z
    joint_root = npj.concatenate((joint_root,z))
    return joint_root

@jit
def reinit_scale(joint,kp2d,camparam,bone,joint_root):
    z0 = joint_root[2:]
    xy0 = joint_root[:2]
    xy = joint[:,:2] * bone
    z = joint[:,2:] * bone
    kp2d = kp2d[0]
    s1 = npj.sum(((kp2d - camparam[0, 0, 2:4])*xy)/(camparam[0, 0, :2]*(z0+z)) - (xy0*xy)/((z0+z)**2))
    s2 = npj.sum((xy**2)/((z0+z)**2))
    s = s1/s2
    bone = bone * npj.max(npj.array([s,0.9]))
    return bone

@jit
def geo(joint):
    idx_a = npj.array([1,5,9,13,17])
    idx_b = npj.array([2,6,10,14,18])
    idx_c = npj.array([3,7,11,15,19])
    idx_d = npj.array([4,8,12,16,20])
    p_a = joint[:,idx_a,:]
    p_b = joint[:,idx_b,:]
    p_c = joint[:,idx_c,:]
    p_d = joint[:,idx_d,:]
    v_ab = p_a - p_b #(B, 5, 3)
    v_bc = p_b - p_c #(B, 5, 3)
    v_cd = p_c - p_d #(B, 5, 3)
    loss_1 = npj.abs(npj.sum(npj.cross(v_ab, v_bc, -1) * v_cd, -1)).mean()
    loss_2 = - npj.clip(npj.sum(npj.cross(v_ab, v_bc, -1) * npj.cross(v_bc, v_cd, -1)), -npj.inf, 0).mean()
    loss = 10000*loss_1 + 100000*loss_2

    return loss

@jit
def residuals(input_list,so3_init,beta_init,joint_root,kp2d,camparam):
    so3 = input_list['so3']
    beta = input_list['beta']
    bone = input_list['bone']
    so3 = so3[npj.newaxis,...]
    beta = beta[npj.newaxis,...]
    _, joint_mano, _, _ = mano_layer(
        pose_coeffs = so3,
        betas = beta
    )
    bone_pred = npj.linalg.norm(joint_mano[:, 0, :] - joint_mano[:, 9, :], axis=1, keepdims=True)
    bone_pred = bone_pred[:,npj.newaxis,...]
    reg = ((so3 - so3_init)**2)
    reg_beta = ((beta - beta_init)**2)
    joint_mano = joint_mano / bone_pred
    joint_mano = joint_mano * bone + joint_root
    geo_reg = geo(joint_mano)
    xy = (joint_mano[...,:2]/joint_mano[...,2:])
    uv = (xy * camparam[:, :, :2] ) + camparam[:, :, 2:4]
    errkp = ((uv - kp2d)**2)
    err = 0.01*reg.mean() + 0.01*reg_beta.mean() + 1*errkp.mean() + 100*geo_reg.mean()
    return err

@jit
def mano_de(params,joint_root,bone):
    so3 = params['so3']
    beta = params['beta']
    if 'quat' in params:
        quat = params['quat'][npj.newaxis,...]
    else:
        quat = None
    verts_mano, joint_mano, _, result = mano_layer(
        pose_coeffs = so3[npj.newaxis,...],
        betas = beta[npj.newaxis,...],
        quat = quat
    )

    bone_pred = npj.linalg.norm(joint_mano[:, 0, :] - joint_mano[:, 9, :],axis=1, keepdims=True)
    bone_pred = bone_pred[:,npj.newaxis,...]
    verts_mano = verts_mano / bone_pred
    verts_mano = verts_mano * bone  + joint_root
    v = verts_mano[0]
    return v, joint_mano, result

@jit
def mano_de_j(so3, beta):
    _, joint_mano, _, _ = mano_layer(
        pose_coeffs = so3[npj.newaxis,...],
        betas = beta[npj.newaxis,...]
    )

    bone_pred = npj.linalg.norm(joint_mano[:, 0, :] - joint_mano[:, 9, :],axis=1, keepdims=True)
    bone_pred = bone_pred[:,npj.newaxis,...]
    joint_mano = joint_mano / bone_pred
    j = joint_mano[0]
    return j


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


def get_pck_with_sigma(predict_labels_dict, gt_labels, sigma_list = np.arange(0, 1, 0.05), save_path = None ):
    """
    Get PCK with different sigma threshold
    :param predict_labels_dict:  dict  element:  'img_name':{'prd_label':[list, coordinates of 21 keypoints],
                                                             'resol': origin image size}
    :param gt_labels:            dict  element:  'img_name': [list, coordinates of 21 keypoints ]
    :param sigma_list:       list    different sigma threshold
    :return:
    """
    pck_dict = {}
    interval = sigma_list
    for im in predict_labels_dict:
        gt_label = gt_labels[im]        # list    len:21      element:[x, y]
        pred_label = predict_labels_dict[im]['prd_label']  # list    len:21      element:[x, y]
        im_size = predict_labels_dict[im]['resol']
        for sigma in interval:
            if sigma not in pck_dict:
                pck_dict[sigma] = []
            pck_dict[sigma].append(PCK(pred_label, gt_label, sigma))
            # Attention!
            # since our cropped image is 2.2 times of hand tightest bounding box,
            # we simply use im_size / 2,2 as the tightest bounding box
    pck_res = np.zeros((len(interval),), dtype=np.float32)
    index = 0
    for sigma in interval:
        pck_res[index] = sum(pck_dict[sigma]) / len(pck_dict[sigma])
        index += 1
    AUC = auc(interval, pck_res)/(interval[-1] - interval[0])
    if save_path:
        # plot it
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(interval,
                pck_res,
                # c=colors[0],
                linestyle='-', linewidth=1)
        plt.xlabel('Normalized distance (px) / ', fontsize=12)
        plt.ylabel('Fraction of frames within distance / %', fontsize=12)
        plt.xlim([interval[0], interval[-1]])
        plt.ylim([0.0, 1.0])
        ax.grid(True)

        # save if required
        fig.savefig(save_path,
                    bbox_extra_artists=None,
                    bbox_inches='tight')

        # show if required
        plt.show(block=False)
        # plt.close(fig)
    pck_res = pck_res.tolist()
    return {'sigma_pck': {str(interval[i]): pck_res[i] for i in range(len(pck_res))}, 'AUC': AUC}


def PCK(predict, target, thre_dis=1):
    """
    Calculate PCK
    :param predict: list    len:21      element:[x, y]
    :param target:  list    len:21      element:[x, y]
    :param bb_size: tightest bounding box length of hand
    :param sigma:   threshold, we use 0.1 in default
    :return: scala range [0,1]
    """
    pck = 0
    for i in range(21):
        pre = predict[i]
        tar = target[i]
        dis = np.sqrt((pre[0] - tar[0]) ** 2 + (pre[1] - tar[1]) ** 2 + (pre[2] - tar[2]) ** 2)
        if dis < thre_dis:
            pck += 1
    return pck / 21.0


def live_application(arg):
    model = HandNet()
    model = model.to(device)
    checkpoint_io = CheckpointIO('.', model=model)
    load_dict = checkpoint_io.load('workspace/checkpoints/model.pt')
    model.eval()

    dd = pickle.load(open("workspace/mano/MANO_RIGHT.pkl", 'rb'), encoding='latin1')
    face = np.array(dd['f'])
    renderer = utils.MeshRenderer(face, img_size=256)
    
    cx = arg.cx
    cy = arg.cy
    fx = arg.fx
    fy = arg.fy
    
    gr = jit(grad(residuals))
    lr = 0.03
    opt_init, opt_update, get_params = optimizers.adam(lr, b1=0.5, b2=0.5)
    opt_init = jit(opt_init)
    opt_update = jit(opt_update)
    get_params = jit(get_params)
    dire = "ground_truth_wristbackward"
    # img_list = glob.glob(f"./workspace/{dire}/color/*")
    # print(img_list)
    # for i, img_path in enumerate(img_list):
    #     os.rename(img_path, '/'.join(img_path.split('/')[:-1]) +'/'+ img_path.split('/')[-1].zfill(10))
    img_list = glob.glob(f"./workspace/{dire}/color/*")
    img_list.sort()


    predict_labels_dict = {}
    gt_labels = {}
    init_root_quat = None
    with torch.no_grad():
        for i, img_path in enumerate(img_list):
            img = np.array(Image.open(img_path))
            if img is None:
                continue
            _cx = cx
            _cy = cy
            if img.shape[0] > img.shape[1]:
                margin = int((img.shape[0] - img.shape[1]) / 2)
                # img = img[margin:-margin]
                img = img[:, :-2*margin]
                _cy = cy - margin
                width = img.shape[1]
            elif img.shape[0] < img.shape[1]:
                margin = int((img.shape[1] - img.shape[0]) / 2)
                # img = img[:, margin:-margin]
                img = img[:, :-2*margin]
                _cx = cx - margin
            width = img.shape[0]

            img = cv2.resize(img, (256, 256),cv2.INTER_LINEAR)
            frame = img.copy()
            meta_info = pkl.load(open(img_path.replace("color", "meta").replace(".jpg", ".pkl"), "rb"))
            # print(meta_info)
            # for i,j in meta_info.items():
            #     print(i, j.shape)
            # exit()
            _cx = (_cx * 256)/width
            _cy = (_cy * 256)/width
            _fx = (fx * 256)/width
            _fy = (fy * 256)/width

            intr = torch.from_numpy(np.array(meta_info["ks"], dtype=np.float32)).unsqueeze(0).to(device)
            intr[:, :2, :] *= 256/width

            _intr = intr.cpu().numpy()

            camparam = np.zeros((1, 21, 4))
            camparam[:, :, 0] = _intr[:, 0, 0]
            camparam[:, :, 1] = _intr[:, 1, 1]
            camparam[:, :, 2] = _intr[:, 0, 2]
            camparam[:, :, 3] = _intr[:, 1, 2]

            img = functional.to_tensor(img).float()
            img = functional.normalize(img, [0.5, 0.5, 0.5], [1, 1, 1])
            img = img.unsqueeze(0).to(device)
            
            hm, so3, beta, joint_root, bone = model(img,intr)
            kp2d = hm_to_kp2d(hm.detach().cpu().numpy())*4
            # kp2d = meta_info["joints_2d_r"][None, :]*256.0/480.0
            so3 = so3[0].detach().cpu().float().numpy()
            # so3 = meta_info["mano_params_r"][0:48]
            beta = beta[0].detach().cpu().float().numpy()
            # beta = meta_info["mano_params_r"][48:58]

            bone = bone[0].detach().cpu().numpy()
            joint_root = joint_root[0].detach().cpu().numpy()
            so3 = npj.array(so3)
            beta = npj.array(beta)
            bone = npj.array(bone)
            joint_root = npj.array(joint_root)
            kp2d = npj.array(kp2d)
            so3_init = so3
            beta_init = beta
            joint_root = reinit_root(joint_root,kp2d, camparam)
            joint = mano_de_j(so3, beta)
            bone = reinit_scale(joint,kp2d,camparam,bone,joint_root)
            params = {'so3':so3, 'beta':beta, 'bone':bone}
            opt_state = opt_init(params)
            n = 0
            while n < 20:
                n = n + 1
                params = get_params(opt_state)
                grads = gr(params,so3_init,beta_init,joint_root,kp2d,camparam)
                opt_state = opt_update(n, grads, opt_state)
            params = get_params(opt_state)

            pred_v, pred_joint_3d, result = mano_de(params, joint_root, bone)
            frame1 = renderer(pred_v*[1, -1, 1], intr[0].cpu(), frame)
            print(pred_v.shape)
            print(pred_v[0])
            cv2.imwrite("pred.jpg", frame1)
            exit()
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


            gt_v, gt_joint_3d, result = mano_de(
                {'so3': np.concatenate((meta_info["mano_params_r"][-3:], meta_info["mano_params_r"][0:45],), axis=-1),
                 'beta': meta_info["mano_params_r"][45:55], 'bone': bone, 'quat':meta_info["mano_params_r"][55:59]}, joint_root, bone)
            frame1 = renderer(gt_v, intr[0].cpu(), frame)
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
                angle = 2*np.arccos(np.abs(np.sum(init_root_quat * root_quat)))*180/np.pi
                print('angle', angle)
                print('euler', euler - init_euler)
                cv2.imwrite(f"workspace/hand-complete/{dire}/{img_path.split('/')[-1]}_gt_{angle}.jpg", np.flip(frame1, -1))
            print()
            predict_labels_dict[img_path] = {}
            predict_labels_dict[img_path]["prd_label"] = pred_joint_3d[0]*1000
            predict_labels_dict[img_path]["resol"] = 480
            gt_labels[img_path] = gt_joint_3d[0]*1000

    # print(get_pck_with_sigma(predict_labels_dict, gt_labels, sigma_list=np.arange(0, 20, 1), save_path=f'workspace/hand-complete/{dire}/pck0-20.jpg'))
    # print(get_pck_with_sigma(predict_labels_dict, gt_labels, sigma_list=np.arange(20, 50, 1), save_path=f'workspace/hand-complete/{dire}/pck20-50.jpg'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cx',
        type=float,
        default=321.2842102050781,
    )

    parser.add_argument(
        '--cy',
        type=float,
        default=235.8609161376953,
    )

    parser.add_argument(
        '--fx',
        type=float,
        default=612.0206298828125,
    )

    parser.add_argument(
        '--fy',
        type=float,
        default=612.2821044921875,
    )

    live_application(parser.parse_args())
