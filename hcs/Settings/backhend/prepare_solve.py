"""
Hand Model Fitting
"""
import torch
import numpy as np
from collections import OrderedDict
from .MANO_SMPL import MANO_SMPL
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
                 step_size=0.01,
                 num_iters=100,
                 threshold=1e-4,
                 w_poseprior=1.0,
                 w_shapeprior=1.0,
                 w_pointcloud=1.0,
                 w_reprojection=1.0,
                 w_silhouette=1.0,
                 lefthand='./.cache/MANO_LEFT.pkl',
                 righthand='./.cache/MANO_RIGHT.pkl',
                 verbose=False,
                 fit_camera=False,
                 use_pcaprior=True):
        self.step_size = step_size
        self.num_iters = num_iters
        self.threshold = threshold
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.lefthand_model = MANO_SMPL(lefthand).to(self.device)
        self.constraints_l = Constraints(right=False)
        self.constraints_r = Constraints(right=True)
        self.righthand_model = MANO_SMPL(righthand).to(self.device)
        self.fit_camera = fit_camera
        self.use_pcaprior = use_pcaprior

        # weights of loss
        self.w_poseprior = w_poseprior
        self.w_shapeprior = w_shapeprior
        self.w_pointcloud = w_pointcloud
        self.w_reprojection = w_reprojection
        self.w_silhouette = w_silhouette

        self.faces = self.lefthand_model.faces.astype(np.int)
        self.verbose = verbose

    def compute_loss(self, hand_mesh, pointcloud, pose_param, shape_param, constraints):
        """
        Compute object function for model fitting
        """
        if not isinstance(pointcloud, torch.Tensor):
            pointcloud = torch.from_numpy(pointcloud).float().to(hand_mesh.device)
        if len(pointcloud.shape) == 2:
            pointcloud = pointcloud.unsqueeze(0)

        # pose prior
        if self.use_pcaprior:
            pose_prior = constraints.getPCAPoseConstraints(pose_param)
        else:
            pose_prior = constraints.getAnglePoseConstraints(pose_param)
        # shape prior
        shape_prior = constraints.getPCAPoseConstraints(shape_param)
        # point cloud matching
        loss_pc = chamfer_loss(hand_mesh.cuda(), pointcloud.cuda()).cpu()

        total_loss = self.w_poseprior * pose_prior + self.w_shapeprior * shape_prior + self.w_pointcloud * loss_pc

        return total_loss

    def __call__(self, learnable_params, item, ks, optimizer=None, method=0):
        # target
        pointcloud_l = item['pointcloud_l']
        pointcloud_r = item['pointcloud_r']

        pose_param_l, shape_param_l, quat_param_l, trans_param_l = learnable_params[:4]
        pose_param_r, shape_param_r, quat_param_r, trans_param_r = learnable_params[4:8]
        if method != 0:
            glb_quat, glb_trans = learnable_params[-2:]

        if 'hand_joints_l' in item:
            gt_joints_uv_l = item['hand_joints_l']
            if isinstance(gt_joints_uv_l, np.ndarray):
                gt_joints_uv_l = torch.from_numpy(gt_joints_uv_l).float().to(self.device)
        if 'hand_joints_r' in item:
            gt_joints_uv_r = item['hand_joints_r']
            if isinstance(gt_joints_uv_r, np.ndarray):
                gt_joints_uv_r = torch.from_numpy(gt_joints_uv_r).float().to(self.device)

        optimizer = create_optimizer(learnable_params, "adam", lr=self.step_size)

        previous_loss = -np.inf
        for i in range(self.num_iters):
            if pose_param_l.shape[-1] == 5 and pose_param_r.shape[-1] == 5:
                pose_param_l_focus = torch.cat((pose_param_l, torch.zeros((1, 40)).to(pose_param_l.device)), 1)
                pose_param_l_focus = pose_param_l_focus[:, valid2full]
                pose_param_r_focus = torch.cat((pose_param_r, torch.zeros((1, 40)).to(pose_param_r.device)), 1)
                pose_param_r_focus = pose_param_r_focus[:, valid2full]
            else:
                pose_param_l_focus = pose_param_l.contiguous()
                pose_param_r_focus = pose_param_r.contiguous()
            
            # 限制旋转和平移
            if method != 0:
                quat_param_l_focus = torch.cat((quat_param_l, torch.zeros((1, 2)).to(quat_param_l.device)), 1)
                quat_param_l_focus = quat_param_l_focus[:, [0, 2, 1, 3]]
                quat_param_l_focus = qmul(glb_quat, quat_param_l_focus)
                quat_param_r_focus = torch.cat((quat_param_r, torch.zeros((1, 2)).to(quat_param_r.device)), 1)
                quat_param_r_focus = quat_param_r_focus[:, [0, 2, 1, 3]]
                quat_param_r_focus = qmul(glb_quat, quat_param_r_focus)
                trans_param_l_focus = torch.cat((trans_param_l, torch.zeros((1, 2)).to(trans_param_l.device)), 1) + glb_trans
                trans_param_r_focus = torch.cat((trans_param_r, torch.zeros((1, 2)).to(trans_param_r.device)), 1) + glb_trans
            else:
                quat_param_l_focus = quat_param_l.contiguous()
                quat_param_r_focus = quat_param_r.contiguous()
                trans_param_l_focus = trans_param_l.contiguous()
                trans_param_r_focus = trans_param_r.contiguous()

            hand_mesh_l, joints_normed_l, Rs_l = self.lefthand_model(shape_param_l,
                                                                     pose_param_l_focus,
                                                                     quat_param_l_focus,
                                                                     get_skin=True,
                                                                     use_pca=False)
            hand_mesh_r, joints_normed_r, Rs_r = self.righthand_model(shape_param_r,
                                                                      pose_param_r_focus,
                                                                      quat_param_r_focus,
                                                                      get_skin=True,
                                                                      use_pca=False)
            joints_normed_l = joints_normed_l[:, mano2cmu, :]
            joints_normed_r = joints_normed_r[:, mano2cmu, :]

            trans_l = trans_param_l_focus.view(joints_normed_l.shape[0], 1, -1)
            trans_r = trans_param_r_focus.view(joints_normed_r.shape[0], 1, -1)

            joints_normed_l = joints_normed_l + trans_l
            joints_normed_r = joints_normed_r + trans_r
            hand_mesh_l = hand_mesh_l + trans_l
            hand_mesh_r = hand_mesh_r + trans_r

            pred_joints_uv_l = projectPoints_batch(joints_normed_l, ks)
            reprojection_loss_l = gmof(gt_joints_uv_l.unsqueeze(0) - pred_joints_uv_l, sigma=100).sum(dim=-1)
            pred_joints_uv_r = projectPoints_batch(joints_normed_r, ks)
            reprojection_loss_r = gmof(gt_joints_uv_r.unsqueeze(0) - pred_joints_uv_r, sigma=100).sum(dim=-1)

            if self.use_pcaprior:
                loss_l = self.compute_loss(hand_mesh_l, pointcloud_l,
                                           self.lefthand_model.get_pcacoff_theta(pose_param_l_focus), shape_param_l,
                                           self.constraints_l)
                loss_r = self.compute_loss(hand_mesh_r, pointcloud_r,
                                           self.righthand_model.get_pcacoff_theta(pose_param_r_focus), shape_param_r,
                                           self.constraints_r)
            else:
                loss_l = self.compute_loss(hand_mesh_l, pointcloud_l, pose_param_l_focus, shape_param_l,
                                           self.constraints_l)
                loss_r = self.compute_loss(hand_mesh_r, pointcloud_r, pose_param_r_focus, shape_param_r,
                                           self.constraints_r)

            loss = loss_l + self.w_reprojection * reprojection_loss_l.sum(dim=-1).sum() + \
                loss_r + self.w_reprojection * reprojection_loss_r.sum(dim=-1).sum()
            

            if self.verbose:
                print("Iteration {}, Fitting Loss: {:.4f}".format(i, loss.detach().item()))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if abs(loss.detach().item() - previous_loss) < self.threshold:
                break
            previous_loss = loss.detach().item()

        # export_ply(hand_mesh_l.detach().cpu().numpy()[0], "./.cache/right.ply", self.faces)
        output = OrderedDict()
        if method == 0:
            opt_param_l = torch.cat([pose_param_l_focus, shape_param_l, quat_param_l_focus, trans_param_l_focus], 1).detach().cpu().numpy()
            opt_param_r = torch.cat([pose_param_r_focus, shape_param_r, quat_param_r_focus, trans_param_r_focus], 1).detach().cpu().numpy()
        else:
            opt_param_l = torch.cat([pose_param_l_focus, shape_param_l, quat_param_l_focus, trans_param_l_focus, glb_quat, glb_trans], 1).detach().cpu().numpy()
            opt_param_r = torch.cat([pose_param_r_focus, shape_param_r, quat_param_r_focus, trans_param_r_focus, glb_quat, glb_trans], 1).detach().cpu().numpy()
        output['opt_params'] = np.concatenate((opt_param_l, opt_param_r), 0)
        hand_mesh_l, extra_verts_l = add_arm_vertices(hand_mesh_l.detach().cpu().numpy()[0], return_faces=False)
        hand_mesh_r, extra_verts_r = add_arm_vertices(hand_mesh_r.detach().cpu().numpy()[0], return_faces=False)
        output['vertices'] = np.concatenate((hand_mesh_l, hand_mesh_r), 0)
        output['extra_verts'] = np.concatenate((extra_verts_l, extra_verts_r), 0)
        # output['vertices'] = torch.cat([hand_mesh_l, hand_mesh_r], 1).detach().cpu().numpy()

        return output


    # def __call__(self, learnable_params, item, ks, optimizer=None):
    #     # target
    #     pointcloud_l = item['pointcloud_l']
    #     pointcloud_r = item['pointcloud_r']

    #     pose_param_l, shape_param_l, quat_param_l, cam_param_l = learnable_params[:4]
    #     pose_param_r, shape_param_r, quat_param_r, cam_param_r = learnable_params[4:]
    #     if 'hand_joints_l' in item:
    #         gt_joints_uv_l = item['hand_joints_l']
    #         if isinstance(gt_joints_uv_l, np.ndarray):
    #             gt_joints_uv_l = torch.from_numpy(gt_joints_uv_l).float().to(self.device)
    #     if 'hand_joints_r' in item:
    #         gt_joints_uv_r = item['hand_joints_r']
    #         if isinstance(gt_joints_uv_r, np.ndarray):
    #             gt_joints_uv_r = torch.from_numpy(gt_joints_uv_r).float().to(self.device)

    #     if self.fit_camera:
    #         camera_opt_params = [quat_param_l, cam_param_l, quat_param_r, cam_param_r]
    #         camera_optimizer = create_optimizer(camera_opt_params, "adam", lr=self.step_size)

    #         previous_loss = -np.inf
    #         for i in range(self.num_iters):
    #             pose_param_l_focus = torch.cat((pose_param_l, torch.zeros((1, 40)).to(pose_param_l.device)), 1)
    #             pose_param_l_focus = pose_param_l_focus[:, valid2full]
    #             pose_param_r_focus = torch.cat((pose_param_r, torch.zeros((1, 40)).to(pose_param_r.device)), 1)
    #             pose_param_r_focus = pose_param_r_focus[:, valid2full]
    #             hand_mesh_l, joints_normed_l, _ = self.lefthand_model(shape_param_l,
    #                                                                   pose_param_l_focus,
    #                                                                   quat_param_l,
    #                                                                   get_skin=True,
    #                                                                   use_pca=False)
    #             hand_mesh_r, joints_normed_r, _ = self.righthand_model(shape_param_r,
    #                                                                    pose_param_r_focus,
    #                                                                    quat_param_r,
    #                                                                    get_skin=True,
    #                                                                    use_pca=False)
    #             joints_normed_l = joints_normed_l[:, mano2cmu, :]
    #             joints_normed_r = joints_normed_r[:, mano2cmu, :]

    #             trans_l = cam_param_l.view(joints_normed_l.shape[0], 1, -1)
    #             trans_r = cam_param_r.view(joints_normed_r.shape[0], 1, -1)

    #             joints_normed_l = joints_normed_l + trans_l
    #             joints_normed_r = joints_normed_r + trans_r

    #             pred_joints_uv_l = projectPoints_batch(joints_normed_l, ks)
    #             reprojection_loss_l = gmof(gt_joints_uv_l.unsqueeze(0) - pred_joints_uv_l, sigma=75).sum(dim=-1)
    #             pred_joints_uv_r = projectPoints_batch(joints_normed_r, ks)
    #             reprojection_ross_r = gmof(gt_joints_uv_r.unsqueeze(0) - pred_joints_uv_r, sigma=75).sum(dim=-1)

    #             loss = self.w_reprojection * reprojection_loss_l.sum(dim=-1).sum() + \
    #                 self.w_reprojection * reprojection_ross_r.sum(dim=-1).sum()

    #             if self.verbose:
    #                 print("Stage 1 Iteration {}, Fitting Loss: {:.4f}".format(i, loss.detach().item()))

    #             camera_optimizer.zero_grad()
    #             loss.backward(retain_graph=True)
    #             camera_optimizer.step()
    #             if abs(loss.detach().item() - previous_loss) < self.threshold:
    #                 break
    #             previous_loss = loss.detach().item()

    #     # Fit the full model
    #     if optimizer is None:
    #         optimizer = create_optimizer(learnable_params, "adam", self.step_size)

    #     self.previous_loss = -np.inf
    #     for i in range(self.num_iters):
    #         pose_param_l_focus = torch.cat((pose_param_l, torch.zeros((1, 40)).to(pose_param_l.device)), 1)
    #         pose_param_l_focus = pose_param_l_focus[:, valid2full]
    #         pose_param_r_focus = torch.cat((pose_param_r, torch.zeros((1, 40)).to(pose_param_r.device)), 1)
    #         pose_param_r_focus = pose_param_r_focus[:, valid2full]
    #         hand_mesh_l, joints_normed_l, Rs_l = self.lefthand_model(shape_param_l,
    #                                                                  pose_param_l_focus,
    #                                                                  quat_param_l,
    #                                                                  get_skin=True,
    #                                                                  use_pca=False)
    #         hand_mesh_r, joints_normed_r, Rs_r = self.righthand_model(shape_param_r,
    #                                                                   pose_param_r_focus,
    #                                                                   quat_param_r,
    #                                                                   get_skin=True,
    #                                                                   use_pca=False)

    #         trans_l = cam_param_l.view(joints_normed_l.shape[0], 1, -1)
    #         trans_r = cam_param_r.view(joints_normed_r.shape[0], 1, -1)

    #         hand_mesh_l = hand_mesh_l + trans_l
    #         hand_mesh_r = hand_mesh_r + trans_r

    #         if self.use_pcaprior:
    #             loss_l = self.compute_loss(hand_mesh_l, pointcloud_l,
    #                                        self.lefthand_model.get_pcacoff_theta(pose_param_l_focus), shape_param_l)
    #             loss_r = self.compute_loss(hand_mesh_r, pointcloud_r,
    #                                        self.righthand_model.get_pcacoff_theta(pose_param_r_focus), shape_param_r)
    #         else:
    #             loss_l = self.compute_loss(hand_mesh_l, pointcloud_l, pose_param_l_focus, shape_param_l)
    #             loss_r = self.compute_loss(hand_mesh_r, pointcloud_r, pose_param_r_focus, shape_param_r)
    #         loss = loss_l + loss_r

    #         optimizer.zero_grad()
    #         loss.backward(retain_graph=True)
    #         optimizer.step()

    #         if self.verbose:
    #             print('Stage 2 Iteration {}, Fitting Loss: {:.4f}'.format(i, loss.detach()))

    #         if abs(loss.detach().item() - self.previous_loss) < self.threshold:
    #             break
    #         self.previous_loss = loss.detach().item()

    #     output = OrderedDict()
    #     opt_param_l = np.concatenate([pose_param_l_focus.detach().cpu().numpy(), shape_param_l.detach().cpu().numpy(),
    #                                   quat_param_l.detach().cpu().numpy(), cam_param_l.detach().cpu().numpy()], 1)  # 1 x 62
    #     opt_param_r = np.concatenate([pose_param_r_focus.detach().cpu().numpy(), shape_param_r.detach().cpu().numpy(),
    #                                   quat_param_r.detach().cpu().numpy(), cam_param_r.detach().cpu().numpy()], 1)  # 1 x 62
    #     output['opt_params'] = np.concatenate((opt_param_l, opt_param_r), 0)  # 2 x 62
    #     output['vertices'] = np.concatenate((hand_mesh_l.detach().cpu().numpy(),
    #                                          hand_mesh_r.detach().cpu().numpy()), 1)
    #     # faces = np.concatenate((self.faces, self.faces + 778), 0)
    #     # export_ply(output['vertices'][0], "./.cache/prepare_twohands.ply", faces)

    #     return output


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
