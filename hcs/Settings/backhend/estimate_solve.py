"MANO New hand model Fitting"
import torch
import numpy as np
from collections import OrderedDict
from .MANO_NEW import MANO_NEW
from utils import Constraints
from utils.loss import chamfer_loss
from utils.optimizer import create_optimizer
from utils.quaternion import qmul
torch.set_num_threads(1)


# valid2full = [19, 20, 0, 21, 22, 1, 23, 24, 2,  # index
#               25, 26, 3, 27, 28, 4, 29, 30, 5,  # middle
#               31, 32, 6, 33, 34, 7, 35, 36, 8,  # pinky
#               37, 38, 9, 39, 40, 10, 41, 42, 11,  # ring
#               12, 13, 14, 15, 43, 16, 44, 17, 18,  # thumb
#               ]
valid2full = [13, 14, 0, 15, 16, 1, 17, 18, 2,  # index
              19, 20, 3, 21, 22, 4, 23, 24, 5,  # middle
              25, 26, 6, 27, 28, 7, 29, 30, 8,  # pinky
              31, 32, 9, 33, 34, 10, 35, 36, 11,  # ring
              12, 37, 38, 39, 40, 41, 42, 43, 44,  # thumb
              ]

mano2cmu = [
    0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20
]


class Solver(object):

    def __init__(self,
                 init_params,
                 step_size=0.1,
                 num_iters=5,
                 threshold=1e-4,
                 w_poseprior=1.0,
                 w_pointcloud=1000.0,
                 w_silhouette=1.0,
                 w_reprojection=0.01,
                 w_temporalprior=1.0,
                 lefthand='./.cache/MANO_LEFT.pkl',
                 righthand='./.cache/MANO_RIGHT.pkl',
                 verbose=False,
                 use_pcaprior=True,
                 temporalprior="params"):
        self.step_size = step_size
        self.num_iters = num_iters
        self.threshold = threshold
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.constraints_l = Constraints(right=False)
        self.constraints_r = Constraints(right=True)

        self.user_shape = init_params[:, 45:55]
        self.init_quat = init_params[:, 55:59]
        self.init_trans = init_params[:, 59:62]
        self.lefthand_model = MANO_NEW(lefthand, self.user_shape[:1], self.init_quat[:1], self.init_trans[:1]).to(self.device)
        self.righthand_model = MANO_NEW(righthand, self.user_shape[1:], self.init_quat[1:], self.init_trans[1:]).to(self.device)
        self.use_pcaprior = use_pcaprior
        self.pprevious_pose_l = None
        self.previous_pose_l = None
        self.pprevious_pose_r = None
        self.previous_pose_r = None

        # weights of loss
        self.w_poseprior = w_poseprior
        self.w_pointcloud = w_pointcloud
        self.w_silhouette = w_silhouette
        self.w_reprojection = w_reprojection
        self.w_temporalprior = w_temporalprior

        self.verbose = verbose
        self.optimizer = None
        self.init_Rs = None
        self.init_glb_rot = None
        self.count = 0

    def compute_loss(self, hand_mesh, pointcloud, pose_param, constraints, pred_silhouette=None, gt_silhouette=None, isfist=True):
        if not isinstance(pointcloud, torch.Tensor):
            pointcloud = torch.from_numpy(pointcloud).float().to(hand_mesh.device)
        if len(pointcloud.shape) == 2:
            pointcloud = pointcloud.unsqueeze(0)
        # pose prior
        if not self.use_pcaprior:
            pose_prior = constraints.getAnglePoseConstraints(pose_param)
        else:
            pose_prior = constraints.getPCAPoseConstraints(pose_param)
        # point cloud matching
        loss_pc = chamfer_loss(hand_mesh.cuda(), pointcloud.cuda()).cpu()
        # silhouette matching
        # total_loss = self.w_pointcloud * loss_pc + self.w_poseprior * pose_prior
        total_loss = 1000.0 * pose_prior if isfist else self.w_pointcloud * loss_pc + self.w_poseprior * pose_prior

        return total_loss

    def __call__(self, learnable_params, item, ks, gesture, optimizer=None):
        # target
        pointcloud_l = item['pointcloud_l']
        pointcloud_r = item['pointcloud_r']
        pose_param_l, quat_param_l, pose_param_r, quat_param_r = learnable_params
        if self.previous_pose_l is None:
            self.pprevious_pose_l = pose_param_l.clone().detach()
            self.pprevious_pose_r = pose_param_r.clone().detach()
        else:
            self.pprevious_pose_l = self.previous_pose_l
            self.pprevious_pose_r = self.previous_pose_r
        self.previous_pose_l = pose_param_l.clone().detach()
        self.previous_pose_r = pose_param_r.clone().detach()

        # Fit the model using pointcloud
        if self.optimizer is None:
            self.optimizer = create_optimizer(learnable_params, "adam", self.step_size)
        if gesture == "fist":
            gt_joints_uv_l = torch.from_numpy(item['hand_joints_l']).float().to(self.device)
            gt_joints_uv_r = torch.from_numpy(item['hand_joints_r']).float().to(self.device)

        self.previous_loss = -np.inf
        for i in range(self.num_iters):
            if gesture == "fist":
                quat_param_l_focus = quat_param_l.contiguous()
                quat_param_r_focus = quat_param_r.contiguous()
                # pose_param_l_focus = torch.cat((pose_param_l, torch.zeros((1, 32)).to(pose_param_l.device)), 1)
                # pose_param_l_focus = pose_param_l_focus[:, valid2full].contiguous()
                # pose_param_r_focus = torch.cat((pose_param_r, torch.zeros((1, 32)).to(pose_param_r.device)), 1)
                # pose_param_r_focus = pose_param_r_focus[:, valid2full].contiguous()
                pose_param_l_focus = self.lefthand_model.get_pcacoff_theta(pose_param_l.contiguous())
                pose_param_r_focus = self.righthand_model.get_pcacoff_theta(pose_param_r.contiguous())
            elif gesture == "wristbackward" or gesture == "wristforward":
                quat_param_l_focus = torch.cat((quat_param_l, torch.zeros((1, 2)).to(quat_param_l.device)), 1)
                quat_param_r_focus = torch.cat((quat_param_r, torch.zeros((1, 2)).to(quat_param_r.device)), 1)
                pose_param_l_focus = pose_param_l.contiguous()
                pose_param_r_focus = pose_param_r.contiguous()
            else:
                quat_param_l_focus = torch.cat((quat_param_l, torch.zeros((1, 2)).to(quat_param_l.device)), 1)
                quat_param_l_focus = quat_param_l_focus[:, [0, 2, 3, 1]].contiguous()
                quat_param_r_focus = torch.cat((quat_param_r, torch.zeros((1, 2)).to(quat_param_r.device)), 1)
                quat_param_r_focus = quat_param_r_focus[:, [0, 2, 3, 1]].contiguous()
                pose_param_l_focus = pose_param_l.contiguous()
                pose_param_r_focus = pose_param_r.contiguous()
            hand_mesh_l, joints_normed_l, Rs_l = self.lefthand_model(pose_param_l_focus,
                                                                     quat_param_l_focus,
                                                                     get_skin=True,
                                                                     use_pca=True if gesture == "fist" else False)
            hand_mesh_r, joints_normed_r, Rs_r = self.righthand_model(pose_param_r_focus,
                                                                      quat_param_r_focus,
                                                                      get_skin=True,
                                                                      use_pca=True if gesture == "fist" else False)

            # Hand joints
            joints_normed_l = joints_normed_l[:, mano2cmu, :]
            joints_normed_r = joints_normed_r[:, mano2cmu, :]
            pred_joints_uv_l = projectPoints_batch(joints_normed_l, ks)
            pred_joints_uv_r = projectPoints_batch(joints_normed_r, ks)


            # if not self.use_pcaprior:
            #     loss_l = self.compute_loss(hand_mesh_l, pointcloud_l, pose_param_l_focus)
            #     loss_r = self.compute_loss(hand_mesh_r, pointcloud_r, pose_param_r_focus)
            # else:
            #     loss_l = self.compute_loss(hand_mesh_l, pointcloud_l,
            #                                self.lefthand_model.get_pcacoff_theta(pose_param_l_focus))
            #     loss_r = self.compute_loss(hand_mesh_r, pointcloud_r,
            #                                self.righthand_model.get_pcacoff_theta(pose_param_r_focus))
            if gesture == "fist":
                if not self.use_pcaprior:
                    loss_l = self.compute_loss(hand_mesh_l, pointcloud_l, self.lefthand_model.get_rotvec_theta(pose_param_l_focus), self.constraints_l)
                    loss_r = self.compute_loss(hand_mesh_r, pointcloud_r, self.righthand_model.get_rotvec_theta(pose_param_r_focus), self.constraints_r)
                else:
                    loss_l = self.compute_loss(hand_mesh_l, pointcloud_l, pose_param_l_focus, self.constraints_l)
                    loss_r = self.compute_loss(hand_mesh_r, pointcloud_r, pose_param_r_focus, self.constraints_r)
                # Reprojection
                reprojection_loss_l = gmof(gt_joints_uv_l.unsqueeze(0) - pred_joints_uv_l, sigma=100).sum(dim=-1)
                reprojection_loss_r = gmof(gt_joints_uv_r.unsqueeze(0) - pred_joints_uv_r, sigma=100).sum(dim=-1)
                # temporal prior
                temporal_prior_l = self.constraints_l.getTemporalConstraints(torch.cat((self.pprevious_pose_l, self.previous_pose_l, pose_param_l_focus), 0))
                temporal_prior_r = self.constraints_l.getTemporalConstraints(torch.cat((self.pprevious_pose_r, self.previous_pose_r, pose_param_r_focus), 0))
                loss = loss_l + 0.1 * reprojection_loss_l.sum(dim=-1).sum() + self.w_temporalprior * temporal_prior_l + \
                    loss_r + 0.1 * reprojection_loss_r.sum(dim=-1).sum() + self.w_temporalprior * temporal_prior_r
            else:
                loss_l = self.compute_loss(hand_mesh_l, pointcloud_l,
                                           self.lefthand_model.get_pcacoff_theta(pose_param_l_focus),
                                           self.constraints_l, isfist=False)
                loss_r = self.compute_loss(hand_mesh_r, pointcloud_r,
                                           self.righthand_model.get_pcacoff_theta(pose_param_r_focus),
                                           self.constraints_r, isfist=False)
                loss = loss_l + loss_r

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.verbose:
                print("Stage 2 Iteration {}, Fitting Loss: {:.4f}".format(i, loss.detach()))

            if abs(loss.detach().item() - self.previous_loss) < self.threshold:
                break
            self.previous_loss = loss.detach().item()

        output = OrderedDict()
        if gesture == "wristside":
            quat_param_l_opt = torch.cat((quat_param_l, torch.zeros((1, 2)).to(quat_param_l.device)), 1)
            quat_param_r_opt = torch.cat((quat_param_r, torch.zeros((1, 2)).to(quat_param_r.device)), 1)
            quat_param_l_opt = quat_param_l_opt[:, [0, 2, 3, 1]].contiguous()
            quat_param_r_opt = quat_param_r_opt[:, [0, 2, 3, 1]].contiguous()
            pose_param_l_opt = pose_param_l.contiguous()
            pose_param_r_opt = pose_param_r.contiguous()
        elif gesture == "wristbackward" or gesture == "wristforward":
            quat_param_l_opt = torch.cat((quat_param_l, torch.zeros((1, 2)).to(quat_param_l.device)), 1)
            quat_param_r_opt = torch.cat((quat_param_r, torch.zeros((1, 2)).to(quat_param_r.device)), 1)
            pose_param_l_opt = pose_param_l.contiguous()
            pose_param_r_opt = pose_param_r.contiguous()
        else:
            # pose_param_l_opt = torch.cat((pose_param_l, torch.zeros((1, 32)).to(pose_param_l.device)), 1)
            # pose_param_r_opt = torch.cat((pose_param_r, torch.zeros((1, 32)).to(pose_param_r.device)), 1)
            # pose_param_l_opt = pose_param_l_opt[:, valid2full].contiguous()
            # pose_param_r_opt = pose_param_r_opt[:, valid2full].contiguous()
            pose_param_l_opt = pose_param_l.contiguous()
            pose_param_r_opt = pose_param_r.contiguous()
            quat_param_l_opt = quat_param_l.contiguous()
            quat_param_r_opt = quat_param_r.contiguous()

        quat_param_l_all = qmul(quat_param_l_opt, torch.from_numpy(self.init_quat[:1]).to(quat_param_l_opt.device))
        opt_param_l = np.concatenate((pose_param_l_opt.detach().cpu().numpy(), self.user_shape[:1],
                                      quat_param_l_all.detach().cpu().numpy(), self.init_trans[:1]), 1)
        quat_param_r_all = qmul(quat_param_r_opt, torch.from_numpy(self.init_quat[1:]).to(quat_param_r_opt.device))
        opt_param_r = np.concatenate((pose_param_r_opt.detach().cpu().numpy(), self.user_shape[1:],
                                      quat_param_r_all.detach().cpu().numpy(), self.init_trans[1:]), 1)
        # opt_param_l = torch.cat((pose_param_l_opt, quat_param_l_opt), 1).detach().cpu().numpy()  # 1 x 49
        # opt_param_r = torch.cat((pose_param_r_opt, quat_param_r_opt), 1).detach().cpu().numpy()  # 1 x 49
        hand_joints = torch.cat((pred_joints_uv_l, pred_joints_uv_r), 0).detach().cpu().numpy()
        Rs = torch.cat((Rs_l, Rs_r), 0).detach().cpu().numpy()  # 2 x 15 x 3 x 3
        glb_rot = torch.cat((quat_param_l_opt, quat_param_r_opt), 0).detach().cpu().numpy()
        if self.init_Rs is None or self.init_glb_rot is None or self.count < 10:
            self.init_Rs = Rs
            self.init_glb_rot = glb_rot
            self.count += 1
        output['opt_params'] = np.concatenate((opt_param_l, opt_param_r), 0)  # 2 x 49
        output['vertices'] = torch.cat((hand_mesh_l, hand_mesh_r), 1).detach().cpu().numpy()
        output['hand_joints'] = hand_joints
        output['Rs'] = np.concatenate((self.init_Rs, Rs), 0)  # 4 x 15 x 3 x 3
        output['glb_rot'] = np.concatenate((self.init_glb_rot, glb_rot), 0)

        return output


def projectPoints_batch(xyz, K, camera_R=None, eps=1e-9):
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
    if camera_R is not None:
        xyz = torch.matmul(xyz, camera_R)
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
