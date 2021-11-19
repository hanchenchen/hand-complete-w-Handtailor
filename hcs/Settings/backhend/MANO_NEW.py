import torch
import numpy as np
import torch.nn as nn
import pickle
from torch.autograd import Variable
import torch.nn.functional as F
from utils.quaternion import qmul


class MANO_NEW(nn.Module):

    def __init__(self, model_path, user_shape, init_quat, init_trans):
        super(MANO_NEW, self).__init__()
        self.model_path = model_path

        # Load the MANO_RIGHT.pkl
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f, encoding='latin1')

        self.faces = model['f']

        self.register_buffer('user_shape',
                             torch.from_numpy(user_shape))
        self.register_buffer('init_quat',
                             torch.from_numpy(init_quat))
        self.register_buffer('init_trans',
                             torch.from_numpy(init_trans))
        np_v_template = np.array(model['v_template'], dtype=np.float)
        self.register_buffer('v_template',
                             torch.from_numpy(np_v_template).float())
        self.size = [np_v_template.shape[0], 3]

        np_shapedirs = np.array(model['shapedirs'], dtype=np.float)
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs',
                             torch.from_numpy(np_shapedirs).float())

        # Adding new joints for the fingertips. Original MANO model provide only 16 skeleton joints.
        np_J_regressor = model['J_regressor'].T.toarray()
        np_J_addition = np.zeros((778, 5))
        np_J_addition[745][0] = 1
        np_J_addition[333][1] = 1
        np_J_addition[444][2] = 1
        np_J_addition[555][3] = 1
        np_J_addition[672][4] = 1
        np_J_regressor = np.concatenate((np_J_regressor, np_J_addition),
                                        axis=1)
        self.register_buffer('J_regressor',
                             torch.from_numpy(np_J_regressor).float())

        np_hand_component = np.array(model['hands_components'], dtype=np.float)
        np_hand_component_inv = np.linalg.inv(np_hand_component)
        self.np_hand_component = np_hand_component
        self.np_hand_component_inv = np_hand_component_inv
        self.register_buffer('hands_comp',
                             torch.from_numpy(np_hand_component).float())
        self.register_buffer('hands_comp_inv',
                             torch.from_numpy(np_hand_component_inv).float())
        np_hand_mean = np.array(model['hands_mean'], dtype=np.float)
        self.register_buffer('hands_mean',
                             torch.from_numpy(np_hand_mean).float())

        np_posedirs = np.array(model['posedirs'], dtype=np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)

        np_weights = np.array(model['weights'], dtype=np.float)

        vertex_count = np_weights.shape[0]
        vertex_component = np_weights.shape[1]

        self.register_buffer(
            'weight',
            torch.from_numpy(np_weights).float().reshape(
                -1, vertex_count, vertex_component))
        self.register_buffer('e3', torch.eye(3).float())
        self.cur_device = None

    def forward(self, theta, delta_quat, get_skin=False, use_pca=False):
        num_batch = theta.shape[0]
        if num_batch > 1:
            beta = self.user_shape.repeat(num_batch, 1)
        else:
            beta = self.user_shape

        v_shaped = torch.matmul(beta, self.shapedirs).view(
            -1, self.size[0], self.size[1]) + self.v_template

        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        if use_pca:
            num_comps = theta.shape[-1]
            Rs = self.batch_rodrigues(torch.matmul(theta, self.hands_comp[:num_comps]).view(
                -1, 3)).view(-1, 15, 3, 3)
        else:
            Rs = self.batch_rodrigues(theta.view(-1, 3)).view(-1, 15, 3, 3)

        pose_feature = (Rs[:, :, :, :]).sub(1.0, self.e3).view(-1, 135)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(
            -1, self.size[0], self.size[1]) + v_shaped

        init_quat = self.init_quat.repeat(num_batch, 1)
        quat = qmul(delta_quat, init_quat)

        self.J_transformed, A = self.batch_global_rigid_transformation(
            torch.cat([self.quat2mat(quat).view(-1, 1, 3, 3), Rs], dim=1),
            J[:, :16, :],
            self.parents,
            rotate_base=False)

        weight = self.weight.repeat(num_batch, 1, 1)
        W = weight.view(num_batch, -1, 16)
        T = torch.matmul(W, A.view(num_batch, 16, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat(
            [v_posed,
             torch.ones(num_batch, v_posed.shape[1], 1).to(v_posed.device)],
            dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]

        joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)

        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        # Do translation
        trans = self.init_trans.view(verts.shape[0], 1, -1)
        verts = verts + trans
        joints = joints + trans

        if get_skin:
            return verts, joints, Rs
        else:
            return joints

    def get_pcacoff_theta(self, theta):
        pca_coff = torch.matmul(theta, self.hands_comp_inv)
        assert pca_coff.shape[-1] == 45
        return pca_coff

    def get_rotvec_theta(self, theta):
        num_comps = theta.shape[-1]
        rotvec = torch.matmul(theta, self.hands_comp[:num_comps])
        assert rotvec.shape[-1] == 45
        return rotvec

    def quat2mat(self, quat):
        """Convert quaternion coefficients to rotation matrix.
        Args:
            quat: size = [B, 4] 4 <===>(w, x, y, z)
        Returns:
            Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
        """
        norm_quat = quat
        norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
        w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

        B = quat.size(0)

        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        rotMat = torch.stack([
            w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
            2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
            2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2
        ], dim=1).view(B, 3, 3)

        return rotMat

    def batch_rodrigues(self, theta):
        l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
        angle = torch.unsqueeze(l1norm, -1)
        normalized = torch.div(theta, angle)
        angle = angle * 0.5
        v_cos = torch.cos(angle)
        v_sin = torch.sin(angle)
        quat = self.quat2mat(torch.cat([v_cos, v_sin * normalized], dim=1))

        return quat

    def batch_global_rigid_transformation(self,
                                          Rs,
                                          Js,
                                          parent,
                                          rotate_base=False):
        N = Rs.shape[0]
        if rotate_base:
            np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                                dtype=np.float)
            np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
            rot_x = Variable(torch.from_numpy(np_rot_x).float()).to(Rs.device)
            root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
        else:
            root_rotation = Rs[:, 0, :, :]
        Js = torch.unsqueeze(Js, -1)

        def make_A(R, t):
            R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
            t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1)).to(t.device)],
                               dim=1)
            return torch.cat([R_homo, t_homo], 2)

        A0 = make_A(root_rotation, Js[:, 0])
        results = [A0]

        for i in range(1, parent.shape[0]):
            j_here = Js[:, i] - Js[:, parent[i]]
            A_here = make_A(Rs[:, i], j_here)
            res_here = torch.matmul(results[parent[i]], A_here)
            results.append(res_here)

        results = torch.stack(results, dim=1)

        new_J = results[:, :, :3, 3]
        Js_w0 = torch.cat([Js, Variable(torch.zeros(N, 16, 1, 1)).to(Js.device)],
                          dim=2)
        init_bone = torch.matmul(results, Js_w0)
        init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
        A = results - init_bone

        return new_J, A
