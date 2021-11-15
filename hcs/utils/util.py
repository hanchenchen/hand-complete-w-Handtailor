# from plyfile import PlyData, PlyElement
import numpy as np
import cv2


wrist_vert_idxs = [78, 79, 108, 120, 119, 117, 118, 122, 38,
                   92, 234, 239, 279, 215, 214, 121]


# def write_ply(save_path, points, text=True):
#     """
#     save_path: path to save: '/yy/XX.ply'
#     pt: pointcloud: size(N, 3)
#     """
#     points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
#     vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
#     el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
#     PlyData([el], text=text).write(save_path)
# 
# 
# def read_ply(filename):
#     """ read XYZ point cloud from filename PLY file """
#     plydata = PlyData.read(filename)
#     pc = plydata['vertex'].data
#     pc_array = np.array([[x, y, z] for x,y,z in pc])
#     return pc_array
# 
# 
# def export_ply(verts, file, faces):
# 
#     verts = [tuple(v) for v in verts]
#     verts = np.array(verts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
# 
#     faces = [(f,) + (f[0] / 778 * 255, f[1] / 778 * 255, f[2] / 778 * 255) for f in faces.tolist()]
#     faces = np.array(faces, dtype=[('vertex_indices', 'i4', (3,)),
#                                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
# 
#     elements = [PlyElement.describe(verts, 'vertex'), PlyElement.describe(faces, 'face')]
# 
#     plydata = PlyData(elements, text=True)
#     plydata.write(file)


def add_arm_vertices(hand_mesh, faces=None, offset=5, return_faces=True):
    wrist_verts = hand_mesh[wrist_vert_idxs, :]
    extra_verts = wrist_verts + (hand_mesh[117, :] - hand_mesh[34, :]) * offset
    new_verts = np.concatenate((hand_mesh, extra_verts), 0)
    if return_faces:
        extra_verts_idx = [len(hand_mesh) + i for i in range(len(wrist_vert_idxs))]
        new_faces = []
        for i in range(len(wrist_vert_idxs) - 1):
            new_faces.append(extra_verts_idx[i:i+2] + [wrist_vert_idxs[i]])
            new_faces.append([extra_verts_idx[i+1]] + wrist_vert_idxs[i:i+2])
        new_faces.append([extra_verts_idx[-1], extra_verts_idx[0], wrist_vert_idxs[-1]])
        new_faces.append([extra_verts_idx[0], wrist_vert_idxs[-1], wrist_vert_idxs[0]])
        new_faces = np.array(new_faces).astype(np.int)
        new_faces = np.concatenate((faces, new_faces), 0)
        return new_verts, new_faces
    else:
        return new_verts, extra_verts


class OneEuroFilter(object):
    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
    
    def compute_alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def process(self, x):
        prev_x = self.x_filter.prev_raw_value
        dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
        edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
        cutoff = self.mincutoff + self.beta * np.abs(edx)
        return self.x_filter.process(x, self.compute_alpha(cutoff))


class LowPassFilter(object):
    def __init__(self):
        self.prev_raw_value = None
        self.prev_filtered_value = None

    def process(self, value, alpha):
        if self.prev_raw_value is None:
            s = value
        else:
            s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
        self.prev_raw_value = value
        self.prev_filtered_value = s
        return s

