import cv2
import numpy as np
import PIL.Image as Image

from worker.handtailor_solve import Solver


def Worker(input_queue, output_queue, proc_id,
           height, width,
           detect_threshold, detect_inputsize, pose_inputsize,
           verbose, detectmodel_path, step_size, num_iters,
           threshold, lefthand, righthand, w_silhouette,
           w_pointcloud, w_poseprior, w_shapeprior, w_reprojection,
           use_pcaprior=True, init_params=None, usage="getshape", all_pose=False, method=0):

    meta = input_queue.get()
    color, depth, Ks = meta['color'][0], meta['depth'][0], meta['ks'][0]

    color = Image.fromarray(cv2.cvtColor(color,cv2.COLOR_BGR2RGB))
    H, W, C = color.shape

    color_left = color[:, :H, :]
    color_right = color[:, H:, :]
    output = {
        "opt_params": [],
        "vertices": [],
        "extra_verts": [],
        "hand_joints": []
    }
    solver = Solver()
    for i, img in enumerate([color_left, color_right]):
        _ = solver(img, Ks, i)
        output["opt_params"].append(_["opt_params"])
        output["vertices"].append(_["vertices"])
        output["extra_verts"].append(_["extra_verts"])
        output["hand_joints"].append(_["hand_joints"])

    for key in output.keys():
        output[key] = np.stack(output[key], 0)

    output_queue.put(output)