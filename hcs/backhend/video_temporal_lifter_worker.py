import numpy as np
import cv2
from utils import Completeness
import PIL.Image as Image

from backhend.video_temporal_lifter_solve import Solver


def Worker(inputs_queue, output_queue, proc_id, init_params, hand_joints, extra_verts,
           wrist_rot_pitch, wrist_rot_yaw, gesture,
           height, width, step_size, num_iters,
           threshold, lefthand, righthand,
           w_silhouette, w_pointcloud, w_poseprior, w_reprojection,  w_temporalprior, left, use_pcaprior=True):

    solver = None
    while True:
        meta = inputs_queue.get()
        if meta == 'STOP':
            print("Quit Estimate Process.")
            break
        else:
            color, depth, Ks = meta['color'][0], meta['depth'][0], meta['ks'][0]

            if solver is None:
                solver = Solver(calibration_mx=Ks)
            _ = solver(color)

            output = {
                "opt_params": [],
                "vertices": [],
                "hand_joints": [],
                "completeness": 0,
                "angles": _["angle"],
                "mismatchness": 0
            }
            output_queue.put(output)
