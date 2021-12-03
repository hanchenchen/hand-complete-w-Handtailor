import numpy as np
import cv2
from utils import Completeness
import PIL.Image as Image

from backhend.handtailor_solve import Solver
import torch


def Worker(inputs_queue, output_queue, solver, gesture, left):

    init_Rs = None
    init_glb_rot = None
    print(gesture)
    completeness_estimator = Completeness(gesture)
    while True:
        meta = inputs_queue.get()
        if meta == 'STOP':
            print("Quit Estimate Process.")
            break
        else:
            color, depth, Ks = meta['color'][0], meta['depth'][0], meta['ks'][0]

            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            H, W, C = color.shape
            color_left = color[:, :H, :]
            color_right = color[:, -1:-H:-1, :]

            output = {
                "opt_params": [],
                "vertices": [],
                "hand_joints": [],
            }
            glb_rot = []
            Rs = []
            for i, img in enumerate([color_left, color_right]):
                _ = solver(img, Ks, i)
                output["opt_params"].append(_["opt_params"])
                output["vertices"].append(_["vertices"])
                output["hand_joints"].append(_["hand_joints"])

                glb_rot.append(_["glb_rot"])
                Rs.append(_["Rs"])
            for key in output.keys():
                output[key] = np.concatenate(output[key], 0)
            glb_rot = np.concatenate(glb_rot, 0)
            Rs = np.concatenate(Rs, 0)
            if init_Rs is None:
                init_Rs = Rs
                init_glb_rot = glb_rot
                output.update({
                    "completeness": 0,
                    "angles": [0, 0],
                    "mismatchness": 0
                })
            else:
                glb_rot = np.concatenate((init_glb_rot, glb_rot), 0)
                Rs = np.concatenate((init_Rs, Rs), 0)
                completeness, sickside_angle, goodside_angle = completeness_estimator(glb_rot.astype('float32'),
                                                                                      Rs.reshape(4, 15, 3, 3).astype('float32'),
                                                                                      left)
                output.update({
                    "completeness": completeness,
                    "angles": [sickside_angle, goodside_angle],
                    "mismatchness": 0
                })
            output_queue.put(output)
