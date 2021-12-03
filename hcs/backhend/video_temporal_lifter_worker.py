import numpy as np
import cv2
from utils import Completeness
import PIL.Image as Image

from backhend.video_temporal_lifter_solve import Solver
import torch


def Worker(inputs_queue, output_queue, solver, gesture, left):

    mode = 2 if gesture == "fist" else 1
    while True:
        meta = inputs_queue.get()
        if meta == 'STOP':
            # if solver is not None:
            #     del solver
            # torch.cuda.empty_cache()
            print("Quit Estimate Process.")
            break
        else:
            color, depth, Ks = meta['color'][0], meta['depth'][0], meta['ks'][0]

            _ = solver(color, mode=mode, left=left)

            output = {
                "opt_params": [],
                "vertices": [],
                "hand_joints": [],
                "completeness": 0,
                "angles": _["angle"],
                "mismatchness": 0
            }
            output_queue.put(output)
