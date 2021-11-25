import os
import cv2
import numpy as np
import json

HUAWEI_DATASET_PATH = "/mnt/hand_data"


def world_to_camera_frame(P, R, T):
  """
  Convert points from world to camera coordinates

  Args
    P: Nx3 3d points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    X_cam: Nx3 3d points in camera coordinates
  """

  assert len(P.shape) == 2
  assert P.shape[1] == 3

  X_cam = R.dot(P.T)   + T # rotate and translate

  return X_cam.T

def convert_cam2img(samples, K):

    K = np.asarray(K)
    samples = np.asarray(samples)
    samples = samples.reshape((-1,21,3))
    res = []
    # print(samples.shape)
    for cam_points in samples:
        img_points = []
        for cam_point in cam_points:
            img_point = np.dot(K, cam_point)
            # print(img_point.shape)
            img_point[0] = img_point[0] / img_point[2]
            img_point[1] = img_point[1] / img_point[2]
            img_points.append(img_point[:2])
        res.append(img_points)
    res = np.asarray(res)
    # print(res.shape)
    return res

# get camera_params
def get_calibration_matrices():
    """
    Returns: 
        dict (subject, seq, camera) to intrinsic camera matrix
    """
    calibs = {}
    for subject in range(1,21):
        path = os.path.join(HUAWEI_DATASET_PATH, "Calibration", "Calibration" + "%03d"%subject + ".json")
        with open(path, 'r') as jsonfile:
            camera_dic = json.load(jsonfile)
        for cam in range(4):
            cam_key = "rgb_" + str(cam)
            calibs[(subject, 0, cam)] = np.asarray(camera_dic[cam_key]["intrinsic"])
            # print(calibs[(subject, 0, cam)])
    return calibs

def train_ground_truth(sub, seq):
    """
    Returns the ground truth annotations. Returns a dict with fields 'annot2', 'annot3', 'univ_annot3'
    """
    json_path = os.path.join(HUAWEI_DATASET_PATH, "Annotation", "inroom2", "%03d"%sub, "%03d"%seq + ".json")
    #TODO: if the path not exist continue return null
    if not os.path.exists(json_path):
        return None

    with open(json_path, 'r') as jsonfile:
        pose_dic = json.load(jsonfile)
    pose_3d = []

    for key, value in pose_dic["Point_3d"].items():
        pose_3d.append(value)
    
    camera_param_path = os.path.join(HUAWEI_DATASET_PATH, "Calibration", "Calibration" + "%03d"%sub + ".json")
    with open(camera_param_path, 'r') as jsonfile:
            camera_dic = json.load(jsonfile)

    univ_annot3 = []
    annot3 = []
    annot2 = []

    for cam in range(4):
        cam_key = "rgb_" + str(cam)
        cam_params = camera_dic[cam_key]
        R = np.asarray(cam_params["rot"])
        T = np.asarray(cam_params["tran"])
        K = np.asarray(cam_params["intrinsic"])
        # print(np.asarray(univ_annot3).shape)
        camera_coord = world_to_camera_frame( np.asarray(pose_3d).reshape(-1,3), R, T)
        pts2d = convert_cam2img(camera_coord, K)

        camera_coord = camera_coord.reshape(-1,21,3)
        pts2d = pts2d.reshape(-1,21,2)
        score = np.ones((pts2d.shape[0], pts2d.shape[1], 1))*0.9
        # print(score.shape)
        # print(pts2d[0][0])
        pts2d = np.stack((pts2d[:,:,0], pts2d[:,:,1], score[:,:,0]), axis = 2)
        # print(pts2d.shape)
        # print(pts2d[0][0])
        


        univ_annot3.append(pose_3d)
        annot3.append(camera_coord.tolist())
        # print(camera_coord)
        annot2.append(pts2d.tolist())
    result = {'annot2': annot2, 'annot3': annot3, 'univ_annot3': univ_annot3}
    return result

        # print(camera_coord)

     






if __name__ == "__main__":
    get_calibration_matrices()
    train_ground_truth(1, 1)