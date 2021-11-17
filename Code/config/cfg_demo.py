from yacs.config import CfgNode as CN

_C = CN()

_C.GPUS = [0]
_C.CAMERA = "webcam"
_C.DISPLAY = "2d"

_C.DETECTOR = CN()
_C.DETECTOR.CUDA = True
_C.DETECTOR.INPUT_DIM = 300
_C.DETECTOR.TRAINED_MODEL = "./cache/ssd_new_mobilenet_FFA.pth"
_C.DETECTOR.THRESHOLD = 0.22
_C.DETECTOR.EXPAND_RATIO = 1.7
_C.DETECTOR.OUTPUT_DIM = 224
_C.DETECTOR.LOST_THRESH = 0.6


_C.POSE = CN()
_C.POSE.CUDA = True
_C.POSE.CFG_FILE = "./cache/rhd_cmu_hrnet_224x224/eval_huawei+public_model_on_huawei.yaml"
_C.POSE.TRAINED_MODEL = "./cache/rhd_cmu_hrnet_224x224/huawei+public_model.pth"


_C.BODY_POSE = CN()
_C.BODY_POSE.TRAINED_MODEL = "./checkpoints/lightweight_human_pose.pth"
_C.BODY_POSE.HEIGHT_SIZE = 256
_C.BODY_POSE.CPU = False
_C.BODY_POSE.TRACK = True
_C.BODY_POSE.SMOOTH = 1
_C.BODY_POSE.SIZE = 150

