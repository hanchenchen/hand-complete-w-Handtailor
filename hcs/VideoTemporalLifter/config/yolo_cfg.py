from yacs.config import CfgNode as CN

_C = CN()
_C.NETWORK = "prn"
_C.CONFIDENCE = 0.25
_C.SIZE = 416
_C.MODEL_PATH =  "/root/workspace/checkpoints/cross-hands-tiny-prn.weights"
_C.CONFIG_PATH = "/root/workspace/checkpoints/cross-hands-tiny-prn.cfg"