import yaml
import numpy as np
from easydict import EasyDict
from pathlib import Path
from pprint import pprint

def read_config(file):
    """
    Reads configuration from yaml file.
    Args:
        file: Path to configuration file.
    Returns:
        An EasyDcit object containing configuration information.
    """
    with open(file, 'r') as f:
        config = EasyDict(yaml.load(f, Loader=yaml.Loader))
    return config


def save_config(cfg, file):
    """
    Saves configuration to file.
    Args:
        cfg: An EasyDcit object containing configuration information.
        file: Path to configuration file.
    """
    with open(file, 'w') as f:
        yaml.dump(easydict2dict(cfg), f, default_flow_style=False)


def easydict2dict(ed):
    """
    Converts EasyDict to dict.
    Args:
        ed: An EasyDict.
    Returns:
        A dict with the same content as ed.
    """
    d = {}
    for key, val in ed.items():
        if isinstance(val, EasyDict):
            d[key] = easydict2dict(val)
        elif isinstance(val, Path):
            d[key] = str(val)
        else:
            d[key] = val
    return d


if __name__ == '__main__':
    cfg = read_config('video_pose_243f.yaml')
    d = easydict2dict(cfg)
    pprint(d)