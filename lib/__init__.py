"""
@Fire
https://github.com/fire717
"""
import os

from lib.data.data import Data
from lib.models.movenet_mobilenetv2 import MoveNet
from lib.task.task import Task


from lib.utils.utils import setRandomSeed, printDash


def init(cfg):

    if cfg["cfg_verbose"]:
        printDash()
        print(cfg)
        printDash()



    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['GPU_ID']
    setRandomSeed(cfg['random_seed'])



    if not os.path.exists(cfg['save_dir']):
        os.makedirs(cfg['save_dir'])