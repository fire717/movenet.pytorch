"""
@Fire
https://github.com/fire717
"""
import os, argparse
import random

from lib import init, Data, MoveNet, Task

from config import cfg
from lib.utils.utils import arg_parser


def main(cfg):
    init(cfg)

    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')

    data = Data(cfg)
    data_loader = data.getEvalDataloader()

    run_task = Task(cfg, model)

    # run_task.modelLoad("/home/ggoyal/data/mpii/output/e300_valacc0.86824.pth")
    run_task.modelLoad(cfg["newest_ckpt"])

    run_task.evaluate(data_loader)


if __name__ == '__main__':
    cfg = arg_parser(cfg)
    main(cfg)
