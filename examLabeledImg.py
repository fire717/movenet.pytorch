"""
@Fire
https://github.com/fire717
"""
import os
import random
import pandas as pd

from lib import init, Data, MoveNet, Task

from config import cfg


def main(cfg):
    init(cfg)

    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')
    #

    data = Data(cfg)
    data_loader = data.getExamDataloader()

    run_task = Task(cfg, model)
    # run_task.modelLoad("/media/ggoyal/Data/data/mpii/output/e27_valacc0.99331.pth")
    run_task.modelLoad(cfg["newest_ckpt"])

    run_task.exam(data_loader, cfg["exam_output_path"])


if __name__ == '__main__':
    main(cfg)
