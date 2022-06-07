"""
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

    run_task.modelLoad('/home/ggoyal/data/h36m_cropped/runs/e48_valacc0.81777.pth')

    # run_task.evaluate(data_loader)
    run_task.infer_video(data_loader,'/home/ggoyal/data/DHP19/tester/out.avi')


if __name__ == '__main__':
    cfg = arg_parser(cfg)
    main(cfg)
