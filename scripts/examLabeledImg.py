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
    run_task.modelLoad("output/test/e105_valacc0.98326.pth")


    run_task.exam(data_loader, "../data/all/exam3")



if __name__ == '__main__':
    main(cfg)