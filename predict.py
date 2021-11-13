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
    
    
    data = Data(cfg)
    test_loader = data.getTestDataloader()


    run_task = Task(cfg, model)
    run_task.modelLoad("output/mbv2_e105_valacc0.80255.pth")


    run_task.predict(test_loader, "output/predict")



if __name__ == '__main__':
    main(cfg)