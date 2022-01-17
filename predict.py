"""
@Fire
https://github.com/fire717
"""

from lib import init, Data, MoveNet, Task

from config import cfg


# Script to create and save as images all the various outputs of the model


def main(cfg):

    init(cfg)


    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')
    
    
    data = Data(cfg)
    test_loader = data.getTestDataloader()
    # _,test_loader = data.getTrainValDataloader()


    run_task = Task(cfg, model)
    run_task.modelLoad("/home/ggoyal/data/mpii/output/e1000_valacc0.66665.pth")
    # run_task.modelLoad("output/mbv2_e105_valacc0.80255.pth") # for coco
    # run_task.modelLoad(cfg["newest_ckpt"])


    run_task.predict(test_loader, cfg["predict_output_path"])
    # run_task.predict(test_loader, "output/predict")



if __name__ == '__main__':
    main(cfg)