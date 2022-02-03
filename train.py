"""
@Fire
https://github.com/fire717
"""
from lib import init, Data, MoveNet, Task

from config import cfg



def main(cfg):

    init(cfg)


    model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')
    # print(model)
    # b


    data = Data(cfg)
    train_loader, val_loader = data.getTrainValDataloader()
    # data.showData(train_loader)
    # b


    run_task = Task(cfg, model)
    if not cfg["from_scratch"]:
        run_task.modelLoad(cfg["newest_ckpt"])
    run_task.train(train_loader, val_loader)




if __name__ == '__main__':
    main(cfg)