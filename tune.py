"""
@Fire
https://github.com/fire717
"""
from lib import init, Data, MoveNet, Task
from config import cfg
from lib.utils.utils import arg_parser,ensure_loc
import random, os



def main(cfg):

    for i in range(10):
        cfg['label'] = 'hypertune_'+str(i)
        cfg['newest_ckpt'] = os.path.join(cfg['save_dir'],cfg['label'],'newest.json')
        #  Hyper parameter tuning for variables:
        cfg['batch_size'] = random.choice([32,64,128])
        cfg['learning_rate'] = random.choice([0.001,0.005,0.01,0.05,0.1])
        cfg['weight_decay'] = random.choice([0.0001,0.0002,0.0005,0.001])
        cfg['scheduler'] = random.choice(['MultiStepLR-70,100-0.1', 'MultiStepLR-30,50,70,90,110-0.5','MultiStepLR-30,70,110-0.2','MultiStepLR-70,110-0.2', 'MultiStepLR-30,70,130-0.2'])
        cfg['w_reg'] = random.choice([1,3,5,10])
        cfg['w_bone'] = random.choice([5,20,40])

        init(cfg)
        ensure_loc(os.path.join(cfg['save_dir'],cfg['label']))
        model = MoveNet(num_classes=cfg["num_classes"],
                        width_mult=cfg["width_mult"],
                        mode='train')

        data = Data(cfg)
        train_loader, val_loader = data.getTrainValDataloader()

        run_task = Task(cfg, model)
        run_task.modelSave('e0_accu0.pth', is_best=True)

        run_task.train(train_loader, val_loader)
        run_task.save_results()

if __name__ == '__main__':
    cfg = arg_parser(cfg)

    main(cfg)
