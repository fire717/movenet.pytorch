from config import cfg
from lib import MoveNet, Task
from pathlib import Path
import json
from lib.utils.utils import arg_parser


def main(cfg):


    if cfg["num_classes"] == 17:
        fullname = 'mbv2_e105_valacc0.80255.pth'
        with open(Path(cfg['newest_ckpt']).resolve(), 'w') as f:
            json.dump(fullname, f, ensure_ascii=False)
    else:
        model = MoveNet(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')

        run_task = Task(cfg, model)
        run_task.modelSave('e0_accu0.pth',is_best=True)


if __name__ == '__main__':
    cfg = arg_parser(cfg)
    main(cfg)