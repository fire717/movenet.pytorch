"""
@Fire
https://github.com/fire717
"""
from lib import init, Data, MoveNet, Task
from config import cfg
import argparse


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
    parser = argparse.ArgumentParser(prog="train")
    ##### Global Setting
    parser.add_argument('--GPU_ID', help='GPUs to use. Example: 0,1,2', default=cfg['GPU_ID'])
    parser.add_argument('--dataset', help='Training dataset.', default=cfg['dataset'],
                        choices=["mpii", "coco", 'h36m', 'dhp19'])
    parser.add_argument('--num_workers', help='Number of workers', default=cfg['num_workers'], type=int)
    parser.add_argument('--random_seed', help='Random seed', default=cfg['random_seed'], type=int)
    parser.add_argument('--cfg_verbose', '-v', help='Verbosity', default=cfg['cfg_verbose'], type=bool)
    parser.add_argument('--save_dir', help='Directory to save model checkpoints', default=cfg['save_dir'], type=str)
    parser.add_argument('--num_classes', help='Number of joints in the dataset', default=cfg['num_classes'], type=int)
    parser.add_argument('--img_size', help='Image size of the square image for training and output',
                        default=cfg['img_size'], type=int)

    ##### Train Setting
    parser.add_argument('--pre-separated_data',
                        help='Set true if the training and validation annotations are in separate files',
                        default=cfg['pre-separated_data'], type=bool)
    parser.add_argument('--training_data_split',
                        help='Percentage of data to use for training. Not used if pre-separated set to True',
                        default=cfg['training_data_split'], type=int)
    parser.add_argument('--newest_ckpt', help='File containing the name of the latest model checkpoints',
                        default=cfg['newest_ckpt'], type=str)
    parser.add_argument('--balance_data', help='Set true for data balancing', default=cfg['balance_data'], type=bool)
    parser.add_argument('--log_interval', help='How frequently to log output', default=cfg['log_interval'], type=int)
    parser.add_argument('--save_best_only', help='Save only the best or improved model checkpoints',
                        default=cfg['save_best_only'], type=bool)
    parser.add_argument('--pin_memory', help='Pin memory', default=cfg['pin_memory'], type=bool)
    parser.add_argument('--th', help='Threshold value, in percentage of head size, for accuracy calculation',
                        default=cfg['th'], type=int)
    parser.add_argument('--from_scratch', help='Set true to begin training from scratch', default=cfg['from_scratch'],
                        type=bool)

    ##### Train Hyperparameters
    parser.add_argument('--learning_rate', help='Initial learning rate of the training paradigm',
                        default=cfg['learning_rate'], type=float)
    parser.add_argument('--batch_size', help='Batch size', default=cfg['batch_size'], type=int)
    parser.add_argument('--epochs', help='Epochs', default=cfg['epochs'], type=int)
    parser.add_argument('--optimizer', help='Optimizer', default=cfg['optimizer'], type=str, choices=['Adam', 'SGD'])
    parser.add_argument('--scheduler', help='scheduler', default=cfg['scheduler'], type=str)
    parser.add_argument('--weight_decay', help='weight decay', default=cfg['weight_decay'], type=float)
    parser.add_argument('--class_weight', help='class weight', default=cfg['class_weight'])
    parser.add_argument('--clip_gradient', help='clip_gradient', default=cfg['clip_gradient'])

    #### Directory paths
    parser.add_argument('--img_path', help='Directory to the images for training and validation',
                        default=cfg['img_path'], type=str)
    parser.add_argument('--train_label_path', help='File (full or relative path) to training annotation file',
                        default=cfg['train_label_path'], type=str)
    parser.add_argument('--val_label_path', help='File (full or relative path) to validation annotation file',
                        default=cfg['val_label_path'], type=str)
    parser.add_argument('--test_img_path', help='Directory to the images for test', default=cfg['test_img_path'],
                        type=str)
    parser.add_argument('--predict_output_path', help='Directory to save images after prediction (without GT)',
                        default=cfg['predict_output_path'], type=str)
    parser.add_argument('--exam_label_path', help='File (full or relative path) to validation annotation file',
                        default=cfg['exam_label_path'], type=str)
    parser.add_argument('--eval_img_path', help='Directory to the images for evaluation', default=cfg['eval_img_path'],
                        type=str)
    parser.add_argument('--eval_label_path', help='File (full or relative path) to evaluation annotation file',
                        default=cfg['eval_label_path'], type=str)

    args = parser.parse_args()
    # cfg['GPU_ID'] = args.GPU_ID
    print(vars(args))

    for key, value in vars(args).items():
        cfg[key] = value
    print(cfg)
    exit()
    main(cfg)
