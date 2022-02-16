"""
@Fire
https://github.com/fire717
"""

# dataset = "coco"
dataset = "mpii"
home = "/media/ggoyal/Data/data/"+dataset+"/"
# home = "/work/ggoyal/Data/"+dataset+"/"

cfg = {
        ##### Global Setting
        'GPU_ID': '',
        "num_workers": 4,
        "random_seed": 42,
        "cfg_verbose": True,
        "save_dir": home + "output/",
        # "num_classes": 17,
        "width_mult": 1.0,
        "img_size": 192,
        'label': 'dev',

        ##### Train Setting
        'pre-separated_data': True,
        'training_data_split': 80,
        "dataset": dataset,
        'balance_data': False,
        'log_interval': 10,
        'save_best_only': True,

        'pin_memory': True,
        'newest_ckpt': home + 'output/newest.json',
        'th':50, # percentage of headsize
        'from_scratch':True,

        ##### Train Hyperparameters
        'learning_rate': 0.001,  # 1.25e-4
        'batch_size': 64,
        'epochs': 300,
        'optimizer': 'Adam',  # Adam  SGD
        'scheduler': 'MultiStepLR-70,100-0.1',  # default  SGDR-5-2  CVPR   step-4-0.8 MultiStepLR
        # multistepLR-<<>milestones>-<<decay multiplier>>
        'weight_decay': 5.e-4,  # 0.0001,
        'class_weight': None,  # [1., 1., 1., 1., 1., 1., 1., ]
        'clip_gradient': 5,  # 1,
        'w_heatmap': 1,
        'w_bone': 40,
        'w_center': 1,
        'w_reg': 3,
        'w_offset': 1,

        ##### Test
        'predict_output_path': home + "/predict/",
        'results_path': home + "/results/",
    }

if dataset == "coco":
    cfg["num_classes"] = 17
    cfg["img_path"] = "cropped/imgs"
    cfg["separated_data"] = True
    cfg["train_label_path"] = home+'cropped/train2017.json'
    cfg["val_label_path"] = home + 'cropped/val2017.json'

    cfg["test_img_path"] = home + "cropped/imgs"
    cfg["exam_label_path"] = home + '/all/data_all_new.json'
    cfg["eval_img_path"] = home + "cropped/imgs"
    cfg["eval_label_path"] = home + 'val.json'

if dataset == "mpii":
    cfg["num_classes"] = 13
    cfg["img_path"] = home + "tos_synthetic_export/"
    cfg["separated_data"] = True
    cfg["train_label_path"] = home + '/anno/train_2.json'
    cfg["val_label_path"] = home + '/anno/val_2.json'

    # cfg["test_img_path"] = home + '/tos_synthetic_export/'
    cfg["test_img_path"] = "/home/ggoyal/data/DHP19/dhp19_s1_2_4-cam3"
    cfg["predict_output_path"] = "/home/ggoyal/data/DHP19/dhp19_s1_2_4-cam3-samples_pred"
    cfg["exam_label_path"] = home + '/anno/val_2.json'
    cfg["eval_img_path"] = home + '/tos_synthetic_export/'
    cfg["eval_label_path"] = "/home/ggoyal/data/mpii/anno/val.json"
    # cfg["eval_label_path"] =  home + 'cropped/val2017.json'

if dataset == 'h36m':
    cfg["num_classes"] = 13
    cfg["img_path"] = home + "eros/"
    cfg["separated_data"] = True
    cfg["train_label_path"] = home + '/anno/train_half.json'
    cfg["val_label_path"] = home + '/anno/val_half.json'

    # cfg["test_img_path"] = home + '/tos_synthetic_export/'
    # cfg["test_img_path"] = "/home/ggoyal/data/DHP19/dhp19_s1_2_4-cam3"
    # cfg["predict_output_path"] = "/home/ggoyal/data/DHP19/dhp19_s1_2_4-cam3-samples_pred"
    # cfg["exam_label_path"] = home + '/anno/val_2.json'
    # cfg["eval_img_path"] = home + '/tos_synthetic_export/'
    # cfg["eval_label_path"] = "/home/ggoyal/data/mpii/anno/val.json"
