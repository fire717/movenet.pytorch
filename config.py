"""
@Fire
https://github.com/fire717
"""

# dataset = "coco"
# dataset = "mpii2"
# dataset = 'h36m'
dataset = 'h36m_cropped/half'
# dataset = 'DHP19'
home = "/media/Data/data/" + dataset + "/"
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
    'save_best_only': False,

    'pin_memory': True,
    'newest_ckpt': home + 'output/newest.json',
    'th': 50,  # percentage of headsize
    'from_scratch': True,

    ##### Train Hyperparameters
    'learning_rate': 0.001,  # 1.25e-4
    'batch_size': 64,
    'epochs': 300,
    'optimizer': 'Adam',  # Adam  SGD
    'scheduler': 'MultiStepLR-70,100-0.1',  # default  SGDR-5-2  CVPR   step-4-0.8 MultiStepLR
    # multistepLR-<<>milestones>-<<decay multiplier>>
    'weight_decay': 0.001,  # 5.e-4,  # 0.0001,
    'class_weight': None,  # [1., 1., 1., 1., 1., 1., 1., ]
    'clip_gradient': 5,  # 1,
    'w_heatmap': 1,
    'w_bone': 20,
    'w_center': 1,
    'w_reg': 3,
    'w_offset': 1,

    ##### Test
    'predict_output_path': home + "/predict/",
    'results_path': home + "/results/",
}

# test_img_path is for prediction script

if dataset == "coco":
    cfg["num_classes"] = 17
    cfg["img_path"] = "cropped/imgs"
    cfg["separated_data"] = True
    cfg["train_label_path"] = home + 'cropped/train2017.json'
    cfg["val_label_path"] = home + 'cropped/val2017.json'

    cfg["test_img_path"] = home + "cropped/imgs"
    cfg["exam_label_path"] = home + '/all/data_all_new.json'
    cfg["exam_output_path"] = home + "/exam/"
    cfg["eval_img_path"] = home + "cropped/imgs"
    cfg["eval_label_path"] = home + 'val.json'

if dataset == "mpii2":
    cfg["num_classes"] = 13
    cfg["img_path"] = home + "eros_synthetic_export_cropped/"
    cfg["separated_data"] = True
    cfg["train_label_path"] = home + '/eros_synthetic_export_cropped/train.json'
    cfg["val_label_path"] = home + '/eros_synthetic_export_cropped/val.json'

    # cfg["test_img_path"] = home + '/tos_synthetic_export/'
    # cfg["test_img_path"] = home + 'eval_subset/'
    # cfg["predict_output_path"] = home + '/pred_mpii2_on-mpii-cropped/'
    cfg["exam_label_path"] = home + 'val_subset.json'
    cfg["exam_img_path"] = home + 'eval_subset/'
    cfg["exam_output_path"] = home + "/exam/"
    cfg["eval_img_path"] = home + '/eros_synthetic_export_cropped/'
    cfg["eval_label_path"] = home + '/eros_synthetic_export_cropped/val.json'
    # cfg["eval_label_path"] =  home + 'cropped/val2017.json'

if dataset == 'h36m':
    cfg["num_classes"] = 13
    cfg["img_path"] = home + "training/h36m_EROS/"
    cfg["separated_data"] = True
    cfg["train_label_path"] = home + '/training/train_subject.json'
    cfg["val_label_path"] = home + '/training/val_subject.json'

    cfg["test_img_path"] = home + '/samples_for_pred/'
    cfg["predict_output_path"] = home + "/pred_output/"
    cfg["exam_label_path"] = home + '/samples_for_pred/poses.json'
    cfg["exam_img_path"] = home + '/samples_for_pred/'
    cfg["exam_output_path"] = home + "/exam/"
    cfg["eval_img_path"] = home + '/training/h36m_EROS/'
    cfg["eval_label_path"] = home + '/training/val_subject.json'

if dataset == 'h36m_cropped/half':
    cfg["num_classes"] = 13
    cfg["img_path"] = home + "images/h36m_EROS"
    cfg["separated_data"] = True
    cfg["train_label_path"] = home + 'train_subject.json'
    cfg["val_label_path"] = home + 'val_subject.json'

    cfg["test_img_path"] = home + ''
    cfg["predict_output_path"] = home + "predictions/"
    cfg["exam_label_path"] = home + ''
    cfg["exam_img_path"] = home + ''
    cfg["exam_output_path"] = home + "examinations/"
    cfg["eval_img_path"] = home + ''
    cfg["eval_label_path"] = home + ''

# samples_for_pred
if dataset == 'DHP19':
    cfg["num_classes"] = 13
    cfg["img_path"] = home + "/samples_for_pred/"
    cfg["separated_data"] = True
    cfg["train_label_path"] = home + '/poses.json'
    cfg["val_label_path"] = home + '/poses.json'

    cfg["test_img_path"] = home + '/samples_for_pred/'
    cfg["predict_output_path"] = home + "/pred/"
    cfg["exam_output_path"] = home + "/exam/"
    cfg["exam_img_path"] = home + 'samples_for_pred/'
    cfg["exam_label_path"] = home + '/poses.json'
    cfg["eval_img_path"] = home + '/samples_for_pred/'
    cfg["eval_label_path"] = home + "/poses.json"

# cfg['newest_ckpt'] = "/home/ggoyal/data/mpii/output/hypertune_0/best.pth"  # mpii original EROS
# cfg['newest_ckpt'] = "/home/ggoyal/data/mpii2/output/best.pth"  # mpii cropped EROS
# cfg['newest_ckpt'] = "/home/ggoyal/data/h36m/output/h36m_subject_sub/best.pth"  # h36m subject fromscratch
cfg['newest_ckpt'] = "/home/ggoyal/data/h36m/output/h36m_finetune_sub/best.pth"  # h36m subject fromscratch

# cfg["exam_output_path"] = home + "/exams_and_evals/exam_on-mpii1/"
# cfg["exam_output_path"] = home + "/exams_and_evals/exam_on-mpii-cropped/"
# cfg["exam_output_path"] = home + "/exams_and_evals/eval_on-h36m-scratch/"
# cfg["exam_output_path"] = home + "/exams_and_evals/eval_on-h36m-finetuned/"