"""
@Fire
https://github.com/fire717
"""

cfg = {
    ##### Global Setting
    'GPU_ID': '0',
    "num_workers":4,
    "random_seed":42,
    "cfg_verbose":True,

    "save_dir": "output/",

    "num_classes": 17,
    "width_mult":1.0,
    "img_size": 192, 
    

    ##### Train Setting
    'img_path':"./data/croped/imgs",
    'train_label_path':'./data/croped/train2017.json',
    'val_label_path':'./data/croped/val2017.json',
    'balance_data':False,

    'log_interval':10,  
    'save_best_only': True,
    
    'pin_memory': True,


    ##### Train Hyperparameters
    'learning_rate':0.001,#1.25e-4
    'batch_size':64,
    'epochs':110,
    'optimizer':'Adam',  #Adam  SGD
    'scheduler':'MultiStepLR-70,100-0.1', #default  SGDR-5-2  CVPR   step-4-0.8 MultiStepLR
    'weight_decay' : 5.e-4,#0.0001,


    'class_weight': None,#[1., 1., 1., 1., 1., 1., 1., ]
    'clip_gradient': 5,#1,   



    ##### Test
    'test_img_path':"./data/croped/imgs",

    #"../data/eval/imgs",
    #"../data/eval/imgs",
    #"../data/all/imgs"
    #"../data/true/mypc/crop_upper1"
    #../data/coco/small_dataset/imgs
    #"../data/testimg"
    'exam_label_path':'../data/all/data_all_new.json',

    'eval_img_path':'../data/eval/imgs',
    'eval_label_path':'../data/eval/mypc.json',
    }
