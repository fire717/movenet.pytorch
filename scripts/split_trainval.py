"""
@Fire
https://github.com/fire717
"""
import os
import json
import pickle
import cv2
import numpy as np
import random


def read_data(path):
    with open(path, 'r') as f:
        data = json.loads(f.readlines()[0])
    print("total: ", len(data))
    return data


def fix_data(data):
    fixed_data = []
    for d in data:
        if type(d) is dict:
            fixed_data.append(d)
        elif type(d) is list:
            fixed_data.extend(d)
    print('files are concatenating correctly:', len(fixed_data))
    return fixed_data

def pick_by_name(data,name):
    data_out = []
    for d in data:
        if name in d["img_name"]:
            data_out.append(d)
            print(d['img_name'])
    return data_out

def file_exists_check(data, path):
    data_fixed = []
    for d in data:
        file = os.path.join(path, d["img_name"])
        if os.path.exists(file):
            data_fixed.append(d)
    print('files that exist:', len(data_fixed))
    return data_fixed


def split(data, val_split=20, mode='ratio', val_subs=None, val_cams=None):
    data_train = []
    data_val = []
    if mode == 'ratio':
        random.shuffle(data)
        val_files_num = int((val_split / 100) * len(data))

        print("Number of validation files", val_files_num)
        print("Percentage of validation files", val_split)

        data_val = data[:val_files_num]
        data_train = data[val_files_num:]

    elif mode == 'subject':

        for d in data:
            if d["img_name"].split('_')[1] in val_subs:
                data_val.append(d)
            else:
                data_train.append(d)
        random.shuffle(data_val)
        random.shuffle(data_train)

    elif mode == 'camera':

        for d in data:
            if d["img_name"].split('_')[0] in val_cams:
                data_val.append(d)
            else:
                data_train.append(d)
        random.shuffle(data_val)
        random.shuffle(data_train)

    return data_train, data_val


### Read and pre-process the data
path = "/home/ggoyal/data/h36m_cropped/poses.json"
# path = r"/home/ggoyal/data/h36m/training/poses.json"
data = read_data(path)
# data = fix_data(data)
# img_path = '/home/ggoyal/data/h36m/training/h36m_EROS/'
# data = file_exists_check(data, img_path)

subs = ['S9', 'S11']
cams = ['cam4']
val_split = 20  # (percentage for validation)
# name = 'cam2_S9_Directions'
#
# data_out = pick_by_name(data,name)
# with open(f"/home/ggoyal/data/h36m_cropped/poses_{name}.json", 'w') as f:
#     json.dump(data_out, f, ensure_ascii=False)


#### Splitting the data
# data_train, data_val = split(data, mode='camera', val_cams=cams)
data_train, data_val = split(data, mode='subject', val_subs=subs)
# data_train, data_val = split(data, val_split=20)


### Saving the final data
print(len(data), len(data_train), len(data_val))
# with open("/home/ggoyal/data/h36m/training/poses_full_clean.json", 'w') as f:
#     json.dump(data, f, ensure_ascii=False)

with open("/home/ggoyal/data/h36m_cropped/train_subject.json", 'w') as f:
    json.dump(data_train, f, ensure_ascii=False)

with open("/home/ggoyal/data/h36m_cropped/val_subject.json", 'w') as f:
    json.dump(data_val, f, ensure_ascii=False)
