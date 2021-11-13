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


read_list = ["13821708-1-64.json",
            "254962538-1-64.json",
            "301370759-1-64.json",
            "347686742-1-64.json",
            "386530533-1-64.json",
            "390420891-1-64.json",
            "label5.json"]#

data_all = []
for name in read_list:
    with open(name,'r') as f:
        data = json.loads(f.readlines()[0])  
    print(name, len(data))

    for d in data:
        data_all.append(d)


print("total: ", len(data_all))
random.shuffle(data_all)
with open("video_all4.json",'w') as f:  
    json.dump(data_all, f, ensure_ascii=False)    