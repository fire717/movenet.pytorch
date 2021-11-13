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

read_list = ["train2014.json","val2014.json","train2017.json","val2017.json"]

data_all = []
for name in read_list:
    with open(name,'r') as f:
        data = json.loads(f.readlines()[0])  
    print(name, len(data))

    for d in data:
        data_all.append(d)


print("total: ", len(data_all))
random.shuffle(data_all)
with open("data_all.json",'w') as f:  
    json.dump(data_all, f, ensure_ascii=False)    