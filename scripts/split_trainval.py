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



with open(r"/home/ggoyal/data/mpii/poses_norm.json",'r') as f:
    data = json.loads(f.readlines()[0])  
print("total: ", len(data))
print(data[0])

random.shuffle(data)
print(data[0])

val_count = 20 #(percentage for validation)
ratio = int((val_count/100) *len(data))

print("val_nums", val_count)
print("ratio", ratio)

data_train = data[:ratio]
data_val = data[ratio:]
# for d in data:
#     if random.random()>ratio:
#         data_train.append(d)
#     else:
#         data_val.append(d)

print(len(data_train), len(data_val))
with open("/home/ggoyal/data/mpii/train.json",'w') as f:
    json.dump(data_train, f, ensure_ascii=False)

with open("/home/ggoyal/data/mpii/val.json",'w') as f:
    json.dump(data_val, f, ensure_ascii=False)