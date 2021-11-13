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



with open(r"data_all_new.json",'r') as f:
    data = json.loads(f.readlines()[0])  
print("total: ", len(data))
print(data[0])

random.shuffle(data)
print(data[0])

val_count = 600
ratio = val_count/len(data)


data_train = []
data_val = []
for d in data:
    if random.random()>ratio:
        data_train.append(d)
    else:
        data_val.append(d)

print(len(data_train), len(data_val))
with open("train.json",'w') as f:  
    json.dump(data_train, f, ensure_ascii=False)    

with open("val.json",'w') as f:  
    json.dump(data_val, f, ensure_ascii=False)    