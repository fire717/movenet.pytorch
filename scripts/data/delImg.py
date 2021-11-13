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





imgs = os.listdir("imgs")
print("total imgs: ", len(imgs))

shows = os.listdir("show")
print("total shows: ", len(shows))


with open("data_all_new.json",'r') as f:
    data = json.loads(f.readlines()[0])  
print("total labels: ",len(data))


new_data = []
for d in data:
    name = d['img_name']
    if name not in shows:
        if name in imgs:
            os.rename(os.path.join("imgs",name),os.path.join("del",name))
    else:
        new_data.append(d)


print("total new_data: ", len(new_data))
with open("data_all_new.json",'w') as f:  
    json.dump(new_data, f, ensure_ascii=False)    