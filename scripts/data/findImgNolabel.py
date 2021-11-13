"""
@Fire
https://github.com/fire717
"""


import os
import json





read_dir = "show"
img_names = os.listdir(read_dir)
print("len img: ",len(img_names))



with open("data_all_new.json",'r') as f:
    data = json.loads(f.readlines()[0])  
    ##print(data[0])
    print("len label: ", len(data))

data_names = set([item['img_name'] for item in data])


for i,img_name in enumerate(img_names):
    if i%10000==0:
        print(i)

    if img_name not in data_names:
        print(img_name)



    


