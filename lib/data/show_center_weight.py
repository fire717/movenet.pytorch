"""
@Fire
https://github.com/fire717
"""
import numpy as np
import cv2



data = np.load("center_weight_origin.npy")
print(data.shape)

data = np.reshape(data, (-1))

print(data)

with open("center_weight_origin.txt",'w') as f:
    f.write(','.join([str(x) for x in data.tolist()]))