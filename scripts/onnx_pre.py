"""
@Fire
https://github.com/fire717
"""
# from __future__ import print_function
import numpy as np
import cv2
# from cv2 import dnn
# import sys
 
# import tensorflow as tf
# from tensorflow.python.framework import graph_util
# import os

# import time


import time

import onnxruntime as rt

model_path = '../output/pose.onnx'
sess=rt.InferenceSession(model_path)#model_path就是模型的地址
input_name=sess.get_inputs()[0].name


img = cv2.imread( '../data/croped/imgs/000000000036_0.jpg')
print("img shape: ", img.shape)
img = cv2.resize(img, ( 192, 192))
img = img[:, :, [ 2, 1, 0]] # BGR2RGB

data = img.reshape( 1, img.shape[ 0], img.shape[ 1], 3)
#print(data.shape)
data = np.transpose(data,(0,3,1,2))
# data = data/255.0
# data = (data-0.5)/0.5
#print(data.shape)
data = data.astype(np.float32)


res = sess.run(None,{input_name:data})

print("res: ", np.array(res[0]).shape)

print(res[0][0][0][0][0])
print(res[1][0][0][0][0],res[2][0][0][0][0],res[3][0][0][0][0])
