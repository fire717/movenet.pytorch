"""
@Fire
https://github.com/fire717
"""
import numpy as np
import cv2



### 先生成一张99*99的大图，然后根据关键点reg的坐标去crop
def new_w():
    ratio = 0.01
    weights = np.zeros((99,99))
    half_w = []
    for i in range(49):
        half_w.append(0.99-i*ratio)
    line = np.array(half_w[::-1].copy()+[1]+half_w.copy())
    weights[49] = line.copy()
    for i in range(1,50):
        weights[49-i] = line.copy()-i*ratio
        weights[49+i] = line.copy()-i*ratio
        
    return weights

new_w = new_w()
print(new_w[45:50,45:50])
print(new_w[:6,0:6])
cv2.imwrite("new_w.jpg", new_w*255)
np.save("my_weight_reg.npy", new_w)
