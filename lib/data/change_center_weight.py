"""
@Fire
https://github.com/fire717
"""
import numpy as np 




_center_weight = np.load('center_weight_origin.npy')
_center_weight = np.reshape(_center_weight,(48,48))
print(_center_weight[0])

_center_weight1 = _center_weight[:,::-1]
# print(_center_weight1[0])


_center_weight2 = _center_weight[::-1,:]
# print(_center_weight2[0])

_center_weight3 = _center_weight[::-1,::-1]
# print(_center_weight3[0])



_center_weight_new = (_center_weight+_center_weight1+_center_weight2+_center_weight3)/4
print(_center_weight_new[0])

print(_center_weight_new[22:28,22:28])

np.save('center_weight.npy',_center_weight_new)