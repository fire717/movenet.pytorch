"""
Functions to visualize the various outputs of movenet
Author: Gaurvi Goyal
"""
import os

import cv2
import numpy as np
import json


def superimpose_pose(img,pose,num_classes=13):
    """ inputs:
            img: rgb image of any size
            pose: numpy array for size (2,:) or 1d array in (x y x y ..) configuration
    """
    pose = np.array(pose)
    pose = pose.squeeze()
    pose = pose.reshape((num_classes,-1))
    h,w,_ = img.shape
    print("w is ", w)
    print("h is ", h)
    for i in range((pose.shape[0])):
        pose[i,0] = int(pose[i,0] * w)
        pose[i,1] = int(pose[i,1] * h)
        # img = cv2.circle(img,(5,5),3,(0,255,0),5)
        img = cv2.circle(img,(int(pose[i,0]),int(pose[i,1])),3,(0,255,0),5)
    cv2.imshow('a',img)
    cv2.waitKey()
    print(pose)

# if __name__ == "__main__":
#     file_img = '/home/ggoyal/data/mpii/tos_synthetic_export/000041029.jpg'
#     pose_path = '/home/ggoyal/data/mpii/poses_norm.json'
#
#     img = cv2.imread(file_img)
#     img_basename = os.path.basename(file_img)
#     with open(pose_path, 'r') as f:
#         train_label_list = json.loads(f.readlines()[0])
#     for line in train_label_list:
#         if line["img_name"] == img_basename:
#             pose = line["keypoints"]
#
#     superimpose_pose(img,pose, cfg["num_classes"])
