"""
@Fire
https://github.com/fire717
"""
import os
import json
import pickle
import cv2
import numpy as np


img_dir = "imgs"
save_dir = "show"

with open(r"video_all4.json",'r') as f:
    data = json.loads(f.readlines()[0])  
print("total: ", len(data))

for item in data:
    """
save_item = {
                         "img_name":save_name,
                         "keypoints":save_keypoints,
                         "center":save_center,
                         "bbox":save_bbox,
                         "other_centers":other_centers,
                         "other_keypoints":other_keypoints,
                        }
    """

    img_name = item['img_name']
    img = cv2.imread(os.path.join(img_dir, img_name))
    h,w = img.shape[:2]

    save_center = item['center']
    save_keypoints = item['keypoints']
    # save_bbox = item['bbox']
    other_centers = item['other_centers']
    other_keypoints = item['other_keypoints']

    cv2.circle(img, (int(save_center[0]*w), int(save_center[1]*h)), 4, (0,255,0), 3)
    for show_kid in range(len(save_keypoints)//3):
        if save_keypoints[show_kid*3+2]==1:
            color = (255,0,0)
        elif save_keypoints[show_kid*3+2]==2:
            color = (0,0,255)
        else:
            continue
        cv2.circle(img, (int(save_keypoints[show_kid*3]*w), 
                    int(save_keypoints[show_kid*3+1]*h)), 3, color, 2)
    # cv2.rectangle(img, (int(save_bbox[0]*w), int(save_bbox[1]*h)), 
    #         (int(save_bbox[2]*w), int(save_bbox[3]*h)), (0,255,0), 2)
    for show_c in other_centers:
        cv2.circle(img, (int(show_c[0]*w), int(show_c[1]*h)), 4, (0,255,255), 3)
    for show_ks in other_keypoints:
        for show_k in show_ks:
            cv2.circle(img, (int(show_k[0]*w), int(show_k[1]*h)), 3, (255,255,0), 2)

    cv2.imwrite(os.path.join(save_dir, img_name), img)