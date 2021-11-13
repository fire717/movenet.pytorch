"""
@Fire
https://github.com/fire717
"""
import os
import json
import pickle
import cv2
import numpy as np
import glob




read_dir = "label5"
save_dir = "croped"


output_name = 'croped/%s.json' % read_dir
output_img_dir = "croped/imgs"


if __name__ == '__main__':


    imgs = glob.glob(read_dir+'/*.jpg')
    print(len(imgs))

    new_label = []
    for i, img_path in enumerate(imgs):
        
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        h,w = img.shape[:2]


        label_path = img_path[:-3]+'txt'
        
        with open(label_path, 'r') as f:
            lines = f.readlines()

        if len(lines)!=8:
            continue
            
        keypoints = []
        for line in lines[1:]:

            x,y = [float(x) for x in line.strip().split(' ')[:2]]
            keypoints.extend([x,y,2]) 

        # center = [(keypoints[2*3]+keypoints[4*3])/2,
        #             (keypoints[2*3+1]+keypoints[4*3+1])/2]

        min_key_x = np.min(np.array(keypoints)[[0,3,6,9,12,15,18]])
        max_key_x = np.max(np.array(keypoints)[[0,3,6,9,12,15,18]])
        min_key_y = np.min(np.array(keypoints)[[1,4,7,10,13,16,19]])
        max_key_y = np.max(np.array(keypoints)[[1,4,7,10,13,16,19]])
        center = [(min_key_x+max_key_x)/2, (min_key_y+max_key_y)/2]


        save_item = {
                     "img_name":img_name,
                     "keypoints":keypoints,
                     "center":center,
                     "other_centers":[],
                     "other_keypoints":[[] for _ in range(7)],
                    }
        # print(save_item)
        new_label.append(save_item)

        cv2.imwrite(os.path.join(output_img_dir, img_name), img)




    with open(output_name,'w') as f:  
        json.dump(new_label, f, ensure_ascii=False)     
    print('Total write ', len(new_label))