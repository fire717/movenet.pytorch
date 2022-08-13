"""
@Fire
https://github.com/fire717
"""
import os
import json
import pickle
import cv2
import numpy as np




"""
segmentation格式取决于这个实例是一个单个的对象（即iscrowd=0，将使用polygons格式）
还是一组对象（即iscrowd=1，将使用RLE格式

iscrowd=1时（将标注一组对象，比如一群人）


标注说明：x,y,v,x,y,v,...
其中v：#0没有标注;1有标注不可见（被遮挡）;2有标注可见

关键点顺序：'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 
    'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 
    'right_ankle']

"""


def main(img_dir, labels_path, output_name, output_img_dir):

    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)


    with open(labels_path, 'r') as f:
        data = json.load(f)

    #print("total: ", len(data)) 5
    #print(data.keys())#['info', 'licenses', 'images', 'annotations', 'categories']
    #print(len(data['annotations']), len(data['images']))#88153 40504
    #print(data['categories'])
    """
    [{'supercategory': 'person', 'name': 'person', 
    'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], 
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], 
    4, 6], [5, 7]], 
    'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 
    'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 
    'right_ankle'], 'id': 1}]
    """
    #print(data['images'][:3])#有filename和id

    img_id_to_name = {}
    img_name_to_id = {}
    for item in data['images']:
        idx = item['id']
        name = item['file_name']
        img_id_to_name[idx] = name
        img_name_to_id[name] = idx
    print(len(img_id_to_name))
    

    anno_by_imgname = {}
    for annotation in data['annotations']:
        name = img_id_to_name[annotation['image_id']]
        if name in anno_by_imgname:
            anno_by_imgname[name] += [annotation]
        else:
            anno_by_imgname[name] = [annotation]
    print(len(anno_by_imgname))



    new_label = []
    for k,v in anno_by_imgname.items():
        #filter out more than 3 people
        if len(v)>3:
            continue

        # print(k)
        # print(v)

        img = cv2.imread(os.path.join(img_dir, k))
        h,w = img.shape[:2]
        for idx,item in enumerate(v):
            if item['iscrowd'] != 0:
                continue

            bbox = [int(x) for x in item['bbox']]#x,y,w,h
            # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,255,0), 2)

            keypoints = item['keypoints']

            # for i in range(len(keypoints)//3):
            #     x = keypoints[i*3]
            #     y = keypoints[i*3+1]
            #     z = keypoints[i*3+2]#0没有标注;1有标注不可见（被遮挡）;2有标注可见
            #     # print(x,y,z)
            #     if z==1:
            #         color = (255,0,0)
            #     elif z==2:
            #         color = (0,0,255)
            #     else:
            #         continue
                # cv2.circle(img, (x, y), 4, color, 3)


            # merge bbox and keypoints to get max bbox 
            keypoints = np.array(keypoints).reshape((17,3))

            keypoints_v = keypoints[keypoints[:,2]>0]
            if len(keypoints_v)<8:#filter out keypoints not enough
                continue
            min_key_x = np.min(keypoints_v[:,0])
            max_key_x = np.max(keypoints_v[:,0])
            min_key_y = np.min(keypoints_v[:,1])
            max_key_y = np.max(keypoints_v[:,1])

            x0 = min(bbox[0], min_key_x)
            x1 = max(bbox[0]+bbox[2], max_key_x)
            y0 = min(bbox[1], min_key_y)
            y1 = max(bbox[1]+bbox[3], max_key_y)
            # cv2.rectangle(img, (x0, y0), (x1, y1), (0,255,255), 2)

            # expand to square then expand
            cx = (x0+x1)/2
            cy = (y0+y1)/2
            
            half_size = ((x1-x0)+(y1-y0))/2 * EXPAND_RATIO
            new_x0 = int(cx - half_size)
            new_x1 = int(cx + half_size)
            new_y0 = int(cy - half_size)
            new_y1 = int(cy + half_size)

            #pad where exceed edge
            pad_top = 0
            pad_left = 0
            pad_right = 0
            pad_bottom = 0
            if new_x0 < 0:
                pad_left = -new_x0+1
            if new_y0 < 0:
                pad_top = -new_y0+1
            if new_x1 > w:
                pad_right = new_x1-w+1
            if new_y1 > h:
                pad_bottom = new_y1-h+1

            pad_img = np.zeros((h+pad_top+pad_bottom, w+pad_left+pad_right, 3))
            pad_img[pad_top:pad_top+h,pad_left:pad_left+w] = img
            new_x0 += pad_left
            new_y0 += pad_top
            new_x1 += pad_left
            new_y1 += pad_top
            # cv2.rectangle(pad_img, (new_x0, new_y0), (new_x1, new_y1), (0,255,0), 2)

            # final save data
            save_name = k[:-4]+"_"+str(idx)+".jpg"
            new_w = new_x1-new_x0
            new_h = new_y1-new_y0
            save_img = pad_img[new_y0:new_y1,new_x0:new_x1]
            save_bbox = [(bbox[0]+pad_left-new_x0)/new_w,
                         (bbox[1]+pad_top-new_y0)/new_h,
                         (bbox[0]+bbox[2]+pad_left-new_x0)/new_w,
                         (bbox[1]+bbox[3]+pad_top-new_y0)/new_h
                        ]
            save_center = [(save_bbox[0]+save_bbox[2])/2,(save_bbox[1]+save_bbox[3])/2]

            save_keypoints = []
            for kid in range(len(keypoints)):
                save_keypoints.extend([(int(keypoints[kid][0])+pad_left-new_x0)/new_w,
                                       (int(keypoints[kid][1])+pad_top-new_y0)/new_h,
                                       int(keypoints[kid][2])
                                      ])
            other_centers = []
            other_keypoints = [[] for _ in range(17)]
            for idx2,item2 in enumerate(v):
                if item2['iscrowd'] != 0 or idx2==idx:
                    continue
                bbox2 = [int(x) for x in item2['bbox']]#x,y,w,h

                save_bbox2 = [(bbox2[0]+pad_left-new_x0)/new_w,
                             (bbox2[1]+pad_top-new_y0)/new_h,
                             (bbox2[0]+bbox2[2]+pad_left-new_x0)/new_w,
                             (bbox2[1]+bbox2[3]+pad_top-new_y0)/new_h
                            ]
                save_center2 = [(save_bbox2[0]+save_bbox2[2])/2,
                                (save_bbox2[1]+save_bbox2[3])/2]
                if save_center2[0]>0 and save_center2[0]<1 and save_center2[1]>0 and save_center2[1]<1:
                    other_centers.append(save_center2)

                keypoints2 = item2['keypoints']
                keypoints2 = np.array(keypoints2).reshape((17,3))
                for kid2 in range(17):
                    if keypoints2[kid2][2]==0:
                        continue
                    kx = (keypoints2[kid2][0]+pad_left-new_x0)/new_w
                    ky = (keypoints2[kid2][1]+pad_top-new_y0)/new_h
                    if kx>0 and kx<1 and ky>0 and ky<1:
                        other_keypoints[kid2].append([kx,ky])

            save_item = {
                         "img_name":save_name,
                         "keypoints":save_keypoints,
                         "center":save_center,
                         "bbox":save_bbox,
                         "other_centers":other_centers,
                         "other_keypoints":other_keypoints,
                        }
            # for k,v in save_item.items():
            #     print(type(v[0]))
            # b
            new_label.append(save_item)



            ###visul for exam, comment when use
            if SHOW_POINTS_ON_IMG:
                cv2.circle(save_img, (int(save_center[0]*new_w), int(save_center[1]*new_h)), 4, (0,255,0), 3)
                for show_kid in range(len(save_keypoints)//3):
                    if save_keypoints[show_kid*3+2]==1:
                        color = (255,0,0)
                    elif save_keypoints[show_kid*3+2]==2:
                        color = (0,0,255)
                    else:
                        continue
                    cv2.circle(save_img, (int(save_keypoints[show_kid*3]*new_w), 
                                int(save_keypoints[show_kid*3+1]*new_h)), 3, color, 2)
                cv2.rectangle(save_img, (int(save_bbox[0]*new_w), int(save_bbox[1]*new_h)), 
                        (int(save_bbox[2]*new_w), int(save_bbox[3]*new_h)), (0,255,0), 2)
                for show_c in other_centers:
                    cv2.circle(save_img, (int(show_c[0]*new_w), int(show_c[1]*new_h)), 4, (0,255,255), 3)
                for show_ks in other_keypoints:
                    for show_k in show_ks:
                        cv2.circle(save_img, (int(show_k[0]*new_w), int(show_k[1]*new_h)), 3, (255,255,0), 2)


            cv2.imwrite(os.path.join(output_img_dir, save_name), save_img)
            
            # print(save_item, save_img.shape)

            # b
        # cv2.imwrite(os.path.join("show", k), pad_img)

    with open(output_name,'w') as f:  
        json.dump(new_label, f, ensure_ascii=False)     
    print('Total write ', len(new_label))


if __name__ == '__main__':

    #### PARAM ####

    SHOW_POINTS_ON_IMG = False
    #whether to show points on img for debug

    EXPAND_RATIO = 1. 
    #person body bbox expand range to image edge

    output_img_dir = "./data/cropped/imgs"


    img_dir = "./data/val2017"
    labels_path = "./data/annotations/person_keypoints_val2017.json"
    output_name = './data/cropped/val2017.json'
    main(img_dir, labels_path, output_name, output_img_dir)

    

    img_dir = "./data/train2017"
    labels_path = "./data/annotations/person_keypoints_train2017.json"
    output_name = './data/cropped/train2017.json'
    main(img_dir, labels_path, output_name, output_img_dir)


    