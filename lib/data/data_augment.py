"""
@Fire
https://github.com/fire717
"""
from PIL import Image
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
import cv2
import albumentations as A
import json
import platform
from copy import deepcopy

###### tools
def Mirror(src,label=None):
    """
    item = {
                     "img_name":save_name,  
                     "keypoints":save_keypoints, relative position
                     "center":save_center,
                     "other_centers":other_centers,
                     "other_keypoints":other_keypoints,
                    }
    # mirror 后左右手顺序就变了！
    """
    keypoints = label['keypoints']
    center = label['center']
    other_centers = label['other_centers']
    other_keypoints = label['other_keypoints']
        
    img = cv2.flip(src, 1)
    if label is None:
        return img,label


    for i in range(len(keypoints)):
        if i%3==0:
            keypoints[i] = 1-keypoints[i]
    try:
        keypoints = [ #17 keypoint skeleton mirror
                keypoints[0],keypoints[1],keypoints[2],
                keypoints[6],keypoints[7],keypoints[8],
                keypoints[3],keypoints[4],keypoints[5],
                keypoints[12],keypoints[13],keypoints[14],
                keypoints[9],keypoints[10],keypoints[11],
                keypoints[18],keypoints[19],keypoints[20],
                keypoints[15],keypoints[16],keypoints[17],
                keypoints[24],keypoints[25],keypoints[26],
                keypoints[21],keypoints[22],keypoints[23],
                keypoints[30],keypoints[31],keypoints[32],
                keypoints[27],keypoints[28],keypoints[29],
                keypoints[36],keypoints[37],keypoints[38],
                keypoints[33],keypoints[34],keypoints[35],
                keypoints[42],keypoints[43],keypoints[44],
                keypoints[39],keypoints[40],keypoints[41],
                keypoints[48],keypoints[49],keypoints[50],
                keypoints[45],keypoints[46],keypoints[47]]
    except IndexError:
        keypoints = [ #13 keypoint skeleton mirror
            keypoints[0], keypoints[1], keypoints[2],
            keypoints[6], keypoints[7], keypoints[8],
            keypoints[3], keypoints[4], keypoints[5],
            keypoints[12], keypoints[13], keypoints[14],
            keypoints[9], keypoints[10], keypoints[11],
            keypoints[18], keypoints[19], keypoints[20],
            keypoints[15], keypoints[16], keypoints[17],
            keypoints[24], keypoints[25], keypoints[26],
            keypoints[21], keypoints[22], keypoints[23],
            keypoints[30], keypoints[31], keypoints[32],
            keypoints[27], keypoints[28], keypoints[29],
            keypoints[36], keypoints[37], keypoints[38],
            keypoints[33], keypoints[34], keypoints[35]]
    #print(center, other_centers, other_keypoints)
    center[0] = 1-center[0]

    for i in range(len(other_centers)):
        other_centers[i][0] = 1 - other_centers[i][0]

    for i in range(len(other_keypoints)):
        for j in range(len(other_keypoints[i])):
            other_keypoints[i][j][0] = 1 - other_keypoints[i][j][0]   
    other_keypoints = other_keypoints[::-1]

    label["keypoints"]=keypoints
    label["center"]=center
    label["other_centers"]=other_centers
    label["other_keypoints"]=other_keypoints



    return img,label


def Padding(src, label, pad_color, max_pad_ratio=0.12):
    """
    item = {
             "img_name":save_name,  
             "keypoints":save_keypoints, relative position
             "center":save_center,
             "other_centers":other_centers,
             "other_keypoints":other_keypoints,
            }
    """
    h,w = src.shape[:2]
    max_size = max(h,w)

    pad_ratio = (random.random()+1)*max_pad_ratio/2
    pad_size = int(max_size*pad_ratio)
    new_size = max_size+pad_size*2

    new_img = np.ones((new_size,new_size,3))*pad_color
    new_x0 = (new_size-w)//2
    new_y0 = (new_size-h)//2
    new_img[new_y0:new_y0+h,new_x0:new_x0+w,:] = src

    new_img = cv2.resize(new_img, (w, h))

    if label is None:
        return new_img.astype(np.uint8)


    keypoints = label['keypoints']
    center = label['center']
    other_centers = label['other_centers']
    other_keypoints = label['other_keypoints']
        

    for i in range(len(keypoints)):
        if i%3==0:
            keypoints[i] = (keypoints[i]*w+new_x0)/new_size
        elif i%3==1:
            keypoints[i] = (keypoints[i]*h+new_y0)/new_size


    center[0] = (center[0]*w+new_x0)/new_size
    center[1] = (center[1]*h+new_y0)/new_size

    for i in range(len(other_centers)):
        other_centers[i][0] = (other_centers[i][0]*w+new_x0)/new_size
        other_centers[i][1] = (other_centers[i][1]*h+new_y0)/new_size


    for i in range(len(other_keypoints)):
        for j in range(len(other_keypoints[i])):
            other_keypoints[i][j][0] = (other_keypoints[i][j][0]*w+new_x0)/new_size  
            other_keypoints[i][j][1] = (other_keypoints[i][j][1]*h+new_y0)/new_size


    label["keypoints"]=keypoints
    label["center"]=center
    label["other_centers"]=other_centers
    label["other_keypoints"]=other_keypoints

    
    return new_img.astype(np.uint8), label


def Crop(src, label, pad_color, max_pad_ratio=0.3):
    """
    item = {
             "img_name":save_name,  
             "keypoints":save_keypoints, relative position
             "center":save_center,
             "other_centers":other_centers,
             "other_keypoints":other_keypoints,
            }
    """
    h,w = src.shape[:2]

    keypoints = label['keypoints']
    center = label['center']
    other_centers = label['other_centers']
    other_keypoints = label['other_keypoints']

    pad_ratio = random.uniform(max_pad_ratio/3, max_pad_ratio)
    if len(other_centers)>0: 
        crop_x = int(w*pad_ratio)
        crop_y = int(h*pad_ratio)
    else:
        crop_x = random.randint(int(w*pad_ratio)//2,int(w*pad_ratio))
        crop_y = random.randint(int(w*pad_ratio)//2,int(h*pad_ratio))

    new_w = int(w-w*pad_ratio*2)
    new_h = int(h-h*pad_ratio*2)

    new_img = src[crop_y:crop_y+new_h,crop_x:crop_x+new_w]
    new_img = cv2.resize(new_img, (w, h))
    ##

    if label is None:
        return new_img.astype(np.uint8)



    for i in range(len(keypoints)):
        if i%3==0:
            keypoints[i] = (keypoints[i]*w-crop_x)/new_w
        elif i%3==1:
            keypoints[i] = (keypoints[i]*h-crop_y)/new_h
    for i in range(len(keypoints)//3):
        if keypoints[i*3]<0 or keypoints[i*3]>=1 or keypoints[i*3+1]<0 or keypoints[i*3+1]>=1:
            keypoints[i*3+2] = 0

    center[0] = min(max(0,(center[0]*w-crop_x)/new_w),1)
    center[1] = min(max(0,(center[1]*h-crop_y)/new_h),1)

    for i in range(len(other_centers)):
        other_centers[i][0] = (other_centers[i][0]*w-crop_x)/new_w
        other_centers[i][1] = (other_centers[i][1]*h-crop_y)/new_h
    other_centers_new = []
    for item in other_centers:
        if item[0]<0 or item[0]>=1 or item[1]<0 or item[1]>=1:
            continue
        else:
            other_centers_new.append(item)
    other_centers = other_centers_new

    for i in range(len(other_keypoints)):
        for j in range(len(other_keypoints[i])):
            other_keypoints[i][j][0] = (other_keypoints[i][j][0]*w-crop_x)/new_w
            other_keypoints[i][j][1] = (other_keypoints[i][j][1]*h-crop_y)/new_h
        other_keypoints_res = []
        for item in other_keypoints[i]:
            if item[0]<0 or item[0]>=1 or item[1]<0 or item[1]>=1:
                continue
            else:
                other_keypoints_res.append(item)
        other_keypoints[i] = other_keypoints_res

    label["keypoints"]=keypoints
    label["center"]=center
    label["other_centers"]=other_centers
    label["other_keypoints"]=other_keypoints

    return new_img.astype(np.uint8), label


def Move(src, label, pad_color, max_move_ratio=0.2):
    """
    item = {
             "img_name":save_name,  
             "keypoints":save_keypoints, relative position
             "center":save_center,
             "other_centers":other_centers,
             "other_keypoints":other_keypoints,
            }
    """
    h,w = src.shape[:2]

    keypoints = label['keypoints']
    center = label['center']

    pad_ratio = random.uniform(max_move_ratio/2, max_move_ratio)

    move_x = random.randint(int(w*pad_ratio)//2,int(w*pad_ratio))*random.choice([-1,1])
    move_y = random.randint(int(w*pad_ratio)//2,int(h*pad_ratio))*random.choice([-1,1])


    M = np.float32([[1, 0, move_x], [0, 1, move_y]])
    src = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]))


    # new_img = src[move_y:move_y+new_h,move_x:move_x+new_w]


    for i in range(len(keypoints)):
        if i%3==0:
            keypoints[i] = keypoints[i]+move_x/w
        elif i%3==1:
            keypoints[i] = keypoints[i]+move_y/h
    for i in range(len(keypoints)//3):
        if keypoints[i*3]<0 or keypoints[i*3]>=1 or keypoints[i*3+1]<0 or keypoints[i*3+1]>=1:
            keypoints[i*3+2] = 0

    center[0] = min(max(0,(center[0]+move_x/w)),1)
    center[1] = min(max(0,(center[1]+move_y/h)),1)


    label["keypoints"]=keypoints
    label["center"]=center


    return src.astype(np.uint8), label

def Rotate(src,angle,pad_color,label=None,center=None,scale=1.0):
    '''
    :param src: src image
    :param label: dict
    :param angle:
    :param center:
    :param scale:
    :return: the rotated image and the points
    '''
    image=src
    (h, w) = image.shape[:2]

    
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    if label is None:
        for i in range(image.shape[2]):
            image[:,:,i] = cv2.warpAffine(image[:,:,i], M, (w, h),
                                          flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=pad_color)
        return image,None
    else:

        keypoints = label['keypoints']
        center = label['center']
        other_centers = label['other_centers']
        other_keypoints = label['other_keypoints']

        
        ####make it as a 3x3 RT matrix
        full_M=np.row_stack((M,np.asarray([0,0,1])))
        img_rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=pad_color)
        ###make the keypoints as 3xN matrix
        keypoints = np.reshape(keypoints, (-1,3))
        keypoints_z = keypoints[:,2].reshape((-1,1))
        keypoints = keypoints[:,:2]
        keypoints[:,0] *= w
        keypoints[:,1] *= h
        keypoints = keypoints.astype(np.int32)

        keypoints = keypoints.T
        full_keypoints = np.row_stack((keypoints, np.ones(shape=(1,keypoints.shape[1]))))
        keypoints_rotated=np.dot(full_M,full_keypoints)
        keypoints_rotated=keypoints_rotated[0:2,:]
        #keypoints_rotated = keypoints_rotated.astype(np.int32)
        keypoints_rotated=keypoints_rotated.T

        keypoints_rotated = keypoints_rotated.astype(np.float32)
        keypoints_rotated[:,0] /=w
        keypoints_rotated[:,1] /=h
        # keypoints_rotated = np.reshape(keypoints_rotated, (-1))
        keypoints = np.concatenate([keypoints_rotated,keypoints_z],-1).reshape((-1)).tolist()
        for i in range(len(keypoints)//3):
            if keypoints[i*3]<0 or keypoints[i*3]>=1 or keypoints[i*3+1]<0 or keypoints[i*3+1]>=1:
                keypoints[i*3+2] = 0

        center = np.reshape(center, (-1,2))
        center[:,0] *= w
        center[:,1] *= h
        center = center.astype(np.int32)

        center = center.T
        full_center = np.row_stack((center, np.ones(shape=(1,center.shape[1]))))
        center_rotated=np.dot(full_M,full_center)
        center_rotated=center_rotated[0:2,:]
        #keypoints_rotated = keypoints_rotated.astype(np.int32)
        center_rotated=center_rotated.T

        center_rotated = center_rotated.astype(np.float32)
        center_rotated[:,0] /=w
        center_rotated[:,1] /=h
        # keypoints_rotated = np.reshape(keypoints_rotated, (-1))
        center = center_rotated.reshape((-1)).tolist()



        other_centers = np.reshape(other_centers, (-1,2))
        other_centers[:,0] *= w
        other_centers[:,1] *= h
        other_centers = other_centers.astype(np.int32)

        other_centers = other_centers.T
        full_center = np.row_stack((other_centers, np.ones(shape=(1,other_centers.shape[1]))))
        center_rotated=np.dot(full_M,full_center)
        center_rotated=center_rotated[0:2,:]
        #keypoints_rotated = keypoints_rotated.astype(np.int32)
        center_rotated=center_rotated.T

        center_rotated = center_rotated.astype(np.float32)
        center_rotated[:,0] /= w
        center_rotated[:,1] /= h
        # keypoints_rotated = np.reshape(keypoints_rotated, (-1))
        other_centers_raw = center_rotated.reshape((-1,2)).tolist()
        other_centers = []
        for item in other_centers_raw:
            if item[0]<0 or item[0]>=1 or item[1]<0 or item[1]>=1:
                continue
            else:
                other_centers.append(item)


        for i in range(len(other_keypoints)):
            if len(other_keypoints[i])>0:
                other_keypointsi = np.reshape(other_keypoints[i], (-1,2))
                other_keypointsi[:,0] *= w
                other_keypointsi[:,1] *= h
                other_keypointsi = other_keypointsi.astype(np.int32)

                other_keypointsi = other_keypointsi.T
                full_center = np.row_stack((other_keypointsi, np.ones(shape=(1,other_keypointsi.shape[1]))))
                center_rotated=np.dot(full_M,full_center)
                center_rotated=center_rotated[0:2,:]
                #keypoints_rotated = keypoints_rotated.astype(np.int32)
                center_rotated=center_rotated.T

                center_rotated = center_rotated.astype(np.float32)
                center_rotated[:,0] /=w
                center_rotated[:,1] /=h
                # keypoints_rotated = np.reshape(keypoints_rotated, (-1))
                other_keypoints_i = center_rotated.reshape((-1,2)).tolist()
                other_keypoints_res = []
                for item in other_keypoints_i:
                    if item[0]<0 or item[0]>=1 or item[1]<0 or item[1]>=1:
                        continue
                    else:
                        other_keypoints_res.append(item)
                other_keypoints[i] = other_keypoints_res

        label["keypoints"]= keypoints
        label["center"]=center
        label["other_centers"]=other_centers
        label["other_keypoints"]=other_keypoints


        return img_rotated, label


def Affine(src,strength,pad_color,label=None):
    image = src
    (h, w) = image.shape[:2]


    keypoints = label['keypoints']
    center = label['center']
    other_centers = label['other_centers']
    other_keypoints = label['other_keypoints']


    pts_base = np.float32([[10,100],[200,50],[100,250]])
    pts1 = np.random.rand(3, 2) * -40 + pts_base
    pts1 = pts1.astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts_base)
    trans_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]) ,
                                            borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=pad_color)
    if label is None:
        return trans_img, label

    keypoints = np.reshape(keypoints, (-1,3))
    keypoints_z = keypoints[:,2].reshape((-1,1))
    keypoints = keypoints[:,:2]
    keypoints[:,0] *= w
    keypoints[:,1] *= h
    keypoints = keypoints.astype(np.int32)

    keypoints=keypoints.T
    full_data = np.row_stack((keypoints, np.ones(shape=(1, keypoints.shape[1]))))
    data_rotated = np.dot(M, full_data)
    #label_rotated = label_rotated.astype(np.int32)
    data_rotated=data_rotated.T

    data_rotated = data_rotated.astype(np.float32)
    data_rotated[:,0] /=w
    data_rotated[:,1] /=h
    keypoints = np.concatenate([data_rotated,keypoints_z],-1).reshape((-1)).tolist()
    for i in range(len(keypoints)//3):
        if keypoints[i*3]<0 or keypoints[i*3]>=1 or keypoints[i*3+1]<0 or keypoints[i*3+1]>=1:
            keypoints[i*3+2] = 0


    center = np.reshape(center, (-1,2))
    center[:,0] *= w
    center[:,1] *= h
    center = center.astype(np.int32)

    center=center.T
    full_data = np.row_stack((center, np.ones(shape=(1, center.shape[1]))))
    data_rotated = np.dot(M, full_data)
    #label_rotated = label_rotated.astype(np.int32)
    data_rotated=data_rotated.T

    data_rotated = data_rotated.astype(np.float32)
    data_rotated[:,0] /=w
    data_rotated[:,1] /=h
    center = data_rotated.reshape((-1)).tolist()



    other_centers = np.reshape(other_centers, (-1,2))
    other_centers[:,0] *= w
    other_centers[:,1] *= h
    other_centers = other_centers.astype(np.int32)

    other_centers = other_centers.T
    full_data = np.row_stack((other_centers, np.ones(shape=(1, other_centers.shape[1]))))
    data_rotated = np.dot(M, full_data)
    #label_rotated = label_rotated.astype(np.int32)
    data_rotated=data_rotated.T

    data_rotated = data_rotated.astype(np.float32)
    data_rotated[:,0] /=w
    data_rotated[:,1] /=h
    other_centers_raw = data_rotated.reshape((-1,2)).tolist()
    other_centers = []
    for item in other_centers_raw:
        if item[0]<0 or item[0]>=1 or item[1]<0 or item[1]>=1:
            continue
        else:
            other_centers.append(item)


    for i in range(len(other_keypoints)):
        if len(other_keypoints[i])>0:
            other_keypointsi = np.reshape(other_keypoints[i], (-1,2))
            other_keypointsi[:,0] *= w
            other_keypointsi[:,1] *= h
            other_keypointsi = other_keypointsi.astype(np.int32)

            other_keypointsi = other_keypointsi.T
            full_data = np.row_stack((other_keypointsi, np.ones(shape=(1, other_keypointsi.shape[1]))))
            data_rotated = np.dot(M, full_data)
            #label_rotated = label_rotated.astype(np.int32)
            data_rotated=data_rotated.T

            data_rotated = data_rotated.astype(np.float32)
            data_rotated[:,0] /=w
            data_rotated[:,1] /=h
            # other_keypoints[i] = data_rotated.reshape((-1,2)).tolist()
            other_keypoints_i = data_rotated.reshape((-1,2)).tolist()
            other_keypoints_res = []
            for item in other_keypoints_i:
                if item[0]<0 or item[0]>=1 or item[1]<0 or item[1]>=1:
                    continue
                else:
                    other_keypoints_res.append(item)
            other_keypoints[i] = other_keypoints_res


    label["keypoints"]=keypoints
    label["center"]=center
    label["other_centers"]=other_centers
    label["other_keypoints"]=other_keypoints

    return trans_img, label


def AID(img, label):
    h,w = img.shape[:2]

    half_size = int(random.uniform(3/192,6/192)*(h+w)/2)


    keypoints = np.array(label["keypoints"]).reshape((-1,3))

    keypoints = keypoints[keypoints[:,2]>0]
    dropout_id = random.randint(0,keypoints.shape[0]-1)


    cx = keypoints[dropout_id][0]*w
    cy = keypoints[dropout_id][1]*h

    x0 = int(max(0,cx-half_size))
    y0 = int(max(0,cy-half_size))
    x1 = int(min(w-1,cx+half_size))
    y1 = int(min(h-1,cy+half_size))

    color = random.randint(0,255)
    img[y0:y1, x0:x1] = (color,color,color)

    return img

def AID2(img, label):
    h,w = img.shape[:2]

    half_size = int(random.uniform(3/192,6/192)*(h+w)/2)


    keypoints = np.array(label["keypoints"]).reshape((-1,3))

    keypoints = keypoints[keypoints[:,2]>0]
    dropout_id = random.randint(0,keypoints.shape[0]-1)


    cx = keypoints[dropout_id][0]*w
    cy = keypoints[dropout_id][1]*h

    x0 = int(max(0,cx-half_size))
    y0 = int(max(0,cy-half_size))
    x1 = int(min(w-1,cx+half_size))
    y1 = int(min(h-1,cy+half_size))
    # print(x0,y0,x1,y1)
    # b
    color = random.randint(0,255)
    dist = half_size**2
    for i in range(y1-y0+1):
        for j in range(x1-x0+1):
            px = x0+j
            py = y0+i
            if ((py-cy)**2+(px-cx)**2)<=dist:
                img[py, px] = (color,color,color)

    return img

def dropout(src,pad_color,max_pattern_ratio=0.05):
    width_ratio = random.uniform(0, max_pattern_ratio)
    height_ratio = random.uniform(0, max_pattern_ratio)
    width=src.shape[1]
    height=src.shape[0]
    block_width=width*width_ratio
    block_height=height*height_ratio
    width_start=int(random.uniform(0,width-block_width))
    width_end=int(width_start+block_width)
    height_start=int(random.uniform(0,height-block_height))
    height_end=int(height_start+block_height)
    src[height_start:height_end,width_start:width_end,:]=np.array(pad_color,dtype=src.dtype)

    return src

def pixel_jitter(src,p=0.5,max_=5.):

    src=src.astype(np.float32)

    pattern=(np.random.rand(src.shape[0], src.shape[1],src.shape[2])-0.5)*2*max_
    img = src + pattern

    img[img<0]=0
    img[img >255] = 255

    img = img.astype(np.uint8)

    return img

def _clip(image):
    """
    Clip and convert an image to np.uint8.
    Args
        image: Image to clip.
    """
    return np.clip(image, 0, 255).astype(np.uint8)

def adjust_contrast(image, factor):
    """ Adjust contrast of an image.
    Args
        image: Image to adjust.
        factor: A factor for adjusting contrast.
    """
    mean = image.mean(axis=0).mean(axis=0)
    return _clip((image - mean) * factor + mean)

def adjust_brightness(image, delta):
    """ Adjust brightness of an image
    Args
        image: Image to adjust.
        delta: Brightness offset between -1 and 1 added to the pixel values.
    """
    return _clip(image + delta * 255)

def adjust_hue(image, delta):
    """ Adjust hue of an image.
    Args
        image: Image to adjust.
        delta: An interval between -1 and 1 for the amount added to the hue channel.
               The values are rotated if they exceed 180.
    """
    image[..., 0] = np.mod(image[..., 0] + delta * 180, 180)
    return image

def adjust_saturation(image, factor):
    """ Adjust saturation of an image.
    Args
        image: Image to adjust.
        factor: An interval for the factor multiplying the saturation values of each pixel.
    """
    image[..., 1] = np.clip(image[..., 1] * factor, 0, 255)
    return image

class ColorDistort():
    def __init__(
            self,
            contrast_range=(0.8, 1.2),
            brightness_range=(-.2, .2),
            hue_range=(-0.1, 0.1),
            saturation_range=(0.8, 1.2)
    ):
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.hue_range = hue_range
        self.saturation_range = saturation_range


    def _uniform(self,val_range):
        """ Uniformly sample from the given range.
        Args
            val_range: A pair of lower and upper bound.
        """
        return np.random.uniform(val_range[0], val_range[1])

    def __call__(self, image):


        if self.contrast_range is not None:
            contrast_factor = self._uniform(self.contrast_range)
            image = adjust_contrast(image,contrast_factor)
        if self.brightness_range is not None:
            brightness_delta = self._uniform(self.brightness_range)
            image = adjust_brightness(image, brightness_delta)

        if self.hue_range is not None or self.saturation_range is not None:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            if self.hue_range is not None:
                hue_delta = self._uniform(self.hue_range)
                image = adjust_hue(image, hue_delta)

            if self.saturation_range is not None:
                saturation_factor = self._uniform(self.saturation_range)
                image = adjust_saturation(image, saturation_factor)

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image



###### Data Aug api
class DataAug:
    def __init__(self, img_size):
        self.h = img_size
        self.w = img_size
        self.color_augmentor = ColorDistort()

    def __call__(self, img, label):
        """
        img: opencv img, BGR
        item = {
                     "img_name":save_name,  
                     "keypoints":save_keypoints, relative position
                     "center":save_center,
                     "other_centers":other_centers,
                     "other_keypoints":other_keypoints,
                    }
        return: same type as img and label
        """
        pad_color = random.randint(0,255)


        new_img = deepcopy(img)
        new_label = deepcopy(label)

        
        if random.random() < 0.5:
            new_img, new_label = Mirror(new_img, label=new_label)
        

        rd = random.random()
        if rd <0.5:
            if random.random() < 0.2:
                new_img, new_label = Padding(new_img, new_label, (0,0,0))
            else:
                new_img, new_label = Crop(new_img, new_label, (0,0,0))
        elif rd < 0.65:
            if random.random() < 0.7:
                if len(new_label["other_centers"])==0:
                    strength = random.uniform(20, 50)
                    new_img, new_label = Affine(new_img, strength=strength, label=new_label, pad_color=(0,0,0))
                
            else:
                angle = random.uniform(-24, 24)
                new_img, new_label = Rotate(new_img, label=new_label, angle=angle, pad_color=(0,0,0))
           
        elif rd < 0.85:
            
            if len(new_label["other_centers"])==0 and (len(new_label["other_keypoints"][0]) + len(new_label["other_keypoints"][3]))==0:
                if random.random() < 0.5:
                    new_img, new_label = Move(new_img, new_label, (0,0,0))
        else:
            pass
        ### in case of no points
        count_zero = 0
        count_zero_new = 0
        for i in range(len(new_label["keypoints"])):
            if i%3==2:
                if label["keypoints"][i]==0:
                    count_zero+=1
                if new_label["keypoints"][i]==0:
                    count_zero_new+=1
        if count_zero_new-count_zero > 3 or count_zero_new>5:
            new_img = img
            new_label = label




        if random.random() < 0.4:
            new_img = self.color_augmentor(new_img)

        if random.random() < 0.4:
            new_img = pixel_jitter(new_img,15)

        # if random.random() < 0.3:
        #     new_img = dropout(new_img, pad_color,0.1)

        if random.random() < 0.3:
            
            new_img = AID(new_img, new_label)

            
        return new_img, new_label




