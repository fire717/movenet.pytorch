"""
@Fire
https://github.com/fire717
"""
import os
import cv2
import numpy as np
from PIL import Image

read_dir = "imgs"
save_dir = "imgs192pil"



if __name__ == "__main__":


    img_names = os.listdir(read_dir)
    print("total: ", len(img_names))

    for i,img_name in enumerate(img_names):
        if i%5000==0:
            print(i)
        img_path = os.path.join(read_dir, img_name)
        save_path = os.path.join(save_dir, img_name)


        img = cv2.imread(img_path)

        # img = cv2.resize(img, (192,192))

        img = Image.fromarray(img)
        img = img.resize((192,192))
        img = np.array(img)

        cv2.imwrite(save_path, img)
