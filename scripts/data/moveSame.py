"""
@Fire
https://github.com/fire717
"""


import os


def getAllName(file_dir, tail_list = ['.jpg','.png','.jpeg']): 
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in tail_list:
                L.append(os.path.join(root, file))
    return L





def main(read_dir, img_dir, label_dir, save_dir):
    # make all

    move_names = os.listdir(read_dir)
    basenames = set([x.split('.')[0] for x in move_names])
    print(len(basenames))


    img_names = os.listdir(img_dir)
    for img_name in img_names:
        if img_name.split('.')[0] in basenames:
            os.rename(os.path.join(img_dir,img_name), os.path.join(save_dir,img_name))
    

    txt_names = os.listdir(label_dir)
    for txt_name in txt_names:
        if txt_name.split('.')[0] in basenames:
            os.rename(os.path.join(label_dir,txt_name), os.path.join(save_dir,txt_name))
    



    



if __name__ == '__main__':

    read_dir = "tmp"

    img_dir = "../crop_imgs/70713034_nb2-1-32"
    label_dir = "txt"
    save_dir = "clean"


    main(read_dir, img_dir, label_dir, save_dir)
    print("------------------")