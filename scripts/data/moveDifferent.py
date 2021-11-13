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





def main(dir_path, label_path, save_dir):
    # make all

    img_names = getAllName(dir_path)
    print("total img_names: ", len(img_names))
    

    label_names = getAllName(label_path)
    print("total label_names: ", len(label_names))
    label_names = set([os.path.basename(name) for name in label_names])



    for i,img_name in enumerate(img_names):

        if i%5000==0:
            print(i, len(img_names))

        name = os.path.basename(img_name)


        if name in label_names:
            os.rename(img_name, os.path.join(save_dir,name))



    



if __name__ == '__main__':

    dir_path = "show"
    label_path = "exam"
    save_dir = "tmp"

    print("Strat: ", dir_path)
    main(dir_path, label_path, save_dir)
    print("------------------")