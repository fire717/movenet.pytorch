"""
@Fire
https://github.com/fire717
"""
import os
import cv2


def video2img(video_path, save_dir):
    cap = cv2.VideoCapture(video_path)

    basename = os.path.basename(video_path)[:-4]
    save_dir = os.path.join(save_dir,basename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    idx = 0
    while(cap.isOpened()):
        if idx%1000==0:
            print(idx)

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if idx%4==1:
            idx+=1
            continue

        save_name = os.path.join(save_dir,basename+"_%d.jpg" % idx)
        cv2.imwrite(save_name, frame)

        idx+=1


video_dir = './video2/'
save_dir = './imgs/'

video_names = os.listdir(video_dir)
print("Total video :", len(video_names))

for i,video_name in enumerate(video_names):
    print("start %d/%d :%s " % (i+1, len(video_names),video_name))
    #video_name = '86460431_nb2-1-64.flv'

    video_path = os.path.join(video_dir, video_name)
    video2img(video_path, save_dir)

