# Movenet.Pytorch

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fire717/Fire/blob/main/LICENSE)

## Intro
![start](/data/imgs/three_pane_aligned.gif)

MoveNet is an ultra fast and accurate model that detects 17 keypoints of a body.
This is A Pytorch implementation of MoveNet from Google. Include training code and pre-train model.

Google just release pre-train models(tfjs or tflite), which cannot be converted to some CPU inference framework such as NCNN,Tengine,MNN,TNN, and we can not add our own custom data to finetune, so there is this repo.


## How To Run

1.Download COCO dataset2017 from https://cocodataset.org/. (You need train2017.zip, val2017.zip and annotations.)Unzip to `movenet.pytorch/data/` like this:

```
├── data
    ├── annotations (person_keypoints_train2017.json, person_keypoints_val2017.json, ...)
    ├── train2017   (xx.jpg, xx.jpg,...)
    └── val2017     (xx.jpg, xx.jpg,...)

```


2.Make data to our data format.
```
python scripts/make_coco_data_17keypooints.py
```
```
Our data format: JSON file
Keypoints order:['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 
    'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 
    'right_ankle']

One item:
[{"img_name": "0.jpg",
  "keypoints": [x0,y0,z0,x1,y1,z1,...],
  #z: 0 for no label, 1 for labeled but invisible, 2 for labeled and visible
  "center": [x,y],
  "bbox":[x0,y0,x1,y1],
  "other_centers": [[x0,y0],[x1,y1],...],
  "other_keypoints": [[[x0,y0],[x1,y1],...],[[x0,y0],[x1,y1],...],...], #lenth = num_keypoints
 },
 ...
]
```

3.You can add your own data to the same format.

4.Setup an initial point for the trainings
```
python initialize.py
```
5.After putting data at right place, you can start training
```
python train.py
```

6.After training finished, you need to change the test model path to test. Such as this in predict.py
```
run_task.modelLoad("output/xxx.pth")
```


7.run predict to show predict result, or run evaluate.py to compute my acc on test dataset.
```
python predict.py
```
8.Convert to onnx.
```
python pth2onnx.py
```

## Training Results

#### Some good samples
![good](/data/imgs/good.png)

#### Some bad cases
![bad](/data/imgs/bad.png)


## Tips to improve
#### 1. Focus on data
* Add COCO2014. (But as I know it has some duplicate data of COCO2017, and I don't know if google use it.)
* Clean the croped COCO2017 data. (Some img just have little points, such as big face, big body,etc.MoveNet is a small network, COCO data is a little hard for it.)
* Add some yoga, fitness, and dance videos frame from YouTube. (Highly Recommened! Cause Google did this on their Movenet and said 'Evaluations on the Active validation dataset show a significant performance boost relative to identical architectures trained using only COCO. ')

#### 2. Change backbone
Try to ransfer Mobilenetv2(original Movenet) to Mobilenetv3 or Shufflenetv2 may get a litte improvement.If you just wanna reproduce the original Movenet, u can ignore this.

#### 3. More fancy loss
Surely this is a muti-task learning. So add some loss to learn together may improve the performence. (Such as BoneLoss which I have added.) And we can never know how Google trained, cause we cannot see it from the pre-train tflite model file, so you can try any loss function you like.


#### 4. Data Again
I just wanna you know the importance of the data. The more time you spend on clean data and add new data, the better performance your model will get! (While tips 2 and 3 may not.)

## Resource
1. [Blog:Next-Generation Pose Detection with MoveNet and TensorFlow.js](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html
)
2. [model card](https://storage.googleapis.com/movenet/MoveNet.SinglePose%20Model%20Card.pdf)
3. [TFHub：movenet/singlepose/lightning
](https://tfhub.dev/google/movenet/singlepose/lightning/4
)
4. [My article share: 2021轻量级人体姿态估计模型修炼之路（附谷歌MoveNet复现经验）](https://zhuanlan.zhihu.com/p/413313925)
