"""
@Fire
https://github.com/fire717
"""
import numpy as np

# from config import cfg


def getDist(pre, labels,num_classes = 13):
    """
    input:
            pre: [batchsize, 14]
            labels: [batchsize, 14]
    return:
            dist: [batchsize, 7]
    """
    pre = pre.reshape([-1, num_classes, 2])
    labels = labels.reshape([-1, num_classes, 2])
    res = np.power(pre[:, :, 0] - labels[:, :, 0], 2) + np.power(pre[:, :, 1] - labels[:, :, 1], 2)
    return res


def getAccRight(dist, th):
    """
    input:
            dist: [batchsize, 7]

    return:
            dist: [7,]    right count of every point
    """
    res = np.zeros(dist.shape[1], dtype=np.int64)
    for i in range(dist.shape[1]):
        res[i] = sum(dist[:, i] < th)

    return res


def myAcc(output, target, th=0.5, num_classes = 13):
    """
    return [7,] ndarray
    """
    # print(output.shape) 
    # 64, 7, 40, 40     gaussian
    # (64, 14)                !gaussian
    # b
    # if hm_type == 'gaussian':
    if len(output.shape) == 4:
        output = heatmap2locate(output)
        target = heatmap2locate(target)

    # offset方式还原坐标
    # [h, ls, rs, lb, rb, lr, rr]
    # output[:,6:10] = output[:,6:10]+output[:,2:6]
    # output[:,10:14] = output[:,10:14]+output[:,6:10]

    dist = getDist(output, target,num_classes)
    cate_acc = getAccRight(dist,th)
    return cate_acc


def pck(output, target, limb_length, threshold=None, num_classes=13, mode='head'):
    if threshold is None:
        if mode == 'head': # for MPII dataset
            threshold = 0.5
        elif mode == 'torso': # for H36M and DHP19
            threshold = 0.2
        else:
            print("Invalid accuracy model")
            return None

    if len(output.shape) == 4:
        output = heatmap2locate(output)
        target = heatmap2locate(target)

    pck={}
    # print(torso_diameter)
    # compute PCK's threshold as percentage of head size in pixels for each pose


    # compute euclidean distances between joints
    output = output.reshape((-1,num_classes,2))
    target = target.reshape((-1,num_classes,2))
    distances = np.linalg.norm( output - target , axis=2)

    try:
        thresholds_val = limb_length * threshold
    except TypeError:
        print('The data type of the length of thresholding limb is incompatible:', type(limb_length))
        print('All thresholds in the batch set to zero.')
        thresholds_val = np.zeros([output.shape[0]])


    # compute correct keypoints
    th_head_expanded = thresholds_val.unsqueeze(1).expand(-1,distances.shape[1])
    correct_keypoints = (distances <= th_head_expanded.numpy()).astype(int)

    # remove not annotated keypoints from pck computation
    correct_keypoints = correct_keypoints * (target[:, :, 0] != -1).astype(int)
    annotated_keypoints_num_samples = np.sum((target[:, :, 0] != -1).astype(int), axis=1)
    annotated_keypoints_num_joints = np.sum((target[:, :, 0] != -1).astype(int), axis=0)
    # print("correct_keypoints: ",correct_keypoints.shape)
    # print("annotated_keypoints_num: ",annotated_keypoints_num.shape)

    # compute pck in all different formats
    pck_per_joint = np.sum(correct_keypoints, axis=0) / annotated_keypoints_num_joints
    pck_per_sample = np.sum(correct_keypoints, axis=1) / annotated_keypoints_num_samples
    # pck_per_keypoint = sum(sum(correct_keypoints))/ sum(annotated_keypoints_num_joints)
    pck["correct_per_sample"] = np.sum(correct_keypoints, axis=1)
    pck["correct_per_joint"] = np.sum(correct_keypoints, axis=0)
    pck["per_joint_mean"] = np.mean(pck_per_joint)
    pck["per_sample_mean"] = np.mean(pck_per_sample)
    pck["total_keypoints"] = sum(annotated_keypoints_num_joints)
    pck["anno_keypoints_per_joint"] = annotated_keypoints_num_joints
    pck["anno_keypoints_per_sample"] = annotated_keypoints_num_samples
    pck["total_correct"] = sum(sum(correct_keypoints))

    # print(pck)
    # print(pck_joints)

    return pck
