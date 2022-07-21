"""
@Fire
https://github.com/fire717
"""
import gc
import os
import torch
import numpy as np
import cv2
from pathlib import Path
import json
import time
import csv

# import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from lib.task.task_tools import getSchedu, getOptimizer, movenetDecode, clipGradient, restore_sizes,superimpose
from lib.loss.movenet_loss import MovenetLoss
from lib.utils.utils import printDash, ensure_loc
from lib.visualization.visualization import superimpose_pose, add_skeleton, movenet_to_hpecore
from lib.utils.metrics import myAcc, pck


class Task():
    def __init__(self, cfg, model):

        self.cfg = cfg
        self.init_epoch = 0
        # if self.cfg['GPU_ID'] != '' :
        #     self.device = torch.device("cuda")
        # else:
        #     self.device = torch.device("cpu")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # edit for Franklin
        self.model = model.to(self.device)

        ############################################################
        # loss
        self.loss_func = MovenetLoss(self.cfg)

        # optimizer
        self.optimizer = getOptimizer(self.cfg['optimizer'],
                                      self.model,
                                      self.cfg['learning_rate'],
                                      self.cfg['weight_decay'])

        self.val_losses = np.zeros([20])
        self.early_stop = not(cfg.get('no_early_stop',True))
        self.early_stop_counter = 0
        self.val_loss_best = np.Inf

        # scheduler
        self.scheduler = getSchedu(self.cfg['scheduler'], self.optimizer)

        # tensorboard
        self.tb = SummaryWriter(comment=self.cfg['label'])
        self.tb.add_graph(self.model, torch.randn(1, 3, 192, 192).to(self.device))
        self.tb.add_text("Hyperparameters: ", str(cfg))
        self.best_train_accuracy = 0
        self.best_val_accuracy = 0

        ensure_loc(os.path.join(self.cfg['save_dir'], self.cfg['label']))

    def train(self, train_loader, val_loader):

        for epoch in range(self.init_epoch, self.init_epoch + self.cfg['epochs']):
            self.onTrainStep(train_loader, epoch)
            self.onValidation(val_loader, epoch)
            if self.early_stop:
                if self.early_stop_counter > 10:
                    break

        self.onTrainEnd()

    def predict(self, data_loader, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.model.eval()
        correct = 0
        size = self.cfg["img_size"]
        with torch.no_grad():

            for (img, img_name) in data_loader:

                # if "yoga_img_483" not in img_name[0]:
                #     continue

                # print(img.shape, img_name)
                img = img.to(self.device)

                output = self.model(img)
                # print(len(output))

                pre = movenetDecode(output, None, mode='output', num_joints=self.cfg["num_classes"])
                # print(pre)
                # print(pre.shape)

                basename = os.path.basename(img_name[0])
                img = np.transpose(img[0].cpu().numpy(), axes=[1, 2, 0])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                h, w = img.shape[:2]

                cv2.imwrite(os.path.join(save_dir, basename[:-4] + "_img.jpg"), img)

                for i in range(len(pre[0]) // 2):
                    x = int(pre[0][i * 2] * w)
                    y = int(pre[0][i * 2 + 1] * h)
                    cv2.circle(img, (x, y), 3, (255, 0, 0), 2)

                cv2.imwrite(os.path.join(save_dir, basename), img)

                ## debug
                heatmaps = output[0].cpu().numpy()[0]
                centers = output[1].cpu().numpy()[0]
                regs = output[2].cpu().numpy()[0]
                offsets = output[3].cpu().numpy()[0]

                # print(centers.shape)

                hm = cv2.resize(np.sum(heatmaps, axis=0), (size, size)) * 255
                cv2.imwrite(os.path.join(save_dir, basename[:-4] + "_heatmaps.jpg"), hm)
                # img[:, :, 0] += hm
                cv2.imwrite(os.path.join(save_dir, basename[:-4] + "_center.jpg"),
                            cv2.resize(centers[0] * 255, (size, size)))
                cv2.imwrite(os.path.join(save_dir, basename[:-4] + "_regs0.jpg"),
                            cv2.resize(regs[0] * 255, (size, size)))

    def label(self, data_loader, save_dir):
        self.model.eval()

        txt_dir = os.path.join(save_dir, 'txt')
        show_dir = os.path.join(save_dir, 'show')

        with torch.no_grad():

            for (img, img_path) in data_loader:
                # print(img.shape, img_path)
                img_path = img_path[0]
                basename = os.path.basename(img_path)

                img = img.to(self.device)

                output = self.model(img)
                # print(len(output))

                pre = movenetDecode(output, None, mode='output')[0]
                # print(pre)
                with open(os.path.join(txt_dir, basename[:-3] + 'txt'), 'w') as f:
                    f.write("7\n")
                    for i in range(len(pre) // 2):
                        vis = 2
                        if pre[i * 2] == -1:
                            vis = 0
                        line = "%f %f %d\n" % (pre[i * 2], pre[i * 2 + 1], vis)
                        f.write(line)

                img = np.transpose(img[0].cpu().numpy(), axes=[1, 2, 0])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                h, w = img.shape[:2]

                for i in range(len(pre) // 2):
                    x = int(pre[i * 2] * w)
                    y = int(pre[i * 2 + 1] * h)
                    cv2.circle(img, (x, y), 3, (255, 0, 0), 2)

                cv2.imwrite(os.path.join(show_dir, basename), img)

                # b

    def exam(self, data_loader, save_dir):
        self.model.eval()
        ensure_loc(save_dir)

        with torch.no_grad():
            for batch_idx, (
            imgs, labels, kps_mask, img_names, torso_diameter, head_size_norm, img_size_original, ts) in enumerate(
                    data_loader):
                if img_size_original == 0:
                    continue

                if batch_idx % 50 == 0 and batch_idx > 0:
                    print('Finish ', batch_idx)
                # if 'mypc'  not in img_names[0]:
                #     continue

                # print('-----------------')

                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                size = self.cfg["img_size"]
                text_location = (10, size * 2 - 10)  # bottom left corner of the image
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.8
                fontColor = (0, 0, 255)
                thickness = 1
                lineType = 2

                output = self.model(imgs)

                pre = movenetDecode(output, kps_mask, mode='output', num_joints=self.cfg["num_classes"])
                gt = movenetDecode(labels, kps_mask, mode='label', num_joints=self.cfg["num_classes"])

                if self.cfg['dataset'] in ['coco', 'mpii','mpii2']:
                    pck_acc = pck(pre, gt, head_size_norm, num_classes=self.cfg["num_classes"], mode='head')
                    th_val = head_size_norm
                else:
                    pck_acc = pck(pre, gt, torso_diameter, num_classes=self.cfg["num_classes"], mode='torso')
                    th_val = torso_diameter
                # print(pck)
                # print(correct,total)

                correct_kps = pck_acc["total_correct"]
                total_kps = pck_acc["total_keypoints"]
                # joint_correct += pck["correct_per_joint"]
                # joint_total += pck["anno_keypoints_per_joint"]

                # if 'mypc1_full_1180' in img_names[0]:
                if 1:
                    # if 0 / 7 < sum(acc) / len(acc) <= 5 / 7:
                    # if sum(acc)/len(acc)==1:
                    # print(pre)
                    # print(gt)
                    # print(acc)
                    img_name = img_names[0]
                    # print(img_name)

                    basename = os.path.basename(img_name)
                    save_name = os.path.join(save_dir, basename)

                    # print(os.path.join(save_dir, basename[:-4] + "_hm_pre.jpg"))
                    # hm = cv2.resize(np.sum(output[0][0].cpu().numpy(), axis=0), (size, size)) * 255
                    # cv2.imwrite(os.path.join(save_dir, basename[:-4] + "_hm_pre.jpg"), hm)
                    # cv2.imshow("prediction",hm)
                    # cv2.waitKey()

                    # hm = cv2.resize(np.sum(labels[0, :, :, :].cpu().numpy(), axis=0), (size, size)) * 255
                    # cv2.imwrite(os.path.join(save_dir, basename[:-4] + "_hm_gt.jpg"), hm)

                    img = np.transpose(imgs[0].cpu().numpy(), axes=[1, 2, 0])
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    h, w = img.shape[:2]

                    for i in range(len(gt[0]) // 2):
                        x = int(gt[0][i * 2] * w)
                        y = int(gt[0][i * 2 + 1] * h)
                        cv2.circle(img, (x, y), 2, (0, 255, 0), 1)  # gt keypoints in green

                        x = int(pre[0][i * 2] * w)
                        y = int(pre[0][i * 2 + 1] * h)
                        cv2.circle(img, (x, y), 2, (0, 0, 255), 1)  # predicted keypoints in red

                img2 = cv2.resize(img, (size * 2, size * 2), interpolation=cv2.INTER_LINEAR)

                if self.cfg['dataset'] != 'DHP19':
                    str = "acc: %.2f" % (pck_acc["total_correct"] / pck_acc["total_keypoints"])
                    # str = "acc: %.2f, th: %.2f " % (pck_acc["total_correct"] / pck_acc["total_keypoints"], th_val)
                    cv2.putText(img2, str,
                                text_location,
                                font,
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)
                cv2.imwrite(save_name, img2)


    def evaluate(self, data_loader,fastmode=False):
        self.model.eval()

        correct_kps = 0.0
        total_kps = 0.0
        joint_correct = np.zeros([self.cfg["num_classes"]])
        joint_total = np.zeros([self.cfg["num_classes"]])
        with torch.no_grad():
            start = time.time()
            for batch_idx, (imgs, labels, kps_mask, img_names, torso_diameter, head_size_norm, img_size_original, ts) in enumerate(data_loader):
                if img_size_original == 0:
                    continue

                start_sample = time.time()
                if batch_idx % 100 == 0 and batch_idx > 10:
                    print('Finished samples: ', batch_idx)
                    if not fastmode:
                        acc_intermediate = correct_kps / total_kps
                        acc_joint_mean_intermediate = np.mean(joint_correct / joint_total)
                        print('[Info] Mean Keypoint Acc: {:.3f}%'.format(100. * acc_intermediate))
                        print('[Info] Mean Joint Acc: {:.3f}%'.format(100. * acc_joint_mean_intermediate))
                        # print('Time since beginning:', time.time()-start)
                        print('[Info] Average Freq:', (batch_idx / (time.time() - start)), '\n')

                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs)

                pre = movenetDecode(output, kps_mask, mode='output', num_joints=self.cfg["num_classes"])
                gt = movenetDecode(labels, kps_mask, mode='label', num_joints=self.cfg["num_classes"])

                if not fastmode:
                    if torso_diameter is None:
                        pck_acc = pck(pre, gt, head_size_norm, num_classes=self.cfg["num_classes"], mode='head')
                    else:
                        pck_acc = pck(pre, gt, torso_diameter, threshold=0.5, num_classes=self.cfg["num_classes"],
                                      mode='torso')
                # print(pre,gt)
                # print(correct,total)

                if not fastmode:
                    correct_kps += pck_acc["total_correct"]
                    total_kps += pck_acc["total_keypoints"]
                    joint_correct += pck_acc["correct_per_joint"]
                    joint_total += pck_acc["anno_keypoints_per_joint"]

                    img_out, pose_gt = restore_sizes(imgs[0], gt, (int(img_size_original[0]), int(img_size_original[1])))
                    # print('gt after restore function', pose_gt)

                _, pose_pre = restore_sizes(imgs[0], pre, (int(img_size_original[0]), int(img_size_original[1])))
                # print('pre after restore function',pose_pre)

                kps_2d = np.reshape(pose_pre, [-1, 2])
                kps_hpecore = movenet_to_hpecore(kps_2d)
                kps_pre_hpecore = np.reshape(kps_hpecore, [-1])
                row = self.create_row(ts,kps_pre_hpecore, delay=time.time()-start_sample)
                sample = '_'.join(os.path.basename(img_names[0]).split('_')[:-1])
                write_path = os.path.join(self.cfg['results_path'],self.cfg['dataset'],sample,'movenet_cam2.csv')
                ensure_loc(os.path.dirname(write_path))
                self.write_results(write_path, row)
                # superimpose_pose(img_out, pose_gt, tensors=False, filename='/home/ggoyal/data/h36m/tests/%s_gt.png' % img_names[0].split('/')[-1].split('.')[0])
                # superimpose_pose(img_out, pose_pre, tensors=False,
                #                  filename=('/media/Data/data/h36m/tests/%s_pre.png' % img_names[0].split('/')[-1].split('.')[0]))

        if not fastmode:
            acc = correct_kps / total_kps
            acc_joint_mean = np.mean(joint_correct / joint_total)
            print('[Info] Mean Keypoint Acc: {:.3f}%'.format(100. * acc))
            print('[Info] Mean Joint Acc: {:.3f}% \n'.format(100. * acc_joint_mean))

    def evaluateTest(self, data_loader):
        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            start = time.time()
            for batch_idx, (imgs, labels, kps_mask, img_names, _, _, _) in enumerate(data_loader):

                if batch_idx % 100 == 0:
                    print('Finish ', batch_idx)
                    # if 'mypc'  not in img_names[0]:
                    #     continue
                    print('[Info] Average Freq:', (batch_idx / (time.time() - start)), '\n')
                # print('-----------------')

                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = (self.model(imgs))
                # print(output)
                # b

                # pre = []
                # for i in range(7):
                #     if output[i * 3 + 2] > 0.1:
                #         pre.extend([output[i * 3], output[i * 3 + 1]])
                #     else:
                #         pre.extend([-1, -1])
                # pre = np.array([pre])

                pre = movenetDecode(output, kps_mask, mode='output', num_joints=self.cfg["num_classes"])
                # gt = movenetDecode(labels, kps_mask, mode='label', num_joints=self.cfg["num_classes"])
                # print(pre, gt)
                # b
                # n
                # acc = myAcc(pre, gt)
                #
                # correct += sum(acc)
                # total += len(acc)

        # acc = correct / total
        # print('[Info] acc: {:.3f}% \n'.format(100. * acc))

    def infer_video(self, data_loader, video_path):
        self.model.eval()

        correct_kps = 0.0
        total_kps = 0.0
        joint_correct = np.zeros([self.cfg["num_classes"]])
        joint_total = np.zeros([self.cfg["num_classes"]])
        size = self.cfg["img_size"]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 50, (size * 2, size * 2))

        text_location = (10, size * 2 - 10)  # bottom left corner of the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        fontColor = (0, 0, 255)
        thickness = 1
        lineType = 2

        with torch.no_grad():
            start = time.time()
            for batch_idx, (imgs, labels, kps_mask, img_names, torso_diameter, head_size_norm, img_size_original,ts) in enumerate(
                    data_loader):
                #### For a single sample inference.
                # sample_name = img_names[0].split('_')[:-1]
                # if batch_idx == 0:
                #     primary_sample = sample_name
                # elif sample_name != primary_sample:
                #     break

                if batch_idx % 100 == 0 and batch_idx > 10:
                    print('Finished samples: ', batch_idx)
                    acc_intermediate = correct_kps / total_kps
                    acc_joint_mean_intermediate = np.mean(joint_correct / joint_total)
                    # print('[Info] Mean Keypoint Acc: {:.3f}%'.format(100. * acc_intermediate))
                    print('[Info] Mean Joint Acc: {:.3f}%'.format(100. * acc_joint_mean_intermediate))
                    print('[Info] Average Freq:', (batch_idx / (time.time() - start)), '\n')


                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs)

                pre = movenetDecode(output, kps_mask, mode='output', num_joints=self.cfg["num_classes"])
                gt = movenetDecode(labels, kps_mask, mode='label', num_joints=self.cfg["num_classes"])

                if self.cfg['dataset'] in ['coco', 'mpii']:
                    pck_acc = pck(pre, gt, head_size_norm, num_classes=self.cfg["num_classes"], mode='head')
                    th_val = head_size_norm
                else:
                    pck_acc = pck(pre, gt, torso_diameter, num_classes=self.cfg["num_classes"], mode='torso')
                    th_val = torso_diameter

                correct_kps += pck_acc["total_correct"]
                total_kps += pck_acc["total_keypoints"]
                joint_correct += pck_acc["correct_per_joint"]
                joint_total += pck_acc["anno_keypoints_per_joint"]

                img = np.transpose(imgs[0].cpu().numpy(), axes=[1, 2, 0])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                h, w = img.shape[:2]
                img = img*3
                img[img>255] = 255
                for i in range(len(gt[0]) // 2):
                    # img = add_skeleton(img, gt, (0, 255, 0),lines=1)
                    img = add_skeleton(img, pre, (0, 0, 255),lines=1)
                    # x = int(gt[0][i * 2] * w)
                    # y = int(gt[0][i * 2 + 1] * h)
                    # cv2.circle(img, (x, y), 2, (0, 255, 0), 1)  # gt keypoints in green

                    # x = int(pre[0][i * 2] * w)
                    # y = int(pre[0][i * 2 + 1] * h)
                    # cv2.circle(img, (x, y), 2, (0, 0, 255), 1)  # predicted keypoints in red
                # # Show center heatmaps
                # centers = output[1].cpu().numpy()[0]
                # from lib.utils.utils import maxPoint
                # cx, cy = maxPoint(centers)
                # # instant['center'] = np.array([cx[0][0], cy[0][0]]) / centers.shape[1]
                # img = superimpose(img, centers[0])

                img2 = cv2.resize(img, (size * 2, size * 2), interpolation=cv2.INTER_LINEAR)
                # str = "acc: %.2f, th: %.2f " % (pck_acc["total_correct"] / pck_acc["total_keypoints"], th_val)
                # cv2.putText(img2, str,
                #             text_location,
                #             font,
                #             fontScale,
                #             fontColor,
                #             thickness,
                #             lineType)
                # cv2.line(img2, [10, 10], [10 + int(head_size_norm * 2), 10], [0, 0, 255], 3)

                # basename = os.path.basename(img_names[0])
                # ensure_loc('eval_result')
                # cv2.imwrite(os.path.join('eval_result', basename), img)
                img2 = np.uint8(img2)
                # cv2.imshow("prediction", img2)
                # cv2.waitKey(1)
                out.write(img2)

        acc = correct_kps / total_kps
        acc_joint_mean = np.mean(joint_correct / joint_total)
        print('[Info] Mean Keypoint Acc: {:.3f}%'.format(100. * acc))
        print('[Info] Mean Joint Acc: {:.3f}% \n'.format(100. * acc_joint_mean))
        out.release()


    def save_video(self, data_loader, video_path):
        self.model.eval()

        size = self.cfg["img_size"]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 50, (size * 2, size * 2))
        with torch.no_grad():
            start = time.time()
            for batch_idx, (imgs, labels, kps_mask, img_names, torso_diameter, head_size_norm, img_size_original, ts) in enumerate(
                    data_loader):
                sample_name = img_names[0].split('_')[:-1]
                if batch_idx == 0:
                    primary_sample = sample_name
                else:
                    if sample_name != primary_sample:
                        break

                if batch_idx % 100 == 0 and batch_idx > 10:
                    print('Finished samples: ', batch_idx)

                img = np.transpose(imgs[0].cpu().numpy(), axes=[1, 2, 0])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img2 = cv2.resize(img, (size * 2, size * 2), interpolation=cv2.INTER_LINEAR)
                img2 = np.uint8(img2)
                out.write(img2)
        out.release()
    ################
    def onTrainStep(self, train_loader, epoch):

        self.model.train()
        correct = 0
        count = 0
        correct_kps = 0.0
        total_kps = 0.0
        joint_correct = np.zeros([self.cfg["num_classes"]])
        joint_total = np.zeros([self.cfg["num_classes"]])
        w_heatmap, w_bone, w_center, w_reg, w_offset = self.cfg['w_heatmap'], self.cfg['w_bone'], \
                                                       self.cfg['w_center'], self.cfg['w_reg'], self.cfg['w_offset']

        heatmap_loss_sum, bone_loss_sum, center_loss_sum, regs_loss_sum, offset_loss_sum = 0, 0, 0, 0, 0

        right_count = np.array([0] * self.cfg['batch_size'], dtype=np.float64)
        total_count = 0

        for batch_idx, (imgs, labels, kps_mask, img_names, torso_diameter, head_size_norm, _, _) in enumerate(
                train_loader):

            # if '000000242610_0' not in img_names[0]:
            #     continue

            labels = labels.to(self.device)
            imgs = imgs.to(self.device)
            kps_mask = kps_mask.to(self.device)

            output = self.model(imgs)

            heatmap_loss, bone_loss, center_loss, regs_loss, offset_loss = self.loss_func(output, labels, kps_mask,
                                                                                          self.cfg['num_classes'])

            total_loss = (w_heatmap * heatmap_loss) + (w_bone * bone_loss) + (w_center * center_loss) \
                         + (w_reg * regs_loss) + (w_offset * offset_loss)

            heatmap_loss_sum += heatmap_loss
            bone_loss_sum += bone_loss
            center_loss_sum += center_loss
            regs_loss_sum += regs_loss
            offset_loss_sum += offset_loss

            if self.cfg['clip_gradient']:
                clipGradient(self.optimizer, self.cfg['clip_gradient'])

            self.optimizer.zero_grad()  # 把梯度置零
            total_loss.backward()  # 计算梯度
            self.optimizer.step()  # 更新参数

            ### evaluate

            pre = movenetDecode(output, kps_mask, mode='output', num_joints=self.cfg["num_classes"])

            gt = movenetDecode(labels, kps_mask, mode='label', num_joints=self.cfg["num_classes"])

            # hm = cv2.resize(np.sum(labels[0,:7,:,:].cpu().detach().numpy(),axis=0),(192,192))*255
            # cv2.imwrite(os.path.join("output/show_img",os.path.basename(img_names[0])[:-4]+"_gt.jpg"),hm)
            # bb
            # print(pre.shape, gt.shape)
            # b
            # acc = myAcc(pre, gt)

            if self.cfg['dataset'] in ['coco', 'mpii']:
                pck_acc = pck(pre, gt, head_size_norm, num_classes=self.cfg["num_classes"], mode='head')
            else:
                pck_acc = pck(pre, gt, torso_diameter, num_classes=self.cfg["num_classes"], mode='torso')

            # right_count += pck
            # total_count += labels.shape[0]
            correct_kps += pck_acc["total_correct"]
            total_kps += pck_acc["total_keypoints"]
            joint_correct += pck_acc["correct_per_joint"]
            joint_total += pck_acc["anno_keypoints_per_joint"]

            if batch_idx % self.cfg['log_interval'] == 0:
                acc_joint_mean_intermediate = np.mean(joint_correct / joint_total)
                print('\r',
                      '%d/%d '
                      '[%d/%d] '
                      'loss: %.4f '
                      '(hm_loss: %.3f '
                      'b_loss: %.3f '
                      'c_loss: %.3f '
                      'r_loss: %.3f '
                      'o_loss: %.3f) - '
                      'acc: %.4f         ' % (epoch + 1, self.cfg['epochs'],
                                              batch_idx, len(train_loader.dataset) / self.cfg['batch_size'],
                                              total_loss.item(),
                                              heatmap_loss.item(),
                                              bone_loss.item(),
                                              center_loss.item(),
                                              regs_loss.item(),
                                              offset_loss.item(),
                                              acc_joint_mean_intermediate),
                      end='', flush=True)
            # break
        total_loss_sum = heatmap_loss_sum + center_loss_sum + regs_loss_sum + offset_loss_sum + bone_loss_sum

        # Tensorboard additions
        self.add_to_tb(heatmap_loss_sum, bone_loss_sum, center_loss_sum, regs_loss_sum, offset_loss_sum,
                       total_loss_sum, acc_joint_mean_intermediate, epoch + 1, label="Train")
        if acc_joint_mean_intermediate > self.best_train_accuracy:
            self.best_train_accuracy = acc_joint_mean_intermediate

    def onTrainEnd(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.tb.flush()
        self.tb.close()

        if self.cfg["cfg_verbose"]:
            printDash()
            print(self.cfg)
            printDash()

    def onValidation(self, val_loader, epoch):

        num_test_batches = 0.0
        self.model.eval()

        correct_kps = 0.0
        total_kps = 0.0
        joint_correct = np.zeros([self.cfg["num_classes"]])
        joint_total = np.zeros([self.cfg["num_classes"]])

        heatmap_loss_sum = bone_loss_sum = center_loss_sum = regs_loss_sum = offset_loss_sum = 0

        right_count = np.array([0] * self.cfg['num_classes'], dtype=np.int64)
        total_count = 0
        with torch.no_grad():
            for batch_idx, (imgs, labels, kps_mask, img_names, torso_diameter, head_size_norm, _) in enumerate(
                    val_loader):
                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs)

                heatmap_loss, bone_loss, center_loss, regs_loss, offset_loss = self.loss_func(output, labels, kps_mask,
                                                                                              self.cfg['num_classes'])
                total_loss = heatmap_loss + center_loss + regs_loss + offset_loss + bone_loss

                heatmap_loss_sum += heatmap_loss
                bone_loss_sum += bone_loss
                center_loss_sum += center_loss
                regs_loss_sum += regs_loss
                offset_loss_sum += offset_loss

                ### evaluate
                # acc1 = myAcc(heatmap2locate(output[0].detach().cpu().numpy()), 
                #                 heatmap2locate(labels[:,:7,:,:].detach().cpu().numpy()))
                pre = movenetDecode(output, kps_mask, mode='output', num_joints=self.cfg["num_classes"])

                gt = movenetDecode(labels, kps_mask, mode='label', num_joints=self.cfg["num_classes"])

                # acc = pckh(pre, gt)
                if self.cfg['dataset'] in ['coco', 'mpii']:
                    pck_acc = pck(pre, gt, head_size_norm, num_classes=self.cfg["num_classes"], mode='head')
                else:
                    pck_acc = pck(pre, gt, torso_diameter, num_classes=self.cfg["num_classes"], mode='torso')
                # right_count += pck
                # total_count += labels.shape[0]
                correct_kps += pck_acc["total_correct"]
                total_kps += pck_acc["total_keypoints"]
                joint_correct += pck_acc["correct_per_joint"]
                joint_total += pck_acc["anno_keypoints_per_joint"]
                acc_joint_mean_intermediate = np.mean(joint_correct / joint_total)

                # right_count += sum(acc)
                # total_count += labels.shape[0]
                # break

            print('LR: %f - '
                  ' [Val] loss: %.5f '
                  '[hm_loss: %.4f '
                  'b_loss: %.4f '
                  'c_loss: %.4f '
                  'r_loss: %.4f '
                  'o_loss: %.4f] - '
                  'acc: %.4f          ' % (
                      self.optimizer.param_groups[0]["lr"],
                      total_loss.item(),
                      heatmap_loss.item(),
                      bone_loss.item(),
                      center_loss.item(),
                      regs_loss.item(),
                      offset_loss.item(),
                      acc_joint_mean_intermediate),
                  )
            print()

        total_loss_sum = heatmap_loss_sum + center_loss_sum + regs_loss_sum + offset_loss_sum + bone_loss_sum

        self.add_to_tb(heatmap_loss_sum, bone_loss_sum, center_loss_sum, regs_loss_sum, offset_loss_sum,
                       total_loss_sum, acc_joint_mean_intermediate, epoch + 1, label="Val")
        if acc_joint_mean_intermediate > self.best_val_accuracy:
            self.best_val_accuracy = acc_joint_mean_intermediate

        if 'default' in self.cfg['scheduler']:
            self.scheduler.step(np.mean(right_count / total_count))
        else:
            self.scheduler.step()

        # save the lastest loass in loss vector
        self.val_losses[:-1] = self.val_losses[1:]
        self.val_losses[-1] = total_loss_sum

        if epoch > 20:
            self.check_early_stop()

        save_name = 'e%d_valacc%.5f.pth' % (epoch + 1, (acc_joint_mean_intermediate))
        self.modelSave(save_name, total_loss_sum < self.val_loss_best)
        self.val_loss_best = min(self.val_loss_best, total_loss_sum)

    def onTest(self, data_loader):
        self.model.eval()

        # predict
        res_list = []
        with torch.no_grad():
            # end = time.time()
            for i, (inputs, target, img_names) in enumerate(data_loader):
                print("\r", str(i) + "/" + str(data_loader.__len__()), end="", flush=True)

                inputs = inputs.cuda()

                output = self.model(inputs)
                output = output.data.cpu().numpy()

                for i in range(output.shape[0]):
                    output_one = output[i][np.newaxis, :]
                    output_one = np.argmax(output_one)

                    res_list.append(output_one)
        return res_list

    def modelLoad(self, model_path, data_parallel=False):

        if os.path.splitext(model_path)[-1] == '.json':
            with open(model_path, 'r') as f:
                model_path = json.loads(f.readlines()[0])
                str1 = ''
            init_epoch = int(str1.join(os.path.basename(model_path).split('_')[0][1:]))
            self.init_epoch = init_epoch
        print(model_path)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        if data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def modelSave(self, save_name, is_best=False):
        if self.cfg['save_best_only']:
            if is_best:
                fullname_best = os.path.join(self.cfg['save_dir'], self.cfg['label'], "best.pth")
                torch.save(self.model.state_dict(), fullname_best)

                fullname = os.path.join(self.cfg['save_dir'], self.cfg['label'], save_name)
                torch.save(self.model.state_dict(), fullname)
                with open(Path(self.cfg['newest_ckpt']).resolve(), 'w') as f:
                    json.dump(fullname, f, ensure_ascii=False)

        else:
            fullname = os.path.join(self.cfg['save_dir'], self.cfg['label'], save_name)
            torch.save(self.model.state_dict(), fullname)

            with open(Path(self.cfg['newest_ckpt']).resolve(), 'w') as f:
                json.dump(fullname, f, ensure_ascii=False)
        # print("Save model to: ",save_name)

    def add_to_tb(self, heatmap_loss, bone_loss, center_loss, regs_loss, offset_loss, total_loss, acc, epoch,
                  label=None):

        if label is not None and label[-1] != " ":
            label = label + " "

        self.tb.add_scalar(label + "Total Loss", total_loss, epoch)
        self.tb.add_scalar(label + "Heatmap Loss", heatmap_loss, epoch)
        self.tb.add_scalar(label + "Bone Loss", bone_loss, epoch)
        self.tb.add_scalar(label + "Center Loss", center_loss, epoch)
        self.tb.add_scalar(label + "Regression Loss", regs_loss, epoch)
        self.tb.add_scalar(label + "Offset Loss", offset_loss, epoch)
        self.tb.add_scalar(label + "Accuracy", acc, epoch)

    def check_early_stop(self):
        losses = self.val_losses
        l = int(len(losses)) // 2
        strip_old = losses[:l]
        strip_new = losses[l:]
        if np.mean(strip_old) <= np.mean(strip_new):
            self.early_stop_counter += 1
        else:
            self.early_stop_counter += 0

    def save_results(self):
        ensure_loc(self.cfg['results_path'])
        with open(os.path.join(self.cfg['results_path'], (self.cfg['label'] + '_log.txt')), 'a') as f:
            f.write(str(self.cfg))
            f.write('Best training accuracy:' + str(self.best_train_accuracy))
            f.write('Best validation accuracy:' + str(self.best_val_accuracy))

    def write_results(self, path, row):
        # Write a data point into a csvfile
        with open(path, 'a') as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerow(row)

    def create_row(self, ts, skt, delay = 0.0):
        # Function to create a row to be written into a csv file.
        row = []
        ts = float(ts)
        row.extend([ts, delay])
        row.extend(skt)
        return row