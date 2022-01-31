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

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from lib.task.task_tools import getSchedu, getOptimizer, movenetDecode, clipGradient
from lib.loss.movenet_loss import MovenetLoss
from lib.utils.utils import printDash
from lib.visualization.visualization import superimpose_pose
from lib.utils.metrics import myAcc, pckh

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

        self.val_losses = np.zeros([10])
        self.early_stop = False
        self.val_loss_best = np.Inf

        # scheduler
        self.scheduler = getSchedu(self.cfg['scheduler'], self.optimizer)

        # tensorboard
        self.tb = SummaryWriter(comment="__Dataset="+self.cfg['optimizer']+"_LR="+str(self.cfg['learning_rate'])+"_optimizer="+self.cfg['optimizer'])

    def train(self, train_loader, val_loader):

        if self.init_epoch == 0:
            dummy_input1 = torch.randn(1, 3, 192, 192).cuda()
            self.tb.add_graph(self.model, dummy_input1)
        print()

        for epoch in range(self.init_epoch,self.init_epoch+self.cfg['epochs']):
            self.onTrainStep(train_loader, epoch)
            self.onValidation(val_loader, epoch)
            if self.early_stop:
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

                pre = movenetDecode(output, None, mode='output',num_joints=self.cfg["num_classes"])
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
                cv2.imwrite(os.path.join(save_dir, basename[:-4] + "_regs0.jpg"), cv2.resize(regs[0] * 255, (size, size)))

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

        with torch.no_grad():
            for batch_idx, (imgs, labels, kps_mask, img_names) in enumerate(data_loader):

                if batch_idx % 5000 == 0:
                    print('Finish ', batch_idx)
                # if 'mypc'  not in img_names[0]:
                #     continue

                # print('-----------------')

                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs)

                pre = movenetDecode(output, kps_mask, mode='output',num_joints=self.cfg["num_classes"])
                gt = movenetDecode(labels, kps_mask, mode='label',num_joints=self.cfg["num_classes"])

                # n
                acc = myAcc(pre, gt)

                # if 'mypc1_full_1180' in img_names[0]:
                if 0 / 7 < sum(acc) / len(acc) <= 5 / 7:
                    # if sum(acc)/len(acc)==1:
                    # print(pre)
                    # print(gt)
                    # print(acc)
                    img_name = img_names[0]
                    # print(img_name)

                    basename = os.path.basename(img_name)
                    save_name = os.path.join(save_dir, basename)

                    # print(os.path.join(save_dir, basename[:-4] + "_hm_pre.jpg"))
                    hm = cv2.resize(np.sum(output[0][0].cpu().numpy(), axis=0), (192, 192)) * 255
                    cv2.imwrite(os.path.join(save_dir, basename[:-4] + "_hm_pre.jpg"), hm)
                    # cv2.imshow("prediction",hm)
                    # cv2.waitKey()

                    hm = cv2.resize(np.sum(labels[0, :7, :, :].cpu().numpy(), axis=0), (192, 192)) * 255
                    cv2.imwrite(os.path.join(save_dir, basename[:-4] + "_hm_gt.jpg"), hm)

                    img = np.transpose(imgs[0].cpu().numpy(), axes=[1, 2, 0])
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    h, w = img.shape[:2]

                    for i in range(len(gt[0]) // 2):
                        x = int(gt[0][i * 2] * w)
                        y = int(gt[0][i * 2 + 1] * h)
                        cv2.circle(img, (x, y), 5, (0, 255, 0), 3)

                        x = int(pre[0][i * 2] * w)
                        y = int(pre[0][i * 2 + 1] * h)
                        cv2.circle(img, (x, y), 3, (0, 0, 255), 2)

                    cv2.imwrite(save_name, img)

                    # bb

    def evaluate(self, data_loader):
        self.model.eval()

        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for batch_idx, (imgs, labels, kps_mask, img_names, head_size,_) in enumerate(data_loader):

                if batch_idx % 100 == 0:
                    print('Finished samples: ', batch_idx)
                    if batch_idx>10:
                        acc_intermediate = correct / total
                        print('[Info] acc: {:.3f}% \n'.format(100. * acc_intermediate))
                # if 'mypc'  not in img_names[0]:
                #     continue

                # print('-----------------')

                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs)

                pre = movenetDecode(output, kps_mask, mode='output',num_joints=self.cfg["num_classes"])
                gt = movenetDecode(labels, kps_mask, mode='label',num_joints=self.cfg["num_classes"])

                # n
                pck = pckh(pre,gt,head_size,self.cfg["th"]/100,self.cfg["num_classes"])
                # print(pck)
                # print(correct,total)
                # exit()
                correct += sum(pck)
                total += len(labels)

        acc = correct / total
        print('[Info] acc: {:.3f}% \n'.format(100. * acc))

    def evaluateTest(self, data_loader):
        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (imgs, labels, kps_mask, img_names) in enumerate(data_loader):

                if batch_idx % 100 == 0:
                    print('Finish ', batch_idx)
                # if 'mypc'  not in img_names[0]:
                #     continue

                # print('-----------------')

                labels = labels.to(self.device)
                imgs = imgs.to(self.device)
                kps_mask = kps_mask.to(self.device)

                output = self.model(imgs).cpu().numpy()
                # print(output)
                # b

                pre = []
                for i in range(7):
                    if output[i * 3 + 2] > 0.1:
                        pre.extend([output[i * 3], output[i * 3 + 1]])
                    else:
                        pre.extend([-1, -1])
                pre = np.array([pre])

                # pre = movenetDecode(output, kps_mask,mode='output',num_joints=self.cfg["num_classes"])
                gt = movenetDecode(labels, kps_mask, mode='label',num_joints=self.cfg["num_classes"])
                # print(pre, gt)
                # b
                # n
                acc = myAcc(pre, gt)

                correct += sum(acc)
                total += len(acc)

        acc = correct / total
        print('[Info] acc: {:.3f}% \n'.format(100. * acc))

    ################
    def onTrainStep(self, train_loader, epoch):

        self.model.train()
        correct = 0
        count = 0

        heatmap_loss_sum = bone_loss_sum = center_loss_sum = regs_loss_sum = offset_loss_sum = 0

        right_count = np.array([0] * self.cfg['batch_size'], dtype=np.float64)
        total_count = 0

        for batch_idx, (imgs, labels, kps_mask, img_names,head_size,_) in enumerate(train_loader):

            # if '000000242610_0' not in img_names[0]:
            #     continue

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
            # print(head_size.shape)
            pck = pckh(pre,gt,head_size,self.cfg["th"]/100,self.cfg["num_classes"])
            right_count += pck
            total_count += labels.shape[0]


            if batch_idx % self.cfg['log_interval'] == 0:
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
                                              sum(right_count) / total_count),
                      end='', flush=True)
            # break
        total_loss_sum = heatmap_loss_sum + center_loss_sum + regs_loss_sum + offset_loss_sum + bone_loss_sum

        # Tensorboard additions
        self.add_to_tb(heatmap_loss_sum, bone_loss_sum, center_loss_sum, regs_loss_sum, offset_loss_sum,
                       total_loss_sum, sum(right_count) / total_count, epoch + 1, label="Train")

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

        heatmap_loss_sum = bone_loss_sum = center_loss_sum = regs_loss_sum = offset_loss_sum = 0

        right_count = np.array([0] * self.cfg['num_classes'], dtype=np.int64)
        total_count = 0
        with torch.no_grad():
            for batch_idx, (imgs, labels, kps_mask, img_names,head_size,_) in enumerate(val_loader):
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
                acc = pckh(pre, gt, head_size, self.cfg["th"]/100)

                # right_count1 += acc1
                right_count += acc
                total_count += labels.shape[0]

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
                      np.mean(right_count / total_count)),
                  )
            print()

        total_loss_sum = heatmap_loss_sum + center_loss_sum + regs_loss_sum + offset_loss_sum + bone_loss_sum

        self.add_to_tb(heatmap_loss_sum, bone_loss_sum, center_loss_sum, regs_loss_sum, offset_loss_sum,
                       total_loss_sum, np.mean(right_count / total_count), epoch + 1, label="Val")

        if 'default' in self.cfg['scheduler']:
            self.scheduler.step(np.mean(right_count / total_count))
        else:
            self.scheduler.step()

        # save the lastest loass in loss vector
        self.val_losses[:-1] = self.val_losses[1:]
        self.val_losses[-1] = total_loss_sum

        self.check_early_stop()

        save_name = 'e%d_valacc%.5f.pth' % (epoch + 1, np.mean(right_count / total_count))
        self.modelSave(save_name,total_loss_sum<self.val_loss_best)
        self.val_loss_best = min(self.val_loss_best, total_loss_sum)

    def onTest(self):
        self.model.eval()

        # predict
        res_list = []
        with torch.no_grad():
            # end = time.time()
            for i, (inputs, target, img_names) in enumerate(data_loader):
                print("\r", str(i) + "/" + str(test_loader.__len__()), end="", flush=True)

                inputs = inputs.cuda()

                output = model(inputs)
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
        self.model.load_state_dict(torch.load(model_path,map_location=self.device))

        if data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def modelSave(self, save_name,is_best=False):
        if self.cfg['save_best_only']:
            if is_best:
                fullname_best = os.path.join(self.cfg['save_dir'], "best.pth")
                torch.save(self.model.state_dict(), fullname_best)
                with open(Path(self.cfg['newest_ckpt']).resolve(), 'w') as f:
                    json.dump(fullname_best, f, ensure_ascii=False)
        else:
            fullname = os.path.join(self.cfg['save_dir'], save_name)
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
        l = int(len(losses))/2
        stripold = losses[:l]
        stripnew = losses[l:]
        if np.mean(stripold) <= np.mean(stripnew) :
            self.early_stop = True