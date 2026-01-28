import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from data_factory.dataloader import get_loader_segment
from model.TFSAD import TFSAD

from metrics.metrics import *
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def forward(self, input, target):
        mse_loss = self.mse(input, target)
        pt = torch.exp(-mse_loss)
        focal_loss = ((1 - pt) ** self.gamma) * mse_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class Solver(object):
    DEFAULTS = {
    }

    def __init__(self, config):
        start_init = time.time()
        self.__dict__.update(Solver.DEFAULTS, **config)
        start_data = time.time()

        # ----------------------------

        # ----------------------------------
        self.train_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                               win_size=self.win_size, mode='train', dataset=self.dataset, )
        self.vali_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.min_size,
                                              win_size=self.win_size, mode='thre', dataset=self.dataset)

        self.sw_max_mean = self.sw_max_mean
        self.sw_loss = self.sw_loss
        self.num_epochs = self.num_epochs
        self.batch = self.batch_size
        self.ratio = self.ratio
        self.build_model()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.init_time = time.time() - start_init

        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
            self.criterion_keep = nn.MSELoss(reduction='none')

    def build_model(self):
        self.model = TFSAD(win_size=self.win_size, patch_size=self.patch_size,
                           batch_size=self.batch_size,
                           channel=self.input)

        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        print('Parameters:', sum(p.numel() for p in self.model.parameters() if p.requires_grad))


    def train(self):
        print("train_loader：", len(self.train_loader))
        print("Start training")
        for epoch in range(self.num_epochs):
            epoch_time = time.time()
            self.model.train()
            for batch_idx, (data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                input = data.float().to(self.device)
                local_point,global_point,local_re_neighbor,global_re_neighbor,local_neighbor = self.model(input)
                loss1 =  (
                     self.criterion(local_point, global_point)
                   + self.criterion(local_point, input)
                   + self.criterion(global_point, input)
                )
                loss2 = (
                        self.criterion(local_re_neighbor, global_re_neighbor)
                     + self.criterion(local_re_neighbor, local_neighbor)
                     + self.criterion(global_re_neighbor, local_neighbor)
                )
                loss = self.ratio*loss1 + (1-self.ratio)*loss2

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        attens_energy = []
        for batch_idx, (data, labels) in enumerate(self.train_loader):
            input = data.float().to(self.device)

            local_point, global_point, local_re_neighbor, global_re_neighbor,local_neighbor= self.model(input)
            loss1 = 0
            loss2 = 0
            loss1 += (
                   self.criterion_keep(local_point, global_point)
                 + self.criterion_keep(local_point, input)
                 + self.criterion_keep(global_point, input)
                        )
            loss2 += (self.criterion_keep(local_re_neighbor, global_re_neighbor)
                      + self.criterion_keep(local_re_neighbor, local_neighbor)
                      + self.criterion_keep(global_re_neighbor, local_neighbor)
                      )
            if (self.sw_max_mean == 0):
                loss1 = torch.mean(loss1, dim=-1)
                loss2 = torch.mean(loss2, dim=-1)
                loss = self.ratio * loss1 + (1-self.ratio) * loss2
            else:
                loss, _ = torch.max(loss, dim=-1)

            metric = torch.softmax(loss, dim=-1)

            cri = metric.detach().cpu().numpy() #
            attens_energy.append(cri)
        print("attens_energy_train:", len(attens_energy))
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        test_labels = []
        attens_energy = []

        for batch_idx, (data, labels) in enumerate(self.thre_loader):
            input = data.float().to(self.device)
            local_point, global_point, local_re_neighbor, global_re_neighbor,local_neighbor= self.model(input)
            test_labels.append(labels)

            loss1 = 0
            loss2 = 0
            loss1 += (
                    self.criterion_keep(local_point, global_point)
                    + self.criterion_keep(local_point, input)
                    + self.criterion_keep(global_point, input)
            )
            loss2 += (
                    self.criterion_keep(local_re_neighbor, global_re_neighbor)
                    + self.criterion_keep(local_re_neighbor, local_neighbor)
                    + self.criterion_keep(global_re_neighbor, local_neighbor)
            )
            if (self.sw_max_mean == 0):
                loss1 = torch.mean(loss1, dim=-1)
                loss2 = torch.mean(loss2, dim=-1)
                loss = self.ratio * loss1 + (1-self.ratio) * loss2
            else:
                loss, _ = torch.max(loss, dim=-1)
            metric = torch.softmax(loss, dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
        print("attens_energy_test: ",len(attens_energy))  # 11
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)

        print("dataset:",self.data_path)
        print("anormly_ratio:", self.anormly_ratio)
        print("Threshold:", thresh)

        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        matrix = [self.index]

        scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        for key, value in scores_simple.items():
            matrix.append(value)
            print('{0:21} : {1:0.4f}'.format(key, value))
        # 后处理优化
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = test_labels.astype(int)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')

        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision,
                                                                                                   recall, f_score))
        results_df = pd.DataFrame({
            'Timestamp': np.arange(len(gt)),  # 如果有实际的时间戳数据，替换为实际的时间戳
            'Actual_Label': gt,
            'Predicted_Label': pred,
            'Energy_Score': test_energy
        })