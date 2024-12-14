# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 03:21:30 2024

@author: Owner
"""

import os
import numpy as np
import pandas as pd
import csv
import torch
from torch.utils.data import Dataset
import random
from sklearn.model_selection import train_test_split
from augmented_pool import *


Train_Data= np.load('D:\\无监督元学习_数据裁剪_域外数据\\HIT航空发动机\\HIT_hangkong_train.npy')
Test_Data = np.load('D:\\无监督元学习_数据裁剪_域外数据\\HIT航空发动机\\HIT_hangkong_test.npy')
combined_array = np.concatenate((Train_Data, Test_Data), axis=0)
#combined_array = np.concatenate((Train_Data, Test_Data), axis=0)

data_array=combined_array.reshape(combined_array.shape[0]*combined_array.shape[1],combined_array.shape[3])

num_samples = data_array.shape[0]
num_classes = 109
samples_per_class = combined_array.shape[1]

# 计算总标签数量
total_labels = num_classes * samples_per_class

# 生成标签数组
labels_array = np.arange(num_classes)  # 生成 [0, 1, 2, ..., num_classes-1]
labels_array = np.repeat(labels_array, samples_per_class)  # 每个标签重复 samples_per_class 次
# 确保标签数量与 num_samples 匹配
labels_array = np.tile(labels_array, num_samples // total_labels) 


class HITDataset(Dataset):
    def __init__(self, data_array, labels_array, mode='train', cls_num=109, test_size=0.5, simclr=False, imb_type='exp', imb_factor=0.01, rand_number=0, num_expert=3, transform=None, target_transform=None):

        self.simclr = simclr
        self.mode = mode
        self.cls_num = cls_num
        self.data = data_array
        self.targets = labels_array.squeeze().astype(np.int64)  # 将标签数组扁平化为一维
        self.new_labels = []

        # 划分训练集和测试集
        indices = np.arange(len(self.data))
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=rand_number, stratify=self.targets)
        
        if mode == 'train':
            self.data = self.data[train_indices]
            self.targets = self.targets[train_indices]
        else:
            self.data = self.data[test_indices]
            self.targets = self.targets[test_indices]

        # 数据不平衡处理
        if mode == 'train':
            np.random.seed(rand_number)
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list, num_expert)

        self.transform = transform
        self.target_transform = target_transform

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.targets) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(int(cls_num - cls_num // 2)):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls, num_expert):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.extend(self.data[selec_idx])
            new_targets.extend([the_class] * the_img_num)
        self.data = np.array(new_data)
        self.targets = np.array(new_targets)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def add_laplace_noise(self, x, u=0, b=0.2):
        laplace_noise = np.random.laplace(u, b, x.shape).reshape(x.shape)
        return laplace_noise + x

    def add_wgn(self, x, snr=15):
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(x ** 2) / len(x)
        npower = xpower / snr
        return x + (np.random.randn(*x.shape) * np.sqrt(npower))

    def Amplitude_scale(self, x, snr=0.05):
        return x * (1 - snr)

    def Translation(self, x, p=0.5):
        a = len(x)
        return np.concatenate((x[int(a*p):], x[:int(a*p)]), axis=0)
    def mask_noise(self,x,probability=0.2):
        x=torch.from_numpy(x)
        masked_data = x.clone()
        
       
        random_numbers = torch.rand_like(masked_data)

        threshold = probability
    
        masked_data[random_numbers < threshold] = 0.5
    
        return masked_data

    def __getitem__(self, index):

        if self.mode == 'train':
            label_ = self.targets[index]
            pic = self.data[index]
            if self.new_labels !=[]:
                new_labels = self.new_labels[index]
            else: 
                new_labels =-1


            if self.simclr:
                ccc = ['self.mask_noise(pic)','self.Amplitude_scale(pic)', 'self.Translation(pic)', 'self.add_wgn(pic)', 'self.add_laplace_noise(pic)']
                n1 = np.random.choice(ccc, 3, replace=False)
                aa = pic.T
                bb = eval(n1[1]).T
                cc=eval(n1[2]).T
                aa2 = np.concatenate((aa, bb, cc), axis=-1)
                aa2=aa2[np.newaxis, :] 
    
                return torch.tensor(aa2, dtype=torch.float).detach(), label_, index,new_labels
        else:
            label_ = self.targets[index]
            pic = self.data[index]
            if self.new_labels !=[]:
                new_labels = self.new_labels[index]
            else: 
                new_labels =-1
            ccc = ['self.Amplitude_scale(pic)', 'self.Translation(pic)', 'self.add_wgn(pic)', 'self.add_laplace_noise(pic)']
            n1 = np.random.choice(ccc, 3, replace=False)
            aa = pic.T
            bb = eval(n1[1]).T
            cc=eval(n1[2]).T
            aa2 = np.concatenate((aa, bb, cc), axis=-1)
            aa2=aa2[np.newaxis, :] 
            
            #pic3 = torch.tensor(pic[np.newaxis, :] , dtype=torch.float)
            return torch.tensor(aa2, dtype=torch.float).detach(), label_, index,new_labels
    def __len__(self):
        return len(self.targets)
    
        
ood_train = Random_HITDataset(data_array, labels_array, rand_number=0, ood=False)   #INDEX

ood_train_tran = Random_HITDataset(data_array, labels_array, rand_number=0, ood=True)  #INDEX=-1

ood_loader = torch.utils.data.DataLoader(
                ood_train, 
                num_workers=0,
                batch_size=64,
                shuffle=False,
                pin_memory=True)
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



from torch.utils.data import TensorDataset, DataLoader

from DRSN_CW import BasicBlock,RSNet

from tqdm import tqdm

from suploss import LogitAdjust, BalSCL,SupConLoss_ccl,BalSCL2,SupConLoss_ccl
import time


dataset_train = HITDataset(data_array, labels_array, mode='train', cls_num=num_classes , 
                          test_size=0.3, simclr=True, imb_type='exp', imb_factor=0.05, 
                          rand_number=0, num_expert=3, transform=None, target_transform=None)   

dataset_test = HITDataset(data_array, labels_array, mode='test', cls_num=num_classes , 
                           test_size=0.3, simclr=True, imb_type='exp', imb_factor=0.05, 
                           rand_number=0, num_expert=3, transform=None, target_transform=None)   
  

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def compute_accuracy(logits, targets):
    """Compute the accuracy"""
    with torch.no_grad():
        _, predictions = torch.max(logits, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()






from balanced_cluster import balanced_kmean
from kmeans_gpu import kmeans




def cluster (train_loader_cluster,model,cluster_number,cls_num_list,cluster_method):
    model.eval()
    features_sum = []
    for i, (inputs, target,_,cluster_target) in enumerate(train_loader_cluster):
        inputs1 = inputs[:, :, :1024]
        inputs1 = inputs1.to(device)
        target =target.to(device)
        with torch.no_grad():
            features,_,_  = model(inputs1)
            features = features.detach()
            features_sum.append(features)
    features= torch.cat(features_sum,dim=0)
    features = torch.split(features, cls_num_list, dim=0)
    target = [[] for i in range(len(cluster_number))]
    for i in range(len(cluster_number)):  
        if cluster_number[i] >1:
          if cluster_method == 'balanced_kmean':
            cluster_ids_x, _ = balanced_kmean(X=features[i], n_clusters=cluster_number[i], distance='cosine', init='k-means++',iol=50,tol=1e-3,device=device)
          else:
            cluster_ids_x, _ = kmeans(X=features[i], num_clusters=cluster_number[i], distance='cosine', tol=1e-3, iter_limit=35, device=device)
            #run faster for cluster
          target[i]=cluster_ids_x
        else:
            target[i] = torch.zeros(1,features[i].size()[0], dtype=torch.int).squeeze(0)
    cluster_number_sum=[sum(cluster_number[:i]) for i in range(len(cluster_number))]
    for i ,k in enumerate(cluster_number_sum):
         target[i] =  torch.add(target[i], k)
    targets=torch.cat(target,dim=0)
    targets = targets.numpy().tolist()

    return targets
    
    


from collections import Counter

# 示例列表

from sklearn.metrics import confusion_matrix
def validate(val_loader, model, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    #losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to evaluate mode
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target,_,_) in enumerate(val_loader):
            INPUT1=input[:, :, :1024]
            input = INPUT1.to(device)
        
            target = target.to(device)
            # compute output
            _,output,_ = model(input)
            #loss = criterion(output,target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            #losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #print(output)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(flag=flag, top1=top1, top5=top5))
        out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        print(output)
        print(out_cls_acc)
       
    return top1.avg





import torch.nn.functional as F
from sklearn.cluster import KMeans as Kmeans_sklearn
###构建数据池
@torch.no_grad()
def sample_batch(train_loader_test_trans, model, sample_loader_test_trans, momentum_weight=None, args=None):
    id_features = []  # 用于存储训练集样本的特征
    id_idxs = []  # 用于存储训练集样本的索引
    k_means_clusters = 20# K-means 聚类的簇数
    batch_size = 64  # 每批次的样本数
    cluster_temperature = 0.02  # 聚类权重的温度参数
    budget = 4000  # 总预算，即最终选择的样本数量

    model.eval()  # 设置模型为评估模式
    for i, (inputs,_, idx,_) in enumerate(train_loader_test_trans):
        inputs1 = inputs[:, :, :1024]  # 提取输入数据的前 1024 个特征
        inputs1 = inputs1.to(device)  # 将数据移动到设备（如 GPU）
        idx = idx.view(-1)  # 将索引展平为一维
        _,features,_ = model.eval()(inputs1)  # 使用模型提取特征

        id_features.append(features.detach())  # 存储特征，detach() 用于断开计算图
        id_idxs.append(idx.detach())  # 存储索引

    id_features = torch.cat(id_features)  # 合并所有特征
    id_features = F.normalize(id_features, dim=-1)  # 对特征进行归一化
    id_idxs = torch.cat(id_idxs)  # 合并所有索引
    id_features = id_features[id_idxs.sort()[1]]  # 按索引排序特征
    momentum_weight = momentum_weight[:id_features.shape[0]]  # 根据样本数量调整 momentum_weight

    ood_features = []  # 用于存储测试集样本的特征
    ood_idxs = []  # 用于存储测试集样本的索引
    for i, (inputs, _, idx,_) in enumerate(sample_loader_test_trans):
        inputs2 = inputs[:, :, :1024]  # 提取输入数据的前 1024 个特征
        inputs2 = inputs2.to(device)  # 将数据移动到设备（如 GPU）
        idx = idx.view(-1).to(device)  # 将索引展平并移动到设备
        _,features,_ = model.eval()(inputs2)  # 使用模型提取特征

        ood_features.append(features.detach())  # 存储特征
        ood_idxs.append(idx.detach())  # 存储索引

    ood_features = torch.cat(ood_features)  # 合并所有特征
    ood_features = F.normalize(ood_features, dim=-1)  # 对特征进行归一化
    ood_idxs = torch.cat(ood_idxs)  # 合并所有索引
    ood_features = ood_features[ood_idxs.sort()[1]]  # 按索引排序特征

    kmeans_features = id_features.detach().cpu().numpy()  # 将特征转换为 NumPy 数组
    kmeans_ins = Kmeans_sklearn(n_clusters=k_means_clusters, random_state=0).fit(kmeans_features)  # 使用 K-means 聚类算法
    cluster_labels = torch.from_numpy(kmeans_ins.labels_).to(device)  # 获取每个样本的簇标签，并转换为 PyTorch 张量
    cluster_weight = torch.zeros(k_means_clusters).to(device)  # 初始化每个簇的权重

    for i in range(k_means_clusters):
        cluster_weight[i] = momentum_weight[cluster_labels == i].mean()  # 计算每个簇的权重

    cluster_weight = -((cluster_weight - torch.mean(cluster_weight)) / torch.std(cluster_weight))  # 标准化权重
    cluster_weight = F.softmax(cluster_weight / cluster_temperature, dim=0)  # 应用 softmax 函数得到最终的簇权重
    cluster_budget = cluster_weight * budget  # 根据权重计算每个簇的预算
    cluster_budget = torch.tensor(cluster_budget, dtype=torch.int64)  # 转换为整型张量
    cluster_budget[-1] += int(budget - cluster_budget.sum())  # 调整最后一个簇的预算以确保总预算为指定值
    cluster_centers = torch.from_numpy(kmeans_ins.cluster_centers_)  # 获取聚类中心
    centers = F.normalize(cluster_centers.to(device), dim=-1)  # 对聚类中心进行归一化
    distance_matrix, index_matrix = (1 - torch.mm(ood_features, centers.t())).t().sort(dim=1)  # 计算距离矩阵和索引矩阵

    sample_idx = []  # 用于存储采样的索引
    sample_labels = []  # 用于存储采样的聚类标签
    mask = torch.ones(distance_matrix.shape[1]).to(device)  # 初始化掩码
    for i, cluster_i_budget in enumerate(cluster_budget):
        idx = index_matrix[i]  # 获取距离矩阵中前几个最近的样本索引
        idx = idx[mask[idx] != 0]  # 根据掩码过滤有效索引
        cluster_i_sample_idx = idx[:cluster_i_budget]  # 选择前 cluster_i_budget 个样本
        sample_idx.append(cluster_i_sample_idx)  # 存储采样的索引
        sample_labels.extend([i] * cluster_i_budget)  # 记录对应的聚类标签
        mask[cluster_i_sample_idx] = 0  # 更新掩码以避免重复选择

    sample_idx = torch.cat(sample_idx)  # 合并所有采样的索引
    sample_labels = torch.tensor(sample_labels).to(device)  # 将标签转换为张量并移动到设备
    assert len(sample_idx) == len(torch.unique(sample_idx)) == budget  # 确保样本数量和预算匹配
    return sample_idx, sample_labels#,cluster_weight,cluster_budget,index_matrix  # 返回采样的索引和聚类标签


def adjust_learning_rate(optimizer, epoch, initial_lr, lr_decay_factor, lr_decay_epoch):
    """
    调整优化器的学习率。

    参数：
    optimizer: 优化器对象，例如 torch.optim.SGD 或 torch.optim.Adam。
    epoch: 当前训练的 epoch 数。
    initial_lr: 初始学习率。
    lr_decay_factor: 学习率衰减因子，每次更新时学习率将乘以该因子。
    lr_decay_epoch: 学习率更新的周期（每多少个 epoch 更新一次）。
    """
    # 判断是否到了更新学习率的 epoch
    if epoch % lr_decay_epoch == 0 and epoch > 0:
        # 计算新的学习率
        new_lr = initial_lr * (lr_decay_factor ** (epoch // lr_decay_epoch))
        # 更新优化器中的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"Epoch {epoch}: 更新学习率为 {new_lr}")



def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def nt_xent(x, t=0.5, features2=None, index=None, sup_weight=0, COLT=None):
    # x: 输入特征，通常是模型的输出
    # t: 温度参数，用于调整对比损失的尺度
    # features2: 另一个特征集，用于计算正样本对
    # index: 用于区分类别标签的索引
    # sup_weight: 权重，通常用于监督损失的加权
    # COLT: 额外的损失计算选项，决定是否使用特定的损失计算方法

    if features2 is None:
        out = F.normalize(x, dim=-1)  # 对输入特征进行归一化
        d = out.size()
        batch_size = d[0] // 2  # 计算批次大小（假设 x 的前半部分是正样本，后半部分是负样本）
        out = out.view(batch_size, 2, -1).contiguous()  # 重新排列特征，使其形状为 (batch_size, 2, feature_dim)
        out_1 = out[:, 0]  # 正样本特征
        out_2 = out[:, 1]  # 负样本特征
    else:
        batch_size = x.shape[0]  # 获取批次大小
        out_1 = F.normalize(x, dim=-1)  # 归一化第一个特征集
        out_2 = F.normalize(features2, dim=-1)  # 归一化第二个特征集

    out = torch.cat([out_1, out_2], dim=0)  # 拼接两个特征集，形成一个大的特征集合

    # 计算所有特征对之间的相似性（点积），然后按温度参数缩放
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / t)

    sup_neg = neg  # 保存原始的负样本对相似性矩阵

    if (index == -1).sum() != 0 and COLT:
        id_mask = (index != -1)  # 识别出在数据集中存在的类别（正样本）
        ood_mask = (index == -1)  # 识别出超出分布的数据（负样本）

        # 计算是否是同一类样本的掩码，排除自身样本对（对角线）
        sup_pos_mask = (((ood_mask.view(-1, 1) & ood_mask.view(1, -1)) | (
                    id_mask.view(-1, 1) & id_mask.view(1, -1))).repeat(2, 2) & (
                            ~(torch.eye(index.shape[0] * 2).bool().to(device))))

        sup_pos_mask = sup_pos_mask.float()

        # 创建一个掩码，用于对正样本对进行加权
        mask_pos_view = torch.zeros_like(sup_pos_mask)
        mask_pos_view[sup_pos_mask.bool()] = 1

    # 计算负样本对掩码
    mask = get_negative_mask(batch_size).to(device)
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # 计算正样本对的相似性（点积），然后拼接
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / t)
    pos = torch.cat([pos, pos], dim=0)

    Ng = neg.sum(dim=-1)  # 计算每个样本的总负样本对相似性
    norm = pos + Ng  # 计算归一化因子

    # 计算负样本对的归一化 logits
    neg_logits = torch.div(neg, norm.view(-1, 1))
    neg_logits = torch.cat([neg_logits[:batch_size].unsqueeze(0), neg_logits[batch_size:].unsqueeze(0)], dim=0)

    # 计算最终的损失
    loss = (- torch.log(pos / (pos + Ng)))
    loss_reshape = loss.clone().detach().view(2, batch_size).mean(0)  # 计算每个正样本对的损失
    loss = loss.mean()  # 计算所有样本对的平均损失

    # 如果 COLT 标志为真且有超出分布的数据，则计算额外的监督损失
    if (index == -1).sum() != 0 and COLT:
        sup_loss = (- torch.log(sup_neg / (pos + Ng)))
        sup_loss = sup_weight * (1/(mask_pos_view.sum(1))) * (mask_pos_view * sup_loss).sum(1)
        loss += sup_loss.mean()  # 将监督损失加入到最终的总损失中

    return neg_logits, loss_reshape, loss

import math

COLT=True



encoder = RSNet(BasicBlock, [2, 2, 2, 2],num_classes).to(device)
            
print(len(dataset_test)) 
   

trloader = DataLoader(
            dataset=dataset_train,
            batch_size=64,
            shuffle=True,
          
            num_workers=0,
            pin_memory=True
        )

train_loader_test_trans= DataLoader(
            dataset=dataset_train,
            batch_size=64,
            shuffle=True,
          
            num_workers=0,
            pin_memory=True
        )


teloader = torch.utils.data.DataLoader(
                dataset_test,
                num_workers=0,
                batch_size=64,
                shuffle=False,
                pin_memory=True)
 
# sample_loader_test_trans = torch.utils.data.DataLoader(
#                 dataset_test,
#                 num_workers=0,
#                 batch_size=64,
#                 shuffle=False,
#                 pin_memory=True,
#                 drop_last=True)     
lr=0.001
optimizer = torch.optim.Adam(
            list(encoder.parameters()),
            lr=lr
        )
        
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, 
#             T_max=len(trloader), 
#             eta_min=0,
#             last_epoch=-1
#         )

cls_num_list =dataset_train.get_cls_num_list()
#print(cls_num_list)
cls_num = len(cls_num_list)
print(cls_num_list,cls_num,sum(cls_num_list))
criterion_ce = LogitAdjust(cls_num_list).to(device)
criterion_scl = BalSCL(cls_num_list, 0.05).to(device)
#criterion_ccl=SupConLoss_ccl(temperature=0.1, grama=0.25, base_temperature=0.07).to(device)
        #self.writer = SummaryWriter('log')
        #self.softmax=nn.Softmax(dim=1)

global_step = 0
t=0.5
beta=0.97
k_largest_logits=10
tatol_epoch=100
preepoch=25
# encoder = RSNet(BasicBlock, [2, 2, 2, 2],num_classes).to(device)
#encoder.load_state_dict(torch.load("D:\\自监督长尾故障诊断\\HITpremodel_exp_0.96875.pth"))

#%%
# 文件路径设置
losses_file = 'D:\\自监督长尾故障诊断\\shuju\\train_losses_record.csv'
# 打开文件并写入头部信息
with open(losses_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Batch', 'CE_Loss', 'SCL_Loss', 'Loss', 'accrucy'])

momentum_tail_score = torch.zeros(tatol_epoch, len(dataset_train)).to(device)
shadow = torch.zeros(len(dataset_train))


for epoch in range(1,preepoch):
    batch_time = AverageMeter('Time', ':6.3f')
    ce_loss_all = AverageMeter('CE_Loss', ':.4e')
    nt_loss_all = AverageMeter('nt_Loss', ':.4e')
    scl_loss_all = AverageMeter('SCL_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    end = time.time()
    encoder.train()
    with tqdm(total=len(trloader)) as pbar:
              
        for i, data in enumerate(trloader):
            inputs, targets,index,cen= data
            # if epoch>0:
            #     print(cen)
            inputs1=inputs[:,:,:1024]
            inputs2=inputs[:,:,1024:1024*2]
            inputs3=inputs[:,:,1024*2:]

            inputs = torch.cat([inputs1, inputs2, inputs3], dim=0)
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = targets.shape[0]
            feat_mlp, logits, centers = encoder(inputs)
        
         
            centers = centers[:cls_num]
            _, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
            features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
            logits, _, __ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)
            out = torch.cat([f2, f3], dim=0)
            if (i + 1) == len(trloader):
                res = len(trloader.dataset) % len(trloader)
                if res == 0:
                    out =out
                    index = index
                else:
                        
                    #last_batch_each_gpu = math.ceil(len(index) / len(trloader))
                    mask = torch.zeros_like(index, dtype=torch.bool)
        
                    for j in range(len(index), res, -1):
                        mask[ j - 1] = True
                        
                    index = index[~mask]
                    out = out[(~mask).repeat(2)]
            
            neg = torch.exp(torch.mm(out, out.t().contiguous()) / t)
            
            # mask = get_negative_mask(batch_size).to(device)
            # neg = neg.masked_select(mask).view(2 * batch_size, -1)
            # pos = torch.exp(torch.sum(f2 * f3, dim=-1) / t)
            # pos = torch.cat([pos, pos], dim=0)
        
            # Ng = neg.sum(dim=-1)
            # norm = pos + Ng
            # neg_logits = torch.div(neg, norm.view(-1, 1))
            # neg_logits = torch.cat([neg_logits[:batch_size].unsqueeze(0), neg_logits[batch_size:].unsqueeze(0)], dim=0)
            
            neg_logits, loss_sample_wise, loss3 = nt_xent(out, t=0.5,
                                                                     index=index,
                                                                     sup_weight=0.2,
                                                                     COLT=True)
            neg_logits = neg_logits.mean(dim=0).detach()
            for count in range(out.shape[0] // 2):
                if not index[count] == -1:
                    if epoch > 1:
                        new_average = (1.0 - beta) * neg_logits[count].sort(descending=True)[0][
                                                                        :k_largest_logits].sum().clone().detach() \
                                      + beta * shadow[index[count]]
                    else:
                        new_average = neg_logits[count].sort(descending=True)[0][
                                      :k_largest_logits].sum().clone().detach()
                    shadow[index[count]] = new_average
            
                    momentum_tail_score[epoch-1, index[count]] = new_average
           
            scl_loss = criterion_scl(centers, features, targets)
                
            ce_loss = criterion_ce(logits, targets)
            loss =  ce_loss + 0.5 * scl_loss
            #loss =   scl_loss
            #loss =  ce_loss + loss3 
            ce_loss_all.update(ce_loss.item(), batch_size)
            #nt_loss_all.update(loss3.item(), batch_size)
            scl_loss_all.update(scl_loss.item(), batch_size)
            acc1,acc5 = accuracy(logits, targets, topk=(1,5))
            top1.update(acc1[0].item(), batch_size)
            top5.update(acc5[0].item(), batch_size)
            
            #top1.update(acc1[0].item(), batch_size)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            batch_time.update(time.time() - end)
            end = time.time()   
            with open(losses_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, i, ce_loss.item(), scl_loss.item(), loss.item(), acc1[0].item()])


            if i % 5== 0:
                 output = ('Epoch: [{0}][{1}/{2}] \t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                     
                      'SCL_Loss {scl_loss.val:.4f} ({scl_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(trloader), batch_time=batch_time,
                ce_loss=ce_loss_all, scl_loss=scl_loss_all, top1=top1, top5=top5,))  # TODO
                 print(output)

            
            pbar.update(1)
            global_step += 1
        momentum_weight = momentum_tail_score[epoch-1]
        
#save_file = 'D:\\自监督长尾故障诊断\\HITpremodel_step_0.96875.pth'.format(epoch) #you should add file
#torch.save(encoder.state_dict(),save_file)   

#encoder.load_state_dict(torch.load("D:\\自监督长尾故障诊断\\HITpremodel_step_0.96875.pth"))

import csv
# 文件路径设置
losses_file = 'D:\\自监督长尾故障诊断\\shuju\\train_losses_record3.csv'
# 打开文件并写入头部信息
with open(losses_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Batch', 'CE_Loss', 'SCL_Loss', 'Loss', 'Loss3','accrucy'])


a=validate(teloader, encoder, flag='val')

sample_idx,clulabel = sample_batch(train_loader_test_trans, encoder,ood_loader,
                                momentum_weight, args=None)
ood_sample_subset = torch.utils.data.Subset(ood_train_tran, sample_idx.tolist())
        
        #ood_sample_subset = dataset_test
del trloader
        

new_train_datasets = torch.utils.data.ConcatDataset([dataset_train, ood_sample_subset])
trloader = torch.utils.data.DataLoader(
            new_train_datasets,
            num_workers=0,
            batch_size=64,
            shuffle=True,
 
            pin_memory=True)


counter  = [new_train_datasets  [i][1] for i in range(len(new_train_datasets ))]
   

# 示例列表
counter = Counter(counter)

# 将统计结果按元素的大小排序
sorted_counter = sorted(counter.items())

# 生成只包含出现次数的列表
cls_num_list = [count for _, count in sorted_counter]
   
cls_num = len(cls_num_list)
#cluster_number= [t//max(min(cls_num_list),70) for t in cls_num_list]

criterion_ce = LogitAdjust(cls_num_list).to(device)
#criterion_scl1 = BalSCL2(cls_num_list, temperature=0.05, grama=0.25, base_temperature=0.07).to(device)
criterion_scl = BalSCL(cls_num_list, 0.05).to(device)
#criterion_ccl = SupConLoss_ccl(temperature=0.1, grama=0.25, base_temperature=0.07).to(device)
#cen=cluster(trloader,encoder,cluster_number,cls_num_list,"kmeans")
#replace_col_dataset = ReplaceColumnConcatDataset(new_train_datasets, cen)

# trloader = torch.utils.data.DataLoader(
#             replace_col_dataset,
#             num_workers=0,
#             batch_size=64,
#             shuffle=True,
#             pin_memory=True)


# a1=len(new_train_datasets)
# momentum_tail_score = torch.zeros(tatol_epoch,a1).to(device)
# shadow = torch.zeros(a1)
#encoder1 = RSNet(BasicBlock, [2, 2, 2, 2],num_classes).to(device)
encoder1 = encoder
lr_decay_factor=0.1
lr_decay_epoch=14

for epoch in range(preepoch,tatol_epoch):
    if  epoch%30==0:
        lr=0.001
    adjust_learning_rate(optimizer, epoch, lr, lr_decay_factor, lr_decay_epoch)
    batch_time = AverageMeter('Time', ':6.3f')
    ce_loss_all = AverageMeter('CE_Loss', ':.4e')
    nt_loss_all = AverageMeter('nt_Loss', ':.4e')
    scl_loss_all = AverageMeter('SCL_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    end = time.time()
    encoder1.train()
    if (epoch-preepoch)>21 and (epoch-preepoch)%30==0:
        a=validate(teloader, encoder, flag='val')
        sample_idx,clulabel = sample_batch(train_loader_test_trans, encoder,ood_loader,
                                momentum_weight, args=None)
        ood_sample_subset = torch.utils.data.Subset(ood_train_tran, sample_idx.tolist())
        #ood_sample_subset = dataset_test
        del trloader
        
        new_train_datasets = torch.utils.data.ConcatDataset([dataset_train, ood_sample_subset])
        trloader = torch.utils.data.DataLoader(
                    new_train_datasets,
                    num_workers=0,
                    batch_size=64,
                    shuffle=True,
                    pin_memory=True)
        
        
        counter  = [new_train_datasets[i][1] for i in range(len(new_train_datasets))]
           
        
        # 示例列表
        counter = Counter(counter)
        
        # 将统计结果按元素的大小排序
        sorted_counter = sorted(counter.items())
        
        # 生成只包含出现次数的列表
        cls_num_list = [count for _, count in sorted_counter]
           
        cls_num = len(cls_num_list)
        #cluster_number= [t//max(min(cls_num_list),70) for t in cls_num_list]
        
        criterion_ce = LogitAdjust(cls_num_list).to(device)
        #criterion_scl1 = BalSCL2(cls_num_list, temperature=0.05, grama=0.25, base_temperature=0.07).to(device)
        criterion_scl = BalSCL(cls_num_list, 0.05).to(device)
        #criterion_ccl = SupConLoss_ccl(temperature=0.1, grama=0.25, base_temperature=0.07).to(device)
        #cen=cluster(trloader,encoder,cluster_number,cls_num_list,"kmeans")
        #replace_col_dataset = ReplaceColumnConcatDataset(new_train_datasets, cen)
        
        # trloader = torch.utils.data.DataLoader(
        #             replace_col_dataset,
        #             num_workers=0,
        #             batch_size=64,
        #             shuffle=True,
        #             pin_memory=True)

        

    with tqdm(total=len(trloader)) as pbar:
              
        for i, data in enumerate(trloader):
            inputs, targets,index,cen= data
            # if epoch>0:
            #     print(cen)
            inputs1=inputs[:,:,:1024]
            inputs2=inputs[:,:,1024:1024*2]
            inputs3=inputs[:,:,1024*2:]

            inputs = torch.cat([inputs1, inputs2, inputs3], dim=0)
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = targets.shape[0]
            feat_mlp, logits, centers = encoder1(inputs)
        
         
            centers = centers[:cls_num]
            _, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
            features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
            logits, _, __ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)
            out = torch.cat([f2, f3], dim=0)
            if (i + 1) == len(trloader):
                res = len(trloader.dataset) % len(trloader)
                #last_batch_each_gpu = math.ceil(len(index) / len(trloader))
                mask = torch.zeros_like(index, dtype=torch.bool)
    
                for j in range(len(index), res, -1):
                    mask[ j - 1] = True
    
                index = index[~mask]
                out = out[(~mask).repeat(2)]
            
            #neg = torch.exp(torch.mm(out, out.t().contiguous()) / t)
            
            # mask = get_negative_mask(batch_size).to(device)
            # neg = neg.masked_select(mask).view(2 * batch_size, -1)
            # pos = torch.exp(torch.sum(f2 * f3, dim=-1) / t)
            # pos = torch.cat([pos, pos], dim=0)
        
            # Ng = neg.sum(dim=-1)
            # norm = pos + Ng
            # neg_logits = torch.div(neg, norm.view(-1, 1))
            # neg_logits = torch.cat([neg_logits[:batch_size].unsqueeze(0), neg_logits[batch_size:].unsqueeze(0)], dim=0)
            
            neg_logits, loss_sample_wise, loss3 = nt_xent(out, t=0.5,
                                                                     index=index,
                                                                     sup_weight=0.2,
                                                                     COLT=True)
            neg_logits = neg_logits.mean(dim=0).detach()
            for count in range(out.shape[0] // 2):
                if not index[count] == -1:
                    if (epoch-preepoch) > 1:
                        new_average = (1.0 - beta) * neg_logits[count].sort(descending=True)[0][
                                                                        :k_largest_logits].sum().clone().detach() \
                                      + beta * shadow[index[count]]
                    else:
                        new_average = neg_logits[count].sort(descending=True)[0][
                                      :k_largest_logits].sum().clone().detach()
                    shadow[index[count]] = new_average
            
                    momentum_tail_score[epoch-1, index[count]] = new_average
           
            scl_loss =  criterion_scl(centers, features, targets)
            ce_loss = criterion_ce(logits, targets)
            loss =  ce_loss + 0.5 * scl_loss+0.7*loss3 
            #loss =   scl_loss
            #loss =  ce_loss+loss3 
            ce_loss_all.update(ce_loss.item(), batch_size)
            #nt_loss_all.update(loss3.item(), batch_size)
            scl_loss_all.update(scl_loss.item(), batch_size)
            acc1,acc5 = accuracy(logits, targets, topk=(1,5))
            top1.update(acc1[0].item(), batch_size)
            top5.update(acc5[0].item(), batch_size)
            
            #top1.update(acc1[0].item(), batch_size)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            batch_time.update(time.time() - end)
            end = time.time()  
            with open(losses_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch, i, ce_loss.item(), scl_loss.item(), loss.item(), loss3.item(),acc1[0].item()])



            if i % 5== 0:
                 output = ('Epoch: [{0}][{1}/{2}] \t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                     
                      'SCL_Loss {scl_loss.val:.4f} ({scl_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(trloader), batch_time=batch_time,
                ce_loss=ce_loss_all, scl_loss=scl_loss_all, top1=top1, top5=top5,))  # TODO
                 print(output)

             
            pbar.update(1)
            global_step += 1
        momentum_weight = momentum_tail_score[epoch-1]
    if epoch %10  == 0:
        a=validate(teloader, encoder, flag='val')
    if (epoch-preepoch) %30 == 0 and (epoch-preepoch) >10:
       
        save_file = 'HITmodel2_epoch_{:03d}_{:.4f}.pth'.format(epoch,a.item()) #you should add file
        torch.save(encoder1.state_dict(),save_file)   
    # if epoch > -1:
        
    #     sample_idx,clulabel = sample_batch(train_loader_test_trans, encoder,teloader,
    #                                     momentum_weight, args=None)
    #     ood_sample_subset = torch.utils.data.Subset(dataset_test, sample_idx.tolist())
    #     #ood_sample_subset = dataset_test

    #     new_train_datasets = torch.utils.data.ConcatDataset([dataset_train, ood_sample_subset])
        
        
    #     counter  = [new_train_datasets  [i][1] for i in range(len(new_train_datasets  ))]
       
        
    #     # 示例列表
    #     counter = Counter(counter)
        
    #     # 将统计结果按元素的大小排序
    #     sorted_counter = sorted(counter.items())
        
    #     # 生成只包含出现次数的列表
    #     cls_num_list = [count for _, count in sorted_counter]
    #     cls_num_list =dataset_train.get_cls_num_list()
    #     cls_num = len(cls_num_list)
        
    #     criterion_ce = LogitAdjust(cls_num_list).to(device)
    #     criterion_scl = BalSCL2(cls_num_list, 0.05).to(device)
    #     #criterion_ccl = SupConLoss_ccl(temperature=0.1, grama=0.25, base_temperature=0.07).to(device)
    #     targets=cluster(new_train_datasets,encoder,180,cls_num_list,"kmeans")
    #     new_train_datasets.new_labels = targets 

    #     del trloader
    #     trloader = torch.utils.data.DataLoader(
    #                 new_train_datasets,
    #                 num_workers=0,
    #                 batch_size=64,
 
    #                 pin_memory=True)

    
    
                

            



### loss













    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
