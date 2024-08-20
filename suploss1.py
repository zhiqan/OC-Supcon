# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 03:05:19 2024

@author: Owner
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



import math
import numpy as np


class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()  # 调用父类的初始化方法
        cls_num_list = torch.FloatTensor(cls_num_list)  # 将类别数量列表转换为浮点型张量
        cls_p_list = cls_num_list / cls_num_list.sum()  # 计算每个类别的样本比例
        m_list = tau * torch.log(cls_p_list)  # 计算 logit 调整值，tau 是缩放因子
        self.m_list = m_list.view(1, -1)  # 调整 logit 的调整值形状为 (1, num_classes)
        self.weight = weight  # 权重用于 cross_entropy

    def forward(self, x, target):
        x_m = x + self.m_list  # 将 logit 调整值添加到输入 logit
        return F.cross_entropy(x_m, target, weight=self.weight)  # 计算带有调整的交叉熵损失



class CCLC(nn.Module):
    def __init__(self, cls_num_list=None, temperature=0.1):
        super(CCLC, self).__init__()
        self.temperature = temperature  # 温度参数，用于调整对比损失的尺度
        self.cls_num_list = cls_num_list  # 每个类别的样本数量，用于类别平衡

    def forward(self, centers1, features, targets,):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))  # 确定计算设备（CPU 或 GPU）
        batch_size = features.shape[0]  # 获取当前批次的样本数量
        targets = targets.contiguous().view(-1, 1)  # 将目标标签展平为二维张量，形状为 (batch_size, 1)
        targets_centers = torch.arange(len(self.cls_num_list), device=device).view(-1, 1)  # 生成类别中心的索引
        targets = torch.cat([targets.repeat(2, 1), targets_centers], dim=0)  # 将目标标签重复一次，并将类别中心索引拼接到一起
        batch_cls_count = torch.eye(len(self.cls_num_list))[targets].sum(dim=0).squeeze()  # 计算每个类别在当前批次中的样本数量

        # 计算正负样本对的掩码
        mask = torch.eq(targets[:2 * batch_size], targets.T).float().to(device)  # 生成目标标签相同的样本对掩码
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )  # 用于掩盖对角线上的自对比（即，样本自身的对比）
        mask = mask * logits_mask  # 应用掩码
        
        # class-complement
        features = torch.cat(torch.unbind(features, dim=1), dim=0)  # 将每个样本的特征沿着类别维度展开
        features = torch.cat([features, centers1], dim=0)  # 将特征和类别中心拼接起来
        logits = features[:2 * batch_size].mm(features.T)  # 计算对比损失的 logits
        logits = torch.div(logits, self.temperature)  # 对 logits 进行温度缩放

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)  # 数值稳定性处理，减去每行的最大值
        logits = logits - logits_max.detach()  # 防止数值溢出

        # class-averaging
        exp_logits = torch.exp(logits) * logits_mask  # 计算 exponentiated logits，并应用掩码
        per_ins_weight = torch.tensor([batch_cls_count[i] for i in targets], device=device).view(1, -1).expand(
            2 * batch_size, 2 * batch_size + len(self.cls_num_list)) - mask  # 计算每个样本的权重
        exp_logits_sum = exp_logits.div(per_ins_weight).sum(dim=1, keepdim=True)  # 对每个样本进行归一化处理
        
        log_prob = logits - torch.log(exp_logits_sum)  # 计算 log probability
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # 计算正样本对的平均 log probability

        loss = - mean_log_prob_pos  # 计算负的平均 log probability 作为损失
        loss = loss.view(2, batch_size).mean()  # 将损失重新形状为 (2, batch_size)，并计算均值
        return loss  

