# -*- coding: utf-8 -*-
"""
Title: Classifiers - Simple Linear Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import number_stability

class Classifier(nn.Module):
    """Multi-view Consistent Classification."""
    def __init__(self, dim_in, num_class):
        super(Classifier, self).__init__()
        # the mlp for classification
        self.classifier_v1 = nn.Sequential(
            nn.Linear(dim_in, num_class)
        )
        self.classifier_v2 = nn.Sequential(
            nn.Linear(dim_in, num_class)
        )
        # define the loss functions
        self.cross_entropy = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss()
        self.init_mlp()

    def init_mlp(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.)

    def get_pred_v1(self, emb):
        return self.classifier_v1(emb)

    def get_pred_v2(self, emb):
        return self.classifier_v2(emb)

    def kl_categorical(self, pred_v1, pred_v2):
        prob_v1 = F.softmax(pred_v1, dim=-1)
        kl = torch.sum(prob_v1 * (F.log_softmax(pred_v1, dim=-1) - 
                                  F.log_softmax(pred_v2, dim=-1)), dim=1)
        return torch.mean(kl)

    def cal_loss_reg(self, emb_v1, emb_v2, train_mask, y):
        """Label invariance constraints."""
        # go through classifier
        pred_v1 = number_stability(self.classifier_v1(emb_v1))
        pred_v2 = number_stability(self.classifier_v2(emb_v2))
        prob_v1 = F.softmax(pred_v1, dim=-1)
        prob_v2 = F.softmax(pred_v2, dim=-1)
        # calculate loss
        loss_train_v1 = self.cross_entropy(prob_v1[train_mask], y[train_mask])
        loss_train_v2 = self.cross_entropy(prob_v2[train_mask], y[train_mask])
        loss = 0.5 * (loss_train_v1 + loss_train_v2)
        return loss
    
    def cal_loss_aug(self, emb_v1, emb_v2):
        """Loss for virtual adversarial augmentation."""
        # go through classifier
        pred_v1 = number_stability(self.classifier_v1(emb_v1))
        pred_v2 = number_stability(self.classifier_v2(emb_v2))
        prob_v1 = F.softmax(pred_v1, dim=-1)
        prob_v2 = F.softmax(pred_v2, dim=-1)
        loss = -0.5 * (self.kl_loss(F.log_softmax(pred_v1, dim=-1), prob_v2) + 
                       self.kl_loss(F.log_softmax(pred_v2, dim=-1), prob_v1))
        return loss

    def forward(self, emb_v1, emb_v2, train_mask, y):
        # go through classifiers
        pred_v1 = number_stability(self.classifier_v1(emb_v1))
        pred_v2 = number_stability(self.classifier_v2(emb_v2))
        prob_v1 = F.softmax(pred_v1, dim=-1)
        prob_v2 = F.softmax(pred_v2, dim=-1)
        # calculate loss
        loss_train_v1 = self.cross_entropy(prob_v1[train_mask], y[train_mask])
        loss_train_v2 = self.cross_entropy(prob_v2[train_mask], y[train_mask])
        loss_cl = 0.5 * (self.kl_loss(F.log_softmax(pred_v1, dim=-1), prob_v2) + 
                         self.kl_loss(F.log_softmax(pred_v2, dim=-1), prob_v1))
        loss = loss_cl + 0.5 * (loss_train_v1 + loss_train_v2)
        return loss
