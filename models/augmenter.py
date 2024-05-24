# -*- coding: utf-8 -*-
"""
Title: Virtual Adversarial Augmentation
"""

import torch
import torch.nn as nn 


class Augmenter(nn.Module):
    """Virtual Adversarial Augmentation"""
    def __init__(self, w_neg_cla=200, w_reg_enc=2, alpha=1e1, eps=1e-1, n_iter=1):
        super(Augmenter, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.w_neg_cla = w_neg_cla
        self.w_reg_enc = w_reg_enc
        self.n_iter = n_iter
    
    def _l2_normalize(self, aug):
        aug_reshaped = aug.view(aug.shape[0], -1, *(1 for _ in range(aug.dim() - 2)))
        aug /= torch.norm(aug_reshaped, dim=1, keepdim=True).detach() + 1e-8
        return aug
    
    def init_perturbation(self, data_x):
        r_aug_v1 = torch.rand(data_x.shape).sub(0.5).to(data_x.device)
        r_aug_v1 = self._l2_normalize(r_aug_v1)
        r_aug_v2 = torch.rand(data_x.shape).sub(0.5).to(data_x.device)
        r_aug_v2 = self._l2_normalize(r_aug_v2)
        return r_aug_v1, r_aug_v2

    def augment_feat(self, data_pyg, encoder, classifier, train_mask, y):
        data_x = data_pyg.x
        data_edge_index = data_pyg.edge_index
        # prepare random perturbation
        r_aug_v1, r_aug_v2 = self.init_perturbation(data_x)
        x_norm = torch.norm(data_x, p=2, dim=1, keepdim=True)
        # calculate adversarial perturbation
        encoder.eval()
        classifier.eval()
        for _ in range(self.n_iter):
            r_aug_v1.requires_grad_()
            r_aug_v2.requires_grad_()
            emb_v1, emb_v2 = encoder.get_adv_embeddings(
                data_x+(r_aug_v1*self.alpha), data_x+(r_aug_v2*self.alpha),
                data_edge_index, data_edge_index
            )
            # calculate the loss
            loss_neg_cla = classifier.cal_loss_aug(emb_v1, emb_v2)
            loss_reg_enc = encoder.cal_loss_reg(emb_v1, emb_v2, data_edge_index)
            loss_reg_cla = classifier.cal_loss_reg(emb_v1, emb_v2, train_mask, y)
            loss = loss_neg_cla * self.w_neg_cla + loss_reg_enc * self.w_reg_enc + loss_reg_cla            
            # calculate perturbation
            loss.backward()
            r_aug_v1 = self._l2_normalize(r_aug_v1.grad) * x_norm
            r_aug_v2 = self._l2_normalize(r_aug_v2.grad) * x_norm
            with torch.cuda.device(train_mask.device):
                torch.cuda.empty_cache()
        # return augmented data
        data_x_v1 = data_x + r_aug_v1 * self.eps
        data_x_v2 = data_x + r_aug_v2 * self.eps
        return data_x_v1, data_x_v2
