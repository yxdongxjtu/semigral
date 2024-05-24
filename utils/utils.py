# -*- coding: utf-8 -*-
"""
Title: Some Helper Functions
"""

import torch
import torch.optim as optim
import torch_geometric.nn as pyg_nn


############################ Model Definition ##########################

def get_activation(act):
    """Get activation function."""
    if act == 'relu':
        return torch.nn.ReLU(inplace=True)
    elif act == 'prelu':
        return torch.nn.PReLU()
    elif act == 'sigmoid':
        return torch.nn.Sigmoid()
    elif act == 'softmax':
        return torch.nn.Softmax()
    elif (act is None) or (act == 'none'):
        return torch.nn.Identity()
    else:
        raise NotImplementedError


def get_conv_model(dim_in, dim_out, model_type):
    """Build enc model for each layer."""
    if model_type == "GCN":
        return pyg_nn.GCNConv(dim_in, dim_out)
    elif model_type == "GAT":
        return pyg_nn.GATConv(dim_in, dim_out)  
    else:
        raise NotImplementedError


def get_optimizer(opt, params, lr, weight_decay):
    """Build optimizer for the given model."""
    params_fn = filter(lambda p: p.requires_grad, list(params))
    if opt == 'adam':
        optimizer = optim.Adam(params_fn, lr=lr, weight_decay=weight_decay)
    elif opt == 'adamw':
        optimizer = optim.AdamW(params_fn, lr=lr, weight_decay=weight_decay)
    return optimizer


def number_stability(mat_th):
    logits_max, _ = torch.max(mat_th, dim=-1, keepdim=True)
    logits = mat_th - logits_max.detach()
    return torch.exp(logits)


def cal_accs(data_pyg, labels, y_pesudo):
    acc_train = (torch.sum(y_pesudo[data_pyg.train_mask]==labels[data_pyg.train_mask]).float() /
                    labels[data_pyg.train_mask].shape[0])
    acc_val = (torch.sum(y_pesudo[data_pyg.val_mask]==labels[data_pyg.val_mask]).float() /
                labels[data_pyg.val_mask].shape[0])
    acc_test = (torch.sum(y_pesudo[data_pyg.test_mask]==labels[data_pyg.test_mask]).float() /
                labels[data_pyg.test_mask].shape[0])
    return acc_train, acc_val, acc_test
