# -*- coding: utf-8 -*-
"""
Title: Data Loading and Pre-processing
"""

import torch
from torch_geometric.datasets import Planetoid, Amazon, Coauthor

DATA_CLASS_DICT = {
    'cora': Planetoid,
    'citeseer': Planetoid,
    'pubmed': Planetoid,
    'Computers': Amazon,
    'Photo': Amazon,
    'CS': Coauthor,
    'Physics': Coauthor,
}


############################ Data Loaders ###########################

def load_data(dataset):
    """Loading dataset (pyg)"""
    if dataset in ['cora', 'citeseer', 'pubmed', 'CS', 'Physics', 'Computers', 'Photo']:
        data_path = f'./data/{dataset}'
        data_pyg = DATA_CLASS_DICT[dataset](data_path, f'{dataset}').data
        return data_pyg
    else:
        raise ValueError(f'Unknown dataset: {dataset}.')


def get_pos_mask(data_pyg):
    """Construct pos mask matrix."""
    labels = data_pyg.y
    class_num = labels.unique().shape[0]
    nodes_num = data_pyg.num_nodes
    train_mask = data_pyg.train_mask
    pos_mask = torch.stack([torch.arange(nodes_num), 
                            torch.arange(nodes_num)]).to(labels.device)
    label_dict = {}
    for class_idx in range(class_num):
        nodes_idx = torch.where(labels==class_idx, True, False) & train_mask
        nodes_idx = torch.where(nodes_idx)[0]
        label_dict[class_idx] = nodes_idx
        pos_mask_idx = gen_pos_mask_idx(nodes_idx)
        pos_mask = torch.cat([pos_mask, pos_mask_idx], dim=-1)
    pos_mask = torch.unique(pos_mask, dim=-1)
    # build pos mask matrix
    pos_mask_mat = build_pos_mask_matrix(pos_mask, nodes_num)
    return pos_mask_mat


def build_pos_mask_matrix(pos_mask_index, nodes_num):
    """build pos mask matrix."""
    pos_mask_mat = torch.sparse_coo_tensor(
        pos_mask_index,
        torch.ones(pos_mask_index.shape[1], dtype=torch.long).to(pos_mask_index.device),
        [nodes_num, nodes_num]
    )
    return pos_mask_mat


def gen_pos_mask_idx(nodes_idx):
    nodes_num = nodes_idx.shape[0]
    row_idx = torch.cat([nodes_idx[i].repeat(nodes_num)
                         for i in range(nodes_num)])
    col_idx = nodes_idx.repeat(nodes_num)
    pos_mask_idx = torch.stack([row_idx, col_idx])
    return pos_mask_idx


def init_pseudo_label(labels, train_mask):
    """Construct pseduo label for training"""
    pseudo_label = labels.clone()
    pseudo_label[~train_mask] = -1  # prevent label leakage
    return pseudo_label
