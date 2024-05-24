# -*- coding: utf-8 -*-
"""
Title: Models in SemiGraL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter

from models.encoder import get_enc
from utils.utils import get_activation


class Encoder(nn.Module):
    """Graph Node Encoder"""
    def __init__(self, args, dim_in, mi_dim=128):
        super(Encoder, self).__init__()
        self.temperature = args.mi_temperature
        # define the encoder
        self.enc = get_enc(args, dim_in)
        # define the projection layer
        self.proj = nn.Sequential(
            nn.Linear(args.hidden_units[-1], mi_dim),
            get_activation(args.enc_act)
        )
        self.triplet_loss = nn.TripletMarginLoss(margin=0.7, p=2)
        # init the MLP layers
        self.init_mlp()

    def init_mlp(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.)
    
    def get_embeddings(self, x, edge_index):
        return self.enc(x, edge_index)
    
    def get_views_embeddings(self, data_pyg_v1, data_pyg_v2):
        emb_v1 = self.enc(data_pyg_v1.x, data_pyg_v1.edge_index)
        emb_v2 = self.enc(data_pyg_v2.x, data_pyg_v2.edge_index)
        return emb_v1, emb_v2

    def get_adv_embeddings(self, x_v1, x_v2, edge_index_v1, edge_index_v2):
        emb_v1 = self.enc(x_v1, edge_index_v1)
        emb_v2 = self.enc(x_v2, edge_index_v2)
        return emb_v1, emb_v2

    def cal_loss_reg(self, emb_v1, emb_v2, edge_index):
        """Node-level equivariance constraints."""
        edge_index_new = remove_self_loops(edge_index)[0]
        emb_v1 = F.normalize(emb_v1, p=2, dim=-1)
        emb_v2 = F.normalize(emb_v2, p=2, dim=-1)
        # triplet loss to define feasible region for data augmentation
        emb_v2_a = emb_v2[edge_index_new[0,:],:]
        emb_v1_p = emb_v1[edge_index_new[0,:],:]
        emb_v1_n = emb_v1[edge_index_new[1,:],:]
        loss = self.triplet_loss(emb_v2_a, emb_v1_p, emb_v1_n)
        return loss

    def cal_loss_mi(self, emb_v1, emb_v2, pos_mask):
        # calculate the similarity matrix
        sim_mat = torch.einsum('ik,jk->ij', emb_v1, emb_v2) / self.temperature
        # for numberical stability
        logits_max, _ = torch.max(sim_mat, dim=1, keepdim=True)
        sim_mat = sim_mat - logits_max.detach()
        sim_mat = torch.exp(sim_mat)
        # calculate the loss
        sim_pos_vec = sim_mat[pos_mask._indices()[0], pos_mask._indices()[1]]
        loss_vec = sim_pos_vec / (sim_mat.sum(dim=1) - 
                   scatter(src=sim_pos_vec, index=pos_mask._indices()[0], reduce='sum')
                   )[pos_mask._indices()[0,:]]
        loss = - (scatter(src=torch.log(loss_vec), index=pos_mask._indices()[0], reduce='sum') /
                  torch.sparse.sum(pos_mask, dim=1)._values()).mean()
        return loss

    def forward(self, data_pyg_v1, data_pyg_v2, pos_mask):
        # go through the encoder
        emb_v1 = self.enc(data_pyg_v1.x, data_pyg_v1.edge_index)
        emb_v2 = self.enc(data_pyg_v2.x, data_pyg_v2.edge_index)
        # calculate MI-loss
        emb_v1_proj = self.proj(emb_v1)
        emb_v2_proj = self.proj(emb_v2)
        emb_v1_proj = F.normalize(emb_v1_proj, p=2, dim=-1)
        emb_v2_proj = F.normalize(emb_v2_proj, p=2, dim=-1)
        loss = 0.5 * (self.cal_loss_mi(emb_v1_proj, emb_v2_proj, pos_mask) + 
                      self.cal_loss_mi(emb_v2_proj, emb_v1_proj, pos_mask))
        return loss
