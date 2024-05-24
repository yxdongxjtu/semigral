# -*- coding: utf-8 -*-
"""
Title: GNN-based Node Encoder
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Sequential

from utils.utils import get_conv_model, get_activation


class GraphENC(nn.Module):
    """Graph Encoder."""
    def __init__(self, dim_in, dim_hiddens, dim_out, enc_types,
                 normalized_embs, dropout, act):
        super(GraphENC, self).__init__()
        # build Graph Encoder
        self.enc_convs = self._build_convs(
            dim_in, dim_hiddens, dim_out, normalized_embs, enc_types,
            act, dropout
        )
    
    @staticmethod
    def _build_convs(dim_in, dim_hiddens, dim_out, normalized_embs, enc_types, act, dropout):
        """Build Graph Encoder."""
        enc_convs = []
        for layer, dim_hidden in enumerate(dim_hiddens):
            # if layer == 0:
            conv = [
                (nn.Dropout(dropout), 'x -> x'),
                (get_conv_model(dim_in, dim_hidden, enc_types[layer]), 
                'x, edge_index -> x')
            ]
            if normalized_embs[layer]:
                conv += [
                    nn.BatchNorm1d(dim_hidden, momentum=0.01),
                    get_activation(act)
                ]
            else:
                conv += [get_activation(act)]
            enc_convs += conv
            dim_in = dim_hidden
        # build the final layer
        enc_convs += [
            (nn.Dropout(dropout), 'x -> x'),
            (get_conv_model(dim_in, dim_out, enc_types[-1]), 'x, edge_index -> x'),
            get_activation(act)
        ]
        return Sequential('x, edge_index', enc_convs)
    
    def forward(self, x, edge_index):
        emb = self.enc_convs(x=x, edge_index=edge_index)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb

############################ Generate Encoder ##########################

def get_enc(args, dim_in):
    enc_type_list, enc_norm = parse_enc_params(
        enc_type=args.enc_type,
        hidden_units=args.hidden_units,
        enc_norm=None
    )
    enc = GraphENC(
        dim_in=dim_in,
        dim_hiddens=args.hidden_units[:-1],
        dim_out=args.hidden_units[-1],
        enc_types=enc_type_list,
        normalized_embs=enc_norm,
        dropout=args.enc_drop,
        act=args.enc_act
    )
    return enc


def parse_enc_params(enc_type, hidden_units, enc_norm):
    enc_type_list = [enc_type] * len(hidden_units)
    if enc_norm is None:
        enc_norm = [True] * len(enc_type_list)
    return enc_type_list, enc_norm
