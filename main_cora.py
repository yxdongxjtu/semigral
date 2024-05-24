# -*- coding: utf-8 -*-
"""
Title: Code for <Semi-Supervised Graph Contrastive Learning with Virtual Adversarial Augmentation>
Description: Interface for Cora Dataset
"""

import os
import argparse
import random

import torch
import numpy as np 
import torch.nn.functional as F

from models.models import Encoder
from models.augmenter import Augmenter
from models.classifier import Classifier
from utils.data_process import load_data, get_pos_mask, init_pseudo_label
from utils.utils import get_optimizer, cal_accs


def argument():
    """Get settings"""
    parser = argparse.ArgumentParser(description='SemiGraL')
    # Data settings
    parser.add_argument('--dataset', type=str, default='cora', help='Name of dataset.')
    # Architecture settings
    parser.add_argument('--mi_temperature', type=float, default=0.3, help='The temperature parameter in MI')
    parser.add_argument('--enc_type', type=str, default='GCN', help='The encoder gnn layer type')
    parser.add_argument('--hidden_units', type=list, default=[512,512], help='The hidden layer size for encoder')
    parser.add_argument('--enc_drop', type=float, default=0.2, help='The dropout value for encoder')
    parser.add_argument('--enc_act', type=str, default='prelu', help='The ctivation function for encoder')
    # Optimization settings
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--aug_iter', type=int, default=1, help='The power iterations M in VAA.')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs.')
    parser.add_argument('--lr', type=float, default=3e-3, help='The learning rate of training.')
    parser.add_argument('--wd', type=float, default=1e-4, help='The weight decay of training.')
    args = parser.parse_args()
    return args


def train_encoder(encoder, opt, data_pyg_v1, data_pyg_v2, pos_mask):
    """Training procedure of Encoder."""
    opt.zero_grad()
    # calculate the loss
    loss = encoder(data_pyg_v1, data_pyg_v2, pos_mask)
    # update the model
    loss.backward()
    opt.step()
    with torch.cuda.device(pos_mask.device):
        torch.cuda.empty_cache()    
    return float(loss.detach().item())


def train_classifier(classifier, opt, data_pyg_v1, data_pyg_v2, encoder, train_mask, y):
    """Training procedure of Classifier."""
    opt.zero_grad()
    # get node embeddings
    emb_v1, emb_v2 = encoder.get_views_embeddings(
        data_pyg_v1, data_pyg_v2
    )
    # calculate the loss 
    loss = classifier(emb_v1, emb_v2, train_mask, y)
    # update the model
    loss.backward()
    opt.step()
    with torch.cuda.device(train_mask.device):
        torch.cuda.empty_cache()
    return float(loss.detach().item())


def main():
    # Load Configs
    args = argument()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    # Set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Load Data
    data_pyg = load_data(args.dataset).to(device)
    print(f'Dataset: {args.dataset}')
    labels = data_pyg.y
    dim_in = data_pyg.x.shape[1]
    class_num = labels.unique().shape[0]
    pos_mask_mat = get_pos_mask(data_pyg)
    train_mask_pseudo = data_pyg.train_mask.clone()
    train_label_pseudo = init_pseudo_label(labels, data_pyg.train_mask)

    # Build models
    augmenter = Augmenter(n_iter=args.aug_iter).to(device)
    encoder = Encoder(args, dim_in).to(device)
    trainable_params_enc = list(encoder.parameters())
    opt_enc = get_optimizer('adamw', trainable_params_enc, lr=args.lr, 
                            weight_decay=args.wd)
    classifier = Classifier(args.hidden_units[-1], class_num).to(device)
    trainable_params_classifier = list(classifier.parameters())
    opt_cla = get_optimizer('adamw', trainable_params_classifier, lr=args.lr, 
                            weight_decay=args.wd)
    
    # Trian models
    best_epoch = 0
    acc_val_best = 0.
    data_pyg_v1 = data_pyg.clone()
    data_pyg_v2 = data_pyg.clone()
    for epoch in range(args.epochs):
        # data augmentation
        data_x_v1, data_x_v2 = augmenter.augment_feat(data_pyg, encoder, classifier, 
                                                      train_mask_pseudo, train_label_pseudo)
        data_pyg_v1.x = data_x_v1.detach()
        data_pyg_v2.x = data_x_v2.detach()
        # train the encoder
        encoder.train()
        classifier.eval()
        augmenter.eval()
        loss_enc = train_encoder(encoder, opt_enc, data_pyg_v1, data_pyg_v2, pos_mask_mat)
        # train the classifier
        classifier.train()
        augmenter.eval()
        encoder.eval()
        loss_cla = train_classifier(classifier, opt_cla, data_pyg_v1, data_pyg_v2, 
                                    encoder, train_mask_pseudo, train_label_pseudo)
        # Info logging
        print(f'Epoch-{epoch:3d}-train: ' + 
              f'Loss-cla={loss_cla:.4f}   ' + f'Loss-enc={loss_enc:.4f}.')
        # Temp evaluation
        classifier.eval()
        encoder.eval()
        with torch.no_grad():
            emb = encoder.get_embeddings(data_pyg.x, data_pyg.edge_index)
            pred_v1 = classifier.get_pred_v1(emb).detach()
            pred_v2 = classifier.get_pred_v2(emb).detach()
            prob_v1 = F.softmax(pred_v1, dim=-1)
            prob_v2 = F.softmax(pred_v2, dim=-1)
            prob_pesudo = (prob_v1 + prob_v2) / 2
            y_pesudo = torch.argmax(prob_pesudo, dim=-1)
        acc_train, acc_val, acc_test = cal_accs(data_pyg, labels, y_pesudo)
        if (float(acc_val) > acc_val_best):    
            # save the best model
            best_epoch = epoch
            torch.save(encoder.state_dict(), os.path.join('./best_models', f'{args.dataset}_best_model.pt'))
            torch.save(classifier.state_dict(), os.path.join('./best_models', f'{args.dataset}_best_classifier.pt'))

    # Final evaluation
    print(f'Loading the best model from {best_epoch}-th epoch...')
    encoder.load_state_dict(torch.load(os.path.join('./best_models', f'{args.dataset}_best_model.pt')))
    classifier.load_state_dict(torch.load(os.path.join('./best_models', f'{args.dataset}_best_classifier.pt')))
    with torch.no_grad():
        emb = encoder.get_embeddings(data_pyg.x, data_pyg.edge_index)#.detach()
        pred_v1 = classifier.get_pred_v1(emb).detach()
        pred_v2 = classifier.get_pred_v2(emb).detach()
        prob_v1 = F.softmax(pred_v1, dim=-1)
        prob_v2 = F.softmax(pred_v2, dim=-1)
        prob_pesudo = (prob_v1 + prob_v2) / 2
        y_pesudo = torch.argmax(prob_pesudo, dim=-1)
    acc_train, acc_val, acc_test = cal_accs(data_pyg, labels, y_pesudo)
    print(f'Train-ACC: {acc_train:.4f}; Val-ACC: {acc_val:.4f}; Test-ACC: {acc_test:.4f}')

if __name__ == '__main__':
    main()
