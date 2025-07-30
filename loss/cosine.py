#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from utils import accuracy

class LossFunction(nn.Module):
    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        self.criterion = nn.MSELoss()

        print('Initialised Cosine Similarity Loss (for disabled adapter)')

    def forward(self, x, label=None):
        
        if x.dim() == 3:
            batch_size, n_per_speaker, embedding_dim = x.size()
            x = x.reshape(-1, embedding_dim)
            if label is not None:
                label = label.repeat_interleave(n_per_speaker)
        
        x_norm = F.normalize(x, p=2, dim=1)
        
        sim_matrix = torch.mm(x_norm, x_norm.t())
        
        batch_size = x.size(0)
        
        if label is not None:
            label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
            target = label_matrix.float()
        else:
            target = torch.eye(batch_size).cuda()
        
        nloss = self.criterion(sim_matrix, target)
        
        # Вычисление accuracy
        with torch.no_grad():
            # Одинаковые спикеры > threshold
            threshold = 0.5
            pred = (sim_matrix > threshold).float()
            
            correct = (pred == target).float()
            
            # Исключаем диагональ (сравнение с самим собой)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)
            prec1 = correct[mask].mean() * 100
        
        return nloss, prec1
