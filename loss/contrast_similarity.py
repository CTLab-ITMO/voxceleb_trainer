#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy


class LossFunction(nn.Module):
    def __init__(self, margin=1.0, **kwargs):
        super(LossFunction, self).__init__()
        self.test_normalize = True
        self.nPerSpeaker = None
        self.margin = margin
        print('Initialised Contrast Similarity Loss')

    def forward(self, outp, label=None):
        batch_size, nPerSpeaker, emb_dim = outp.size()

        # 1. Внутригрупповые расстояния
        intra_loss = 0.0
        if nPerSpeaker > 1:
            # Reshape для вычисления всех внутригрупповых расстояний
            all_intra_dists = []
            for i in range(batch_size):
                dists = torch.pdist(outp[i], p=2)  # Только уникальные пары; аудиозаписи внутри каждого спикера: аудио1-аудио2, аудио2-аудио3...
                all_intra_dists.append(dists)

            if all_intra_dists:
                intra_dists = torch.cat(all_intra_dists)
                intra_loss = (intra_dists ** 2).mean() # Штрафуем за наличие расстояния между аудиозаписями внутри спикера

        # 2. Межгрупповые расстояния
        inter_loss = 0.0
        if batch_size > 1:
            speaker_centers = outp.mean(dim=1)
            inter_dists = torch.pdist(speaker_centers, p=2)  # Берем среднее аудио по спикеру и сравниваем между спикерами: спикер1-спикер2...

            if len(inter_dists) > 0: # Штрафуем, если расстояние между разными спикерами меньше порога
                inter_loss = (F.relu(self.margin - inter_dists) ** 2).mean()

        return intra_loss + inter_loss, torch.tensor(0)