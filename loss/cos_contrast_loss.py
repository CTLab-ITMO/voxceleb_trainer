#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy


class LossFunction(nn.Module):
    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()
        self.test_normalize = True
        self.nPerSpeaker = None
        print('Initialised Cosin Contrast Similarity Loss')

    def forward(self, outp, label=None):
        if torch.isnan(outp).any() or torch.isinf(outp).any():
            raise Exception("outp contains NaN or Inf")

        if outp.dim() == 3:
            # Получаем размеры
            batch_size, n_per_spk, features = outp.shape

            # Убираем размерность с индексом 1 (объединяем batch и sequence)
            outp = outp.reshape(-1, features)

            # Создаем labels: для каждого элемента в batch повторяем номер batch
            labels = torch.arange(batch_size).repeat_interleave(n_per_spk)

        else:
            # Для 2D случая - просто создаем последовательные labels
            batch_size = outp.shape[0]
            labels = torch.arange(batch_size)

        norm_emb = nn.functional.normalize(outp, p=2, dim=1)
        cos_similarity = torch.mm(norm_emb, norm_emb.t())

        mask = labels.unsqueeze(1) == labels.unsqueeze(0)

        same_distances = cos_similarity[mask]
        diff_distances = cos_similarity[~mask]

        if diff_distances.numel() == 0:
            diff_mean = torch.tensor(0.0, device=cos_similarity.device)
        else:
            diff_mean = diff_distances.mean()

        if same_distances.numel() == 0:
            same_mean = torch.tensor(1.0, device=cos_similarity.device)
        else:
            same_mean = same_distances.mean()

        contrastive_loss = torch.clamp(diff_mean, min=0) - same_mean + 1.0
        loss = contrastive_loss

        if torch.isnan(loss):
            raise Exception("loss values contains NaN or Inf")
        return loss, torch.tensor(0)
