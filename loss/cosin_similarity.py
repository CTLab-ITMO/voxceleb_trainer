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
        print('Initialised Cosin Similarity Loss')

    def forward(self, outp, label=None):
        # Гарантируем 3D-тензор
        if outp.dim() == 2:
            outp = outp.unsqueeze(1)  # [batch, 1, emb_dim], если nPerSpeaker=1
        self.nPerSpeaker = outp.size(1)

        # Нормализация эмбеддингов
        outp_norm = F.normalize(outp, p=2, dim=-1)

        # 1. Внутригрупповая схожесть (должна → 1 для всех элементов, т.к. голоса внутри группы принадлежат одному спикеру)
        intra_sim = torch.bmm(outp_norm, outp_norm.transpose(1, 2))
        intra_target = torch.ones_like(intra_sim).cuda()
        intra_loss = F.mse_loss(intra_sim, intra_target)

        # 2. Межгрупповая схожесть (должна → 0, т.к. эмбеддинги из разных спикеров, 1 только по диагонали, т.к. по ней сравнение элемента с самим собой)
        inter_sim = torch.mm(
            outp_norm.mean(dim=1),  # Усредненные по спикеру (по группе голосов) эмбеддинги [batch_size, emb_dim]
            outp_norm.mean(dim=1).t()
        )
        inter_target = torch.eye(inter_sim.size(0)).cuda()
        inter_loss = F.mse_loss(inter_sim, inter_target)

        # 3. Расчет accuracy
        with torch.no_grad():
            # Для внутригрупповой accuracy
            intra_pred = (intra_sim > 0.5).float() # Если больше 0.5, считаем, что элементы признаны равными
            intra_correct = (intra_pred == intra_target).float()
            intra_acc = intra_correct.mean() * 100

            # Для межгрупповой accuracy
            inter_pred = (inter_sim > 0.5).float()
            inter_correct = (inter_pred == inter_target).float()
            inter_acc = inter_correct.mean() * 100

            '''
            Коэффициенты, учитывающие долю внутренней и внешней точности
            Позже будут введены, пока что будем учитывать только межгрупповую accuracy
            '''
            intra_coef = 0
            inter_coef = 1

            # Общая accuracy (взвешенное среднее)
            total_acc = (intra_acc * intra_coef + inter_acc * inter_coef) / \
                        (intra_coef + inter_coef)

        # Комбинированный лосс
        total_loss = intra_loss + inter_loss

        return total_loss, total_acc