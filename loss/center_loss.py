#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from utils import accuracy
import os

class CenterLoss(nn.Module):
	def __init__(self, num_classes, feat_dim, alpha=1):
		super(CenterLoss, self).__init__()
		self.num_classes = num_classes
		self.feat_dim = feat_dim
		self.alpha = alpha # Скорость обновления центров классов

		# Центры классов - обучаемые параметры
		self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

	def forward(self, x, labels):
		"""
        x: эмбеддинги [batch_size, feat_dim]
        labels: метки классов [batch_size]
        """
		batch_size = x.size(0)

		# Получаем центры классов для каждого примера в батче
		centers_batch = self.centers[labels]  # [batch_size, feat_dim]

		# Вычисляем center loss (L2 расстояние)
		center_loss = 0.5 * torch.sum(torch.pow(x - centers_batch, 2))

		return center_loss

	def update_centers(self, x, labels):
		"""
        Обновляем центры вручную
        """
		#print("МЕТКИ БАТЧА")
		#print(labels)
		#print("ЦЕНТР 0ГО СПИКЕРА")
		#print(self.centers[0])
		with torch.no_grad():
			for j in range(self.num_classes):
				# Маска для текущего класса
				mask = (labels == j)  # Находим все примеры класса j в батче
				#print(f"МАСКА ДЛЯ {j}: столько примеров этого класса в батче")
				#print(mask)

				if mask.sum() > 0:
					# Фичи текущего класса
					class_features = x[mask]

					# Δc_j = Σ(c_j - x_i) / (1 + count)
					delta = (self.centers[j] - class_features).sum(dim=0)
					delta = delta / (1 + mask.sum())

					# c_j = c_j - α * Δc_j
					self.centers[j] -= self.alpha * delta


class LossFunction(nn.Module):
	def __init__(self, nOut, nClasses, lambda_center=0.003, **kwargs):
		super(LossFunction, self).__init__()
		self.test_normalize = True
		self.criterion = torch.nn.CrossEntropyLoss()
		self.fc = nn.Linear(nOut, nClasses)

		# Center Loss параметры
		self.lambda_center = lambda_center
		self.center_loss = CenterLoss(nClasses, nOut)

		print('Initialised Softmax + Center Loss')

	def forward(self, x, label=None):
		# x - эмбеддинги

		# Softmax branch
		logits = self.fc(x)
		softmax_loss = self.criterion(logits, label)

		# Center Loss branch
		center_loss = self.center_loss(x, label)

		# Total loss
		total_loss = softmax_loss + self.lambda_center * center_loss

		prec1 = accuracy(logits.detach(), label.detach(), topk=(1,))[0]

		return [total_loss, softmax_loss], prec1