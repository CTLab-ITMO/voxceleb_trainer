#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

class ReDimNetHubAdapter(nn.Module):
	def __init__(self, nOut=192, redim_model_name='b0', redim_train_type='ptn', redim_dataset='vox2', **kwargs):
		super().__init__()
		# Load upstream ReDimNet via torch.hub (pretrained init). Falls back to raising if no internet.
		self.base = torch.hub.load(
			'IDRnD/ReDimNet',
			'ReDimNet',
			model_name=redim_model_name,
			train_type=redim_train_type,
			dataset=redim_dataset
		)
		# Adapt embedding dimensionality to nOut if needed
		base_dim = getattr(self.base, 'linear', None).out_features if hasattr(self.base, 'linear') else nOut
		if base_dim != nOut:
			self.proj = nn.Linear(base_dim, nOut)
		else:
			self.proj = nn.Identity()

	def forward(self, x):
		# x: [B, T] raw waveform (16kHz), consistent with existing pipeline
		emb = self.base(x)
		emb = self.proj(emb)
		return emb

def MainModel(nOut=192, **kwargs):
	# Pass through optional redim_* kwargs; ignore unknown keys gracefully
	return ReDimNetHubAdapter(nOut=nOut, **kwargs)
