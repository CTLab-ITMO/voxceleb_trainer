#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import accuracy


class LossFunction(nn.Module):
    def __init__(
        self,
        nOut,
        nClasses,
        range_alpha=1e-5,
        range_beta=1e-4,
        range_lambda=1.0,
        range_margin=0.5,
        range_k=2,
        **kwargs,
    ):
        super().__init__()

        self.test_normalize = True

        self.ce_criterion = nn.CrossEntropyLoss()
        self.fc = nn.Linear(nOut, nClasses)

        self.range_alpha = range_alpha
        self.range_beta = range_beta
        self.range_lambda = range_lambda
        self.range_margin = range_margin
        self.range_k = max(range_k, 1)

        print(
            "Initialised Range Loss (Softmax + intra/inter regularisation) "
            f"[alpha={self.range_alpha}, beta={self.range_beta}, "
            f"lambda={self.range_lambda}, margin={self.range_margin}, k={self.range_k}]"
        )

    def forward(self, x, label=None):
        if label is None:
            raise ValueError("Range loss requires ground-truth labels during training.")

        if x.dim() == 3:
            batch_size, n_per_speaker, emb_dim = x.size()
            x_flat = x.reshape(batch_size * n_per_speaker, emb_dim)
            label_flat = label.unsqueeze(1).repeat(1, n_per_speaker).reshape(-1)
        else:
            x_flat = x
            label_flat = label

        logits = self.fc(x_flat)
        ce_loss = self.ce_criterion(logits, label_flat)

        range_loss = torch.zeros(1, device=x_flat.device, dtype=x_flat.dtype)
        if self.range_lambda != 0.0 and (self.range_alpha != 0.0 or self.range_beta != 0.0):
            range_loss = self._compute_range_terms(x_flat, label_flat)

        total_loss = ce_loss + self.range_lambda * range_loss
        prec1 = accuracy(logits.detach(), label_flat.detach(), topk=(1,))[0]

        return total_loss, prec1

    def _compute_range_terms(self, embeddings, labels):
        unique_labels = torch.unique(labels)

        intra_terms = []
        centers = []

        for lbl in unique_labels:
            mask = labels == lbl
            feats = embeddings[mask]
            if feats.numel() == 0:
                continue

            centers.append(feats.mean(dim=0))

            if feats.size(0) < 2 or self.range_alpha == 0.0:
                continue

            pairwise = torch.pdist(feats, p=2)
            if pairwise.numel() == 0:
                continue

            k = min(self.range_k, pairwise.numel())
            topk = torch.topk(pairwise, k, largest=True).values

            inv = torch.reciprocal(torch.clamp(topk, min=1e-6))
            harmonic_mean = k / inv.sum()
            intra_terms.append(harmonic_mean)

        intra_loss = embeddings.new_tensor(0.0)
        if intra_terms and self.range_alpha != 0.0:
            intra_loss = torch.stack(intra_terms).mean()
            intra_loss = self.range_alpha * intra_loss

        inter_loss = embeddings.new_tensor(0.0)
        if len(centers) > 1 and self.range_beta != 0.0:
            center_stack = torch.stack(centers)
            distances = torch.cdist(center_stack, center_stack, p=2) ** 2
            mask = torch.eye(distances.size(0), device=distances.device, dtype=torch.bool)
            distances = distances.masked_fill(mask, float("inf"))

            min_dist = torch.min(distances)
            if torch.isfinite(min_dist):
                inter_loss = F.relu(self.range_margin - min_dist)
                inter_loss = self.range_beta * inter_loss

        return intra_loss + inter_loss

