import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunction(nn.Module):
    def __init__(self, margin=1.0, **kwargs):
        super(LossFunction, self).__init__()
        
        self.margin = float(kwargs.get('margin', margin))
        self.test_normalize = True
        print(f'Initialised Contrastive Loss (margin={self.margin})')

    def forward(self, x, label=None):
        # x shapes: (S, K, D)  S=batchsize, K=nPerSpeaker, D=emb dim
        assert label is not None, "Contrastive loss requires labels"

        if x.dim() == 3:
            S, K, D = x.size()
            if self.test_normalize:
                x = F.normalize(x, p=2, dim=2)
            X = x.reshape(S * K, D)                        
            labels_flat = label.view(-1, 1).repeat(1, K).view(-1) 
        elif x.dim() == 2:
            S, D = x.size()
            K = 1
            if self.test_normalize:
                x = F.normalize(x, p=2, dim=1)
            X = x                                      
            labels_flat = label
        else:
            raise ValueError("Unexpected embedding tensor shape for contrastive loss")

        N = X.size(0)
        if N < 2:
            zero = X.new_zeros(())
            return zero, zero

        Dmat = torch.cdist(X, X, p=2)

        tri_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=X.device), diagonal=1)

        same = (labels_flat.unsqueeze(0) == labels_flat.unsqueeze(1)) & tri_mask
        diff = (~(labels_flat.unsqueeze(0) == labels_flat.unsqueeze(1))) & tri_mask

        pos_loss = (Dmat[same] ** 2).sum()
        neg_part = torch.clamp(self.margin - Dmat[diff], min=0.0)
        neg_loss = (neg_part ** 2).sum()

        num_pairs = tri_mask.sum().clamp(min=1).float()
        loss = (pos_loss + neg_loss) / (2.0 * num_pairs)

        with torch.no_grad():
            thr = self.margin / 2.0
            preds_same = (Dmat < thr) & tri_mask
            correct = ((preds_same & same) | ((~preds_same) & diff)).sum().float()
            acc = (correct / num_pairs) * 100.0

        return loss, acc
