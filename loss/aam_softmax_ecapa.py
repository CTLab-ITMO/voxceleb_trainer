
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy


class LossFunction(nn.Module):
    def __init__(self, nClasses, margin, scale, **kwargs):
        super(LossFunction, self).__init__()
        self.m = 0.2
        self.s = 30
        n_class = 5994 # Параметры в ECAPA
        self.test_normalize = True
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output
