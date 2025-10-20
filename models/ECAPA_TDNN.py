#! /usr/bin/python
# -*- encoding: utf-8 -*-

import math
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        scale = self.se(inputs)
        return inputs * scale


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, dilation=1, scale=8):
        super().__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)

        self.nums = scale - 1
        convs = []
        bns = []
        pad = math.floor(kernel_size / 2) * dilation
        for _ in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, inputs):
        residual = inputs
        out = self.conv1(inputs)
        out = self.relu(out)
        out = self.bn1(out)

        splits = torch.split(out, self.width, dim=1)
        agg = None
        outputs = []
        for idx in range(self.nums):
            part = splits[idx] if agg is None else agg + splits[idx]
            part = self.convs[idx](part)
            part = self.relu(part)
            part = self.bns[idx](part)
            agg = part
            outputs.append(part)
        outputs.append(splits[self.nums])
        out = torch.cat(outputs, dim=1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class PreEmphasis(nn.Module):
    def __init__(self, coef=0.97):
        super().__init__()
        self.coef = coef
        filt = torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0)
        self.register_buffer("filter", filt)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1)
        inputs = F.pad(inputs, (1, 0), mode="reflect")
        return F.conv1d(inputs, self.filter).squeeze(1)


class FbankAug(nn.Module):
    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        super().__init__()
        self.freq_mask_width = freq_mask_width
        self.time_mask_width = time_mask_width

    @staticmethod
    def _mask_along_axis(x, width_range, dim):
        original_size = x.shape

        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() > 3:
            x = x.reshape(-1, original_size[-2], original_size[-1])

        batch, feat, time = x.shape
        if batch == 0 or feat == 0 or time == 0:
            return x.view(*original_size)

        low, high = width_range
        if high <= low:
            high = low + 1

        if dim == 1:
            D = feat
        else:
            D = time

        high = min(high, D)
        if high <= 0:
            return x.view(*original_size)

        mask_len = torch.randint(low, max(high, 1), (batch, 1), device=x.device).clamp_(min=0, max=D)
        mask_len = mask_len.unsqueeze(2)

        max_start = max(1, D - int(mask_len.max().item()))
        mask_pos = torch.randint(0, max_start, (batch, 1), device=x.device).unsqueeze(2)

        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) & (arange < (mask_pos + mask_len))

        if dim == 1:
            mask = mask.transpose(1, 2)

        x = x.masked_fill(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self._mask_along_axis(x, self.time_mask_width, dim=2)
        x = self._mask_along_axis(x, self.freq_mask_width, dim=1)
        return x


class ECAPATDNN(nn.Module):
    def __init__(
        self,
        channels=512,
        nOut=192,
        n_mels=80,
        apply_spec_aug=True,
        preemphasis_coef=0.97,
        **kwargs,
    ):
        super().__init__()

        self.apply_spec_aug = apply_spec_aug

        self.torchfbank = nn.Sequential(
            PreEmphasis(preemphasis_coef),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=512,
                win_length=400,
                hop_length=160,
                f_min=20,
                f_max=7600,
                window_fn=torch.hamming_window,
                n_mels=n_mels,
            ),
        )

        self.specaug = FbankAug() if apply_spec_aug else None

        self.conv1 = nn.Conv1d(n_mels, channels, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(channels)

        self.layer1 = Bottle2neck(channels, channels, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(channels, channels, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(channels, channels, kernel_size=3, dilation=4, scale=8)

        self.layer4 = nn.Conv1d(channels * 3, 1536, kernel_size=1)

        self.attention = nn.Sequential(
            nn.Conv1d(1536 * 3, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )

        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, nOut)
        self.bn6 = nn.BatchNorm1d(nOut)

    def forward(self, x):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                feats = self.torchfbank(x) + 1e-6
                feats = feats.log()
                feats = feats - feats.mean(dim=-1, keepdim=True)
        if self.training and self.apply_spec_aug and self.specaug is not None:
            feats = self.specaug(feats)

        x = self.conv1(feats)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.layer4(x)
        x = self.relu(x)

        t = x.size(-1)
        global_x = torch.cat(
            [
                x,
                x.mean(dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(x.var(dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t),
            ],
            dim=1,
        )

        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))

        x = torch.cat([mu, sg], dim=1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x


def MainModel(nOut=192, channels=512, **kwargs):
    return ECAPATDNN(channels=channels, nOut=nOut, **kwargs)
