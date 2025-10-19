import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from typing import Union, List

from tdnn_models.utils import LayerNorm

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAP_BN_FC_BN(nn.Module):
    def __init__(self, channel_size, intermediate_size, embeding_size):
        super(VAP_BN_FC_BN, self).__init__()
        self.channel_size = channel_size
        self.intermediate_size = intermediate_size
        self.embeding_size = embeding_size

        self.conv_1 = nn.Conv1d(self.channel_size, self.intermediate_size, kernel_size=1)
        self.conv_2 = nn.Conv1d(self.intermediate_size, self.channel_size, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(self.intermediate_size)
        self.tanh = nn.Tanh()
        self.bn2 = nn.BatchNorm1d(self.channel_size * 2)

        self.fc = nn.Linear(self.channel_size * 2, self.embeding_size)
        self.bn3 = nn.BatchNorm1d(self.embeding_size)

    def forward(self, x):
        """
        Args:
            x: (batch_size, channel_size, T)
        Returns:
            x: (batch_size, embeding_size)
        """
        assert x.dim() == 3, "x.dim() must be 3"

        attn = self.conv_2(self.tanh(self.bn1(self.conv_1(x))))
        # self.conv_1(x).shape : (batch_size, intermediate_size, T)
        # self.conv_2(self.tapn(self.conv_1(x))).shape : (batch_size, channel_size, T)
        attn = F.softmax(attn, dim=2)  # (batch_size, channel_size, T)

        mu = torch.sum(x * attn, dim=2)  # (batch_size, channel_size)
        rh = torch.sqrt((torch.sum((x ** 2) * attn, dim=2) - mu ** 2).clamp(min=1e-5))

        x = torch.cat((mu, rh), dim=1)  # (batch_size, channel_size*2)
        x = self.bn2(x)

        x = self.fc(x)  # (batch_size, embeding_size)
        x = self.bn3(x)

        return x


def Aggregation(channel_size=3 * 128, intermediate_size=int(3 * 128 / 8), embeding_size=192):
    return VAP_BN_FC_BN(channel_size, intermediate_size, embeding_size)


class NeXtTDNN(nn.Module):  #
    """ NeXt-TDNN / NeXt-TDNN-Light model.

    Args:
        in_chans (int): Number of input channels. Default: 80
        depths (tuple(int)): Number of blocks at each stage. Default: [1, 1, 1]
        dims (int): Feature dimension at each stage. Default: [256, 256, 256]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
    """

    def __init__(self, in_chans=80,  # in_channels: 3 -> 80
                 depths=[1, 1, 1], dims=[128, 128, 128],
                 drop_path_rate=0.,  #
                 kernel_size: Union[int, List[int]] = 65,
                 block="TSConvNeXt_light",  # TSConvNeXt_light or TSConvNeXt
                 ):
        super().__init__()
        self.depths = depths
        self.stem = nn.ModuleList()
        Conv1DLayer = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")  # ⚡
        )
        self.stem.append(Conv1DLayer)
        self.aggregate = Aggregation()

        block = importlib.import_module(f"tdnn_models.{block}").__getattribute__(block)  # TSConvNeXt_light or TSConvNeXt

        self.stages = nn.ModuleList()  # 3 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(self.depths)):  # ⚡
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j], kernel_size=kernel_size) for j in range(depths[i])]
                # ⚡
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

        # MFA layer
        self.MFA = nn.Sequential(  # ⚡
            nn.Conv1d(3 * dims[-1], int(3 * dims[-1]), kernel_size=1),
            LayerNorm(int(3 * dims[-1]), eps=1e-6, data_format="channels_first")  # ⚡
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):  # ⚡
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        print(f"Stem input shape: {x.shape}")  # Должно быть (N, C, T)
        x = self.stem[0](x)
        print(f"After stem shape: {x.shape}")

        mfa_in = []
        for i in range(len(self.depths)):  # ⚡ 4 -> 3(len(self.depths))
            x = self.stages[i](x)
            mfa_in.append(x)

        return mfa_in  # ⚡

    def forward(self, x):
        # Если вход 2D, reshape в 3D
        '''
        if x.dim() == 2:
            # Предполагаем: x.shape = (batch_size, 80 * time_steps)
            batch_size = x.size(0)
            x = x.view(batch_size, 80, -1)  # -> (batch_size, 80, time_steps)
            print(f"Reshaped to: {x.shape}")
        '''
        x = self.forward_features(x)  # ⚡

        # MFA layer
        x = torch.cat(x, dim=1)
        x = self.MFA(x)  # Conv1d + LayerNorm_TDNN

        x = self.aggregate.forward(x)

        return x


def MainModel(**kwargs):
    model = NeXtTDNN()
    return model