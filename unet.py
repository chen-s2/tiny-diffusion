import math
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from time_embed import Time, TimeLinearEmbedder, REF_get_timestep_embedding
from torch.nn import functional as F
import matplotlib.pyplot as plt
from attention import SelfAttention, CrossAttention
from utils import *

time_emb_dim = None

class Conv2dDouble(nn.Module):
    def __init__(self, in_chan, out_chan, mid_chan=None):
        super(Conv2dDouble, self).__init__()
        if not mid_chan:
            mid_chan = out_chan

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_chan),
            nn.ReLU(),
            nn.Conv2d(in_chan, mid_chan, kernel_size=3, padding=1, bias=False))

        self.time_fc = Time(time_emb_dim, mid_chan)

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(),
            nn.Conv2d(mid_chan, out_chan, kernel_size=3, padding=1, bias=False))

        self.in_chan = in_chan
        self.mid_chan = mid_chan
        self.out_chan = out_chan
        # self.drop = nn.Dropout(p=0.1, inplace=True)

    def forward(self, x, t_emb):
        x = self.conv1(x)

        t_emb_lin = self.time_fc(t_emb)
        t_emb_lin = t_emb_lin.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1, -1)
        t_emb_lin = F.interpolate(t_emb_lin, size=(x.shape[2:4]), mode='bilinear', align_corners=False)
        x = x + t_emb_lin
        # x = self.drop(x)

        x = self.conv2(x)
        return x

class DownModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv_double = Conv2dDouble(in_chan, out_chan)
        self.skip = nn.Identity() if in_chan == out_chan else nn.Conv2d(in_chan, out_chan, 1)

    def forward(self, x, t_emb):
        x = self.maxpool(x)
        skip = self.skip(x)
        x = self.conv_double(x, t_emb)
        return x + skip

class UpModule(nn.Module):
    def __init__(self,in_chan, mid_chan, out_chan):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_double = Conv2dDouble(in_chan, out_chan, mid_chan)
        self.skip = nn.Identity() if mid_chan == out_chan else nn.Conv2d(mid_chan, out_chan, 1)

    def forward(self, x, horziontal_residual, t_emb):
        # horziontal_residual is passed from a parallel downsampling layer to this upsampling layer
        x = self.upsample(x)

        skip = self.skip(x)

        x = torch.cat([horziontal_residual,x], dim=1)

        x = self.conv_double(x, t_emb)

        return x + skip
        # return x

class MiddleModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv_double_1 = Conv2dDouble(in_chan, out_chan)
        self.conv_double_2 = Conv2dDouble(in_chan, out_chan)
        self.skip = nn.Identity() if in_chan == out_chan else nn.Conv2d(in_chan, out_chan, 1)

    def forward(self, x, t_emb):
        skip = self.skip(x)
        x = self.conv_double_1(x, t_emb)
        x = self.conv_double_2(x, t_emb)
        return x + skip

class UNet(nn.Module):
    def __init__(self, n_channels, time_emb_dim_param, device):
        super().__init__()
        global time_emb_dim
        time_emb_dim = time_emb_dim_param
        self.inc = Conv2dDouble(n_channels, 64)

        self.down1 = DownModule(64, 128)
        self.down2 = DownModule(128, 256)
        self.down3 = DownModule(256, 512)
        self.down4 = DownModule(512, 512)

        self.middle = MiddleModule(512, 512)

        self.up1 = UpModule(1024, 512, 256)
        self.up2 = UpModule(512, 256, 128) # in_chan of curr layer != out_chan of previous layer, since in the upstream layers we *concatenate the input tensor with a residual from the parallel downstream layer*
        self.up3 = UpModule(256, 128, 64)
        self.up4 = UpModule(128, 64, 64)

        self.out = Conv2dDouble(64, n_channels)

        self.time_linear_embedder = TimeLinearEmbedder(1, time_emb_dim)
        self.time_emb_dim = time_emb_dim

        self.device = device
        total_params = sum(p.numel() for p in self.parameters())
        print("model's total params:", total_params)

    def forward(self, image, t):
        t_emb = REF_get_timestep_embedding(timesteps=[t], embed_dim=self.time_emb_dim, dtype=torch.float32, device=self.device)
        t_emb = self.time_linear_embedder(t_emb.T)

        x1 = self.inc(image, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        x5 = self.down4(x4, t_emb)

        xm = self.middle(x5, t_emb)

        x = self.up1(xm, x4, t_emb)
        x = self.up2(x, x3, t_emb)
        x = self.up3(x, x2, t_emb)
        x = self.up4(x, x1, t_emb)
        x = self.out(x, t_emb)
        return x
