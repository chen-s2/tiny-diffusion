import math
from torch import nn
from time_embed import Time, TimeLinearEmbedder, REF_get_timestep_embedding
from torch.nn import functional as F
from utils import *

time_emb_dim = None

class Conv2dDouble(nn.Module):
    def __init__(self, in_chan, out_chan, mid_chan=None):
        super().__init__()
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

    def forward(self, x, t_emb):
        x = self.conv1(x)

        t_emb_lin = self.time_fc(t_emb)
        t_emb_lin = t_emb_lin.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1, -1)
        t_emb_lin = F.interpolate(t_emb_lin, size=(x.shape[2:4]), mode='bilinear', align_corners=False)
        x = x + t_emb_lin

        x = self.conv2(x)
        return x

class OutLayer(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_chan),
            nn.ReLU(),
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1, bias=False))

        self.in_chan = in_chan
        self.out_chan = out_chan
    def forward(self, x):
        x = self.conv(x)
        return x

class DownModule(nn.Module):
    def __init__(self, in_chan, out_chan, apply_attn):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv_double = Conv2dDouble(in_chan, out_chan)
        self.skip = nn.Identity() if in_chan == out_chan else nn.Conv2d(in_chan, out_chan, 1)
        self.attn = AttentionModule(out_chan)
        self.apply_attn = apply_attn

    def forward(self, x, t_emb):
        x = self.maxpool(x)

        skip = self.skip(x)

        x = self.conv_double(x, t_emb)

        x = x + skip

        if self.apply_attn:
            x = self.attn(x)

        return x

class UpModule(nn.Module):
    def __init__(self,in_chan, mid_chan, out_chan, apply_attn):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_double = Conv2dDouble(in_chan, out_chan, mid_chan)
        self.skip = nn.Identity() if mid_chan == out_chan else nn.Conv2d(mid_chan, out_chan, 1)
        self.attn = AttentionModule(out_chan)
        self.apply_attn = apply_attn

    def forward(self, x, horziontal_residual, t_emb):
        x = self.upsample(x)

        skip = self.skip(x)

        x = torch.cat([horziontal_residual,x], dim=1) # horziontal_residual is passed from a parallel downsampling layer to this upsampling layer

        x = self.conv_double(x, t_emb)

        x = x + skip

        if self.apply_attn:
            x = self.attn(x)

        return x

class MiddleModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv_double_1 = Conv2dDouble(in_chan, out_chan)
        self.attn = AttentionModule(out_chan)  # in_chan = out_chan in the middle-module anyhow
        self.conv_double_2 = Conv2dDouble(in_chan, out_chan)
        self.skip = nn.Identity() if in_chan == out_chan else nn.Conv2d(in_chan, out_chan, 1)

    def forward(self, x, t_emb):
        skip = self.skip(x)

        x = self.conv_double_1(x, t_emb)

        x = self.attn(x)

        x = self.conv_double_2(x, t_emb)

        x = x + skip

        return x

class AttentionModule(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        self.in_chan = in_chan

        self.norm = nn.BatchNorm2d(in_chan)

        self.proj_in = nn.Conv2d(in_chan, 3 * in_chan, 1)

        self.proj_out = nn.Conv2d(in_chan, in_chan, 1)

        self.skip = nn.Identity()

    @staticmethod
    def qkv(q, k, v):
        B, C, H, W = q.shape
        w = torch.einsum("bchw, bcHW -> bhwHW", q, k)
        w = torch.softmax(w.reshape(B,H,W,H*W) / math.sqrt(C), dim=-1).reshape(B,H,W,H,W)
        out = torch.einsum("bhwHW, bcHW -> bchw", w, v)
        return out.contiguous()

    def forward(self, x, **kwargs):
        skip = self.skip(x)

        C = x.shape[1]
        assert C == self.in_chan

        x = self.proj_in(x)

        q, k, v = x.chunk(3,dim=1)

        x = self.qkv(q,k,v)

        x = self.proj_out(x)

        x = x + skip

        return x

class UNet(nn.Module):
    def __init__(self, n_channels, time_emb_dim_param, device, apply_attn):
        super().__init__()
        global time_emb_dim
        time_emb_dim = time_emb_dim_param
        self.apply_attn = apply_attn
        self.inc = Conv2dDouble(n_channels, 64)

        self.down1 = DownModule(64, 128, apply_attn[0])
        self.down2 = DownModule(128, 256, apply_attn[1])
        self.down3 = DownModule(256, 512, apply_attn[2])
        self.down4 = DownModule(512, 512, apply_attn[3])

        self.middle = MiddleModule(512, 512)

        self.up1 = UpModule(1024, 512, 256, apply_attn[0])
        self.up2 = UpModule(512, 256, 128, apply_attn[1]) # in_chan of current layer != out_chan of previous layer, since in the upstream layers we *concatenate the input tensor with a residual from the parallel downstream layer*
        self.up3 = UpModule(256, 128, 64, apply_attn[2])
        self.up4 = UpModule(128, 64, 64, apply_attn[3])

        self.out = OutLayer(64, n_channels)

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
        x = self.out(x)

        return x
