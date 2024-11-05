import math
from torch import nn
import torch

class Conv2dDouble(nn.Module):
    def __init__(self, in_chan, out_chan, mid_chan=None):
        super().__init__()
        if not mid_chan:
            mid_chan = out_chan

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_chan),
            nn.ReLU(),
            nn.Conv2d(in_chan, mid_chan, kernel_size=3, padding=1, bias=False))

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(),
            nn.Conv2d(mid_chan, out_chan, kernel_size=3, padding=1, bias=False))

        self.in_chan = in_chan
        self.mid_chan = mid_chan
        self.out_chan = out_chan

    def forward(self, x):
        x = self.conv1(x)
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

    def forward(self, x):
        x = self.maxpool(x)
        skip = self.skip(x)
        x = self.conv_double(x)
        x = x + skip
        if self.apply_attn:
            x = self.attn(x)
        return x

class UpModule(nn.Module):
    def __init__(self,in_chan, out_chan, apply_attn):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_double = Conv2dDouble(in_chan, out_chan)
        self.skip = nn.Conv2d(in_chan, out_chan, 1)
        self.attn = AttentionModule(out_chan)
        self.apply_attn = apply_attn

    def forward(self, x):
        x = self.upsample(x)
        skip = self.skip(x)
        x = self.conv_double(x)
        x = x + skip
        if self.apply_attn:
            x = self.attn(x)
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


class Autoencoder(nn.Module):
    def __init__(self, apply_attn, n_channels, device):
        super().__init__()

        self.inc = Conv2dDouble(n_channels, 64)

        self.down1 = DownModule(64, 128, apply_attn[0])
        self.down2 = DownModule(128, 256, apply_attn[1])
        self.down3 = DownModule(256, 512, apply_attn[2])
        self.down4 = DownModule(512, 1024, apply_attn[3])

        self.up1 = UpModule(1024, 1024, apply_attn[0])
        self.up2 = UpModule(1024, 512, apply_attn[1])
        self.up3 = UpModule(512, 128, apply_attn[2])
        self.up4 = UpModule(128, 64, apply_attn[3])

        self.out = OutLayer(64, n_channels)

        self.device = device

    def encode(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x

    def decode(self, z):
        z = self.up1(z)
        z = self.up2(z)
        z = self.up3(z)
        z = self.up4(z)
        z = self.out(z)
        return z

device = 'cuda'
torch.set_default_device(device)

apply_attn = [False, False, True, False]
n_channels = 1
image_size = 64

ae_model = Autoencoder(apply_attn=apply_attn, n_channels=n_channels, device=device)

x = torch.randn(torch.Size([1, n_channels, image_size, image_size]), dtype=torch.float32, device=device)
z = ae_model.encode(x)
x_rec = ae_model.decode(z)

# print("z:",z.shape)
# print("x_rec:", x_rec.shape)

# print(ae_model)