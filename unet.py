from torch import nn
import torch
from tqdm import tqdm

from time_embedding import TimeEmbedding
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention
from dataset import create_dataloader

class Conv2dDouble(nn.Module):
    def __init__(self, in_chan, out_chan, mid_chan=None):
        super().__init__()
        if not mid_chan:
            mid_chan = out_chan
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(),
            nn.Conv2d(mid_chan, out_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class DownModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        conv_double = Conv2dDouble(in_chan, out_chan)
        self.pool_and_conv = nn.Sequential(nn.MaxPool2d(2), conv_double)
    def forward(self, x):
        return self.pool_and_conv(x)

class UpModule(nn.Module):
    def __init__(self,in_chan, mid_chan, out_chan):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_double = Conv2dDouble(in_chan, out_chan, mid_chan)

    def forward(self, input, residual):
        input = self.upsample(input)

        output = torch.cat([residual,input], dim=1)

        output = self.conv_double(output)

        return output

class UNet(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.inc = Conv2dDouble(n_channels, 64)

        self.down1 = DownModule(64, 128)
        self.down2 = DownModule(128, 256)
        self.down3 = DownModule(256, 512)
        self.down4 = DownModule(512, 512)

        self.up1 = UpModule(1024, 512, 256)
        self.up2 = UpModule(512, 256, 128) # in_chan of curr layer != out_chan of previous layer, since in the upstream layers we *concatenate the input tensor with a residual from the parallel downstream layer*
        self.up3 = UpModule(256, 128, 64)
        self.up4 = UpModule(128, 64, 64)

        self.out = Conv2dDouble(64, n_channels)



    def forward(self, image):
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x

class UNetDiffusion(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet(n_channels=n_channels)

    def forward(self, image):
        output = self.unet(image)
        return output

def train(model, optimizer, loss_function, training_loader, epochs_num, device):
    running_loss = 0.
    last_loss = 0.

    model.to(device)
    model.train()

    for epoch in range(epochs_num):
        print("epoch:", epoch)
        for i, data in tqdm(enumerate(training_loader)):
            optimizer.zero_grad()

            clean_images, _ = data
            clean_images = clean_images.to(device)

            noisy_images = torch.randn(clean_images.shape, dtype=clean_images.dtype, device=device) * 0.5 + 0.5

            denoised_images = model(image=noisy_images)

            loss = loss_function(denoised_images, clean_images)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print("loss:", running_loss/len(training_loader))
        running_loss = 0

def loss_function_mse(model_out, target):
    loss = F.mse_loss(model_out, target, reduction='none')
    return loss.mean()

data_path = "../dit/data/train"
image_size = 32
batch_size = 16
epochs_num = 10
c_latent = 3
device = 'cuda'

model = UNetDiffusion(n_channels=c_latent)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

training_loader = create_dataloader(data_path, image_size, batch_size)

train(model=model, optimizer=optimizer, loss_function=loss_function_mse, training_loader=training_loader, epochs_num=epochs_num, device=device)

print("done")