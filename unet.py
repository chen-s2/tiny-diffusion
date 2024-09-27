from torch import nn
import torch
from time_embedding import TimeEmbedding
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention
from dataset import create_dataloader

class Conv2dDouble(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1, bias=False),
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
    def __init__(self,in_chan, out_chan):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_double = Conv2dDouble(in_chan, out_chan)

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

        self.up1 = UpModule(1024, 256)
        self.up2 = UpModule(512, 128)
        self.up3 = UpModule(256, 64)
        self.up4 = UpModule(128, 64)

    def forward(self, image_latent, text_context, time_embed):
        '''
        :param image_latent: (B,4,h/8,w/8)
        :param text_context: (B,seq_len,embed_dim)
        :param time_embed: (1,1280)
        '''
        x1 = self.inc(image_latent)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

class UNetDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()

    def forward(self, image_latent, text_context, time_embed):
        '''
        image_latent: (B, 4, H/8, W/8), the latent representation of the image, after the VAE-encoder, plus added noise
        text_context: (B, seq_len, embed_len), the prompt after passing CLIP encoding, seq_len is the max len of clip's output
        time_embed: (1, 320), the current timestep's in some (?) space
        '''

        # (1,320) -> (1,1280)
        time_embed = self.time_embedding(time_embed)

        # (B, 4, H/8, W/8) -> (B, 320, H/8, W/8)
        output = self.unet(image_latent, text_context, time_embed)

        return output

def train(model, optimizer, loss_function, training_loader, epochs_num, device):
    running_loss = 0.
    last_loss = 0.

    model.to(device)
    model.train()

    for epoch in range(epochs_num):
        for i, data in enumerate(training_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            '''
            image_latent: (B, 4, H/8, W/8), the latent representation of the image, after the VAE-encoder, plus added noise
            text_context: (B, seq_len, embed_len), the prompt after passing CLIP encoding, seq_len is the max len of clip's output
            time_embed: (1, 320), the current timestep's in some (?) space
            '''

            seq_len, embed_len = 80, 512
            text_context = torch.ones((batch_size, seq_len, embed_len)).to(device)
            time_embed = torch.ones((1,320)).to(device)

            outputs = model(image_latent=inputs, text_context=text_context, time_embed=time_embed)

            loss = loss_function(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.

            return None

    return last_loss

def loss_function_mse(model_out, target):
    loss = F.mse_loss(model_out, target, reduction='none')
    return loss.mean()

data_path = "../dit/data/tiny"
image_size = 256
batch_size = 16
epochs_num = 1
c_latent = 3
device = 'cuda'

model = UNetDiffusion()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

training_loader = create_dataloader(data_path, image_size, batch_size)

train(model=model, optimizer=optimizer, loss_function=loss_function_mse, training_loader=training_loader, epochs_num=epochs_num, device=device)

print("done")