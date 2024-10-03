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

time_emb_dim = None

class Conv2dDouble(nn.Module):
    def __init__(self, in_chan, out_chan, mid_chan=None, ):
        super().__init__()
        if not mid_chan:
            mid_chan = out_chan
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU())
        self.time_fc = Time(time_emb_dim, mid_chan)
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU())
        self.in_chan = in_chan
        self.mid_chan = mid_chan
        self.out_chan = out_chan

    def forward(self, x, t_emb):
        x = self.conv1(x)
        t_emb_lin = self.time_fc(t_emb)
        t_emb_lin = t_emb_lin.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1, -1)
        t_emb_lin = F.interpolate(t_emb_lin, size=(x.shape[2:4]), mode='bilinear', align_corners=False)
        x = self.conv2(x + t_emb_lin)
        return x

class DownModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv_double = Conv2dDouble(in_chan, out_chan)

    def forward(self, x, t_emb):
        y = self.maxpool(x)
        y = self.conv_double(y, t_emb)
        return y

class UpModule(nn.Module):
    def __init__(self,in_chan, mid_chan, out_chan):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_double = Conv2dDouble(in_chan, out_chan, mid_chan)

    def forward(self, input, residual, t_emb):
        input = self.upsample(input)
        output = torch.cat([residual,input], dim=1)
        output = self.conv_double(output, t_emb)
        return output

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
        x = self.up1(x5, x4, t_emb)
        x = self.up2(x, x3, t_emb)
        x = self.up3(x, x2, t_emb)
        x = self.up4(x, x1, t_emb)
        x = self.out(x, t_emb)
        return x

def show_diffusion_chain(clean_image, epsilon, T):
    num_noise_levels = 20
    beta = np.linspace(1e-4, 0.02, num=T)
    timesteps = np.linspace(T-1, 1, num=num_noise_levels).astype('int')
    device = clean_image.device
    print("image shape:", clean_image.shape, ", epsilon shape:", epsilon.shape)

    fig, axes = plt.subplots(2, num_noise_levels//2, figsize=(16, 3))

    for noise_index, t in enumerate(timesteps):
        alpha_1_to_t_array = []
        for i in range(1, t + 1):
            alpha_t = 1 - beta[i]
            alpha_1_to_t_array.append(alpha_t)
        alpha_t_bar = torch.prod(torch.Tensor(alpha_1_to_t_array))

        noisy_image = torch.sqrt(alpha_t_bar) * clean_image + torch.sqrt(1 - alpha_t_bar) * epsilon

        noisy_image_transpose = np.transpose(dt(noisy_image), (1, 2, 0))
        noisy_image_transpose = torch.Tensor(noisy_image_transpose).to(device)

        axes[math.floor(noise_index/10.0), noise_index%10].imshow(dt(noisy_image_transpose), cmap='gray')
        axes[math.floor(noise_index/10.0), noise_index%10].axis('off')

    plt.tight_layout()
    plt.show()

def train(model, optimizer, loss_function, training_loader, epochs_num, device, T):
    running_loss = 0.
    last_loss = 0.
    beta = np.linspace(1e-4, 0.02, num=T)

    model.to(device)
    model.train()

    dataset_name = training_loader.dataset_name

    for epoch in range(epochs_num):
        print("epoch:", epoch)
        for i, data in tqdm(enumerate(training_loader)):
            optimizer.zero_grad()
            x0, _ = data

            x0 = x0.to(device)
            epsilon = torch.randn(x0.shape, dtype=x0.dtype, device=device)
            t = int(torch.randint(1, T, (1,), dtype=x0.dtype, device=device))

            alpha_1_to_t_array = []
            for i in range(1,t+1):
                alpha_t = 1-beta[i]
                alpha_1_to_t_array.append(alpha_t)
            alpha_t_bar = torch.prod(torch.Tensor(alpha_1_to_t_array))

            noisy_image = torch.sqrt(alpha_t_bar) * x0 + torch.sqrt(1-alpha_t_bar) * epsilon

            # show_diffusion_chain(clean_image=x0[0], epsilon=epsilon[0], T=T)
            # quit()

            epsilon_pred = model(image=noisy_image, t=t)

            loss = loss_function(epsilon, epsilon_pred)  # todo: why don't we minimize the diff between epsilon_pred and sqrt(1-alpha_t)*epsilon?

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        last_loss = running_loss/len(training_loader)
        print("loss:", last_loss)
        running_loss = 0

    model_name = './models/model_' + dataset_name + "_" + str(round(last_loss,4)) + '.pth'
    torch.save(model, model_name)
    print('saved model to:', model_name)

def loss_function_mse(epsilon, epsilon_pred):
    loss = F.mse_loss(epsilon, epsilon_pred, reduction='none')
    return loss.mean()

