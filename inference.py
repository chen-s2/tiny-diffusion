import numpy as np
from matplotlib import pyplot as plt

from unet import *

import torch
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention
from dataset import create_dataloader

# model_path = './models/model_0.2711.pth'
# model_path = './models/model_0.5413.pth'
model_path = './models/model_0.6079.pth'
T = 50
image_size=48

model = torch.load(model_path)
device = 'cuda'

timesteps = np.linspace(start=T,stop=1, num=T).astype('int')
beta = np.linspace(1e-4, 0.02, num=T)
beta = np.concatenate(([0.0],beta)) # for indexing between [1,T] instead of [0,T-1]
beta = torch.Tensor(beta)

for _ in range(5):
    x_t = torch.randn(torch.Size([1, 3, image_size, image_size]), dtype=torch.float32, device=device)

    for t in tqdm(timesteps):
        z = torch.randn(torch.Size([1, 3, image_size, image_size]), dtype=torch.float32, device=device)

        sigma_t = torch.sqrt(beta[t])
        sigma_t = sigma_t if t != timesteps[-1] else 0

        alpha_1_to_t_array = []
        for i in range(1, t+1):
            alpha_t = 1 - beta[i]
            alpha_1_to_t_array.append(alpha_t)
        alpha_t_bar = torch.prod(torch.Tensor(alpha_1_to_t_array))

        x_t_minus_1 = (1/torch.sqrt(alpha_t)) * (x_t - ((1-alpha_t) / (1-torch.sqrt(alpha_t_bar))) * model(x_t, t)) + sigma_t * z

    generated_image = x_t_minus_1
    generated_image = generated_image.detach().cpu().numpy()
    # print("img mean/std:", np.mean(generated_image), np.std(generated_image))
    # print("img min/max:", np.min(generated_image), np.max(generated_image))
    # print("img shape:", generated_image.shape)

    min_img, max_img = np.min(generated_image), np.max(generated_image)
    generated_image = 255.0*(generated_image/(max_img-min_img))

    r,g,b = generated_image.astype('uint8').squeeze()

    generated_image = generated_image.astype('uint8').squeeze()
    generated_image = np.transpose(generated_image, (1, 2, 0))
    print("generated_image mean/std:", np.mean(generated_image), np.std(generated_image))
    print("generated_image min/max:", np.min(generated_image), np.max(generated_image))
    print("generated_image shape:", generated_image.shape)

    plt.figure()
    plt.imshow(generated_image)
    plt.show(block=False)

plt.show()
print("done")