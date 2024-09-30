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

# noisy_image = torch.randn(torch.Size([1,3,32,32]), dtype=torch.float32, device=device) * 0.5 + 0.5
noisy_image = torch.randn(torch.Size([1,3,image_size,image_size]), dtype=torch.float32, device=device)
# print("noisy_image:")
# noisy_np = noisy_image.detach().cpu().numpy()
# print("img mean/std:", np.mean(noisy_np), np.std(noisy_np))
# print("img min/max:", np.min(noisy_np), np.max(noisy_np))
# print("img shape:", noisy_np.shape)

timesteps = np.linspace(start=T,stop=0, num=T).astype('int')
for t in tqdm(timesteps):
    noisy_image = model(noisy_image, t)

generated_image = noisy_image
generated_image = generated_image.detach().cpu().numpy()
print("img mean/std:", np.mean(generated_image), np.std(generated_image))
print("img min/max:", np.min(generated_image), np.max(generated_image))
print("img shape:", generated_image.shape)

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
plt.show()

print("done")