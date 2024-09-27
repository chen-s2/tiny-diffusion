import numpy as np
from matplotlib import pyplot as plt

from unet import *

import torch
from torch import nn
from tqdm import tqdm
from time_embedding import TimeEmbedding
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention
from dataset import create_dataloader

model_path = './models/model_0.2711.pth'
model = torch.load(model_path)
device = 'cuda'

noisy_image = torch.randn(torch.Size([1,3,32,32]), dtype=torch.float32, device=device) * 0.5 + 0.5
generated_image = model(noisy_image)
generated_image = generated_image.detach().cpu().numpy()
print("img mean/std:", np.mean(generated_image), np.std(generated_image))
print("img min/max:", np.min(generated_image), np.max(generated_image))
print("img shape:", generated_image.shape)

min_img, max_img = np.min(generated_image), np.max(generated_image)
generated_image = 255.0*(generated_image/(max_img-min_img))

print("new img mean/std:", np.mean(generated_image), np.std(generated_image))
print("new img min/max:", np.min(generated_image), np.max(generated_image))

r,g,b = generated_image.astype('uint8').squeeze()
plt.figure()
plt.imshow(r, cmap='gray')
plt.show()

print("done")