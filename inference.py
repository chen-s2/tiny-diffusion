import numpy as np
from matplotlib import pyplot as plt
from utils import *
from unet import *
import torch
from tqdm import tqdm

model_path = './models/model_0.5343.pth'
T = 200
image_size = 48

model = torch.load(model_path)
device = 'cuda'

timesteps = np.linspace(start=T,stop=1, num=T).astype('int')
beta = np.linspace(1e-4, 0.02, num=T)
beta = np.concatenate(([0.0],beta)) # for indexing between [1,T] instead of [0,T-1]
beta = torch.Tensor(beta)

num_images_generated = 2
fig, axes = plt.subplots(1, num_images_generated, figsize=(16, 3))

for img_index in range(num_images_generated):
    x_t = torch.randn(torch.Size([1, 1, image_size, image_size]), dtype=torch.float32, device=device)

    for t in tqdm(timesteps):
        z = torch.randn(torch.Size([1, 1, image_size, image_size]), dtype=torch.float32, device=device)

        sigma_t = torch.sqrt(beta[t])
        sigma_t = sigma_t if t != timesteps[-1] else 0

        alpha_1_to_t_array = []
        for i in range(1, t+1):
            alpha_t = 1 - beta[i]
            alpha_1_to_t_array.append(alpha_t)
        alpha_t_bar = torch.prod(torch.Tensor(alpha_1_to_t_array))

        # x_t_minus_1 = (1/torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1-alpha_t_bar)) * model(x_t, t)) + sigma_t * z
        x_t_minus_1 = x_t - ((1 - alpha_t) / (1 - torch.sqrt(alpha_t_bar))) * model(x_t, t) + sigma_t * z # works but the denominator is incorrect according to algo 2
        # x_t_minus_1 = 0.2 * (x_t - ((1 - alpha_t) / torch.sqrt(1-alpha_t_bar)) * model(x_t, t)) # + 0.1 * sigma_t * z

        if t%20 == 0:
            print("t:", t, ", mean/std:", torch.mean(x_t_minus_1), torch.std(x_t_minus_1))
        #     print("\nt:", t)
        #     print("eps_pred coeffcient:", ((1-alpha_t) / (1-torch.sqrt(alpha_t_bar))))
        #     print("z coeffcient:", sigma_t)
        #     print("denoised_image coefficent:", (1/torch.sqrt(alpha_t)))

        x_t = x_t_minus_1

    generated_image = x_t_minus_1
    generated_image = generated_image.detach().cpu().numpy()

    min_img, max_img = np.min(generated_image), np.max(generated_image)
    generated_image = 255.0*(generated_image/(max_img-min_img))

    gray = generated_image.astype('uint8').squeeze()

    generated_image = generated_image.astype('uint8').squeeze()
    show_stats_np_tensor(generated_image, "generated_image")

    axes[img_index].imshow(gray, cmap='gray')
    axes[img_index].axis('off')

plt.tight_layout()
plt.show()
print("done")