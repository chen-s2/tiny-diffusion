import gc

import numpy as np
from matplotlib import pyplot as plt
from utils import *
from unet import *
import torch
from tqdm import tqdm

model_path = get_last_created_model()
print("using model:", model_path)

T = 1000
num_intervals = T
image_size = 64
batch_size = 32
transition_between_two_latent_values = True

model = torch.load(model_path)
device = 'cuda'

timesteps = np.linspace(start=T,stop=1, num=num_intervals).astype('int')
beta = np.linspace(1e-4, 0.02, num=T)
beta = np.concatenate(([0.0],beta)) # for indexing between [1,T] instead of [0,T-1]
beta = torch.Tensor(beta)

num_images_generated = batch_size
fig, axes = plt.subplots(np.ceil(batch_size/8.0).astype('int'), np.min((num_images_generated,8)), figsize=(16, 3))

if transition_between_two_latent_values:
    x_t = torch.randn(torch.Size([1, image_size, image_size]), dtype=torch.float32, device=device)
    x_t = x_t.unsqueeze(0).repeat(batch_size, 1, 1, 1)
else:
    x_t = torch.randn(torch.Size([batch_size, 1, image_size, image_size]), dtype=torch.float32, device=device)
generated_images = []

def calculate_latent_transition(z):
    batchsize = z.shape[0]
    start_latent = z[0]
    end_latent = z[batchsize-1]
    latents_transition_interpolated = torch.stack([(1 - alpha) * start_latent + alpha * end_latent for alpha in torch.linspace(0, 1, batchsize)])
    return latents_transition_interpolated

with torch.no_grad():
    for t in tqdm(timesteps):
        z = torch.randn(torch.Size([batch_size, 1, image_size, image_size]), dtype=torch.float32, device=device)
        if transition_between_two_latent_values:
            z = calculate_latent_transition(z)

        sigma_t = torch.sqrt(beta[t])
        sigma_t = sigma_t if t != timesteps[-1] else 0

        alpha_1_to_t_array = []
        for i in range(1, t+1):
            alpha_t = 1 - beta[i]
            alpha_1_to_t_array.append(alpha_t)
        alpha_t_bar = torch.prod(torch.Tensor(alpha_1_to_t_array))

        x_t_minus_1 = (1/torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1-alpha_t_bar)) * model(x_t, t)) + sigma_t * z

        x_t, alpha_t_bar, alpha_t, sigma_t, z = [None for i in range(5)]
        if t%100 == 0:
            print("t:", t, ", mean/std:", dt(torch.mean(x_t_minus_1)), dt(torch.std(x_t_minus_1)))
            with torch.no_grad():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()

        x_t = x_t_minus_1

        del x_t_minus_1

generated_image = x_t
generated_image = generated_image.detach().cpu().numpy()

min_img, max_img = np.min(generated_image), np.max(generated_image)
generated_image = 255.0*((generated_image-min_img)/(max_img-min_img))
show_stats_np_tensor(generated_image, "generated_image")

generated_image = generated_image.astype('uint8').squeeze()
show_stats_np_tensor(generated_image, "generated_image")

for img_index in range(batch_size):
    if batch_size > 8:
        axes[np.floor(img_index/8.0).astype('int'), img_index%8].imshow(generated_image[img_index], cmap='gray')
        axes[np.floor(img_index/8.0).astype('int'), img_index%8].axis('off')
    elif 8 >= batch_size > 1:
        axes[img_index].imshow(generated_image[img_index], cmap='gray')
        axes[img_index].axis('off')
    else:
        plt.imshow(generated_image, cmap='gray')

plt.tight_layout()
plt.title(os.path.basename(model_path))
plt.show()
print("done")