import gc
import shutil
import time
from pathlib import Path

from unet import *
import torch
from tqdm import tqdm

model_path = get_last_created_model()
print("using model:", model_path)

T = 1000
num_intervals = T
image_size = 64
batch_size = 16
transition_between_two_latent_values = True
channels_num = 3

model = torch.load(model_path)
device = 'cuda'

timesteps = np.linspace(start=T,stop=1, num=num_intervals).astype('int')
beta = np.linspace(1e-4, 0.02, num=T)
beta = np.concatenate(([0.0],beta))  # index between [1,T] instead of [0,T-1]
beta = torch.Tensor(beta)

def calculate_latent_transition(z):
    batchsize = z.shape[0]
    start_latent = z[0]
    end_latent = z[batchsize-1]
    latents_transition_interpolated = torch.stack([(1 - alpha) * start_latent + alpha * end_latent for alpha in torch.linspace(0, 1, batchsize)])
    return latents_transition_interpolated

def run_inference():
    if transition_between_two_latent_values:
        x_t = torch.randn(torch.Size([channels_num, image_size, image_size]), dtype=torch.float32, device=device)
        x_t = x_t.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        x_t = torch.randn(torch.Size([batch_size, channels_num, image_size, image_size]), dtype=torch.float32, device=device)

    with torch.no_grad():
        for t in tqdm(timesteps):
            z = torch.randn(torch.Size([batch_size, channels_num, image_size, image_size]), dtype=torch.float32, device=device)
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

    return generated_image

def save_image(generated_image, img_index, saved_dir, saved_img_name):
    noisy_image_transposed = np.transpose(generated_image[img_index], (1, 2, 0))
    normalized_image = np.zeros(noisy_image_transposed.shape)

    for channel in range(channels_num):
        min_img_in_channel = noisy_image_transposed[:, :, channel].min()
        max_img_in_channel = noisy_image_transposed[:, :, channel].max()
        normalized_image[:, :, channel] = 255.0 * ((noisy_image_transposed[:, :, channel] - min_img_in_channel) / (max_img_in_channel - min_img_in_channel))
    noisy_image_transposed = normalized_image
    noisy_image_transposed = noisy_image_transposed.astype('uint8').squeeze()

    saved_frame_path = os.path.join(saved_dir, saved_img_name.replace('.png', '_frame_' + f"{img_index:03}" + '.png'))
    Path(os.path.dirname(saved_frame_path)).mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    plt.imshow(noisy_image_transposed, cmap='gray')
    plt.axis('off')

    plt.savefig(saved_frame_path, transparent=True, bbox_inches='tight', pad_inches=0)
    print("saved to:", saved_frame_path)
    plt.close(fig)

model_name = os.path.basename(model_path).replace('.pth','')

saved_dir = os.path.join('results', model_name)

if os.path.exists(saved_dir):
    shutil.rmtree(saved_dir)
os.makedirs(saved_dir)

generated_image = run_inference()

timestr = time.strftime("%Y%m%d-%H%M%S")

num_images_generated = batch_size
saved_img_name = os.path.basename(model_path).replace('.pth','') + "_" + timestr + ".png"
saved_img_out_path = os.path.join('results', saved_img_name)

for img_index in range(batch_size):
    save_image(generated_image, img_index, saved_dir, saved_img_name)

create_video_from_images_dir(saved_dir, fps=10)