import math
import torch
from utils import *

def get_selection_schedule(schedule, size, timesteps):
    '''
    schedule: type of schedule (linear, etc.)
    size: length of sub-sequence chosen
    timesteps: num of timesteps using in *training* the ddpm model
    '''

    assert schedule in {"linear", "quadratic"}

    if schedule == "linear":
        subsequence = torch.arange(0,timesteps, timesteps//size)
    else:
        subsequence = torch.pow(torch.linspace(0, math.sqrt(timesteps*0.8), size), 2).round().to(torch.int64)

    return subsequence

class GaussianDiffusion:
    def __init__(self, betas, model_mean_type, model_var_type, loss_type, **kwargs):

if __name__ == "__main__":
    subsequence = get_selection_schedule("linear", 10, 1000)
    print("subsequence:", subsequence[0:5], "...", subsequence[-5:])

    betas = get_beta_schedule("linear", 0.0001, 0.02, 1000)
    print("betas:", betas[0:5], "...", betas[-5:])

    batch_size = 1
    in_channels = 1
    image_res = 32

    input_shape = (in_channels, image_res, image_res)
    diffusion_kwargs = meta_config["diffusion"]
    beta_schedule = diffusion_kwargs.pop("beta_schedule")
    beta_start = diffusion_kwargs.pop("beta_start")
    beta_end = diffusion_kwargs.pop("beta_end")
    num_diffusion_timesteps = diffusion_kwargs.pop("timesteps")

    diffusion_kwargs["model_var_type"] = "fixed-small"
    skip_schedule = args.skip_schedule
    eta = args.eta
    subseq_size = 10
    subsequence = get_selection_schedule(skip_schedule, size=subseq_size, timesteps=num_diffusion_timesteps)

    diffusion = DDIM(betas, **diffusion_kwargs, eta=eta, subsequence=subsequence)

    shape = (batch_size,) + input_shape
    x = diffusion.p_sample(model, shape=shape, device=device, noise=torch.randn(shape, device=device)).cpu()
    x = (x * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()

    show_image(x)
    # diffusion = GaussianDiffusion(betas, "eps", "fixed-small", "mse")
    # print(diffusion.__dict__)