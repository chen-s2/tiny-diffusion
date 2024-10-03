import numpy as np
import torch
from matplotlib import pyplot as plt

def show_stats_np_tensor(np_tensor, title):
    print(title, "mean/std:", np.mean(np_tensor), np.std(np_tensor))
    print(title, "min/max:", np.min(np_tensor), np.max(np_tensor))
    print(title, "shape:", np_tensor.shape)

def show_image(img, title, block=False):
    get_tensor_stats(img, title)
    plt.figure()
    plt.imshow(img[0,0,:,:].detach().cpu().numpy())
    plt.title(title)
    plt.show(block=block)

def get_tensor_stats(tensor, title):
    print("tensor:", title)
    print("mean/std:", torch.mean(tensor), ",", torch.std(tensor))
    print("min/max:", torch.min(tensor), torch.max(tensor))

def dt(tensor):
    return tensor.detach().cpu().numpy()