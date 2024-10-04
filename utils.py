import numpy as np
import torch
from matplotlib import pyplot as plt
import os

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

def get_last_created_model():
    directory = './models'
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    sorted_files = sorted(files, key=os.path.getmtime, reverse=True)
    return sorted_files[0]

def rename_images_in_directory(directory, start_number=5000):
    files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])
    for idx, filename in enumerate(files, start=start_number):
        new_filename = f"Image_{idx}.jpg"
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        os.rename(old_filepath, new_filepath)