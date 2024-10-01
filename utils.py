import numpy as np

def show_stats_np_tensor(np_tensor, title):
    print(title, "mean/std:", np.mean(np_tensor), np.std(np_tensor))
    print(title, "min/max:", np.min(np_tensor), np.max(np_tensor))
    print(title, "shape:", np_tensor.shape)
