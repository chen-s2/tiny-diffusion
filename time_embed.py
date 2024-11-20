from torch.nn import functional as F
import torch
from torch import nn
import math

class Time(nn.Module):
    def __init__(self, time_emb_dim, out_chan):
        super().__init__()
        self.fc_time = nn.Linear(time_emb_dim, time_emb_dim)
        self.relu = nn.ReLU()

    def forward(self, t_emb):
        return self.fc_time(self.relu(t_emb))

class TimeLinearEmbedder(nn.Module):
    def __init__(self, hid_channels, time_emb_dim):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Linear(hid_channels, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

    def forward(self, t_emb):
        return self.embedder(t_emb)

def get_timestep_embedding(timesteps, embed_dim, dtype, device):
    """
    adapted from: https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/sinusoidal_positional_embedding.py
    """
    timesteps = torch.Tensor(timesteps).to(device)
    half_dim = embed_dim // 2
    embedding = math.log(10000) / (half_dim - 1)
    embedding = torch.exp(-torch.arange(half_dim, dtype=dtype, device=device) * embedding)
    embedding = torch.outer(timesteps.ravel().to(dtype), embedding)
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
    if embed_dim % 2 == 1:
        embedding = F.pad(embedding, [0, 1])  # padding the last dimension
    assert embedding.dtype == dtype
    return embedding



