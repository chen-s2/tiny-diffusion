from torch.nn import functional as F
import torch
from torch import nn
import math

class Time(nn.Module):
    def __init__(self, time_emb_dim, out_chan, hid_channels):
        super().__init__()
        self.fc_time = nn.Linear(time_emb_dim, out_chan)

    def forward(self, t_emb):
        return self.fc_time(nn.relu(t_emb))[:, :, None, None]

class TimeLinearEmbedder(nn.Module):
    def __init__(self, hid_channels, time_emb_dim):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Linear(hid_channels, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

    def forward(self, t_emb):
        self.embedder(t_emb)

# TODO: rebuild yourself
def REF_get_timestep_embedding(timesteps, embed_dim, dtype):
    """
    Adapted from fairseq/fairseq/modules/sinusoidal_positional_embedding.py
    The implementation is slightly different from the decription in Section 3.5 of [1]
    [1] Vaswani, Ashish, et al. "Attention is all you need."
     Advances in neural information processing systems 30 (2017).
    """
    half_dim = embed_dim // 2
    embed = math.log(10000) / (half_dim - 1)
    embed = torch.exp(-torch.arange(half_dim, dtype=dtype, device=timesteps.device) * embed)
    embed = torch.outer(timesteps.ravel().to(dtype), embed)
    embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=1)
    if embed_dim % 2 == 1:
        embed = F.pad(embed, [0, 1])  # padding the last dimension
    assert embed.dtype == dtype
    return embed



