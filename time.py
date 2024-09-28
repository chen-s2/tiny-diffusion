from torch.nn import functional as F
import torch
from torch import nn
import math

class Time(nn.Module):
    def __init__(self, time_emb_dim, out_chan):
        super().__init__()
        self.fc_time = nn.Linear(time_emb_dim, out_chan)

    def forward(self, t_emb):
        return self.fc_time(nn.relu(t_emb))[:, :, None, None]

# TODO: rebuild yourself
def REFERENCE_get_timestep_embedding(timesteps, embed_dim: int, dtype: torch.dtype = DEFAULT_DTYPE):
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

def get_timestep_embedding(timesteps, embed_dim, dtype):
    pass

# beginning :
self.embed = Sequential(
    Linear(self.hid_channels, self.time_embedding_dim),
    self.nonlinearity,
    Linear(self.time_embedding_dim, self.time_embedding_dim)
)

t_emb = get_timestep_embedding(t, self.hid_channels)
t_emb = self.embed(t_emb)

# residual block:
def forward(self, x, t_emb):
    skip = self.skip(x)
    x = self.conv1(self.nonlinearity(self.norm1(x)))
    x += self.fc(self.nonlinearity(t_emb))[:, :, None, None]
    x = self.dropout(self.nonlinearity(self.norm2(x)))
    x = self.conv2(x)
    return x + skip
