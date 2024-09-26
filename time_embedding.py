from torch import nn
from torch.nn import functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, 4*n_embd)

    def forward(self, x):
        '''
        x: (1,n_embd), by default it's (1,320)
        '''

        # (1,320) -> (1,1280)
        x = self.linear_1(x)

        # (1,1280) -> (1,1280)
        x = F.silu(x)

        # (1,1280) -> (1,1280)
        x = self.linear_2(x)

        return x