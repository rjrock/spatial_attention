'''attention.py'''

import torch.nn as nn


class SpatialAttention(nn.Module):
    '''See section 2.2 of 'Knowing When to Look'.'''
    def __init__(self, k, d):
        super(SpatialAttention, self).__init__()
        self.W_v     = nn.Linear(d, k)
        self.W_g     = nn.Linear(d, k)
        self.w_h     = nn.Linear(k, 1)
        self.tanh    = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, V, h):
        a = self.W_v(V)
        b = self.W_g(h).unsqueeze(-1).expand(a.shape)
        z = self.w_h(self.tanh(a + b)).squeeze(dim=-1)
        α = self.softmax(z)
        c = (α.unsqueeze(-1) * V).sum(dim=1)
        return c, α
