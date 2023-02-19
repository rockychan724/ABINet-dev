import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class Embedding(nn.Module):
    def __init__(self, in_timestep, in_planes, mid_dim = 4096, embed_dim=300):
        super(Embedding, self).__init__()
        self.in_timestep = in_timestep
        self.in_planes = in_planes
        self.embed_dim = embed_dim
        self.mid_dim = mid_dim
        self.eEmbed = nn.Linear(in_timestep * in_planes, self.embed_dim)  # Embed encoder output to a word-embedding like

    def forward(self, x):
        x = x.reshape((x.shape[0], -1))
        x = self.eEmbed(x)
        return x
