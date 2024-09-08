import torch
import torch.nn as nn


class NeRF(nn.Module):
    def __init__(self, hidden_dim=128, n_layers=8):
        super(NeRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.in_layer = nn.Linear(3, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.out_layer = nn.Linear(hidden_dim, 4)
        
    def forward(self, x):
        x = self.in_layer(x)
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.out_layer(x)
        
        return x



