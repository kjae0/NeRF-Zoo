import torch
import torch.nn as nn

from nerf.nn import mlp

class NeRF(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, volume_dim=1, radiance_dim=3, n_layers=8, skip_concat_connection=[4], skip_connection=[]):
        super(NeRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # embedding layer
        self.embedding = None
        
        # nerf
        self.body = mlp.MLP(input_dim=input_dim, 
                            hidden_dim=hidden_dim, 
                            output_dim=hidden_dim, 
                            num_layers=n_layers, 
                            skip_concat_connection=skip_concat_connection, 
                            skip_connection=skip_connection)
        self.RGB_layer = mlp.MLP(input_dim=hidden_dim, 
                                 hidden_dim=int(hidden_dim // 2), 
                                 output_dim=radiance_dim, 
                                 num_layers=2)
        self.sigma_layer = nn.Linear(hidden_dim, volume_dim)
        
    def forward(self, xyz, direction):
        xyz = self.embedding(xyz) # B x 3 -> B x hidden_dim
        direction = self.embedding(direction) # B x 2 -> B x hidden_dim
        
        hs = self.body(xyz)
        sigma = self.sigma_layer(hs)
        rgb = self.RGB_layer(torch.concat([hs, direction], dim=-1))
        
        return sigma, rgb



