import torch
import torch.nn as nn

from nerf.nn import mlp
from nerf.nn import embedding
from nerf.nn import lr_mlp

class LowRankNeRF(nn.Module):
    def __init__(self, xyz_dim=3, xyz_embedding_dim=10, 
                 direction_dim=3, direction_embedding_dim=4, 
                 hidden_dim=256, volume_dim=1, radiance_dim=3, 
                 n_layers=8, skip_concat_connection=[4], skip_connection=[]):
        super(LowRankNeRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        input_xyz_dim = xyz_dim * xyz_embedding_dim * 2
        input_direction_dim = direction_dim * direction_embedding_dim * 2
        
        # embedding layer
        self.xyz_embedding = embedding.SinusoidalEmbedding(xyz_embedding_dim)
        self.direction_embedding = embedding.SinusoidalEmbedding(direction_embedding_dim)
        
        # nerf
        self.body = lr_mlp.MLP(input_dim=input_xyz_dim, 
                            hidden_dim=hidden_dim, 
                            output_dim=hidden_dim, 
                            num_layers=n_layers, 
                            skip_concat_connection=skip_concat_connection, 
                            skip_connection=skip_connection)
        self.RGB_layer = mlp.MLP(input_dim=hidden_dim+input_direction_dim, 
                                 hidden_dim=int(hidden_dim // 2), 
                                 output_dim=radiance_dim, 
                                 num_layers=2)
        self.sigma_layer = nn.Linear(hidden_dim, volume_dim)
        
    def forward(self, xyz, direction):
        xyz = self.xyz_embedding(xyz) # B x 3 -> B x hidden_dim
        direction = self.direction_embedding(direction) # B x 3 -> B x hidden_dim

        hs = self.body(xyz)
        sigma = self.sigma_layer(hs)
        rgb = self.RGB_layer(torch.concat([hs, direction], dim=-1))
        
        return rgb, sigma

if __name__ == '__main__':
    # Example Usage
    model = LowRankNeRF()
    print(model)
    xyz = torch.randn(1, 3)
    direction = torch.randn(1, 3)
    rgb, sigma = model(xyz, direction)
    print(rgb.shape, sigma.shape)

