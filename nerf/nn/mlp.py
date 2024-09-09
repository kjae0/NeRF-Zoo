import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, act_fn=nn.ReLU, skip_concat_connection=[], skip_connection=[]):
        super().__init__()
        assert num_layers >= 2, "Number of layers must be greater than 1"
        
        self.skip_concat_connection = skip_concat_connection
        self.skip_connection = skip_connection
        
        self.layer = nn.ModuleList([nn.Linear(input_dim, hidden_dim), act_fn()] + 
            [nn.Linear(hidden_dim, hidden_dim), act_fn() for _ in range(num_layers - 2)] + \
                [nn.Linear(hidden_dim, output_dim)])
    
    def forward(self, x):
        for i, layer in enumerate(self.layer):
            if i in self.skip_concat_connection:
                x = torch.concat([x, layer(x)], dim=-1)
                
            x = layer(x)
            
            if i in self.skip_connection:
                x = x + layer(x)
        return x
    