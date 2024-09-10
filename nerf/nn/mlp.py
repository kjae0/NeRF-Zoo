import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, act_fn=nn.ReLU(), skip_concat_connection=[], skip_connection=[]):
        super().__init__()
        assert num_layers >= 2, "Number of layers must be greater than 1"
        
        self.skip_concat_connection = skip_concat_connection
        self.skip_connection = skip_connection
        
        self.layer = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.act_fn = act_fn
        
        for i in range(num_layers - 1):
            if (i+1) in skip_concat_connection:
                current_hidden_dim = hidden_dim + input_dim
            else:
                current_hidden_dim = hidden_dim
                
            if i == num_layers - 2:
                current_output_dim = output_dim
            else:
                current_output_dim = hidden_dim
                
            self.layer.append(nn.Linear(current_hidden_dim, current_output_dim))
    
    def forward(self, x):
        out = x.clone()
        for i, layer in enumerate(self.layer):
            if i in self.skip_concat_connection:
                out = torch.concat([x, out], dim=-1)

            out = layer(out)
            
            if i < len(self.layer) - 1:
                out = self.act_fn(out)
                
            if i in self.skip_connection:
                out = x + out
                
        return out
    