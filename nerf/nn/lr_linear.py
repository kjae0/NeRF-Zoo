import torch
import torch.nn as nn

# Custom Linear Layer with Decomposed Weights
class DecomposedLinear(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if hidden_dim is None:
            hidden_dim = 2
        
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w = nn.Linear(hidden_dim, output_dim, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.randn(output_dim), requires_grad=True)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # Compute w^T x
        y = self.v(x)
        y = self.w(y)
        
        # Add bias if applicable
        if self.bias is not None:
            y = y + self.bias.unsqueeze(0)
        return y
    