import torch
import torch.nn as nn

# Custom Linear Layer with Decomposed Weights
class DecomposedLinear(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Initialize the decomposed weight vectors v and w
        self.v = nn.Parameter(torch.randn(output_dim))
        self.w = nn.Parameter(torch.randn(input_dim))
        if bias:
            self.bias = nn.Parameter(torch.randn(output_dim))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # Compute w^T x
        wx = x @ self.w  # Shape: (batch_size,)
        # Compute v * (w^T x)
        y = self.v.unsqueeze(0) * wx.unsqueeze(1)  # Shape: (batch_size, output_dim)
        # Add bias if applicable
        if self.bias is not None:
            y = y + self.bias.unsqueeze(0)
        return y
    