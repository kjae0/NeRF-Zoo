import math
import torch
import torch.nn as nn

class SinusoidalEmbedding(torch.nn.Module):
    def __init__(self, n_dim):
        super(SinusoidalEmbedding, self).__init__()
        self.n_dim = n_dim
        self.exp = torch.FloatTensor([2 ** (i // 2) * math.pi for i in range(2 * n_dim)]).unsqueeze(0) # 1 x 2L
        self.exp = nn.Parameter(self.exp, requires_grad=False)

    def forward(self, p):
        """
        p: A tensor containing positions to encode, typically a scalar or tensor with positions (e.g., input indices).
        """
        # p: B x input_dim
        B, _ = p.size()
        p = p.unsqueeze(2) # B x input_dim x 1
        p = (p * self.exp).view(B, -1)
        
        for i in range(p.shape[1] // 2):
            p[:, 2 * i] = torch.sin(p[:, 2 * i])
            p[:, 2 * i + 1] = torch.cos(p[:, 2 * i + 1])
        
        return p
        