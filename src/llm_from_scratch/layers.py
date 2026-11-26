import torch.nn as nn
import torch

from .attention import MultiHeadAttention

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) *
        (x + 0.044715 * torch.pow(x, 3))
        ))

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class FeedForward(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            GELU(),
            nn.Linear(emb_size * 4, emb_size)
        )
        

    def forward(self, x):
        output = self.network(x)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, out_dim, n_heads, drop_rate, context_length):
        super().__init__()
        self.attention = MultiHeadAttention(emb_dim, emb_dim, context_length, drop_rate, n_heads)
        self.dropout = nn.Dropout(drop_rate)
        self.feedfordward = FeedForward(emb_dim)
        self.layer_norm = LayerNorm(emb_dim)


    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.attention(x)
        x = self.dropout(x)

        x += residual
        residual = x

        x = self.layer_norm(x)
        x = self.feedfordward(x)
        x = self.dropout(x)

        x += residual
        return x