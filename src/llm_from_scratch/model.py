import torch
import torch.nn as nn

from .layers import TransformerBlock, LayerNorm


class GPTModel(nn.Module):
    def __init__(self, emb_dim, n_heads, n_layers, vocab_size, drop_rate, context_lenght):
        super().__init__()
        self.embeding = nn.Embedding(vocab_size, emb_dim)
        self.pos_embedding = nn.Embedding(context_lenght, emb_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.backbone = nn.Sequential(*[TransformerBlock(emb_dim, emb_dim, n_heads, drop_rate, context_lenght) for _ in range(n_layers)])
        self.lm_head =  nn.Linear(emb_dim, vocab_size)
        self.layer_norm = LayerNorm(emb_dim)

    def forward(self, x):
        emb = self.embeding(x)
        pos_embedding = self.pos_embedding(torch.arange(x.shape[-1]))

        x = emb + pos_embedding
        x = self.dropout(x)

        x = self.backbone(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)

        return logits

        