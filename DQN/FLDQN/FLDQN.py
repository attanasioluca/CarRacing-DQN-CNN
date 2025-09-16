import torch
import torch.nn as nn
from typing import Union

class FLDQN(nn.Module):
    def __init__(self, in_dim: Union[int, object], out_dim: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        n_states = in_dim
        self.emb = nn.Embedding(num_embeddings=n_states, embedding_dim=embed_dim)
        self.q_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, a=-0.01, b=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dtype != torch.long:
            x = x.long()
        z = self.emb(x)
        q = self.q_head(z)
        return q