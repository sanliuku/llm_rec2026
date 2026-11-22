import torch
import torch.nn as nn


class BPRMF(nn.Module):
    def __init__(self,
                 num_users,
                 num_items,
                 emb_dim,
                 drop_rate):
        super(BPRMF, self).__init__()
        self.emb_dim = emb_dim
        self.Emb_u = nn.Embedding(num_users, emb_dim)
        self.Emb_k = nn.Embedding(num_items, emb_dim)
        self.Dropout = nn.Dropout(drop_rate)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Emb_u.weight)
        nn.init.xavier_uniform_(self.Emb_k.weight)

    def forward(self):
        u_embs = self.Emb_u.weight
        i_embs = self.Emb_k.weight
        if self.training:
            u_embs = self.Dropout(u_embs)
            i_embs = self.Dropout(i_embs)
        return u_embs, i_embs
