import torch
import torch.nn as nn


class lightGCN(nn.Module):
    def __init__(self,
                 num_users,
                 num_items,
                 emb_dim,
                 num_layers,
                 drop_rate):
        super(lightGCN, self).__init__()
        self.Emb_u = nn.Embedding(num_users, emb_dim)
        self.Emb_k = nn.Embedding(num_items, emb_dim)
        self.num_layers = num_layers
        self.dropout = drop_rate
        self.Dropout = nn.Dropout(drop_rate)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.normal_(self.Emb_u.weight, std=0.1)
        # nn.init.normal_(self.Emb_k.weight, std=0.1)
        nn.init.xavier_uniform_(self.Emb_u.weight)
        nn.init.xavier_uniform_(self.Emb_k.weight)

    def Dropout_with_Graph(self, H):
        device = H.device
        sp_mat = H.coalesce()
        if self.dropout == 0:
            return sp_mat
        p = torch.ones(sp_mat._nnz(), device=device) * self.dropout
        keep_mask = torch.bernoulli(1 - p).to(device)
        fill_values = torch.logical_not(keep_mask) * 0.0
        new_sp_mat = torch.sparse_coo_tensor(
            sp_mat._indices(),
            sp_mat._values() * keep_mask + fill_values,
            size=sp_mat.size(),
            device=sp_mat.device,
            dtype=sp_mat.dtype,
        )
        return new_sp_mat

    def forward(self, H):
        u_embs = self.Emb_u.weight
        i_embs = self.Emb_k.weight
        all_embs = torch.cat([u_embs, i_embs], dim=0)

        embs = [all_embs]
        for _ in range(self.num_layers):
            if self.training:
                H = self.Dropout_with_Graph(H)
            all_embs = torch.sparse.mm(H, all_embs)
            embs.append(all_embs)
        embs = torch.stack(embs, dim=1)
        embs = torch.mean(embs, dim=1)

        u_embs, i_embs = torch.split(embs, [u_embs.size(0), i_embs.size(0)], dim=0)
        return u_embs, i_embs
