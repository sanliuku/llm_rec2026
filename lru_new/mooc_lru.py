from typing import Any
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch_geometric.nn import GATConv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class MOOCLRU(nn.Module):
    def __init__(
        self,
        config: Any,
        edge_index: torch.Tensor,
        emb_path: str = None,
    ):
        super().__init__()
        self.config = config
        self.device = config.device

        self.embedding = LRUEmbedding(config, edge_index, emb_path)

        self.model = LRUModel(config)
        self.truncated_normal_init(std=0.02, lower=-3 * 0.02, upper=3 * 0.02)

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for n, p in self.named_parameters():
                if not 'layer_norm' in n and 'params_log' not in n:
                    if torch.is_complex(p):
                        p.real.uniform_(2 * l - 1, 2 * u - 1)
                        p.imag.uniform_(2 * l - 1, 2 * u - 1)
                        p.real.erfinv_()
                        p.imag.erfinv_()
                        p.real.mul_(std * math.sqrt(2.))
                        p.imag.mul_(std * math.sqrt(2.))
                        p.real.add_(mean)
                        p.imag.add_(mean)
                    else:
                        p.uniform_(2 * l - 1, 2 * u - 1)
                        p.erfinv_()
                        p.mul_(std * math.sqrt(2.))
                        p.add_(mean)

    def forward(self, log_seqs, pos_seqs, neg_seqs, pos_concepts, neg_concepts):
        log_feats, Emb_v, Emb_k, mask = self.embedding(log_seqs)
        log_feats = self.model(log_feats, mask)

        pos_embs = Emb_v[pos_seqs]
        neg_embs = Emb_v[neg_seqs]

        pos_concept_embs = Emb_k[pos_concepts]
        neg_concept_embs = Emb_k[neg_concepts]

        pos_logits = torch.sum(log_feats * pos_embs, dim=-1)
        neg_logits = torch.sum(log_feats * neg_embs, dim=-1)

        log_last_feats = log_feats[:, -1, :].unsqueeze(1)
        pos_concept_logits = torch.sum(log_last_feats * pos_concept_embs, dim=-1)
        neg_concept_logits = torch.sum(log_last_feats * neg_concept_embs, dim=-1)
        return pos_logits, neg_logits, pos_concept_logits, neg_concept_logits

    def predict(self, log_seqs):
        log_feats, Emb_v, Emb_k, mask = self.embedding(log_seqs)
        log_feats = self.model(log_feats, mask)
        final_feat = log_feats[:, -1, :].view(log_feats.shape[0], -1) # (U, C)
        logits = final_feat.mm(Emb_v.T)
        return logits # preds # (U, I)

    def predict_concept(self, log_seqs):
        log_feats, Emb_v, Emb_k, mask = self.embedding(log_seqs)
        log_feats = self.model(log_feats, mask)
        final_feat = log_feats[:, -1, :].view(log_feats.shape[0], -1)
        logits = final_feat.mm(Emb_k.T)
        return logits

class LRUEmbedding(nn.Module):
    def __init__(
        self,
        config: Any,
        edge_index: torch.Tensor,
        emb_path: str = None,
    ):
        super().__init__()
        self.edge_index = edge_index.to(config.device)
        embed_size = config.emb_dim
        self.video_num = config.video_num
        self.concept_num = config.concept_num

        self.pad_emb = nn.Parameter(torch.zeros(1, embed_size), requires_grad=False)
        self.video_emb = nn.Embedding(config.video_num, embed_size)
        if emb_path is None:
            emb_path = os.path.join(PROJECT_ROOT, 'data/MOOCCube/mid_data/need_data/embs/concept_embs_local.npy') 
        self.concept_emb = nn.Embedding.from_pretrained(torch.from_numpy(np.load(emb_path)).float(), freeze=True)
        
        self.rd_linear = nn.Linear(config.bert_dim, config.emb_dim, bias=False)
        self.gat = GATConv(embed_size, embed_size, config.num_heads, concat=False, add_self_loops=True).to(config.device)

        self.layer_norm = nn.LayerNorm(embed_size)
        self.embed_dropout = nn.Dropout(config.dropout_rate)

    def get_item_embeddings(self):
        video_emb0 = self.video_emb.weight
        concept_emb = self.concept_emb.weight
        concept_emb0 = self.rd_linear(concept_emb)
        item_emb = torch.cat([video_emb0, concept_emb0], dim=0)
        new_agg = self.gat(item_emb, self.edge_index)
        video_emb, concept_emb = new_agg[:self.video_num], new_agg[self.video_num:]
        # res   having self-loop dont need residual
        video_emb = 1/2 * (video_emb + video_emb0)
        concept_emb = 1/2 * (concept_emb + concept_emb0)

        video_emb = torch.cat([self.pad_emb, video_emb], dim=0)
        concept_emb = torch.cat([self.pad_emb, concept_emb], dim=0)
        return video_emb, concept_emb

    def get_mask(self, x):
        return (x > 0)

    def forward(self, log_seqs):
        Emb_v, Emb_k = self.get_item_embeddings()
        mask = self.get_mask(log_seqs)

        x = Emb_v[log_seqs]
        if self.training:
            return self.layer_norm(self.embed_dropout(x)), Emb_v, Emb_k, mask
        else:
            return self.layer_norm(x), Emb_v, Emb_k, mask

class LRUModel(nn.Module):
    def __init__(
        self,
        config: Any
    ):
        super().__init__()
        self.hidden_size = config.emb_dim
        layers = config.num_blocks

        self.lru_blocks = nn.ModuleList([LRUBlock(config) for _ in range(layers)])
        # self.bias = torch.nn.Parameter(torch.zeros(config.video_num + 1))

    def forward(self, x, mask):
        # left padding to the power of 2
        seq_len = x.size(1)
        log2_L = int(np.ceil(np.log2(seq_len)))
        x = F.pad(x, (0, 0, 2 ** log2_L - x.size(1), 0, 0, 0))
        mask_ = F.pad(mask, (2 ** log2_L - mask.size(1), 0, 0, 0))

        # LRU blocks with pffn
        for lru_block in self.lru_blocks:
            x = lru_block.forward(x, mask_)
        x = x[:, -seq_len:]  # B x L x D (64)
        return x


class LRUBlock(nn.Module):
    def __init__(
        self,
        config: Any
    ):
        super().__init__()
        hidden_size = config.emb_dim
        self.lru_layer = LRULayer(
            d_model=hidden_size, dropout=config.dropout_rate)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden_size, d_ff=hidden_size * 4, dropout=config.dropout_rate)

    def forward(self, x, mask):
        x = self.lru_layer(x, mask)
        x = self.feed_forward(x)
        return x


class LRULayer(nn.Module):
    def __init__(self,
                 d_model,
                 dropout=0.1,
                 use_bias=True,
                 r_min=0.8,
                 r_max=0.99):
        super().__init__()
        self.embed_size = d_model
        self.hidden_size = 2 * d_model
        self.use_bias = use_bias

        # init nu, theta, gamma
        # 欧拉公式
        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))  # gamma = 1 - |lambda|^2
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

        # Init B, C, D
        self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=use_bias).to(torch.cfloat)
        self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=use_bias).to(torch.cfloat)
        # self.out_vector = nn.Parameter(torch.rand(self.embed_size))
        self.out_vector = nn.Identity()

        # Dropout and layer norm
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    def lru_parallel(self, i, h, lamb, mask, B, L, D):
        # Parallel algorithm, see: https://kexue.fm/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96
        # The original implementation is slightly slower and does not consider 0 padding
        l = 2 ** i
        h = h.reshape(B * L // l, l, D)  # (B, L, D) -> (B * L // 2, 2, D)
        mask_ = mask.reshape(B * L // l, l)  # (B, L) -> (B * L // 2, 2)
        h1, h2 = h[:, :l // 2], h[:, l // 2:]  # Divide data in half

        if i > 1: lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1:l // 2].unsqueeze(-1)
        h = torch.cat([h1, h2], axis=1)
        return h, lamb

    def forward(self, x, mask):
        # compute bu and lambda
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = self.in_proj(x.to(torch.cfloat)) * gamma  # bu

        # compute h in parallel
        log2_L = int(np.ceil(np.log2(h.size(1))))
        B, L, D = h.size(0), h.size(1), h.size(2)
        for i in range(log2_L):
            h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)
        if self.training:
            x = self.dropout(self.out_proj(h).real) + self.out_vector(x)
        else:
            x = self.out_proj(h).real + self.out_vector(x)
        return self.layer_norm(x)  # residual connection introduced above


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.w_1(x)))
        if self.training:
            return self.layer_norm(self.dropout(self.w_2(x_)) + x)
        else:
            return self.layer_norm(self.w_2(x_) + x)