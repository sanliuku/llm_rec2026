"""原始 LRU 链路：concept 序列 + 纯 ID 嵌入，无 BERT / GAT / video 分支。
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LRUOld(nn.Module):
    """concept id 已整体 +1，0 为 padding。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device

        self.embedding = LRUEmbedding(config)
        self.model = LRUModel(config)
        self.truncated_normal_init(std=0.02, lower=-3 * 0.02, upper=3 * 0.02)
        # 与 sasrec_old 的 init_weight 一致：截断初始化后再把 padding 行清零
        with torch.no_grad():
            self.embedding.concept_emb.weight[0].zero_()

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for name, p in self.named_parameters():
                if 'layer_norm' in name or 'params_log' in name:
                    continue
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

    def get_item_embeddings(self):
        return self.embedding.concept_emb.weight  # (concept_num+1, emb_dim)，第 0 行为 padding

    def log2feats(self, log_seqs):
        x, mask = self.embedding(log_seqs)
        log_feats = self.model(x, mask)
        return log_feats, self.get_item_embeddings()

    def forward(self, log_seqs, next_concepts, next_neg_concepts):
        # 与 sasrec_old 一致：窗口内逐位置预测下一个 concept
        log_feats, Emb_k = self.log2feats(log_seqs)
        next_pos_logits = torch.sum(log_feats * Emb_k[next_concepts], dim=-1)
        next_neg_logits = torch.sum(log_feats * Emb_k[next_neg_concepts], dim=-1)
        return next_pos_logits, next_neg_logits

    def predict_concept(self, log_seqs):
        log_feats, Emb_k = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :].view(log_feats.shape[0], -1)
        return final_feat.mm(Emb_k.T)


class LRUEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_size = config.emb_dim
        self.concept_num = config.concept_num

        # 纯 ID 嵌入：第 0 行是 padding，不参与梯度
        self.concept_emb = nn.Embedding(config.concept_num + 1, embed_size, padding_idx=0)

        self.layer_norm = nn.LayerNorm(embed_size)
        self.embed_dropout = nn.Dropout(config.dropout_rate)

    def get_mask(self, x):
        return x > 0

    def forward(self, log_seqs):
        x = self.concept_emb(log_seqs)
        mask = self.get_mask(log_seqs)
        if self.training:
            return self.layer_norm(self.embed_dropout(x)), mask
        return self.layer_norm(x), mask


class LRUModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.emb_dim
        layers = config.num_blocks

        self.lru_blocks = nn.ModuleList([LRUBlock(config) for _ in range(layers)])

    def forward(self, x, mask):
        # left padding to the power of 2
        seq_len = x.size(1)
        log2_L = int(np.ceil(np.log2(seq_len)))
        x = F.pad(x, (0, 0, 2 ** log2_L - x.size(1), 0, 0, 0))
        mask_ = F.pad(mask, (2 ** log2_L - mask.size(1), 0, 0, 0))

        for lru_block in self.lru_blocks:
            x = lru_block(x, mask_)
        x = x[:, -seq_len:]  # B x L x D
        return x


class LRUBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.emb_dim
        self.lru_layer = LRULayer(
            d_model=hidden_size, dropout=config.dropout_rate,
            r_min=config.lru_r_min, r_max=config.lru_r_max)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden_size, d_ff=hidden_size * 4, dropout=config.dropout_rate)

    def forward(self, x, mask):
        x = self.lru_layer(x, mask)
        x = self.feed_forward(x)
        return x


class LRULayer(nn.Module):
    def __init__(self, d_model, dropout=0.1, use_bias=True, r_min=0.8, r_max=0.99):
        super().__init__()
        self.embed_size = d_model
        self.hidden_size = 2 * d_model
        self.use_bias = use_bias

        # init nu, theta, gamma
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
        self.out_vector = nn.Identity()

        # Dropout and layer norm
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    def lru_parallel(self, i, h, lamb, mask, B, L, D):
        # 并行化实现（考虑 0 padding），见 https://kexue.fm/archives/9554
        l = 2 ** i
        h = h.reshape(B * L // l, l, D)
        mask_ = mask.reshape(B * L // l, l)
        h1, h2 = h[:, :l // 2], h[:, l // 2:]

        if i > 1:
            lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
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
        return self.layer_norm(self.w_2(x_) + x)
