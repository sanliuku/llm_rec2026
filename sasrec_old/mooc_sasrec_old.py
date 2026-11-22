"""原始 SASRec 链路：concept 序列 + 纯 ID 嵌入，无 BERT / GAT / video 分支。
"""

from typing import Any

import torch
import torch.nn as nn

from torch import Tensor


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRecOld(nn.Module):
    """concept id 已整体 +1，0 为 padding。"""

    def __init__(self, config: Any):
        super(SASRecOld, self).__init__()
        self.concept_num = config.concept_num
        self.emb_dim = config.emb_dim
        self.device = config.device

        # 纯 ID 嵌入：第 0 行是 padding，不参与梯度
        self.concept_emb = nn.Embedding(config.concept_num + 1, config.emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(config.max_len + 1, config.emb_dim, padding_idx=0)

        self.emb_dropout = nn.Dropout(p=config.dropout_rate)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(self.emb_dim, eps=1e-8)

        for _ in range(config.num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(self.emb_dim, eps=1e-8))
            self.attention_layers.append(nn.MultiheadAttention(self.emb_dim, config.num_heads,
                                                               config.dropout_rate))
            self.forward_layernorms.append(nn.LayerNorm(self.emb_dim, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(self.emb_dim, config.dropout_rate))

        self.init_weight()

    def init_weight(self):
        mean, std = 0, 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean, std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean, std)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)

    def get_item_embeddings(self):
        return self.concept_emb.weight  # (concept_num+1, emb_dim)，第 0 行为 padding

    def log2feats(self, log_seqs):
        Emb_k = self.get_item_embeddings()

        seqs = Emb_k[log_seqs]
        seqs = seqs * (self.emb_dim ** 0.5)

        poss = torch.arange(1, log_seqs.shape[1] + 1, device=log_seqs.device).view(1, -1)
        poss = poss.repeat(log_seqs.shape[0], 1)
        poss = poss * (log_seqs != 0)
        seqs = seqs + self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=log_seqs.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)
        return log_feats, Emb_k

    def forward(self, log_seqs, next_concepts, next_neg_concepts):
        log_feats, Emb_k = self.log2feats(log_seqs)

        # 逐位置预测下一个 concept（窗口末尾接 label video 的首个 concept）
        next_pos_logits = torch.sum(log_feats * Emb_k[next_concepts], dim=-1)
        next_neg_logits = torch.sum(log_feats * Emb_k[next_neg_concepts], dim=-1)
        return next_pos_logits, next_neg_logits

    def predict_concept(self, log_seqs):
        log_feats, Emb_k = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :].view(log_feats.shape[0], -1)
        return final_feat.mm(Emb_k.T)
