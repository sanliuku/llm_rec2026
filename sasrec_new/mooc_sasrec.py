import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class MOOCSASRec(nn.Module):
    def __init__(
        self,
        config: Any,
        edge_index: torch.Tensor,
        emb_path: str = None,
    ):
        super(MOOCSASRec, self).__init__()
        # load embedding
        self.video_num = config.video_num
        self.concept_num = config.concept_num
        self.emb_dim = config.emb_dim
        self.pad_emb = nn.Parameter(torch.zeros(1, self.emb_dim), requires_grad=False)
        self.video_emb = nn.Embedding(config.video_num, self.emb_dim)
        if emb_path is None:
            emb_path = os.path.join(PROJECT_ROOT, 'data/MOOCCube/mid_data/need_data/embs/concept_embs_local.npy')
        self.concept_emb = nn.Embedding.from_pretrained(torch.from_numpy(np.load(emb_path)).float(), freeze=True)

        self.pos_emb = nn.Embedding(config.max_len+1, self.emb_dim, padding_idx=0)
        # load components
        self.rd_linear = nn.Linear(config.bert_dim, config.emb_dim, bias=False)
        self.gat = GATConv(self.emb_dim, self.emb_dim, config.num_heads, concat=False, add_self_loops=True).to(config.device)

        self.emb_dropout = nn.Dropout(p=config.dropout_rate)
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(self.emb_dim, eps=1e-8)

        for _ in range(config.num_blocks):
            new_attn_layernorm = nn.LayerNorm(self.emb_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer =  nn.MultiheadAttention(self.emb_dim, config.num_heads, config.dropout_rate)
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = nn.LayerNorm(self.emb_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(self.emb_dim, config.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        self.device = config.device
        self.edge_index = edge_index.to(config.device)
        # self.init_weight()

    def init_weight(self):
        mean = 0
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean, std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean, std)

    def get_item_embeddings(self):
        video_emb0 = self.video_emb.weight
        concept_emb = self.concept_emb.weight
        concept_emb0 = self.rd_linear(concept_emb)
        # concept_emb0 = self.concept_emb.weight
        item_emb = torch.cat([video_emb0, concept_emb0], dim=0)
        new_agg = self.gat(item_emb, self.edge_index)
        video_emb, concept_emb = new_agg[:self.video_num], new_agg[self.video_num:]

        # res   having self-loop dont need residual
        video_emb = 1/2 * (video_emb + video_emb0)
        concept_emb = 1/2 * (concept_emb + concept_emb0)

        video_emb = torch.cat([self.pad_emb, video_emb], dim=0)
        concept_emb = torch.cat([self.pad_emb, concept_emb], dim=0)
        return video_emb, concept_emb

    def log2feats(self, log_seqs):
        Emb_v, Emb_k = self.get_item_embeddings()
        seqs = Emb_v[log_seqs]
        seqs *= self.emb_dim ** 0.5
        poss = torch.arange(1, log_seqs.shape[1]+1).view(1, -1).repeat(log_seqs.shape[0], 1).to(self.device)
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool)).to(self.device)

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        return Emb_v, Emb_k, log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs, pos_concepts, neg_concepts): # for training
        Emb_v, Emb_k, log_feats = self.log2feats(log_seqs)

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

    def predict(self, log_seqs): # for inference
        Emb_v, Emb_k, log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :].view(log_feats.shape[0], -1) # (U, C)
        logits = final_feat.mm(Emb_v.T)
        return logits # preds # (U, I)

    def predict_concept(self, log_seqs):
        Emb_v, Emb_k, log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :].view(log_feats.shape[0], -1)
        logits = final_feat.mm(Emb_k.T)
        return logits

# 1. true loop/
# 2. false loop / video res concept res
# 3. true / video concept
