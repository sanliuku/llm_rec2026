"""A-LLMRec 第一阶段：协同嵌入与文本嵌入对齐（论文 4.1 节）。
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def make_activation(name: str) -> nn.Module:
    key = (name or 'none').strip().lower()
    table = {
        'none': lambda: nn.Identity(),
        'identity': lambda: nn.Identity(),
        'linear': lambda: nn.Identity(),
        'sigmoid': lambda: nn.Sigmoid(),
        'tanh': lambda: nn.Tanh(),
        'gelu': lambda: nn.GELU(),
        'relu': lambda: nn.ReLU(),
        'leakyrelu': lambda: nn.LeakyReLU(0.2),
        'leaky': lambda: nn.LeakyReLU(0.2),
    }
    if key not in table:
        raise ValueError(f'未知的 enc_act: {name}（可选 {sorted(table)}）')
    return table[key]()


class TwoLayerMLP(nn.Module):
    def __init__(self, dim: int, align_dim: int, act: str = 'leakyrelu'):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(dim, align_dim)
        self.fc2 = nn.Linear(align_dim, dim)
        self.act = make_activation(act)

    def encode(self, x: Tensor) -> Tensor:
        return self.act(self.fc1(x))

    def forward(self, x: Tensor) -> tuple:
        """返回 (对齐向量, 重建向量)。"""
        enc = self.encode(x)
        return enc, self.fc2(enc)


class MoocAlign(nn.Module):
    def __init__(self, cf_model: nn.Module, text_model: nn.Module, config: Any):
        super(MoocAlign, self).__init__()
        self.config = config
        self.cf = cf_model
        self.concept_num = cf_model.concept_num
        self.emb_dim = cf_model.emb_dim
        self.align_dim = config.align_dim
        self.text_model = text_model  

        act = getattr(config, 'enc_act', 'leakyrelu')
        self.item_mlp = TwoLayerMLP(self.emb_dim, config.align_dim, act=act)
        self.text_mlp = TwoLayerMLP(text_model.emb_dim, config.align_dim, act=act)

    def train(self, mode: bool = True):
        super(MoocAlign, self).train(mode)
        self.cf.eval()
        return self

    def forward(self, log_seqs: Tensor, pos_concepts: Tensor, neg_concepts: Tensor) -> dict:
        with torch.no_grad():
            log_feats, Emb_k = self.cf.log2feats(log_seqs)
            user_feat = log_feats[:, -1, :]  # x^u
            pos_emb = Emb_k[pos_concepts]
            neg_emb = Emb_k[neg_concepts]

        pos_match, pos_recon = self.item_mlp(pos_emb)
        neg_match, neg_recon = self.item_mlp(neg_emb)

        # Q_i 每步现算：梯度经 matching 回传到 SBERT（源码 pos_text_embedding 同此）
        pos_text = self.text_model(pos_concepts)
        neg_text = self.text_model(neg_concepts)
        pos_text_match, pos_text_recon = self.text_mlp(pos_text)
        neg_text_match, neg_text_recon = self.text_mlp(neg_text)

        bce = F.binary_cross_entropy_with_logits
        pos_logits = torch.sum(user_feat * pos_recon, dim=-1)
        neg_logits = torch.sum(user_feat * neg_recon, dim=-1)
        rec = bce(pos_logits, torch.ones_like(pos_logits)) + \
            bce(neg_logits, torch.zeros_like(neg_logits))

        matching = F.mse_loss(pos_match, pos_text_match) + F.mse_loss(neg_match, neg_text_match)
        item_recon = F.mse_loss(pos_recon, pos_emb) + F.mse_loss(neg_recon, neg_emb)
        text_recon = F.mse_loss(pos_text_recon, pos_text.detach()) + \
            F.mse_loss(neg_text_recon, neg_text.detach())

        diag = self._diagnostics(pos_match, pos_recon, pos_emb, pos_logits, neg_logits)

        return {
            'rec': rec,
            'matching': matching,
            'item_recon': item_recon,
            'text_recon': text_recon,
            'total': rec + matching
                     + self.config.alpha * item_recon + self.config.beta * text_recon,
            'diag': diag,
        }

    @staticmethod
    def _diagnostics(pos_match: Tensor, pos_recon: Tensor, pos_emb: Tensor,
                     pos_logits: Tensor, neg_logits: Tensor) -> dict:
        with torch.no_grad():
            e = pos_match.detach()
            nan = float('nan')
            if e.size(0) > 1:
                code_std = float(e.std(dim=0).mean())
                perm = torch.randperm(e.size(0), device=e.device)
                cos_ee = float(F.cosine_similarity(e, e[perm], dim=-1).mean())
            else:
                code_std, cos_ee = nan, nan
            rel = (pos_recon.detach() - pos_emb).norm(dim=-1) / pos_emb.norm(dim=-1).clamp_min(1e-8)
            return {
                'code_std': code_std,       # 码字逐维 std（塌缩时 →0）
                'cos_ee': cos_ee,           # 随机 concept 对的 cos（塌缩时 →1）
                'recon_rel': float(rel.median()),  # E_i' 与 E_i 的相对误差（重建质量）
                'logit_pos': float(pos_logits.detach().mean()),
                'logit_neg': float(neg_logits.detach().mean()),
            }

    # ------------------------------------------------------------------ 导出
    def joint_item_embeddings(self) -> Tensor:
        with torch.no_grad():
            return self.item_mlp.encode(self.cf.get_item_embeddings())

    def joint_text_embeddings(self) -> Tensor:
        with torch.no_grad():
            return self.text_mlp.encode(self.text_model.encode_all())

    def rec_item_embeddings(self) -> Tensor:
        with torch.no_grad():
            _, recon = self.item_mlp(self.cf.get_item_embeddings())
        return recon

    def predict_concept(self, log_seqs: Tensor) -> Tensor:
        log_feats, _ = self.cf.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :].view(log_feats.shape[0], -1)
        return final_feat.mm(self.rec_item_embeddings().T)
