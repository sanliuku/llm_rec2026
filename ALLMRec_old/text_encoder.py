"""文本侧：concept 文本 -> SBERT 编码 Q_i（源码 self.sbert），默认参与微调。
"""

import os
import pickle as pkl

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

# 离线加载本地权重，别在训练里等网络
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def load_concept_texts(concept_csv: str, text_pkl: str, num_concept: int) -> list:
    df = pd.read_csv(concept_csv, header=None, names=['index', 'id']).sort_values('index')
    explain = pkl.load(open(text_pkl, 'rb'))
    names = df['id'].tolist()
    if len(names) < num_concept:
        raise ValueError(f'concepts.csv 只有 {len(names)} 行，少于 ckpt 的 concept 数 {num_concept}')

    texts = []
    for name in names[:num_concept]:
        desc = explain.get(name)
        texts.append(f'{name}, {desc}' if desc else str(name))
    return [''] + texts


def resolve_model_path(name: str) -> str:
    if os.path.isdir(name):
        return name
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(name, local_files_only=True)
    except Exception:
        return name


class ConceptTextEncoder(nn.Module):

    def __init__(self, model_name: str, texts: list, max_length: int = 128, finetune: bool = True,
                 freeze_layers: int = 0, freeze_embeddings: bool = True):
        super(ConceptTextEncoder, self).__init__()
        path = resolve_model_path(model_name)
        from sentence_transformers import SentenceTransformer  # 只有在线这条路才需要
        self.sbert = SentenceTransformer(path, device='cpu', local_files_only=True)
        self.emb_dim = self.sbert.get_sentence_embedding_dimension()
        self.finetune = finetune
        self._cache = None
        self._frozen_modules = []

        if finetune:
            self._freeze_bottom_layers(freeze_layers, freeze_embeddings)

        tokens = self.sbert.tokenizer(texts, padding='max_length', truncation=True,
                                      max_length=max_length, return_tensors='pt')
        self.register_buffer('input_ids', tokens['input_ids'], persistent=False)
        self.register_buffer('attention_mask', tokens['attention_mask'], persistent=False)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'text encoder: {model_name} (dim={self.emb_dim}, max_len={max_length}, '
              f'finetune={finetune}, trainable={trainable:,}, texts={len(texts)})')

    def _freeze_bottom_layers(self, freeze_layers: int, freeze_embeddings: bool) -> None:
        auto_model = getattr(self.sbert[0], 'auto_model', None)
        if auto_model is None or not hasattr(auto_model, 'encoder'):
            raise RuntimeError(f'{type(self.sbert[0]).__name__} 不是 BERT 系结构，无法按层冻结')

        total = len(auto_model.encoder.layer)
        n = max(0, min(int(freeze_layers), total))
        if n and not freeze_embeddings:
            print('[warn] freeze_layers>0 但 embedding 没冻，梯度仍要穿透底层，冻层不省时间')

        targets = []
        if freeze_embeddings:
            targets.append(auto_model.embeddings)
        if getattr(auto_model, 'pooler', None) is not None:
            targets.append(auto_model.pooler)  # mean/CLS pooling 都不走 pooler，它本来就没梯度
        targets.extend(auto_model.encoder.layer[i] for i in range(n))

        for module in targets:
            for param in module.parameters():
                param.requires_grad = False
        self._frozen_modules = targets
        print(f'freeze sbert: bottom {n}/{total} layers + '
              f'{"embeddings + " if freeze_embeddings else ""}pooler')

    def train(self, mode: bool = True):
        super(ConceptTextEncoder, self).train(mode)
        for module in self._frozen_modules:
            module.eval()
        return self

    def to(self, *args, **kwargs):
        self.sbert = self.sbert.to(*args, **kwargs)
        self._cache = None
        return super(ConceptTextEncoder, self).to(*args, **kwargs)

    def _encode(self, ids: Tensor, mask: Tensor) -> Tensor:
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=ids.is_cuda):
            emb = self.sbert({'input_ids': ids, 'attention_mask': mask})['sentence_embedding']
        return emb.float()

    def forward(self, concept_ids: Tensor) -> Tensor:
        if not self.finetune:
            if self._cache is None:
                self._cache = self.encode_all()
            return self._cache[concept_ids]
        return self._encode(self.input_ids[concept_ids], self.attention_mask[concept_ids])

    def encode_all(self) -> Tensor:
        """全部 concept 的 Q_i（含 padding 行），(concept_num+1, emb_dim)。"""
        feats = []
        with torch.no_grad():
            for i in range(0, self.input_ids.shape[0], 256):
                feats.append(self._encode(self.input_ids[i:i + 256], self.attention_mask[i:i + 256]))
        return torch.cat(feats, dim=0)


class PrecomputedTextEncoder(nn.Module):

    def __init__(self, path: str, num_concept: int):
        super(PrecomputedTextEncoder, self).__init__()
        table = np.load(path)
        if table.ndim != 2:
            raise ValueError(f'{path} 不是二维嵌入表，shape={table.shape}')
        if table.shape[0] < num_concept:
            raise ValueError(f'{path} 只有 {table.shape[0]} 行，少于 concept 数 {num_concept}')

        table = np.asarray(table[:num_concept], dtype=np.float32)
        self.emb_dim = int(table.shape[1])
        pad = np.zeros((1, self.emb_dim), dtype=np.float32)
        # 不写进 state_dict：它是数据不是参数
        self.register_buffer('table', torch.from_numpy(np.vstack([pad, table])), persistent=False)
        print(f'text encoder: precomputed {path} '
              f'(dim={self.emb_dim}, rows={table.shape[0]}+1 pad, frozen)')

    def forward(self, concept_ids: Tensor) -> Tensor:
        return self.table[concept_ids]

    def encode_all(self) -> Tensor:
        """全部 concept 的 Q_i（含 padding 行），(concept_num+1, emb_dim)。"""
        return self.table
