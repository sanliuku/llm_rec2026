"""A-LLMRec 第二阶段的模型本体：冻结 LLM + 两个投影层。
"""

import inspect
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _logits_keep_kwarg(model) -> str:
    """LM head 只算尾部若干位置 logits 的参数名（不同 transformers 版本叫法不同）。"""
    params = inspect.signature(model.forward).parameters
    for name in ('logits_to_keep', 'num_logits_to_keep'):
        if name in params:
            return name
    return ''


def _activation(name: str) -> nn.Module:
    act = {'leakyrelu': nn.LeakyReLU, 'gelu': nn.GELU, 'relu': nn.ReLU,
           'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}.get((name or '').lower())
    if act is None:
        raise ValueError(f'unknown activation: {name}')
    return act()


class ProjLayer(nn.Module):
    """官方 F_U / F_I：Linear(d→H) → LayerNorm(H) → act → Linear(H→H)，两个 Linear 只对 weight 做 xavier。"""

    def __init__(self, input_size: int, hidden_size: int, act: str = 'gelu'):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.act = _activation(act)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        return self.fc2(self.act(self.norm(self.fc1(x))))


class PTReasoner(nn.Module):
    def __init__(self, model_path: str, config: Any, device: str = 'cuda:0',
                 is_lora: bool = False, gradient_checkpointing: bool = True):
        super().__init__()
        self.token_size = config.token_size
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        tokenizer.add_tokens('<user_des>')
        tokenizer.add_tokens('<his_des>')
        tokenizer.add_tokens('<concept_des>')
        tokenizer.padding_side = 'left'

        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        # LLM 只当固定解码器，但梯度要穿过它回传到 inputs_embeds，所以必须开 checkpointing，
        # 否则 24 层激活全留在显存里（这一项就是几 GB 的差别）
        self._logits_keep_kw = _logits_keep_kwarg(model)
        for param in model.parameters():
            param.requires_grad = False

        self.is_lora = is_lora
        if is_lora:
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias='none',
                task_type='CAUSAL_LM',
            ))
        if gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={'use_reentrant': False})
            except TypeError:  # 老版本 transformers 不认这个参数
                model.gradient_checkpointing_enable()
        model.config.use_cache = False

        self.device = device
        self.llm = model
        self.tokenizer = tokenizer
        self.user_token_id = tokenizer.convert_tokens_to_ids('<user_des>')
        self.his_token_id = tokenizer.convert_tokens_to_ids('<his_des>')
        self.concept_token_id = tokenizer.convert_tokens_to_ids('<concept_des>')

        # 三个占位符的原始 embedding 置零，等价于官方「整位替换」
        with torch.no_grad():
            emb = self.llm.get_input_embeddings().weight
            for token_id in (self.user_token_id, self.his_token_id, self.concept_token_id):
                emb[token_id].fill_(0)

        self.embedding_converter_user = ProjLayer(
            config.user_embed_size, config.token_size, config.user_act)
        self.embedding_converter_concept = ProjLayer(
            config.item_embed_size, config.token_size, config.item_act)

    def forward(self, inputs, logits_to_keep: int = 0):
        kwargs = {self._logits_keep_kw: logits_to_keep} if self._logits_keep_kw and logits_to_keep else {}
        return self.llm(inputs_embeds=self.get_embs(inputs),
                        attention_mask=inputs['attention_mask'], **kwargs)

    def get_embs(self, inputs):
        user_embs = self.embedding_converter_user(inputs['user_emb'].to(torch.float32))
        his = inputs['his_embs']
        his_embs = self.embedding_converter_concept(his[torch.any(his != 0, dim=-1)].to(torch.float32))
        # candidate_embs 是 (B, 候选数, D)，要按 batch-major 摊平才和 concept_pos 的行序对得上
        cand = inputs['candidate_embs']
        concept_embs = self.embedding_converter_concept(
            cand.reshape(-1, cand.shape[-1]).to(torch.float32))

        inputs_embs = self.llm.get_input_embeddings()(inputs['input_ids'])
        user_pos = (inputs['input_ids'] == self.user_token_id).nonzero()
        his_pos = (inputs['input_ids'] == self.his_token_id).nonzero()
        concept_pos = (inputs['input_ids'] == self.concept_token_id).nonzero()

        dtype = inputs_embs.dtype
        user_mask = torch.zeros_like(inputs_embs).to(self.device).to(dtype)
        his_mask = torch.zeros_like(inputs_embs).to(self.device).to(dtype)
        concept_mask = torch.zeros_like(inputs_embs).to(self.device).to(dtype)

        user_mask[user_pos[:, 0], user_pos[:, 1]] = user_embs.to(dtype)
        his_mask[his_pos[:, 0], his_pos[:, 1]] = his_embs.to(dtype)
        concept_mask[concept_pos[:, 0], concept_pos[:, 1]] = concept_embs.to(dtype)

        return inputs_embs + user_mask + his_mask + concept_mask

    def save_pretrained(self, output_dir, save_embedding_layers=False):
        if self.is_lora:
            self.llm.save_pretrained(output_dir, save_embedding_layers=save_embedding_layers)
        torch.save(self.embedding_converter_user.state_dict(), f'{output_dir}/user_adaptor.pth')
        torch.save(self.embedding_converter_concept.state_dict(), f'{output_dir}/concept_adaptor.pth')

    def load_model(self, config):
        """只加载两个投影层；LoRA 权重走 PeftModel。"""
        for layer, name in ((self.embedding_converter_user, 'user_adaptor.pth'),
                            (self.embedding_converter_concept, 'concept_adaptor.pth')):
            state = torch.load(f'{config.lora_weights_path}/{name}', map_location=self.device)
            layer.load_state_dict(state)
            layer.to(self.device)

        if not self.is_lora:
            return
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model, torch_dtype=torch.bfloat16, device_map=config.device)
        model.resize_token_embeddings(len(self.tokenizer))
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model = PeftModel.from_pretrained(model, config.lora_weights_path, torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()
        for param in model.parameters():
            param.requires_grad = False
        self.llm = model
        with torch.no_grad():
            emb = self.llm.get_input_embeddings().weight
            for token_id in (self.user_token_id, self.his_token_id, self.concept_token_id):
                emb[token_id].fill_(0)
