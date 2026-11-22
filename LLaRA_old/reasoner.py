import inspect
import os
from typing import Any, List, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from pydantic import BaseModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import math

def _logits_keep_kwarg(model) -> str:
    """LM head 只算尾部若干位置 logits 的参数名（不同 transformers 版本叫法不同）。"""
    params = inspect.signature(model.forward).parameters
    for name in ('logits_to_keep', 'num_logits_to_keep'):
        if name in params:
            return name
    return ''


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"load model !Trainable params: {trainable_params} || All params: {all_param} || Trainable %: {100 * trainable_params / all_param:.1f}%")

class QMoEMlpLayer(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=768):
        super(QMoEMlpLayer, self).__init__()
        # self.gate_proj = nn.Linear(input_size, hidden_dim, bias=False)
        self.up_proj = nn.Linear(input_size, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, output_size, bias=False)
        self.act_fn = nn.GELU()
        # self.act_fn = nn.SiLU()

    def forward(self, x):
        # return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(self.act_fn(self.up_proj(x)))

class LlaRA(nn.Module):
    def __init__(
        self,
        model_path: str,
        config: Any,
        device: str = 'cpu',
        is_lora: bool = False,
    ):
        super().__init__()
        self.token_size = config.token_size
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        tokenizer.add_tokens("<his_des>")
        tokenizer.add_tokens("<concept_des>")
        tokenizer.add_tokens("<PH>")
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        # LM head 只算尾部若干位置 logits 的参数名（老版本 transformers 可能没有）
        self._logits_keep_kw = _logits_keep_kwarg(model)
        # freeze parameters in llama
        for param in model.parameters():
            param.requires_grad = False

        self.is_lora = is_lora
        if is_lora:
            model = prepare_model_for_kbit_training(model)
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)

        print_trainable_parameters(model)
        model.config.use_cache = False

        self.device = device
        self.llm = model
        self.tokenizer = tokenizer

        self.soft_token_id = tokenizer.convert_tokens_to_ids("<PH>")
        self.his_token_id = tokenizer.convert_tokens_to_ids("<his_des>")
        self.concept_token_id = tokenizer.convert_tokens_to_ids("<concept_des>")

        with torch.no_grad():
            self.llm.get_input_embeddings().weight[self.soft_token_id].fill_(0)
            self.llm.get_input_embeddings().weight[self.his_token_id].fill_(0)
            self.llm.get_input_embeddings().weight[self.concept_token_id].fill_(0)

        #### load moe layer
        self.embedding_converter_concept = QMoEMlpLayer(input_size=config.embed_size, output_size=config.token_size, hidden_dim=config.hidden_size)

    def forward(self, inputs, logits_to_keep: int = 0):
        inputs_embs = self.get_embs(inputs)
        attention_mask = inputs['attention_mask']
        kwargs = {self._logits_keep_kw: logits_to_keep} if self._logits_keep_kw and logits_to_keep else {}
        return self.llm(inputs_embeds=inputs_embs, attention_mask=attention_mask, **kwargs)

    def get_embs(self, inputs):
        # fast stable
        if "his_embs" not in inputs or "candidate_embs" not in inputs:
            return self.llm.get_input_embeddings()(inputs['input_ids'])

        his_embs = inputs["his_embs"]
        non_zero_rows = torch.any(his_embs != 0, dim=-1)
        his_embs = his_embs[non_zero_rows]
        his_embs = self.embedding_converter_concept(his_embs.to(torch.float32)).view(-1, self.token_size)
        concept_embs = inputs["candidate_embs"]
        concept_embs = self.embedding_converter_concept(concept_embs.to(torch.float32)).view(-1, self.token_size)

        inputs_embs = self.llm.get_input_embeddings()(inputs['input_ids'])
        his_emb_position = (inputs['input_ids'] == self.his_token_id).nonzero()
        concept_emb_position = (inputs['input_ids'] == self.concept_token_id).nonzero()
        if his_emb_position.shape[0] == 0 or concept_emb_position.shape[0] == 0:
            return inputs_embs

        his_emb_mask = torch.zeros_like(inputs_embs).to(self.device).to(inputs_embs.dtype)
        concept_emb_mask = torch.zeros_like(inputs_embs).to(self.device).to(inputs_embs.dtype)

        his_embs = his_embs.to(inputs_embs.dtype)
        concept_embs = concept_embs.to(inputs_embs.dtype)

        his_emb_mask[his_emb_position[:, 0], his_emb_position[:, 1]] = his_embs
        concept_emb_mask[concept_emb_position[:, 0], concept_emb_position[:, 1]] = concept_embs

        inputs_embs_new = inputs_embs + his_emb_mask + concept_emb_mask
        return inputs_embs_new

    def save_pretrained(self, output_dir, save_embedding_layers=False):
        if self.is_lora:
            self.llm.save_pretrained(output_dir, save_embedding_layers=save_embedding_layers)
        torch.save(self.embedding_converter_concept.state_dict(), output_dir+"/"+'concept_moe_adaptor.pth')

    def load_model(self, config, is_train=False):
        self.embedding_converter_concept.load_state_dict(torch.load(config.lora_weights_path+"/"+'concept_moe_adaptor.pth'))
        self.embedding_converter_concept.to(self.device)
        if self.is_lora:
            self.llm = None
            if is_train:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    config.base_model,
                    quantization_config=bnb_config,
                    device_map=self.device,
                    torch_dtype=torch.bfloat16,
                )
                model.resize_token_embeddings(len(self.tokenizer))
                model.config.pad_token_id = self.tokenizer.pad_token_id
                model = prepare_model_for_kbit_training(model)
                model = PeftModel.from_pretrained(model, config.lora_weights_path, is_trainable=True)

            else:
                model = AutoModelForCausalLM.from_pretrained(
                    config.base_model,
                    device_map=self.device,
                    torch_dtype=torch.bfloat16,
                )
                model.resize_token_embeddings(len(self.tokenizer))
                model.config.pad_token_id = self.tokenizer.pad_token_id
                model = PeftModel.from_pretrained(model, config.lora_weights_path)
                model.merge_and_unload()
                model.generation_config.cache_implementation = "static"
                model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                for name, param in model.named_parameters():
                    # freeze base model's layers
                    param.requires_grad = False

            model.config.use_cache = False
            print_trainable_parameters(model)
            self.llm=model

        with torch.no_grad():
            self.llm.get_input_embeddings().weight[self.soft_token_id].fill_(0)
            self.llm.get_input_embeddings().weight[self.his_token_id].fill_(0)
            self.llm.get_input_embeddings().weight[self.concept_token_id].fill_(0)