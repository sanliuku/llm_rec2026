"""A-LLMRec 第二阶段的模型：冻结 LLM + LoRA + 三个占位符的适配器。
"""

from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params} || All params: {all_param} "
          f"|| Trainable %: {100 * trainable_params / all_param:.1f}%")


class Adapter(nn.Module):
    """单隐层的投影层：up_proj -> act -> down_proj，无 bias（= 你的 QMoEMlpLayer）。

    官方 F_U / F_I 是 Linear -> LayerNorm -> LeakyReLU/GELU -> Linear；这里省掉 LayerNorm，
    激活可选 sigmoid（你的写法，默认）/ gelu（LLaRA 的写法）。
    """

    def __init__(self, input_size: int, output_size: int, hidden_dim: int, act: str = 'sigmoid'):
        super(Adapter, self).__init__()
        self.up_proj = nn.Linear(input_size, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, output_size, bias=False)
        self.act_fn = nn.Sigmoid() if act == 'sigmoid' else nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class ALLMRecReasoner(nn.Module):
    """把 x^u / e_i 投影成 token 向量，替换掉 prompt 里的三个占位符后送进 LLM。"""

    def __init__(self, model_path: str, config: Any, device: str = 'cpu', is_lora: bool = True):
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
        tokenizer.add_tokens("<user_des>")
        tokenizer.add_tokens("<his_des>")
        tokenizer.add_tokens("<concept_des>")
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
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
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        print_trainable_parameters(model)
        model.config.use_cache = False

        hidden_size = getattr(model.config, 'hidden_size', None)
        if hidden_size is not None and hidden_size != config.token_size:
            raise ValueError(f'token_size={config.token_size} 与底座 hidden_size={hidden_size} 不一致，'
                             f'适配器输出必须等于 LLM 的 hidden')

        self.device = device
        self.llm = model
        self.tokenizer = tokenizer
        self.user_token_id = tokenizer.convert_tokens_to_ids("<user_des>")
        self.his_token_id = tokenizer.convert_tokens_to_ids("<his_des>")
        self.concept_token_id = tokenizer.convert_tokens_to_ids("<concept_des>")

        # 占位符本身的词嵌入置零，后面用「零掩码相加」把适配器输出写到对应位置
        with torch.no_grad():
            emb = self.llm.get_input_embeddings().weight
            emb[self.user_token_id].fill_(0)
            emb[self.his_token_id].fill_(0)
            emb[self.concept_token_id].fill_(0)

        self.user_proj = Adapter(config.user_embed_size, config.token_size,
                                 config.adapter_hidden, config.adapter_act)
        self.item_proj = Adapter(config.item_embed_size, config.token_size,
                                 config.adapter_hidden, config.adapter_act)

    def forward(self, inputs):
        inputs_embs = self.get_embs(inputs)
        return self.llm(inputs_embeds=inputs_embs, attention_mask=inputs['attention_mask'])

    def get_embs(self, inputs):
        """input_ids 里的占位符位置，用适配器输出覆盖掉（其余位置保持原词嵌入）。"""
        inputs_embs = self.llm.get_input_embeddings()(inputs['input_ids'])
        positions = {
            self.user_token_id: (inputs.get('user_emb'), self.user_proj),
            self.his_token_id: (inputs.get('his_embs'), self.item_proj),
            self.concept_token_id: (inputs.get('candidate_embs'), self.item_proj),
        }

        replaced = torch.zeros_like(inputs_embs)
        for token_id, (raw, proj) in positions.items():
            if raw is None:
                continue
            where = (inputs['input_ids'] == token_id).nonzero()
            if where.shape[0] == 0:
                continue
            # 历史侧 batch 内做过左补零，整行为 0 的是 padding，要剔掉
            raw = raw.reshape(-1, raw.shape[-1])
            if raw.shape[0] != where.shape[0]:
                raw = raw[torch.any(raw != 0, dim=-1)]
            if raw.shape[0] != where.shape[0]:
                raise ValueError(
                    f'占位符数量 {where.shape[0]} 与向量行数 {raw.shape[0]} 不一致（token={token_id}）')
            vec = proj(raw.to(torch.float32)).view(-1, self.token_size).to(inputs_embs.dtype)
            replaced[where[:, 0], where[:, 1]] = vec

        return inputs_embs + replaced

    # ------------------------------------------------------------------ 存取
    def save_pretrained(self, output_dir, save_embedding_layers=False):
        if self.is_lora:
            self.llm.save_pretrained(output_dir, save_embedding_layers=save_embedding_layers)
        torch.save(self.user_proj.state_dict(), output_dir + "/" + 'user_adaptor.pth')
        torch.save(self.item_proj.state_dict(), output_dir + "/" + 'item_adaptor.pth')

    def load_model(self, config, merge: bool = True):
        """加载训练好的 LoRA + 适配器；merge=True 时把 LoRA 合进底座（推理用）。"""
        self.user_proj.load_state_dict(torch.load(config.lora_weights_path + "/" + 'user_adaptor.pth'))
        self.item_proj.load_state_dict(torch.load(config.lora_weights_path + "/" + 'item_adaptor.pth'))
        self.user_proj.to(self.device)
        self.item_proj.to(self.device)

        if not self.is_lora:
            return
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model, torch_dtype=torch.bfloat16, device_map=config.device)
        model.resize_token_embeddings(len(self.tokenizer))
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model = PeftModel.from_pretrained(model, config.lora_weights_path, torch_dtype=torch.bfloat16)
        if merge:
            model = model.merge_and_unload()
        model.config.use_cache = True
        for param in model.parameters():
            param.requires_grad = False
        self.llm = model

        with torch.no_grad():
            emb = self.llm.get_input_embeddings().weight
            emb[self.user_token_id].fill_(0)
            emb[self.his_token_id].fill_(0)
            emb[self.concept_token_id].fill_(0)
