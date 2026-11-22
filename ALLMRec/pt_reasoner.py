import os
from typing import Any, List, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from pydantic import BaseModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params} || All params: {all_param} || Trainable %: {100 * trainable_params / all_param:.1f}%")

class QMoEMlpLayer(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=768):
        super(QMoEMlpLayer, self).__init__()
        # self.gate_proj = nn.Linear(input_size, hidden_dim, bias=False)
        self.up_proj = nn.Linear(input_size, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, output_size, bias=False)
        # self.act_fn = nn.GELU()
        self.act_fn = nn.Sigmoid()

    def forward(self, x):
        # return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(self.act_fn(self.up_proj(x)))

class QMoEAdapterLayer(nn.Module):
    def __init__(self, n_experts=8, input_size=256, output_size=1536, dropout=0.1, top_k=2):
        super(QMoEAdapterLayer, self).__init__()
        self.num_experts = n_experts
        self.top_k = top_k
        self.norm_topk_prob = False
        self.input_size = input_size
        self.output_size = output_size
        self.experts = nn.ModuleList([QMoEMlpLayer(input_size, output_size) for _ in range(n_experts)])
        self.gate = nn.Linear(input_size, n_experts, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, hidden_dim = x.size()
        sequence_length = 1
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, self.output_size), dtype=x.dtype, device=x.device
        )
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = x[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))
        if self.training:
            final_hidden_states = self.dropout(final_hidden_states)
        return final_hidden_states

class PTReasoner(nn.Module):
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
        tokenizer.add_tokens("<user_des>")
        tokenizer.add_tokens("<his_des>")
        tokenizer.add_tokens("<concept_des>")
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
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
        self.user_token_id = tokenizer.convert_tokens_to_ids("<user_des>")
        self.his_token_id = tokenizer.convert_tokens_to_ids("<his_des>")
        self.concept_token_id = tokenizer.convert_tokens_to_ids("<concept_des>")

        with torch.no_grad():
            self.llm.get_input_embeddings().weight[self.user_token_id].fill_(0)
            self.llm.get_input_embeddings().weight[self.his_token_id].fill_(0)
            self.llm.get_input_embeddings().weight[self.concept_token_id].fill_(0)


        #### load moe layer
        self.embedding_converter_user = QMoEMlpLayer(input_size=config.embed_size, output_size=config.token_size, hidden_dim=config.hidden_size)
        self.embedding_converter_concept = QMoEMlpLayer(input_size=config.embed_size, output_size=config.token_size, hidden_dim=config.hidden_size)


    def forward(self, inputs):
        inputs_embs = self.get_embs(inputs)
        attention_mask = inputs['attention_mask']
        return self.llm(inputs_embeds=inputs_embs, attention_mask=attention_mask)

    def get_embs(self, inputs):
        # fast stable
        user_embs = inputs["user_emb"]
        user_embs = self.embedding_converter_user(user_embs.to(torch.float32)).view(-1, self.token_size)
        his_embs = inputs["his_embs"]
        non_zero_rows = torch.any(his_embs != 0, dim=-1)
        his_embs = his_embs[non_zero_rows]
        his_embs = self.embedding_converter_concept(his_embs.to(torch.float32)).view(-1, self.token_size)
        concept_embs = inputs["candidate_embs"]
        concept_embs = self.embedding_converter_concept(concept_embs.to(torch.float32)).view(-1, self.token_size)

        inputs_embs = self.llm.get_input_embeddings()(inputs['input_ids'])
        user_emb_position = (inputs['input_ids'] == self.user_token_id).nonzero()
        his_emb_position = (inputs['input_ids'] == self.his_token_id).nonzero()
        concept_emb_position = (inputs['input_ids'] == self.concept_token_id).nonzero()

        user_emb_mask = torch.zeros_like(inputs_embs).to(self.device).to(inputs_embs.dtype)
        his_emb_mask = torch.zeros_like(inputs_embs).to(self.device).to(inputs_embs.dtype)
        concept_emb_mask = torch.zeros_like(inputs_embs).to(self.device).to(inputs_embs.dtype)

        user_embs = user_embs.to(inputs_embs.dtype)
        his_embs = his_embs.to(inputs_embs.dtype)
        concept_embs = concept_embs.to(inputs_embs.dtype)

        user_emb_mask[user_emb_position[:, 0], user_emb_position[:, 1]] = user_embs
        his_emb_mask[his_emb_position[:, 0], his_emb_position[:, 1]] = his_embs
        concept_emb_mask[concept_emb_position[:, 0], concept_emb_position[:, 1]] = concept_embs

        inputs_embs_new = inputs_embs + user_emb_mask + his_emb_mask + concept_emb_mask

        return inputs_embs_new

    def save_pretrained(self, output_dir, save_embedding_layers=False):
        if self.is_lora:
            self.llm.save_pretrained(output_dir, save_embedding_layers=save_embedding_layers)
        torch.save(self.embedding_converter_user.state_dict(), output_dir+"/"+'user_moe_adaptor.pth')
        torch.save(self.embedding_converter_concept.state_dict(), output_dir+"/"+'concept_moe_adaptor.pth')

    def load_model(self, config):
        self.embedding_converter_user.load_state_dict(torch.load(config.lora_weights_path+"/"+'user_moe_adaptor.pth'))
        self.embedding_converter_concept.load_state_dict(torch.load(config.lora_weights_path+"/"+'concept_moe_adaptor.pth'))
        self.embedding_converter_user.to(self.device)
        self.embedding_converter_concept.to(self.device)
        if self.is_lora:
            self.llm = None
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                torch_dtype=torch.bfloat16,
                device_map=config.device,
            )
            model.resize_token_embeddings(len(self.tokenizer))
            model.config.pad_token_id = self.tokenizer.pad_token_id
            model = PeftModel.from_pretrained(model, config.lora_weights_path, torch_dtype=torch.bfloat16)
            model.merge_and_unload()
            model.generation_config.cache_implementation = "static"
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
            for name, param in model.named_parameters():
                # freeze base model's layers
                param.requires_grad = False
            print_trainable_parameters(model)
            self.llm=model

            with torch.no_grad():
                self.llm.get_input_embeddings().weight[self.user_token_id].fill_(0)
                self.llm.get_input_embeddings().weight[self.his_token_id].fill_(0)
                self.llm.get_input_embeddings().weight[self.concept_token_id].fill_(0)