import copy
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Any

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase,
)

import os
import torch
import torch.nn as nn
from datasets import load_dataset
from safetensors.torch import load_file

def prepare_hh_dpo_dataset(tokenizer, max_samples=None, max_length=512, num_proc=8):
    ds = load_dataset("Anthropic/hh-rlhf", split="train")

    def split_prompt_chosen_rejected(example):
        marker = "Assistant:"

        chosen = example["chosen"]
        rejected = example["rejected"]

        idx_c = chosen.rfind(marker)
        idx_r = rejected.rfind(marker)
        if idx_c == -1 or idx_r == -1:
            return {"prompt": None, "chosen": None, "rejected": None}

        prompt_c = chosen[:idx_c].strip()
        prompt_r = rejected[:idx_r].strip()

        # В HH prompt совпадает, но на всякий
        if prompt_c != prompt_r:
            return {"prompt": None, "chosen": None, "rejected": None}

        prompt = prompt_c
        chosen_resp = chosen[idx_c + len(marker):].strip()
        rejected_resp = rejected[idx_r + len(marker):].strip()

        return {
            "prompt": prompt,
            "chosen": chosen_resp,
            "rejected": rejected_resp,
        }

    ds = ds.map(split_prompt_chosen_rejected, num_proc=num_proc)
    ds = ds.filter(lambda x: x["prompt"] is not None)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    def tokenize_dpo(example):

        prompt_text = example["prompt"] + "\nAssistant:"

        chosen_text = example["chosen"]
        rejected_text = example["rejected"]

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        chosen_ids = tokenizer(chosen_text, add_special_tokens=False)["input_ids"]
        rejected_ids = tokenizer(rejected_text, add_special_tokens=False)["input_ids"]

        chosen_full = prompt_ids + chosen_ids
        rejected_full = prompt_ids + rejected_ids

        chosen_full = chosen_full[:max_length]
        rejected_full = rejected_full[:max_length]

        chosen_attn = [1] * len(chosen_full)
        rejected_attn = [1] * len(rejected_full)

        prompt_len = len(prompt_ids)
        chosen_answer_mask = [0] * min(prompt_len, len(chosen_full))
        chosen_answer_mask += [1] * max(0, len(chosen_full) - len(chosen_answer_mask))
        chosen_answer_mask = chosen_answer_mask[:len(chosen_full)]

        rejected_answer_mask = [0] * min(prompt_len, len(rejected_full))
        rejected_answer_mask += [1] * max(0, len(rejected_full) - len(rejected_answer_mask))
        rejected_answer_mask = rejected_answer_mask[:len(rejected_full)]

        return {
            "chosen_input_ids": chosen_full,
            "chosen_attention_mask": chosen_attn,
            "chosen_answer_mask": chosen_answer_mask,
            "rejected_input_ids": rejected_full,
            "rejected_attention_mask": rejected_attn,
            "rejected_answer_mask": rejected_answer_mask,
        }

    ds = ds.map(
        tokenize_dpo,
        num_proc=num_proc,
        remove_columns=ds.column_names,
    )

    return ds

@dataclass
class DPODataCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}

        for key in [
            "chosen_input_ids",
            "chosen_attention_mask",
            "chosen_answer_mask",
            "rejected_input_ids",
            "rejected_attention_mask",
            "rejected_answer_mask",
        ]:
            sequences = [torch.tensor(f[key], dtype=torch.long) for f in features]
            # pad до max длины в батче
            padded = torch.nn.utils.rnn.pad_sequence(
                sequences,
                batch_first=True,
                padding_value=0,
            )
            batch[key] = padded

        return batch

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        in_f = base_layer.in_features
        out_f = base_layer.out_features

        device = base_layer.weight.device

        self.lora_A = nn.Parameter(torch.randn(in_f, r, device=device) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_f, device=device))


        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):

        out = self.base(x)


        lora = x @ self.lora_A @ self.lora_B

        return out + self.scale * lora



def add_lora_to_neox(model, r=8, alpha=16, target_mlp=False):
    for layer_idx, layer in enumerate(model.gpt_neox.layers):
        

        qkv = layer.attention.query_key_value
        layer.attention.query_key_value = LoRALinear(qkv, r=r, alpha=alpha)


        dense = layer.attention.dense
        layer.attention.dense = LoRALinear(dense, r=r, alpha=alpha)


        if target_mlp:
            h4 = layer.mlp.dense_h_to_4h
            h = layer.mlp.dense_4h_to_h

            layer.mlp.dense_h_to_4h = LoRALinear(h4, r=r, alpha=alpha)
            layer.mlp.dense_4h_to_h = LoRALinear(h, r=r, alpha=alpha)


    for p in model.parameters():
        p.requires_grad = False


    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True

    return model


class DPOTrainer(Trainer):
    def __init__(self, ref_model, beta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.ref_model = ref_model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.to(self.args.device)

    @staticmethod
    def _logps_from_logits(logits, input_ids, answer_mask):

        # сдвигаем, как в LM: предсказываем токен t+1 из позиции t
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        answer_mask = answer_mask[:, 1:]

        logprobs = F.log_softmax(logits, dim=-1)
        token_logprobs = logprobs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

        # Маскируем только токены ответа
        token_logprobs = token_logprobs * answer_mask

        # лог-правдоподобие ответа = сумма по токенам ответа
        return token_logprobs.sum(dim=-1)  # [B]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        c_ids = inputs["chosen_input_ids"]
        c_attn = inputs["chosen_attention_mask"]
        c_ans_mask = inputs["chosen_answer_mask"]

        r_ids = inputs["rejected_input_ids"]
        r_attn = inputs["rejected_attention_mask"]
        r_ans_mask = inputs["rejected_answer_mask"]

        # Policy модель
        policy_chosen = model(
            input_ids=c_ids,
            attention_mask=c_attn,
            use_cache=False,
        )
        policy_rejected = model(
            input_ids=r_ids,
            attention_mask=r_attn,
            use_cache=False,
        )

        # Reference (замороженная SFT модель)
        with torch.no_grad():
            ref_chosen = self.ref_model(
                input_ids=c_ids,
                attention_mask=c_attn,
                use_cache=False,
            )
            ref_rejected = self.ref_model(
                input_ids=r_ids,
                attention_mask=r_attn,
                use_cache=False,
            )

        log_pi_c = self._logps_from_logits(policy_chosen.logits, c_ids, c_ans_mask)
        log_pi_r = self._logps_from_logits(policy_rejected.logits, r_ids, r_ans_mask)

        log_pref_c = self._logps_from_logits(ref_chosen.logits, c_ids, c_ans_mask)
        log_pref_r = self._logps_from_logits(ref_rejected.logits, r_ids, r_ans_mask)


        pi_logratios = log_pi_c - log_pi_r

        ref_logratios = log_pref_c - log_pref_r


        beta = self.beta
        dpo_logits = beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(dpo_logits).mean()

        if return_outputs:
            return loss, {
                "log_pi_c": log_pi_c.detach(),
                "log_pi_r": log_pi_r.detach(),
                "log_pref_c": log_pref_c.detach(),
                "log_pref_r": log_pref_r.detach(),
            }
        return loss




def main_dpo():
    model_name = "EleutherAI/pythia-1.4b"
    sft_ckpt_dir = "./pythia-1.4b-hh-lora"
    dpo_output_dir = "./pythia-1.4b-hh-dpo"

    max_length = 512
    num_proc = 8
    max_samples = None  


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

 
    print("Preparing DPO dataset...")
    dpo_ds = prepare_hh_dpo_dataset(
        tokenizer=tokenizer,
        max_samples=max_samples,
        max_length=max_length,
        num_proc=num_proc,
    )

    data_collator = DPODataCollator(tokenizer=tokenizer)


    print("Loading base model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    )


    policy_model = add_lora_to_neox(policy_model, r=8, alpha=16, target_mlp=False)
    policy_model.to(torch.bfloat16)


    print("Loading SFT LoRA checkpoint...")
    state = load_file(f"{sft_ckpt_dir}/model.safetensors")
    missing, unexpected = policy_model.load_state_dict(state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)


    print("Cloning reference model...")
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    training_args = TrainingArguments(
        output_dir=dpo_output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        weight_decay=0.0,
        warmup_ratio=0.03,
        logging_steps=50,
        save_steps=10_000,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        report_to="none",
        optim="adamw_torch_fused",
        ddp_find_unused_parameters=False,
    
        remove_unused_columns=False,
    )


    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        beta=0.1,
        args=training_args,
        train_dataset=dpo_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    print("Saving DPO model to", dpo_output_dir)
    trainer.save_model(dpo_output_dir)
    tokenizer.save_pretrained(dpo_output_dir)

if __name__ == "__main__":
    main_dpo()

