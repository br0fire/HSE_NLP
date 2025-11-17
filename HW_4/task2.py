import os
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

def prepare_hh_dataset(max_samples=None, max_length=512, num_proc=8):
    ds = load_dataset("Anthropic/hh-rlhf", split="train")

    def split_prompt_response(example):
        text = example["chosen"]
        marker = "Assistant:"
        idx = text.rfind(marker)
        if idx == -1:
            return {"prompt": None, "response": None}

        prompt = text[:idx].strip()
        response = text[idx + len(marker):].strip()
        return {"prompt": prompt, "response": response}

    ds = ds.map(split_prompt_response, num_proc=num_proc)
    ds = ds.filter(lambda x: x["prompt"] is not None and x["response"] is not None)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    return ds



def tokenize_dataset(ds, tokenizer, max_length=512, num_proc=8):
    assistant_prefix = "\nAssistant: "

    def tokenize_fn(example):
        prompt = example["prompt"] + assistant_prefix
        answer = example["response"]


        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]


        input_ids = prompt_ids + answer_ids
        input_ids = input_ids[:max_length]


        labels = [-100] * len(prompt_ids) + answer_ids
        labels = labels[:max_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    tokenized = ds.map(
        tokenize_fn,
        remove_columns=ds.column_names,
        num_proc=num_proc,
    )

    return tokenized




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


def print_trainable_params(model, prefix=""):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{prefix}Total params:     {total:,}")
    print(f"{prefix}Trainable params: {trainable:,}")
    print(f"{prefix}Percent:          {100*trainable/total:.4f}%")


# ---------------------------
# 4. Main training entry
# ---------------------------

def main():

    from transformers import set_seed
    set_seed(42)

    model_name = "EleutherAI/pythia-1.4b"
    max_length = 512
    num_proc = 8 
    max_samples = None 


    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main_process = (local_rank == 0) or (local_rank == -1)

    if is_main_process:
        print("Loading dataset...")
    ds = prepare_hh_dataset(max_samples=max_samples, max_length=max_length, num_proc=num_proc)

    if is_main_process:
        print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main_process:
        print("Tokenizing dataset...")
    tokenized = tokenize_dataset(ds, tokenizer, max_length=max_length, num_proc=num_proc)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=512,
        label_pad_token_id=-100,
    )
    

    if is_main_process:
        print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16, 
    )


    model = add_lora_to_neox(model, r=8, alpha=16, target_mlp=False)

    if is_main_process:
        print_trainable_params(model, prefix="[LoRA] ")



    output_dir = "./pythia-1.4b-hh-lora"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=32,  
        gradient_accumulation_steps=1,   
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=50,
        save_steps=10_000,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        gradient_checkpointing=False,
        optim="adamw_torch_fused",
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    if is_main_process:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Training done. Model saved to", output_dir)


if __name__ == "__main__":
    main()
