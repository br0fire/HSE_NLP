import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    BertConfig, BertForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from datasets import load_dataset
from seqeval.metrics import f1_score
import numpy as np



dataset = load_dataset("conll2003", trust_remote_code=True)
dataset = dataset.remove_columns(["id", "pos_tags", "chunk_tags"])

label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

split = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
temp_dataset = split["test"]
split_val_test = temp_dataset.train_test_split(test_size=0.5, seed=42)
val_dataset = split_val_test["train"]
test_dataset = split_val_test["test"]


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def align_labels_with_tokens(labels, word_ids):
    previous_word_idx = None
    aligned_labels = []
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            aligned_labels.append(labels[word_idx])
        else:
            label_id = labels[word_idx]
            label_name = label_names[label_id]
            if label_name.startswith("B-"):
                label_name = label_name.replace("B-", "I-")
                label_id = label_names.index(label_name)
            aligned_labels.append(label_id)
        previous_word_idx = word_idx
    return aligned_labels


def tokenize_and_align_labels(batch):
    tokenized_inputs = tokenizer(
        batch["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=128
    )
    labels = [
        align_labels_with_tokens(l, tokenized_inputs.word_ids(batch_index=i))
        for i, l in enumerate(batch["ner_tags"])
    ]
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_preds = [
        [label_names[pred] for (pred, lab) in zip(pred_row, label_row) if lab != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]
    return {"f1": f1_score(true_labels, true_preds)}



train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)



teacher_ckpt = "./bert-ner/checkpoint-132"
teacher = AutoModelForTokenClassification.from_pretrained(
    teacher_ckpt,
    num_labels=len(label_names)
).eval()

for p in teacher.parameters():
    p.requires_grad = False


student_cfg = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=256,             
    num_hidden_layers=4,       
    num_attention_heads=4,       
    intermediate_size=1024, 
    max_position_embeddings=512,
    type_vocab_size=2,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)
student_cfg.num_labels = len(label_names)
student = BertForTokenClassification(student_cfg)



def count_params(m): return sum(p.numel() for p in m.parameters())
print(f"Student params: {count_params(student):,}")


class KDTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature=2.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.temperature = temperature
        self.alpha = alpha
        self.kldiv = nn.KLDivLoss(reduction="batchmean")
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs_s = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits_s = outputs_s.logits

        with torch.no_grad():
            outputs_t = self.teacher(**{k: v for k, v in inputs.items() if k != "labels"})
            logits_t = outputs_t.logits

        T = self.temperature

        kd_loss = self.kldiv(
            F.log_softmax(logits_s / T, dim=-1),
            F.softmax(logits_t / T, dim=-1)
        ) * (T * T)

        ce_loss = self.ce(logits_s.view(-1, logits_s.size(-1)), labels.view(-1))
        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        return (loss, outputs_s) if return_outputs else loss



training_args = TrainingArguments(
    output_dir="./student-ner-kd",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-4,                
    num_train_epochs=20,               
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_ratio=0.1,                  
    bf16=True,                       
    logging_steps=100,
    report_to="tensorboard",
    logging_dir="./logs_student_kd",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher.to(device)
student.to(device)


trainer = KDTrainer(
    model=student,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
    teacher_model=teacher,
    temperature=2.0,   
    alpha=0.5         
)

trainer.train()


test_metrics = trainer.evaluate(test_dataset)
print({k: round(v, 4) for k, v in test_metrics.items()})
print("Final TEST F1:", round(test_metrics["eval_f1"], 4))
