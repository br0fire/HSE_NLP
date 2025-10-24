from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
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

train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)


model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased", num_labels=len(label_names)
)

print("Число параметров:", sum(p.numel() for p in model.parameters()))


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_preds = [
        [label_names[pred] for (pred, lab) in zip(pred_row, label_row) if lab != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]
    return {"f1": f1_score(true_labels, true_preds)}


training_args = TrainingArguments(
    output_dir="./bert-ner",
    eval_strategy="epoch",
    save_strategy="epoch",
    overwrite_output_dir=True,
    learning_rate=3e-4,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,
    bf16=True,                         
    logging_steps=50,                 
    logging_dir="./logs",               
    report_to="tensorboard",        
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
)

    
# 8️⃣ Тренировка
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

# 9️⃣ Оценка на тестовом сете (честная)
metrics = trainer.evaluate(test_dataset)
print(f"\nFinal test F1: {metrics['eval_f1']:.4f}")
