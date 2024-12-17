import os
import wandb
import torch
from datasets import load_dataset

from transformers import (
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score
import torch

# ----------------------------
# Initialize Hyperparameters
# ----------------------------
MODEL_NAME = "gpt2-xl"
DATASET_NAME = "piqa"
OUTPUT_DIR = "./gpt2_piqa_lora"
NUM_LABELS = 2
MAX_LENGTH = 256
LEARNING_RATE = 3e-6
BATCH_SIZE = 8
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = [
    "transformer.h.0.attn.c_attn", 
    "transformer.h.0.mlp.c_fc"
]

# ----------------------------
# Load Tokenizer and Dataset
# ----------------------------
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

# Load dataset
dataset = load_dataset(DATASET_NAME)

# ----------------------------
# Preprocess Dataset
# ----------------------------
def preprocess_function(examples):
    """Preprocess the dataset for multiple choice tasks."""
    # Duplicate the goal for each solution pair
    first_sentences = [[context] * 2 for context in examples["goal"]]
    second_sentences = [
        [examples["sol1"][i], examples["sol2"][i]] for i in range(len(examples["sol1"]))
    ]
    labels = examples["label"]  # Labels should remain flat

    # Flatten lists for tokenization
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    
    # Tokenize inputs
    inputs = tokenizer(
        first_sentences, 
        second_sentences, 
        truncation=True, 
        padding="max_length", 
        max_length=MAX_LENGTH
    )
    
    # Group inputs into pairs for multiple-choice format
    inputs = {k: [v[i : i + 2] for i in range(0, len(v), 2)] for k, v in inputs.items()}
    
    # Add labels
    inputs["labels"] = labels
    return inputs

# Tokenize dataset
tokenized_datasets = dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=dataset["train"].column_names
)

# Convert datasets to torch format
tokenized_datasets = tokenized_datasets.with_format("torch")

# ----------------------------
# Load Model with LoRA
# ----------------------------
model = GPT2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.config.pad_token_id = tokenizer.pad_token_id

# Apply LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
)
model = get_peft_model(model, lora_config)

# ----------------------------
# Training Arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="constant",
    warmup_ratio=0.0,
    max_grad_norm=0.3,    
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="wandb",  # Log to W&B
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    gradient_checkpointing=True,
    no_cuda=not torch.cuda.is_available(),
)

# ----------------------------
# Metrics Calculation
# ----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# ----------------------------
# Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ----------------------------
# Train and Evaluate the Model
# ----------------------------
trainer.train()
metrics = trainer.evaluate(tokenized_datasets["validation"])
print("Evaluation Metrics:", metrics)

# ----------------------------
# Save Model
# ----------------------------
trainer.save_model(OUTPUT_DIR)
print("Model saved to:", OUTPUT_DIR)
