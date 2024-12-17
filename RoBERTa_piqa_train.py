import os
import wandb
import torch
from datasets import load_dataset
from transformers import (
    RobertaForMultipleChoice,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score

# ----------------------------
# Hyperparameters Initialization
# ----------------------------
MODEL_NAME = "roberta-large"
DATASET_NAME = "piqa"
OUTPUT_DIR = "./roberta_piqa_lora"
MAX_LENGTH = 256
LEARNING_RATE = 3e-6
BATCH_SIZE = 8
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = [
    "roberta.encoder.layer.0.attention.self.query", 
    "roberta.encoder.layer.0.attention.self.value"
]

# ----------------------------
# Load Tokenizer and Dataset
# ----------------------------
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

# Load PIQA dataset
dataset = load_dataset(DATASET_NAME)

# ----------------------------
# Preprocessing Function
# ----------------------------
def preprocess_function(examples):
    """Preprocess dataset for multiple choice tasks."""
    first_sentences = [[context] * 2 for context in examples["goal"]]
    second_sentences = [
        [examples["sol1"][i], examples["sol2"][i]] for i in range(len(examples["sol1"]))
    ]
    labels = examples["label"]

    # Flatten inputs
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize inputs
    inputs = tokenizer(
        first_sentences, second_sentences, 
        truncation=True, padding="max_length", max_length=MAX_LENGTH
    )

    # Reshape inputs for multiple-choice format
    inputs = {k: [v[i : i + 2] for i in range(0, len(v), 2)] for k, v in inputs.items()}
    inputs["labels"] = labels
    return inputs

# Tokenize and preprocess the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True, 
                                 remove_columns=dataset["train"].column_names)
tokenized_datasets = tokenized_datasets.with_format("torch")

# ----------------------------
# Model Initialization with LoRA
# ----------------------------
def print_model_parameters(model):
    """Print model architecture and parameter counts."""
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

# Load RoBERTa model
model = RobertaForMultipleChoice.from_pretrained(MODEL_NAME)
print("Original Model:")
print_model_parameters(model)

# Apply LoRA for parameter-efficient fine-tuning
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
)
model = get_peft_model(model, lora_config)

print("Model with LoRA Applied:")
print_model_parameters(model)

# ----------------------------
# Training Configuration
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
    report_to="wandb",  # Log to Weights & Biases
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    gradient_checkpointing=True,
    no_cuda=not torch.cuda.is_available(),
)

# ----------------------------
# Metrics Calculation
# ----------------------------
def compute_metrics(eval_pred):
    """Calculate accuracy for predictions."""
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# ----------------------------
# Trainer Initialization and Training
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting Training...")
trainer.train()

# Evaluate the fine-tuned model
print("Evaluating Model...")
metrics = trainer.evaluate(tokenized_datasets["validation"])
print("Evaluation Results:", metrics)

# Save the fine-tuned model
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")







