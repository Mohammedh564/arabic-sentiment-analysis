"""
Arabic Sentiment Analysis - Fine-tuning AraBERT
Transformer-based Model for 3-class Sentiment Classification
"""

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================================
# Configuration
# ============================================================================
DATA_PATH = r"data/processed/"
MODEL_NAME = 'aubmindlab/bert-base-arabertv02'
NUM_LABELS = 3
BATCH_SIZE = 2
EPOCHS = 4
LEARNING_RATE = 1e-4
OUTPUT_DIR = r"models/transformer"

# Label mappings
id2label = {0: 'Positive', 1: 'Mixed', 2: 'Negative'}
label2id = {'Positive': 0, 'Mixed': 1, 'Negative': 2}

# ============================================================================
# Load Data
# ============================================================================
print("Loading datasets...")
train_df = pd.read_csv(f"{DATA_PATH}train.csv")
val_df = pd.read_csv(f"{DATA_PATH}val.csv") 
test_df = pd.read_csv(f"{DATA_PATH}test.csv")

# Convert text labels to integers
def convert_labels(df):
    # Handle different possible label formats
    label_mapping = {
        'positive': 0, 'Positive': 0, 'POSITIVE': 0,
        'mixed': 1, 'Mixed': 1, 'MIXED': 1,
        'negative': 2, 'Negative': 2, 'NEGATIVE': 2
    }
    df['label'] = df['label'].map(label_mapping)
    return df

train_df = convert_labels(train_df)
val_df = convert_labels(val_df)
test_df = convert_labels(test_df)

print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")
print(f"Label distribution in train:\n{train_df['label'].value_counts().sort_index()}\n")

dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'validation': Dataset.from_pandas(val_df),
    'test': Dataset.from_pandas(test_df)
})

# ============================================================================
# Tokenization
# ============================================================================
print("Tokenizing datasets...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512)

tokenized_data = dataset.map(preprocess_function, batched=True)
dataCollator = DataCollatorWithPadding(tokenizer=tokenizer)

# ============================================================================
# Model Setup
# ============================================================================
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)

# Apply LoRA for parameter-efficient fine-tuning
print("Applying LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"]
)
model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())


# ============================================================================
# Evaluation Metrics
# ============================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted")
    }

# ============================================================================
# Training Configuration
# ============================================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",      # save every N steps
    save_steps=500,              # adjust based on batch size & dataset
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    report_to="none",
    seed=42,
    gradient_accumulation_steps=8,
    fp16=True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['validation'],
    data_collator=dataCollator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)
# ============================================================================
# Training
# ============================================================================
print("Starting training...\n")
trainer.train()

# ============================================================================
# Evaluation on Test Set
# ============================================================================
print("\n" + "="*60)
print("Test Set Evaluation")
print("="*60)

test_results = trainer.evaluate(tokenized_data['test'])
print(f"\nAccuracy: {test_results['eval_accuracy']:.4f}")
print(f"F1 Macro: {test_results['eval_f1_macro']:.4f}")
print(f"F1 Weighted: {test_results['eval_f1_weighted']:.4f}")

# Detailed classification report
predictions = trainer.predict(tokenized_data['test'])
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

print("\n" + classification_report(
    true_labels, 
    pred_labels, 
    target_names=list(id2label.values()),
    digits=4
))

# ============================================================================
# Save Model
# ============================================================================
print("="*60)
print("Saving final model and tokenizer...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✓ Model saved to: {OUTPUT_DIR}")
print("="*60)
