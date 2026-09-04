import os
import sys
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

# ──────────────────────────────────────────────────────────────
# 1. CONFIG
# ──────────────────────────────────────────────────────────────
MODEL_NAME    = "roberta-base"
BATCH_SIZE    = 32
NUM_EPOCHS    = 5
MAX_LEN       = 128
LEARNING_RATE = 2e-5
SEED          = 42

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
# Look for the CSV one folder up, in the Reproduction folder
CSV_PATH   = os.path.join(os.path.dirname(OUTPUT_DIR), "Reproduction", "cartography_metrics.csv")

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[GPU] Using device: {device}\n")

# ──────────────────────────────────────────────────────────────
# 2. LOAD & FILTER DATA (using exact mapped regions)
# ──────────────────────────────────────────────────────────────
print("[Step 1/5] Loading cartography metrics...")
if not os.path.exists(CSV_PATH):
    print(f"ERROR: Could not find cartography_metrics.csv at {CSV_PATH}")
    sys.exit(1)

# Load the CSV
df = pd.read_csv(CSV_PATH)

# EXACT MAPPED REGION TEST: Filter for rows labeled "Ambiguous" in our plot
df_ambig = df[df["region"] == "Ambiguous"].copy()

# HuggingFace datasets expect the target column to be named "label"
df_ambig = df_ambig.rename(columns={"true_label": "label"})

print(f"  Total original examples: {len(df)}")
print(f"  Ambiguous examples for training: {len(df_ambig)} ({(len(df_ambig)/len(df))*100:.1f}%)")

# Convert pandas DataFrame directly into a HuggingFace Dataset
train_dataset = Dataset.from_pandas(df_ambig)

# Load standard validation set from Hub
print("[Step 2/5] Loading validation set...")
raw_dataset = load_dataset("nyu-mll/glue", "sst2")
val_dataset = raw_dataset["validation"]

# ──────────────────────────────────────────────────────────────
# 3. TOKENIZE
# ──────────────────────────────────────────────────────────────
print("[Step 3/5] Tokenizing datasets...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=MAX_LEN)

train_tokenized = train_dataset.map(tokenize, batched=True)
val_tokenized   = val_dataset.map(tokenize, batched=True)

train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(train_tokenized, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_tokenized, batch_size=BATCH_SIZE)

# ──────────────────────────────────────────────────────────────
# 4. INITIALIZE FRESH MODEL
# ──────────────────────────────────────────────────────────────
print(f"\n[Step 4/5] Initializing fresh {MODEL_NAME} model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

# ──────────────────────────────────────────────────────────────
# 5. TRAINING LOOP
# ──────────────────────────────────────────────────────────────
print(f"\n[Step 5/5] Training for {NUM_EPOCHS} epochs on Ambiguous subset ONLY...")
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
    for batch in bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        if "label" in batch:
            batch["labels"] = batch.pop("label")
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()
        bar.set_postfix({"loss": f"{loss.item():.4f}"})

# ──────────────────────────────────────────────────────────────
# 6. EVALUATION
# ──────────────────────────────────────────────────────────────
print("\nTraining complete! Evaluating on full validation set...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        if "label" in batch:
            batch["labels"] = batch.pop("label")
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += len(preds)

accuracy = correct / total

print("\n" + "="*60)
print(f" BASELINE MODEL ACCURACY (100% Data, 67k examples) : 93.12%")
print(f" AMBIGUOUS MODEL ACCURACY (~50% Data, {len(df_ambig)} examples): {accuracy*100:.2f}%")
print("="*60 + "\n")
