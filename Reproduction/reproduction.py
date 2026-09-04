"""
reproduction.py
===============
Reproduction of: "Dataset Cartography: Mapping and Diagnosing Datasets with
Training Dynamics" (Swayamdipta et al., 2020 | arXiv:2009.10795)

What this script does:
  1. Loads the SST-2 sentiment classification dataset (67k examples).
  2. Fine-tunes roberta-base for NUM_EPOCHS epochs on your RTX 4050 GPU.
  3. After EVERY epoch, records the Softmax probability of the TRUE label
     for every training example.
  4. After training, computes:
       - Confidence (mu): Mean of the per-epoch true-label probabilities.
       - Variability (sigma): Std Dev of the per-epoch true-label probabilities.
  5. Saves the cartography metrics to 'cartography_metrics.csv'.
  6. Plots and saves the Dataset Cartography map as 'reproduction_map.png'.

GPU NOTE:
  This script will hard-crash with a clear error if no CUDA GPU is found.
  It will NOT silently fall back to CPU.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# 0. GPU ENFORCEMENT — will hard crash if GPU is not available
# ──────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    print("=" * 60)
    print("  FATAL ERROR: No CUDA GPU detected!")
    print("  This script requires an NVIDIA GPU (RTX 4050 or similar).")
    print("  Make sure your PyTorch installation includes CUDA support.")
    print("  Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("=" * 60)
    sys.exit(1)

DEVICE = torch.device("cuda")
print(f"[GPU] Using device    : {torch.cuda.get_device_name(0)}")
print(f"[GPU] VRAM available  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

# ──────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────────
MODEL_NAME    = "roberta-base"
DATASET_NAME  = "sst2"           # SST-2: 67k sentence-level sentiment examples
NUM_EPOCHS    = 5                 # Paper used 6; 5 is enough to see the dynamics
BATCH_SIZE    = 32                # Fits comfortably in 6GB VRAM for roberta-base
MAX_LEN       = 128               # Max token length (SST-2 sentences are short)
LEARNING_RATE = 2e-5              # Standard fine-tuning LR for RoBERTa
WARMUP_RATIO  = 0.06              # 6% of total steps for LR warmup
SEED          = 42
OUTPUT_DIR    = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────────
# 2. REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ──────────────────────────────────────────────────────────────
# 3. LOAD DATASET & TOKENIZER
# ──────────────────────────────────────────────────────────────
print("[Step 1/5] Loading SST-2 dataset and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

raw_dataset = load_dataset("nyu-mll/glue", DATASET_NAME)
train_data  = raw_dataset["train"]
val_data    = raw_dataset["validation"]

print(f"  Training examples  : {len(train_data)}")
print(f"  Validation examples: {len(val_data)}\n")

def tokenize(batch):
    return tokenizer(
        batch["sentence"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

# Apply tokenization
train_tokenized = train_data.map(tokenize, batched=True)
val_tokenized   = val_data.map(tokenize, batched=True)

# HuggingFace datasets -> PyTorch tensors
train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# We keep two DataLoaders for the training set:
#   - Shuffled: used during the actual training pass (better gradient diversity)
#   - Ordered : used during the probability-recording pass (stable example indices)
train_loader_shuffled = DataLoader(train_tokenized, batch_size=BATCH_SIZE, shuffle=True)
train_loader_ordered  = DataLoader(train_tokenized, batch_size=BATCH_SIZE, shuffle=False)
val_loader            = DataLoader(val_tokenized,   batch_size=BATCH_SIZE, shuffle=False)

# ──────────────────────────────────────────────────────────────
# 4. LOAD MODEL
# ──────────────────────────────────────────────────────────────
print(f"[Step 2/5] Loading {MODEL_NAME} for sequence classification...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model = model.to(DEVICE)
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")

# ──────────────────────────────────────────────────────────────
# 5. OPTIMIZER & SCHEDULER
# ──────────────────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

total_steps  = len(train_loader_shuffled) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

# ──────────────────────────────────────────────────────────────
# 6. TRAINING LOOP + PROBABILITY RECORDING
#
# The SECRET SAUCE of Dataset Cartography:
# After every epoch we do a separate inference pass over the ENTIRE
# training set (with shuffle=False so example indices stay stable),
# and we record the Softmax probability assigned to the TRUE label
# for every single training example.
#
# After NUM_EPOCHS, prob_records[i] is a list of NUM_EPOCHS floats:
#   [P(true_label_i @ epoch1), P(true_label_i @ epoch2), ...]
# ──────────────────────────────────────────────────────────────
print("[Step 3/5] Training model and recording per-epoch probabilities...")
num_train    = len(train_tokenized)
prob_records = [[] for _ in range(num_train)]

for epoch in range(1, NUM_EPOCHS + 1):

    # ── Training pass (shuffled) ──────────────────────────────
    model.train()
    epoch_loss = 0.0
    train_bar  = tqdm(train_loader_shuffled,
                      desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]", leave=False)

    for batch in train_bar:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()

        # Gradient clipping — standard for transformer fine-tuning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = epoch_loss / len(train_loader_shuffled)

    # ── Validation accuracy ───────────────────────────────────
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = outputs.logits.argmax(dim=-1)
            correct       += (preds == labels).sum().item()
            total         += labels.size(0)

    val_acc = correct / total
    print(f"  Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ── Probability recording pass (ordered, NO shuffle) ──────
    # We iterate in the SAME order every epoch so that index i
    # always refers to the same training example.
    model.eval()
    idx = 0
    with torch.no_grad():
        for batch in tqdm(train_loader_ordered,
                          desc=f"Epoch {epoch}/{NUM_EPOCHS} [Recording probs]", leave=False):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Softmax converts raw logits -> proper probabilities summing to 1
            probs   = F.softmax(outputs.logits, dim=-1)  # shape: (batch_size, 2)

            for i in range(len(labels)):
                true_label           = labels[i].item()
                prob_of_true_label   = probs[i, true_label].item()
                prob_records[idx + i].append(prob_of_true_label)

            idx += len(labels)

print()

# ──────────────────────────────────────────────────────────────
# 7. COMPUTE CARTOGRAPHY METRICS
#
# For each training example i:
#   confidence_i (mu_i)    = mean(prob_records[i])
#   variability_i (sigma_i) = std(prob_records[i])
# ──────────────────────────────────────────────────────────────
print("[Step 4/5] Computing cartography metrics...")
prob_array  = np.array(prob_records)       # shape: (num_train, NUM_EPOCHS)
confidence  = prob_array.mean(axis=1)      # mu  for each example
variability = prob_array.std(axis=1)       # sigma for each example
true_labels = np.array(train_tokenized["label"])

# ── Classify into regions (following the paper's thresholds) ──
median_var = np.median(variability)
region     = np.where(confidence < 0.25, "Hard",
             np.where(variability > median_var, "Ambiguous", "Easy"))

# ── Save metrics to CSV ───────────────────────────────────────
metrics_df = pd.DataFrame({
    "example_idx" : np.arange(num_train),
    "sentence"    : train_data["sentence"],
    "true_label"  : true_labels,
    "confidence"  : confidence,
    "variability" : variability,
    "region"      : region,
})

csv_path = os.path.join(OUTPUT_DIR, "cartography_metrics.csv")
metrics_df.to_csv(csv_path, index=False)
print(f"  Metrics saved to: {csv_path}")

print(f"\n  Dataset Region Summary (using AdamW optimizer):")
print(f"  Easy       : {(region == 'Easy').sum():>6} ({(region == 'Easy').mean()*100:.1f}%)")
print(f"  Ambiguous  : {(region == 'Ambiguous').sum():>6} ({(region == 'Ambiguous').mean()*100:.1f}%)")
print(f"  Hard       : {(region == 'Hard').sum():>6} ({(region == 'Hard').mean()*100:.1f}%)\n")

# ──────────────────────────────────────────────────────────────
# 8. PLOT THE DATASET CARTOGRAPHY MAP
#
# X-axis: Variability (sigma) — how much does the model bounce?
# Y-axis: Confidence (mu)     — how often is the model right?
#
# Colour: Green=Easy, Orange=Ambiguous, Red=Hard
# ──────────────────────────────────────────────────────────────
print("[Step 5/5] Plotting Dataset Cartography map...")

COLORS = {"Easy": "#2ecc71", "Ambiguous": "#f39c12", "Hard": "#e74c3c"}
LABELS = {
    "Easy"     : f"Easy-to-learn  (n={(region == 'Easy').sum()})",
    "Ambiguous": f"Ambiguous      (n={(region == 'Ambiguous').sum()})",
    "Hard"     : f"Hard-to-learn  (n={(region == 'Hard').sum()})",
}

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#16213e")

for reg in ["Easy", "Ambiguous", "Hard"]:
    mask = region == reg
    ax.scatter(
        variability[mask],
        confidence[mask],
        c=COLORS[reg],
        label=LABELS[reg],
        alpha=0.35,
        s=6,
        linewidths=0,
    )

# Annotation boxes for the three regions
annotations = [
    ("Easy-to-learn\n(high confidence,\nlow variability)",  (0.02, 0.93), (0.14, 0.78)),
    ("Ambiguous\n(high variability)",                        (0.60, 0.52), (0.55, 0.40)),
    ("Hard-to-learn\n(low confidence,\nlow variability)",   (0.03, 0.06), (0.16, 0.18)),
]
for text, xy, xytext in annotations:
    ax.annotate(
        text,
        xy=xy, xycoords="axes fraction",
        xytext=xytext, textcoords="axes fraction",
        fontsize=9, color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc="#0f3460", ec="#888", alpha=0.85),
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.2),
    )

ax.set_xlabel(
    "Variability  (σ — std dev of P(true label) across epochs)",
    fontsize=12, color="#ecf0f1", labelpad=10
)
ax.set_ylabel(
    "Confidence  (μ — mean of P(true label) across epochs)",
    fontsize=12, color="#ecf0f1", labelpad=10
)
ax.set_title(
    f"Dataset Cartography Map  —  SST-2 / {MODEL_NAME}  /  {NUM_EPOCHS} epochs\n"
    f"Optimizer: AdamW  |  Reproduction of Swayamdipta et al. (2020) arXiv:2009.10795",
    fontsize=12, color="white", pad=14
)

ax.tick_params(colors="#ecf0f1")
ax.spines[:].set_color("#444")
ax.set_xlim(-0.01, max(variability.max() * 1.05, 0.1))
ax.set_ylim(-0.02, 1.05)

ax.legend(
    fontsize=10, framealpha=0.3, loc="center right",
    facecolor="#0f3460", edgecolor="gray", labelcolor="white"
)

ax.text(
    0.01, 0.01,
    f"GPU: {torch.cuda.get_device_name(0)}",
    transform=ax.transAxes, fontsize=7.5, color="#888",
    verticalalignment="bottom"
)

plt.tight_layout()
map_path = os.path.join(OUTPUT_DIR, "reproduction_map.png")
plt.savefig(map_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print(f"  Map saved to: {map_path}")

print("\n" + "=" * 60)
print("  REPRODUCTION COMPLETE")
print(f"  Cartography map : reproduction_map.png")
print(f"  Metrics CSV     : cartography_metrics.csv")
print("=" * 60)
