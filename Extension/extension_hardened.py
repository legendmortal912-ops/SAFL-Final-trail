import os, sys, re, random
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

MODEL_NAME = 'roberta-base'
NUM_EPOCHS = 3
BATCH_SIZE = 32
MAX_LEN = 128
LR = 2e-5
WARMUP_RATIO = 0.06
SEED = 42
REPRO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Reproduction')
METRICS_CSV = os.path.join(REPRO_DIR, 'cartography_metrics.csv')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
if not torch.cuda.is_available():
    print('ERROR: No CUDA GPU.'); sys.exit(1)
DEVICE = torch.device('cuda')
print('[GPU]', torch.cuda.get_device_name(0))

print('[Step 1] Loading cartography metrics...')
df = pd.read_csv(METRICS_CSV)
df_ambig = df[df['region'] == 'Ambiguous'].copy()
df_easy  = df[df['region'] == 'Easy'].copy()
print('  Ambiguous:', len(df_ambig), ' Easy:', len(df_easy))

print('[Step 2] Perturbing easy examples...')
INTENSITY_MAP = {
    'amazing': 'decent', 'brilliant': 'fine', 'excellent': 'okay',
    'fantastic': 'alright', 'wonderful': 'nice', 'great': 'alright',
    'perfect': 'acceptable', 'superb': 'good', 'outstanding': 'average',
    'extraordinary': 'ordinary', 'spectacular': 'adequate',
    'incredible': 'passable', 'awesome': 'fine', 'marvelous': 'decent',
    'phenomenal': 'reasonable', 'terrific': 'okay', 'magnificent': 'reasonable',
    'masterpiece': 'competent work', 'stunning': 'pleasing',
    'beautiful': 'pleasant', 'delightful': 'agreeable', 'exceptional': 'adequate',
    'breathtaking': 'pleasant', 'terrible': 'disappointing',
    'awful': 'mediocre', 'horrible': 'subpar', 'dreadful': 'underwhelming',
    'atrocious': 'poor', 'appalling': 'lacking', 'disgusting': 'unimpressive',
    'pathetic': 'weak', 'abysmal': 'unremarkable', 'horrendous': 'poor',
    'wretched': 'below par', 'hideous': 'unappealing', 'ghastly': 'mediocre',
    'dull': 'uninspired', 'boring': 'flat', 'deplorable': 'substandard',
}
HEDGES = ['In some ways, ', 'To an extent, ', 'I suppose ', 'Somewhat ', 'More or less, ', 'In a sense, ']

def perturb_sentence(sentence, idx):
    s = sentence
    for strong, weak in INTENSITY_MAP.items():
        s = re.sub(r'\b' + re.escape(strong) + r'\b', weak, s, flags=re.IGNORECASE)
    if idx % 3 == 0:
        s = HEDGES[idx % len(HEDGES)] + s
    return s

perturbed = [perturb_sentence(row, i) for i, row in enumerate(df_easy['sentence'].tolist())]
n_changed = sum(1 for o, n in zip(df_easy['sentence'].tolist(), perturbed) if o != n)
df_easy = df_easy.copy()
df_easy['sentence'] = perturbed
print('  Modified', n_changed, 'sentences')

print('[Step 3] Building combined dataset...')
df_combined = pd.concat([
    df_ambig[['sentence', 'true_label']].rename(columns={'true_label': 'label'}),
    df_easy[['sentence',  'true_label']].rename(columns={'true_label': 'label'}),
], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
print('  Combined size:', len(df_combined))

print('[Step 4] Tokenizing (approx 60 seconds for 66k examples)...')
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
raw_data  = load_dataset('nyu-mll/glue', 'sst2')
val_raw   = raw_data['validation']

class SentimentDataset(Dataset):
    def __init__(self, sentences, labels):
        enc = tokenizer(sentences, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
        self.input_ids      = enc['input_ids']
        self.attention_mask = enc['attention_mask']
        self.labels         = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx], 'labels': self.labels[idx]}

train_dataset = SentimentDataset(df_combined['sentence'].tolist(), df_combined['label'].tolist())
val_dataset   = SentimentDataset(list(val_raw['sentence']), list(val_raw['label']))
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print('  Train batches:', len(train_loader))

print('[Step 5] Loading roberta-base...')
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)
optimizer    = AdamW(model.parameters(), lr=LR)
total_steps  = len(train_loader) * NUM_EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)
scheduler    = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

print('[Step 6] Training...')
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    bar = tqdm(train_loader, desc='Epoch ' + str(epoch) + '/' + str(NUM_EPOCHS))
    for batch in bar:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
        bar.set_postfix(loss=round(loss.item(), 4))
    avg_loss = epoch_loss / len(train_loader)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            preds = model(**batch).logits.argmax(dim=-1)
            correct += (preds == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    val_acc = correct / total
    print('  Epoch', epoch, '| Train Loss:', round(avg_loss, 4), '| Val Acc:', round(val_acc, 4))

print('[Step 7] Final evaluation...')
model.eval()
correct = total = 0
with torch.no_grad():
    for batch in tqdm(val_loader, desc='Evaluating'):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        preds = model(**batch).logits.argmax(dim=-1)
        correct += (preds == batch['labels']).sum().item()
        total += batch['labels'].size(0)
accuracy = correct / total
print('')
print('=' * 65)
print('  BASELINE           : 93.12  (100pct data, 67349 examples)')
print('  AMBIGUOUS ONLY     : 94.50  ( 49pct data, 33209 examples)')
print('  AMBIGUOUS+HARDENED :', str(round(accuracy * 100, 2)) + 'pct  ( 98pct data, ' + str(len(df_combined)) + ' examples)')
print('=' * 65)