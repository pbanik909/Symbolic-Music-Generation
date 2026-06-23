import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.config import RAW_MAESTRO_DIR, PROCESSED_DIR, CHECKPOINTS_DIR
from src.preprocessing.tokenizer import get_remi_tokenizer
from src.models.transformer import MusicTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---- Settings ----
SUBSET_FILES = 120      # original used only 50
SEQ_LEN = 256
BATCH = 32
EPOCHS = 12
LR = 5e-4

tokenizer = get_remi_tokenizer()
vocab_size = len(tokenizer)
print("Vocab size:", vocab_size)

# ---- Tokenize (cached so re-runs are instant) ----
cache = PROCESSED_DIR / "transformer_tokens.npy"
if cache.exists():
    all_tokens = np.load(cache)
    print("Loaded cached tokens:", len(all_tokens))
else:
    df = pd.read_csv(RAW_MAESTRO_DIR / "maestro-v3.0.0.csv")
    composers = ["Johann Sebastian Bach","Wolfgang Amadeus Mozart","Ludwig van Beethoven",
                 "Fr\u00e9d\u00e9ric Chopin","Franz Liszt","Franz Schubert","Robert Schumann",
                 "Claude Debussy","Sergei Rachmaninoff","Joseph Haydn"]
    df = df[df["canonical_composer"].isin(composers)].sample(n=SUBSET_FILES, random_state=42)
    toks = []
    for i, (_, row) in enumerate(df.iterrows()):
        p = RAW_MAESTRO_DIR / row["midi_filename"]
        if p.exists():
            try:
                t = tokenizer(p)
                if hasattr(t, "ids"):
                    toks.extend(t.ids)
                elif isinstance(t, list) and len(t) > 0 and hasattr(t[0], "ids"):
                    toks.extend(t[0].ids)
            except Exception as e:
                print("skip:", e)
        if (i + 1) % 20 == 0:
            print(f"  tokenized {i+1}/{len(df)} files, {len(toks)} tokens")
    all_tokens = np.array(toks, dtype=np.int64)
    np.save(cache, all_tokens)
    print("Saved tokens:", len(all_tokens))

class TokenDataset(Dataset):
    def __init__(self, data, seq_len): self.data = data; self.seq_len = seq_len
    def __len__(self): return (len(self.data) - 1) // self.seq_len
    def __getitem__(self, idx):
        i = idx * self.seq_len
        return (torch.tensor(self.data[i:i+self.seq_len], dtype=torch.long),
                torch.tensor(self.data[i+1:i+self.seq_len+1], dtype=torch.long))

loader = DataLoader(TokenDataset(all_tokens, SEQ_LEN), batch_size=BATCH, shuffle=True, drop_last=True)
print("Batches per epoch:", len(loader))

model = MusicTransformer(vocab_size=vocab_size, d_model=256, nhead=8, num_layers=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def forward_causal(model, x):
    seq_len = x.size(1)
    cmask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
    e = model.embedding(x) * math.sqrt(model.d_model)
    e = model.positional_encoding(e)
    e = model.transformer_encoder(e, mask=cmask)
    return model.output_projection(e)

out_path = CHECKPOINTS_DIR / "task3_transformer_fixed.pth"
print("Training WITH causal mask... (perplexity should DROP each epoch)")
t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = forward_causal(model, x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    ppl = math.exp(total / len(loader))
    print(f"Epoch [{epoch+1}/{EPOCHS}] perplexity={ppl:.2f} ({time.time()-t0:.0f}s)")
    if (epoch + 1) % 2 == 0:
        torch.save(model.state_dict(), out_path)

torch.save(model.state_dict(), out_path)
print("Saved fixed transformer ->", out_path)
print(f"Total time: {time.time()-t0:.0f}s")