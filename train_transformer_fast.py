import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.config import PROCESSED_DIR, CHECKPOINTS_DIR
from src.preprocessing.tokenizer import get_remi_tokenizer
from src.models.transformer import MusicTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

SEQ_LEN = 256
BATCH = 32
EPOCHS = 6
LR = 5e-4
TOKEN_LIMIT = 300000   # use a slice for a fast first model (tokens are cached)

tokenizer = get_remi_tokenizer()
vocab_size = len(tokenizer)

all_tokens = np.load(PROCESSED_DIR / "transformer_tokens.npy")
print("Cached tokens available:", len(all_tokens))
all_tokens = all_tokens[:TOKEN_LIMIT]
print("Using slice:", len(all_tokens))

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
    torch.save(model.state_dict(), out_path)   # save EVERY epoch
    print(f"  saved checkpoint")

print("Saved fixed transformer ->", out_path)
print(f"Total time: {time.time()-t0:.0f}s")