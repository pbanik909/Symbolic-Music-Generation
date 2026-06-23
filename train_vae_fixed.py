import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.config import PROCESSED_DIR, CHECKPOINTS_DIR
from src.models.vae import LSTM_VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

SUBSET = 20000
EPOCHS = 30
BATCH = 128
LR = 1e-3
KL_WARMUP = 8
FREE_BITS = 0.5

all_data = np.load(PROCESSED_DIR / "maestro_train_windows.npy")
print("Full train windows:", all_data.shape)
if SUBSET and SUBSET < len(all_data):
    rng = np.random.default_rng(0)
    all_data = all_data[rng.choice(len(all_data), SUBSET, replace=False)]
    print("Using subset:", all_data.shape)

class DS(Dataset):
    def __init__(self, arr): self.arr = arr
    def __len__(self): return len(self.arr)
    def __getitem__(self, i): return torch.tensor(self.arr[i], dtype=torch.float32)

loader = DataLoader(DS(all_data), batch_size=BATCH, shuffle=True, drop_last=True)
print("Batches per epoch:", len(loader))

class Focal(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__(); self.a=alpha; self.g=gamma
    def forward(self, logits, t):
        ce = nn.functional.binary_cross_entropy_with_logits(logits, t, reduction="none")
        p = torch.sigmoid(logits); pt = p*t + (1-p)*(1-t)
        at = self.a*t + (1-self.a)*(1-t)
        return (at * (1-pt)**self.g * ce).mean()

model = LSTM_VAE().to(device)
recon_criterion = Focal().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def tf_forward(model, batch):
    B, T, P = batch.shape
    mu, log_var = model.encode(batch)
    z = model.reparameterize(mu, log_var)
    h0 = model.decoder_linear(z).unsqueeze(0).repeat(model.num_layers, 1, 1).contiguous()
    c0 = torch.zeros_like(h0)
    start = torch.zeros(B, 1, P, device=batch.device)
    dec_in = torch.cat([start, batch[:, :-1, :]], dim=1)
    dec_out, _ = model.decoder_lstm(dec_in, (h0, c0))
    return model.output_layer(dec_out), mu, log_var

out_path = CHECKPOINTS_DIR / "task2_lstm_vae_fixed.pth"
print("Training... WATCH KL: it should settle near 64 and STAY (not drop to 0).")
t0 = time.time()
for epoch in range(EPOCHS):
    model.train()
    beta = min(1.0, epoch / KL_WARMUP)
    er, ek, n = 0.0, 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits, mu, log_var = tf_forward(model, batch)
        recon = recon_criterion(logits, batch)
        kl_dim = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp())).mean(dim=0)
        kl = torch.clamp(kl_dim, min=FREE_BITS).sum()
        (recon + beta * kl).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        er += recon.item(); ek += kl.item(); n += 1
    print(f"Epoch [{epoch+1}/{EPOCHS}] beta={beta:.2f} recon={er/n:.4f} KL={ek/n:.3f} ({time.time()-t0:.0f}s)")
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), out_path)

torch.save(model.state_dict(), out_path)
print("Saved fixed VAE ->", out_path)
print(f"Total time: {time.time()-t0:.0f}s")
