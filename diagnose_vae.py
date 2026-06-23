import os, sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import CHECKPOINTS_DIR
from src.models.vae import LSTM_VAE

device = torch.device("cpu")
model = LSTM_VAE().to(device)
ckpt = CHECKPOINTS_DIR / "task2_lstm_vae.pth"
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()
print("Loaded VAE checkpoint OK")

with torch.no_grad():
    z = torch.randn(1, model.latent_dim).to(device)
    logits = model.decode(z, seq_len=128)
    probs = torch.sigmoid(logits).cpu().numpy()[0]

print("Max prob:   ", round(float(probs.max()), 4))
print("Mean prob:  ", round(float(probs.mean()), 4))
print("Notes >0.4: ", int((probs > 0.4).sum()))
print("Notes >0.1: ", int((probs > 0.1).sum()))
print("Notes >0.05:", int((probs > 0.05).sum()))
print("Notes >0.02:", int((probs > 0.02).sum()))
